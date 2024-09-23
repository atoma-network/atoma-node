use atoma_types::AtomaStreamingData;
use candle::{DType, Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::llama::LlamaEosToks};
use cudarc::{driver::safe::CudaDevice, nccl::result::NcclError};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::{path::PathBuf, rc::Rc, str::FromStr, thread, time::Instant};

use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, mpsc};
use tracing::{error, info, Span};

use crate::models::{
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{ModelType, TextModelInput, TextModelOutput},
    ModelError, ModelTrait,
};
use cudarc::driver::DriverError;
use cudarc::nccl::safe::{Comm, Id};

use super::{hub_load_safetensors, llama_nccl_model as model};

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "</s>";

/// `LlamaModel` - encapsulates a Llama model
/// together with additional metadata, necessary
/// to run inference
pub struct LlamaNcclModel {
    /// The model's unique identifier
    model_type: ModelType,
    to_workers_sender: broadcast::Sender<TextModelInput>,
    output_receiver: mpsc::Receiver<TextModelOutput>,
}

struct LlamaNcclWorker {
    rank: usize,
    device: Device,
    config: model::Config,
    tokenizer: TokenOutputStream,
    model: model::Llama,
    cache: model::Cache,
}

impl LlamaNcclWorker {
    pub fn new(
        rank: usize,
        num_shards: usize,
        id: Id,
        dtype: DType,
        config_file_path: &PathBuf,
        model_weights_file_paths: &[PathBuf],
        tokenizer_file_path: &PathBuf,
        device_id: usize,
        stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError> {
        let device = CudaDevice::new(rank)?;
        // Initialize the Communicator from Nvidia Collective Communication Library. This is for the inter gpu communication.
        // For more information visit https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
        let comm =
            Rc::new(Comm::from_rank(device, rank, num_shards, id).map_err(ModelError::NcclError)?);
        info!("Rank {rank:?} spawned");
        let device = Device::new_cuda(device_id)?;
        let config: model::Config = serde_json::from_slice(&std::fs::read(config_file_path)?)?;
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                model_weights_file_paths,
                dtype,
                &device,
            )?
        };
        let cache = model::Cache::new(dtype, &config, &device)?;
        let model = model::Llama::load(vb, &cache, &config, &comm)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file_path)?;

        Ok(Self {
            rank,
            device,
            config,
            model,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
            cache,
        })
    }

    pub fn run(&mut self, input: TextModelInput) -> Result<TextModelOutput, ModelError> {
        self.tokenizer.clear();
        self.cache.clear();

        let bos_token_id = self
            .config
            .bos_token_id
            .or_else(|| self.tokenizer.tokenizer().token_to_id(BOS_TOKEN))
            .unwrap();
        let eos_token_id = self.config.eos_token_id.clone().or_else(|| {
            self.tokenizer
                .tokenizer()
                .token_to_id(EOS_TOKEN)
                .map(LlamaEosToks::Single)
        });
        let prompt_ids = self
            .tokenizer
            .tokenizer()
            .encode(input.prompt.clone(), true)?
            .get_ids()
            .to_vec();
        let mut tokens = if input.pre_prompt_tokens.is_empty() {
            prompt_ids
        } else {
            [input.pre_prompt_tokens, vec![bos_token_id], prompt_ids].concat()
        };
        let input_tokens = tokens.len();

        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, Some(input.temperature), input.top_p);
        let mut index_pos = 0;
        let mut res = String::new();

        let request_id = Some(input.request_id).filter(|_| input.should_stream_output);
        let start_gen = Instant::now();
        for index in 0..input.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, index_pos)?;
            let logits = logits.squeeze(0)?;
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            match eos_token_id {
                Some(LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                    break;
                }
                Some(LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                    break;
                }
                _ => (),
            }
            if let Some(t) = self.tokenizer.next_token(next_token, request_id.clone())? {
                res += &t;
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest(request_id.clone())? {
            res += &rest;
        }
        let generated_tokens = self.tokenizer.get_num_generated_tokens();
        let dt = start_gen.elapsed();
        if self.rank == 0 {
            info!(
                "{generated_tokens} tokens generated ({} token/s)\n",
                generated_tokens as f64 / dt.as_secs_f64(),
            );

            if input.should_stream_output {
                info!("Ending stream");
                self.tokenizer.end_stream(request_id.unwrap())?;
            }
        }

        Ok(TextModelOutput {
            text: res,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
            input_tokens,
            tokens: if input.chat { tokens } else { vec![] },
        })
    }
}

#[derive(Clone)]
pub struct LlmLoadData {
    /// The `DType`, representing the decimal
    /// precision in which the model is supposed to run
    pub dtype: DType,
    /// Vector of all the downloaded model weights file paths
    pub file_paths: Vec<PathBuf>,
    /// The model type, to identify the model (e.g. Llama3-8b)
    pub model_type: ModelType,
    pub device_ids: Vec<usize>,
}

impl ModelTrait for LlamaNcclModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
    type LoadData = LlmLoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        let dtype = DType::from_str(&config.dtype())?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let model_type = ModelType::from_str(&config.model_id())?;
        let repo_id = model_type.repo().to_string();
        let revision = model_type.default_revision().to_string();

        let repo = api.repo(Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        let model_weights_file_paths = if &repo_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0" {
            vec![repo.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        let mut file_paths = Vec::with_capacity(2 + model_weights_file_paths.len());
        file_paths.extend(vec![config_file_path, tokenizer_file_path]);
        file_paths.extend(model_weights_file_paths);

        Ok(Self::LoadData {
            dtype,
            file_paths,
            model_type: ModelType::from_str(&config.model_id())?,
            device_ids: config.device_ids(),
        })
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn load(
        load_data: Self::LoadData,
        stream_tx: tokio::sync::mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError> {
        info!("Loading Llama model ...");
        let start = Instant::now();
        let dtype = load_data.dtype;
        let num_shards = load_data.device_ids.len();
        let id = Id::new().unwrap();

        let (to_workers_sender, _) = broadcast::channel(100);
        let (output_sender, output_receiver) = mpsc::channel(100);

        for rank in 0..num_shards {
            let file_paths = load_data.file_paths.clone();
            let mut to_workers_receiver = to_workers_sender.subscribe();
            let output_sender = output_sender.clone();
            let stream_tx = stream_tx.clone();
            let device_id = load_data.device_ids[rank];
            let span = Span::current();
            thread::spawn(move || -> Result<(), ModelError> {
                let _enter = span.enter();
                let llama_worker = LlamaNcclWorker::new(
                    rank,
                    num_shards,
                    id,
                    dtype,
                    &file_paths[0],
                    &file_paths[2..],
                    &file_paths[1],
                    device_id,
                    stream_tx,
                );
                if let Err(e) = &llama_worker {
                    error!("Error: {:?}", e);
                }
                let mut llama_worker = llama_worker?;
                info!("Starting inference loop on rank {rank}");
                loop {
                    let input = to_workers_receiver.blocking_recv()?;
                    let output = llama_worker.run(input)?;
                    if rank == 0 {
                        output_sender.blocking_send(output)?;
                    }
                }
            });
        }

        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self {
            model_type: load_data.model_type,
            to_workers_sender,
            output_receiver,
        })
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        self.to_workers_sender.send(input).map_err(Box::new)?;
        self.output_receiver
            .blocking_recv()
            .ok_or_else(|| ModelError::Msg("Something went wrong".to_string()))
    }
}

#[derive(Error, Debug)]
pub enum LlamaNcclError {
    #[error("DriverError error: `{0}`")]
    DriverError(#[from] DriverError),
    #[error("Nccl error: `{}`", 0.0)]
    NcclError(NcclError),
    #[error("Candle error: `{0}`")]
    CandleError(#[from] candle::Error),
    #[error("SerdeJsonError error: `{0}`")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("IoError error: `{0}`")]
    IoError(#[from] std::io::Error),
    #[error("Error: `{0}`")]
    BoxedError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Tokio error: `{0}`")]
    RecvError(#[from] tokio::sync::broadcast::error::RecvError),
    #[error("ModelError error: `{0}`")]
    ModelError(#[from] ModelError),
}
