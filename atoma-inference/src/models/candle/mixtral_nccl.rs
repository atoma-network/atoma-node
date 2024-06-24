use atoma_types::Digest;
use candle::{DType, Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, utils::apply_repeat_penalty};
use cudarc::{driver::safe::CudaDevice, nccl::result::NcclError};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::{path::PathBuf, rc::Rc, str::FromStr, thread, time::Instant};

use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, mpsc};
use tracing::{error, info};

use crate::{
    bail,
    models::{
        candle::mixtral_nccl_model::Config,
        config::ModelConfig,
        token_output_stream::TokenOutputStream,
        types::{ModelType, TextModelInput, TextModelOutput},
        ModelError, ModelTrait,
    },
};
use cudarc::driver::DriverError;
use cudarc::nccl::safe::{Comm, Id};

use super::{hub_load_safetensors, mixtral_nccl_model as model};

/// `MixtralModel` - encapsulates a Mixtral model
/// together with additional metadata, necessary
/// to run inference
pub struct MixtralNcclModel {
    /// The model's unique identifier
    model_type: ModelType,
    to_workers_sender: broadcast::Sender<TextModelInput>,
    output_receiver: mpsc::Receiver<TextModelOutput>,
}

#[derive(Error, Debug)]
pub enum MixtralNcclError {
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

struct MixtralNcclWorker {
    rank: usize,
    device: Device,
    tokenizer: TokenOutputStream,
    model: model::Model,
    model_type: ModelType,
    dtype: DType,
}

impl MixtralNcclWorker {
    pub fn new(
        rank: usize,
        num_shards: usize,
        id: Id,
        dtype: DType,
        model_weights_file_paths: &[PathBuf],
        tokenizer_file_path: &PathBuf,
        model_type: ModelType,
        device_id: usize,
        stream_tx: tokio::sync::mpsc::Sender<(Digest, String)>,
    ) -> Result<Self, ModelError> {
        let device = CudaDevice::new(rank)?;
        let comm =
            Rc::new(Comm::from_rank(device, rank, num_shards, id).map_err(ModelError::NcclError)?);
        info!("Rank {rank:?} spawned");
        let device = Device::new_cuda(device_id)?;
        let config = Config::v0_1_8x7b(true);
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                model_weights_file_paths,
                dtype,
                &device,
            )?
        };
        let cache = model::Cache::new(dtype, &config, &device)?;
        let model = model::Model::new(rank, &config, vb, &comm, &cache)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file_path)?;

        Ok(Self {
            rank,
            device,
            model,
            model_type,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
            dtype,
        })
    }

    pub fn run(&mut self, input: TextModelInput) -> Result<TextModelOutput, ModelError> {
        self.tokenizer.clear();

        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, Some(input.temperature), input.top_p);
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(input.prompt.clone(), true)?
            .get_ids()
            .to_vec();
        let mut tokens = [input.pre_prompt_tokens, tokens].concat();

        let input_tokens = tokens.len();

        let mut generated_tokens = 0_usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => bail!("cannot find the </s> token"),
        };

        let request_id = Some(input.request_id).filter(|_| input.should_stream_output);
        let mut output = String::new();
        let start_gen = std::time::Instant::now();
        for index in 0..input.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input_ids = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_ids, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(self.dtype)?;
            let logits = if input.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(input.repeat_last_n);
                apply_repeat_penalty(&logits, input.repeat_penalty, &tokens[start_at..])?
            };

            let next_token = logits_processor.sample(&logits)?;
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            tokens.push(next_token);
            if let Some(word) = self.tokenizer.next_token(next_token, request_id.clone())? {
                print!("{word}");
                output.push_str(&word);
            }
        }
        println!();
        let dt = start_gen.elapsed();
        if self.rank == 0 {
            if let Some(rest) = self.tokenizer.decode_rest(request_id.clone())? {
                output.push_str(&rest);
            }

            info!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );

            if input.should_stream_output {
                info!("Ending stream");
                self.tokenizer.end_stream(request_id.unwrap())?;
            }
        }

        Ok(TextModelOutput {
            text: output,
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
    /// The model type, to identify the model (e.g. Mixtral8x7)
    pub model_type: ModelType,
    pub device_ids: Vec<usize>,
}

impl ModelTrait for MixtralNcclModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
    type LoadData = LlmLoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        info!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;
        let repo_id = ModelType::Mixtral8x7b.repo().to_string();
        let revision = ModelType::Mixtral8x7b.default_revision().to_string();
        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weight_filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
        let mut file_paths = Vec::with_capacity(1 + weight_filenames.len());
        file_paths.push(tokenizer_filename);
        file_paths.extend(weight_filenames);

        Ok(Self::LoadData {
            dtype: DType::from_str(&config.dtype())?,
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
        stream_tx: tokio::sync::mpsc::Sender<(Digest, String)>,
    ) -> Result<Self, ModelError> {
        info!("Loading Mixtral model ...");
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
            let model_type = load_data.model_type.clone();
            let stream_tx = stream_tx.clone();
            let device_id = load_data.device_ids[rank];
            thread::spawn(move || -> Result<(), ModelError> {
                let mixtral_worker = MixtralNcclWorker::new(
                    rank,
                    num_shards,
                    id,
                    dtype,
                    &file_paths[1..],
                    &file_paths[0],
                    model_type,
                    device_id,
                    stream_tx,
                );
                if let Err(e) = &mixtral_worker {
                    error!("Error: {:?}", e);
                }
                let mut mixtral_worker = mixtral_worker?;
                info!("Starting inference loop on rank {rank}");
                loop {
                    let input = to_workers_receiver.blocking_recv()?;
                    let output = mixtral_worker.run(input)?;
                    if rank == 0 {
                        output_sender.blocking_send(output)?;
                    }
                }
            });
        }

        info!("Loaded Mixtral model in {:?}", start.elapsed());

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
