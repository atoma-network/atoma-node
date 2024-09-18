use std::str::FromStr;

use atoma_types::AtomaStreamingData;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use candle_transformers::models::qwen2_moe::{Config as ConfigMoe, Model as ModelMoe};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{info, instrument};

use crate::bail;
use crate::models::candle::{device, hub_load_safetensors};
use crate::models::token_output_stream::TokenOutputStream;
use crate::models::types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput};
use crate::models::{ModelError, ModelTrait};

/// Helper enum to differentiate between Qwen's
/// MoE model and the base model
pub enum Model {
    Base(ModelBase),
    MoE(ModelMoe),
}

impl Model {
    /// Performs a forward pass, based on the type of Qwen model available
    fn forward(&mut self, input: &Tensor, start_pos: usize) -> Result<Tensor, ModelError> {
        match self {
            Model::Base(ref mut base) => base
                .forward(input, start_pos)
                .map_err(ModelError::CandleError),
            Model::MoE(ref mut moe) => moe
                .forward(input, start_pos)
                .map_err(ModelError::CandleError),
        }
    }

    /// Clears the key-value cache
    fn clear_kv_cache(&mut self) {
        match self {
            Model::Base(m) => m.clear_kv_cache(),
            Model::MoE(m) => m.clear_kv_cache(),
        }
    }
}

/// `QwenModel` - encapsulates a Qwen model
/// together with additional metadata, necessary
/// to run inference
pub struct QwenModel {
    /// The model's unique identifier
    model_type: ModelType,
    /// The actual Qwen model
    model: Model,
    /// The device holding the model
    /// weights, while running inference
    device: Device,
    /// The model weights decimal precision
    dtype: DType,
    /// Tokenizer, with streaming functionality
    tokenizer: TokenOutputStream,
}

impl QwenModel {
    pub fn new(
        model: Model,
        model_type: ModelType,
        device: Device,
        dtype: DType,
        tokenizer: Tokenizer,
        stream_tx: mpsc::Sender<AtomaStreamingData>,
    ) -> Self {
        Self {
            model,
            model_type,
            device,
            dtype,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
        }
    }
}

impl ModelTrait for QwenModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
    type LoadData = LlmLoadData;

    #[instrument(skip_all)]
    fn fetch(
        api_key: String,
        cache_dir: std::path::PathBuf,
        config: crate::models::config::ModelConfig,
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

        let model_type = ModelType::from_str(&config.model_id())?;
        let repo_id = model_type.repo().to_string();
        let revision = model_type.default_revision().to_string();
        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = match model_type {
            ModelType::QwenW0_5b | ModelType::QwenW1_8b => vec![repo.get("model.safetensors")?],
            ModelType::QwenW4b
            | ModelType::QwenW7b
            | ModelType::QwenW14b
            | ModelType::QwenW72b
            | ModelType::QwenMoeA27b => {
                hub_load_safetensors(&repo, "model.safetensors.index.json")?
            }
            _ => return Err(ModelError::InvalidModelType(model_type.to_string())),
        };
        let config_filename = repo.get("config.json")?;

        let device = device(config.device_first_id())?;
        let dtype = DType::from_str(&config.dtype())?;

        let mut file_paths = Vec::with_capacity(2 + filenames.len());
        file_paths.push(config_filename);
        file_paths.push(tokenizer_filename);
        file_paths.extend(filenames);

        Ok(LlmLoadData {
            device,
            dtype,
            file_paths,
            model_type,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    #[instrument(skip_all)]
    fn load(
        load_data: Self::LoadData,
        stream_tx: mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError> {
        let device = load_data.device;
        let dtype = load_data.dtype;

        let start = std::time::Instant::now();
        let tokenizer = Tokenizer::from_file(load_data.file_paths[1].clone())?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&load_data.file_paths[2..], dtype, &device)?
        };
        let model = match load_data.model_type {
            ModelType::QwenMoeA27b => {
                let config: ConfigMoe =
                    serde_json::from_slice(&std::fs::read(&load_data.file_paths[0])?)?;
                Model::MoE(ModelMoe::new(&config, vb)?)
            }
            ModelType::QwenW0_5b
            | ModelType::QwenW1_8b
            | ModelType::QwenW4b
            | ModelType::QwenW7b
            | ModelType::QwenW14b
            | ModelType::QwenW72b => {
                let config: ConfigBase =
                    serde_json::from_slice(&std::fs::read(&load_data.file_paths[0])?)?;
                Model::Base(ModelBase::new(&config, vb)?)
            }
            _ => {
                return Err(ModelError::InvalidModelType(
                    load_data.model_type.to_string(),
                ))
            }
        };

        info!("Loaded the model in {:?}", start.elapsed());
        Ok(Self::new(
            model,
            load_data.model_type,
            device,
            dtype,
            tokenizer,
            stream_tx,
        ))
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    #[instrument(skip_all)]
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(input.prompt, true)?
            .get_ids()
            .to_vec();
        let mut tokens = [input.pre_prompt_tokens, tokens].concat();
        let input_tokens = tokens.len();

        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, Some(input.temperature), input.top_p);

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => bail!("cannot find the <|endoftext|> token"),
        };

        let request_id = Some(input.request_id).filter(|_| input.should_stream_output);
        let mut output = String::new();
        let start_gen = std::time::Instant::now();

        for index in 0..input.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input_tensor = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_tensor, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(self.dtype)?;
            let logits = if input.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(input.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    input.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            tokens.push(next_token);
            if let Some(t) = self.tokenizer.next_token(next_token, request_id.clone())? {
                output.push_str(&t);
            }
        }

        let dt = start_gen.elapsed();
        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        self.model.clear_kv_cache();
        if input.should_stream_output {
            info!("Ending stream");
            self.tokenizer.end_stream(request_id.unwrap())?;
        }

        Ok(Self::Output {
            text: output,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
            input_tokens,
            tokens: if input.chat { tokens } else { vec![] },
        })
    }
}
