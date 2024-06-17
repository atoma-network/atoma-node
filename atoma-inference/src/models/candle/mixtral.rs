use std::str::FromStr;

use atoma_types::Digest;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::mixtral::{Config, Model},
    utils::apply_repeat_penalty,
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{info, instrument};

use crate::{
    bail,
    models::{
        candle::{device, hub_load_safetensors},
        token_output_stream::TokenOutputStream,
        types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
        ModelError, ModelTrait,
    },
};

/// `MixtralModel` - encapsulates a Mixtral model
/// together with additional metadata, necessary
/// to run inference
pub struct MixtralModel {
    /// The actual Mistral model
    model: Model,
    /// The device holding the model
    /// weights, while running inference
    device: Device,
    /// The model weights decimal precision
    dtype: DType,
    /// Tokenizer, with streaming functionality
    tokenizer: TokenOutputStream,
}

impl MixtralModel {
    pub fn new(model: Model, device: Device, dtype: DType, tokenizer: TokenOutputStream) -> Self {
        Self {
            model,
            device,
            dtype,
            tokenizer,
        }
    }
}

impl ModelTrait for MixtralModel {
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
        let repo_id = ModelType::Mixtral8x7b.repo().to_string();
        let revision = ModelType::Mixtral8x7b.default_revision().to_string();
        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weight_filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
        let mut file_paths = Vec::with_capacity(1 + weight_filenames.len());
        file_paths.push(tokenizer_filename);
        file_paths.extend(weight_filenames);

        let device = device(config.device_id())?;

        Ok(Self::LoadData {
            model_type: ModelType::Mixtral8x7b,
            file_paths,
            device,
            dtype: DType::from_str(&config.dtype())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    #[instrument(skip_all)]
    fn load(
        load_data: Self::LoadData,
        stream_tx: mpsc::Sender<(Digest, String)>,
    ) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let device = load_data.device;
        let dtype = load_data.dtype;

        let start = std::time::Instant::now();

        let config = Config::v0_1_8x7b(load_data.use_flash_attention);
        let tokenizer = Tokenizer::from_file(load_data.file_paths[0].clone())?;
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&load_data.file_paths[1..], dtype, &device)?
        };
        let model = Model::new(&config, var_builder)?;

        info!("Loaded the model in {:?}", start.elapsed());
        Ok(Self {
            model,
            device,
            dtype,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
        })
    }

    fn model_type(&self) -> ModelType {
        ModelType::Mixtral8x7b
    }

    #[instrument(skip_all)]
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        info!("Running inference on prompt: {}", input.prompt);
        // clean tokenizer state
        self.tokenizer.clear();

        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, Some(input.temperature), input.top_p);
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(input.prompt, true)?
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
            let ctx = &tokens[start_pos..];
            let input_ids = Tensor::new(ctx, &self.device)?.unsqueeze(0)?;
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
                output.push_str(&word);
            }
        }

        let dt = start_gen.elapsed();
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

        Ok(TextModelOutput {
            text: output,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
            input_tokens,
            tokens: if input.chat { tokens } else { vec![] },
        })
    }
}
