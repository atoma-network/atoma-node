use std::str::FromStr;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor, models::phi3::Model, utils::apply_repeat_penalty,
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    bail,
    models::{
        candle::{device, hub_load_safetensors},
        token_output_stream::TokenOutputStream,
        types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
        ModelError, ModelTrait,
    },
};

pub struct Phi3Model {
    model: Model,
    device: Device,
    dtype: DType,
    tokenizer: TokenOutputStream,
}

impl ModelTrait for Phi3Model {
    type Input = TextModelInput;
    type LoadData = LlmLoadData;
    type Output = TextModelOutput;

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

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weight_filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
        let mut file_paths = Vec::with_capacity(1 + weight_filenames.len());
        file_paths.push(config_filename);
        file_paths.push(tokenizer_filename);
        file_paths.extend(weight_filenames);

        let device = device(config.device_id())?;

        Ok(Self::LoadData {
            model_type,
            file_paths,
            device,
            dtype: DType::from_str(&config.dtype())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let Self::LoadData {
            file_paths,
            device,
            dtype,
            ..
        } = load_data;

        let start = std::time::Instant::now();

        let config = serde_json::from_slice(&std::fs::read(&file_paths[0])?)?;
        let tokenizer = Tokenizer::from_file(file_paths[1].clone())?;
        let var_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&file_paths[2..], dtype, &device)? };
        let model = Model::new(&config, var_builder)?;

        info!("Loaded the model in {:?}", start.elapsed());
        Ok(Self {
            model,
            device,
            dtype,
            tokenizer: TokenOutputStream::new(tokenizer),
        })
    }

    fn model_type(&self) -> ModelType {
        ModelType::Phi3Mini
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        info!("Running inference on prompt: {}, with inputs = {:?}", input.prompt, input);
        // clean tokenizer state
        self.tokenizer.clear();

        let tokens = self.tokenizer.tokenizer().encode(input.prompt, true)?;
        if tokens.is_empty() {
            bail!("Empty prompts are not supported in the phi model.")
        }
        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, Some(input.temperature), input.top_p);
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => bail!("cannot find the endoftext token"),
        };

        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        let mut output = String::new();
        for index in 0..input.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input_tensor = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_tensor, pos)?.i((.., 0, ..))?;
            let logits = logits.squeeze(0)?.to_dtype(self.dtype)?;
            let logits = if input.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(input.repeat_last_n);
                apply_repeat_penalty(&logits, input.repeat_penalty, &tokens[start_at..])?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(token) = self.tokenizer.next_token(next_token)? {
                output.push_str(&token)
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(TextModelOutput {
            text: output,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
        })
    }
}
