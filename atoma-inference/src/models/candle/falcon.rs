use std::{path::PathBuf, str::FromStr, time::Instant};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::falcon::{Config, Falcon},
    utils::apply_repeat_penalty,
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::models::{
    config::ModelConfig,
    types::{LlmLoadData, ModelType, TextModelInput},
    ModelError, ModelTrait,
};

use super::device;

pub struct FalconModel {
    model: Falcon,
    device: Device,
    dtype: DType,
    model_type: ModelType,
    tokenizer: Tokenizer,
}

impl FalconModel {
    pub fn new(
        model: Falcon,
        device: Device,
        dtype: DType,
        model_type: ModelType,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            model,
            device,
            dtype,
            tokenizer,
            model_type,
        }
    }
}

impl ModelTrait for FalconModel {
    type Input = TextModelInput;
    type Output = String;
    type LoadData = LlmLoadData;

    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        let device = device(config.device_id())?;
        let dtype = DType::from_str(&config.dtype())?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let model_type = ModelType::from_str(&config.model_id())?;
        let repo_id = model_type.repo().to_string();
        let revision = model_type.default_revision().to_string();

        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;
        let model_weights_file_path = repo.get("model.safetensors")?;

        Ok(Self::LoadData {
            device,
            dtype,
            file_paths: vec![
                config_file_path,
                tokenizer_file_path,
                model_weights_file_path,
            ],
            model_type: ModelType::from_str(&config.model_id())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Falcon model ...");

        let start = Instant::now();

        let config_filename = load_data.file_paths[0].clone();
        let tokenizer_filename = load_data.file_paths[1].clone();
        let weights_filenames = load_data.file_paths[2..].to_vec();

        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

        let config: Config =
            serde_json::from_slice(&std::fs::read(config_filename)?)?;
        config.validate()?;

        if load_data.dtype != DType::BF16 || load_data.dtype != DType::F32 {
            panic!("Invalid dtype, it must be either BF16 or F32 precision");
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weights_filenames,
                load_data.dtype,
                &load_data.device,
            )?
        };
        let model = Falcon::load(vb, config.clone())?;
        info!("loaded the model in {:?}", start.elapsed());

        Ok(Self::new(
            model,
            load_data.device,
            load_data.dtype,
            load_data.model_type,
            tokenizer,
        ))
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let TextModelInput {
            prompt,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            top_p,
            ..
        } = input;

        let mut logits_processor =
            LogitsProcessor::new(random_seed, Some(temperature), Some(top_p));
        info!("Running inference on prompt: {:?}", prompt);
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let mut output = String::new();

        let start_gen = Instant::now();
        for index in 0..max_tokens {
            let start_gen = Instant::now();
            let context_size = if self.model.config().use_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctx = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctx, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(self.dtype)?;
            let logits = if repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(repeat_last_n);
                apply_repeat_penalty(&logits, repeat_penalty, &tokens[start_at..])?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            debug!("> {:?}", start_gen);
            output.push_str(
                &self
                    .tokenizer
                    .decode(&[next_token], true)?,
            );
        }
        let dt = start_gen.elapsed();

        info!(
            "{max_tokens} tokens generated ({} token/s)\n----\n{}\n----",
            max_tokens as f64 / dt.as_secs_f64(),
            self.tokenizer
                .decode(&new_tokens, true)?,
        );

        Ok(output)
    }
}
