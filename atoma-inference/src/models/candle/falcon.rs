use std::{path::PathBuf, time::Instant};

use candle::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::falcon::{Config, Falcon},
    utils::apply_repeat_penalty,
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::models::{types::{LlmFetchData, LlmLoadData, TextModelInput}, ModelError, ModelId, ModelTrait};

pub struct FalconModel {
    model: Falcon,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    which: Which,
}

impl FalconModel {
    pub fn new(
        model: Falcon,
        config: Config,
        device: Device,
        dtype: DType,
        tokenizer: Tokenizer,
    ) -> Self {
        let which = Which::from_config(&config);
        Self {
            model,
            device,
            dtype,
            tokenizer,
            which,
        }
    }
}

impl ModelTrait for FalconModel {
    type FetchData = LlmFetchData;
    type Input = TextModelInput;
    type Output = String;
    type LoadData = LlmLoadData;

    fn fetch(fetch_data: &Self::FetchData) -> Result<Vec<PathBuf>, ModelError> {
        let api_key = fetch_data.api_key;
        let cache_dir = fetch_data.cache_dir;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let repo = api.repo(Repo::with_revision(
            fetch_data.model_id.clone(),
            RepoType::Model,
            fetch_data.revision,
        ));

        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;
        let model_weights_file_path = repo.get("model.safetensors")?;

        Ok(vec![
            config_file_path,
            tokenizer_file_path,
            model_weights_file_path,
        ])
    }

    fn load(
        load_data: Self::LoadData
    ) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Falcon model ...");

        let start = Instant::now();

        let config_filename = load_data.file_paths[0].clone();
        let tokenizer_filename = load_data.file_paths[1].clone();
        let weights_filenames = load_data.file_paths[2..].to_vec();

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(ModelError::BoxedError)?;

        let config: Config =
            serde_json::from_slice(&std::fs::read(config_filename).map_err(ModelError::IoError)?)
                .map_err(ModelError::DeserializeError)?;
        config.validate()?;

        let device = if cuda_is_available() {
            Device::new_cuda(load_data.device_id).map_err(ModelError::CandleError)?
        } else if metal_is_available() {
            Device::new_metal(load_data.device_id).map_err(ModelError::CandleError)?
        } else {
            Device::Cpu
        };

        if load_data.dtype != DType::BF16 || load_data.dtype != DType::F32 {
            panic!("Invalid dtype, it must be either BF16 or F32 precision");
        }

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights_filenames, load_data.dtype, &device)? };
        let model = Falcon::load(vb, config.clone())?;
        info!("loaded the model in {:?}", start.elapsed());

        Ok(Self::new(model, config, device, load_data.dtype, tokenizer))
    }

    fn model_id(&self) -> ModelId {
        self.which.model_id().to_string()
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
            .encode(prompt, true)
            .map_err(ModelError::BoxedError)?
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
                    .decode(&[next_token], true)
                    .map_err(ModelError::BoxedError)?,
            );
        }
        let dt = start_gen.elapsed();

        info!(
            "{max_tokens} tokens generated ({} token/s)\n----\n{}\n----",
            max_tokens as f64 / dt.as_secs_f64(),
            self.tokenizer
                .decode(&new_tokens, true)
                .map_err(ModelError::BoxedError)?,
        );

        Ok(output)
    }
}

enum Which {
    Falcon7b,
    Falcon40b,
    Falcon180b,
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Falcon7b => "tiiuae/falcon-7b",
            Self::Falcon40b => "tiiuae/falcon-40b",
            Self::Falcon180b => "tiiuae/falcon-180b",
        }
    }

    fn from_config(config: &Config) -> Self {
        match config.hidden_size {
            4544 => Self::Falcon7b,
            8192 => Self::Falcon40b,
            14848 => Self::Falcon180b,
            _ => panic!("Invalid config hidden size value"),
        }
    }
}
