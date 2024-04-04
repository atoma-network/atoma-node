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
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::{
    models::types::{PrecisionBits, TextModelInput},
    models::{ModelError, ModelId, ModelTrait},
};

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
    type Fetch = ();
    type Input = TextModelInput;
    type Output = String;

    fn fetch(_fetch: &Self::Fetch) -> Result<(), ModelError> {
        Ok(())
    }

    fn load(
        filenames: Vec<PathBuf>,
        precision: PrecisionBits,
        device_id: usize,
    ) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Falcon model ...");

        let start = Instant::now();

        let config_filename = filenames[0].clone();
        let tokenizer_filename = filenames[1].clone();
        let weights_filenames = filenames[2..].to_vec();

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(ModelError::BoxedError)?;

        let config: Config =
            serde_json::from_slice(&std::fs::read(config_filename).map_err(ModelError::IoError)?)
                .map_err(ModelError::DeserializeError)?;
        config.validate()?;

        let device = if cuda_is_available() {
            Device::new_cuda(device_id).map_err(ModelError::CandleError)?
        } else if metal_is_available() {
            Device::new_metal(device_id).map_err(ModelError::CandleError)?
        } else {
            Device::Cpu
        };

        let dtype = precision.into_dtype();
        if dtype != DType::BF16 || dtype != DType::F32 {
            panic!("Invalid dtype, it must be either BF16 or F32 precision");
        }

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights_filenames, dtype, &device)? };
        let model = Falcon::load(vb, config.clone())?;
        info!("loaded the model in {:?}", start.elapsed());

        Ok(Self::new(model, config, device, dtype, tokenizer))
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
