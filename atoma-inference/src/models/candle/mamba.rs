use std::{path::PathBuf, time::Instant};

use candle::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::mamba::{Config, Model, State},
    utils::apply_repeat_penalty,
};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    bail,
    models::types::{PrecisionBits, TextModelInput},
    models::{token_output_stream::TokenOutputStream, ModelError, ModelId, ModelTrait},
};

pub struct MambaModel {
    model: Model,
    config: Config,
    device: Device,
    dtype: DType,
    tokenizer: TokenOutputStream,
    which: Which,
}

impl MambaModel {
    pub fn new(
        model: Model,
        config: Config,
        device: Device,
        dtype: DType,
        tokenizer: Tokenizer,
    ) -> Self {
        let which = Which::from_config(&config);
        Self {
            model,
            config,
            device,
            dtype,
            tokenizer: TokenOutputStream::new(tokenizer),
            which,
        }
    }
}

impl ModelTrait for MambaModel {
    type Input = TextModelInput;
    type Output = String;

    fn load(filenames: Vec<PathBuf>, precision: PrecisionBits) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Mamba model ...");

        let start = Instant::now();

        let config_filename = filenames[0].clone();
        let tokenizer_filename = filenames[1].clone();
        let weights_filenames = filenames[2..].to_vec();

        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).map_err(ModelError::TokenizerError)?;

        let config: Config =
            serde_json::from_slice(&std::fs::read(config_filename).map_err(ModelError::IoError)?)
                .map_err(ModelError::DeserializeError)?;
        let device = if cuda_is_available() {
            Device::new_cuda(0).map_err(ModelError::CandleError)?
        } else if metal_is_available() {
            Device::new_metal(0).map_err(ModelError::CandleError)?
        } else {
            Device::Cpu
        };
        let dtype = precision.into_dtype();

        info!("Loading model weights..");
        let var_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights_filenames, dtype, &device)? };
        let model = Model::new(&config, var_builder.pp("backbone"))?;
        info!("Loaded Mamba model in {:?}", start.elapsed());

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

        info!("Running inference on prompt: {:?}", prompt);

        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(ModelError::TokenizerError)?
            .get_ids()
            .to_vec();
        let mut logits_processor =
            LogitsProcessor::new(random_seed, Some(temperature), Some(top_p));

        let mut generated_tokens = 0_usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => bail!("Invalid eos token"),
        };

        let mut state = State::new(1, &self.config, &self.device)?; // TODO: handle larger batch sizes

        let mut next_logits = None;
        let mut output = String::new();

        for &token in tokens.iter() {
            let input = Tensor::new(&[token], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(token)? {
                output.push_str(t.as_str());
            }
        }

        let start_gen = Instant::now();
        for _ in 0..max_tokens {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => bail!("cannot work on an empty prompt"),
            };

            let logits = logits.squeeze(0)?.to_dtype(self.dtype)?;
            let logits = if repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(repeat_last_n);
                apply_repeat_penalty(&logits, repeat_penalty, &tokens[start_at..])?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if next_token == eos_token {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                output.push_str(t.as_str());
            }

            let input = Tensor::new(&[next_token], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?);
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest()? {
            output.push_str(rest.as_str());
        }

        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(output)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    // Mamba2_8bSlimPj, TODO: add this
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            // Self::Mamba2_8bSlimPj => "state-spaces/mamba-2.8b-slimpj'",
        }
    }

    fn from_config(config: &Config) -> Self {
        match config.d_model {
            768 => Self::Mamba130m,
            1024 => Self::Mamba370m,
            1536 => Self::Mamba790m,
            2048 => Self::Mamba1_4b,
            2560 => Self::Mamba2_8b,
            _ => panic!("Invalid config d_model value"),
        }
    }
}
