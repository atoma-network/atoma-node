use std::{path::PathBuf, time::Instant};

use candle::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device,
};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::mamba::{Config, Model},
};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    models::{ModelError, ModelId, ModelTrait},
    types::PrecisionBits,
};

pub struct MambaModel {
    model: Model,
    config: Config,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
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
            tokenizer,
            which,
        }
    }
}

pub struct MambaInput {
    prompt: String,
    temperature: f64,
    random_seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    top_p: f64,
}

impl MambaInput {
    pub fn new(
        prompt: String,
        temperature: f64,
        random_seed: u64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        top_p: f64,
    ) -> Self {
        Self {
            prompt,
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            top_p,
        }
    }
}

impl ModelTrait for MambaModel {
    type Input = MambaInput;
    type Output = String;

    fn load(filenames: Vec<PathBuf>, precision: PrecisionBits) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Mamba model ...");

        let start = Instant::now();

        let tokenizer_filename = filenames[0].clone();
        let config_filename = filenames[1].clone();
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

        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_filenames, dtype, &device)
                .map_err(ModelError::CandleError)?
        };
        let model = Model::new(&config, var_builder).map_err(ModelError::CandleError)?;
        info!("Loaded Mamba model in {:?}", start.elapsed());

        Ok(Self::new(model, config, device, dtype, tokenizer))
    }

    fn model_id(&self) -> ModelId {
        self.which.model_id().to_string()
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output, ModelError> {
        todo!()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    Mamba2_8bSlimPj,
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            Self::Mamba2_8bSlimPj => "state-spaces/mamba-2.8b-slimpj'",
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
