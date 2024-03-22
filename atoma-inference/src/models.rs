use std::{error::Error, fmt::Display};

use candle::Device;
use candle_transformers::{
    models::{
        llama::{Config as LlamaConfig, Llama},
        llama2_c::{Config as Llama2Config, Llama as Llama2},
        mamba::{Config as MambaConfig, Model as MambaModel},
        mistral::{Config as MistralConfig, Model as MistralModel},
        mixtral::{Config as MixtralConfig, Model as MixtralModel},
        stable_diffusion::StableDiffusionConfig,
    },
    quantized_var_builder::VarBuilder,
};

use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub enum ModelType {
    Llama(usize),
    Llama2(usize),
    Mamba(usize),
    Mixtral8x7b,
    Mistral(usize),
    StableDiffusionV1_5,
    StableDiffusionV2_1,
    StableDiffusionXl,
    StableDiffusionTurbo,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(size) => write!(f, "llama({})", size),
            Self::Llama2(size) => write!(f, "llama2({})", size),
            Self::Mamba(size) => write!(f, "mamba({})", size),
            Self::Mixtral8x7b => write!(f, "mixtral_8x7b"),
            Self::Mistral(size) => write!(f, "mistral({})", size),
            Self::StableDiffusionV1_5 => write!(f, "stable_diffusion_v1_5"),
            Self::StableDiffusionV2_1 => write!(f, "stable_diffusion_v2_1"),
            Self::StableDiffusionXl => write!(f, "stable_diffusion_xl"),
            Self::StableDiffusionTurbo => write!(f, "stable_diffusion_turbo"),
        }
    }
}

#[derive(Clone)]
pub enum ModelConfig {
    Llama(LlamaConfig),
    Llama2(Llama2Config),
    Mamba(MambaConfig),
    Mixtral8x7b(MixtralConfig),
    Mistral(MistralConfig),
    StableDiffusion(StableDiffusionConfig),
}

impl From<ModelType> for ModelConfig {
    fn from(_model_type: ModelType) -> Self {
        todo!()
    }
}

pub trait ModelApi {
    fn load(model_specs: ModelSpecs, var_builder: VarBuilder) -> Self;
    fn run(&self, input: String) -> Result<String, Box<dyn Error + Send + Sync>>;
}

#[allow(dead_code)]
pub struct ModelSpecs {
    pub(crate) config: ModelConfig,
    pub(crate) device: Device,
    pub(crate) tokenizer: Tokenizer,
}

pub enum Model {
    Llama {
        model_specs: ModelSpecs,
        model: Llama,
    },
    Llama2 {
        model_specs: ModelSpecs,
        model: Llama2,
    },
    Mamba {
        model_specs: ModelSpecs,
        model: MambaModel,
    },
    Mixtral8x7b {
        model_specs: ModelSpecs,
        model: MixtralModel,
    },
    Mistral {
        model_specs: ModelSpecs,
        model: MistralModel,
    },
}

impl ModelApi for Model {
    fn load(model_specs: ModelSpecs, var_builder: VarBuilder) -> Self {
        let model_config = model_specs.config.clone();
        match model_config {
            ModelConfig::Llama(config) => {
                let model = load_llama_model(config, var_builder);
                Self::Llama { model, model_specs }
            }
            ModelConfig::Llama2(config) => {
                let model = load_llama2_model(config, var_builder);
                Self::Llama2 { model_specs, model }
            }
            ModelConfig::Mamba(config) => {
                let model = load_mamba_model(config, var_builder);
                Self::Mamba { model_specs, model }
            }
            ModelConfig::Mistral(config) => {
                let model = load_mistral(config, var_builder);
                Self::Mistral { model_specs, model }
            }
            ModelConfig::Mixtral8x7b(config) => {
                let model = load_mixtral(config, var_builder);
                Self::Mixtral8x7b { model_specs, model }
            }
            ModelConfig::StableDiffusion(config) => {
                panic!("TODO: implement it")
            }
        }
    }

    fn run(&self, input: String) -> Result<String, Box<dyn Error + Send + Sync>> {
        todo!()
    }
}

fn load_llama_model(config: LlamaConfig, var_builder: VarBuilder) -> Llama {
    todo!()
}

fn load_llama2_model(config: Llama2Config, var_builder: VarBuilder) -> Llama2 {
    todo!()
}

fn load_mamba_model(config: MambaConfig, var_builder: VarBuilder) -> MambaModel {
    todo!()
}
fn load_mistral(config: MistralConfig, var_builder: VarBuilder) -> MistralModel {
    todo!()
}

fn load_mixtral(config: MixtralConfig, var_builder: VarBuilder) -> MixtralModel {
    todo!()
}
