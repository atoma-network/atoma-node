use std::fmt::Display;

use candle::{DType, Device, Error as CandleError, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{
        llama::{Config as LlamaConfig, Llama},
        llama2_c::{Config as Llama2Config, Llama as Llama2},
        mamba::{Config as MambaConfig, Model as MambaModel},
        mistral::{Config as MistralConfig, Model as MistralModel},
        mixtral::{Config as MixtralConfig, Model as MixtralModel},
        stable_diffusion::StableDiffusionConfig,
    },
};
use thiserror::Error;

use tokenizers::Tokenizer;

use crate::types::Temperature;

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
    fn run(
        &self,
        input: String,
        max_tokens: usize,
        random_seed: usize,
        temperature: Temperature,
        top_p: f32,
    ) -> Result<String, ModelError>;
}

#[allow(dead_code)]
pub struct ModelSpecs {
    pub(crate) config: ModelConfig,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
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
                let model = Llama::load(var_builder, &config).expect("Failed to load LlaMa model");
                Self::Llama { model, model_specs }
            }
            ModelConfig::Llama2(config) => {
                let model = Llama2::load(var_builder, config).expect("Failed to load LlaMa2 model");
                Self::Llama2 { model_specs, model }
            }
            ModelConfig::Mamba(config) => {
                let model =
                    MambaModel::new(&config, var_builder).expect("Failed to load Mamba model");
                Self::Mamba { model_specs, model }
            }
            ModelConfig::Mistral(config) => {
                let model =
                    MistralModel::new(&config, var_builder).expect("Failed to load Mistral model");
                Self::Mistral { model_specs, model }
            }
            ModelConfig::Mixtral8x7b(config) => {
                let model =
                    MixtralModel::new(&config, var_builder).expect("Failed to load Mixtral model");
                Self::Mixtral8x7b { model_specs, model }
            }
            ModelConfig::StableDiffusion(_) => {
                panic!("TODO: implement it")
            }
        }
    }

    fn run(
        &self,
        input: String,
        max_tokens: usize,
        random_seed: usize,
        temperature: Temperature,
        top_p: f32,
    ) -> Result<String, ModelError> {
        match self {
            Self::Llama { model_specs, model } => {
                let mut tokens = model_specs
                    .tokenizer
                    .encode(input, true)
                    .map_err(ModelError::TokenizerError)?
                    .get_ids()
                    .to_vec();

                let mut logits = LogitsProcessor::new(
                    random_seed as u64,
                    Some(temperature as f64),
                    Some(top_p as f64),
                );

                let start = std::time::Instant::now();

                let index_pos = 0;
                let mut tokens_generated = 0;

                let mut output = String::with_capacity(max_tokens);

                for index in 0..max_tokens {
                    let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                        (1, index_pos)
                    } else {
                        (tokens.len(), 0)
                    };
                    let ctx = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctx, &model_specs.device)?.unsqueeze(0)?;
                    let logits = model.forward(&input, context_index, &mut cache)?;
                    let logits = logits.squeeze(0).map_err(ModelError::LogitsError)?;
                    let logits = if repeat_penalty == 1. {
                        logits
                    } else {
                        let start_at = tokens.len().saturating_sub(repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            repeat_penalty,
                            &tokens[start_at..],
                        )?
                    };
                    index_pos += ctx.len();
            
                    let next_token = logits_processor.sample(&logits)?;
                    token_generated += 1;
                    tokens.push(next_token);
            
                    if Some(next_token) == eos_token_id {
                        break;
                    }
                    if let Some(t) = model_specs.tokenizer(next_token)? {
                        print!("{t}");
                        std::io::stdout().flush()?;
                    }
                }
                todo!()
            }
            Self::Llama2 { model_specs, model } => {
                todo!()
            }
            Self::Mamba { model_specs, model } => {
                todo!()
            }
            Self::Mistral { model_specs, model } => {
                todo!()
            }
            Self::Mixtral8x7b { model_specs, model } => {
                todo!()
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Failed to load error: `{0}`")]
    LoadError(CandleError),
    #[error("Failed input tokenization: `{0}`")]
    TokenizerError(Box<dyn std::error::Error + Send + Sync>),
    #[error("Logits error: `{0}`")]
    LogitsError(CandleError)
}
