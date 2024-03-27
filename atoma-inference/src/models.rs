use std::fmt::Display;

use candle::{DType, Device, Error as CandleError, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{
        llama::{Cache as LlamaCache, Config as LlamaConfig, Llama},
        mamba::{Config as MambaConfig, Model as MambaModel},
        mistral::{Config as MistralConfig, Model as MistralModel},
        mixtral::{Config as MixtralConfig, Model as MixtralModel},
        stable_diffusion::StableDiffusionConfig,
    },
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use tokenizers::Tokenizer;

use crate::types::Temperature;

const EOS_TOKEN: &str = "</s>";

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ModelType {
    Llama2_7b,
    Mamba3b,
    Mixtral8x7b,
    Mistral7b,
    StableDiffusion2,
    StableDiffusionXl,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama2_7b => write!(f, "llama2_7b"),
            Self::Mamba3b => write!(f, "mamba_3b"),
            Self::Mixtral8x7b => write!(f, "mixtral_8x7b"),
            Self::Mistral7b => write!(f, "mistral_7b"),
            Self::StableDiffusion2 => write!(f, "stable_diffusion_2"),
            Self::StableDiffusionXl => write!(f, "stable_diffusion_xl"),
        }
    }
}

impl ModelType {
    pub(crate) fn model_config(&self) -> ModelConfig {
        match self {
            Self::Llama2_7b => ModelConfig::Llama(LlamaConfig::config_7b_v2(false)), // TODO: add the case for flash attention
            Self::Mamba3b => todo!(),
            Self::Mistral7b => ModelConfig::Mistral(MistralConfig::config_7b_v0_1(false)), // TODO: add the case for flash attention
            Self::Mixtral8x7b => {
                ModelConfig::Mixtral8x7b(Box::new(MixtralConfig::v0_1_8x7b(false)))
            } // TODO: add the case for flash attention
            Self::StableDiffusion2 => ModelConfig::StableDiffusion(Box::new(
                StableDiffusionConfig::v2_1(None, None, None),
            )),
            Self::StableDiffusionXl => ModelConfig::StableDiffusion(Box::new(
                StableDiffusionConfig::sdxl_turbo(None, None, None),
            )),
        }
    }
}

#[derive(Clone)]
pub enum ModelConfig {
    Llama(LlamaConfig),
    Mamba(MambaConfig),
    Mixtral8x7b(Box<MixtralConfig>),
    Mistral(MistralConfig),
    StableDiffusion(Box<StableDiffusionConfig>),
}

impl From<ModelType> for ModelConfig {
    fn from(_model_type: ModelType) -> Self {
        todo!()
    }
}

pub trait ModelApi {
    fn load(model_specs: ModelSpecs, var_builder: VarBuilder) -> Self;

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        input: String,
        max_tokens: usize,
        random_seed: usize,
        repeat_last_n: usize,
        repeat_penalty: f32,
        temperature: Temperature,
        top_p: f32,
    ) -> Result<String, ModelError>;
}

#[allow(dead_code)]
pub struct ModelSpecs {
    pub(crate) cache: Option<LlamaCache>,
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
            // ModelConfig::Llama2(config) => {
            //     let model = Llama2::load(var_builder, config).expect("Failed to load LlaMa2 model");
            //     Self::Llama2 { model_specs, model }
            // }
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
        repeat_last_n: usize,
        repeat_penalty: f32,
        temperature: Temperature,
        top_p: f32,
    ) -> Result<String, ModelError> {
        match self {
            Self::Llama { model_specs, model } => {
                let mut cache = model_specs.cache.clone().expect("Failed to get cache");
                let mut tokens = model_specs
                    .tokenizer
                    .encode(input, true)
                    .map_err(ModelError::TokenizerError)?
                    .get_ids()
                    .to_vec();

                let mut logits_processor = LogitsProcessor::new(
                    random_seed as u64,
                    Some(temperature as f64),
                    Some(top_p as f64),
                );

                let eos_token_id = model_specs.tokenizer.token_to_id(EOS_TOKEN);

                let start = std::time::Instant::now();

                let mut index_pos = 0;
                let mut tokens_generated = 0;

                let mut output = Vec::with_capacity(max_tokens);

                for index in 0..max_tokens {
                    let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                        (1, index_pos)
                    } else {
                        (tokens.len(), 0)
                    };
                    let ctx = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctx, &model_specs.device)
                        .map_err(ModelError::TensorError)?
                        .unsqueeze(0)
                        .map_err(ModelError::TensorError)?;
                    let logits = model
                        .forward(&input, context_index, &mut cache)
                        .map_err(ModelError::TensorError)?;
                    let logits = logits.squeeze(0).map_err(ModelError::LogitsError)?;
                    let logits = if repeat_penalty == 1. {
                        logits
                    } else {
                        let start_at = tokens.len().saturating_sub(repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            repeat_penalty,
                            &tokens[start_at..],
                        )
                        .map_err(ModelError::TensorError)?
                    };
                    index_pos += ctx.len();

                    let next_token = logits_processor
                        .sample(&logits)
                        .map_err(ModelError::TensorError)?;
                    tokens_generated += 1;
                    tokens.push(next_token);

                    if Some(next_token) == eos_token_id {
                        break;
                    }
                    // TODO: possibly do this in batches will speed up the process
                    if let Ok(t) = model_specs.tokenizer.decode(&[next_token], true) {
                        output.push(t);
                    }
                    let dt = start.elapsed();
                    tracing::info!(
                        "Generated {tokens_generated} tokens ({} tokens/s)",
                        tokens_generated as f64 / dt.as_secs_f64()
                    );
                }
                Ok(output.join(" "))
            }
            Self::Mamba { .. } => {
                todo!()
            }
            Self::Mistral { .. } => {
                todo!()
            }
            Self::Mixtral8x7b { .. } => {
                todo!()
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Cache error: `{0}`")]
    CacheError(String),
    #[error("Failed to load error: `{0}`")]
    LoadError(CandleError),
    #[error("Logits error: `{0}`")]
    LogitsError(CandleError),
    #[error("Tensor error: `{0}`")]
    TensorError(CandleError),
    #[error("Failed input tokenization: `{0}`")]
    TokenizerError(Box<dyn std::error::Error + Send + Sync>),
}
