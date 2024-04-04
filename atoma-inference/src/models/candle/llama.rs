#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Cache, LlamaConfig},
};
use hf_hub::{api::sync::Api, Repo, RepoType};

use candle_transformers::models::llama as model;
use tokenizers::Tokenizer;

use crate::models::{
    token_output_stream::TokenOutputStream, types::PrecisionBits, ModelError, ModelTrait,
};

use super::{device, hub_load_safetensors};

const EOS_TOKEN: &str = "</s>";

#[allow(dead_code)]
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
enum Which {
    V1,
    V2,
    Solar10_7B,
    TinyLlama1_1BChat,
}

pub struct Config {}

pub struct Llama {
    device: Device,
    tokenizer: Tokenizer,
    llama: model::Llama,
    cache: Cache,
}

pub struct Input {
    prompt: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
    sample_len: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Input {
    pub fn default_prompt(prompt: String) -> Self {
        Self {
            prompt,
            temperature: None,
            top_p: None,
            seed: 0,
            sample_len: 10000,
            repeat_penalty: 1.,
            repeat_last_n: 64,
        }
    }
}

pub struct Fetch {
    model_id: Option<String>,
    revision: Option<String>,
    which: Which,
    use_flash_attn: bool,
    dtype: Option<String>,
}

impl Default for Fetch {
    fn default() -> Self {
        Self {
            model_id: None,
            revision: None,
            which: Which::TinyLlama1_1BChat,
            use_flash_attn: false,
            dtype: None,
        }
    }
}

impl ModelTrait for Llama {
    type Input = Input;
    type Fetch = Fetch;
    type Output = Vec<Tensor>;

    fn fetch(fetch: &Self::Fetch) -> Result<(), ModelError> {
        let device = device()?;
        let dtype = match fetch.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => Err(ModelError::Config(format!("Invalid dtype : {dtype}")))?,
            None => DType::F16,
        };
        let api = Api::new()?;
        let model_id = fetch.model_id.clone().unwrap_or_else(|| match fetch.which {
            Which::V1 => "Narsil/amall-7b".to_string(),
            Which::V2 => "meta-llama/Llama-2-7b-hf".to_string(),
            Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0".to_string(),
            Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        });
        let revision = fetch.revision.clone().unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        api.get("tokenizer.json")?;
        let config_filename = api.get("config.json")?;
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(fetch.use_flash_attn);
        let filenames = match fetch.which {
            Which::V1 | Which::V2 | Which::Solar10_7B => {
                hub_load_safetensors(&api, "model.safetensors.index.json")?
            }
            Which::TinyLlama1_1BChat => vec![api.get("model.safetensors")?],
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        model::Llama::load(vb, &config)?;
        Ok(())
    }

    fn model_id(&self) -> crate::models::ModelId {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string()
    }

    fn load(filenames: Vec<PathBuf>, precision: PrecisionBits) -> Result<Self, ModelError> {
        let device = device()?;
        let dtype = precision.into_dtype();
        let (llama, tokenizer_filename, cache) = {
            let tokenizer_filename = filenames[0].clone();
            let config_filename = filenames[1].clone();
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let config = config.into_config(false); // TODO: use from config

            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&filenames[2..], dtype, &device)? };
            let cache = model::Cache::new(true, dtype, &config, &device)?; // TODO: use from config
            (model::Llama::load(vb, &config)?, tokenizer_filename, cache)
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        Ok(Llama {
            device,
            tokenizer,
            llama,
            cache,
        })
    }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN);
        let mut tokens = self
            .tokenizer
            .encode(input.prompt.clone(), true)?
            .get_ids()
            .to_vec();

        let mut tokenizer = TokenOutputStream::new(self.tokenizer.clone());
        let mut logits_processor = LogitsProcessor::new(input.seed, input.temperature, input.top_p);
        let mut index_pos = 0;
        let mut res = String::new();
        let mut result = Vec::new();
        for index in 0..input.sample_len {
            let (context_size, context_index) = if self.cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input_tensor = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self
                .llama
                .forward(&input_tensor, context_index, &mut self.cache)?;
            let logits = logits.squeeze(0)?;
            let logits = if input.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(input.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    input.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();
            let next_token = logits_processor.sample(&logits)?;
            result.push(logits);
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                res += &t;
            }
        }
        if let Some(rest) = tokenizer.decode_rest()? {
            res += &rest;
        }
        println!("Result {}", res);
        Ok(result)
    }
}
