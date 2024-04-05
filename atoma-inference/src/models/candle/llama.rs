#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::{path::PathBuf, str::FromStr};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Cache, LlamaConfig},
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use candle_transformers::models::llama as model;
use tokenizers::Tokenizer;

use crate::models::{
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{LlmLoadData, ModelType, TextModelInput},
    ModelError, ModelTrait,
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

pub struct LlamaModel {
    cache: Cache,
    device: Device,
    dtype: DType,
    model: model::Llama,
    model_type: ModelType,
    tokenizer: Tokenizer,
}

impl ModelTrait for LlamaModel {
    type Input = TextModelInput;
    type Output = String;
    type LoadData = LlmLoadData;

    fn fetch(cache_dir: PathBuf, config: ModelConfig) -> Result<Self::LoadData, ModelError> {
        let device = device(config.device_id())?;
        let dtype = DType::from_str(&config.dtype())?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(config.api_key()))
            .with_cache_dir(cache_dir)
            .build()?;

        let api = api.repo(Repo::with_revision(
            config.model_id(),
            RepoType::Model,
            config.revision(),
        ));
        let config_file_path = api.get("tokenizer.json")?;
        let tokenizer_file_path = api.get("config.json")?;

        let model_weights_file_paths = if &config.model_id() == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        {
            vec![api.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&api, "model.safetensors.index.json")?
        };

        let mut file_paths = Vec::with_capacity(2 + model_weights_file_paths.len());
        file_paths.extend(vec![config_file_path, tokenizer_file_path]);
        file_paths.extend(model_weights_file_paths);

        Ok(Self::LoadData {
            device,
            dtype,
            file_paths,
            model_type: ModelType::from_str(&config.model_id())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError> {
        let device = load_data.device;
        let dtype = load_data.dtype;
        let (model, tokenizer_filename, cache) = {
            let config_filename = load_data.file_paths[0].clone();
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

            let tokenizer_filename = load_data.file_paths[1].clone();
            let config = config.into_config(load_data.use_flash_attention);

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&load_data.file_paths[2..], dtype, &device)?
            };
            let cache = model::Cache::new(true, dtype, &config, &device)?; // TODO: use from config
            (model::Llama::load(vb, &config)?, tokenizer_filename, cache)
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        Ok(Self {
            cache,
            device,
            dtype,
            model,
            model_type: load_data.model_type,
            tokenizer,
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
        let mut logits_processor = LogitsProcessor::new(
            input.random_seed,
            Some(input.temperature),
            Some(input.top_p),
        );
        let mut index_pos = 0;
        let mut res = String::new();
        for index in 0..input.max_tokens {
            let (context_size, context_index) = if self.cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input_tensor = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
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
        Ok(res)
    }
}
