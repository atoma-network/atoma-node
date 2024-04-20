use std::{path::PathBuf, str::FromStr, time::Instant};

use candle::{DType, Device, Tensor};
use candle_examples::llama_multiprocess::model::Llama as ShardedLlama;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Config, LlamaConfig},
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use candle_transformers::models::llama as model;
use tokenizers::Tokenizer;
use tracing::info;

use crate::models::{
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
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

pub struct LlamaModel {
    device: Device,
    model: model::Llama,
    model_type: ModelType,
    tokenizer: Tokenizer,
    config: Config,
    dtype: DType,
}

impl ModelTrait for LlamaModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
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

        let repo = api.repo(Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        let model_weights_file_paths = if &repo_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0" {
            vec![repo.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
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
        info!("Loading Llama model ...");

        let start = Instant::now();

        let device = load_data.device;
        let dtype = load_data.dtype;
        let (model, tokenizer_filename, config) = {
            let config_filename = load_data.file_paths[0].clone();
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

            let tokenizer_filename = load_data.file_paths[1].clone();
            let config = config.into_config(load_data.use_flash_attention);

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&load_data.file_paths[2..], dtype, &device)?
            };
            (model::Llama::load(vb, &config)?, tokenizer_filename, config)
        };
        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self {
            device,
            model,
            model_type: load_data.model_type,
            tokenizer,
            config,
            dtype,
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
        let mut generated_tokens = 0;

        let start_gen = Instant::now();
        let mut cache = model::Cache::new(true, self.dtype, &self.config, &self.device)?;
        for index in 0..input.max_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input_tensor = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input_tensor, context_index, &mut cache)?;
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

            generated_tokens += 1;
        }
        if let Some(rest) = tokenizer.decode_rest()? {
            res += &rest;
        }

        let dt = start_gen.elapsed();
        info!(
            "{generated_tokens} tokens generated ({} token/s)\n",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(TextModelOutput {
            text: res,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_model_interface() {
        let api_key = "".to_string();
        let cache_dir: PathBuf = "./test_llama_cache_dir/".into();
        let model_id = "llama_tiny_llama_1_1b_chat".to_string();
        let dtype = "f32".to_string();
        let revision = "main".to_string();
        let device_id = 0;
        let use_flash_attention = false;
        let config = ModelConfig::new(
            model_id,
            dtype.clone(),
            revision,
            device_id,
            use_flash_attention,
        );
        let load_data = LlamaModel::fetch(api_key, cache_dir.clone(), config)
            .expect("Failed to fetch llama model");

        println!("model device = {:?}", load_data.device);
        let should_be_device = device(device_id).unwrap();
        if should_be_device.is_cpu() {
            assert!(load_data.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(load_data.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(load_data.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(load_data.file_paths.len(), 3);
        assert_eq!(load_data.use_flash_attention, use_flash_attention);
        assert_eq!(load_data.model_type, ModelType::LlamaTinyLlama1_1BChat);

        let should_be_dtype = DType::from_str(&dtype).unwrap();
        assert_eq!(load_data.dtype, should_be_dtype);
        let mut model = LlamaModel::load(load_data).expect("Failed to load model");

        if should_be_device.is_cpu() {
            assert!(model.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(model.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(model.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(model.model_type, ModelType::LlamaTinyLlama1_1BChat);

        let prompt = "Write a hello world rust program: ".to_string();
        let temperature = 0.6;
        let random_seed = 42;
        let repeat_penalty = 1.0;
        let repeat_last_n = 20;
        let max_tokens = 128;
        let top_k = 10;
        let top_p = 0.6;

        let input = TextModelInput::new(
            prompt.clone(),
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            top_k,
            top_p,
        );
        let output = model.run(input).expect("Failed to run inference");
        println!("{output}");

        assert!(output.text.len() > 1);
        assert!(output.text.split(' ').collect::<Vec<_>>().len() <= max_tokens);

        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
