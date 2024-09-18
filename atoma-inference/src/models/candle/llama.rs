use std::{path::PathBuf, str::FromStr, time::Instant};

use atoma_types::AtomaStreamingData;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Config, LlamaConfig},
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use candle_transformers::models::llama as model;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{info, instrument};

use crate::models::{
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
    ModelError, ModelTrait,
};

use super::{device, hub_load_safetensors};

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "</s>";

/// `LlamaModel` - encapsulates a Llama model
/// together with additional metadata, necessary
/// to run inference
pub struct LlamaModel {
    /// The device holding the model
    /// weights, while running inference
    device: Device,
    /// The actual Llama model
    model: model::Llama,
    /// The model's unique identifier
    model_type: ModelType,
    /// Tokenizer, with streaming functionality
    tokenizer: TokenOutputStream,
    /// Llama's configuration
    config: Config,
    /// The model weights decimal precision
    dtype: DType,
}

impl ModelTrait for LlamaModel {
    type Input = TextModelInput;
    type Output = TextModelOutput;
    type LoadData = LlmLoadData;

    #[instrument(skip_all)]
    fn fetch(
        api_key: String,
        cache_dir: PathBuf,
        config: ModelConfig,
    ) -> Result<Self::LoadData, ModelError> {
        let device = device(config.device_first_id())?;
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

    #[instrument(skip_all)]
    fn load(
        load_data: Self::LoadData,
        stream_tx: mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError> {
        info!("Loading Llama model ...");

        let start = Instant::now();

        let device = load_data.device;
        let dtype = load_data.dtype;
        let (model, tokenizer, config) = {
            let config_filename = load_data.file_paths[0].clone();
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

            let config = config.into_config(load_data.use_flash_attention);

            let tokenizer_filename = load_data.file_paths[1].clone();
            let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&load_data.file_paths[2..], dtype, &device)?
            };
            (model::Llama::load(vb, &config)?, tokenizer, config)
        };
        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self {
            device,
            model,
            model_type: load_data.model_type,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
            config,
            dtype,
        })
    }

    #[instrument(skip_all)]
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        info!("Running inference on prompt: {:?}", input.prompt);
        self.tokenizer.clear();

        let bos_token_id = self
            .config
            .bos_token_id
            .or_else(|| self.tokenizer.tokenizer().token_to_id(BOS_TOKEN))
            .unwrap();
        let eos_token_id = self.config.eos_token_id.clone().or_else(|| {
            self.tokenizer
                .tokenizer()
                .token_to_id(EOS_TOKEN)
                .map(model::LlamaEosToks::Single)
        });
        let prompt_ids = self
            .tokenizer
            .tokenizer()
            .encode(input.prompt.clone(), true)?
            .get_ids()
            .to_vec();
        let tokens = if self.model_type == ModelType::Llama3_8b {
            vec![bos_token_id].into_iter().chain(prompt_ids).collect()
        } else {
            prompt_ids
        };
        let mut tokens = [input.pre_prompt_tokens, tokens].concat();
        let input_tokens = tokens.len();

        let mut logits_processor =
            LogitsProcessor::new(input.random_seed, Some(input.temperature), input.top_p);
        let mut index_pos = 0;
        let mut res = String::new();
        let mut generated_tokens = 0;

        let request_id = Some(input.request_id).filter(|_| input.should_stream_output);
        let mut cache = model::Cache::new(true, self.dtype, &self.config, &self.device)?;
        let start_gen = Instant::now();

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

            match eos_token_id {
                Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                    break;
                }
                Some(model::LlamaEosToks::Multiple(ref eos_ids))
                    if eos_ids.contains(&next_token) =>
                {
                    break;
                }
                _ => (),
            }
            if let Some(t) = self.tokenizer.next_token(next_token, request_id.clone())? {
                res += &t;
            }

            generated_tokens += 1;
        }
        if let Some(rest) = self.tokenizer.decode_rest(request_id.clone())? {
            res += &rest;
        }

        let dt = start_gen.elapsed();
        info!(
            "{generated_tokens} tokens generated ({} token/s)\n",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        if input.should_stream_output {
            info!("Ending stream");
            self.tokenizer.end_stream(request_id.unwrap())?;
        }

        Ok(TextModelOutput {
            text: res,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
            input_tokens,
            tokens: if input.chat { tokens } else { vec![] },
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
            vec![device_id],
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

        let (stream_tx, _) = mpsc::channel(1);
        let mut model = LlamaModel::load(load_data, stream_tx).expect("Failed to load model");

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
            String::new(),
            prompt.clone(),
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            Some(top_k),
            Some(top_p),
            false,
            vec![],
            false,
        );
        let output = model.run(input).expect("Failed to run inference");
        println!("{output}");

        assert!(output.text.len() > 1);
        assert!(output.text.split(' ').collect::<Vec<_>>().len() <= max_tokens);

        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
