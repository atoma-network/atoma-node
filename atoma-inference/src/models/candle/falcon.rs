use std::{path::PathBuf, str::FromStr, time::Instant};

use atoma_types::AtomaStreamingData;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::falcon::{Config, Falcon},
    utils::apply_repeat_penalty,
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument};

use crate::models::{
    candle::hub_load_safetensors,
    config::ModelConfig,
    token_output_stream::TokenOutputStream,
    types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
    ModelError, ModelTrait,
};

use super::device;

/// `FalconModel` - encapsulates a Falcon model
/// together with additional metadata, necessary
/// to run inference
pub struct FalconModel {
    /// The actual Falcon model
    model: Falcon,
    /// The device holding the model
    /// weights, while running inference
    device: Device,
    /// The decimal precisions of the model's weights
    dtype: DType,
    /// The model's unique identifier
    model_type: ModelType,
    /// Tokenizer, with streaming functionality
    tokenizer: TokenOutputStream,
}

impl FalconModel {
    /// Constructor
    pub fn new(
        model: Falcon,
        device: Device,
        dtype: DType,
        model_type: ModelType,
        tokenizer: Tokenizer,
        stream_tx: mpsc::Sender<AtomaStreamingData>,
    ) -> Self {
        Self {
            model,
            device,
            dtype,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
            model_type,
        }
    }
}

impl ModelTrait for FalconModel {
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

        info!("{repo_id} <> {revision}");

        let mut file_paths = vec![];
        let repo = api.repo(Repo::new(repo_id.clone(), RepoType::Model));
        file_paths.push(repo.get("config.json")?);

        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));
        file_paths.push(repo.get("tokenizer.json")?);

        file_paths.extend(
            hub_load_safetensors(&repo, "model.safetensors.index.json").map_err(|e| {
                error!("{e}");
                e
            })?,
        );

        Ok(Self::LoadData {
            device,
            dtype,
            file_paths,
            model_type: ModelType::from_str(&config.model_id())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    #[instrument(skip_all)]
    fn load(
        load_data: Self::LoadData,
        stream_tx: mpsc::Sender<AtomaStreamingData>,
    ) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Falcon model ...");

        let start = Instant::now();

        let config_filename = load_data.file_paths[0].clone();
        let tokenizer_filename = load_data.file_paths[1].clone();
        let weights_filenames = load_data.file_paths[2..].to_vec();

        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
        let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        config.validate()?;

        if load_data.dtype != DType::BF16 && load_data.dtype != DType::F32 {
            panic!("Invalid dtype, it must be either BF16 or F32 precision");
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weights_filenames,
                load_data.dtype,
                &load_data.device,
            )?
        };
        let model = Falcon::load(vb, config.clone())?;
        info!("Loaded Falcon model in {:?}", start.elapsed());

        Ok(Self::new(
            model,
            load_data.device,
            load_data.dtype,
            load_data.model_type,
            tokenizer,
            stream_tx,
        ))
    }

    fn model_type(&self) -> ModelType {
        self.model_type.clone()
    }

    #[instrument(skip_all)]
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        info!("Running inference on prompt: {:?}", input.prompt);
        self.tokenizer.clear();

        self.model.clear_kv_cache();
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

        let mut logits_processor = LogitsProcessor::new(random_seed, Some(temperature), top_p);
        info!("Running inference on prompt: {:?}", prompt);
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)?
            .get_ids()
            .to_vec();
        let mut tokens = [input.pre_prompt_tokens, tokens].concat();

        let input_tokens = tokens.len();

        let request_id = Some(input.request_id).filter(|_| input.should_stream_output);
        let mut generated_tokens = 0;
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
            let input_tensor = Tensor::new(ctx, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_tensor)?;
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

            if let Some(t) = self.tokenizer.next_token(next_token, request_id.clone())? {
                output += &t;
            }

            generated_tokens += 1;
        }
        if let Some(rest) = self.tokenizer.decode_rest(request_id.clone())? {
            output += &rest;
        }
        let dt = start_gen.elapsed();

        info!(
            "{generated_tokens} tokens generated ({} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        if input.should_stream_output {
            info!("Ending stream");
            self.tokenizer.end_stream(request_id.unwrap())?;
        }

        Ok(TextModelOutput {
            text: output,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
            input_tokens,
            tokens: if input.chat { tokens } else { vec![] },
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "metal")]
    fn test_falcon_model_interface_with_metal() {
        use super::*;

        let api_key = "".to_string();
        let cache_dir: PathBuf = "./test_falcon_cache_dir/".try_into().unwrap();
        let model_id = "falcon_7b".to_string();
        let dtype = "f32".to_string();
        let revision = "refs/pr/43".to_string();
        let device_id = 0;
        let use_flash_attention = false;
        let config = ModelConfig::new(
            model_id,
            dtype.clone(),
            revision,
            vec![device_id],
            use_flash_attention,
        );
        let load_data = FalconModel::fetch(api_key, cache_dir.clone(), config)
            .expect("Failed to fetch falcon model");

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

        assert_eq!(load_data.file_paths.len(), 4);
        assert_eq!(load_data.use_flash_attention, use_flash_attention);
        assert_eq!(load_data.model_type, ModelType::Falcon7b);

        let should_be_dtype = DType::from_str(&dtype).unwrap();
        assert_eq!(load_data.dtype, should_be_dtype);
        let (sender, _) = mpsc::channel(1);
        let mut model = FalconModel::load(load_data, sender).expect("Failed to load model");

        if should_be_device.is_cpu() {
            assert!(model.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(model.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(model.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(model.dtype, should_be_dtype);
        assert_eq!(model.model_type, ModelType::Falcon7b);

        let prompt = "Write a hello world rust program: ".to_string();
        let temperature = 0.6;
        let random_seed = 42;
        let repeat_penalty = 1.0;
        let repeat_last_n = 20;
        let max_tokens = 1;
        let top_k = 10;
        let top_p = 0.6;

        let input = TextModelInput::new(
            "".to_string(),
            prompt.clone(),
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            Some(top_k),
            Some(top_p as f64),
            false,
            vec![],
            false,
        );
        let output = model.run(input).expect("Failed to run inference");

        assert!(output.text.len() >= 1);
        assert!(output.text.split(" ").collect::<Vec<_>>().len() <= max_tokens);

        std::fs::remove_dir_all(cache_dir).unwrap();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_falcon_model_interface_with_cuda() {
        use super::*;

        let api_key = "".to_string();
        let cache_dir: PathBuf = "./test_falcon_cache_dir/".try_into().unwrap();
        let model_id = "falcon_7b".to_string();
        let dtype = "f32".to_string();
        let revision = "refs/pr/43".to_string();
        let device_id = 0;
        let use_flash_attention = false;
        let config = ModelConfig::new(
            model_id,
            dtype.clone(),
            revision,
            vec![device_id],
            use_flash_attention,
        );
        let load_data = FalconModel::fetch(api_key, cache_dir.clone(), config)
            .expect("Failed to fetch falcon model");

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
        assert_eq!(load_data.model_type, ModelType::Mamba130m);

        let should_be_dtype = DType::from_str(&dtype).unwrap();
        assert_eq!(load_data.dtype, should_be_dtype);
        let (sender, _) = mpsc::channel(1);
        let mut model = FalconModel::load(load_data, sender).expect("Failed to load model");

        if should_be_device.is_cpu() {
            assert!(model.device.is_cpu());
        } else if should_be_device.is_cuda() {
            assert!(model.device.is_cuda());
        } else if should_be_device.is_metal() {
            assert!(model.device.is_metal());
        } else {
            panic!("Invalid device")
        }

        assert_eq!(model.dtype, should_be_dtype);
        assert_eq!(model.model_type, ModelType::Mamba130m);

        let prompt = "Write a hello world rust program: ".to_string();
        let temperature = 0.6;
        let random_seed = 42;
        let repeat_penalty = 1.0;
        let repeat_last_n = 20;
        let max_tokens = 1;
        let top_k = 10;
        let top_p = 0.6;

        let input = TextModelInput::new(
            "".to_string(),
            prompt.clone(),
            temperature,
            random_seed,
            repeat_penalty,
            repeat_last_n,
            max_tokens,
            Some(top_k),
            Some(top_p as f64),
            false,
            vec![],
            false,
        );
        let output = model.run(input).expect("Failed to run inference");
        println!("{output}");

        assert!(output.text.len() >= 1);
        assert!(output.text.split(" ").collect::<Vec<_>>().len() <= max_tokens);

        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
