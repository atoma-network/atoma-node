use std::{path::PathBuf, str::FromStr, sync::mpsc, time::Instant};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::mamba::{Config, Model, State},
    utils::apply_repeat_penalty,
};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    bail,
    models::{
        candle::device,
        config::ModelConfig,
        token_output_stream::TokenOutputStream,
        types::{LlmLoadData, ModelType, TextModelInput, TextModelOutput},
        ModelError, ModelTrait,
    },
};

pub struct MambaModel {
    model: Model,
    config: Config,
    device: Device,
    dtype: DType,
    model_type: ModelType,
    tokenizer: TokenOutputStream,
}

impl MambaModel {
    pub fn new(
        model: Model,
        config: Config,
        device: Device,
        dtype: DType,
        model_type: ModelType,
        tokenizer: Tokenizer,
        stream_tx: mpsc::Sender<String>,
    ) -> Self {
        Self {
            model,
            config,
            device,
            dtype,
            model_type,
            tokenizer: TokenOutputStream::new(tokenizer, stream_tx),
        }
    }
}

impl ModelTrait for MambaModel {
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

        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = api
            .model("EleutherAI/gpt-neox-20b".to_string())
            .get("tokenizer.json")?;
        let model_weights_file_path = repo.get("model.safetensors")?;

        Ok(Self::LoadData {
            device,
            dtype,
            file_paths: vec![
                config_file_path,
                tokenizer_file_path,
                model_weights_file_path,
            ],
            model_type: ModelType::from_str(&config.model_id())?,
            use_flash_attention: config.use_flash_attention(),
        })
    }

    fn load(load_data: Self::LoadData, stream_tx: mpsc::Sender<String>) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Mamba model ...");

        let start = Instant::now();

        let config_filename = load_data.file_paths[0].clone();
        let tokenizer_filename = load_data.file_paths[1].clone();
        let weights_filenames = load_data.file_paths[2..].to_vec();

        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

        let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;

        info!("Loading model weights..");
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weights_filenames,
                load_data.dtype,
                &load_data.device,
            )?
        };
        let model = Model::new(&config, var_builder.pp("backbone"))?;
        info!("Loaded Mamba model in {:?}", start.elapsed());

        Ok(Self::new(
            model,
            config,
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

        // clean tokenizer state
        self.tokenizer.clear();

        info!("Running inference on prompt: {:?}", prompt);

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)?
            .get_ids()
            .to_vec();
        let mut logits_processor = LogitsProcessor::new(random_seed, Some(temperature), top_p);

        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => bail!("Invalid eos token"),
        };

        let mut state = State::new(1, &self.config, self.dtype, &self.device)?; // TODO: handle larger batch sizes

        let mut next_logits = None;
        let mut output = String::new();

        for &token in tokens.iter() {
            let input_tensor = Tensor::new(&[token], &self.device)?;
            let logits = self.model.forward(&input_tensor, &mut state)?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(token, input.stream)? {
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

            if next_token == eos_token {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token, input.stream)? {
                output.push_str(t.as_str());
            }

            let input = Tensor::new(&[next_token], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?);
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest(input.stream)? {
            output.push_str(rest.as_str());
        }

        let generated_tokens = self.tokenizer.get_num_generated_tokens();
        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(TextModelOutput {
            text: output,
            time: dt.as_secs_f64(),
            tokens_count: generated_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_model_interface() {
        let api_key = "".to_string();
        let cache_dir: PathBuf = "./test_mamba_cache_dir/".into();
        let model_id = "mamba_130m".to_string();
        let dtype = "f32".to_string();
        let revision = "refs/pr/1".to_string();
        let device_id = 0;
        let use_flash_attention = false;
        let config = ModelConfig::new(
            model_id,
            dtype.clone(),
            revision,
            device_id,
            use_flash_attention,
        );
        let load_data = MambaModel::fetch(api_key, cache_dir.clone(), config)
            .expect("Failed to fetch mamba model");

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

        let (stream_tx, _) = mpsc::channel();
        let mut model = MambaModel::load(load_data, stream_tx).expect("Failed to load model");

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
            Some(top_k),
            Some(top_p),
            false,
        );
        let output = model.run(input).expect("Failed to run inference");
        println!("{output}");

        assert!(output.text.contains(&prompt));
        assert!(output.text.len() > prompt.len());

        std::fs::remove_dir_all(cache_dir).unwrap();
    }
}
