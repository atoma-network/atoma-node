use std::{path::PathBuf, time::Instant};

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
        token_output_stream::TokenOutputStream,
        types::{LlmFetchData, LlmLoadData, TextModelInput},
        ModelError, ModelId, ModelTrait,
    },
};

pub struct MambaModel {
    model: Model,
    config: Config,
    device: Device,
    dtype: DType,
    tokenizer: TokenOutputStream,
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
            tokenizer: TokenOutputStream::new(tokenizer),
            which,
        }
    }
}

impl ModelTrait for MambaModel {
    type FetchData = LlmFetchData;
    type Input = TextModelInput;
    type Output = String;
    type LoadData = LlmLoadData;

    fn fetch(fetch_data: &Self::FetchData) -> Result<Vec<PathBuf>, ModelError> {
        let api_key = fetch_data.api_key;
        let cache_dir = fetch_data.cache_dir;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let repo = api.repo(Repo::with_revision(
            fetch_data.model_id.clone(),
            RepoType::Model,
            fetch_data.revision,
        ));

        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = api
            .model("EleutherAI/gpt-neox-20b".to_string())
            .get("tokenizer.json")?;
        let model_weights_file_path = repo.get("model.safetensors")?;

        Ok(vec![
            config_file_path,
            tokenizer_file_path,
            model_weights_file_path,
        ])
    }

    fn load(load_data: Self::LoadData) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        info!("Loading Mamba model ...");

        let start = Instant::now();

        let config_filename = load_data.file_paths[0].clone();
        let tokenizer_filename = load_data.file_paths[1].clone();
        let weights_filenames = load_data.file_paths[2..].to_vec();

        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

        let config: Config =
            serde_json::from_slice(&std::fs::read(config_filename).map_err(ModelError::IoError)?)
                .map_err(ModelError::DeserializeError)?;
        let device = device(load_data.device_id)?;
        let dtype = load_data.dtype;

        info!("Loading model weights..");
        let var_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights_filenames, dtype, &device)? };
        let model = Model::new(&config, var_builder.pp("backbone"))?;
        info!("Loaded Mamba model in {:?}", start.elapsed());

        Ok(Self::new(model, config, device, dtype, tokenizer))
    }

    fn model_id(&self) -> ModelId {
        self.which.model_id().to_string()
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

        info!("Running inference on prompt: {:?}", prompt);

        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)?
            .get_ids()
            .to_vec();
        let mut logits_processor =
            LogitsProcessor::new(random_seed, Some(temperature), Some(top_p));

        let mut generated_tokens = 0_usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => bail!("Invalid eos token"),
        };

        let mut state = State::new(1, &self.config, &self.device)?; // TODO: handle larger batch sizes

        let mut next_logits = None;
        let mut output = String::new();

        for &token in tokens.iter() {
            let input = Tensor::new(&[token], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(token)? {
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
            generated_tokens += 1;

            if next_token == eos_token {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                output.push_str(t.as_str());
            }

            let input = Tensor::new(&[next_token], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?);
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest()? {
            output.push_str(rest.as_str());
        }

        info!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(output)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    // Mamba2_8bSlimPj, TODO: add this
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            // Self::Mamba2_8bSlimPj => "state-spaces/mamba-2.8b-slimpj'",
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
