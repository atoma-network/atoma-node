use atoma_paged_attention::models::{llama::Config, Llama};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::info;

use crate::model_executor::{ModelExecutor, ModelExecutorError, ModelLoader, ModelMetadata};

pub struct LlamaModel {
    config: Config,
    device: Device,
    dtype: DType,
    model: Llama,
}

impl ModelLoader for LlamaModel {
    type FilePaths = Vec<PathBuf>;

    fn fetch<T: AsRef<Path>>(
        api_key: String,
        cache_dir: T,
        model_id: String,
        revision: String,
    ) -> Result<Self::FilePaths, ModelLoaderError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir)
            .build()?;

        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
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

        Ok(file_paths)
    }

    fn load(
        device: Device,
        dtype: DType,
        file_paths: Self::FilePaths,
    ) -> Result<Self, ModelLoaderError>
    where
        Self: Sized,
    {
        info!("Loading Llama model ...");
        let start = Instant::now();

        let (model, tokenizer, config) = {
            let config_filename = file_paths[0].clone();
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;

            let tokenizer_filename = file_paths[1].clone();
            let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&file_paths[2..], dtype, &device)? };
            (model::Llama::load(vb, &config)?, tokenizer, config)
        };
        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self {
            model,
            config,
            device,
            dtype,
        })
    }
}

impl ModelMetadata for LlamaModel {
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn softmax_scale(&self) -> f32 {
        let head_dim = self.hidden_size() / self.num_attention_heads();
        1f32 / (head_dim as f32).sqrt()
    }

    fn sliding_window(&self) -> Option<usize> {
        None
    }
}

impl ModelExecutor for LlamaModel {
    fn forward(
        &mut self,
        input: &Tensor,
        input_positions: &Tensor,
        selected_token_positions: &Tensor,
        kv_cache: Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError> {
        Ok(self.model.forward(
            input,
            input_positions,
            selected_token_positions,
            kv_cache,
            attention_metadata,
        )?)
    }
}
