use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use std::path::Path;
/// `ModelConfig` - Model configuration values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hugging Face api key, optional
    pub api_key: Option<String>,
    /// Cache directory, where model data is stored
    pub cache_dir: String,
    /// If set to true, on shutdown, the model cache
    /// should be deleted
    pub flush_storage: bool,
    /// The model's name, according to HuggingFace's model hub
    pub model_name: String,
    /// The model's revision, used to fetch the model from HuggingFace's API
    pub revision: String,
    /// The device ids. If the model is running on multiple GPUs
    /// it should contains all the GPU device ids, otherwise
    /// it should contain a single device id, for the
    /// corresponding GPU.
    pub device_ids: Vec<usize>,
    /// The data type, either `bf16` or `f16`
    pub dtype: String,
}

impl ModelConfig {
    /// Creates a new `ModelConfig`
    pub fn new(
        api_key: Option<String>,
        cache_dir: String,
        flush_storage: bool,
        model_name: String,
        revision: String,
        device_ids: Vec<usize>,
        dtype: String,
    ) -> Self {
        Self {
            api_key,
            cache_dir,
            flush_storage,
            model_name,
            revision,
            device_ids,
            dtype,
        }
    }
}

impl ModelConfig {
    /// Creates a new instance of `ModelsConfig` from a file path, containing the
    /// contents of a configuration file, with the above parameters specified.
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        config
            .get::<Self>("inference")
            .expect("Failed to generated config file")
    }

    pub fn from_env_file() -> Self {
        dotenv().ok();

        let api_key = std::env::var("API_KEY").ok();
        let cache_dir = std::env::var("CACHE_DIR")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let flush_storage = std::env::var("FLUSH_STORAGE")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let model_name = serde_json::from_str(
            &std::env::var("MODEL_NAME")
                .expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let revision = serde_json::from_str(
            &std::env::var("REVISION").expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let device_ids = serde_json::from_str(
            &std::env::var("DEVICE_IDS")
                .expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let dtype = serde_json::from_str(
            &std::env::var("DTYPE").expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();

        Self {
            api_key,
            cache_dir,
            flush_storage,
            model_name,
            revision,
            device_ids,
            dtype,
        }
    }
}

/// `CacheConfig` - Cache configuration values
/// to manage how we handle KV cache memory
/// management efficiently
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CacheConfig {
    /// The device id
    pub device_id: usize,
    /// Model's Cache dtype
    pub dtype: String,
    /// Maximum (individual) sequence length
    pub max_seq_len: usize,
}
