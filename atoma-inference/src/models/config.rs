use std::path::{Path, PathBuf};

use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};

use crate::models::ModelId;

type Revision = String;

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Model configuration.
///
/// It contains parameters such as:
/// `device_id: usize` - ordinal specifying which GPU to host the current model
/// `dtype: String` - corresponds to the precision of which the model
///     should run (e.g., `f16`, `bf16`, `f32`, etc)
/// `model_id: String` - the model's identifier (from HuggingFace)
/// `revision: String` - needed to fetch model from HF's API
/// `use_flash_attention: bool` - if set to true, it compiles the code to run flash attention on, a highly optimized,
///     inference (and training) GPU handling of (K, V, Q) matrices multiplications. Only available if code is compiled
///     with `cuda` and `flash-attn` features on
pub struct ModelConfig {
    /// Device id that maps to the actual
    /// physical GPU card handling the current model
    device_id: usize,
    /// Dtype for decimal precision
    dtype: String,
    /// The model's identifier
    model_id: ModelId,
    /// Revision, to fetch the model from HF API
    revision: Revision,
    /// Use flash attention boolean value
    use_flash_attention: bool,
    num_shards: usize,
}

impl ModelConfig {
    /// Constructor
    pub fn new(
        model_id: ModelId,
        dtype: String,
        revision: Revision,
        device_id: usize,
        use_flash_attention: bool,
        num_shards: usize,
    ) -> Self {
        Self {
            dtype,
            model_id,
            revision,
            device_id,
            use_flash_attention,
            num_shards,
        }
    }

    /// Getter for `dtype`
    pub fn dtype(&self) -> String {
        self.dtype.clone()
    }

    /// Getter for `model_id`
    pub fn model_id(&self) -> ModelId {
        self.model_id.clone()
    }

    /// Getter for `revision`
    pub fn revision(&self) -> Revision {
        self.revision.clone()
    }

    /// Getter for `device_id`
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Getter for `use_flash_attention`
    pub fn use_flash_attention(&self) -> bool {
        self.use_flash_attention
    }

    pub fn num_shards(&self) -> usize {
        self.num_shards
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Configuration for hosting multiple models.
pub struct ModelsConfig {
    /// Node's own api key, used to fetch models
    /// from HF
    api_key: String,
    /// The path to which the models should be stored (locally)
    cache_dir: PathBuf,
    /// If set to true, on smooth shutdown, the model cache
    /// should be deleted
    flush_storage: bool,
    /// Vector holding `ModelConfig`'s for each `Model`
    /// hoster
    models: Vec<ModelConfig>,
    /// Node's JRPC service port
    jrpc_port: u64,
}

impl ModelsConfig {
    /// Constructor
    pub fn new(
        api_key: String,
        cache_dir: PathBuf,
        flush_storage: bool,
        models: Vec<ModelConfig>,
        jrpc_port: u64,
    ) -> Self {
        Self {
            api_key,
            cache_dir,
            flush_storage,
            models,
            jrpc_port,
        }
    }

    /// Getter for `api_key`
    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    /// Getter for `cache_dir`
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone()
    }

    /// Getter for `flush_storage`
    pub fn flush_storage(&self) -> bool {
        self.flush_storage
    }

    /// Getter for `models`
    pub fn models(&self) -> Vec<ModelConfig> {
        self.models.clone()
    }

    /// Getter for `jrpc_port`
    pub fn jrpc_port(&self) -> u64 {
        self.jrpc_port
    }

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

        let api_key = std::env::var("API_KEY")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let cache_dir = std::env::var("CACHE_DIR")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let flush_storage = std::env::var("FLUSH_STORAGE")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let models = serde_json::from_str(
            &std::env::var("MODELS").expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let jrpc_port = std::env::var("JRPC_PORT")
            .expect("Failed to retrieve jrpc port from .env file")
            .parse()
            .unwrap();

        Self {
            api_key,
            cache_dir,
            flush_storage,
            models,
            jrpc_port,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = ModelsConfig::new(
            "my_key".to_string(),
            "/".to_string().into(),
            true,
            vec![ModelConfig::new(
                "F16".to_string(),
                "Llama2_7b".to_string(),
                "".to_string(),
                0,
                true,
            )],
            18001,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "api_key = \"my_key\"\ncache_dir = \"/\"\nflush_storage = true\njrpc_port = 18001\n\n[[models]]\ndevice_id = 0\ndtype = \"Llama2_7b\"\nmodel_id = \"F16\"\nrevision = \"\"\nuse_flash_attention = true\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
