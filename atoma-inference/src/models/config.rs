use std::path::{Path, PathBuf};

use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};

use crate::models::ModelId;

type Revision = String;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    device_id: usize,
    dtype: String,
    model_id: ModelId,
    revision: Revision,
    use_flash_attention: bool,
}

impl ModelConfig {
    pub fn new(
        model_id: ModelId,
        dtype: String,
        revision: Revision,
        device_id: usize,
        use_flash_attention: bool,
    ) -> Self {
        Self {
            dtype,
            model_id,
            revision,
            device_id,
            use_flash_attention,
        }
    }

    pub fn dtype(&self) -> String {
        self.dtype.clone()
    }

    pub fn model_id(&self) -> ModelId {
        self.model_id.clone()
    }

    pub fn revision(&self) -> Revision {
        self.revision.clone()
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn use_flash_attention(&self) -> bool {
        self.use_flash_attention
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelsConfig {
    api_key: String,
    cache_dir: PathBuf,
    flush_storage: bool,
    models: Vec<ModelConfig>,
    tracing: bool,
    jrpc_port: u64,
}

impl ModelsConfig {
    pub fn new(
        api_key: String,
        cache_dir: PathBuf,
        flush_storage: bool,
        models: Vec<ModelConfig>,
        tracing: bool,
        jrpc_port: u64,
    ) -> Self {
        Self {
            api_key,
            cache_dir,
            flush_storage,
            models,
            tracing,
            jrpc_port,
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone()
    }

    pub fn flush_storage(&self) -> bool {
        self.flush_storage
    }

    pub fn models(&self) -> Vec<ModelConfig> {
        self.models.clone()
    }

    pub fn tracing(&self) -> bool {
        self.tracing
    }

    pub fn jrpc_port(&self) -> u64 {
        self.jrpc_port
    }

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
        let tracing = std::env::var("TRACING")
            .unwrap_or_default()
            .parse()
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
            tracing,
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
            true,
            18001,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "api_key = \"my_key\"\ncache_dir = \"/\"\nflush_storage = true\ntracing = true\njrpc_port = 18001\n\n[[models]]\ndevice_id = 0\ndtype = \"Llama2_7b\"\nmodel_id = \"F16\"\nrevision = \"\"\nuse_flash_attention = true\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
