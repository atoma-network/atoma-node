use std::path::PathBuf;

use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};

use crate::models::ModelId;

type Revision = String;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    api_key: String,
    cache_dir: String,
    device_id: usize,
    dtype: String,
    model_id: ModelId,
    revision: Revision,
    use_flash_attention: bool,
    sliced_attention_size: Option<usize>,
}

impl ModelConfig {
    pub fn new(
        api_key: String,
        cache_dir: String,
        model_id: ModelId,
        dtype: String,
        revision: Revision,
        device_id: usize,
        use_flash_attention: bool,
        sliced_attention_size: Option<usize>,
    ) -> Self {
        Self {
            api_key,
            cache_dir,
            dtype,
            model_id,
            revision,
            device_id,
            use_flash_attention,
            sliced_attention_size
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn cache_dir(&self) -> String {
        self.cache_dir.clone()
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

    pub fn sliced_attention_size(&self) -> Option<usize> { 
        self.sliced_attention_size
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelsConfig {
    flush_storage: bool,
    models: Vec<ModelConfig>,
    tracing: bool,
}

impl ModelsConfig {
    pub fn new(flush_storage: bool, models: Vec<ModelConfig>, tracing: bool) -> Self {
        Self {
            flush_storage,
            models,
            tracing,
        }
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

    pub fn from_file_path(config_file_path: PathBuf) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.to_str().as_ref().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        config
            .try_deserialize::<Self>()
            .expect("Failed to generated config file")
    }

    pub fn from_env_file() -> Self {
        dotenv().ok();

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

        Self {
            flush_storage,
            models,
            tracing,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = ModelsConfig::new(
            true,
            vec![ModelConfig::new(
                "my_key".to_string(),
                "/".to_string(),
                "F16".to_string(),
                "Llama2_7b".to_string(),
                "".to_string(),
                0,
                true,
                Some(0)
            )],
            true,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "api_key = \"my_key\"\nflush_storage = true\nmodels = [[\"Llama2_7b\", \"F16\", \"\"]]\nstorage_path = \"storage_path\"\ntracing = true\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
