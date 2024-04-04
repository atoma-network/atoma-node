use std::path::PathBuf;

use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};

use crate::{models::types::PrecisionBits, models::ModelId};

type Revision = String;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    model_id: ModelId,
    precision: PrecisionBits,
    revision: Revision,
    device_id: usize,
}

impl ModelConfig {
    pub fn new(
        model_id: ModelId,
        precision: PrecisionBits,
        revision: Revision,
        device_id: usize,
    ) -> Self {
        Self {
            model_id,
            precision,
            revision,
            device_id,
        }
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    pub fn precision(&self) -> PrecisionBits {
        self.precision
    }

    pub fn revision(&self) -> Revision {
        self.revision.clone()
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelsConfig {
    api_key: String,
    flush_storage: bool,
    models: Vec<ModelConfig>,
    storage_path: PathBuf,
    tracing: bool,
}

impl ModelsConfig {
    pub fn new(
        api_key: String,
        flush_storage: bool,
        models: Vec<ModelConfig>,
        storage_path: PathBuf,
        tracing: bool,
    ) -> Self {
        Self {
            api_key,
            flush_storage,
            models,
            storage_path,
            tracing,
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn flush_storage(&self) -> bool {
        self.flush_storage
    }

    pub fn models(&self) -> Vec<ModelConfig> {
        self.models.clone()
    }

    pub fn storage_path(&self) -> PathBuf {
        self.storage_path.clone()
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

        let api_key = std::env::var("API_KEY").expect("Failed to retrieve api key, from .env file");
        let flush_storage = std::env::var("FLUSH_STORAGE")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let models = serde_json::from_str(
            &std::env::var("MODELS").expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let storage_path = std::env::var("STORAGE_PATH")
            .expect("Failed to retrieve storage path, from .env file")
            .parse()
            .unwrap();
        let tracing = std::env::var("TRACING")
            .unwrap_or_default()
            .parse()
            .unwrap();

        Self {
            api_key,
            flush_storage,
            models,
            storage_path,
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
            String::from("my_key"),
            true,
            vec![ModelConfig::new(
                "Llama2_7b".to_string(),
                PrecisionBits::F16,
                "".to_string(),
                0,
            )],
            "storage_path".parse().unwrap(),
            true,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "api_key = \"my_key\"\nflush_storage = true\nmodels = [[\"Llama2_7b\", \"F16\", \"\"]]\nstorage_path = \"storage_path\"\ntracing = true\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
