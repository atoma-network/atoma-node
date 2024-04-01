use std::path::PathBuf;

use config::Config;
use serde::{Deserialize, Serialize};

use crate::{models::ModelId, types::PrecisionBits};

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    api_key: String,
    models: Vec<(ModelId, PrecisionBits)>,
    storage_path: PathBuf,
    tracing: bool,
}

impl ModelConfig {
    pub fn new(
        api_key: String,
        models: Vec<(ModelId, PrecisionBits)>,
        storage_path: PathBuf,
        tracing: bool,
    ) -> Self {
        Self {
            api_key,
            models,
            storage_path,
            tracing,
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn model_ids(&self) -> Vec<(ModelId, PrecisionBits)> {
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
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = ModelConfig::new(
            String::from("my_key"),
            vec![("Llama2_7b".to_string(), PrecisionBits::F16)],
            "storage_path".parse().unwrap(),
            true,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "api_key = \"my_key\"\nmodels = [[\"Llama2_7b\", \"F16\"]]\nstorage_path = \"storage_path\"\ntracing = true\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
