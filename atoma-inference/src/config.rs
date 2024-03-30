use std::path::PathBuf;

use config::Config;
use serde::{Deserialize, Serialize};

use crate::{models::ModelType, types::PrecisionBits};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelTokenizer {
    pub(crate) model_type: ModelType,
    pub(crate) tokenizer: PathBuf,
    pub(crate) precision: PrecisionBits,
    pub(crate) use_kv_cache: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InferenceConfig {
    api_key: String,
    models: Vec<ModelTokenizer>,
    storage_folder: PathBuf,
    tracing: bool,
}

impl InferenceConfig {
    pub fn new(
        api_key: String,
        models: Vec<ModelTokenizer>,
        storage_folder: PathBuf,
        tracing: bool,
    ) -> Self {
        Self {
            api_key,
            models,
            storage_folder,
            tracing,
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn models(&self) -> Vec<ModelTokenizer> {
        self.models.clone()
    }

    pub fn storage_folder(&self) -> PathBuf {
        self.storage_folder.clone()
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
        let config = InferenceConfig::new(
            String::from("my_key"),
            vec![ModelTokenizer {
                model_type: ModelType::Llama2_7b,
                tokenizer: "tokenizer".parse().unwrap(),
                precision: PrecisionBits::BF16,
                use_kv_cache: Some(true),
            }],
            "storage_folder".parse().unwrap(),
            true,
        );

        let toml_str = toml::to_string(&config).unwrap();
        let should_be_toml_str = "api_key = \"my_key\"\nstorage_folder = \"storage_folder\"\ntracing = true\n\n[[models]]\nmodel_type = \"Llama2_7b\"\ntokenizer = \"tokenizer\"\nprecision = \"BF16\"\nuse_kv_cache = true\n";
        assert_eq!(toml_str, should_be_toml_str);
    }
}
