use std::path::PathBuf;

use config::Config;
use serde::Deserialize;

use crate::models::ModelType;

#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    api_key: String,
    models: Vec<ModelType>,
    storage_base_path: PathBuf,
    tokenizer_file_path: PathBuf,
    tracing: bool,
}

impl InferenceConfig {
    pub fn new(
        api_key: String,
        models: Vec<ModelType>,
        storage_base_path: PathBuf,
        tokenizer_file_path: PathBuf,
        tracing: bool,
    ) -> Self {
        Self {
            api_key,
            models,
            storage_base_path,
            tokenizer_file_path,
            tracing,
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn models(&self) -> Vec<ModelType> {
        self.models.clone()
    }

    pub fn storage_base_path(&self) -> PathBuf {
        self.storage_base_path.clone()
    }

    pub fn tokenizer_file_path(&self) -> PathBuf {
        self.tokenizer_file_path.clone()
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
