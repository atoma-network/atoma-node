use std::path::PathBuf;

use config::Config;
use serde::Deserialize;

use crate::{models::ModelType, types::PrecisionBits};

#[derive(Clone, Debug, Deserialize)]
pub struct ModelTokenizer {
    pub(crate) model_type: ModelType,
    pub(crate) tokenizer: PathBuf,
    pub(crate) precision: PrecisionBits,
    pub(crate) use_kv_cache: Option<bool>,
}

#[derive(Debug, Deserialize)]
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
