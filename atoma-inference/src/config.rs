use std::path::PathBuf;

use crate::{
    models::ModelType,
    specs::{HardwareSpec, SoftwareSpec},
};

pub struct InferenceConfig {
    api_key: String,
    hardware_specs: HardwareSpec,
    models: Vec<ModelType>,
    software_specs: SoftwareSpec,
    storage_base_path: PathBuf,
    tokenizer_file_path: PathBuf,
    tracing: bool,
}

impl InferenceConfig {
    pub fn new(
        api_key: String,
        hardware_specs: HardwareSpec,
        models: Vec<ModelType>,
        software_specs: SoftwareSpec,
        storage_base_path: PathBuf,
        tokenizer_file_path: PathBuf,
        tracing: bool,
    ) -> Self {
        Self {
            api_key,
            hardware_specs,
            models,
            software_specs,
            storage_base_path,
            tokenizer_file_path,
            tracing,
        }
    }

    pub fn api_key(&self) -> String {
        self.api_key.clone()
    }

    pub fn hardware(&self) -> HardwareSpec {
        self.hardware_specs.clone()
    }

    pub fn models(&self) -> Vec<ModelType> {
        self.models.clone()
    }

    pub fn software(&self) -> SoftwareSpec {
        self.software_specs.clone()
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
}
