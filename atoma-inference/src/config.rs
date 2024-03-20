use std::path::PathBuf;

use crate::{
    specs::{HardwareSpec, SoftwareSpec},
    types::Model,
};

pub struct InferenceConfig {
    pub(crate) api_key: String,
    hardware_specs: HardwareSpec,
    models: Vec<Model>,
    software_specs: SoftwareSpec,
    storage_base_path: PathBuf,
}

impl InferenceConfig {
    pub fn new(
        api_key: String,
        hardware_specs: HardwareSpec,
        models: Vec<Model>,
        software_specs: SoftwareSpec,
        storage_base_path: PathBuf,
    ) -> Self {
        Self {
            api_key,
            hardware_specs,
            models,
            software_specs,
            storage_base_path,
        }
    }

    pub fn hardware(&self) -> HardwareSpec {
        self.hardware_specs.clone()
    }

    pub fn software(&self) -> SoftwareSpec {
        self.software_specs.clone()
    }

    pub fn storage_base_path(&self) -> PathBuf {
        self.storage_base_path.clone()
    }

    pub fn models(&self) -> Vec<Model> {
        self.models.clone()
    }
}
