use std::{path::PathBuf, str::FromStr};

use crate::specs::{CudaVersion, HardwareSpec, SoftwareSpec};

pub struct InferenceConfig {
    hardware_specs: HardwareSpec,
    hugging_face_api_key: String, // TODO: make this generic for any available Web2/Web3 API
    software_specs: SoftwareSpec,
    storage_base_path: PathBuf,
}

impl InferenceConfig {
    pub fn new(
        hardware_specs: HardwareSpec,
        hugging_face_api_key: String,
        software_specs: SoftwareSpec,
        storage_base_path: PathBuf,
    ) -> Self {
        Self {
            hardware_specs,
            hugging_face_api_key,
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

    pub fn hugging_face_api_key(&self) -> String {
        self.hugging_face_api_key.clone()
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            hardware_specs: HardwareSpec::Gpu(crate::specs::GpuModel::Nvidia(
                crate::specs::NvidiaModel::Rtx4090(2),
            )),
            hugging_face_api_key: String::new(), // TODO: for now it is an empty key
            software_specs: SoftwareSpec::Cuda(CudaVersion::v11),
            storage_base_path: PathBuf::from_str("./").unwrap(),
        }
    }
}
