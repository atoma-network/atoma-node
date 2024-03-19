use std::{path::PathBuf, str::FromStr};

use crate::specs::{CudaVersion, HardwareSpec, SoftwareSpec};

pub struct InferenceConfig {
    hardware_specs: HardwareSpec,
    software_specs: SoftwareSpec,
    storage_base_path: PathBuf,
}

impl InferenceConfig {
    pub fn new(
        hardware_specs: HardwareSpec,
        software_specs: SoftwareSpec,
        storage_base_path: PathBuf,
    ) -> Self {
        Self {
            hardware_specs,
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
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            hardware_specs: HardwareSpec::Gpu(crate::specs::GpuModel::Nvidia(
                crate::specs::NvidiaModel::Rtx4090(2),
            )),
            software_specs: SoftwareSpec::Cuda(CudaVersion::v11),
            storage_base_path: PathBuf::from_str("./").unwrap(),
        }
    }
}
