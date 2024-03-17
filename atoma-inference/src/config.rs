use crate::specs::{CudaVersion, HardwareSpec, SoftwareSpec};

pub struct InferenceConfig {
    hardware_specs: HardwareSpec,
    software_specs: SoftwareSpec,
}

impl InferenceConfig { 
    pub fn new(hardware_specs: HardwareSpec, software_specs: SoftwareSpec) -> Self { 
        Self { hardware_specs, software_specs }
    }

    pub fn hardware(&self) -> HardwareSpec { 
        self.hardware_specs.clone()
    }

    pub fn software(&self) -> SoftwareSpec {
        self.software_specs.clone()
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            hardware_specs: HardwareSpec::Gpu(crate::specs::GpuModel::Nvidia(
                crate::specs::NvidiaModel::Rtx4090(2),
            )),
            software_specs: SoftwareSpec::Cuda(CudaVersion::v11),
        }
    }
}
