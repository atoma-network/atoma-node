#![allow(non_camel_case_types)]

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub enum HardwareSpec {
    Cpu(CpuModel),
    Gpu(GpuModel),
}

#[derive(Clone, Debug, Deserialize)]
pub enum CpuModel {
    Intel(IntelModel),
    Arm(ArmModel),
}

#[derive(Clone, Debug, Deserialize)]
pub enum GpuModel {
    Nvidia(NvidiaModel),
    Amd(AmdModel),
}

#[derive(Clone, Debug, Deserialize)]
pub enum IntelModel {
    x86(usize),
}

#[derive(Clone, Debug, Deserialize)]
pub enum ArmModel {}

#[derive(Debug, Clone, Deserialize)]
pub enum NvidiaModel {
    Rtx3090(usize),
    Rtx4090(usize),
    V100(usize),
    A100(usize),
    H100(usize),
}

#[derive(Clone, Debug, Deserialize)]
pub enum AmdModel {}

#[derive(Clone, Debug, Deserialize)]
pub enum SoftwareSpec {
    Jax(JaxVersion),
    Cuda(CudaVersion),
    Xla(XlaVersion),
}

#[derive(Clone, Debug, Deserialize)]
pub enum JaxVersion {}

#[derive(Clone, Debug, Deserialize)]
pub enum CudaVersion {
    v11,
}

#[derive(Clone, Debug, Deserialize)]
pub enum XlaVersion {}
