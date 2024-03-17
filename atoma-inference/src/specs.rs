#![allow(non_camel_case_types)]

#[derive(Clone, Debug)]
pub enum HardwareSpec {
    Cpu(CpuModel),
    Gpu(GpuModel),
}

#[derive(Clone, Debug)]
pub enum CpuModel {
    Intel(IntelModel),
    Arm(ArmModel),
}

#[derive(Clone, Debug)]
pub enum GpuModel {
    Nvidia(NvidiaModel),
    Amd(AmdModel),
}

#[derive(Clone, Debug)]
pub enum IntelModel {
    x86(usize),
}

#[derive(Clone, Debug)]
pub enum ArmModel {}

#[derive(Debug, Clone)]
pub enum NvidiaModel {
    Rtx3090(usize),
    Rtx4090(usize),
    V100(usize),
    A100(usize),
    H100(usize),
}

#[derive(Clone, Debug)]
pub enum AmdModel {}

#[derive(Clone, Debug)]
pub enum SoftwareSpec {
    Jax(JaxVersion),
    Cuda(CudaVersion),
    Xla(XlaVersion),
}

#[derive(Clone, Debug)]
pub enum JaxVersion {}

#[derive(Clone, Debug)]
pub enum CudaVersion {
    v11,
}

#[derive(Clone, Debug)]
pub enum XlaVersion {}
