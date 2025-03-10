//! NVIDIA Management Library (NVML) Confidential Computing Module
//!
//! This module provides functionality for interacting with NVIDIA GPUs to obtain
//! attestation reports for confidential computing. It handles both synchronous and
//! asynchronous fetching of attestation reports, which are used to verify the
//! integrity and authenticity of GPU hardware and its execution environment.
//!
//! The attestation reports can be used in remote attestation protocols to establish
//! trust with external services or validators.

use nvml_wrapper::Nvml;

type Result<T> = std::result::Result<T, AttestationError>;

/// Fetch the attestation report for a given device index and nonce
///
/// This function performs a blocking call to the NVML library to retrieve an
/// attestation report from the specified GPU device. The nonce is used to ensure
/// freshness of the attestation report and prevent replay attacks.
///
/// # Arguments
///
/// * `index` - The index of the device to fetch the attestation report for
/// * `nonce` - The nonce to use for the attestation report (typically a cryptographic challenge)
///
/// # Returns
///
/// * `Vec<u8>` - The raw attestation report for the given device index and nonce
///
/// # Errors
///
/// * `AttestationError::NvmlError` - If the NVML library returns an error
/// * `AttestationError::InvalidDeviceIndex` - If the device index cannot be converted to u32
pub fn fetch_attestation_report(index: usize, nonce: [u8; 32]) -> Result<Vec<u8>> {
    let nvml = Nvml::init()?;
    let device = nvml.device_by_index(u32::try_from(index)?)?;
    let report = device.confidential_compute_gpu_attestation_report(nonce)?;
    let attestation_report_size = report.attestation_report_size as usize;
    let report_bytes = report.attestation_report[0..attestation_report_size].to_vec();
    Ok(report_bytes)
}

/// Fetch the attestation report for a given device index and nonce asynchronously,
/// using a blocking call to avoid blocking the main thread.
///
/// This function offloads the blocking NVML operation to Tokio's dedicated thread pool
/// for CPU-bound tasks, allowing the caller to continue processing other async tasks
/// while waiting for the attestation report.
///
/// # Arguments
///
/// * `index` - The index of the device to fetch the attestation report for
/// * `nonce` - The nonce to use for the attestation report (typically a cryptographic challenge)
///
/// # Returns
///
/// * `Vec<u8>` - The raw attestation report for the given device index and nonce
///
/// # Errors
///
/// * `AttestationError::NvmlError` - If the NVML library returns an error
/// * `AttestationError::InvalidDeviceIndex` - If the device index cannot be converted to u32
/// * `AttestationError::JoinError` - If the spawned blocking task fails to complete
pub async fn fetch_attestation_report_async(index: usize, nonce: [u8; 32]) -> Result<Vec<u8>> {
    let join_handle = tokio::task::spawn_blocking(move || fetch_attestation_report(index, nonce));
    join_handle.await?
}

#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("NVML error: {0}")]
    NvmlError(#[from] nvml_wrapper::error::NvmlError),
    #[error("Invalid device index: {0}")]
    InvalidDeviceIndex(#[from] std::num::TryFromIntError),
    #[error("Join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
}
