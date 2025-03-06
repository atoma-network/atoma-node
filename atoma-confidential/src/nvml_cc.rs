use nvml_wrapper::Nvml;

type Result<T> = std::result::Result<T, AttestationError>;

/// Fetch the attestation report for a given device index and nonce
///
/// # Arguments
///
/// * `index` - The index of the device to fetch the attestation report for
/// * `nonce` - The nonce to use for the attestation report
///
/// # Returns
///
/// * `Vec<u8>` - The attestation report for the given device index and nonce
///
/// # Errors
///
/// * `AttestationError::NvmlError` - If the NVML library returns an error
pub fn fetch_attestation_report(index: usize, nonce: [u8; 32]) -> Result<Vec<u8>> {
    let nvml = Nvml::init()?;
    let device = nvml.device_by_index(u32::try_from(index)?)?;
    let report = device.confidential_compute_gpu_attestation_report(nonce)?;
    let attestation_report_size = report.attestation_report_size as usize;
    let report_bytes = report.attestation_report[0..attestation_report_size].to_vec();
    Ok(report_bytes)
}

/// Fetch the attestation report for a given device index and nonce asynchronously
///
/// # Arguments
///
/// * `index` - The index of the device to fetch the attestation report for
/// * `nonce` - The nonce to use for the attestation report
///
/// # Returns
///
/// * `Vec<u8>` - The attestation report for the given device index and nonce
///
/// # Errors
///
/// * `AttestationError::JoinError` - If the join handle returns an error
pub async fn fetch_attestation_report_async(index: usize, nonce: [u8; 32]) -> Result<Vec<u8>> {
    let join_handle = tokio::spawn(async move { fetch_attestation_report(index, nonce) });
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
