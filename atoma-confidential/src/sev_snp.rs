use crate::ToBytes;

use sev::{
    attestation::{AttestationReport, AttestationReportSignature, CertTableEntry},
    error::SevError,
    firmware::Firmware,
};

use thiserror::Error;

pub const SEV_SNP_REPORT_DATA_SIZE: usize = 64;

type Result<T> = std::result::Result<T, SevError>;

/// Generates an SEV-SNP attestation report for the given compute data.
///
/// This function takes a slice of bytes representing the attested data and returns
/// an AttestationReport. The attested data must be exactly 64 bytes long,
/// matching the SEV_SNP_REPORT_DATA_SIZE constant.
///
/// # Arguments
///
/// * `attested_data` - A slice of bytes containing the data to be attested (public key)
///
/// # Returns
///
/// * `Result<AttestationReport>` - On success, returns an AttestationReport.
///                                 On failure, returns a SevError.
///
/// # Errors
///
/// Returns `SevError::InvalidAttestedDataSize` if the input data size is not 64 bytes.
/// Returns `SevError::FailedVerification` if the attestation report verification fails.
/// Returns `SevError::FailedToOpenFirmware` if the firmware interface cannot be opened.
/// Returns `SevError::FailedToGetAttestationReport` if the attestation report cannot be generated.
///
/// # Example
///
/// ```
/// let data_to_attest = [1u8; 64];
/// match get_compute_data_attestation(&data_to_attest) {
///     Ok(report) => println!("Attestation successful"),
///     Err(e) => eprintln!("Attestation failed: {}", e),
/// }
/// ```
pub fn get_compute_data_attestation(attested_data: &[u8]) -> Result<AttestationReport> {
    if attested_data.len() != SEV_SNP_REPORT_DATA_SIZE {
        return Err(SevError::InvalidAttestedDataSize(attested_data.len()));
    }

    let mut firmware = Firmware::open().map_err(SevError::FailedToOpenFirmware)?;

    // Ask the PSP to generate an attestation report for the given data
    let (attestation_report, cert_chain): (AttestationReport, Vec<CertTableEntry>) = firmware
        .get_ext_report(Some(1), Some(attested_data), Some(1))
        .map_err(SevError::FailedToGetAttestationReport)?;

    // Verify the attestation report signature using the VCEK.
    // This uses the crypto_nossl feature of the sev crate for detailed error handling and a pure Rust implementation.
    (attestation_report, cert_chain)
        .verify()
        .map_err(|e| SevError::FailedVerification(format!("Attestation report verification failed: {}", e)))?;

    Ok(attestation_report)
}

#[derive(Error, Debug)]
pub enum SevError {
    #[error("Invalid attested data size: expected {}, got {0}", SEV_SNP_REPORT_DATA_SIZE)]
    InvalidAttestedDataSize(usize),
    #[error("Failed to verify attestation report: {0}")]
    FailedVerification(#[source] std::io::Error),
    #[error("Failed to open firmware interface: {0}")]
    FailedToOpenFirmware(#[source] std::io::Error),
    #[error("Failed to get attestation report: {0}")]
    FailedToGetAttestationReport(#[source] std::io::Error),
}

// TODO
impl ToBytes for AttestationReport {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}

// TODO
impl ToBytes for CertTableEntry {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
}
