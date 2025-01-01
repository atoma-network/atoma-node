use crate::ToBytes;
use dcap_rs::types::quotes::{body::QuoteBody, version_4::QuoteV4};
use tdx::{device::{Device, DeviceOptions}, error::TdxError as DeviceError};
use thiserror::Error;

/// The size of the data to be attested, for a intel TDX quote.
pub const TDX_REPORT_DATA_SIZE: usize = 64;

type Result<T> = std::result::Result<T, TdxError>;

/// Generates a TDX attestation report for the given compute data.
///
/// This function takes a slice of bytes representing the attested data and returns
/// a QuoteV4 attestation report. The attested data must be exactly 64 bytes long,
/// matching the TDX_REPORT_DATA_SIZE constant.
///
/// # Arguments
///
/// * `attested_data` - A slice of bytes containing the data to be attested.
///
/// # Returns
///
/// * `Result<QuoteV4>` - On success, returns a QuoteV4 attestation report.
///                       On failure, returns a TdxError.
///
/// # Errors
///
/// Returns `TdxError::InvalidAttestedDataSize` if the input data size is not 64 bytes.
///
/// # Example
///
/// ```
/// let data_to_attest = [1u8; 64];
/// match get_compute_data_attestation(&data_to_attest) {
///     Ok(quote) => println!("Attestation successful"),
///     Err(e) => eprintln!("Attestation failed: {}", e),
/// }
/// ```
pub fn get_compute_data_attestation(attested_data: &[u8]) -> Result<QuoteV4> {
    if attested_data.len() != TDX_REPORT_DATA_SIZE {
        return Err(TdxError::InvalidAttestedDataSize(attested_data.len()));
    }
    let mut report_data = [0u8; TDX_REPORT_DATA_SIZE];
    report_data[..attested_data.len()].copy_from_slice(attested_data);
    let device = Device::new(DeviceOptions {
            report_data: Some(report_data),
        })?;
    device
        .get_attestation_report()
}

impl ToBytes for QuoteV4 {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.header.to_bytes());
        // TODO: Do we need to handle the enum identifier as well ?
        match self.quote_body {
            QuoteBody::SGXQuoteBody(enclave_report) => {
                bytes.extend_from_slice(&enclave_report.to_bytes())
            }
            QuoteBody::TD10QuoteBody(td10_report) => {
                bytes.extend_from_slice(&td10_report.to_bytes())
            }
        }
        bytes.extend_from_slice(&self.signature_len.to_le_bytes());
        bytes.extend_from_slice(&self.signature.quote_signature);
        bytes.extend_from_slice(&self.signature.ecdsa_attestation_key);
        bytes.extend_from_slice(&self.signature.qe_cert_data.cert_data_type.to_le_bytes());
        bytes.extend_from_slice(&self.signature.qe_cert_data.cert_data_size.to_le_bytes());
        bytes.extend_from_slice(&self.signature.qe_cert_data.cert_data);
        bytes
    }
}

#[derive(Error, Debug)]
pub enum TdxError {
    #[error("Invalid attested data size: {0}")]
    InvalidAttestedDataSize(usize),
    #[error("TDX device error: {0}")]
    TdxDeviceError(#[from] DeviceError),
}
