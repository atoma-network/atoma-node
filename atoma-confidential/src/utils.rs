use flate2::{write::ZlibEncoder, Compression};
use std::io::Write;
use thiserror::Error;
use tracing::error;

type Result<T> = std::result::Result<T, CompressionError>;

/// Compress the attestation report bytes using zlib compression
///
/// # Arguments
///
/// * `report_bytes` - The bytes to compress
///
/// # Returns
///
/// * `Vec<u8>` - The compressed bytes
///
/// # Errors
///
/// * `Error::CompressError` - If the compression fails
pub fn compress_attestation_report_bytes(report_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(report_bytes)?;
    Ok(encoder.finish()?)
}

#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("Failed to compress attestation report: {0}")]
    CompressError(#[from] std::io::Error),
}
