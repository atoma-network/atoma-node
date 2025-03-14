use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use std::io::{Read, Write};
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
pub fn compress_bytes(report_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(report_bytes)
        .map_err(|e| CompressionError::CompressError(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| CompressionError::CompressError(e.to_string()))
}

/// Decompress the attestation report bytes using zlib decompression
///
/// # Arguments
///
/// * `compressed_bytes` - The compressed bytes to decompress
///
/// # Returns
///
/// * `Vec<u8>` - The decompressed bytes
///
/// # Errors
///
/// * `Error::DecompressError` - If the decompression fails
pub fn decompress_bytes(compressed_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(compressed_bytes);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| CompressionError::DecompressError(e.to_string()))?;
    Ok(decompressed)
}

#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("Failed to compress attestation report: {0}")]
    CompressError(String),

    // Note: We're reusing the same error type since std::io::Error covers both
    // compression and decompression errors, but with a different message
    #[error("Failed to decompress attestation report: {0}")]
    DecompressError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_decompression_roundtrip() {
        // Test data of various sizes
        let test_cases = vec![
            vec![],              // Empty data
            vec![1, 2, 3, 4, 5], // Small data
            vec![42; 1000],      // Medium data with repetition (compressible)
            (0..10000)
                .map(|i| u8::try_from(i % 256).unwrap())
                .collect::<Vec<_>>(), // Large data with some patterns
            // Random-like data (less compressible)
            (0..5000u64)
                .map(|i| u8::try_from((i * 1_664_525 + 1_013_904_223) % 256).unwrap())
                .collect::<Vec<_>>(),
        ];

        for original_data in test_cases {
            // Compress the data
            let compressed = compress_bytes(&original_data).expect("Compression should succeed");

            // For non-empty data, compressed data should be different from original
            if !original_data.is_empty() {
                assert_ne!(
                    compressed, original_data,
                    "Compressed data should differ from original"
                );
            }

            // Decompress the data
            let decompressed = decompress_bytes(&compressed).expect("Decompression should succeed");

            // Verify the roundtrip produces the original data
            assert_eq!(
                decompressed, original_data,
                "Decompressed data should match original"
            );
        }
    }

    #[test]
    fn test_compression_reduces_size_for_repetitive_data() {
        // Create highly compressible data (lots of repetition)
        let repetitive_data = vec![0; 10000];

        let compressed = compress_bytes(&repetitive_data).expect("Compression should succeed");

        // Highly repetitive data should compress significantly
        assert!(
            compressed.len() < repetitive_data.len() / 2,
            "Compression should reduce size significantly for repetitive data"
        );
    }

    #[test]
    fn test_invalid_compressed_data() {
        // Invalid compressed data
        let invalid_data = vec![1, 2, 3, 4, 5]; // Not valid zlib compressed data

        let result = decompress_bytes(&invalid_data);
        assert!(result.is_err(), "Decompression of invalid data should fail");
    }

    #[test]
    fn test_empty_input() {
        // Empty input should work for both functions
        let empty = vec![];

        let compressed = compress_bytes(&empty).expect("Compressing empty data should succeed");

        let decompressed = decompress_bytes(&compressed).expect("Decompressing should succeed");

        assert_eq!(
            decompressed, empty,
            "Decompressed empty data should remain empty"
        );
    }
}
