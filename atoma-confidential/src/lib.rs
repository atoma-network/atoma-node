#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod key_management;
pub mod nvml_cc;
pub mod service;
#[cfg(feature = "cc")]
pub mod tdx;
pub mod types;
pub mod utils;

pub use config::AtomaConfidentialComputeConfig;
pub use service::AtomaConfidentialCompute;

/// Trait for converting types into a byte representation
///
/// This trait provides a standard way to serialize types into a sequence of bytes.
/// It is particularly useful for cryptographic operations and data serialization
/// where a consistent byte representation is needed.
pub trait ToBytes {
    /// Converts the implementing type into a vector of bytes
    ///
    /// # Returns
    /// * `Vec<u8>` - The byte representation of the implementing type
    fn to_bytes(&self) -> Vec<u8>;
}
