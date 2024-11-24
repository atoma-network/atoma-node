#[cfg(feature = "tdx")]
pub mod key_rotation;
#[cfg(feature = "tdx")]
pub mod service;

#[cfg(feature = "tdx")]
use dcap_rs::types::quotes::body::QuoteBody;
#[cfg(feature = "tdx")]
use tdx::QuoteV4;

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

#[cfg(feature = "tdx")]
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
