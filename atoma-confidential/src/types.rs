use atoma_utils::constants::{NONCE_SIZE, SALT_SIZE};

/// Size of a Diffie-Hellman public key in bytes
pub const DH_PUBLIC_KEY_SIZE: usize = 32;

/// A request for confidential computation that includes encrypted data and key exchange parameters
///
/// This struct contains all necessary components for secure communication:
/// * Encrypted payload (ciphertext)
/// * Cryptographic nonce for the encryption
/// * Salt for key derivation
/// * Public key for Diffie-Hellman key exchange
pub struct ConfidentialComputeDecryptionRequest {
    /// The encrypted data to be processed
    pub ciphertext: Vec<u8>,

    /// Cryptographic nonce used in the encryption process
    pub nonce: [u8; NONCE_SIZE],

    /// Salt value used in key derivation
    pub salt: [u8; SALT_SIZE],

    /// Public key component for Diffie-Hellman key exchange from the client
    pub client_dh_public_key: [u8; DH_PUBLIC_KEY_SIZE],

    /// Public key component for Diffie-Hellman key exchange from the node
    pub node_dh_public_key: [u8; DH_PUBLIC_KEY_SIZE],
}

/// Response containing the decrypted data from a confidential computation request
///
/// This struct contains the decrypted plaintext after successful processing of a
/// ConfidentialComputeDecryptionRequest. The plaintext represents the original
/// data that was encrypted in the request.
pub struct ConfidentialComputeDecryptionResponse {
    /// The decrypted data resulting from the confidential computation
    pub plaintext: Vec<u8>,
}

/// A request for confidential computation that includes plaintext data and key exchange parameters
///
/// This struct contains all necessary components for secure communication:
/// * Plaintext payload to be encrypted
/// * Salt for key derivation
/// * Public key for Diffie-Hellman key exchange
pub struct ConfidentialComputeEncryptionRequest {
    /// The plaintext data to be encrypted
    pub plaintext: Vec<u8>,
    /// Salt value used in key derivation
    pub salt: [u8; SALT_SIZE],
    /// Client's public key component for Diffie-Hellman key exchange
    pub client_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
}

/// Response containing the encrypted data from a confidential computation request
///
/// This struct contains the encrypted ciphertext after successful processing of a
/// ConfidentialComputeEncryptionRequest.
pub struct ConfidentialComputeEncryptionResponse {
    /// The encrypted data resulting from the confidential computation
    pub ciphertext: Vec<u8>,
    /// Cryptographic nonce used in the encryption process
    pub nonce: [u8; NONCE_SIZE],
}

/// Request for the shared secret from a confidential computation request
///
/// This struct contains the public key of the proxy
pub struct ConfidentialComputeSharedSecretRequest {
    /// Client's public key component for Diffie-Hellman key exchange
    pub client_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
}

/// Response containing the shared secret from a confidential computation request
///
/// This struct contains the shared secret after successful processing of a
/// ConfidentialComputeSharedSecretRequest.
pub struct ConfidentialComputeSharedSecretResponse {
    /// The shared secret resulting from the confidential computation
    pub shared_secret: x25519_dalek::SharedSecret,
    /// Cryptographic nonce used in the encryption process
    pub nonce: [u8; NONCE_SIZE],
}

/// Represents the type of Trusted Execution Environment (TEE) provider used by a node.
///
/// This enum identifies different TEE providers that can be used for secure computation
/// and attestation in the Atoma network. Each variant corresponds to a specific TEE
/// technology from different hardware vendors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TEEProvider {
    /// Intel Trust Domain Extensions (TDX) technology
    Tdx = 0,
    /// AMD SEV with Secure Nested Paging (SNP) technology
    Snp = 1,
    /// ARM TrustZone technology
    Arm = 2,
}

impl TEEProvider {
    /// Creates a TEEProvider from its byte representation.
    ///
    /// # Arguments
    /// * `byte` - A byte containing the TEE provider identifier (0, 1, or 2)
    ///
    /// # Returns
    /// * `Ok(TEEProvider)` - The corresponding TEE provider variant
    /// * `Err(anyhow::Error)` - If the byte value is not recognized as a valid TEE provider
    ///
    /// # Errors
    /// Returns an error if the input byte does not correspond to a known TEE provider variant.
    pub fn from_byte(byte: u8) -> Result<Self, anyhow::Error> {
        Ok(match byte {
            0 => Self::Tdx,
            1 => Self::Snp,
            2 => Self::Arm,
            _ => anyhow::bail!("Invalid TEE provider"),
        })
    }

    /// Converts the TEE provider enum to a byte representation.
    ///
    /// # Returns
    /// A byte representing the TEE provider variant.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        self as u8
    }
}
