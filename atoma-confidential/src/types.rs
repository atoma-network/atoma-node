use atoma_utils::constants::{NONCE_SIZE, SALT_SIZE};
use remote_attestation_verifier::{DeviceEvidence, NvSwitchEvidence};
use serde::{Deserialize, Serialize};

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

/// Combined evidence from a device and an NVSwitch
///
/// This enum represents the evidence from a device and an NVSwitch, which is used to verify the integrity and authenticity of the GPU hardware and its execution environment.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "evidence_type")]
pub enum CombinedEvidence {
    /// Evidence from a device
    #[serde(rename = "device")]
    Device(DeviceEvidence),

    /// Evidence from an NVSwitch
    #[serde(rename = "nvswitch")]
    NvSwitch(NvSwitchEvidence),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeAttestation {
    pub node_small_id: i64,
    pub attestation: Vec<u8>,
}
