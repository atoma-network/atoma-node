/// Size of a Diffie-Hellman public key in bytes
pub const DH_PUBLIC_KEY_SIZE: usize = 32;

/// Size of a cryptographic nonce in bytes
pub const NONCE_SIZE: usize = 12;

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
    pub nonce: Vec<u8>,
    /// Salt value used in key derivation
    pub salt: Vec<u8>,
    /// Public key component for Diffie-Hellman key exchange
    pub proxy_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
    /// Public key component for Diffie-Hellman key exchange
    pub node_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
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
    pub salt: Vec<u8>,
    /// Public key component for Diffie-Hellman key exchange
    pub proxy_x25519_public_key: [u8; DH_PUBLIC_KEY_SIZE],
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
