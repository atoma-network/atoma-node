use aes_gcm::{aead::Aead, Aes256Gcm, Error as AesError, KeyInit};
use hkdf::Hkdf;
use sha2::Sha256;
use thiserror::Error;
use x25519_dalek::SharedSecret;

const NONCE_BYTE_SIZE: usize = 12;

type Result<T> = std::result::Result<T, EncryptionError>;

/// Decrypts ciphertext using AES-256-GCM with a derived key from a shared secret.
///
/// This function performs the following steps:
/// 1. Derives a symmetric key from the shared secret using HKDF-SHA256
/// 2. Initializes an AES-256-GCM cipher with the derived key
/// 3. Decrypts the ciphertext using the provided nonce
///
/// # Arguments
///
/// * `shared_secret` - The shared secret derived from X25519 key exchange
/// * `ciphertext` - The encrypted data to decrypt
/// * `salt` - Salt value used in the key derivation process
/// * `nonce` - Unique nonce (number used once) for AES-GCM
///
/// # Returns
///
/// Returns the decrypted plaintext as a vector of bytes, or a `DecryptionError` if the operation fails.
///
/// # Example
///
/// ```rust,ignore
/// use atoma_tdx::decryption::decrypt_cyphertext;
/// # use your_crate::SharedSecret;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let shared_secret = SharedSecret::new();
/// let ciphertext = vec![/* encrypted data */];
/// let salt = b"unique_salt_value";
/// let nonce = b"unique_nonce_12"; // Must be 12 bytes for AES-GCM
///
/// let plaintext = decrypt_cyphertext(
///     shared_secret,
///     &ciphertext,
///     salt,
///     nonce
/// )?;
/// # Ok(())
/// # }
/// ```
///
/// # Security Considerations
///
/// - The nonce must be unique for each encryption operation
/// - The salt should be randomly generated for each key derivation
/// - The shared secret should be derived using secure key exchange
pub fn decrypt_cyphertext(
    shared_secret: SharedSecret,
    ciphertext: &[u8],
    salt: &[u8],
    nonce: &[u8],
) -> Result<Vec<u8>> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), shared_secret.as_bytes());
    let mut symmetric_key = [0u8; 32];
    hkdf.expand(b"", &mut symmetric_key)
        .map_err(EncryptionError::KeyExpansionFailed)?;

    let cipher = Aes256Gcm::new(&symmetric_key.into());
    cipher
        .decrypt(nonce.into(), ciphertext)
        .map_err(EncryptionError::DecryptionFailed)
}

/// Encrypts plaintext using AES-256-GCM with a derived key from a shared secret.
///
/// This function performs the following steps:
/// 1. Derives a symmetric key from the shared secret using HKDF-SHA256
/// 2. Initializes an AES-256-GCM cipher with the derived key
/// 3. Generates a random nonce
/// 4. Encrypts the plaintext using the generated nonce
///
/// # Arguments
///
/// * `plaintext` - The data to encrypt
/// * `shared_secret` - The shared secret derived from X25519 key exchange
/// * `salt` - Salt value used in the key derivation process
///
/// # Returns
///
/// Returns a tuple containing the encrypted ciphertext and the generated nonce, or a `DecryptionError` if the operation fails.
///
/// # Example
///
/// ```rust,ignore
/// use atoma_tdx::decryption::encrypt_plaintext;
/// # use your_crate::SharedSecret;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let shared_secret = SharedSecret::new();
/// let plaintext = b"secret message";
/// let salt = b"unique_salt_value";
///
/// let (ciphertext, nonce) = encrypt_plaintext(
///     plaintext,
///     shared_secret,
///     salt
/// )?;
/// # Ok(())
/// # }
/// ```
///
/// # Security Considerations
///
/// - The salt should be randomly generated for each key derivation
/// - The shared secret should be derived using secure key exchange
/// - The generated nonce is guaranteed to be unique for each encryption operation
pub fn encrypt_plaintext(
    plaintext: &[u8],
    shared_secret: SharedSecret,
    salt: &[u8],
) -> Result<(Vec<u8>, [u8; NONCE_BYTE_SIZE])> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), shared_secret.as_bytes());
    let mut symmetric_key = [0u8; 32];
    hkdf.expand(b"", &mut symmetric_key)
        .map_err(EncryptionError::KeyExpansionFailed)?;

    let cipher = Aes256Gcm::new(&symmetric_key.into());
    let nonce = rand::random::<[u8; NONCE_BYTE_SIZE]>().into();
    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(EncryptionError::EncryptionFailed)?;

    Ok((ciphertext, nonce.into()))
}

#[derive(Debug, Error)]
pub enum EncryptionError {
    #[error("Failed to decrypt ciphertext, with error: `{0}`")]
    DecryptionFailed(AesError),
    #[error("Failed to encrypt plaintext, with error: `{0}`")]
    EncryptionFailed(AesError),
    #[error("Failed to expand key, with error: `{0}`")]
    KeyExpansionFailed(hkdf::InvalidLength),
}
