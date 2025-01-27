use aes_gcm::{aead::Aead, Aes256Gcm, Error as AesError, KeyInit};
use hkdf::Hkdf;
use sha2::Sha256;
use x25519_dalek::SharedSecret;

pub const NONCE_BYTE_SIZE: usize = 12;

type Result<T> = std::result::Result<T, Error>;

/// Decrypts a ciphertext using the provided shared secret and nonce.
///
/// # Arguments
/// * `shared_secret` - The shared secret key for decryption
/// * `ciphertext` - The encrypted data to decrypt
/// * `salt` - Salt used in key derivation
/// * `nonce` - Nonce used in encryption
///
/// # Returns
/// The decrypted plaintext as a byte vector
///
/// # Errors
/// Returns an error if:
/// - Key derivation fails
/// - Decryption fails due to invalid data or parameters
pub fn decrypt_ciphertext(
    shared_secret: &SharedSecret,
    ciphertext: &[u8],
    salt: &[u8],
    nonce: &[u8],
) -> Result<Vec<u8>> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), shared_secret.as_bytes());
    let mut symmetric_key = [0u8; 32];
    hkdf.expand(b"", &mut symmetric_key)
        .map_err(Error::KeyExpansionFailed)?;

    let cipher = Aes256Gcm::new(&symmetric_key.into());
    cipher
        .decrypt(nonce.into(), ciphertext)
        .map_err(Error::DecryptionFailed)
}

/// Encrypts plaintext using the provided shared secret.
///
/// # Arguments
/// * `plaintext` - The data to encrypt
/// * `shared_secret` - The shared secret key for encryption
/// * `salt` - Salt for key derivation
/// * `nonce` - Optional nonce (generated if None)
///
/// # Returns
/// Tuple of (encrypted data, nonce used)
///
/// # Errors
/// Returns an error if:
/// - Key derivation fails
/// - Encryption operation fails
pub fn encrypt_plaintext(
    plaintext: &[u8],
    shared_secret: &SharedSecret,
    salt: &[u8],
    nonce: Option<[u8; NONCE_BYTE_SIZE]>,
) -> Result<(Vec<u8>, [u8; NONCE_BYTE_SIZE])> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), shared_secret.as_bytes());
    let mut symmetric_key = [0u8; 32];
    hkdf.expand(b"", &mut symmetric_key)
        .map_err(Error::KeyExpansionFailed)?;

    let cipher = Aes256Gcm::new(&symmetric_key.into());
    let nonce = nonce.unwrap_or_else(rand::random::<[u8; NONCE_BYTE_SIZE]>);
    let ciphertext = cipher
        .encrypt(&nonce.into(), plaintext)
        .map_err(Error::EncryptionFailed)?;

    Ok((ciphertext, nonce))
}

/// Errors that can occur during encryption/decryption operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to decrypt ciphertext, with error: `{0}`")]
    DecryptionFailed(AesError),
    #[error("Failed to encrypt plaintext, with error: `{0}`")]
    EncryptionFailed(AesError),
    #[error("Failed to expand key, with error: `{0}`")]
    KeyExpansionFailed(hkdf::InvalidLength),
}
