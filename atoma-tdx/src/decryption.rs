use aes_gcm::{aead::Aead, Aes256Gcm, Error as AesError, KeyInit};
use hkdf::Hkdf;
use sha2::Sha256;
use thiserror::Error;
use x25519_dalek::SharedSecret;

type Result<T> = std::result::Result<T, DecryptionError>;

pub fn decrypt_cyphertext(
    shared_secret: SharedSecret,
    ciphertext: &[u8],
    salt: &[u8],
    nonce: &[u8],
) -> Result<Vec<u8>> {
    let hkdf = Hkdf::<Sha256>::new(Some(salt), shared_secret.as_bytes());
    let mut symmetric_key = [0u8; 32];
    hkdf.expand(b"", &mut symmetric_key)?;

    let cipher = Aes256Gcm::new(&symmetric_key.into());
    cipher
        .decrypt(nonce.into(), ciphertext)
        .map_err(DecryptionError::DecryptionFailed)
}

#[derive(Debug, Error)]
pub enum DecryptionError {
    #[error("Failed to decrypt ciphertext, with error: `{0}`")]
    DecryptionFailed(AesError),
    #[error("Failed to expand key, with error: `{0}`")]
    KeyExpansionFailed(#[from] hkdf::InvalidLength),
}
