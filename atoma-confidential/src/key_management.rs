use atoma_utils::encryption::{
    decrypt_cyphertext, encrypt_plaintext, EncryptionError, NONCE_BYTE_SIZE,
};
use thiserror::Error;
use x25519_dalek::{PublicKey, SharedSecret, StaticSecret};

type Result<T> = std::result::Result<T, KeyManagementError>;

/// A struct that manages X25519 key pair operations.
///
/// `X25519KeyPairManager` handles:
/// - X25519 key pair management
/// - Key generation and rotation
/// - Public key access for key exchange
/// - Shared secret computation
pub struct X25519KeyPairManager {
    /// The X25519 static secret used for cryptographic operations.
    /// This secret key can be rotated using the `rotate_keys()` method.
    secret_key: StaticSecret,
}

impl X25519KeyPairManager {
    /// Constructor
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let secret_key = StaticSecret::random_from_rng(&mut rng);
        Self { secret_key }
    }

    /// Returns a reference to the current X25519 public key.
    ///
    /// This public key corresponds to the current keypair and can be used for:
    /// - Verifying signatures
    /// - Establishing secure communications
    /// - Identity verification
    ///
    /// The public key will change when `rotate_keys()` is called.
    pub fn get_public_key(&self) -> PublicKey {
        PublicKey::from(&self.secret_key)
    }

    /// Rotates the X25519 keypair by generating a new random keypair and updates device report data.
    ///
    /// This method:
    /// - Generates a new random X25519 keypair using a secure random number generator
    /// - Replaces the existing keypair with the newly generated one
    /// - Updates the device's report data with the new public key
    /// - Invalidates any previous signatures or cryptographic operations
    /// - Generates and returns a new attestation report
    ///
    /// After rotation:
    /// - The previous public key will no longer be valid
    /// - The device's attestation report will include the new public key
    /// - Any systems relying on the previous public key must be updated
    ///
    /// # Returns
    /// - `Ok(QuoteV4)` - A new attestation report containing the rotated key
    /// - `Err(AttestationManagerError)` - If key rotation or report generation fails
    ///
    /// Note: This operation automatically updates the device options to ensure
    /// the attestation report reflects the new keypair.
    pub fn rotate_keys(&mut self) {
        let mut rng = rand::thread_rng();
        self.secret_key = StaticSecret::random_from_rng(&mut rng);
    }

    /// Computes the shared secret between the current secret key and a given public key.
    ///
    /// This method uses the X25519 Diffie-Hellman key agreement protocol to derive a shared secret.
    /// The shared secret is returned as a byte vector.
    ///
    /// # Returns
    /// - `SharedSecret` - The shared secret
    pub fn compute_shared_secret(&self, public_key: &PublicKey) -> SharedSecret {
        self.secret_key.diffie_hellman(&public_key)
    }

    /// Decrypts a ciphertext using X25519 key exchange and symmetric encryption.
    ///
    /// This method:
    /// 1. Converts the provided raw public key bytes into an X25519 public key
    /// 2. Computes a shared secret using the current secret key and the provided public key
    /// 3. Uses the shared secret to decrypt the ciphertext
    ///
    /// # Arguments
    /// * `public_key` - The sender's X25519 public key as a 32-byte array
    /// * `ciphertext` - The encrypted data to be decrypted
    /// * `salt` - Salt value used in the encryption process
    /// * `nonce` - Unique nonce (number used once) for the encryption
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - The decrypted plaintext as a byte vector
    /// * `Err(KeyManagementError)` - If decryption fails
    ///
    /// # Example
    /// ```rust,ignore
    /// # use your_crate::X25519KeyPairManager;
    /// let manager = X25519KeyPairManager::new();
    /// let public_key = [0u8; 32]; // Sender's public key
    /// let ciphertext = vec![/* encrypted data */];
    /// let salt = vec![/* salt bytes */];
    /// let nonce = vec![/* nonce bytes */];
    ///
    /// let plaintext = manager.decrypt_cyphertext(public_key, &ciphertext, &salt, &nonce)?;
    /// ```
    pub fn decrypt_cyphertext(
        &self,
        public_key: [u8; 32],
        ciphertext: &[u8],
        salt: &[u8],
        nonce: &[u8],
    ) -> Result<Vec<u8>> {
        let public_key = PublicKey::from(public_key);
        let shared_secret = self.compute_shared_secret(&public_key);
        Ok(decrypt_cyphertext(shared_secret, ciphertext, salt, nonce)?)
    }

    /// Encrypts plaintext using X25519 key exchange and symmetric encryption.
    ///
    /// This method:
    /// 1. Converts the provided raw public key bytes into an X25519 public key
    /// 2. Computes a shared secret using the current secret key and the provided public key
    /// 3. Uses the shared secret to encrypt the plaintext
    ///
    /// # Arguments
    /// * `public_key` - The recipient's X25519 public key as a 32-byte array
    /// * `plaintext` - The data to be encrypted
    /// * `salt` - Salt value to be used in the encryption process
    ///
    /// # Returns
    /// * `Ok((Vec<u8>, [u8; NONCE_BYTE_SIZE]))` - A tuple containing:
    ///   - The encrypted ciphertext as a byte vector
    ///   - A randomly generated nonce used in the encryption
    /// * `Err(KeyManagementError)` - If encryption fails
    ///
    /// # Example
    /// ```rust,ignore
    /// # use your_crate::X25519KeyPairManager;
    /// let manager = X25519KeyPairManager::new();
    /// let recipient_public_key = [0u8; 32];
    /// let plaintext = b"Secret message";
    /// let salt = vec![/* salt bytes */];
    ///
    /// let (ciphertext, nonce) = manager.encrypt_plaintext(recipient_public_key, plaintext, &salt)?;
    /// ```
    pub fn encrypt_plaintext(
        &self,
        public_key: [u8; 32],
        plaintext: &[u8],
        salt: &[u8],
    ) -> Result<(Vec<u8>, [u8; NONCE_BYTE_SIZE])> {
        let public_key = PublicKey::from(public_key);
        let shared_secret = self.compute_shared_secret(&public_key);
        Ok(encrypt_plaintext(plaintext, shared_secret, salt)?)
    }
}

#[derive(Debug, Error)]
pub enum KeyManagementError {
    #[error("Encryption error: `{0}`")]
    EncryptionError(#[from] EncryptionError),
}
