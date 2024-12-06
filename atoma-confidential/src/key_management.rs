use atoma_utils::encryption::{
    decrypt_ciphertext, encrypt_plaintext, EncryptionError, NONCE_BYTE_SIZE,
};
use thiserror::Error;
use x25519_dalek::{PublicKey, SharedSecret, StaticSecret};

/// The size of the X25519 secret key in bytes.
const DH_SECRET_KEY_SIZE: usize = 32;

/// The directory where the private key file is stored.
const KEY_FILE_DIR: &str = "keys";

/// The name of the private key file.
const KEY_FILE_NAME: &str = "dh_privkey";

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
    #[allow(clippy::new_without_default)]
    pub fn new() -> Result<Self> {
        let path = Self::get_key_file_path();

        if path.exists() {
            // Read the existing key from the file
            let key_bytes = std::fs::read(&path).map_err(KeyManagementError::IoError)?;
            let mut key_bytes_array: [u8; DH_SECRET_KEY_SIZE] = [0u8; DH_SECRET_KEY_SIZE];
            key_bytes_array.copy_from_slice(&key_bytes[..DH_SECRET_KEY_SIZE]);
            let secret_key = StaticSecret::from(key_bytes_array);
            Ok(Self { secret_key })
        } else {
            // Generate a new key
            let mut rng = rand::thread_rng();
            let secret_key = StaticSecret::random_from_rng(&mut rng);
            let this = Self { secret_key };
            this.write_private_key_to_file()?;
            Ok(this)
        }
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
    pub fn rotate_keys(&mut self) -> Result<()> {
        let mut rng = rand::thread_rng();
        self.secret_key = StaticSecret::random_from_rng(&mut rng);
        self.write_private_key_to_file()
    }

    /// Computes the shared secret between the current secret key and a given public key.
    ///
    /// This method uses the X25519 Diffie-Hellman key agreement protocol to derive a shared secret.
    /// The shared secret is returned as a byte vector.
    ///
    /// # Returns
    /// - `SharedSecret` - The shared secret
    pub fn compute_shared_secret(&self, public_key: &PublicKey) -> SharedSecret {
        self.secret_key.diffie_hellman(public_key)
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
    /// let plaintext = manager.decrypt_ciphertext(public_key, &ciphertext, &salt, &nonce)?;
    /// ```
    pub fn decrypt_ciphertext(
        &self,
        public_key: [u8; 32],
        ciphertext: &[u8],
        salt: &[u8],
        nonce: &[u8],
    ) -> Result<Vec<u8>> {
        let public_key = PublicKey::from(public_key);
        let shared_secret = self.compute_shared_secret(&public_key);
        Ok(decrypt_ciphertext(shared_secret, ciphertext, salt, nonce)?)
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

    /// Returns the file path where the private key should be stored.
    ///
    /// This method constructs a path by:
    /// 1. Starting from the current working directory
    /// 2. Adding a "keys" subdirectory
    /// 3. Adding the "dh_privkey" file name
    ///
    /// # Returns
    /// * `PathBuf` - Path to the private key file: `./keys/dh_privkey`
    ///
    /// # Note
    /// Currently uses a hardcoded path relative to the current directory.
    /// This is primarily intended for development/testing purposes.
    /// Production environments should use a more secure and configurable location.
    fn get_key_file_path() -> std::path::PathBuf {
        // Use a more appropriate path, possibly from config
        std::env::current_dir()
            .unwrap_or_default()
            .join(KEY_FILE_DIR)
            .join(KEY_FILE_NAME)
    }

    /// Writes the current private key to a file at the root directory.
    ///
    /// # Warning
    /// This function is intended for development/testing purposes only.
    /// Writing private keys to disk in production is a security risk.
    ///
    /// # Returns
    /// * `Ok(())` if the write was successful
    /// * `Err(KeyManagementError)` if the write failed
    pub fn write_private_key_to_file(&self) -> Result<()> {
        use std::fs::{self, create_dir_all};
        use std::os::unix::fs::PermissionsExt; // Unix-specific permissions

        let path = Self::get_key_file_path();

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            create_dir_all(parent).map_err(KeyManagementError::IoError)?;
        }

        // Write key with restricted permissions
        fs::write(&path, self.secret_key.to_bytes()).map_err(KeyManagementError::IoError)?;

        // Set file permissions to owner read/write only (0600)
        #[cfg(unix)]
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .map_err(KeyManagementError::IoError)?;

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum KeyManagementError {
    #[error("Encryption error: `{0}`")]
    EncryptionError(#[from] EncryptionError),
    #[error("IO error: `{0}`")]
    IoError(#[from] std::io::Error),
}
