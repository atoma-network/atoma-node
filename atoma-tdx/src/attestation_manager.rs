use atoma_utils::hashing::blake2b_hash;
use tdx::device::{Device, DeviceOptions};
use thiserror::Error;
use x25519_dalek::{PublicKey, StaticSecret};

type Result<T> = std::result::Result<T, AttestationManagerError>;

/// A struct that manages cryptographic key rotation and remote attestation for TDX (Trust Domain Extensions).
///
/// `AttestationManager` combines an X25519 keypair with a TDX device to facilitate:
/// - Secure key generation and rotation
/// - Remote attestation report generation
/// - Public key access
/// - Device report data updates
pub struct AttestationManager {
    /// The X25519 static secret used for cryptographic operations.
    /// This secret key can be rotated using the `rotate_keys()` method.
    secret_key: StaticSecret,

    /// The TDX device instance used for generating attestation reports
    /// and managing device-specific operations.
    device: Device,
}

impl AttestationManager {
    /// Constructor
    pub fn new() -> Result<Self> {
        let mut rng = rand::thread_rng();
        let secret_key = StaticSecret::random_from_rng(&mut rng);
        let device = Device::new(DeviceOptions::default())?;
        let mut this = Self { secret_key, device };
        this.update_device_options();
        Ok(this)
    }

    /// Returns a reference to the current X25519 public key.
    ///
    /// This public key corresponds to the current keypair and can be used for:
    /// - Verifying signatures
    /// - Establishing secure communications
    /// - Identity verification
    ///
    /// The public key will change when `rotate_keys()` is called.
    pub fn get_public_key(&self) -> &PublicKey {
        &PublicKey::from(&self.secret_key)
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
    pub fn rotate_keys(&mut self) -> Result<QuoteV4> {
        let mut rng = rand::thread_rng();
        self.secret_key = StaticSecret::random_from_rng(&mut rng);
        self.update_device_options();
        self.generate_remote_attestation_report()
    }

    /// Generates a remote attestation report using the TDX device.
    ///
    /// This method creates an attestation report that:
    /// - Proves the authenticity of the TDX environment
    /// - Includes the current public key in the report data
    /// - Can be verified by remote parties to establish trust
    ///
    /// The generated report contains:
    /// - A quote structure with measurement data
    /// - The device's report data (including the public key)
    /// - Additional metadata for verification
    ///
    /// # Returns
    /// - `Ok(QuoteV4)` containing the attestation report
    /// - `Err(AttestationManagerError)` if report generation fails
    pub fn generate_remote_attestation_report(&self) -> Result<QuoteV4> {
        self.device.get_attestation_report()
    }

    /// Updates the TDX device's report data with the current public key.
    ///
    /// This method embeds the X25519 public key into the TDX device's report data field,
    /// which is a 64-byte buffer used during attestation. The public key (32 bytes) is
    /// copied into the beginning of the buffer, with the remaining bytes zeroed.
    ///
    /// # Implementation Details
    /// - Derives the X25519 public key from the current secret key
    /// - Initializes a zero-filled 64-byte report data buffer
    /// - Copies the 32-byte public key into the first half of the buffer
    /// - Updates the device options with the new report data
    ///
    /// This method is called automatically during:
    /// - Key manager initialization (`new()`)
    /// - Key rotation (`rotate_keys()`)
    ///
    /// The report data binding ensures that attestation reports cryptographically
    /// prove possession of the corresponding private key within the TDX environment.
    pub fn update_device_options(&mut self) {
        let public_key = PublicKey::from(&self.secret_key);
        let mut report_data = [0u8; 64];
        report_data[..public_key.as_ref().len()].copy_from_slice(public_key.as_ref());
        self.device.update_options(DeviceOptions {
            report_data: Some(report_data),
        });
    }

    /// Computes the shared secret between the current secret key and a given public key.
    ///
    /// This method uses the X25519 Diffie-Hellman key agreement protocol to derive a shared secret.
    /// The shared secret is returned as a byte vector.
    ///
    /// # Returns
    /// - `Vec<u8>` - The shared secret as a byte vector
    pub fn compute_shared_secret(&self, public_key: &PublicKey) -> Vec<u8> {
        let shared_secret = self.secret_key.diffie_hellman(&public_key);
        shared_secret.to_bytes().to_vec()
    }

    pub fn get_compute_data_attestation(&self, service_data: &[u8]) -> Result<QuoteV4> {
        let mut report_data = [0u8; 64];
        let public_key_bytes = self.get_public_key().as_ref();
        report_data[..public_key_bytes.len()].copy_from_slice(public_key_bytes);
        let service_data_hash = blake2b_hash(service_data);
        report_data[public_key_bytes.len()..].copy_from_slice(&service_data_hash);
        let device = Device::new(DeviceOptions {
            report_data: Some(report_data),
        });
        device.get_attestation_report()
    }
}

#[derive(Debug, Error)]
pub enum AttestationManagerError {
    #[error("Failed to generate key rotation report")]
    FailedToGenerateReport,
    #[error("Failed to create device: `{0}`")]
    TdxError(#[from] tdx::error::TdxError),
}
