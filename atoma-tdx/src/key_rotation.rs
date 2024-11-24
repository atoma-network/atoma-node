use fastcrypto::{
    ed25519::{Ed25519KeyPair, Ed25519PublicKey},
    traits::{EncodeDecodeBase64, KeyPair},
};
use tdx::device::{Device, DeviceOptions};
use thiserror::Error;

type Result<T> = std::result::Result<T, KeyManagerError>;

/// A struct that manages cryptographic key rotation and remote attestation for TDX (Trust Domain Extensions).
///
/// `KeyManager` combines an Ed25519 keypair with a TDX device to facilitate:
/// - Secure key generation and rotation
/// - Remote attestation report generation
/// - Public key access
/// - Device report data updates
pub struct KeyManager {
    /// The Ed25519 keypair used for cryptographic operations.
    /// This keypair can be rotated using the `rotate_keys()` method.
    keypair: Ed25519KeyPair,

    /// The TDX device instance used for generating attestation reports
    /// and managing device-specific operations.
    device: Device,
}

impl KeyManager {
    /// Constructor
    pub fn new() -> Result<Self> {
        let mut rng = rand::thread_rng();
        let keypair = Ed25519KeyPair::generate(&mut rng);
        let device = Device::new(DeviceOptions::default())?;
        let mut this = Self { keypair, device };
        this.update_device_options();
        Ok(this)
    }

    /// Returns a reference to the current Ed25519 public key.
    ///
    /// This public key corresponds to the current keypair and can be used for:
    /// - Verifying signatures
    /// - Establishing secure communications
    /// - Identity verification
    ///
    /// The public key will change when `rotate_keys()` is called.
    pub fn get_public_key(&self) -> &Ed25519PublicKey {
        &self.keypair.public()
    }

    /// Rotates the Ed25519 keypair by generating a new random keypair and updates device report data.
    ///
    /// This method:
    /// - Generates a new random Ed25519 keypair using a secure random number generator
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
    /// - `Err(KeyManagerError)` - If key rotation or report generation fails
    ///
    /// Note: This operation automatically updates the device options to ensure
    /// the attestation report reflects the new keypair.
    pub fn rotate_keys(&mut self) -> Result<QuoteV4> {
        let mut rng = rand::thread_rng();
        self.keypair = Ed25519KeyPair::generate(&mut rng);
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
    /// - `Err(KeyManagerError)` if report generation fails
    pub fn generate_remote_attestation_report(&self) -> Result<QuoteV4> {
        self.device.get_attestation_report()
    }

    /// Updates the TDX device options with the current public key as report data.
    ///
    /// This method:
    /// - Retrieves the current Ed25519 public key
    /// - Creates a 64-byte report data buffer
    /// - Copies the public key bytes into the report data buffer
    /// - Updates the device options with the new report data
    ///
    /// The report data is used in attestation reports to bind the public key
    /// to the TDX environment, ensuring that remote parties can verify both:
    /// - The authenticity of the TDX environment
    /// - The ownership of the public key
    ///
    /// This method is automatically called during key rotation and initialization
    /// to maintain consistency between the keypair and device attestation.
    pub fn update_device_options(&mut self) {
        let public_key = self.keypair.public();
        let mut report_data = [0u8; 64];
        report_data[..public_key.as_ref().len()].copy_from_slice(public_key.as_ref());
        self.device.update_options(DeviceOptions {
            report_data: Some(report_data),
        });
    }
}

#[derive(Debug, Error)]
pub enum KeyManagerError {
    #[error("Failed to generate key rotation report")]
    FailedToGenerateReport,
    #[error("Failed to create device: `{0}`")]
    TdxError(#[from] tdx::error::TdxError),
}
