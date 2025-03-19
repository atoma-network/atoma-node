use crate::{
    key_management::{KeyManagementError, X25519KeyPairManager},
    nvml_cc::{
        check_confidential_compute_status, fetch_attestation_report_async,
        fetch_device_certificate_chain_async, num_devices, AttestationError,
    },
    types::{
        ConfidentialComputeDecryptionRequest, ConfidentialComputeDecryptionResponse,
        ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
        ConfidentialComputeSharedSecretRequest, ConfidentialComputeSharedSecretResponse,
    },
};
use atoma_sui::client::Client;
use atoma_sui::{client::AtomaSuiClientError, events::AtomaEvent};
use atoma_utils::{
    compression::{compress_bytes, CompressionError},
    constants::NONCE_SIZE,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc::UnboundedReceiver, oneshot, RwLock};
use tracing::instrument;
use x25519_dalek::PublicKey;

/// The key for the certificate in the evidence data
const CERTIFICATE_KEY: &str = "certificate";

/// The key for the evidence in the evidence data
const EVIDENCE_KEY: &str = "evidence";

/// Intel CC CPU device slot [0, 100)
#[allow(dead_code)]
const INTEL_CC_CPU_DEVICE_SLOT: u16 = 0;

/// AMD CC CPU device slot [100, 200)
#[allow(dead_code)]
const AMD_CC_CPU_DEVICE_SLOT: u16 = 100;

/// ARM CC CPU device slot [200, 300)
#[allow(dead_code)]
const ARM_CC_CPU_DEVICE_SLOT: u16 = 200;

/// NVIDIA CC GPU device slot [300, 10_000)
#[allow(dead_code)]
const NVIDIA_CC_GPU_DEVICE_SLOT: u16 = 300;

/// NVIDIA CC NVSwitch device slot [10_000, 16_000)
#[allow(dead_code)]
const NVIDIA_CC_NVSWITCH_DEVICE_SLOT: u16 = 10_000;

type Result<T> = std::result::Result<T, AtomaConfidentialComputeError>;

type ServiceDecryptionRequest = (
    ConfidentialComputeDecryptionRequest,
    oneshot::Sender<anyhow::Result<ConfidentialComputeDecryptionResponse>>,
);

type ServiceEncryptionRequest = (
    ConfidentialComputeEncryptionRequest,
    oneshot::Sender<anyhow::Result<ConfidentialComputeEncryptionResponse>>,
);

type ServiceSharedSecretRequest = (
    ConfidentialComputeSharedSecretRequest,
    oneshot::Sender<ConfidentialComputeSharedSecretResponse>,
);

/// A service that manages Intel's TDX (Trust Domain Extensions) operations and key rotations.
///
/// The `AtomaConfidentialCompute` handles:
/// - Processing Atoma events related to key rotations
/// - Managing TDX key rotations and attestations
/// - Submitting attestations to the Sui blockchain
/// - Graceful shutdown handling
pub struct AtomaConfidentialCompute {
    /// Client for interacting with the Sui blockchain to submit attestations and transactions
    /// NOTE: We disable clippy's `dead_code` lint warning here, as the `sui_client` is used
    /// in the `submit_node_key_rotation_tdx_attestation` and `submit_nvidia_cc_attestation` methods,
    /// when the cc feature is enabled.
    sui_client: Arc<RwLock<Client>>,

    /// Whether confidential computing is supported on the node
    is_cc_supported: bool,

    /// Current key rotation counter
    key_rotation_counter: u64,

    /// Manages CC key operations including key rotation and attestation generation
    key_manager: X25519KeyPairManager,

    /// Channel receiver for incoming Atoma events that need to be processed
    event_receiver: UnboundedReceiver<AtomaEvent>,

    /// The number of devices supported by the NVML library
    num_devices: u32,

    /// Channel receiver for incoming Atoma service requests for decryption and processing
    service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,

    /// Channel receiver for incoming Atoma service requests for encryption and processing
    service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,

    /// Channel receiver for incoming Atoma service requests for shared secret computation
    service_shared_secret_receiver: UnboundedReceiver<ServiceSharedSecretRequest>,

    /// Signal receiver for coordinating graceful shutdown of the service
    shutdown_signal: tokio::sync::watch::Receiver<bool>,
}

impl AtomaConfidentialCompute {
    /// Constructor
    ///
    /// # Arguments
    /// * `sui_client` - Configuration settings for the client
    /// * `config` - Configuration settings for the confidential compute service
    /// * `key_rotation_counter` - The current key rotation counter of the Atoma smart contract
    /// * `event_receiver` - Channel receiver for Atoma events
    /// * `service_decryption_receiver` - Channel receiver for decryption requests
    /// * `service_encryption_receiver` - Channel receiver for encryption requests
    /// * `service_shared_secret_receiver` - Channel receiver for shared secret requests
    /// * `shutdown_signal` - Channel receiver for shutdown signals
    ///
    /// # Returns
    /// A new client instance
    ///
    /// # Errors
    /// Returns `AtomaConfidentialComputeError` if:
    /// - Key manager initialization fails
    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "info", skip_all, fields(
        num_devices = num_devices().unwrap_or(0),
    ))]
    pub fn new(
        sui_client: Arc<RwLock<Client>>,
        key_rotation_counter: u64,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
        service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
        service_shared_secret_receiver: UnboundedReceiver<ServiceSharedSecretRequest>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Self> {
        let key_manager = X25519KeyPairManager::new()?;
        // NOTE: If the NVML library is not available, we return 0 devices
        let num_devices = num_devices().unwrap_or(0);
        let mut is_cc_supported = num_devices > 0;
        for index in 0..num_devices {
            is_cc_supported &= check_confidential_compute_status(index).unwrap_or(false);
        }
        tracing::info!(
            target = "atoma-confidential-compute-service",
            event = "new_confidential_compute_service",
            "New confidential compute service created, with num_devices: {num_devices} and is_cc_supported: {is_cc_supported}"
        );
        Ok(Self {
            sui_client,
            is_cc_supported,
            key_rotation_counter,
            key_manager,
            event_receiver,
            num_devices,
            service_decryption_receiver,
            service_encryption_receiver,
            service_shared_secret_receiver,
            shutdown_signal,
        })
    }

    /// Initializes and starts the confidential compute service.
    ///
    /// This method performs the following steps:
    /// 1. Creates a new service instance
    /// 2. Submits an initial node key rotation attestation
    /// 3. Starts the main service event loop
    ///
    /// # Arguments
    /// * `sui_client` - Arc-wrapped RwLock containing the Sui blockchain client
    /// * `event_receiver` - Channel receiver for Atoma events
    /// * `service_decryption_receiver` - Channel receiver for decryption requests
    /// * `service_encryption_receiver` - Channel receiver for encryption requests
    /// * `service_shared_secret_receiver` - Channel receiver for shared secret computation requests
    /// * `shutdown_signal` - Watch channel receiver for coordinating service shutdown
    ///
    /// # Returns
    /// * `Ok(())` if the service starts and runs successfully
    /// * `Err(AtomaConfidentialComputeError)` if initialization, attestation, or running fails
    ///
    /// # Errors
    /// This function can return:
    /// * `AtomaConfidentialComputeError::KeyManagementError` if key initialization fails
    /// * `AtomaConfidentialComputeError::SuiClientError` if attestation submission fails
    #[instrument(
        level = "info",
        skip_all,
        fields(
            num_devices = num_devices().unwrap_or(0),
        )
    )]
    pub async fn start_confidential_compute_service(
        sui_client: Arc<RwLock<Client>>,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
        service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
        service_shared_secret_receiver: UnboundedReceiver<ServiceSharedSecretRequest>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()> {
        // NOTE: Submit the first node key rotation attestation, because the node is starting up afresh
        let last_key_rotation_data = {
            let mut client = sui_client.write().await;
            client.get_last_key_rotation_event().await?
        };
        let service = if let Some((key_rotation_counter, nonce)) = last_key_rotation_data {
            let mut service = Self::new(
                sui_client,
                key_rotation_counter,
                event_receiver,
                service_decryption_receiver,
                service_encryption_receiver,
                service_shared_secret_receiver,
                shutdown_signal,
            )?;
            if service.is_cc_supported {
                service.key_manager.rotate_keys();
                service.submit_nvidia_cc_attestation(nonce).await?;
            }
            service
        } else {
            Self::new(
                sui_client,
                0,
                event_receiver,
                service_decryption_receiver,
                service_encryption_receiver,
                service_shared_secret_receiver,
                shutdown_signal,
            )?
        };

        service.run().await?;

        Ok(())
    }

    /// Returns the current public key used by the confidential compute service
    ///
    /// This method provides access to the X25519 public key that is currently being used
    /// for encryption and decryption operations. The public key can be shared with clients
    /// who need to establish secure communication with this service.
    ///
    /// # Returns
    /// - `x25519_dalek::PublicKey`: The current public key from the key manager
    #[must_use]
    pub fn get_public_key(&self) -> x25519_dalek::PublicKey {
        self.key_manager.get_public_key()
    }

    /// Returns the shared secret between the node and the proxy
    ///
    /// This method computes the shared secret using the node's secret key and the proxy's public key
    /// and returns it as a `x25519_dalek::StaticSecret`
    ///
    /// # Returns
    /// - `x25519_dalek::StaticSecret`: The shared secret between the node and the proxy
    #[must_use]
    pub fn compute_shared_secret(
        &self,
        client_x25519_public_key: &PublicKey,
    ) -> x25519_dalek::SharedSecret {
        self.key_manager
            .compute_shared_secret(client_x25519_public_key)
    }

    /// Starts the Confidential Compute service event loop that processes Atoma events and handles graceful shutdown.
    ///
    /// This method runs continuously until a shutdown signal is received, processing two types of events:
    /// 1. Atoma events from the event receiver:
    ///    - `NewKeyRotationEvent`: Triggers a node key rotation attestation submission
    ///    - Other events are logged as warnings
    ///
    /// 2. Shutdown signals that terminate the service loop when received
    ///
    /// # Returns
    /// - `Ok(())` when the service shuts down gracefully
    /// - `Err(AtomaConfidentialComputeError)` if there's an error during key rotation or attestation submission
    ///
    /// # Errors
    /// This method can return:
    /// - `AtomaConfidentialComputeError::KeyManagerError` if key rotation fails
    /// - `AtomaConfidentialComputeError::SuiClientError` if attestation submission fails
    ///
    /// Note: Channel receive errors are logged but don't cause the service to return an error.
    #[instrument(level = "info", skip_all)]
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(
            target = "atoma-confidential-compute-service",
            event = "confidential_compute_service_run",
            "Running confidential compute service"
        );

        loop {
            tokio::select! {
                Some((decryption_request, sender)) = self.service_decryption_receiver.recv() => {
                    self.handle_decryption_request(decryption_request, sender)?;
                }
                Some((encryption_request, sender)) = self.service_encryption_receiver.recv() => {
                    self.handle_encryption_request(encryption_request, sender)?;
                }
                Some((shared_secret_request, sender)) = self.service_shared_secret_receiver.recv() => {
                    self.handle_shared_secret_request(shared_secret_request, sender)?;
                }
                Some(event) = self.event_receiver.recv() => {
                    self.handle_atoma_event(event).await?;
                }
                shutdown_result = self.shutdown_signal.changed() => {
                    match shutdown_result {
                        Ok(()) => {
                            if *self.shutdown_signal.borrow() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-confidential-compute-service",
                                event = "shutdown_signal_error",
                                error = %e,
                                "Shutdown signal channel closed"
                            );
                        }
                    }
                }
            }
        }
        tracing::info!(
            target = "atoma-confidential-compute-service",
            event = "confidential_compute_service_finished",
            "Confidential compute service finished"
        );
        Ok(())
    }

    /// Submits NVIDIA confidential computing attestation reports to the Sui blockchain.
    ///
    /// This method performs the following steps:
    /// 1. Collects evidence data from all configured GPU devices:
    ///    - Generates a unique nonce hash using the provided nonce, public key, and device index
    ///    - Fetches an attestation report from each NVIDIA GPU device
    ///    - Fetches the certificate chain from each NVIDIA GPU device
    ///    - Encodes both the attestation report and certificate chain in base64
    ///    - Combines them into a JSON structure for each device
    /// 2. Serializes all evidence data into a single JSON array
    /// 3. Compresses the combined evidence data to reduce size
    /// 4. Submits the compressed evidence to the Sui blockchain with device-specific information
    ///
    /// Unlike the previous implementation that submitted separate attestations for each device,
    /// this method now collects evidence from all devices and submits them in a single transaction.
    /// This approach is more efficient and ensures atomic verification of all GPU devices.
    ///
    /// # Arguments
    /// * `nonce` - A unique value provided by the Atoma contract to prevent replay attacks
    ///
    /// # Returns
    /// * `Ok(())` if the attestation was successfully submitted
    /// * `Err(AtomaConfidentialComputeError)` if any step fails
    ///
    /// # Errors
    /// This function can return:
    /// * `AtomaConfidentialComputeError::AttestationError` if fetching attestation reports or certificate chains fails
    /// * `AtomaConfidentialComputeError::CompressionError` if compressing the evidence data fails
    /// * `AtomaConfidentialComputeError::SerializationError` if serializing the evidence data fails
    /// * `AtomaConfidentialComputeError::SuiClientError` if the attestation submission to Sui fails
    /// * `AtomaConfidentialComputeError::InvalidDeviceIndex` if a device index cannot be converted to u16
    #[instrument(level = "info", skip_all, fields(
        nonce = nonce,
        num_devices = self.num_devices,
    ))]
    async fn submit_nvidia_cc_attestation(&mut self, nonce: u64) -> Result<()> {
        let public_key_bytes = self.key_manager.get_public_key().to_bytes();
        let nonce_le_bytes = nonce.to_le_bytes();
        let nonce_blake3_hash = blake3::hash(
            &[
                &nonce_le_bytes[..],
                &public_key_bytes,
                &NVIDIA_CC_GPU_DEVICE_SLOT.to_le_bytes()[..],
            ]
            .concat(),
        );
        let mut evidence_data = Vec::with_capacity(self.num_devices as usize);
        for device_index in 0..self.num_devices {
            let attestation_report =
                fetch_attestation_report_async(device_index, *nonce_blake3_hash.as_bytes()).await?;
            let certificate_chain = fetch_device_certificate_chain_async(device_index).await?;
            let attestation_report_base64 = STANDARD.encode(attestation_report);
            let certificate_chain_base64 = STANDARD.encode(certificate_chain);
            evidence_data.push(serde_json::json!({
                CERTIFICATE_KEY: certificate_chain_base64,
                EVIDENCE_KEY: attestation_report_base64,
            }));
        }
        let evidence_data_bytes = serde_json::to_vec(&evidence_data)?;
        let compressed_evidence_data = compress_bytes(&evidence_data_bytes)?;
        let response = {
            self.sui_client
                .write()
                .await
                .submit_key_rotation_remote_attestation(
                    public_key_bytes,
                    compressed_evidence_data,
                    self.key_rotation_counter,
                    NVIDIA_CC_GPU_DEVICE_SLOT,
                    None,
                    None,
                    None,
                )
                .await
        };
        match response {
            Ok((digest, key_rotation_counter)) => {
                tracing::info!(
                    target = "atoma-nvidia-cc-service",
                    digest = digest,
                    key_rotation_counter = key_rotation_counter,
                    "Submitted NVIDIA CC attestation successfully for node, with nonce: {nonce}",
                );
            }
            Err(e) => {
                tracing::error!(
                    target = "atoma-nvidia-cc-service",
                    error = %e,
                    "Failed to submit NVIDIA CC attestation"
                );
                return Err(AtomaConfidentialComputeError::SuiClientError(e));
            }
        }
        Ok(())
    }

    /// Handles a decryption request from a client by validating the node's public key and decrypting the ciphertext.
    ///
    /// This method performs the following steps:
    /// 1. Validates that the provided node public key matches the service's current public key
    /// 2. If valid, attempts to decrypt the ciphertext using the key manager
    /// 3. Sends the decryption result back through the provided sender channel
    ///
    /// # Arguments
    /// * `decryption_request` - A tuple containing:
    ///   - The decryption request with ciphertext, nonce, salt, and public keys
    ///   - A oneshot sender channel for returning the decryption response
    ///
    /// # Returns
    /// * `Ok(())` if the request was handled and response sent successfully
    /// * `Err(AtomaConfidentialComputeError)` if key validation fails, decryption fails, or the response cannot be sent
    ///
    /// # Errors
    /// This function can return `AtomaConfidentialComputeError::SenderError` if the response channel is closed
    #[instrument(
        level = "debug",
        name = "handle_decryption_request",
        skip_all,
        fields(
            client_public_key = ?decryption_request.client_dh_public_key,
            node_public_key = ?self.key_manager.get_public_key().as_bytes()
        )
    )]
    fn handle_decryption_request(
        &mut self,
        decryption_request: ConfidentialComputeDecryptionRequest,
        sender: oneshot::Sender<anyhow::Result<ConfidentialComputeDecryptionResponse>>,
    ) -> Result<()> {
        let ConfidentialComputeDecryptionRequest {
            ciphertext,
            nonce,
            salt,
            client_dh_public_key,
            node_dh_public_key,
        } = decryption_request;
        let result = if PublicKey::from(node_dh_public_key) == self.key_manager.get_public_key() {
            self.key_manager
                .decrypt_ciphertext(client_dh_public_key, &ciphertext, &salt, &nonce)
                .map_err(|e| {
                    tracing::error!(
                        target = "atoma-confidential-compute-service",
                        event = "confidential_compute_service_decryption_error",
                        "Failed to decrypt cyphertext, with error: {:?}",
                        e
                    );
                    anyhow::anyhow!(e)
                })
        } else {
            tracing::error!(
                target = "atoma-confidential-compute-service",
                event = "confidential_compute_service_decryption_error",
                "Failed to decrypt request: node public key mismatch"
            );
            Err(anyhow::anyhow!(
                "Node X25519 public key does not match the expected key"
            ))
        };
        let message = result
            .map(|plaintext| ConfidentialComputeDecryptionResponse { plaintext })
            .map_err(|e| anyhow::anyhow!(e));
        sender
            .send(message)
            .map_err(|_| AtomaConfidentialComputeError::SenderError)
    }

    /// Handles an encryption request from a client by encrypting the provided plaintext using the key manager.
    ///
    /// This method performs the following steps:
    /// 1. Extracts the plaintext, salt, and proxy public key from the request
    /// 2. Encrypts the plaintext using the key manager with the provided parameters
    /// 3. Sends the encryption result (ciphertext and nonce) back through the provided sender channel
    ///
    /// # Arguments
    /// * `encryption_request` - The encryption request containing:
    ///   - plaintext: The data to be encrypted
    ///   - salt: The salt value for the encryption
    ///   - client_x25519_public_key: The public key of the proxy requesting encryption
    /// * `sender` - A oneshot sender channel for returning the encryption response
    ///
    /// # Returns
    /// * `Ok(())` if the request was handled and response sent successfully
    /// * `Err(AtomaConfidentialComputeError)` if encryption fails or the response cannot be sent
    ///
    /// # Errors
    /// This function can return:
    /// * `AtomaConfidentialComputeError::KeyManagementError` if encryption fails
    /// * `AtomaConfidentialComputeError::SenderError` if the response channel is closed
    #[instrument(
        level = "debug",
        name = "handle_encryption_request",
        skip_all,
        fields(
            client_public_key = ?encryption_request.client_x25519_public_key,
            node_public_key = ?self.key_manager.get_public_key().as_bytes()
        )
    )]
    fn handle_encryption_request(
        &mut self,
        encryption_request: ConfidentialComputeEncryptionRequest,
        sender: oneshot::Sender<anyhow::Result<ConfidentialComputeEncryptionResponse>>,
    ) -> Result<()> {
        let ConfidentialComputeEncryptionRequest {
            plaintext,
            salt,
            client_x25519_public_key,
        } = encryption_request;
        let result = self
            .key_manager
            .encrypt_plaintext(client_x25519_public_key, &plaintext, &salt)
            .map_err(|e| {
                tracing::error!(
                    target = "atoma-confidential-compute-service",
                    event = "confidential_compute_service_encryption_error",
                    "Failed to encrypt plaintext, with error: {:?}",
                    e
                );
                AtomaConfidentialComputeError::KeyManagementError(e)
            });
        let message = result
            .map(|(ciphertext, nonce)| ConfidentialComputeEncryptionResponse { ciphertext, nonce })
            .map_err(|e| anyhow::anyhow!(e));
        sender
            .send(message)
            .map_err(|_| AtomaConfidentialComputeError::SenderError)
    }

    /// Handles a shared secret request from a client by computing the shared secret and sending it back.
    ///
    /// This method performs the following steps:
    /// 1. Computes the shared secret using the node's secret key and the proxy's public key
    /// 2. Generates a random nonce
    /// 3. Sends the shared secret and nonce back through the provided sender channel
    ///
    /// # Arguments
    /// * `shared_secret_request` - The shared secret request containing:
    ///   - client_x25519_public_key: The public key of the proxy requesting the shared secret
    /// * `sender` - A oneshot sender channel for returning the shared secret response
    ///
    /// # Returns
    /// * `Ok(())` if the request was handled and response sent successfully
    /// * `Err(AtomaConfidentialComputeError)` if the response cannot be sent
    ///
    /// # Errors
    /// This function can return `AtomaConfidentialComputeError::SenderError` if the response channel is closed
    #[instrument(
        level = "debug",
        name = "handle_shared_secret_request",
        skip_all,
        fields(
            client_public_key = ?shared_secret_request.client_x25519_public_key,
            node_public_key = ?self.key_manager.get_public_key().as_bytes()
        )
    )]
    fn handle_shared_secret_request(
        &mut self,
        shared_secret_request: ConfidentialComputeSharedSecretRequest,
        sender: oneshot::Sender<ConfidentialComputeSharedSecretResponse>,
    ) -> Result<()> {
        let ConfidentialComputeSharedSecretRequest {
            client_x25519_public_key,
        } = shared_secret_request;
        let shared_secret = self.compute_shared_secret(&PublicKey::from(client_x25519_public_key));
        let nonce = rand::random::<[u8; NONCE_SIZE]>();
        sender
            .send(ConfidentialComputeSharedSecretResponse {
                shared_secret,
                nonce,
            })
            .map_err(|_| AtomaConfidentialComputeError::SenderError)?;
        Ok(())
    }

    /// Processes incoming Atoma events and handles key rotation requests.
    ///
    /// This method handles two types of events:
    /// 1. `NewKeyRotationEvent`: Triggers a node key rotation attestation submission
    /// 2. Other events: Logged as warnings and ignored
    ///
    /// # Arguments
    /// * `event` - The Atoma event to be processed
    ///
    /// # Returns
    /// - `Ok(())` if the event was processed successfully
    /// - `Err(AtomaConfidentialComputeError)` if there's an error during key rotation or attestation submission
    ///
    /// # Errors
    /// This method can return:
    /// - `AtomaConfidentialComputeError::KeyManagerError` if key rotation fails
    /// - `AtomaConfidentialComputeError::SuiClientError` if attestation submission fails
    #[instrument(
        level = "info",
        name = "handle_atoma_event",
        skip_all,
        fields(
            event_type = ?std::any::type_name_of_val(&event)
        )
    )]
    async fn handle_atoma_event(&mut self, event: AtomaEvent) -> Result<()> {
        // NOTE: If confidential computing is not supported on the node, we skip the key rotations
        // and attestation submissions
        if !self.is_cc_supported {
            return Ok(());
        }
        match event {
            AtomaEvent::NewKeyRotationEvent(event) => {
                tracing::info!(
                    target = "atoma-confidential-compute-service",
                    event = "new_key_rotation_event",
                    "New key rotation event received from event receiver"
                );
                // NOTE: Make sure to submit a node key rotation to the Atoma contract
                // if and only if the current key rotation counter is `None` (in which
                // case your node has not submitted a valid public key for encryption yet)
                // or if the current key rotation counter is less than the received key rotation
                // counter (in which case your node has submitted a public key for encryption
                // for a previous key rotation counter and not for the current one).
                if self.key_rotation_counter < event.key_rotation_counter {
                    self.key_rotation_counter = event.key_rotation_counter;
                    self.key_manager.rotate_keys();
                    self.submit_nvidia_cc_attestation(event.nonce).await?;
                }
            }
            _ => {
                tracing::warn!(
                    target = "atoma-confidential-compute-service",
                    event = "unhandled_event",
                    "Unhandled event received from event receiver"
                );
            }
        }
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum AtomaConfidentialComputeError {
    #[error("Sui client error: {0}")]
    SuiClientError(#[from] AtomaSuiClientError),
    #[error("Key management error: {0}")]
    KeyManagementError(#[from] KeyManagementError),
    #[error("Sender error")]
    SenderError,
    #[error("Attestation error: {0}")]
    AttestationError(#[from] AttestationError),
    #[error("Compression error: {0}")]
    CompressionError(#[from] CompressionError),
    #[error("Invalid device index: {0}")]
    InvalidDeviceIndex(#[from] std::num::TryFromIntError),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
