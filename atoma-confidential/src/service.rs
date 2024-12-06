#[cfg(feature = "tdx")]
use crate::tdx::get_compute_data_attestation;
use crate::{
    key_management::{KeyManagementError, X25519KeyPairManager},
    types::{
        ConfidentialComputeDecryptionRequest, ConfidentialComputeDecryptionResponse,
        ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
    },
};
use atoma_sui::client::AtomaSuiClient;
use atoma_sui::{client::AtomaSuiClientError, events::AtomaEvent};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc::UnboundedReceiver, oneshot, RwLock};
use tracing::instrument;
use x25519_dalek::PublicKey;

// TODO: How large can the `ServiceData` be ? Is it feasible to use a Flume channel ?

type Result<T> = std::result::Result<T, AtomaConfidentialComputeError>;
type ServiceDecryptionRequest = (
    ConfidentialComputeDecryptionRequest,
    oneshot::Sender<anyhow::Result<ConfidentialComputeDecryptionResponse>>,
);
type ServiceEncryptionRequest = (
    ConfidentialComputeEncryptionRequest,
    oneshot::Sender<anyhow::Result<ConfidentialComputeEncryptionResponse>>,
);

/// A service that manages Intel's TDX (Trust Domain Extensions) operations and key rotations.
///
/// The `AtomaConfidentialCompute` handles:
/// - Processing Atoma events related to key rotations
/// - Managing TDX key rotations and attestations
/// - Submitting attestations to the Sui blockchain
/// - Graceful shutdown handling
pub struct AtomaConfidentialComputeService {
    /// Client for interacting with the Sui blockchain to submit attestations and transactions
    sui_client: Arc<RwLock<AtomaSuiClient>>,
    /// Manages TDX key operations including key rotation and attestation generation
    key_manager: X25519KeyPairManager,
    /// Channel receiver for incoming Atoma events that need to be processed
    event_receiver: UnboundedReceiver<AtomaEvent>,
    /// Channel receiver for incoming Atoma service requests for decryption and processing
    service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
    /// Channel receiver for incoming Atoma service requests for encryption and processing
    service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
    /// Signal receiver for coordinating graceful shutdown of the service
    shutdown_signal: tokio::sync::watch::Receiver<bool>,
}

impl AtomaConfidentialComputeService {
    /// Constructor
    pub fn new(
        sui_client: Arc<RwLock<AtomaSuiClient>>,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
        service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Self> {
        let key_manager = X25519KeyPairManager::new()?;
        Ok(Self {
            sui_client,
            key_manager,
            event_receiver,
            service_decryption_receiver,
            service_encryption_receiver,
            shutdown_signal,
        })
    }

    /// Returns the current public key used by the confidential compute service
    ///
    /// This method provides access to the X25519 public key that is currently being used
    /// for encryption and decryption operations. The public key can be shared with clients
    /// who need to establish secure communication with this service.
    ///
    /// # Returns
    /// - `x25519_dalek::PublicKey`: The current public key from the key manager
    pub fn get_public_key(&self) -> x25519_dalek::PublicKey {
        self.key_manager.get_public_key()
    }

    /// Starts the TDX service event loop that processes Atoma events and handles graceful shutdown.
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
            "Running confidential compute service, with dh public key: {:?}",
            self.key_manager.get_public_key().as_bytes()
        );
        loop {
            tokio::select! {
                Some((decryption_request, sender)) = self.service_decryption_receiver.recv() => {
                    self.handle_decryption_request(decryption_request, sender)?;
                }
                Some((encryption_request, sender)) = self.service_encryption_receiver.recv() => {
                    self.handle_encryption_request(encryption_request, sender)?;
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
                                target = "atoma-tdx-service",
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

    /// Submits a node key rotation attestation to the Sui blockchain.
    ///
    /// This method performs the following steps:
    /// 1. Rotates the TDX keys using the key manager
    /// 2. Generates a TDX quote from the rotated keys
    /// 3. Retrieves the public key associated with the rotated keys
    /// 4. Submits the attestation to the Sui blockchain with the quote and public key
    ///
    /// # Returns
    /// - `Ok(())` if the attestation was successfully submitted
    /// - `Err(AtomaConfidentialComputeError)` if any step fails, including key rotation or Sui client errors
    ///
    /// # Errors
    /// This function can return:
    /// - `AtomaConfidentialComputeError::KeyManagerError` if key rotation or public key retrieval fails
    /// - `AtomaConfidentialComputeError::SuiClientError` if the attestation submission to Sui fails
    #[instrument(level = "debug", skip_all)]
    async fn submit_node_key_rotation_tdx_attestation(&mut self) -> Result<()> {
        self.key_manager.rotate_keys()?;
        let public_key = self.key_manager.get_public_key();
        let public_key_bytes = public_key.to_bytes();
        #[cfg(feature = "tdx")]
        let tdx_quote_bytes = {
            let tdx_quote = get_compute_data_attestation(&public_key_bytes)?;
            tdx_quote.to_bytes()
        };
        #[cfg(not(feature = "tdx"))]
        let tdx_quote_bytes = {
            const TDX_QUOTE_V4_SIZE: usize = 512;
            vec![0u8; TDX_QUOTE_V4_SIZE]
        };
        match self
            .sui_client
            .write()
            .await
            .submit_key_rotation_remote_attestation(
                public_key_bytes,
                tdx_quote_bytes,
                None,
                None,
                None,
            )
            .await
        {
            Ok(digest) => {
                tracing::info!(
                    target = "atoma-tdx-service",
                    digest = digest,
                    "Submitted node key rotation attestation successfully"
                );
                Ok(())
            }
            Err(e) => {
                tracing::error!(
                    target = "atoma-tdx-service",
                    error = %e,
                    "Failed to submit node key rotation attestation"
                );
                Err(AtomaConfidentialComputeError::SuiClientError(e))
            }
        }
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
            proxy_public_key = ?decryption_request.proxy_x25519_public_key,
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
            proxy_x25519_public_key,
            node_x25519_public_key,
        } = decryption_request;
        let result = if PublicKey::from(node_x25519_public_key) != self.key_manager.get_public_key()
        {
            tracing::error!(
                target = "atoma-confidential-compute-service",
                event = "confidential_compute_service_decryption_error",
                "Node X25519 public key does not match the expected key: {:?} != {:?}",
                node_x25519_public_key,
                self.key_manager.get_public_key().as_bytes()
            );
            Err(anyhow::anyhow!(
                "Node X25519 public key does not match the expected key"
            ))
        } else {
            self.key_manager
                .decrypt_cyphertext(proxy_x25519_public_key, &ciphertext, &salt, &nonce)
                .map_err(|e| {
                    tracing::error!(
                        target = "atoma-confidential-compute-service",
                        event = "confidential_compute_service_decryption_error",
                        "Failed to decrypt cyphertext, with error: {:?}",
                        e
                    );
                    anyhow::anyhow!(e)
                })
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
    ///   - proxy_x25519_public_key: The public key of the proxy requesting encryption
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
            proxy_public_key = ?encryption_request.proxy_x25519_public_key,
            proxy_public_key = ?self.key_manager.get_public_key().as_bytes()
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
            proxy_x25519_public_key,
        } = encryption_request;
        let result = self
            .key_manager
            .encrypt_plaintext(proxy_x25519_public_key, &plaintext, &salt)
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
        level = "debug",
        name = "handle_atoma_event",
        skip_all,
        fields(
            event_type = ?std::any::type_name_of_val(&event)
        )
    )]
    async fn handle_atoma_event(&mut self, event: AtomaEvent) -> Result<()> {
        match event {
            AtomaEvent::NewKeyRotationEvent(_event) => {
                tracing::trace!(
                    target = "atoma-tdx-service",
                    event = "new_key_rotation_event",
                    "New key rotation event received from event receiver"
                );
                self.submit_node_key_rotation_tdx_attestation().await?
            }
            _ => {
                tracing::warn!(
                    target = "atoma-tdx-service",
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
}
