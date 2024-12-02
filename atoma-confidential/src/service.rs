#[cfg(feature = "tdx")]
use crate::tdx::get_compute_data_attestation;
use crate::{
    key_management::{KeyManagementError, X25519KeyPairManager},
    types::{
        ConfidentialComputeDecryptionRequest, ConfidentialComputeDecryptionResponse,
        ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
    },
};
#[cfg(feature = "tdx")]
use atoma_sui::client::AtomaSuiClient;
use atoma_sui::{client::AtomaSuiClientError, events::AtomaEvent};
use thiserror::Error;
use tokio::sync::{mpsc::UnboundedReceiver, oneshot};
use tracing::instrument;

// TODO: How large can the `ServiceData` be ? Is it feasible to use a Flume channel ?

type Result<T> = std::result::Result<T, AtomaConfidentialComputeError>;
type ServiceDecryptionRequest = (
    ConfidentialComputeDecryptionRequest,
    oneshot::Sender<ConfidentialComputeDecryptionResponse>,
);
type ServiceEncryptionRequest = (
    ConfidentialComputeEncryptionRequest,
    oneshot::Sender<ConfidentialComputeEncryptionResponse>,
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
    #[cfg(feature = "tdx")]
    sui_client: AtomaSuiClient,
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
        #[cfg(feature = "tdx")] sui_client: AtomaSuiClient,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
        service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Self> {
        let key_manager = X25519KeyPairManager::new();
        Ok(Self {
            #[cfg(feature = "tdx")]
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
    ///    - `NodeKeyRotationEvent`: Currently unimplemented (TODO)
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
    #[instrument(level = "trace", skip_all)]
    pub async fn run(mut self) -> Result<()> {
        loop {
            tokio::select! {
                Some((decryption_request, sender)) = self.service_decryption_receiver.recv() => {
                    let ConfidentialComputeDecryptionRequest { ciphertext, nonce, salt, diffie_hellman_public_key } = decryption_request;
                    let plaintext = self.key_manager.decrypt_cyphertext(diffie_hellman_public_key, &ciphertext, &salt, &nonce)?;
                    sender
                        .send(ConfidentialComputeDecryptionResponse { plaintext })
                        .map_err(|_| AtomaConfidentialComputeError::SenderError)?;
                }
                Some((encryption_request, sender)) = self.service_encryption_receiver.recv() => {
                    let ConfidentialComputeEncryptionRequest { plaintext, salt, diffie_hellman_public_key } = encryption_request;
                    let (ciphertext, nonce) = self.key_manager.encrypt_plaintext(diffie_hellman_public_key, &plaintext, &salt)?;
                    sender
                        .send(ConfidentialComputeEncryptionResponse { ciphertext, nonce })
                        .map_err(|_| AtomaConfidentialComputeError::SenderError)?;
                }
                event_result = self.event_receiver.recv() => {
                    if let Some(event) = event_result {
                        match event {
                            AtomaEvent::NewKeyRotationEvent(_event) => {
                                tracing::trace!(
                                    target = "atoma-tdx-service",
                                    event = "new_key_rotation_event",
                                    "New key rotation event received from event receiver"
                                );
                                #[cfg(feature = "tdx")]
                                self.submit_node_key_rotation_tdx_attestation()?;
                            }
                            _ => {
                                tracing::warn!(
                                    target = "atoma-tdx-service",
                                    event = "unhandled_event",
                                    "Unhandled event received from event receiver"
                                );
                            }
                        }
                    }
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
    #[instrument(level = "trace", skip_all)]
    #[cfg(feature = "tdx")]
    async fn submit_node_key_rotation_tdx_attestation(&self) -> Result<()> {
        self.key_manager.rotate_keys();
        let public_key = self.key_manager.get_public_key();
        let public_key_bytes = public_key.to_bytes();
        let tdx_quote = get_compute_data_attestation(&public_key_bytes)?;
        let tdx_quote_bytes = tdx_quote.to_bytes();
        match self
            .sui_client
            .submit_key_rotation_remote_attestation(
                tdx_quote_bytes,
                public_key_bytes,
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
