use crate::{
    key_rotation::{KeyManager, KeyManagerError},
    ToBytes,
};
use atoma_sui::client::{AtomaSuiClient, AtomaSuiClientError};
use atoma_sui::events::AtomaEvent;
use flume::Receiver as FlumeReceiver;
use thiserror::Error;
use tokio::sync::{mpsc::UnboundedReceiver, oneshot};
use tracing::instrument;

// TODO: How large can the `ServiceData` be ? 

type Result<T> = std::result::Result<T, TdxServiceError>;
type ServiceData = Vec<u8>;

/// A service that manages Intel's TDX (Trust Domain Extensions) operations and key rotations.
///
/// The `TdxService` handles:
/// - Processing Atoma events related to key rotations
/// - Managing TDX key rotations and attestations
/// - Submitting attestations to the Sui blockchain
/// - Graceful shutdown handling
pub struct TdxService {
    /// Client for interacting with the Sui blockchain to submit attestations and transactions
    sui_client: AtomaSuiClient,
    /// Manages TDX key operations including key rotation and attestation generation
    key_manager: KeyManager,
    /// Channel receiver for incoming Atoma events that need to be processed
    event_receiver: UnboundedReceiver<AtomaEvent>,
    /// Channel receiver for incoming Atoma service requests for decryption and processing
    service_receiver: FlumeReceiver<(ServiceData, oneshot::Sender)>,
    /// Signal receiver for coordinating graceful shutdown of the service
    shutdown_signal: tokio::sync::watch::Receiver<bool>,
}

impl TdxService {
    /// Constructor
    pub fn new(
        sui_client: AtomaSuiClient,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_receiver: FlumeReceiver<(ServiceData, oneshot::Sender)>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Self> {
        let key_manager = KeyManager::new()?;
        Ok(Self {
            sui_client,
            key_manager,
            event_receiver,
            service_receiver,
            shutdown_signal,
        })
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
    /// - `Err(TdxServiceError)` if there's an error during key rotation or attestation submission
    ///
    /// # Errors
    /// This method can return:
    /// - `TdxServiceError::KeyManagerError` if key rotation fails
    /// - `TdxServiceError::SuiClientError` if attestation submission fails
    ///
    /// Note: Channel receive errors are logged but don't cause the service to return an error.
    #[instrument(level = "trace", skip_all)]
    pub async fn run(self) -> Result<()> {
        loop {
            tokio::select! {
                event = self.event_receiver.recv() => {
                    match event {
                        Ok(event) => {
                            tracing::trace!(
                                target = "atoma-tdx-service",
                                event = "event_received",
                                event = %event,
                                "Received event from event receiver"
                            );
                            match event {
                                AtomaEvent::NewKeyRotationEvent(event) => {
                                    tracing::trace!(
                                        target = "atoma-tdx-service",
                                        event = "new_key_rotation_event",
                                        event = %event,
                                        "New key rotation event received from event receiver"
                                    );
                                    self.submit_node_key_rotation_attestation()?;
                                }
                                AtomaEvent::NodeKeyRotationEvent(event) => {
                                    tracing::trace!(
                                        target = "atoma-tdx-service",
                                        event = "node_key_rotation_event",
                                        event = %event,
                                        "Node key rotation event received from event receiver"
                                    );
                                    // TODO: Handle node key rotation event
                                    // Should we even handle it here ?
                                }
                                _ => {
                                    tracing::warn!(
                                        target = "atoma-tdx-service",
                                        event = "unhandled_event",
                                        event = %event,
                                        "Unhandled event received from event receiver"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                target = "atoma-tdx-service",
                                event = "event_receiver_error",
                                error = %e,
                                "Error receiving event from event receiver"
                            );
                        }
                    }
                }
                shutdown_signal_changed = self.shutdown_signal.changed() => {
                    match shutdown_signal_changed {
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
    /// - `Err(TdxServiceError)` if any step fails, including key rotation or Sui client errors
    ///
    /// # Errors
    /// This function can return:
    /// - `TdxServiceError::KeyManagerError` if key rotation or public key retrieval fails
    /// - `TdxServiceError::SuiClientError` if the attestation submission to Sui fails
    #[instrument(level = "trace", skip_all)]
    fn submit_node_key_rotation_attestation(&self) -> Result<()> {
        let tdx_quote = self.key_manager.rotate_keys()?;
        let tdx_quote_bytes = tdx_quote.to_bytes();
        let public_key = self.key_manager.get_public_key();
        let public_key_bytes = public_key.to_bytes();
        match self.sui_client.submit_node_key_rotation_attestation(
            tdx_quote_bytes,
            public_key_bytes,
            None,
            None,
            None,
        ) {
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
                Err(TdxServiceError::SuiClientError(e))
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum TdxServiceError {
    #[error("Sui client error: {0}")]
    SuiClientError(#[from] AtomaSuiClientError),
    #[error("Key manager error: {0}")]
    KeyManagerError(#[from] KeyManagerError),
}
