use crate::{
    key_management::{KeyManagementError, X25519KeyPairManager},
    types::{
        ConfidentialComputeDecryptionRequest, ConfidentialComputeDecryptionResponse,
        ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
        ConfidentialComputeSharedSecretRequest, ConfidentialComputeSharedSecretResponse,
    },
};
#[cfg(feature = "sev-snp")]
use crate::{
    sev_snp::{get_compute_data_attestation as snp_attestation, SnpError},
    ToBytes,
};
#[cfg(feature = "tdx")]
use crate::{
    tdx::{get_compute_data_attestation as tdx_attestation, TdxError},
    ToBytes,
};
use atoma_sui::client::Client;
use atoma_sui::{client::AtomaSuiClientError, events::AtomaEvent};
use atoma_utils::constants::NONCE_SIZE;
use std::sync::Arc;
use strum::EnumString;
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
    /// in the `submit_node_key_rotation_tdx_attestation` method, when the tdx feature is enabled.
    #[allow(dead_code)]
    sui_client: Arc<RwLock<Client>>,
    /// Current key rotation counter
    key_rotation_counter: Option<u64>,
    /// Manages TDX key operations including key rotation and attestation generation
    key_manager: X25519KeyPairManager,
    /// Channel receiver for incoming Atoma events that need to be processed
    event_receiver: UnboundedReceiver<AtomaEvent>,
    /// Channel receiver for incoming Atoma service requests for decryption and processing
    service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
    /// Channel receiver for incoming Atoma service requests for encryption and processing
    service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
    /// Channel receiver for incoming Atoma service requests for shared secret computation
    service_shared_secret_receiver: UnboundedReceiver<ServiceSharedSecretRequest>,
    /// Signal receiver for coordinating graceful shutdown of the service
    shutdown_signal: tokio::sync::watch::Receiver<bool>,
    /// Provider for the confidential compute service
    confidential_compute_provider: Option<AtomaConfidentialComputeProvider>,
}

/// Represents the supported confidential compute providers.
///
/// This enum defines the different types of confidential computing technologies
/// that can be used for secure computation and attestation.
///
/// # Variants
/// * `IntelTdx` - Intel Trust Domain Extensions (TDX) technology
/// * `AmdSevSnp` - AMD Secure Encrypted Virtualization with Secure Nested Paging (SEV-SNP)
///
/// The variants are serialized in kebab-case format (e.g., "intel-tdx", "amd-sev-snp")
/// when converting to/from strings using the `EnumString` trait from the `strum` crate.
#[derive(Debug, Clone, Copy, EnumString)]
#[strum(serialize_all = "kebab-case")]
pub enum AtomaConfidentialComputeProvider {
    IntelTdx,  // intel-tdx
    AmdSevSnp, // amd-sev-snp
}

impl AtomaConfidentialCompute {
    /// Constructor
    ///
    /// # Arguments
    /// * `sui_client` - Configuration settings for the client
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
    pub fn new(
        sui_client: Arc<RwLock<Client>>,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
        service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
        service_shared_secret_receiver: UnboundedReceiver<ServiceSharedSecretRequest>,
        confidential_compute_provider: Option<AtomaConfidentialComputeProvider>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Self> {
        let key_manager = X25519KeyPairManager::new()?;
        Ok(Self {
            sui_client,
            key_rotation_counter: None,
            key_manager,
            event_receiver,
            service_decryption_receiver,
            service_encryption_receiver,
            service_shared_secret_receiver,
            shutdown_signal,
            confidential_compute_provider,
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
    #[instrument(level = "info", skip_all)]
    pub async fn start_confidential_compute_service(
        sui_client: Arc<RwLock<Client>>,
        event_receiver: UnboundedReceiver<AtomaEvent>,
        service_decryption_receiver: UnboundedReceiver<ServiceDecryptionRequest>,
        service_encryption_receiver: UnboundedReceiver<ServiceEncryptionRequest>,
        service_shared_secret_receiver: UnboundedReceiver<ServiceSharedSecretRequest>,
        confidential_compute_provider: Option<AtomaConfidentialComputeProvider>,
        shutdown_signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()> {
        let mut service = Self::new(
            sui_client,
            event_receiver,
            service_decryption_receiver,
            service_encryption_receiver,
            service_shared_secret_receiver,
            confidential_compute_provider,
            shutdown_signal,
        )?;

        service.submit_node_key_rotation_attestation().await?;

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
    ///
    ///
    /// Make sure this is under a feature flag for TDX, and add a different CC implemntation for ADM SEV-SNP
    #[instrument(level = "debug", skip_all)]
    async fn submit_node_key_rotation_attestation(&mut self) -> Result<()> {
        self.key_manager.rotate_keys();

        if let Some(_cc_provider) = self.confidential_compute_provider {
            #[cfg(feature = "tdx")]
            if matches!(_cc_provider, AtomaConfidentialComputeProvider::IntelTdx) {
                return self.submit_tdx_attestation().await;
            }
            #[cfg(feature = "sev-snp")]
            if matches!(_cc_provider, AtomaConfidentialComputeProvider::AmdSevSnp) {
                return self.submit_sev_snp_attestation().await;
            }
        }

        // If we get here, either the feature flags don't match or the provider isn't supported
        tracing::warn!(
            target = "atoma-confidential-compute-service",
            provider = ?self.confidential_compute_provider,
            "No attestation implementation available for provider"
        );
        Ok(())
    }

    #[cfg(feature = "tdx")]
    async fn submit_tdx_attestation(&mut self) -> Result<()> {
        let public_key = self.key_manager.get_public_key();
        let public_key_bytes = public_key.to_bytes();
        let tdx_quote = tdx_attestation(&public_key_bytes)?;
        let tdx_quote_bytes = tdx_quote.to_bytes();

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
            Ok((digest, key_rotation_counter)) => {
                tracing::info!(
                    target = "atoma-confidential-compute-service",
                    digest = digest,
                    key_rotation_counter = key_rotation_counter,
                    "Submitted TDX attestation successfully"
                );
                self.key_rotation_counter = Some(key_rotation_counter);
                Ok(())
            }
            Err(e) => {
                tracing::error!(
                    target = "atoma-confidential-compute-service",
                    error = %e,
                    "Failed to submit TDX attestation"
                );
                Err(AtomaConfidentialComputeError::SuiClientError(e))
            }
        }
    }

    #[cfg(feature = "sev-snp")]
    async fn submit_sev_snp_attestation(&mut self) -> Result<()> {
        let public_key = self.key_manager.get_public_key();
        let public_key_bytes = public_key.to_bytes();
        let sev_snp_quote = snp_attestation(&public_key_bytes)?;
        let sev_snp_quote_bytes = sev_snp_quote.to_bytes();

        match self
            .sui_client
            .write()
            .await
            .submit_key_rotation_remote_attestation(
                public_key_bytes,
                sev_snp_quote_bytes,
                None,
                None,
                None,
            )
            .await
        {
            Ok((digest, key_rotation_counter)) => {
                tracing::info!(
                    target = "atoma-confidential-compute-service",
                    digest = digest,
                    key_rotation_counter = key_rotation_counter,
                    "Submitted SEV-SNP attestation successfully"
                );
                self.key_rotation_counter = Some(key_rotation_counter);
                Ok(())
            }
            Err(e) => {
                tracing::error!(
                    target = "atoma-confidential-compute-service",
                    error = %e,
                    "Failed to submit SEV-SNP attestation"
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
        level = "debug",
        name = "handle_atoma_event",
        skip_all,
        fields(
            event_type = ?std::any::type_name_of_val(&event)
        )
    )]
    async fn handle_atoma_event(&mut self, event: AtomaEvent) -> Result<()> {
        match event {
            AtomaEvent::NewKeyRotationEvent(event) => {
                tracing::trace!(
                    target = "atoma-tdx-service",
                    event = "new_key_rotation_event",
                    "New key rotation event received from event receiver"
                );
                // NOTE: Make sure to submit a node key rotation to the Atoma contract
                // if and only if the current key rotation counter is `None` (in which
                // case your node has not submitted a valid public key for encryption yet)
                // or if the current key rotation counter is less than the received key rotation
                // counter (in which case your node has submitted a public key for encryption
                // for a previous key rotation counter and not for the current one).
                if self
                    .key_rotation_counter
                    .map_or(true, |counter| counter < event.key_rotation_counter)
                {
                    self.submit_node_key_rotation_attestation().await?;
                }
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
    #[cfg(feature = "tdx")]
    #[error("TDX device error: {0}")]
    TdxDeviceError(#[from] TdxError),
    #[cfg(feature = "sev-snp")]
    #[error("SEV-SNP device error: {0}")]
    SevSnpDeviceError(#[from] SnpError),
}
