use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use atoma_confidential::types::{
    ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
};
use atoma_state::types::AtomaAtomaStateManagerEvent;
use atoma_utils::hashing::blake2b_hash;
use axum::body::Bytes;
use axum::{response::sse::Event, Error};
use flume::Sender as FlumeSender;
use futures::Stream;
use prometheus::HistogramTimer;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tokio::sync::{
    mpsc::UnboundedSender,
    oneshot::{self, error::TryRecvError},
};
use tracing::{error, instrument};

use crate::{
    handlers::prometheus::{
        CHAT_COMPLETIONS_DECODING_TIME, CHAT_COMPLETIONS_INPUT_TOKENS_METRICS,
        CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS,
    },
    middleware::EncryptionMetadata,
    server::utils,
    server::EncryptionRequest,
};

/// The chunk that indicates the end of a streaming response
const DONE_CHUNK: &str = "[DONE]";
/// The prefix for the data chunk
const DATA_PREFIX: &str = "data: ";
/// The keep-alive chunk
const KEEP_ALIVE_CHUNK: &[u8] = b": keep-alive\n\n";
/// The choices key
const CHOICES: &str = "choices";
/// The usage key
const USAGE: &str = "usage";

/// A structure for streaming chat completion chunks.
pub struct Streamer {
    /// The stream of bytes from the inference service
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    /// The accumulated response for final processing
    accumulated_response: Vec<Value>,
    /// Current status of the stream
    status: StreamStatus,
    /// The stack small id for the request
    stack_small_id: i64,
    /// The estimated total compute units for the request
    estimated_total_compute_units: i64,
    /// The request payload hash
    payload_hash: [u8; 32],
    /// The sender for the state manager
    state_manager_sender: FlumeSender<AtomaAtomaStateManagerEvent>,
    /// The keystore
    keystore: Arc<FileBasedKeystore>,
    /// The address index
    address_index: usize,
    /// The model for the inference request
    model: String,
    /// The first token generation (prefill phase) timer for the request.
    /// We need store it as an option because we need to consume its value
    /// once the first token is generated
    first_token_generation_timer: Option<HistogramTimer>,
    /// The decoding phase timer for the request.
    decoding_phase_timer: Option<HistogramTimer>,
    /// The client encryption metadata for the request
    client_encryption_metadata: Option<EncryptionMetadata>,
    /// Confidential compute encryption sender
    confidential_compute_encryption_sender: UnboundedSender<EncryptionRequest>,
    /// The receiver for the encryption response
    encryption_response_receiver: Option<oneshot::Receiver<ConfidentialComputeEncryptionResponse>>,
    /// Boolean value flagging if the stream is currently waiting for a chunk encryption
    waiting_for_encrypted_chunk: bool,
}

/// Represents the various states of a streaming process
#[derive(Debug, PartialEq, Eq)]
pub enum StreamStatus {
    /// Stream has not started
    NotStarted,
    /// Stream is actively receiving data
    Started,
    /// Stream has completed successfully
    Completed,
    /// Stream failed with an error
    Failed(String),
}

impl Streamer {
    /// Creates a new Streamer instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        state_manager_sender: FlumeSender<AtomaAtomaStateManagerEvent>,
        stack_small_id: i64,
        estimated_total_compute_units: i64,
        payload_hash: [u8; 32],
        keystore: Arc<FileBasedKeystore>,
        address_index: usize,
        model: String,
        client_encryption_metadata: Option<EncryptionMetadata>,
        first_token_generation_timer: HistogramTimer,
        confidential_compute_encryption_sender: UnboundedSender<EncryptionRequest>,
    ) -> Self {
        Self {
            stream: Box::pin(stream),
            accumulated_response: Vec::new(),
            status: StreamStatus::NotStarted,
            stack_small_id,
            estimated_total_compute_units,
            payload_hash,
            state_manager_sender,
            keystore,
            address_index,
            model,
            first_token_generation_timer: Some(first_token_generation_timer),
            decoding_phase_timer: None,
            client_encryption_metadata,
            confidential_compute_encryption_sender,
            encryption_response_receiver: None,
            waiting_for_encrypted_chunk: false,
        }
    }

    /// Processes the final chunk of a streaming response, performing signature generation,
    /// token counting, and state updates.
    ///
    /// This method:
    /// 1. Signs the accumulated response data
    /// 2. Extracts and validates token usage information
    /// 3. Updates the state manager with token counts
    /// 4. Calculates a total hash combining payload and response hashes
    /// 5. Updates the state manager with the total hash
    /// 6. Creates a final SSE message containing signature and metadata
    ///
    /// # Arguments
    ///
    /// * `usage` - A JSON Value containing token usage information, expected to have a
    ///             "total_tokens" field with an integer value
    ///
    /// # Returns
    ///
    /// Returns a `Result<Event, Error>` where:
    /// * `Event` - An SSE event containing the final message with signature
    /// * `Error` - An error that can occur during:
    ///   - Response signing
    ///   - Token usage extraction
    ///   - JSON serialization
    ///
    /// # State Updates
    ///
    /// This method sends two events to the state manager:
    /// * `UpdateStackNumTokens` - Updates the token count for the stack
    /// * `UpdateStackTotalHash` - Updates the combined hash of payload and response
    #[instrument(
        level = "info",
        skip(self, usage),
        fields(
            endpoint = "handle_final_chunk",
            stack_small_id = self.stack_small_id,
            estimated_total_compute_units = self.estimated_total_compute_units,
            payload_hash = hex::encode(self.payload_hash)
        )
    )]
    fn handle_final_chunk(&mut self, usage: &Value) -> Result<String, Error> {
        // Record the decoding phase timer
        if let Some(timer) = self.decoding_phase_timer.take() {
            timer.observe_duration();
        }

        // Sign the accumulated response
        let (response_hash, signature) = utils::sign_response_body(
            &json!(self.accumulated_response),
            &self.keystore,
            self.address_index,
        )
        .map_err(|e| {
            error!("Error signing response: {}", e);
            Error::new(format!("Error signing response: {}", e))
        })?;

        // Get total tokens
        let mut total_compute_units = 0;
        if let Some(prompt_tokens) = usage.get("prompt_tokens") {
            let prompt_tokens = prompt_tokens.as_u64().unwrap_or(0);
            CHAT_COMPLETIONS_INPUT_TOKENS_METRICS
                .with_label_values(&[&self.model, &self.stack_small_id.to_string()])
                .inc_by(prompt_tokens as f64);
            total_compute_units += prompt_tokens;
        } else {
            error!("Error getting prompt tokens from usage");
            return Err(Error::new("Error getting prompt tokens from usage"));
        }
        if let Some(completion_tokens) = usage.get("completion_tokens") {
            let completion_tokens = completion_tokens.as_u64().unwrap_or(0);
            CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS
                .with_label_values(&[&self.model, &self.stack_small_id.to_string()])
                .inc_by(completion_tokens as f64);
            total_compute_units += completion_tokens;
        } else {
            error!("Error getting completion tokens from usage");
            return Err(Error::new("Error getting completion tokens from usage"));
        }

        // Update stack num tokens
        if let Err(e) = self.state_manager_sender.send(
            AtomaAtomaStateManagerEvent::UpdateStackNumComputeUnits {
                stack_small_id: self.stack_small_id,
                estimated_total_compute_units: self.estimated_total_compute_units,
                total_compute_units: total_compute_units as i64,
            },
        ) {
            error!("Error updating stack num tokens: {}", e);
        }

        // Calculate and update total hash
        let total_hash = blake2b_hash(&[self.payload_hash, response_hash].concat());
        let total_hash_bytes: [u8; 32] = total_hash
            .as_slice()
            .try_into()
            .expect("Invalid BLAKE2b hash length");

        // Update stack total hash
        if let Err(e) =
            self.state_manager_sender
                .send(AtomaAtomaStateManagerEvent::UpdateStackTotalHash {
                    stack_small_id: self.stack_small_id,
                    total_hash: total_hash_bytes,
                })
        {
            error!("Error updating stack total hash: {}", e);
        }

        Ok(signature)
    }

    /// Handles the processing of an encrypted chunk response from the confidential compute service.
    ///
    /// This method attempts to receive and process an encrypted response from a previously initiated
    /// encryption request. It manages the oneshot channel receiver that was set up to receive the
    /// encrypted data.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Option<Value>, Error>` where:
    /// * `Ok(Some(Value))` - Successfully received and processed an encrypted chunk
    /// * `Ok(None)` - No encrypted chunk is available yet (receiver is empty)
    /// * `Err(Error)` - The channel has been dropped or another error occurred
    ///
    /// # State Changes
    ///
    /// * Consumes `encryption_response_receiver` using `take()` to process the response
    ///
    /// # Example Response Format
    ///
    /// When successful, returns a JSON object containing:
    /// ```json
    /// {
    ///     "ciphertext": "encrypted_data_here",
    ///     "nonce": "nonce_value_here"
    /// }
    /// ```
    #[instrument(
        level = "debug",
        skip(self),
        fields(path = "streamer-handle_encrypted_chunk")
    )]
    fn handle_encrypted_chunk(&mut self) -> Result<Option<Value>, Error> {
        if let Some(mut receiver) = self.encryption_response_receiver.take() {
            match receiver.try_recv() {
                Ok(ConfidentialComputeEncryptionResponse { ciphertext, nonce }) => {
                    // Construct encrypted JSON
                    let encrypted_chunk = json!({
                        "ciphertext": ciphertext,
                        "nonce": nonce,
                    });

                    return Ok(Some(encrypted_chunk));
                }
                Err(e) => {
                    if e == TryRecvError::Empty {
                        return Ok(None);
                    }
                    return Err(Error::new(
                        "Oneshot sender channel has been dropped".to_string(),
                    ));
                }
            }
        }
        Ok(None)
    }

    /// Handles the encryption request for a chunk of streaming data.
    ///
    /// This method initiates the encryption process for a given chunk by:
    /// 1. Creating a oneshot channel for receiving the encryption response
    /// 2. Sending the encryption request with the chunk data to the confidential compute service
    /// 3. Setting up the streamer to wait for the encrypted response
    ///
    /// # Arguments
    ///
    /// * `chunk` - The JSON value containing the data to be encrypted
    /// * `proxy_x25519_public_key` - The X25519 public key of the proxy (32 bytes)
    /// * `salt` - The salt value used in the encryption process
    ///
    /// # Returns
    ///
    /// Returns a `Result<(), Error>` where:
    /// * `Ok(())` - The encryption request was successfully sent
    /// * `Err(Error)` - An error occurred while sending the encryption request
    ///
    /// # State Changes
    ///
    /// * Sets `waiting_for_encrypted_chunk` to `true`
    /// * Updates `encryption_response_receiver` with the new receiver
    #[instrument(
        level = "debug",
        skip_all,
        fields(
            proxy_x25519_public_key = ?proxy_x25519_public_key
        )
    )]
    fn handle_encryption_request(
        &mut self,
        chunk: &Value,
        proxy_x25519_public_key: [u8; 32],
        salt: Vec<u8>,
    ) -> Result<(), Error> {
        let (sender, receiver) = oneshot::channel();
        self.confidential_compute_encryption_sender
            .send((
                ConfidentialComputeEncryptionRequest {
                    plaintext: chunk.to_string().into(),
                    proxy_x25519_public_key,
                    salt,
                },
                sender,
            ))
            .map_err(|e| {
                error!("Error sending encryption request: {}", e);
                Error::new(format!("Error sending encryption request: {}", e))
            })?;
        self.waiting_for_encrypted_chunk = true;
        self.encryption_response_receiver = Some(receiver);
        Ok(())
    }
}

impl Stream for Streamer {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamStatus::Completed {
            return Poll::Ready(None);
        }

        match self.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if self.waiting_for_encrypted_chunk {
                    match self.handle_encrypted_chunk() {
                        Ok(Some(chunk)) => {
                            self.waiting_for_encrypted_chunk = false;
                            return Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)));
                        }
                        Err(e) => return Poll::Ready(Some(Err(e))),
                        Ok(None) => return Poll::Pending,
                    }
                }

                if self.status != StreamStatus::Started {
                    self.status = StreamStatus::Started;
                }

                if chunk.as_ref() == KEEP_ALIVE_CHUNK {
                    return Poll::Pending;
                }

                let chunk_str = match std::str::from_utf8(&chunk) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("Invalid UTF-8 sequence: {}", e);
                        return Poll::Ready(Some(Err(Error::new(format!(
                            "Invalid UTF-8 sequence: {}",
                            e
                        )))));
                    }
                };

                let chunk_str = chunk_str.strip_prefix(DATA_PREFIX).unwrap_or(chunk_str);

                if chunk_str.starts_with(DONE_CHUNK) {
                    // This is the last chunk, meaning the inference streaming is complete
                    self.status = StreamStatus::Completed;
                    return Poll::Ready(None);
                }
                let mut chunk = serde_json::from_str::<Value>(chunk_str).map_err(|e| {
                    error!("Error parsing chunk: {}", e);
                    Error::new(format!("Error parsing chunk: {}", e))
                })?;

                // Observe the first token generation timer
                if let Some(timer) = self.first_token_generation_timer.take() {
                    timer.observe_duration();
                    let timer = CHAT_COMPLETIONS_DECODING_TIME
                        .with_label_values(&[&self.model])
                        .start_timer();
                    self.decoding_phase_timer = Some(timer);
                }

                let choices = match chunk.get(CHOICES).and_then(|choices| choices.as_array()) {
                    Some(choices) => choices,
                    None => {
                        error!("Error getting choices from chunk");
                        return Poll::Ready(Some(Err(Error::new(
                            "Error getting choices from chunk",
                        ))));
                    }
                };

                if choices.is_empty() {
                    // Check if this is a final chunk with usage info
                    if let Some(usage) = chunk.get(USAGE) {
                        self.status = StreamStatus::Completed;
                        let signature = self.handle_final_chunk(usage)?;
                        chunk["signature"] = json!(signature);
                        let client_encryption_metadata = self.client_encryption_metadata.clone();
                        if let Some(EncryptionMetadata {
                            proxy_x25519_public_key,
                            salt,
                        }) = client_encryption_metadata
                        {
                            // NOTE: We only need to perform chunk encryption when sending the chunk back to the client
                            self.handle_encryption_request(&chunk, proxy_x25519_public_key, salt)?;
                            // NOTE: We don't expect the encryption to be ready immediately, so we return pending
                            // for now, so next time we poll, we'll check if the encryption is ready
                            Poll::Pending
                        } else {
                            Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
                        }
                    } else {
                        error!("Error getting usage from chunk");
                        Poll::Ready(Some(Err(Error::new("Error getting usage from chunk"))))
                    }
                } else {
                    // Accumulate regular chunks
                    self.accumulated_response.push(chunk.clone());
                    let should_encrypt = self
                        .client_encryption_metadata
                        .as_ref()
                        .map(|metadata| (metadata.proxy_x25519_public_key, metadata.salt.clone()));
                    if let Some((proxy_x25519_public_key, salt)) = should_encrypt {
                        // NOTE: We only need to perform chunk encryption when sending the chunk back to the client
                        self.handle_encryption_request(&chunk, proxy_x25519_public_key, salt)?;
                        // NOTE: We don't expect the encryption to be ready immediately, so we return pending
                        // for now, so next time we poll, we'll check if the encryption is ready
                        Poll::Pending
                    } else {
                        Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
                    }
                }
            }
            Poll::Ready(Some(Err(e))) => {
                self.status = StreamStatus::Failed(e.to_string());
                Poll::Ready(None)
            }
            Poll::Ready(None) => {
                self.status = StreamStatus::Completed;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
