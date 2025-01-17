use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use atoma_state::types::AtomaAtomaStateManagerEvent;
use atoma_utils::{
    constants::{NONCE_SIZE, PAYLOAD_HASH_SIZE, SALT_SIZE},
    encryption::encrypt_plaintext,
    hashing::blake2b_hash,
};
use axum::body::Bytes;
use axum::{response::sse::Event, Error};
use base64::{engine::general_purpose::STANDARD, Engine};
use flume::Sender as FlumeSender;
use futures::Stream;
use prometheus::HistogramTimer;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tracing::{error, info, instrument};
use x25519_dalek::SharedSecret;

use crate::{
    handlers::{
        prometheus::{
            CHAT_COMPLETIONS_DECODING_TIME, CHAT_COMPLETIONS_INPUT_TOKENS_METRICS,
            CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS,
        },
        update_stack_num_compute_units, USAGE_KEY,
    },
    server::utils,
};

/// The chunk that indicates the end of a streaming response
const DONE_CHUNK: &str = "[DONE]";

/// The prefix for the data chunk
const DATA_PREFIX: &str = "data: ";

/// The keep-alive chunk
const KEEP_ALIVE_CHUNK: &[u8] = b": keep-alive\n\n";

/// The keep-alive-text chunk (used by mistralrs)
const KEEP_ALIVE_TEXT_CHUNK: &[u8] = b"keep-alive-text\n";

/// The choices key
const CHOICES: &str = "choices";

/// The ciphertext key
const CIPHERTEXT_KEY: &str = "ciphertext";

/// The nonce key
const NONCE_KEY: &str = "nonce";

/// The response hash key
const RESPONSE_HASH_KEY: &str = "response_hash";

/// The signature key
const SIGNATURE_KEY: &str = "signature";

/// Metadata required for encrypting streaming responses to clients.
///
/// This structure contains the cryptographic elements needed to establish
/// secure communication during streaming operations.
pub struct StreamingEncryptionMetadata {
    /// The shared secret key derived from ECDH key exchange
    pub shared_secret: SharedSecret,
    /// A unique nonce value to prevent replay attacks
    pub nonce: [u8; NONCE_SIZE],
    /// Additional randomness used in the encryption process
    pub salt: [u8; SALT_SIZE],
}

/// A structure for streaming chat completion chunks.
pub struct Streamer {
    /// The stream of bytes from the inference service
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    /// Current status of the stream
    status: StreamStatus,
    /// The stack small id for the request
    stack_small_id: i64,
    /// The estimated total compute units for the request
    estimated_total_compute_units: i64,
    /// The request payload hash
    payload_hash: [u8; PAYLOAD_HASH_SIZE],
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
    streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
    /// The endpoint for the request
    endpoint: String,
    /// A chunk buffer (needed as some chunks might be split into multiple parts)
    chunk_buffer: String,
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
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        keystore: Arc<FileBasedKeystore>,
        address_index: usize,
        model: String,
        streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
        endpoint: String,
        first_token_generation_timer: HistogramTimer,
    ) -> Self {
        Self {
            stream: Box::pin(stream),
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
            streaming_encryption_metadata,
            endpoint,
            chunk_buffer: String::new(),
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
    fn handle_final_chunk(
        &mut self,
        usage: &Value,
        response_hash: [u8; PAYLOAD_HASH_SIZE],
    ) -> Result<(), Error> {
        // Record the decoding phase timer
        if let Some(timer) = self.decoding_phase_timer.take() {
            timer.observe_duration();
        }

        // Get total tokens
        let mut total_compute_units = 0;
        if let Some(prompt_tokens) = usage.get("prompt_tokens") {
            let prompt_tokens = prompt_tokens.as_u64().unwrap_or(0);
            CHAT_COMPLETIONS_INPUT_TOKENS_METRICS
                .with_label_values(&[&self.model])
                .inc_by(prompt_tokens as f64);
            total_compute_units += prompt_tokens;
        } else {
            error!(
                target = "atoma-service-streamer",
                level = "error",
                "Error getting prompt tokens from usage"
            );
            return Err(Error::new("Error getting prompt tokens from usage"));
        }
        if let Some(completion_tokens) = usage.get("completion_tokens") {
            let completion_tokens = completion_tokens.as_u64().unwrap_or(0);
            CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS
                .with_label_values(&[&self.model])
                .inc_by(completion_tokens as f64);
            total_compute_units += completion_tokens;
        } else {
            error!(
                target = "atoma-service-streamer",
                level = "error",
                "Error getting completion tokens from usage"
            );
            return Err(Error::new("Error getting completion tokens from usage"));
        }

        info!(
            target = "atoma-service-streamer",
            level = "info",
            endpoint = self.endpoint,
            stack_small_id = self.stack_small_id,
            estimated_total_compute_units = self.estimated_total_compute_units,
            payload_hash = hex::encode(self.payload_hash),
            "Handle final chunk: Total compute units: {}",
            total_compute_units,
        );

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
            error!(
                target = "atoma-service-streamer",
                level = "error",
                endpoint = self.endpoint,
                "Error updating stack total hash: {}",
                e
            );
        }

        // Update stack num tokens
        if let Err(e) = update_stack_num_compute_units(
            &self.state_manager_sender,
            self.stack_small_id,
            self.estimated_total_compute_units,
            total_compute_units as i64,
            &self.endpoint,
        ) {
            error!(
                target = "atoma-service-streamer",
                level = "error",
                "Error updating stack num tokens: {}",
                e
            );
        }

        Ok(())
    }

    /// Signs the accumulated response  
    /// This is used when the streaming is complete and we need to send the final chunk back to the client
    /// with the signature and response hash
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// * A base64-encoded string of the signature
    /// * A base64-encoded string of the response hash
    #[instrument(level = "debug", skip_all)]
    pub fn sign_chunk(&self, chunk: &Value) -> Result<(String, [u8; PAYLOAD_HASH_SIZE]), Error> {
        // Sign the accumulated response
        let (response_hash, signature) =
            utils::sign_response_body(&chunk, &self.keystore, self.address_index).map_err(|e| {
                error!(
                    target = "atoma-service-streamer",
                    level = "error",
                    "Error signing response: {}",
                    e
                );
                Error::new(format!("Error signing response: {}", e))
            })?;

        Ok((signature, response_hash))
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
    /// * `usage` - The usage of the chunk
    /// * `streaming_encryption_metadata` - The streaming encryption metadata
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
    #[instrument(level = "debug", skip_all)]
    fn handle_encryption_request(
        chunk: &Value,
        usage: Option<&Value>,
        streaming_encryption_metadata: &StreamingEncryptionMetadata,
    ) -> Result<Value, Error> {
        let StreamingEncryptionMetadata {
            shared_secret,
            nonce,
            salt,
        } = streaming_encryption_metadata;
        // NOTE: We remove the usage key from the chunk before encryption
        // because we need to send the usage key back to the client in the final chunk
        let (encrypted_chunk, nonce) = if usage.is_some() {
            let mut chunk = chunk.clone();
            chunk.as_object_mut().map(|obj| obj.remove(USAGE_KEY));
            encrypt_plaintext(
                chunk.to_string().as_bytes(),
                shared_secret,
                salt,
                Some(*nonce),
            )
        } else {
            encrypt_plaintext(
                chunk.to_string().as_bytes(),
                shared_secret,
                salt,
                Some(*nonce),
            )
        }
        .map_err(|e| {
            error!(
                target = "atoma-service",
                level = "error",
                "Error encrypting chunk: {}",
                e
            );
            Error::new(format!("Error encrypting chunk: {}", e))
        })?;
        if let Some(usage) = usage {
            Ok(json!({
                CIPHERTEXT_KEY: STANDARD.encode(encrypted_chunk),
                NONCE_KEY: STANDARD.encode(nonce),
                USAGE_KEY: usage.clone(),
            }))
        } else {
            Ok(json!({
                CIPHERTEXT_KEY: STANDARD.encode(encrypted_chunk),
                NONCE_KEY: STANDARD.encode(nonce),
            }))
        }
    }

    /// Processes an individual chunk from the streaming response.
    ///
    /// This method handles the processing of each chunk received from the streaming response,
    /// including parsing, validation, and transformation of the data. It supports both regular
    /// streaming chunks and final chunks containing usage information.
    ///
    /// # Processing Flow
    /// 1. Updates stream status to Started if not already set
    /// 2. Handles keep-alive messages by returning Poll::Pending
    /// 3. Parses the chunk from UTF-8 bytes and removes any data prefix
    /// 4. Handles the [DONE] marker indicating end of stream
    /// 5. Parses the chunk as JSON, managing partial chunks using a buffer
    /// 6. Records timing metrics for the first token and decoding phase
    /// 7. Processes either:
    ///    - Regular chunks: Accumulates and encrypts if needed
    ///    - Final chunks: Handles usage info, signatures, and state updates
    ///
    /// # Arguments
    /// * `chunk` - The raw bytes received from the stream
    ///
    /// # Returns
    /// Returns a `Poll` containing:
    /// * `Some(Ok(Event))` - A successfully processed chunk ready to send to the client
    /// * `Some(Err(Error))` - An error occurred during processing
    /// * `None` - Stream has completed ([DONE] received)
    /// * `Poll::Pending` - More data needed to complete chunk processing
    ///
    /// # Error Handling
    /// The method handles several types of errors:
    /// * UTF-8 parsing errors
    /// * JSON parsing errors (including partial chunks)
    /// * Missing required fields (choices, usage)
    /// * Encryption errors
    ///
    /// # State Changes
    /// * Updates `status` field
    /// * Manages `chunk_buffer` for partial chunks
    /// * Manages timing metrics via `first_token_generation_timer` and `decoding_phase_timer`
    #[instrument(
        level = "info",
        skip_all,
        fields(
            endpoint = self.endpoint,
            stack_small_id = self.stack_small_id,
            estimated_total_compute_units = self.estimated_total_compute_units,
            payload_hash = hex::encode(self.payload_hash)
        )
    )]
    fn handle_streaming_chunk(&mut self, chunk: Bytes) -> Poll<Option<Result<Event, Error>>> {
        if self.status != StreamStatus::Started {
            self.status = StreamStatus::Started;
        }

        if chunk.as_ref() == KEEP_ALIVE_CHUNK || chunk.as_ref() == KEEP_ALIVE_TEXT_CHUNK {
            return Poll::Pending;
        }

        let chunk_str = match std::str::from_utf8(&chunk) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    target = "atoma-service",
                    level = "error",
                    "Invalid UTF-8 sequence: {}",
                    e
                );
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

        let chunk = match serde_json::from_str::<Value>(chunk_str) {
            Ok(chunk) => {
                if !self.chunk_buffer.is_empty() {
                    error!(
                        target = "atoma-service-streamer",
                        level = "error",
                        "Error parsing previous chunk(s), as chunk buffer is not empty: {}",
                        self.chunk_buffer
                    );
                    self.chunk_buffer.clear();
                }
                chunk
            }
            Err(e) => {
                if e.is_eof() {
                    info!(
                        target = "atoma-service-streamer",
                        parse_chunk = "eof_chunk",
                        "EOF reached, pushing chunk to buffer: {}",
                        chunk_str
                    );
                    self.chunk_buffer.push_str(chunk_str);
                    return Poll::Pending;
                }

                if self.chunk_buffer.is_empty() {
                    error!(
                        target = "atoma-service-streamer",
                        level = "error",
                        "Error parsing chunk {chunk_str}: {}",
                        e
                    );
                    return Poll::Ready(Some(Err(Error::new(format!(
                        "Error parsing chunk: {}",
                        e
                    )))));
                }

                self.chunk_buffer.push_str(chunk_str);
                match serde_json::from_str::<Value>(&self.chunk_buffer) {
                    Ok(chunk) => {
                        info!(
                            target = "atoma-service-streamer",
                            parse_chunk = "eof_chunk",
                            "Chunk parsed successfully, clearing buffer: {}",
                            self.chunk_buffer
                        );
                        self.chunk_buffer.clear();
                        chunk
                    }
                    Err(e) => {
                        if e.is_eof() {
                            // NOTE: We don't need to push the chunk to the buffer, as it was pushed already
                            return Poll::Pending;
                        }
                        error!(
                            target = "atoma-service-streamer",
                            level = "error",
                            "Error parsing chunk {}: {}",
                            self.chunk_buffer,
                            e
                        );
                        self.chunk_buffer.clear();
                        return Poll::Ready(Some(Err(Error::new(format!(
                            "Error parsing chunk: {}",
                            e
                        )))));
                    }
                }
            }
        };

        // Observe the first token generation timer
        if let Some(timer) = self.first_token_generation_timer.take() {
            timer.observe_duration();
            let timer = CHAT_COMPLETIONS_DECODING_TIME
                .with_label_values(&[&self.model])
                .start_timer();
            self.decoding_phase_timer = Some(timer);
        }

        let (signature, response_hash) = self.sign_chunk(&chunk)?;

        let choices = match chunk.get(CHOICES).and_then(|choices| choices.as_array()) {
            Some(choices) => choices,
            None => {
                error!(
                    target = "atoma-service",
                    level = "error",
                    endpoint = self.endpoint,
                    "Error getting choices from chunk"
                );
                return Poll::Ready(Some(Err(Error::new("Error getting choices from chunk"))));
            }
        };

        if choices.is_empty() {
            // Check if this is a final chunk with usage info
            if let Some(usage) = chunk.get(USAGE_KEY) {
                self.status = StreamStatus::Completed;
                let mut chunk = if let Some(streaming_encryption_metadata) =
                    self.streaming_encryption_metadata.as_ref()
                {
                    // NOTE: We only need to perform chunk encryption when sending the chunk back to the client
                    Self::handle_encryption_request(
                        &chunk,
                        Some(usage),
                        streaming_encryption_metadata,
                    )?
                } else {
                    chunk.clone()
                };
                self.handle_final_chunk(usage, response_hash)?;
                update_chunk(&mut chunk, signature, response_hash);
                Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
            } else {
                error!(
                    target = "atoma-service",
                    level = "error",
                    endpoint = self.endpoint,
                    "Error getting usage from chunk"
                );
                Poll::Ready(Some(Err(Error::new("Error getting usage from chunk"))))
            }
        } else {
            let mut chunk = if let Some(streaming_encryption_metadata) =
                self.streaming_encryption_metadata.as_ref()
            {
                // NOTE: We only need to perform chunk encryption when sending the chunk back to the client
                Self::handle_encryption_request(&chunk, None, streaming_encryption_metadata)?
            } else {
                chunk
            };
            update_chunk(&mut chunk, signature, response_hash);
            Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
        }
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
                match self.handle_streaming_chunk(chunk) {
                    Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(Ok(event))),
                    Poll::Ready(Some(Err(e))) => {
                        self.status = StreamStatus::Failed(e.to_string());
                        // NOTE: We need to update the stack number of tokens as the service failed to generate
                        // a proper response. For this reason, we set the total number of tokens to 0.
                        // This will ensure that the stack number of tokens is not updated, and the stack
                        // will not be penalized for the failed request.
                        if let Err(e) = update_stack_num_compute_units(
                            &self.state_manager_sender,
                            self.stack_small_id,
                            self.estimated_total_compute_units,
                            0,
                            &self.endpoint,
                        ) {
                            error!(
                                target = "atoma-service-streamer",
                                level = "error",
                                "Error updating stack num tokens: {}",
                                e
                            );
                        }
                        Poll::Ready(Some(Err(e)))
                    }
                    Poll::Ready(None) => Poll::Ready(None),
                    Poll::Pending => Poll::Pending,
                }
            }
            Poll::Ready(Some(Err(e))) => {
                self.status = StreamStatus::Failed(e.to_string());
                // NOTE: We need to update the stack number of tokens as the service failed to generate
                // a proper response. For this reason, we set the total number of tokens to 0.
                // This will ensure that the stack number of tokens is not updated, and the stack
                // will not be penalized for the failed request.
                if let Err(e) = update_stack_num_compute_units(
                    &self.state_manager_sender,
                    self.stack_small_id,
                    self.estimated_total_compute_units,
                    0,
                    &self.endpoint,
                ) {
                    error!(
                        target = "atoma-service-streamer",
                        level = "error",
                        "Error updating stack num tokens: {}",
                        e
                    );
                }
                Poll::Ready(None)
            }
            Poll::Ready(None) => {
                if !self.chunk_buffer.is_empty() {
                    error!(
                        target = "atoma-service-streamer",
                        level = "error",
                        "Stream ended, but the chunk buffer is not empty, this should not happen: {}",
                        self.chunk_buffer
                    );
                }
                self.status = StreamStatus::Completed;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Updates the final chunk with the signature and response hash    
/// This is used when the streaming is complete and we need to send the final chunk back to the client
/// with the signature and response hash
///
/// # Arguments
///
/// * `chunk` - The chunk to update (mut ref, as we update the chunk in place)
/// * `signature` - The signature to update the chunk with
/// * `response_hash` - The response hash to update the chunk with
fn update_chunk(chunk: &mut Value, signature: String, response_hash: [u8; PAYLOAD_HASH_SIZE]) {
    chunk[SIGNATURE_KEY] = json!(signature);
    chunk[RESPONSE_HASH_KEY] = json!(STANDARD.encode(response_hash));
}
