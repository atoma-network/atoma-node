use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use atoma_state::types::AtomaAtomaStateManagerEvent;
use atoma_utils::{
    constants::{NONCE_SIZE, PAYLOAD_HASH_SIZE, SALT_SIZE},
    encryption::encrypt_plaintext,
};
use axum::body::Bytes;
use axum::{response::sse::Event, Error};
use base64::{engine::general_purpose::STANDARD, Engine};
use dashmap::{DashMap, DashSet};
use flume::Sender as FlumeSender;
use futures::Stream;
use opentelemetry::KeyValue;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tracing::{error, info, instrument};
use x25519_dalek::SharedSecret;

use crate::{
    handlers::{
        handle_concurrent_requests_count_decrement,
        metrics::{
            CHAT_COMPLETIONS_DECODING_TIME, CHAT_COMPLETIONS_INPUT_TOKENS_METRICS,
            CHAT_COMPLETIONS_INTER_TOKEN_GENERATION_TIME, CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS,
            CHAT_COMPLETIONS_STREAMING_LATENCY_METRICS, CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN,
            TOTAL_COMPLETED_REQUESTS,
        },
        update_fiat_amount, update_stack_num_compute_units, USAGE_KEY,
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

/// The ciphertext key
const CIPHERTEXT_KEY: &str = "ciphertext";

/// The prompt tokens key
const PROMPT_TOKENS_KEY: &str = "prompt_tokens";

/// The completion tokens key
const COMPLETION_TOKENS_KEY: &str = "completion_tokens";

/// The total tokens key
const TOTAL_TOKENS_KEY: &str = "total_tokens";

/// The model key
const MODEL_KEY: &str = "model";

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
    /// The number of concurrent requests for the stack
    concurrent_requests: Arc<DashMap<i64, u64>>,
    /// The client dropped streamer connections
    client_dropped_streamer_connections: Arc<DashSet<String>>,
    /// The stream of bytes from the inference service
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    /// Current status of the stream
    status: StreamStatus,
    /// The stack small id for the request
    stack_small_id: Option<i64>,
    /// The estimated output tokens for the request
    estimated_output_tokens: i64,
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
    first_token_generation_timer: Option<Instant>,
    /// The decoding phase timer for the request.
    decoding_phase_timer: Option<Instant>,
    /// The client encryption metadata for the request
    streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
    /// The endpoint for the request
    endpoint: String,
    /// The request ID for the request
    request_id: String,
    /// A chunk buffer (needed as some chunks might be split into multiple parts)
    chunk_buffer: String,
    /// Timer for measuring time between token generations
    inter_stream_token_latency_timer: Option<Instant>,
    /// Whether the final chunk has been handled already or not,
    /// useful in situations where the client kills the connection
    /// before the final chunk is sent
    is_final_chunk_handled: bool,
    /// The number of tokens computed so far, this is used when the client
    /// kills the connection before the final chunk is sent. If, instead,
    /// the last chunk is handled, the value is updated to the actual number of tokens
    /// returned by the LLM inference service
    streamer_computed_num_tokens: i64,
    /// The number of input tokens for the request
    num_input_tokens: i64,
    /// The price per one million tokens for the request
    price_per_one_million_tokens: i64,
    /// The user address for the request
    user_address: String,
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
        concurrent_requests: Arc<DashMap<i64, u64>>,
        client_dropped_streamer_connections: Arc<DashSet<String>>,
        stack_small_id: Option<i64>,
        num_input_tokens: i64,
        estimated_output_tokens: i64,
        payload_hash: [u8; PAYLOAD_HASH_SIZE],
        keystore: Arc<FileBasedKeystore>,
        address_index: usize,
        model: String,
        streaming_encryption_metadata: Option<StreamingEncryptionMetadata>,
        endpoint: String,
        request_id: String,
        first_token_generation_timer: Instant,
        price_per_one_million_tokens: i64,
        user_address: String,
    ) -> Self {
        Self {
            concurrent_requests,
            client_dropped_streamer_connections,
            stream: Box::pin(stream),
            status: StreamStatus::NotStarted,
            stack_small_id,
            estimated_output_tokens,
            payload_hash,
            state_manager_sender,
            keystore,
            address_index,
            model,
            first_token_generation_timer: Some(first_token_generation_timer),
            decoding_phase_timer: None,
            streaming_encryption_metadata,
            endpoint,
            request_id,
            chunk_buffer: String::new(),
            inter_stream_token_latency_timer: None,
            is_final_chunk_handled: false,
            streamer_computed_num_tokens: 0,
            num_input_tokens,
            price_per_one_million_tokens,
            user_address,
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
    /// 7. Updates the stack num tokens
    /// 8. Sets the variable `is_final_chunk_handled` to `true`
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
    #[instrument(
        level = "info",
        skip(self, usage),
        fields(
            endpoint = "handle_final_chunk",
            stack_small_id = self.stack_small_id,
            estimated_output_tokens = self.estimated_output_tokens,
            payload_hash = hex::encode(self.payload_hash)
        ),
        err
    )]
    fn handle_final_chunk(
        &mut self,
        usage: &Value,
        response_hash: [u8; PAYLOAD_HASH_SIZE],
    ) -> Result<(), Error> {
        let privacy_level = if self.streaming_encryption_metadata.is_some() {
            "confidential"
        } else {
            "non-confidential"
        };

        // Record the decoding phase timer
        if let Some(timer) = self.decoding_phase_timer.take() {
            CHAT_COMPLETIONS_DECODING_TIME.record(
                timer.elapsed().as_secs_f64(),
                &[
                    KeyValue::new("model", self.model.clone()),
                    KeyValue::new("privacy_level", privacy_level),
                ],
            );
        }

        // Get total tokens
        let mut input_tokens = 0;
        let mut output_tokens = 0;
        if let Some(prompt_tokens) = usage.get("prompt_tokens") {
            let prompt_tokens = prompt_tokens.as_u64().unwrap_or(0);
            CHAT_COMPLETIONS_INPUT_TOKENS_METRICS.add(
                prompt_tokens,
                &[
                    KeyValue::new("model", self.model.clone()),
                    KeyValue::new("privacy_level", privacy_level),
                ],
            );
            input_tokens += prompt_tokens;
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
            CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS.add(
                completion_tokens,
                &[
                    KeyValue::new("model", self.model.clone()),
                    KeyValue::new("privacy_level", privacy_level),
                ],
            );
            output_tokens += completion_tokens;
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
            estimated_total_tokens = self.num_input_tokens + self.estimated_output_tokens,
            payload_hash = hex::encode(self.payload_hash),
            "Handle final chunk: Total tokens: {}",
            input_tokens + output_tokens,
        );

        if let Some(stack_small_id) = self.stack_small_id {
            // Update stack num tokens
            let concurrent_requests = handle_concurrent_requests_count_decrement(
                &self.concurrent_requests,
                stack_small_id,
                &self.endpoint,
            );
            if let Err(e) = update_stack_num_compute_units(
                &self.state_manager_sender,
                stack_small_id,
                self.num_input_tokens + self.estimated_output_tokens,
                (input_tokens + output_tokens) as i64,
                &self.endpoint,
                concurrent_requests,
            ) {
                error!(
                    target = "atoma-service-streamer",
                    level = "error",
                    "Error updating stack num tokens: {}",
                    e
                );
            }
        } else if let Err(e) = update_fiat_amount(
            &self.state_manager_sender,
            self.user_address.clone(),
            self.num_input_tokens,
            input_tokens as i64,
            self.estimated_output_tokens,
            output_tokens as i64,
            self.price_per_one_million_tokens,
            &self.endpoint,
        ) {
            error!(
                target = "atoma-service-streamer",
                level = "error",
                endpoint = self.endpoint,
                "Error updating fiat amount: {}",
                e
            );
        }

        self.is_final_chunk_handled = true;

        CHAT_COMPLETIONS_STREAMING_LATENCY_METRICS.record(
            self.inter_stream_token_latency_timer
                .unwrap()
                .elapsed()
                .as_secs_f64(),
            &[
                KeyValue::new("model", self.model.clone()),
                KeyValue::new("privacy_level", privacy_level),
            ],
        );

        Ok(())
    }

    /// Signs the each chunk of the response
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// * A base64-encoded string of the signature
    /// * A base64-encoded string of the response hash
    #[instrument(level = "debug", skip_all, err)]
    pub fn sign_chunk(&self, chunk: &Value) -> Result<(String, [u8; PAYLOAD_HASH_SIZE]), Error> {
        let (response_hash, signature) =
            utils::sign_response_body(chunk, &self.keystore, self.address_index).map_err(|e| {
                error!(
                    target = "atoma-service-streamer",
                    level = "error",
                    "Error signing response: {}",
                    e
                );
                Error::new(format!("Error signing response: {e}"))
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
    #[instrument(level = "debug", skip_all, err)]
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
        let (encrypted_chunk, nonce) = encrypt_plaintext(
            chunk.to_string().as_bytes(),
            shared_secret,
            salt,
            Some(*nonce),
        )
        .map_err(|e| {
            error!(
                target = "atoma-service",
                level = "error",
                "Error encrypting chunk: {}",
                e
            );
            Error::new(format!("Error encrypting chunk: {e}"))
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
        level = "trace",
        skip_all,
        fields(
            endpoint = self.endpoint,
            stack_small_id = self.stack_small_id,
            estimated_total_tokens = self.num_input_tokens + self.estimated_output_tokens,
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
                    "Invalid UTF-8 sequence: {e}",
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
                    return Poll::Ready(Some(Err(Error::new(
                        format!("Error parsing chunk: {e}",),
                    ))));
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
                            "Error parsing chunk: {e}",
                        )))));
                    }
                }
            }
        };

        // Observe the first token generation timer
        if let Some(timer) = self.first_token_generation_timer.take() {
            CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN.record(
                timer.elapsed().as_secs_f64(),
                &[
                    KeyValue::new("model", self.model.clone()),
                    KeyValue::new(
                        "privacy_level",
                        if self.streaming_encryption_metadata.is_some() {
                            "confidential"
                        } else {
                            "non-confidential"
                        },
                    ),
                ],
            );
            self.decoding_phase_timer = Some(timer);
        }

        let (signature, response_hash) = self.sign_chunk(&chunk)?;

        // Check if this is a final chunk with usage info
        if let Some(usage) = chunk.get(USAGE_KEY).filter(|v| !v.is_null()) {
            self.status = StreamStatus::Completed;
            let mut chunk = if let Some(streaming_encryption_metadata) =
                self.streaming_encryption_metadata.as_ref()
            {
                // NOTE: We only need to perform chunk encryption when sending the chunk back to the client
                Self::handle_encryption_request(&chunk, Some(usage), streaming_encryption_metadata)?
            } else {
                chunk.clone()
            };
            self.handle_final_chunk(usage, response_hash)?;
            update_chunk(&mut chunk, &signature, response_hash);
            Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
        } else {
            let mut chunk = if let Some(streaming_encryption_metadata) =
                self.streaming_encryption_metadata.as_ref()
            {
                // NOTE: We only need to perform chunk encryption when sending the chunk back to the client
                Self::handle_encryption_request(&chunk, None, streaming_encryption_metadata)?
            } else {
                chunk
            };
            update_chunk(&mut chunk, &signature, response_hash);
            // NOTE: We increment the number of tokens computed so far, as we are processing a new chunk
            // which corresponds to a new generated token.
            self.streamer_computed_num_tokens += 1;
            if let Some(_client_dropped_streamer_connection) = self
                .client_dropped_streamer_connections
                .remove(&self.request_id)
            {
                info!(
                    target = "atoma-service-streamer",
                    level = "info",
                    endpoint = self.endpoint,
                    "Client dropped streamer connection, updating usage"
                );
                self.status = StreamStatus::Completed;
                chunk[USAGE_KEY] = json!({
                    PROMPT_TOKENS_KEY: self.num_input_tokens,
                    COMPLETION_TOKENS_KEY: self.streamer_computed_num_tokens,
                    TOTAL_TOKENS_KEY: self.num_input_tokens + self.streamer_computed_num_tokens,
                });
                // NOTE: At this point, we will need to re-sign the chunk, as we added the usage key.
                // This is also the last chunk, as the connection was dropped, and therefore, there is
                // little to no latency overhead to do it once more.
                //
                // 1. Remove the previous signature and response hash from the chunk
                if let Some(obj) = chunk.as_object_mut() {
                    obj.remove(SIGNATURE_KEY);
                    obj.remove(RESPONSE_HASH_KEY);
                }
                // 2. Sign the chunk again
                let (signature, response_hash) = self.sign_chunk(&chunk)?;
                // 3. Update the chunk with the new signature and response hash
                update_chunk(&mut chunk, &signature, response_hash);
                info!(
                    target = "atoma-service-streamer",
                    level = "info",
                    endpoint = self.endpoint,
                    "Client dropped streamer connection, updating usage, chunk = {chunk}"
                );
                return Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)));
            }
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
            Poll::Ready(Some(Ok(chunk))) => self.handle_poll_chunk(chunk),
            Poll::Ready(Some(Err(e))) => self.handle_poll_error(&e),
            Poll::Ready(None) => self.handle_poll_complete(),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Streamer {
    /// Handles a successful chunk from the stream
    fn handle_poll_chunk(&mut self, chunk: Bytes) -> Poll<Option<Result<Event, Error>>> {
        match self.handle_streaming_chunk(chunk) {
            Poll::Ready(Some(Ok(event))) => self.handle_successful_event(event),
            Poll::Ready(Some(Err(e))) => self.handle_streaming_error(e),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Handles a successful event, updating timers
    fn handle_successful_event(&mut self, event: Event) -> Poll<Option<Result<Event, Error>>> {
        // Observe the previous timer if it exists
        if let Some(timer) = self.inter_stream_token_latency_timer.take() {
            let elapsed = timer.elapsed();
            let privacy_level = if self.streaming_encryption_metadata.is_some() {
                "confidential"
            } else {
                "non-confidential"
            };
            CHAT_COMPLETIONS_INTER_TOKEN_GENERATION_TIME.record(
                elapsed.as_secs_f64(),
                &[
                    KeyValue::new("model", self.model.clone()),
                    KeyValue::new("privacy_level", privacy_level),
                ],
            );
        }
        // Start the timer after we've processed this chunk
        self.inter_stream_token_latency_timer = Some(Instant::now());

        Poll::Ready(Some(Ok(event)))
    }

    /// Handles errors during streaming
    fn handle_streaming_error(&mut self, e: Error) -> Poll<Option<Result<Event, Error>>> {
        self.status = StreamStatus::Failed(e.to_string());
        self.update_balance_on_error();
        Poll::Ready(Some(Err(e)))
    }

    /// Handles stream poll errors
    fn handle_poll_error(&mut self, e: &reqwest::Error) -> Poll<Option<Result<Event, Error>>> {
        self.status = StreamStatus::Failed(e.to_string());
        self.update_balance_on_error();
        Poll::Ready(None)
    }

    /// Handles stream completion
    fn handle_poll_complete(&mut self) -> Poll<Option<Result<Event, Error>>> {
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

    /// Updates balance (stack or fiat) when an error occurs
    #[instrument(
        level = "error",
        skip_all,
        fields(
            endpoint = self.endpoint,
            stack_small_id = self.stack_small_id,
            estimated_total_tokens = self.num_input_tokens + self.estimated_output_tokens,
            payload_hash = hex::encode(self.payload_hash)
        )
    )]
    fn update_balance_on_error(&self) {
        if let Some(stack_small_id) = self.stack_small_id {
            // NOTE: We need to update the stack number of tokens as the service failed to generate
            // a proper response. For this reason, we set the total number of tokens to 0.
            // This will ensure that the stack number of tokens is not updated, and the stack
            // will not be penalized for the failed request.
            //
            // NOTE: We also decrement the concurrent requests count, as we are done processing the request.
            let concurrent_requests = handle_concurrent_requests_count_decrement(
                &self.concurrent_requests,
                stack_small_id,
                &self.endpoint,
            );
            if let Err(e) = update_stack_num_compute_units(
                &self.state_manager_sender,
                stack_small_id,
                self.num_input_tokens + self.estimated_output_tokens,
                0,
                &self.endpoint,
                concurrent_requests,
            ) {
                error!(
                    target = "atoma-service-streamer",
                    level = "error",
                    "Error updating stack num tokens: {}",
                    e
                );
            }
        } else if let Err(e) = update_fiat_amount(
            &self.state_manager_sender,
            self.user_address.clone(),
            self.num_input_tokens,
            0,
            self.estimated_output_tokens,
            0,
            self.price_per_one_million_tokens,
            &self.endpoint,
        ) {
            error!(
                target = "atoma-service-streamer",
                level = "error",
                endpoint = self.endpoint,
                "Error updating fiat amount: {}",
                e
            );
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
fn update_chunk(chunk: &mut Value, signature: &str, response_hash: [u8; PAYLOAD_HASH_SIZE]) {
    chunk[SIGNATURE_KEY] = json!(signature);
    chunk[RESPONSE_HASH_KEY] = json!(STANDARD.encode(response_hash));
}

impl Drop for Streamer {
    /// Implements the Drop trait to handle cleanup when the streamer is dropped
    ///
    /// This method ensures proper resource cleanup and state management when the streamer
    /// is dropped, particularly in cases where the final chunk hasn't been handled.
    ///
    /// # Implementation Details
    ///
    /// - Checks if the final chunk has already been handled to prevent duplicate cleanup
    /// - If the final chunk has not been handled, it records the decoding phase timer taken so far
    /// - Decrements the concurrent request counter for the associated stack
    /// - Updates the stack's compute units through the state manager
    /// - Sets the stream status to Completed
    ///
    /// # Instrumentation
    ///
    /// This method is instrumented with the following fields for tracing:
    /// - `endpoint`: The API endpoint being accessed
    /// - `stack_small_id`: The unique identifier for the stack
    /// - `estimated_total_tokens`: Expected tokens for the operation
    /// - `payload_hash`: Hex-encoded hash of the request payload
    ///
    /// # Error Handling
    ///
    /// If updating the stack's tokens fails, the error is logged but not propagated
    /// since this is a destructor implementation.
    #[instrument(
        level = "info",
        skip_all,
        fields(
            endpoint = self.endpoint,
            stack_small_id = self.stack_small_id,
            estimated_total_tokens = self.num_input_tokens + self.estimated_output_tokens,
            payload_hash = hex::encode(self.payload_hash)
        )
    )]
    fn drop(&mut self) {
        if self.is_final_chunk_handled || matches!(self.status, StreamStatus::Failed(_)) {
            TOTAL_COMPLETED_REQUESTS.add(1, &[KeyValue::new(MODEL_KEY, self.model.clone())]);
            return;
        }
        if let Some(timer) = self.decoding_phase_timer.take() {
            CHAT_COMPLETIONS_DECODING_TIME.record(
                timer.elapsed().as_secs_f64(),
                &[
                    KeyValue::new("model", self.model.clone()),
                    KeyValue::new(
                        "privacy_level",
                        if self.streaming_encryption_metadata.is_some() {
                            "confidential"
                        } else {
                            "non-confidential"
                        },
                    ),
                ],
            );
        }
        if let Some(stack_small_id) = self.stack_small_id {
            let concurrent_requests = handle_concurrent_requests_count_decrement(
                &self.concurrent_requests,
                stack_small_id,
                &self.endpoint,
            );
            if let Err(e) = update_stack_num_compute_units(
                &self.state_manager_sender,
                stack_small_id,
                self.num_input_tokens + self.estimated_output_tokens,
                self.num_input_tokens + self.streamer_computed_num_tokens,
                &self.endpoint,
                concurrent_requests,
            ) {
                error!(
                    target = "atoma-service-streamer",
                    level = "error",
                    "Error updating stack num tokens: {}",
                    e
                );
            }
        } else if let Err(e) = update_fiat_amount(
            &self.state_manager_sender,
            self.user_address.clone(),
            self.num_input_tokens,
            self.num_input_tokens,
            self.estimated_output_tokens,
            self.streamer_computed_num_tokens,
            self.price_per_one_million_tokens,
            &self.endpoint,
        ) {
            error!(
                target = "atoma-service-streamer",
                level = "error",
                endpoint = self.endpoint,
                "Error updating fiat amount: {}",
                e
            );
        }
        self.status = StreamStatus::Completed;
    }
}
