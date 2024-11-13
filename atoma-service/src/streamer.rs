use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use atoma_state::types::AtomaAtomaStateManagerEvent;
use axum::body::Bytes;
use axum::{response::sse::Event, Error};
use blake2::{
    digest::generic_array::{typenum::U32, GenericArray},
    Blake2b, Digest,
};
use flume::Sender as FlumeSender;
use futures::Stream;
use reqwest;
use serde_json::{json, Value};
use sui_keys::keystore::FileBasedKeystore;
use tracing::{error, instrument};

use crate::server::utils;

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
    pub fn new(
        stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
        state_manager_sender: FlumeSender<AtomaAtomaStateManagerEvent>,
        stack_small_id: i64,
        estimated_total_compute_units: i64,
        payload_hash: [u8; 32],
        keystore: Arc<FileBasedKeystore>,
        address_index: usize,
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
        let total_compute_units = usage
            .get("total_tokens")
            .and_then(|t| t.as_i64())
            .ok_or_else(|| {
                error!("Error getting total tokens from usage");
                Error::new("Error getting total tokens from usage")
            })?;

        // Update stack num tokens
        if let Err(e) = self.state_manager_sender.send(
            AtomaAtomaStateManagerEvent::UpdateStackNumComputeUnits {
                stack_small_id: self.stack_small_id,
                estimated_total_compute_units: self.estimated_total_compute_units,
                total_compute_units,
            },
        ) {
            error!("Error updating stack num tokens: {}", e);
        }

        // Calculate and update total hash
        let mut blake2b = Blake2b::new();
        blake2b.update([self.payload_hash, response_hash].concat());
        let total_hash: GenericArray<u8, U32> = blake2b.finalize();
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
}

impl Stream for Streamer {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamStatus::Completed {
            return Poll::Ready(None);
        }

        match self.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
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
                        Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
                    } else {
                        error!("Error getting usage from chunk");
                        Poll::Ready(Some(Err(Error::new("Error getting usage from chunk"))))
                    }
                } else {
                    // Accumulate regular chunks
                    self.accumulated_response.push(chunk.clone());
                    Poll::Ready(Some(Ok(Event::default().json_data(&chunk)?)))
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