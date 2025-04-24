pub mod chat_completions;
pub mod embeddings;
pub mod image_generations;
pub mod metrics;
pub mod request_model;
pub mod stop_streamer;

use std::sync::Arc;

use atoma_confidential::types::{
    ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
};
use atoma_utils::hashing::blake2b_hash;
use base64::{engine::general_purpose::STANDARD, Engine};
use dashmap::DashMap;
use flume::Sender;
use hyper::StatusCode;
use image_generations::CONFIDENTIAL_IMAGE_GENERATIONS_PATH;
use serde_json::{json, Value};
use tracing::{info, instrument};

use crate::{
    error::AtomaServiceError,
    middleware::EncryptionMetadata,
    server::{utils, AppState},
};
use atoma_state::types::AtomaAtomaStateManagerEvent;

/// Key for the ciphertext in the response body
const CIPHERTEXT_KEY: &str = "ciphertext";

/// The default max tokens for a chat completion request
const DEFAULT_MAX_TOKENS: u64 = 8_192;

/// Key for the nonce in the response body
const NONCE_KEY: &str = "nonce";

/// Key for the response hash in the response body
const RESPONSE_HASH_KEY: &str = "response_hash";

/// Key for the signature in the response body
const SIGNATURE_KEY: &str = "signature";

/// Key for the usage in the response body
pub const USAGE_KEY: &str = "usage";

/// Updates response signature and stack hash state
///
/// # Arguments
///
/// * `response_body` - Mutable reference to the response JSON
/// * `payload_hash` - Hash of the original request payload
/// * `state` - Application state containing keystore and state manager
/// * `stack_small_id` - Identifier for the current stack
///
/// # Returns
///
/// Returns Result<(), AtomaServiceError> indicating success or failure
#[instrument(
    level = "info",
    skip(response_body, state),
    fields(event = "sign-response-and-update-stack-hash",),
    err
)]
async fn sign_response_and_update_stack_hash(
    response_body: &mut Value,
    payload_hash: [u8; 32],
    state: &AppState,
    stack_small_id: i64,
    endpoint: String,
) -> Result<(), AtomaServiceError> {
    // Sign the response body byte content and add the base64 encoded signature to the response body
    let (response_hash, signature) =
        utils::sign_response_body(response_body, &state.keystore, state.address_index).map_err(
            |e| AtomaServiceError::InternalError {
                message: format!("Error signing response body: {}", e),
                endpoint: endpoint.clone(),
            },
        )?;
    response_body[SIGNATURE_KEY] = json!(signature);
    response_body[RESPONSE_HASH_KEY] = json!(STANDARD.encode(response_hash));

    // Update the stack total hash
    let total_hash = blake2b_hash(&[payload_hash, response_hash].concat());
    let total_hash_bytes: [u8; 32] = total_hash
        .as_slice()
        .try_into()
        .expect("Invalid BLAKE2b hash length");

    // Send the update stack total hash event to the state manager
    state
        .state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateStackTotalHash {
            stack_small_id,
            total_hash: total_hash_bytes,
        })
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error updating stack total hash: {}", e),
            endpoint: endpoint.clone(),
        })?;

    Ok(())
}

/// Handles the encryption of response data for confidential compute requests
///
/// This function processes the response body based on the presence of encryption metadata.
/// If encryption metadata is provided, it encrypts the response using the provided
/// public key and salt. Otherwise, it returns the response body unchanged.
///
/// # Arguments
///
/// * `state` - Reference to the application state containing encryption channels
/// * `response_body` - The response data to potentially encrypt
/// * `client_encryption_metadata` - Optional metadata containing encryption parameters
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * An encrypted response as JSON with `nonce` and `ciphertext` fields
/// * The original response body if no encryption was requested
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// * Failed to send encryption request through the channel
/// * Failed to receive encryption response
///
/// # Example Response
///
/// When encryption is performed:
/// ```json
/// {
///     "nonce": "base64_encoded_nonce",
///     "ciphertext": "base64_encoded_encrypted_data"
/// }
/// ```
#[instrument(
    level = "info",
    skip(state, response_body, client_encryption_metadata),
    fields(event = "confidential-compute-encryption-response"),
    err
)]
pub async fn handle_confidential_compute_encryption_response(
    state: &AppState,
    mut response_body: Value,
    client_encryption_metadata: Option<EncryptionMetadata>,
    endpoint: String,
) -> Result<Value, AtomaServiceError> {
    if let Some(EncryptionMetadata {
        client_x25519_public_key,
        salt,
    }) = client_encryption_metadata
    {
        info!(
            target = "atoma-service",
            event = "confidential-compute-encryption-response",
            "Confidential AI inference response, with client x25519 public key: {:#?}",
            client_x25519_public_key
        );

        // Extract signature and response_hash before encryption
        let signature = response_body.get(SIGNATURE_KEY).cloned();
        let response_hash = response_body.get(RESPONSE_HASH_KEY).cloned();

        // Remove signature and response_hash from the body that will be encrypted
        if signature.is_some() {
            response_body
                .as_object_mut()
                .map(|obj| obj.remove(SIGNATURE_KEY));
        }
        if response_hash.is_some() {
            response_body
                .as_object_mut()
                .map(|obj| obj.remove(RESPONSE_HASH_KEY));
        }

        let (sender, receiver) = tokio::sync::oneshot::channel();
        let usage =
            if endpoint == CONFIDENTIAL_IMAGE_GENERATIONS_PATH {
                None
            } else {
                Some(response_body.get(USAGE_KEY).ok_or_else(|| {
                    AtomaServiceError::InvalidBody {
                        message: "Usage not found in response body".to_string(),
                        endpoint: endpoint.clone(),
                    }
                })?)
            };
        state
            .encryption_sender
            .send((
                ConfidentialComputeEncryptionRequest {
                    plaintext: response_body.to_string().as_bytes().to_vec(),
                    salt,
                    client_x25519_public_key,
                },
                sender,
            ))
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Error sending encryption request: {e}"),
                endpoint: endpoint.clone(),
            })?;
        let result = receiver
            .await
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Error receiving encryption response: {e}"),
                endpoint: endpoint.clone(),
            })?;
        match result {
            Ok(ConfidentialComputeEncryptionResponse { ciphertext, nonce }) => {
                let nonce = STANDARD.encode(nonce);
                let ciphertext = STANDARD.encode(ciphertext);
                let mut response_body = json!({
                    NONCE_KEY: nonce,
                    CIPHERTEXT_KEY: ciphertext
                });
                if let Some(response_hash) = response_hash {
                    response_body[RESPONSE_HASH_KEY] = response_hash;
                }
                if let Some(signature) = signature {
                    response_body[SIGNATURE_KEY] = signature;
                }
                if let Some(usage) = usage {
                    response_body[USAGE_KEY] = usage.clone();
                }
                Ok(response_body)
            }
            Err(e) => {
                return Err(AtomaServiceError::InternalError {
                    message: format!("Failed to encrypt confidential compute response: {:?}", e),
                    endpoint: endpoint.clone(),
                })
            }
        }
    } else {
        Ok(response_body)
    }
}

/// Updates the compute units used by a stack in the state manager.
///
/// This function sends an update event to the state manager to record compute unit usage for a stack.
/// It's typically used when handling errors to ensure the estimated compute units are still tracked,
/// even if the actual computation failed.
///
/// # Arguments
///
/// * `state` - Application state containing the state manager channel
/// * `stack_small_id` - Unique identifier for the stack being updated
/// * `estimated_total_compute_units` - The estimated number of compute units that would have been used
/// * `endpoint` - The API endpoint path where the request was received
///
/// # Returns
///
/// Returns `Ok(())` if the update was successful, or an `AtomaServiceError` if the update failed.
///
/// # Errors
///
/// Returns `AtomaServiceError::InternalError` if:
/// - Failed to send the update event to the state manager
/// - Failed to receive confirmation from the state manager
/// - The state manager returned an error during the update
///
/// # Instrumentation
///
/// This function is instrumented with info-level tracing that includes:
/// - stack_small_id
/// - estimated_total_compute_units
/// - total_compute_units (always 0 for error cases)
/// - payload_hash
/// - endpoint
///
/// # Example
///
/// ```rust,ignore
/// use crate::AppState;
///
/// async fn handle_error(state: &AppState, stack_id: i64) -> Result<(), AtomaServiceError> {
///     // When an error occurs, update the stack with estimated units
///     update_stack_num_compute_units(
///         state,
///         stack_id,
///         100, // estimated units
///         "/v1/chat/completions"
///     ).await?;
///     Ok(())
/// }
/// ```
#[instrument(
    level = "info",
    skip_all,
    fields(
        stack_small_id,
        estimated_total_compute_units,
        total_compute_units,
        payload_hash,
        endpoint
    ),
    err
)]
pub fn update_stack_num_compute_units(
    state_manager_sender: &Sender<AtomaAtomaStateManagerEvent>,
    stack_small_id: i64,
    estimated_total_compute_units: i64,
    total_compute_units: i64,
    endpoint: &str,
    concurrent_requests: &Arc<DashMap<i64, u64>>,
) -> Result<(), AtomaServiceError> {
    state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateStackNumComputeUnits {
            stack_small_id,
            total_compute_units,
            estimated_total_compute_units,
            concurrent_requests: Arc::clone(concurrent_requests),
        })
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error sending update stack num compute units event: {e}"),
            endpoint: endpoint.to_string(),
        })
}

/// Atomically decrements the concurrent request count for a specific stack ID in a `DashMap`.
///
/// If the count reaches zero after decrementing, the entry for the stack ID is removed from the map.
/// This function uses the `DashMap::entry` API to ensure atomicity of the check-decrement-update/remove
/// operation for a given key.
///
/// # Arguments
///
/// * `concurrent_requests_per_stack` - A reference to the `DashMap` storing `stack_small_id` -> `count`.
/// * `stack_small_id` - The identifier of the stack whose concurrent request count should be decremented.
/// * `endpoint` - A string representing the API endpoint, used for logging context.
///
/// # Returns
/// Returns the new concurrent request count after decrement
///
/// Returns the concurrent request count for the `stack_small_id` *after* the decrement operation.
/// If the entry did not exist, or if the count was already zero, it returns 0. If the count reached zero
/// due to this decrement, it returns 0 after removing the entry.
///
/// # Concurrency
///
/// The `DashMap::entry` method locks the relevant map shard, ensuring that the read, potential decrement,
/// update, and potential removal operations occur atomically for the given `stack_small_id`.
///
/// # Logging
///
/// - Logs an error if an attempt is made to decrement the count for a `stack_small_id` that is not
///   present in the map (implying a count of 0).
/// - Logs an error if an attempt is made to decrement a count that is already 0.
/// - Logs informational message when a count reaches 0 and the corresponding entry is removed.
#[instrument(level = "info", skip_all, fields(stack_small_id, endpoint))]
pub fn handle_concurrent_requests_count_decrement(
    concurrent_requests_per_stack: &DashMap<i64, u64>,
    stack_small_id: i64,
    endpoint: &str,
) {
    let entry = concurrent_requests_per_stack.entry(stack_small_id);
    match entry {
        dashmap::mapref::entry::Entry::Occupied(mut occupied_entry) => {
            let current_count = *occupied_entry.get();
            if current_count == 0 {
                tracing::error!(
                    target = "atoma-service",
                    level = "error",
                    endpoint = endpoint,
                );
                occupied_entry.remove();
            } else {
                let new_count = current_count.saturating_sub(1);
                occupied_entry.insert(new_count);
                if new_count == 0 {
                    tracing::info!(
                        target = "atoma-service",
                        level = "info",
                        endpoint = endpoint,
                        stack_small_id,
                        "Concurrent requests count reached 0 for stack, removing entry and updating stack num compute units.",
                    );
                    occupied_entry.remove();
                }
            }
        }
        dashmap::mapref::entry::Entry::Vacant(_) => {
            // Entry doesn't exist. This implies a count of 0.
            tracing::error!(
                target = "atoma-service",
                level = "error",
                endpoint = endpoint,
                stack_small_id,
                "Attempted to decrement concurrent requests for non-existent stack entry (implies count is 0).",
            );
        }
    }
}

/// Handles the status code returned by the inference service.
///
/// This function maps the status code to an appropriate error variant.
///
/// # Arguments
///
/// * `status_code` - The status code returned by the inference service
/// * `endpoint` - The API endpoint path where the request was received
/// * `error` - The error message returned by the inference service
///
/// # Returns
///
/// Returns an `AtomaServiceError` variant based on the status code.
#[instrument(level = "info", skip_all, fields(endpoint), err)]
pub fn handle_status_code_error(
    status_code: StatusCode,
    endpoint: &str,
    error: &str,
) -> Result<(), AtomaServiceError> {
    match status_code {
        StatusCode::UNAUTHORIZED => Err(AtomaServiceError::AuthError {
            auth_error: format!("Unauthorized response from inference service: {error}"),
            endpoint: endpoint.to_string(),
        }),
        StatusCode::INTERNAL_SERVER_ERROR => Err(AtomaServiceError::InternalError {
            message: format!("Inference service returned internal server error: {error}"),
            endpoint: endpoint.to_string(),
        }),
        StatusCode::BAD_REQUEST => Err(AtomaServiceError::InvalidBody {
            message: format!("Inference service returned bad request error: {error}"),
            endpoint: endpoint.to_string(),
        }),
        _ => Err(AtomaServiceError::InternalError {
            message: format!("Inference service returned non-success error: {error}"),
            endpoint: endpoint.to_string(),
        }),
    }
}

mod vllm_metrics {
    use futures::{stream::FuturesUnordered, StreamExt};
    use hyper::StatusCode;
    use once_cell::sync::Lazy;
    use prometheus_http_query::Client;
    use tracing::{info, instrument};

    pub type Result<T> = std::result::Result<T, VllmMetricsError>;

    /// The timeout for the Prometheus metrics queries
    const METRICS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    /// Prometheus url for the atoma-node instance
    const PROMETHEUS_URL: &str = "http://prometheus:9090";

    /// The HTTP client for the Prometheus metrics queries
    static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
        Client::from(
            reqwest::Client::builder()
                .timeout(METRICS_TIMEOUT)
                .build()
                .expect("Failed to create HTTP client"),
            PROMETHEUS_URL,
        )
        .expect("Failed to create HTTP client")
    });

    /// Retrieves metrics data from a chat completions service endpoint.
    ///
    /// This function sends a GET request to the specified endpoint with the provided query parameters,
    /// and returns the metrics data as a `MetricsDataAndUrl` struct.
    ///
    /// # Arguments
    ///
    /// * `client` - The HTTP client to use for the request
    /// * `query` - The Prometheus query to execute
    /// * `endpoint` - The URL of the chat completions service endpoint
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the metrics data as a `MetricsDataAndUrl` struct.
    ///
    /// # Errors
    ///
    /// Returns a `VllmMetricsError` if the request fails or the response is not valid.
    #[instrument(level = "info", skip_all, fields(job=job))]
    async fn get_metrics(
        client: &Client,
        job: &str,
        chat_completions_service_url: &str,
    ) -> Result<(String, f64)> {
        info!(
            target = "atoma-service",
            module = "vllm_metrics",
            level = "info",
            "Getting metrics for job: {job}"
        );
        let query = format!(
            "histogram_quantile(0.90, sum(rate(vllm:request_queue_time_seconds_bucket{{job=\"{job}\"}}[30s])) by (le))"
        );
        let response = client.query(&query).get().await?;
        response.data().as_vector().map_or_else(
            || Err(VllmMetricsError::NoMetricsFound(job.to_string())),
            |data_vector| {
                data_vector.first().map_or_else(
                    || Err(VllmMetricsError::NoMetricsFound(job.to_string())),
                    |value| {
                        let sample = value.sample();
                        let value = sample.value();
                        Ok((chat_completions_service_url.to_string(), value))
                    },
                )
            },
        )
    }

    /// Retrieves the best available chat completions service URL for a given model.
    ///
    /// This function iterates through a list of chat completions service URLs,
    /// executes a Prometheus query to retrieve request queue time metrics,
    /// and selects the URL with the lowest request queue time.
    ///
    /// # Arguments
    ///
    /// * `chat_completions_service_urls` - A list of chat completions service URLs together with respective job names to query
    /// * `model` - The name of the model to query
    ///
    /// # Returns
    ///
    /// Returns the URL of the chat completions service with the lowest request queue time.
    ///
    /// # Errors
    ///
    /// Returns a `VllmMetricsError` if the list of chat completions service URLs is empty.
    #[instrument(level = "info", skip_all, fields(model=model, chat_completions_service_urls=chat_completions_service_urls.len()))]
    pub async fn get_best_available_chat_completions_service_url(
        chat_completions_service_urls: &[(String, String)],
        model: &str,
    ) -> Result<(String, StatusCode)> {
        const MAX_ALLOWED_REQUEST_QUEUE_TIME_SECONDS: f64 = 4.0; // Default to 4 seconds
        if chat_completions_service_urls.is_empty() {
            return Err(VllmMetricsError::NoChatCompletionsServiceUrlsFound(
                model.to_string(),
            ));
        }
        let mut min_request_queue_time_seconds = MAX_ALLOWED_REQUEST_QUEUE_TIME_SECONDS;
        let mut best_url = chat_completions_service_urls[0].0.clone();
        let mut futures: FuturesUnordered<_> = chat_completions_service_urls
            .iter()
            .map(|(chat_completions_service_url, job_name)| {
                get_metrics(&HTTP_CLIENT, job_name, chat_completions_service_url)
            })
            .collect();
        while let Some(request_queue_time_seconds) = futures.next().await {
            let (chat_completions_service_url, request_queue_time_seconds) =
                match request_queue_time_seconds {
                    Ok((chat_completions_service_url, request_queue_time_seconds)) => {
                        info!(
                            target = "atoma-service",
                            module = "vllm_metrics",
                            level = "info",
                            "Received vLLM request queue time metrics response for {chat_completions_service_url}: {request_queue_time_seconds}"
                        );
                        (chat_completions_service_url, request_queue_time_seconds)
                    }
                    Err(e) => {
                        tracing::error!(
                            target = "atoma-service",
                            module = "vllm_metrics",
                            level = "error",
                            "Failed to get metrics for chat completions service url with error: {e}",
                        );
                        continue;
                    }
                };
            if request_queue_time_seconds < min_request_queue_time_seconds
                || request_queue_time_seconds.is_nan()
            {
                min_request_queue_time_seconds = if request_queue_time_seconds.is_nan() {
                    0.0
                } else {
                    request_queue_time_seconds
                };
                best_url.clone_from(&chat_completions_service_url);
            }
        }
        tracing::info!(
            target = "atoma-service",
            level = "info",
            "Best available chat completions service URL for model: {model} is: {best_url}",
            model = model,
            best_url = best_url
        );
        if min_request_queue_time_seconds > MAX_ALLOWED_REQUEST_QUEUE_TIME_SECONDS {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "Best available chat completions service URL for model: {model} has a request queue time of at least {min_request_queue_time_seconds} seconds",
            );
            return Ok((best_url, StatusCode::TOO_MANY_REQUESTS));
        }
        Ok((best_url, StatusCode::OK))
    }

    #[derive(Debug, thiserror::Error)]
    pub enum VllmMetricsError {
        #[error("Failed to get metrics: {0}")]
        GetMetricsError(#[from] reqwest::Error),
        #[error("No chat completions service urls found for model: {0}")]
        NoChatCompletionsServiceUrlsFound(String),
        #[error("Invalid metrics value: {0}")]
        InvalidMetricsValue(#[from] std::num::ParseFloatError),
        #[error("Invalid metrics response: {0}")]
        InvalidMetricsResponse(#[from] serde_json::Error),
        #[error("Failed to create HTTP client: {0}")]
        FailedToCreateHttpClient(#[from] prometheus_http_query::Error),
        #[error("No metrics found for job: {0}")]
        NoMetricsFound(String),
    }
}
