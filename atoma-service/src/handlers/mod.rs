#![allow(clippy::duplicate_mod)]
pub mod chat_completions;
pub mod completions;
pub mod embeddings;
pub mod image_generations;
pub mod metrics;
pub mod request_model;
pub mod stop_streamer;

use atoma_confidential::types::{
    ConfidentialComputeEncryptionRequest, ConfidentialComputeEncryptionResponse,
};
use atoma_p2p::constants::ONE_MILLION;
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

/// Key for the prompt tokens in the usage in the response body
pub const PROMPT_TOKENS_KEY: &str = "prompt_tokens";

/// Key for the completion tokens in the usage in the response body
pub const COMPLETION_TOKENS_KEY: &str = "completion_tokens";

#[derive(Debug, Clone)]
pub enum InferenceService {
    Vllm,
    SgLang,
}

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
pub async fn sign_response_and_update_stack_hash(
    response_body: &mut Value,
    payload_hash: [u8; 32],
    state: &AppState,
    stack_small_id: Option<i64>,
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
/// * `estimated_total_tokens` - The estimated number of tokens that would have been used
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
/// - estimated_total_tokens
/// - total_tokens (always 0 for error cases)
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
        estimated_total_tokens,
        total_tokens,
        payload_hash,
        endpoint
    ),
    err
)]
pub fn update_stack_num_compute_units(
    state_manager_sender: &Sender<AtomaAtomaStateManagerEvent>,
    stack_small_id: i64,
    estimated_total_tokens: i64,
    total_tokens: i64,
    endpoint: &str,
    concurrent_requests: u64,
) -> Result<(), AtomaServiceError> {
    state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateStackNumTokens {
            stack_small_id,
            total_tokens,
            estimated_total_tokens,
            concurrent_requests,
        })
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error sending update stack num compute units event: {e}"),
            endpoint: endpoint.to_string(),
        })
}

/// Updates the fiat amount for a user in the state manager.
///
/// This function sends an update event to the state manager to record the fiat amount
/// associated with a user's request. It calculates the estimated total amount based on
/// the estimated compute units and the price per million compute units.
///
/// # Arguments
///
/// * `state_manager_sender` - The channel to send events to the state manager
/// * `user_address` - The address of the user for whom the fiat amount is being updated
/// * `estimated_total_tokens` - The estimated number of tokens for the request
/// * `total_amount` - The total amount in fiat currency
/// * `price_per_one_million_compute_units` - The price per million compute units
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
/// # Example
///
/// ```rust,ignore
/// use crate::AppState;
///
/// async fn handle_request(state: &AppState, user_address: String) -> Result<(), AtomaServiceError> {
///     // When a request is made, update the fiat amount for the user
///     update_fiat_amount(
///         &state.state_manager_sender,
///         user_address,
///         100, // estimated units
///         10, // total amount
///         0.01, // price per million compute units
///         "/v1/chat/completions"
///     ).await?;
///     Ok(())
/// }
/// ```
#[instrument(
    level = "info",
    skip_all,
    fields(
        user_address,
        estimated_total_tokens,
        total_amount,
        price_per_one_million_compute_units,
        endpoint
    ),
    err
)]
#[allow(clippy::too_many_arguments)]
pub fn update_fiat_amount(
    state_manager_sender: &Sender<AtomaAtomaStateManagerEvent>,
    user_address: String,
    estimated_input_tokens: i64,
    input_tokens: i64,
    estimated_output_tokens: i64,
    output_tokens: i64,
    price_per_one_million_compute_units: i64,
    endpoint: &str,
) -> Result<(), AtomaServiceError> {
    let estimated_input_amount = (estimated_input_tokens as u128
        * price_per_one_million_compute_units as u128
        / ONE_MILLION) as i64;
    let input_amount =
        (input_tokens as u128 * price_per_one_million_compute_units as u128 / ONE_MILLION) as i64;
    let estimated_output_amount = (estimated_output_tokens as u128
        * price_per_one_million_compute_units as u128
        / ONE_MILLION) as i64;
    let output_amount =
        (output_tokens as u128 * price_per_one_million_compute_units as u128 / ONE_MILLION) as i64;
    state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateFiatAmount {
            user_address,
            estimated_input_amount,
            input_amount,
            estimated_output_amount,
            output_amount,
        })
        .map_err(|e| AtomaServiceError::InternalError {
            message: format!("Error sending update fiat amount event: {e}"),
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
pub fn handle_concurrent_requests_count_decrement(
    concurrent_requests_per_stack: &DashMap<i64, u64>,
    stack_small_id: i64,
    endpoint: &str,
) -> u64 {
    let entry = concurrent_requests_per_stack.entry(stack_small_id);
    match entry {
        dashmap::mapref::entry::Entry::Occupied(mut occupied_entry) => {
            let current_count = *occupied_entry.get();
            if current_count == 0 {
                tracing::error!(
                    target = "atoma-service",
                    level = "error",
                    endpoint = endpoint,
                    stack_small_id,
                    "Attempted to decrement concurrent requests for non-existent stack entry (implies count is 0).",
                );
                occupied_entry.remove();
                0
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
                new_count
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
            0
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

pub mod inference_service_metrics {
    use futures::future::join_all;
    use opentelemetry::KeyValue;
    use prometheus_parse::Scrape;
    use prometheus_parse::Value;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::time::Duration;
    use tokio::sync::RwLock;
    use tokio::time;

    use crate::handlers::metrics::CHAT_COMPLETIONS_TOO_MANY_REQUESTS;
    use hyper::StatusCode;
    use tracing::{info, instrument};

    use super::InferenceService;

    pub type Result<T> = std::result::Result<T, ChatCompletionsMetricsError>;
    type MetricValue = ChatCompletionsMetrics;
    type MetricResult = Result<MetricValue>;
    type MetricsVec = Vec<MetricResult>;
    type CachedMetrics = Option<MetricsVec>;
    type MetricsLock = Arc<RwLock<CachedMetrics>>;

    /// The default interval for updating the metrics
    const DEFAULT_METRICS_UPDATE_INTERVAL_MILLIS: u64 = 1_000;

    /// Chat completions metrics
    #[derive(Debug, Clone)]
    struct ChatCompletionsMetrics {
        /// The model name  
        model: String,
        /// The chat completions service url
        chat_completions_service_url: String,
        /// The number of queue requests
        num_queued_requests: f64,
        /// The number of running requests
        num_running_requests: f64,
    }

    /// Cache structure to store metrics
    #[derive(Debug, Default)]
    struct MetricsCache {
        metrics: MetricsLock,
    }

    impl MetricsCache {
        fn new() -> Self {
            Self {
                metrics: Arc::new(RwLock::new(None)),
            }
        }

        async fn get_metrics(&self) -> Option<MetricsVec> {
            self.metrics.read().await.clone()
        }

        async fn update_metrics(&self, new_metrics: Vec<Result<ChatCompletionsMetrics>>) {
            *self.metrics.write().await = Some(new_metrics);
        }
    }

    /// Global metrics cache
    #[allow(clippy::redundant_closure)]
    static VLLM_METRICS_CACHE: LazyLock<MetricsCache> = LazyLock::new(|| MetricsCache::new());

    /// Global metrics cache
    #[allow(clippy::redundant_closure)]
    static SGLANG_METRICS_CACHE: LazyLock<MetricsCache> = LazyLock::new(|| MetricsCache::new());

    /// Start the background task to update metrics every 500 milliseconds
    ///
    /// # Arguments
    ///
    /// * `chat_completions_service_urls` - A vector of tuples containing the chat completions service URL and the job name.
    /// * `metrics_update_interval` - The interval in seconds to update the metrics.
    #[instrument(level = "info", skip_all)]
    pub fn start_metrics_updater(
        chat_completions_service_urls: Vec<(String, String, String)>,
        metrics_update_interval: Option<u64>,
    ) {
        type ChatCompletionsServiceUrls = Vec<(String, String, String)>;
        info!(
            target = "atoma-service",
            module = "inference_service_metrics",
            level = "info",
            "Starting metrics updater with {chat_completions_service_urls:?}"
        );
        let (vllm_chat_completions_service_urls, sglang_chat_completions_service_urls): (
            ChatCompletionsServiceUrls,
            ChatCompletionsServiceUrls,
        ) = chat_completions_service_urls
            .iter()
            .cloned()
            .partition(|(_, _, job)| job.contains("vllm"));
        info!(
            target = "atoma-service",
            module = "inference_service_metrics",
            level = "info",
            "Partitioned chat completions service urls: vllm: {vllm_chat_completions_service_urls:?}, sglang: {sglang_chat_completions_service_urls:?}"
        );
        let vllm_chat_completions_service_urls = Arc::new(vllm_chat_completions_service_urls);
        let sglang_chat_completions_service_urls = Arc::new(sglang_chat_completions_service_urls);
        tokio::spawn(async move {
            let metrics_interval =
                metrics_update_interval.unwrap_or(DEFAULT_METRICS_UPDATE_INTERVAL_MILLIS);
            info!(
                target = "atoma-service",
                module = "inference_service_metrics",
                level = "info",
                "Metrics update interval: {metrics_interval} seconds"
            );
            let mut interval = time::interval(Duration::from_millis(metrics_interval));
            loop {
                interval.tick().await;
                if !vllm_chat_completions_service_urls.is_empty() {
                    let vllm_metrics =
                        get_metrics(InferenceService::Vllm, &vllm_chat_completions_service_urls)
                            .await;
                    if vllm_metrics.iter().any(std::result::Result::is_ok) {
                        VLLM_METRICS_CACHE.update_metrics(vllm_metrics).await;
                    } else {
                        tracing::warn!(
                            "Failed to retrieve any valid vLLM metrics, not updating cache"
                        );
                    }
                }
                if !sglang_chat_completions_service_urls.is_empty() {
                    let sglang_metrics = get_metrics(
                        InferenceService::SgLang,
                        &sglang_chat_completions_service_urls,
                    )
                    .await;
                    if sglang_metrics.iter().any(std::result::Result::is_ok) {
                        SGLANG_METRICS_CACHE.update_metrics(sglang_metrics).await;
                    } else {
                        tracing::warn!(
                            "Failed to retrieve any valid SgLang metrics, not updating cache"
                        );
                    }
                }
            }
        });
    }

    /// Fetches metrics from the specified chat completions service URL.
    ///
    /// This function retrieves metrics from the specified chat completions service URL
    /// and parses the response to extract relevant metrics such as the number of queue
    /// requests and running requests. It handles errors gracefully and returns a vector
    /// of results, where each result contains the metrics for a specific service URL.
    ///
    /// # Arguments
    ///
    /// * `inference_service` - The inference service type (vLLM or SgLang).
    /// * `jobs_with_url` - A slice of tuples containing the chat completions service URL
    ///   and the job name (e.g., "vllm-service", "sglang-service").
    ///
    /// # Returns
    ///
    /// Returns a `Vec<Result<ChatCompletionsMetrics>>`, where each result contains
    /// the metrics for a specific service URL. If an error occurs while fetching or parsing
    /// the metrics, the error is returned in the result.
    ///
    /// # Errors
    ///
    /// *   `ChatCompletionsMetricsError::NoMetricsFound`: If no metrics are found for the
    ///     specified job or if the metrics response is invalid.
    /// *   Other variants of `ChatCompletionsMetricsError` may be returned if underlying
    ///     issues occur during metric collection from Prometheus (e.g., network errors,
    ///     parsing errors), though the function attempts to handle missing individual metrics
    ///     gracefully.
    async fn get_metrics(
        inference_service: InferenceService,
        jobs_with_url: &[(String, String, String)], // (model, url, job)
    ) -> Vec<Result<ChatCompletionsMetrics>> {
        let (num_queue_requests_metric, num_running_requests_metric, service_prefix) =
            match inference_service {
                InferenceService::SgLang => ("num_queue_reqs", "num_running_reqs", "sglang:"),
                InferenceService::Vllm => ("num_requests_waiting", "num_requests_running", "vllm:"),
            };

        let tasks =
            jobs_with_url
                .iter()
                .map(|(model, chat_completions_service_url, job)| async move {
                    let response = reqwest::get(format!("{chat_completions_service_url}/metrics"))
                        .await
                        .map_err(|_| {
                            ChatCompletionsMetricsError::NoMetricsFound(job.to_string())
                        })?;
                    let body = response.text().await?;
                    let lines = body
                        .lines()
                        .map(|line| Ok(line.replace(service_prefix, "")));
                    let metrics = Scrape::parse(lines).unwrap();
                    let get_metric = |name: &str| -> Result<f64> {
                        metrics
                            .samples
                            .iter()
                            .find(|s| s.metric == name)
                            .ok_or_else(|| ChatCompletionsMetricsError::NoMetricsFound(job.clone()))
                            .and_then(|sample| {
                                if let Value::Gauge(value) = sample.value {
                                    Ok(value)
                                } else {
                                    Err(ChatCompletionsMetricsError::NoMetricsFound(job.clone()))
                                }
                            })
                    };
                    let num_queued_requests = get_metric(num_queue_requests_metric)?;
                    let num_running_requests = get_metric(num_running_requests_metric)?;

                    Ok(ChatCompletionsMetrics {
                        model: model.clone(),
                        chat_completions_service_url: chat_completions_service_url.clone(),
                        num_queued_requests,
                        num_running_requests,
                    })
                });
        join_all(tasks).await
    }

    /// Selects the best available chat completions service URL for a given model based on performance metrics.
    ///
    /// This function aims to distribute load and ensure optimal response times by choosing
    /// the service instance that is currently performing best. The selection process prioritizes
    /// services with lower requests running and queue lengths.
    ///
    /// # Metrics and Selection Logic:
    ///
    /// 1.  **Metrics Source**: Metrics for each service (vLLM or SgLang) are retrieved directly from the inference
    ///     service URL.
    ///
    /// 2.  **Priority of Metrics for "Best" Service Selection**:
    ///     *   **No Load**: If a service has zero running requests (`num_running_requests` is 0.0),
    ///         it's considered the best.
    ///     *   **Number of Queued Requests**: If number of running requests are equivalent, the service
    ///         with the fewest `num_queue_requests` is selected.
    ///
    /// 3.  **Handling Missing or Invalid Metrics**:
    ///     *   If, after checking all services, no valid metrics are found for the specified `model`,
    ///         a service URL is chosen randomly from the initial list.
    ///
    /// # Load Thresholds and Behavior:
    ///
    /// The function defines several thresholds to manage high load scenarios:
    /// *   `MAX_ALLOWED_NUM_QUEUED_REQUESTS` (1.0)
    ///
    /// If the determined "best" service (or all services) exceeds these
    /// thresholds, the function returns the first URL from the input `chat_completions_service_urls`
    /// list along with a `StatusCode::TOO_MANY_REQUESTS`. The `CHAT_COMPLETIONS_TOO_MANY_REQUESTS`
    /// metric counter is also incremented for the model.
    ///
    /// # Arguments
    ///
    /// * `chat_completions_service_urls`: A slice of tuples `(String, String)`, where each tuple
    ///   represents a service. The first `String` is the service URL, and the second `String`
    ///   is the job name (e.g., "vllm-service", "sglang-service"), used to determine the
    ///   metrics querying strategy.
    /// * `model`: A string slice representing the name of the model for which the best service
    ///   URL is being requested. The comparison is case-insensitive.
    ///
    /// # Returns
    ///
    /// Returns a `Result<(String, StatusCode), ChatCompletionsMetricsError>`:
    /// *   `Ok((String, StatusCode::OK))`: On success, containing the URL of the best available
    ///     service and an OK status.
    /// *   `Ok((String, StatusCode::TOO_MANY_REQUESTS))`: If the system is determined to be under
    ///     high load based on the metrics thresholds. The returned `String` will be the first URL
    ///     from the `chat_completions_service_urls` input.
    /// *   `Err(ChatCompletionsMetricsError)`: If an error occurs, such as no service URLs
    ///     being provided or issues during metrics fetching that are not handled by fallback mechanisms.
    ///
    /// # Errors
    ///
    /// *   `ChatCompletionsMetricsError::NoChatCompletionsServiceUrlsFound`: If the input
    ///     `chat_completions_service_urls` slice is empty.
    /// *   Other variants of `ChatCompletionsMetricsError` may be returned if underlying issues
    ///     occur during metric collection from Prometheus (e.g., network errors, parsing errors),
    ///     though the function attempts to handle missing individual metrics gracefully.g
    #[instrument(level = "info", skip_all, fields(model=model))]
    #[allow(clippy::float_cmp)]
    pub async fn get_best_available_chat_completions_service_url(
        chat_completions_service_urls: &[(String, String)],
        model: &str,
    ) -> Result<(String, StatusCode)> {
        const MAX_ALLOWED_NUM_QUEUED_REQUESTS: f64 = 1.0; // Default to 1 request

        type ChatCompletionsServiceUrls = Vec<(String, String)>;

        if chat_completions_service_urls.is_empty() {
            return Err(
                ChatCompletionsMetricsError::NoChatCompletionsServiceUrlsFound(model.to_string()),
            );
        }
        tracing::debug!(
            target = "atoma-service",
            module = "inference_service_metrics",
            level = "info",
            "Getting best available chat completions service URL for model: {model} and urls: {chat_completions_service_urls:?}"
        );
        let (vllm_chat_completions_service_urls, sglang_chat_completions_service_urls): (
            ChatCompletionsServiceUrls,
            ChatCompletionsServiceUrls,
        ) = chat_completions_service_urls
            .iter()
            .cloned()
            .partition(|(_, job)| job.contains("vllm"));

        tracing::debug!(
            target = "atoma-service",
            module = "inference_service_metrics",
            level = "info",
            "Partitioned chat completions service urls: vllm: {vllm_chat_completions_service_urls:?}, sglang: {sglang_chat_completions_service_urls:?}"
        );

        // Get cached metrics
        let vllm_metrics = if vllm_chat_completions_service_urls.is_empty() {
            vec![]
        } else if let Some(metrics) = VLLM_METRICS_CACHE.get_metrics().await {
            metrics
        } else {
            info!(
                target = "atoma-service",
                module = "inference_service_metrics",
                level = "info",
                "No cached vLLM metrics, getting them directly"
            );
            let vllm_chat_completions_service_urls_with_model: Vec<(String, String, String)> =
                vllm_chat_completions_service_urls
                    .iter()
                    .map(|(url, job)| (model.to_string(), url.clone(), job.clone()))
                    .collect();
            get_metrics(
                InferenceService::Vllm,
                &vllm_chat_completions_service_urls_with_model,
            )
            .await
        };
        let sglang_metrics = if sglang_chat_completions_service_urls.is_empty() {
            vec![]
        } else if let Some(metrics) = SGLANG_METRICS_CACHE.get_metrics().await {
            metrics
        } else {
            info!(
                target = "atoma-service",
                module = "inference_service_metrics",
                level = "info",
                "No cached SgLang metrics, getting them directly"
            );
            let sglang_chat_completions_service_urls_with_model: Vec<(String, String, String)> =
                sglang_chat_completions_service_urls
                    .iter()
                    .map(|(url, job)| (model.to_string(), url.clone(), job.clone()))
                    .collect();
            get_metrics(
                InferenceService::SgLang,
                &sglang_chat_completions_service_urls_with_model,
            )
            .await
        };

        tracing::debug!(
            target = "atoma-service",
            module = "inference_service_metrics",
            level = "info",
            "Received vLLM metrics: {vllm_metrics:?}, SgLang metrics: {sglang_metrics:?}"
        );

        let mut metrics_results = Vec::new();
        for metric in vllm_metrics.into_iter().chain(sglang_metrics.into_iter()) {
            match metric {
                Ok(ChatCompletionsMetrics {
                    model: current_model,
                    chat_completions_service_url,
                    num_queued_requests,
                    num_running_requests,
                }) => {
                    tracing::info!(
                        target = "atoma-service",
                        module = "inference_service_metrics",
                        level = "info",
                        "current_model = {current_model}, model = {model}, they are equal = {}",
                        current_model == model
                    );
                    if current_model.to_lowercase() != model.to_lowercase() {
                        // NOTE: We only want to consider metrics for the current model
                        continue;
                    }
                    info!(
                        target = "atoma-service",
                        module = "vllm_metrics",
                        level = "info",
                        "Received vLLM/SgLang metrics response for {chat_completions_service_url}:\n
                            num_queued_requests={num_queued_requests},
                            num_running_requests={num_running_requests}"
                    );
                    metrics_results.push(ChatCompletionsMetrics {
                        model: current_model,
                        chat_completions_service_url,
                        num_queued_requests,
                        num_running_requests,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        target = "atoma-service",
                        module = "vllm_metrics",
                        level = "error",
                        "Failed to get metrics for chat completions service url with error: {e}",
                    );
                }
            }
        }

        if metrics_results.is_empty() {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "No metrics found for model: {model}",
            );
            // NOTE: In this case, we pick one of the urls at random
            let random_index = rand::thread_rng().gen_range(0..chat_completions_service_urls.len());
            let best_url = chat_completions_service_urls[random_index].0.clone();
            return Ok((best_url, StatusCode::OK));
        }

        let mut rng = rand::thread_rng();
        // If there are several minimum metrics, shuffle them to get a random one, the min_by selects the first one
        metrics_results.shuffle(&mut rng);

        // Select the best available chat completions service URL based on the number of queued and running requests.
        // If some service has 0 running requests it can still have a lot of queued requests, so we need to check for that
        // and select the one with the least number of queued requests.
        let best_metrics = metrics_results
            .iter()
            .min_by_key(|metric| (metric.num_queued_requests + metric.num_running_requests) as i64)
            .unwrap();

        if best_metrics.num_queued_requests > MAX_ALLOWED_NUM_QUEUED_REQUESTS {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "Node is currently under high load, the best available chat completions service URL for model: {model} has a num queue requests of at least {} requests",
                best_metrics.num_queued_requests
            );
            CHAT_COMPLETIONS_TOO_MANY_REQUESTS.add(1, &[KeyValue::new("model", model.to_string())]);
            return Ok((
                chat_completions_service_urls[0].0.clone(),
                StatusCode::TOO_MANY_REQUESTS,
            ));
        }

        let best_url = best_metrics.chat_completions_service_url.clone();
        tracing::info!(
            target = "atoma-service",
            level = "info",
            "Best available chat completions service URL for model: {model} is: {best_url} with and {} queue requests",
            best_metrics.num_queued_requests
        );

        Ok((best_url, StatusCode::OK))
    }

    #[derive(Debug, thiserror::Error, Clone)]
    pub enum ChatCompletionsMetricsError {
        #[error("Failed to get metrics: {0}")]
        GetMetricsError(String),
        #[error("No chat completions service urls found for model: {0}")]
        NoChatCompletionsServiceUrlsFound(String),
        #[error("Invalid metrics value: {0}")]
        InvalidMetricsValue(String),
        #[error("Invalid metrics response: {0}")]
        InvalidMetricsResponse(String),
        #[error("Failed to create HTTP client: {0}")]
        FailedToCreateHttpClient(String),
        #[error("No metrics found for job: {0}")]
        NoMetricsFound(String),
        #[error("Failed to acquire semaphore permit")]
        AcquireError,
    }

    // From implementations to handle conversions from error types to our cloneable error type
    impl From<reqwest::Error> for ChatCompletionsMetricsError {
        fn from(err: reqwest::Error) -> Self {
            Self::GetMetricsError(err.to_string())
        }
    }

    impl From<std::num::ParseFloatError> for ChatCompletionsMetricsError {
        fn from(err: std::num::ParseFloatError) -> Self {
            Self::InvalidMetricsValue(err.to_string())
        }
    }

    impl From<serde_json::Error> for ChatCompletionsMetricsError {
        fn from(err: serde_json::Error) -> Self {
            Self::InvalidMetricsResponse(err.to_string())
        }
    }

    impl From<prometheus_http_query::Error> for ChatCompletionsMetricsError {
        fn from(err: prometheus_http_query::Error) -> Self {
            Self::FailedToCreateHttpClient(err.to_string())
        }
    }
}
