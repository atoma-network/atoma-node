#![allow(clippy::duplicate_mod)]
pub mod chat_completions;
pub mod completions;
pub mod embeddings;
pub mod image_generations;
pub mod metrics;
pub mod request_batcher;
pub mod request_counter;
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

const VLLM_RUNNING_REQUESTS_QUERY: &str = "num_requests_running";
const VLLM_QUEUED_REQUESTS_QUERY: &str = "num_requests_waiting";
const VLLM_MEMORY_USAGE_QUERY: &str = "gpu_cache_usage_perc";
const VLLM_SERVICE_PREFIX: &str = "vllm:";
const SGLANG_RUNNING_REQUESTS_QUERY: &str = "num_running_reqs";
const SGLANG_QUEUED_REQUESTS_QUERY: &str = "num_queue_reqs";
const SGLANG_MEMORY_USAGE_QUERY: &str = "token_usage";
const SGLANG_SERVICE_PREFIX: &str = "sglang:";

#[derive(Debug, Clone)]
pub enum InferenceService {
    Vllm,
    SgLang,
}

impl InferenceService {
    #[must_use]
    pub const fn get_queued_requests_metric_name(&self) -> &'static str {
        match self {
            Self::Vllm => VLLM_QUEUED_REQUESTS_QUERY,
            Self::SgLang => SGLANG_QUEUED_REQUESTS_QUERY,
        }
    }

    #[must_use]
    pub const fn get_running_requests_metric_name(&self) -> &'static str {
        match self {
            Self::Vllm => VLLM_RUNNING_REQUESTS_QUERY,
            Self::SgLang => SGLANG_RUNNING_REQUESTS_QUERY,
        }
    }

    #[must_use]
    pub const fn get_usage(&self) -> &'static str {
        match self {
            Self::Vllm => VLLM_MEMORY_USAGE_QUERY,
            Self::SgLang => SGLANG_MEMORY_USAGE_QUERY,
        }
    }

    #[must_use]
    pub const fn get_service_prefix(&self) -> &'static str {
        match self {
            Self::Vllm => VLLM_SERVICE_PREFIX,
            Self::SgLang => SGLANG_SERVICE_PREFIX,
        }
    }
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
/// * `user_id` - The id of the user for whom the fiat amount is being updated
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
    user_id: Option<i64>,
    user_address: String,
    model_name: String,
    estimated_input_tokens: i64,
    input_tokens: i64,
    estimated_output_tokens: i64,
    output_tokens: i64,
    price_per_one_million_compute_units: i64,
    endpoint: &str,
) -> Result<(), AtomaServiceError> {
    if let Some(user_id) = user_id {
        let estimated_input_amount = (estimated_input_tokens as u128
            * price_per_one_million_compute_units as u128
            / ONE_MILLION) as i64;
        let input_amount = (input_tokens as u128 * price_per_one_million_compute_units as u128
            / ONE_MILLION) as i64;
        let estimated_output_amount = (estimated_output_tokens as u128
            * price_per_one_million_compute_units as u128
            / ONE_MILLION) as i64;
        let output_amount = (output_tokens as u128 * price_per_one_million_compute_units as u128
            / ONE_MILLION) as i64;
        state_manager_sender
            .send(AtomaAtomaStateManagerEvent::UpdateFiatAmount {
                user_id,
                user_address,
                model_name,
                estimated_input_amount,
                input_amount,
                input_tokens,
                estimated_output_amount,
                output_amount,
                output_tokens,
            })
            .map_err(|e| AtomaServiceError::InternalError {
                message: format!("Error sending update fiat amount event: {e}"),
                endpoint: endpoint.to_string(),
            })?;
        Ok(())
    } else {
        Err(AtomaServiceError::InvalidBody {
            message: "User ID not found in request body".to_string(),
            endpoint: endpoint.to_string(),
        })
    }
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
    use dashmap::DashMap;
    use futures::future::join_all;
    use prometheus_parse::Scrape;
    use prometheus_parse::Value;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::time::Duration;
    use std::time::Instant;
    use tokio::sync::RwLock;
    use tokio::time;
    use tracing::info;

    use crate::handlers::metrics::NUM_RATE_LIMITED_REQUESTS;
    use crate::handlers::InferenceService;
    use hyper::StatusCode;
    use tracing::instrument;

    use super::request_counter::RequestCounter;
    use rand::seq::SliceRandom;

    pub type Result<T> = std::result::Result<T, ChatCompletionsMetricsError>;
    type MetricValue = ChatCompletionsMetrics;
    type MetricResult = Result<MetricValue>;
    type MetricsVec = Vec<MetricResult>;
    type CachedMetrics = Option<MetricsVec>;
    type MetricsLock = Arc<RwLock<CachedMetrics>>;

    /// The default interval for updating the metrics
    const DEFAULT_METRICS_UPDATE_INTERVAL_MILLIS: u64 = 35;

    /// The timeout for the Prometheus metrics queries
    const METRICS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    /// The HTTP client for the metrics queries
    static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
        reqwest::Client::builder()
            .timeout(METRICS_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client")
    });

    /// Chat completions metrics
    #[derive(Debug, Clone, PartialEq)]
    pub struct ChatCompletionsMetrics {
        /// The model name  
        model: String,
        /// The chat completions service url
        chat_completions_service_url: String,
        /// The number of queue requests
        num_queued_requests: f64,
        /// The number of running requests
        num_running_requests: f64,
        /// The memory usage in fraction, e.g. 1.00 means 100% memory usage
        memory_usage: f64,
        /// The maximum number of running requests allowed for this url.
        max_number_of_running_requests: usize,
    }

    impl Eq for ChatCompletionsMetrics {}

    impl PartialOrd for ChatCompletionsMetrics {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for ChatCompletionsMetrics {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.num_queued_requests
                .total_cmp(&other.num_queued_requests)
                .then_with(|| {
                    self.memory_usage
                        .total_cmp(&other.memory_usage)
                        .then_with(|| {
                            self.num_running_requests
                                .total_cmp(&other.num_running_requests)
                        })
                })
        }
    }

    impl ChatCompletionsMetrics {
        #[must_use]
        pub fn above_upper_threshold_exceeded(&self, threshold: f64) -> bool {
            self.memory_usage > threshold
        }

        #[must_use]
        pub fn under_lower_threshold(&self, threshold: f64) -> bool {
            self.memory_usage <= threshold
        }
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
    /// * `chat_completions_service_urls` - A vector of tuples containing the model name, the chat completions service URL and the job name.
    /// * `metrics_update_interval` - The interval in seconds to update the metrics.
    #[instrument(level = "info", skip_all)]
    pub fn start_metrics_updater(
        chat_completions_service_urls: Vec<(String, String, String, usize)>,
        metrics_update_interval: Option<u64>,
    ) {
        type ChatCompletionsServiceUrls = Vec<(String, String, String, usize)>;
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
            .partition(|(_, _, job, _)| job.contains("vllm"));
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
                "Metrics update interval: {metrics_interval} milliseconds"
            );
            let mut interval = time::interval(Duration::from_millis(metrics_interval));
            loop {
                interval.tick().await;
                if !vllm_chat_completions_service_urls.is_empty() {
                    let vllm_metrics =
                        get_metrics(&InferenceService::Vllm, &vllm_chat_completions_service_urls)
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
                        &InferenceService::SgLang,
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
    /// * `jobs_with_url` - A slice of tuples containing model name, the chat completions service URL
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
        inference_service: &InferenceService,
        jobs_with_url: &[(String, String, String, usize)], // (model, url, job, max_concurrent_requests)
    ) -> Vec<Result<ChatCompletionsMetrics>> {
        let tasks =
            jobs_with_url
                .iter()
                .map(|(model, chat_completions_service_url, job, max_number_of_running_requests)| async move {
                    let response = HTTP_CLIENT
                        .get(format!("{chat_completions_service_url}/metrics"))
                        .send()
                        .await
                        .map_err(|_| {
                            ChatCompletionsMetricsError::NoMetricsFound(job.to_string())
                        })?;
                    let body = response.text().await?;
                    let lines = body
                        .lines()
                        .map(|line| Ok(line.replace(inference_service.get_service_prefix(), "")));
                    let metrics = Scrape::parse(lines).unwrap();
                    let num_queued_requests = extract_metric(
                        &metrics,
                        inference_service.get_queued_requests_metric_name(),
                        job,
                    )?;
                    let num_running_requests = extract_metric(
                        &metrics,
                        inference_service.get_running_requests_metric_name(),
                        job,
                    )?;
                    let memory_usage = extract_metric(
                        &metrics,
                        inference_service.get_usage(),
                        job,
                    )?;

                    Ok(ChatCompletionsMetrics {
                        model: model.clone(),
                        chat_completions_service_url: chat_completions_service_url.clone(),
                        num_queued_requests,
                        num_running_requests,
                        memory_usage,
                        max_number_of_running_requests: *max_number_of_running_requests,
                    })
                });
        join_all(tasks).await
    }

    /// Extracts a specific metric from the Scrape response.
    ///
    /// This function searches for a metric with the specified name in the
    /// Scrape response and returns its value if found.
    ///
    /// # Arguments
    ///
    /// * `metrics` - The Scrape response containing the metrics.
    /// * `name` - The name of the metric to extract.
    /// * `job` - The job name used for error reporting.
    ///
    /// # Returns
    ///
    /// Returns a `Result<f64>` containing the metric value if found,
    /// or an error if not found or if the value is not a Gauge.
    ///
    /// # Errors
    ///
    /// *   `ChatCompletionsMetricsError::NoMetricsFound`: If the specified metric is not found
    ///     or if the value is not a Gauge.
    /// *   Other variants of `ChatCompletionsMetricsError` may be returned if underlying
    ///     issues occur during metric collection from Prometheus (e.g., network errors,
    ///     parsing errors),
    ///     though the function attempts to handle missing individual metrics gracefully.
    fn extract_metric(metrics: &Scrape, name: &str, job: &str) -> Result<f64> {
        metrics
            .samples
            .iter()
            .find(|s| s.metric == name)
            .ok_or_else(|| ChatCompletionsMetricsError::NoMetricsFound(job.to_string()))
            .and_then(|sample| {
                if let Value::Gauge(value) = sample.value {
                    Ok(value)
                } else {
                    Err(ChatCompletionsMetricsError::NoMetricsFound(job.to_string()))
                }
            })
    }

    /// Retrieves all chat completions metrics for the specified model.
    ///
    /// This function fetches metrics from both vLLM and SgLang services,
    /// partitions the service URLs based on the job type, and retrieves metrics
    /// for each service. It returns a vector of `ChatCompletionsMetrics` for the specified model.
    ///
    /// # Arguments
    ///
    /// * `chat_completions_service_urls` - A vector of tuples containing the chat completions service URLs,
    ///   job names, and maximum concurrent requests.
    /// * `model` - The model name for which to retrieve metrics.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Vec<ChatCompletionsMetrics>>` containing the metrics for the specified model.
    ///
    /// # Errors
    ///
    /// *   `ChatCompletionsMetricsError::NoChatCompletionsServiceUrlsFound`: If no chat completions service URLs are provided.
    /// *   Other variants of `ChatCompletionsMetricsError` may be returned if underlying
    ///     issues occur during metric collection from Prometheus (e.g., network errors,
    ///     parsing errors), though the function attempts to handle missing individual metrics gracefully.
    #[instrument(
        level = "info",
        skip(chat_completions_service_urls, model),
        fields(model = model)
    )]
    pub async fn get_all_metrics(
        chat_completions_service_urls: &[(String, String, usize)], // (url, job, max_concurrent_requests)
        model: &str,
    ) -> Result<Vec<ChatCompletionsMetrics>> {
        type ChatCompletionsServiceUrls = Vec<(String, String, usize)>;

        if chat_completions_service_urls.is_empty() {
            tracing::warn!(
                target = "atoma-service",
                model = model,
                "No chat completions service URLs provided for model."
            );
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
            .partition(|(_, job, _)| job.contains("vllm"));

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
            let vllm_chat_completions_service_urls_with_model: Vec<(
                String,
                String,
                String,
                usize,
            )> = vllm_chat_completions_service_urls
                .iter()
                .map(|(url, job, max_concurrent_requests)| {
                    (
                        model.to_string(),
                        url.clone(),
                        job.clone(),
                        *max_concurrent_requests,
                    )
                })
                .collect();
            get_metrics(
                &InferenceService::Vllm,
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
            let sglang_chat_completions_service_urls_with_model: Vec<(
                String,
                String,
                String,
                usize,
            )> = sglang_chat_completions_service_urls
                .iter()
                .map(|(url, job, max_concurrent_requests)| {
                    (
                        model.to_string(),
                        url.clone(),
                        job.clone(),
                        *max_concurrent_requests,
                    )
                })
                .collect();
            get_metrics(
                &InferenceService::SgLang,
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
                    memory_usage,
                    max_number_of_running_requests,
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
                        memory_usage,
                        max_number_of_running_requests,
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
        Ok(metrics_results)
    }

    /// Checks if the number of requests is within the allowed limits.
    ///
    /// This function checks if the number of requests is within the allowed limits
    /// defined by `limit_request_interval_ms` and `limit_number_of_requests_per_interval`.
    ///
    /// # Arguments
    ///
    /// * `requests_limiter_times` - A reference to a shared map that tracks request times for each URL.
    /// * `limit_request_interval_ms` - The time interval in milliseconds within which the number of requests is limited.
    /// * `limit_number_of_requests_per_interval` - The maximum number of requests allowed within the specified interval.
    ///
    /// # Returns
    ///
    /// Returns `true` if the request can be made (i.e., the number of requests is within the allowed limits),
    /// or `false` if the request exceeds the limit.
    #[instrument(level = "debug")]
    fn check_rate_limit(
        requests_times: &mut VecDeque<Instant>,
        limit_request_interval_ms: u128,
        limit_number_of_requests_per_interval: usize,
    ) -> bool {
        while !requests_times.is_empty()
            && requests_times.front().unwrap().elapsed().as_millis() >= limit_request_interval_ms
        {
            requests_times.pop_front();
        }
        requests_times.len() < limit_number_of_requests_per_interval
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
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::significant_drop_tightening)]
    pub async fn get_best_available_chat_completions_service_url(
        running_num_requests: &RequestCounter,
        requests_limiter_times: &Arc<DashMap<String, VecDeque<Instant>>>,
        limit_request_interval_ms: u128,
        limit_number_of_requests_per_interval: usize,
        chat_completions_service_urls: &[(String, String, usize)], // (url, job, max_concurrent_requests)
        model: &str,
        memory_upper_threshold: f64,
        max_num_queued_requests: f64,
    ) -> Result<(String, StatusCode)> {
        let mut metrics_results = get_all_metrics(chat_completions_service_urls, model)
            .await
            .map_err(|e| {
                tracing::error!(
                    target = "atoma-service",
                    level = "error",
                    "Failed to get metrics for model: {model} with error: {e}"
                );
                e
            })?;
        if metrics_results.is_empty() {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "No metrics found for model: {model}",
            );
            // NOTE: In this case, we pick one of the urls at random
            let mut indices: Vec<usize> = (0..chat_completions_service_urls.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            for index in indices {
                let (best_url, _, max_concurrent_requests) = &chat_completions_service_urls[index];
                let mut requests_times_guard =
                    requests_limiter_times.entry(best_url.clone()).or_default();
                if !check_rate_limit(
                    &mut requests_times_guard,
                    limit_request_interval_ms,
                    limit_number_of_requests_per_interval,
                ) {
                    tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Too many requests in the last {limit_request_interval_ms} milliseconds for model: {model} at url: {best_url}",
                );
                    continue;
                }
                if running_num_requests.increment(best_url.as_str(), *max_concurrent_requests) {
                    requests_times_guard.push_back(Instant::now());
                    return Ok((best_url.clone(), StatusCode::OK));
                }
            }

            NUM_RATE_LIMITED_REQUESTS.increment(model);
            return Ok((String::new(), StatusCode::TOO_MANY_REQUESTS));
        }

        // Select the best available chat completions service URL based on the number of queued and running requests.
        metrics_results.sort();

        for metric in metrics_results {
            let best_url = metric.chat_completions_service_url.clone();
            let mut requests_times_guard =
                requests_limiter_times.entry(best_url.clone()).or_default();
            if !check_rate_limit(
                &mut requests_times_guard,
                limit_request_interval_ms,
                limit_number_of_requests_per_interval,
            ) {
                tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Too many requests in the last {limit_request_interval_ms} milliseconds for model: {model} at url: {}",
                    metric.chat_completions_service_url
                );
                continue;
            }

            if metric.num_queued_requests > max_num_queued_requests {
                tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Number of queued requests for model: {model} is too high: {}",
                    metric.num_queued_requests
                );
                continue;
            }
            if metric.above_upper_threshold_exceeded(memory_upper_threshold) {
                tracing::debug!(
                    target = "atoma-service",
                    level = "debug",
                    "Memory usage for model: {model} is too high: {}",
                    metric.memory_usage
                );
                continue;
            }
            if running_num_requests.increment(
                &metric.chat_completions_service_url,
                metric.max_number_of_running_requests,
            ) {
                requests_times_guard.push_back(Instant::now());
                tracing::info!(
                    target = "atoma-service",
                    level = "info",
                    "Best available chat completions service URL for model: {model} is: {best_url} with and {} queue requests",
                    metric.num_queued_requests
                );
                return Ok((best_url, StatusCode::OK));
            }
        }

        NUM_RATE_LIMITED_REQUESTS.increment(model);
        return Ok((String::new(), StatusCode::TOO_MANY_REQUESTS));
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
