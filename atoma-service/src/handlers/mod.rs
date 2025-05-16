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
    concurrent_requests: u64,
) -> Result<(), AtomaServiceError> {
    state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateStackNumComputeUnits {
            stack_small_id,
            total_compute_units,
            estimated_total_compute_units,
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
/// * `estimated_total_compute_units` - The estimated number of compute units for the request
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
        estimated_total_compute_units,
        total_amount,
        price_per_one_million_compute_units,
        endpoint
    ),
    err
)]
pub fn update_fiat_amount(
    state_manager_sender: &Sender<AtomaAtomaStateManagerEvent>,
    user_address: String,
    estimated_total_compute_units: i64,
    total_amount: i64,
    price_per_one_million_compute_units: i64,
    endpoint: &str,
) -> Result<(), AtomaServiceError> {
    let estimated_total_amount = (estimated_total_compute_units as u128
        * price_per_one_million_compute_units as u128
        / ONE_MILLION) as i64;
    let total_amount =
        (total_amount as u128 * price_per_one_million_compute_units as u128 / ONE_MILLION) as i64;
    state_manager_sender
        .send(AtomaAtomaStateManagerEvent::UpdateFiatAmount {
            user_address,
            total_amount,
            estimated_total_amount,
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
    use opentelemetry::KeyValue;
    use prometheus_http_query::response::PromqlResult;
    use rand::Rng;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::time::Duration;
    use tokio::sync::RwLock;
    use tokio::time;

    use crate::handlers::metrics::CHAT_COMPLETIONS_TOO_MANY_REQUESTS;
    use hyper::StatusCode;
    use prometheus_http_query::Client;
    use tracing::{info, instrument};

    pub type Result<T> = std::result::Result<T, ChatCompletionsMetricsError>;
    type MetricValue = ChatCompletionsMetrics;
    type MetricResult = Result<MetricValue>;
    type MetricsVec = Vec<MetricResult>;
    type CachedMetrics = Option<MetricsVec>;
    type MetricsLock = Arc<RwLock<CachedMetrics>>;

    /// The timeout for the Prometheus metrics queries
    const METRICS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

    /// Prometheus url for the atoma-node instance
    const PROMETHEUS_URL: &str = "http://prometheus:9090";

    /// The HTTP client for the Prometheus metrics queries
    static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
        Client::from(
            reqwest::Client::builder()
                .timeout(METRICS_TIMEOUT)
                .build()
                .expect("Failed to create HTTP client"),
            PROMETHEUS_URL,
        )
        .expect("Failed to create HTTP client")
    });

    /// Chat completions metrics
    #[derive(Debug, Clone)]
    struct ChatCompletionsMetrics {
        /// The model name  
        model: String,
        /// The chat completions service url
        chat_completions_service_url: String,
        /// The waiting queue time
        waiting_queue_time: f64,
        /// The number of queue requests
        num_queue_requests: f64,
        /// The time to first token
        time_to_first_token_seconds: f64,
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

    /// Start the background task to update metrics every 30 seconds
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
            let metrics_interval = metrics_update_interval.unwrap_or(30);
            let mut interval = time::interval(Duration::from_secs(metrics_interval));
            loop {
                interval.tick().await;
                if !vllm_chat_completions_service_urls.is_empty() {
                    let vllm_metrics =
                        get_metrics_vllm(&HTTP_CLIENT, &vllm_chat_completions_service_urls).await;
                    if vllm_metrics.iter().any(std::result::Result::is_ok) {
                        VLLM_METRICS_CACHE.update_metrics(vllm_metrics).await;
                    } else {
                        tracing::warn!(
                            "Failed to retrieve any valid vLLM metrics, not updating cache"
                        );
                    }
                }
                if !sglang_chat_completions_service_urls.is_empty() {
                    let sglang_metrics =
                        get_metrics_sglang(&HTTP_CLIENT, &sglang_chat_completions_service_urls)
                            .await;
                    info!(
                        target = "atoma-service",
                        module = "inference_service_metrics",
                        level = "info",
                        "Received SgLang metrics response for {sglang_chat_completions_service_urls:?}, {sglang_metrics:?}"
                    );
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

    /// Extracts the metrics from the response  
    ///
    /// # Arguments
    ///
    /// * `response` - The response from the Prometheus query.
    /// * `job` - The job name.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a tuple `(String, f64)` on success, where:
    ///   - The `String` is the `job` passed as input.
    ///   - The `f64` is the calculated average of the request queue time in seconds.
    ///
    /// # Errors
    ///  
    /// Returns a `ChatCompletionsMetricsError` if:
    ///   - The Prometheus query fails.
    ///   - No metrics data is found for the specified job.
    ///   - The response data cannot be parsed correctly.
    #[instrument(level = "info", skip_all)]
    fn extract_metrics(
        response: &std::result::Result<PromqlResult, prometheus_http_query::Error>,
        job: &str,
    ) -> Result<f64> {
        response
            .as_ref()
            .map_err(|_| ChatCompletionsMetricsError::NoMetricsFound(job.to_string()))
            .and_then(|response| {
                response.data().as_vector().map_or_else(
                    || Ok(-1.0),
                    |vector| {
                        vector
                            .iter()
                            .find(|instant| instant.metric().get("job") == Some(&job.to_string()))
                            .map_or_else(
                                || Ok(-1.0),
                                |value| {
                                    let sample = value.sample();
                                    Ok(sample.value())
                                },
                            )
                    },
                )
            })
    }

    /// Retrieves both request queue time and TTFT metrics from a vLLM service.
    #[instrument(
        level = "info",
        skip_all,
        fields(
            jobs_with_url=models_with_urls_and_job
                .iter()
                .map(|(model, url, job)| format!("{job}={url},{model}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    )]
    async fn get_metrics_vllm(
        client: &Client,
        models_with_urls_and_job: &[(String, String, String)],
    ) -> Vec<Result<ChatCompletionsMetrics>> {
        let jobs = models_with_urls_and_job
            .iter()
            .map(|(_model, _url, job)| job.as_str())
            .collect::<Vec<_>>()
            .join("|");
        tracing::debug!(
            target = "atoma-service",
            module = "vllm_metrics",
            level = "info",
            "Getting VLLM metrics for jobs: {jobs}"
        );

        let queue_time_query = format!(
            "histogram_quantile(0.90, sum by (le,job) (rate(vllm:request_queue_time_seconds_bucket{{job=~\"{jobs}\"}}[30s])))"
        );
        let num_running_requests_query = format!(
            "quantile_over_time(
                0.90,
                vllm:num_requests_running{{job=~\"{jobs}\"}}[30s]
            )"
        );
        let num_queue_requests_query = format!(
            "quantile_over_time(
                0.90,
                vllm:num_queue_reqs{{job=~\"{jobs}\"}}[30s]
            )"
        );
        let ttft_query =
            format!("histogram_quantile(0.90, sum by (le,job) (rate(vllm:time_to_first_token_seconds_bucket{{job=~\"{jobs}\"}}[30s])))");

        let (
            queue_time_response,
            num_running_requests_response,
            num_queue_requests_response,
            ttft_response,
        ) = tokio::join!(
            client.query(&queue_time_query).get(),
            client.query(&num_running_requests_query).get(),
            client.query(&num_queue_requests_query).get(),
            client.query(&ttft_query).get(),
        );

        models_with_urls_and_job
            .iter()
            .map(|(model, url, job)| {
                let queue_time = extract_metrics(&queue_time_response, job);
                let ttft = extract_metrics(&ttft_response, job);
                let num_running_requests = extract_metrics(&num_running_requests_response, job);
                let num_queue_requests = extract_metrics(&num_queue_requests_response, job);

                queue_time.and_then(|qt| {
                    ttft.and_then(|ttf| {
                        num_running_requests.and_then(|nrr| {
                            num_queue_requests.map(|nqr| ChatCompletionsMetrics {
                                model: model.to_string(),
                                chat_completions_service_url: url.to_string(),
                                waiting_queue_time: qt,
                                time_to_first_token_seconds: ttf,
                                num_running_requests: nrr,
                                num_queue_requests: nqr,
                            })
                        })
                    })
                })
            })
            .collect()
    }

    /// Retrieves the average request queue time metric from a sglang service and time to first token metric from a SgLang service.
    ///
    /// This function executes a Prometheus query against the configured Prometheus instance
    /// to get the `sglang:avg_request_queue_latency` histogram for the specified `job`,
    /// calculates the average, and returns it along with the service URL.
    ///
    /// # Arguments
    ///
    /// * `client` - The Prometheus HTTP query client.
    /// * `jobs_with_url` - A vector of tuples containing the job name and the service URL.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a tuple `(String, f64)` on success, where:
    ///   - The `String` is the `job` passed as input.
    ///   - The `f64` is the calculated average of the request queue time in seconds.
    ///
    /// # Errors
    ///
    /// Returns a `ChatCompletionsMetricsError` if:
    ///   - The Prometheus query fails.
    ///   - No metrics data is found for the specified job.
    ///   - The response data cannot be parsed correctly.
    #[instrument(
        level = "info",
        skip_all,
        fields(
            jobs=models_with_urls_and_jobs
                .iter()
                .map(|(_, _, job)| job.as_str())
                .collect::<Vec<_>>()
                .join(",")
        )
    )]
    async fn get_metrics_sglang(
        client: &Client,
        models_with_urls_and_jobs: &[(String, String, String)],
    ) -> Vec<Result<ChatCompletionsMetrics>> {
        let jobs = models_with_urls_and_jobs
            .iter()
            .map(|(_model, _url, job)| job.as_str())
            .collect::<Vec<_>>()
            .join("|");
        tracing::debug!(
            target = "atoma-service",
            module = "sglang_metrics",
            level = "info",
            "Getting metrics for jobs: {jobs}" // jobs = "http://host.docker.internal:3000"
        );
        tracing::debug!(
            target = "atoma-service",
            module = "sglang_metrics",
            level = "info",
            "prometheus queries: 
            
            quantile_over_time(
                0.90,
                sglang:num_queue_reqs{{job=\"{jobs}\"}}[30s]
            )
            
            histogram_quantile(0.90, sum by (le,job) (rate(sglang:time_to_first_token_seconds_bucket{{job=\"{jobs}\"}}[30s])))"
        );
        let waiting_queue_time = format!(
            "quantile_over_time(
                0.90,
                sglang:avg_request_queue_latency{{job=\"{jobs}\"}}[30s]
            )"
        );
        let num_queue_requests = format!(
            "quantile_over_time(
                0.90,
                sglang:num_queue_reqs{{job=\"{jobs}\"}}[30s]
            )"
        );
        let num_running_requests = format!(
            "quantile_over_time(
                0.90,
                sglang:num_running_reqs{{job=\"{jobs}\"}}[30s]
            )"
        );
        let ttft = format!(
            "histogram_quantile(0.90, sum by (le,job) (rate(sglang:time_to_first_token_seconds_bucket{{job=\"{jobs}\"}}[30s])))",
        );
        let (
            waiting_queue_time_response,
            num_queue_requests_response,
            num_running_requests_response,
            ttft_response,
        ) = tokio::join!(
            client.query(&waiting_queue_time).get(),
            client.query(&num_queue_requests).get(),
            client.query(&num_running_requests).get(),
            client.query(&ttft).get()
        );
        tracing::debug!(
            target = "atoma-service",
            module = "sglang_metrics",
            level = "info",
            "Received sglang metrics response for {jobs}:\n 
                waiting_queue_time: {waiting_queue_time_response:?}, 
                num_queue_requests: {num_queue_requests_response:?}, 
                num_running_requests: {num_running_requests_response:?}, 
                ttft: {ttft_response:?}"
        );
        models_with_urls_and_jobs
            .iter()
            .map(|(model, url, job)| {
                let waiting_queue_time = extract_metrics(&waiting_queue_time_response, job);
                let num_queue_requests = extract_metrics(&num_queue_requests_response, job);
                let num_running_requests = extract_metrics(&num_running_requests_response, job);
                let ttft = extract_metrics(&ttft_response, job);

                waiting_queue_time.and_then(|wqt| {
                    num_queue_requests.and_then(|nqr| {
                        num_running_requests.and_then(|nrr| {
                            ttft.map(|ttf| ChatCompletionsMetrics {
                                model: model.to_string(),
                                chat_completions_service_url: url.to_string(),
                                waiting_queue_time: wqt,
                                time_to_first_token_seconds: ttf,
                                num_running_requests: nrr,
                                num_queue_requests: nqr,
                            })
                        })
                    })
                })
            })
            .collect()
    }

    /// Retrieves the best available chat completions service URL for a given model.
    #[instrument(level = "info", skip_all, fields(model=model))]
    #[allow(clippy::float_cmp)]
    pub async fn get_best_available_chat_completions_service_url(
        chat_completions_service_urls: &[(String, String)],
        model: &str,
    ) -> Result<(String, StatusCode)> {
        const MAX_ALLOWED_NUM_QUEUE_REQUESTS: f64 = 16.0; // Default to 6 requests
        const MAX_ALLOWED_TIME_TO_FIRST_TOKEN_SECONDS: f64 = 2.0; // Default to 4 seconds
        const MAX_ALLOWED_WAITING_TIME_SECONDS: f64 = 2.0; // Default to 6 seconds

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

        let mut min_time_to_first_token_seconds = f64::INFINITY;

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
            // If no cached metrics, get them directly
            let vllm_chat_completions_service_urls_with_model: Vec<(String, String, String)> =
                vllm_chat_completions_service_urls
                    .iter()
                    .map(|(url, job)| (model.to_string(), url.clone(), job.clone()))
                    .collect();
            get_metrics_vllm(&HTTP_CLIENT, &vllm_chat_completions_service_urls_with_model).await
        };
        let sglang_metrics = if sglang_chat_completions_service_urls.is_empty() {
            vec![]
        } else if let Some(metrics) = SGLANG_METRICS_CACHE.get_metrics().await {
            metrics
        } else {
            // If no cached metrics, get them directly (this is the "None" case)
            let sglang_chat_completions_service_urls_with_model: Vec<(String, String, String)> =
                sglang_chat_completions_service_urls
                    .iter()
                    .map(|(url, job)| (model.to_string(), url.clone(), job.clone()))
                    .collect();
            get_metrics_sglang(
                &HTTP_CLIENT,
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
            let (
                chat_completions_service_url,
                waiting_queue_time,
                num_queue_requests,
                num_running_requests,
                time_to_first_token_seconds,
            ) = match metric {
                Ok(ChatCompletionsMetrics {
                    model: current_model,
                    chat_completions_service_url,
                    waiting_queue_time,
                    num_queue_requests,
                    time_to_first_token_seconds,
                    num_running_requests,
                }) => {
                    info!(
                        target = "atoma-service",
                        module = "inference_service_metrics",
                        level = "info",
                        "current_model = {current_model}, model = {model}"
                    );
                    if current_model != model {
                        // NOTE: We only want to consider metrics for the current model
                        continue;
                    }
                    info!(
                        target = "atoma-service",
                        module = "vllm_metrics",
                        level = "info",
                        "Received vLLM/SgLang metrics response for {chat_completions_service_url}:\n
                            num_waiting_queue_time={waiting_queue_time},
                            num_queue_requests={num_queue_requests},
                            num_running_requests={num_running_requests},
                            time_to_first_token_seconds={time_to_first_token_seconds}"
                    );
                    (
                        chat_completions_service_url,
                        waiting_queue_time,
                        num_queue_requests,
                        num_running_requests,
                        time_to_first_token_seconds,
                    )
                }
                Err(e) => {
                    tracing::warn!(
                        target = "atoma-service",
                        module = "vllm_metrics",
                        level = "error",
                        "Failed to get metrics for chat completions service url with error: {e}",
                    );
                    continue;
                }
            };

            let mut waiting_queue_time = waiting_queue_time;
            let mut num_queue_requests = num_queue_requests;

            // 1. First check if the number of running requests is NaN or 0.0
            if num_running_requests.is_nan() || num_running_requests == 0.0 {
                // NOTE: The minimum time to first token is never 0.0, but in this case, since there are no running requests, we can safely assume that it's 0.0
                min_time_to_first_token_seconds = 0.0;
                metrics_results.push(ChatCompletionsMetrics {
                    model: model.to_string(),
                    chat_completions_service_url,
                    waiting_queue_time: 0.0,
                    num_queue_requests: 0.0,
                    num_running_requests: 0.0,
                    time_to_first_token_seconds: 0.0,
                });
                break;
            }

            // 2. Check if the number of queue requests is NaN or smaller than the current minimum
            if num_queue_requests.is_nan() {
                num_queue_requests = 0.0;
            }

            // 3. Check if the time to first token is NaN or smaller than the current minimum
            if time_to_first_token_seconds.is_nan() || time_to_first_token_seconds == 0.0 {
                min_time_to_first_token_seconds = 0.0;
                metrics_results.push(ChatCompletionsMetrics {
                    model: model.to_string(),
                    chat_completions_service_url,
                    waiting_queue_time,
                    num_queue_requests,
                    num_running_requests,
                    time_to_first_token_seconds: 0.0,
                });
                break;
            }

            // 4. Check if the waiting time is NaN or smaller than the current minimum
            if waiting_queue_time.is_nan() {
                waiting_queue_time = 0.0;
            }

            // 5. In this case, we have valid metrics, so we can add them to the results
            metrics_results.push(ChatCompletionsMetrics {
                model: model.to_string(),
                chat_completions_service_url,
                waiting_queue_time,
                num_queue_requests,
                num_running_requests,
                time_to_first_token_seconds,
            });

            // 6. Update the minimum TTFT if necessary
            if time_to_first_token_seconds < min_time_to_first_token_seconds {
                min_time_to_first_token_seconds = time_to_first_token_seconds;
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

        if min_time_to_first_token_seconds > MAX_ALLOWED_TIME_TO_FIRST_TOKEN_SECONDS {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "Node is currently under high load, the best available chat completions service URL for model: {model} has a TTFT of at least {min_time_to_first_token_seconds} seconds",
            );
            CHAT_COMPLETIONS_TOO_MANY_REQUESTS.add(1, &[KeyValue::new("model", model.to_string())]);
            return Ok((
                chat_completions_service_urls[0].0.clone(),
                StatusCode::TOO_MANY_REQUESTS,
            ));
        }

        // NOTE: At this point, we should have a list of metrics with the minimum TTFT, since we already handled the case where `metrics_results` is empty
        let best_ttft_metrics = metrics_results
            .iter()
            .filter(|m| m.time_to_first_token_seconds == min_time_to_first_token_seconds)
            .collect::<Vec<_>>();

        // NOTE: We made sure above that TTFT is not NaN, so we can safely unwrap. Also from the previous NOTE, a minimum TTFT is guaranteed
        let best_metrics = best_ttft_metrics
            .iter()
            .min_by(|a, b| {
                let a_predicate = a.waiting_queue_time == -1.0;
                let b_predicate = b.waiting_queue_time == -1.0;

                match (a_predicate, b_predicate) {
                    (true | false, true) | (true, false) => a
                        .num_queue_requests
                        .partial_cmp(&b.num_queue_requests)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    (false, false) => a
                        .waiting_queue_time
                        .partial_cmp(&b.waiting_queue_time)
                        .unwrap_or(std::cmp::Ordering::Equal),
                }
            })
            .unwrap();

        if best_metrics.num_queue_requests > MAX_ALLOWED_NUM_QUEUE_REQUESTS {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "Node is currently under high load, the best available chat completions service URL for model: {model} has a num queue requests of at least {} requests",
                best_metrics.num_queue_requests
            );
            CHAT_COMPLETIONS_TOO_MANY_REQUESTS.add(1, &[KeyValue::new("model", model.to_string())]);
            return Ok((
                chat_completions_service_urls[0].0.clone(),
                StatusCode::TOO_MANY_REQUESTS,
            ));
        }

        if best_metrics.waiting_queue_time > MAX_ALLOWED_WAITING_TIME_SECONDS {
            tracing::warn!(
                target = "atoma-service",
                level = "warn",
                "Node is currently under high load, the best available chat completions service URL for model: {model} has a waiting time of at least {} seconds",
                best_metrics.waiting_queue_time
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
            "Best available chat completions service URL for model: {model} is: {best_url} with a TTFT of {min_time_to_first_token_seconds} seconds, waiting queue time of {} seconds and {} queue requests",
            best_metrics.waiting_queue_time,
            best_metrics.num_queue_requests
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
