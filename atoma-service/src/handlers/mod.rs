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
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::time::Duration;
    use tokio::sync::RwLock;
    use tokio::time;
    use tracing::debug;

    use crate::handlers::metrics::CHAT_COMPLETIONS_TOO_MANY_REQUESTS;
    use hyper::StatusCode;
    use prometheus_http_query::Client;
    use tracing::{info, instrument};

    pub type Result<T> = std::result::Result<T, VllmMetricsError>;
    type MetricValue = (String, (f64, f64));
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

        async fn update_metrics(&self, new_metrics: Vec<Result<(String, (f64, f64))>>) {
            *self.metrics.write().await = Some(new_metrics);
        }
    }

    /// Global metrics cache
    #[allow(clippy::redundant_closure)]
    static METRICS_CACHE: LazyLock<MetricsCache> = LazyLock::new(|| MetricsCache::new());

    /// Start the background task to update metrics every 30 seconds
    ///
    /// # Arguments
    ///
    /// * `chat_completions_service_urls` - A vector of tuples containing the chat completions service URL and the job name.
    /// * `metrics_update_interval` - The interval in seconds to update the metrics.
    #[instrument(level = "info", skip_all)]
    pub fn start_metrics_updater(
        chat_completions_service_urls: Vec<(String, String)>,
        metrics_update_interval: Option<u64>,
    ) {
        let chat_completions_service_urls = Arc::new(chat_completions_service_urls);
        tokio::spawn(async move {
            let metrics_interval = metrics_update_interval.unwrap_or(30);
            let mut interval = time::interval(Duration::from_secs(metrics_interval));
            loop {
                interval.tick().await;
                let metrics = get_metrics(&HTTP_CLIENT, &chat_completions_service_urls).await;
                if metrics.iter().any(std::result::Result::is_ok) {
                    METRICS_CACHE.update_metrics(metrics).await;
                } else {
                    tracing::warn!("Failed to retrieve any valid metrics, not updating cache");
                }
            }
        });
    }

    /// Retrieves both request queue time and TTFT metrics from a vLLM service.
    #[instrument(level = "info", skip_all, fields(jobs_with_url=jobs_with_url.iter().map(|(url, job)| format!("{job}={url}")).collect::<Vec<_>>().join(",")))]
    async fn get_metrics(
        client: &Client,
        jobs_with_url: &[(String, String)],
    ) -> Vec<Result<(String, (f64, f64))>> {
        let jobs = jobs_with_url
            .iter()
            .map(|(_url, job)| job.as_str())
            .collect::<Vec<_>>()
            .join("|");

        let queue_time_query = format!(
            "histogram_quantile(0.90, sum by (le,job) (rate(vllm:request_queue_time_seconds_bucket{{job=~\"{jobs}\"}}[30s])))"
        );

        let ttft_query =
            format!("histogram_quantile(0.90, sum by (le,job) (rate(vllm:time_to_first_token_seconds_bucket{{job=~\"{jobs}\"}}[30s])))");

        let (queue_time_response, ttft_response) = tokio::join!(
            client.query(&queue_time_query).get(),
            client.query(&ttft_query).get()
        );

        jobs_with_url
            .iter()
            .map(|(url, job)| {
                let queue_time = queue_time_response
                    .as_ref()
                    .map_err(|_| VllmMetricsError::NoMetricsFound(job.to_string()))
                    .and_then(|response| {
                        response
                            .data()
                            .as_vector()
                            .ok_or_else(|| VllmMetricsError::NoMetricsFound(job.to_string()))
                            .and_then(|vector| {
                                vector
                                    .iter()
                                    .find(|instant| {
                                        instant.metric().get("job") == Some(&job.to_string())
                                    })
                                    .ok_or_else(|| {
                                        VllmMetricsError::NoMetricsFound(job.to_string())
                                    })
                                    .map(|value| {
                                        let sample = value.sample();
                                        sample.value()
                                    })
                            })
                    });

                let ttft = ttft_response
                    .as_ref()
                    .map_err(|_| VllmMetricsError::NoMetricsFound(job.to_string()))
                    .and_then(|response| {
                        response
                            .data()
                            .as_vector()
                            .ok_or_else(|| VllmMetricsError::NoMetricsFound(job.to_string()))
                            .and_then(|vector| {
                                vector
                                    .iter()
                                    .find(|instant| {
                                        instant.metric().get("job") == Some(&job.to_string())
                                    })
                                    .ok_or_else(|| {
                                        VllmMetricsError::NoMetricsFound(job.to_string())
                                    })
                                    .map(|value| {
                                        let sample = value.sample();
                                        sample.value()
                                    })
                            })
                    });

                queue_time.and_then(|qt| ttft.map(|gc| (url.to_string(), (qt, gc))))
            })
            .collect()
    }

    /// Retrieves the 90th percentile request queue time metric from a vLLM service.
    ///
    /// This function executes a Prometheus query against the configured Prometheus instance
    /// to get the `vllm:request_queue_time_seconds` histogram for the specified `job`,
    /// calculates the 90th percentile, and returns it along with the service URL.
    ///
    /// # Arguments
    ///
    /// * `client` - The Prometheus HTTP query client.
    /// * `job` - The Prometheus job name corresponding to the vLLM service instance.
    /// * `chat_completions_service_url` - The URL of the vLLM service instance, returned alongside the metric.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a tuple `(String, f64)` on success, where:
    ///   - The `String` is the `chat_completions_service_url` passed as input.
    ///   - The `f64` is the calculated 90th percentile of the request queue time in seconds.
    ///
    /// # Errors
    ///
    /// Returns a `VllmMetricsError` if:
    ///   - The Prometheus query fails.
    ///   - No metrics data is found for the specified job.
    ///   - The response data cannot be parsed correctly.
    #[instrument(level = "info", skip_all, fields(job=job))]
    async fn get_sglang_request_queue_latency(
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
            "histogram_quantile(
                0.90,
                sum(rate(sglang:avg_request_queue_latency{{job=\"{job}\"}}[30s])
                    or vector(0)) by (le)
            )"
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
    #[instrument(level = "info", skip_all, fields(model=model))]
    #[allow(clippy::float_cmp)]
    pub async fn get_best_available_chat_completions_service_url(
        chat_completions_service_urls: &[(String, String)],
        model: &str,
    ) -> Result<(String, StatusCode)> {
        const MAX_ALLOWED_REQUEST_QUEUE_TIME_SECONDS: f64 = 2.0; // Default to 2 seconds

        if chat_completions_service_urls.is_empty() {
            return Err(VllmMetricsError::NoChatCompletionsServiceUrlsFound(
                model.to_string(),
            ));
        }

        let mut min_request_queue_time_seconds = f64::INFINITY;
        let mut min_time_to_first_token_seconds = f64::INFINITY;
        let mut best_url = chat_completions_service_urls[0].0.clone();

        // Get cached metrics
        let metrics = match METRICS_CACHE.get_metrics().await {
            Some(metrics) => metrics,
            None => {
                // If no cached metrics, get them directly
                get_metrics(&HTTP_CLIENT, chat_completions_service_urls).await
            }
        };

        for metric in metrics {
            let (
                chat_completions_service_url,
                (request_queue_time_seconds, time_to_first_token_seconds),
            ) = match metric {
                Ok((
                    chat_completions_service_url,
                    (request_queue_time_seconds, time_to_first_token_seconds),
                )) => {
                    info!(
                        target = "atoma-service",
                        module = "vllm_metrics",
                        level = "info",
                        "Received vLLM metrics response for {chat_completions_service_url}: queue_time={request_queue_time_seconds}, time_to_first_token_seconds={time_to_first_token_seconds}"
                    );
                    (
                        chat_completions_service_url,
                        (request_queue_time_seconds, time_to_first_token_seconds),
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

            // Handle NaN case first
            if request_queue_time_seconds.is_nan() {
                min_request_queue_time_seconds = 0.0;
                best_url.clone_from(&chat_completions_service_url);
                break;
            }

            if time_to_first_token_seconds < min_time_to_first_token_seconds {
                min_time_to_first_token_seconds = time_to_first_token_seconds;
            }

            // Update min time and best URL if we found a better option
            if request_queue_time_seconds < min_request_queue_time_seconds
                || (request_queue_time_seconds == min_request_queue_time_seconds
                    && time_to_first_token_seconds < min_time_to_first_token_seconds)
            {
                debug!(
                    target = "atoma-service",
                    module = "vllm_metrics",
                    level = "debug",
                    "Updating best chat completions service url to {chat_completions_service_url} with request queue time {request_queue_time_seconds} and time to first token {time_to_first_token_seconds}"
                );
                min_request_queue_time_seconds = request_queue_time_seconds;
                best_url.clone_from(&chat_completions_service_url);
            }
        }

        // If we never found valid metrics, default to 0.0
        if min_request_queue_time_seconds == f64::INFINITY {
            min_request_queue_time_seconds = 0.0;
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
            CHAT_COMPLETIONS_TOO_MANY_REQUESTS.add(1, &[KeyValue::new("model", model.to_string())]);
            return Ok((best_url, StatusCode::TOO_MANY_REQUESTS));
        }
        Ok((best_url, StatusCode::OK))
    }

    #[derive(Debug, thiserror::Error, Clone)]
    pub enum VllmMetricsError {
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
    impl From<reqwest::Error> for VllmMetricsError {
        fn from(err: reqwest::Error) -> Self {
            Self::GetMetricsError(err.to_string())
        }
    }

    impl From<std::num::ParseFloatError> for VllmMetricsError {
        fn from(err: std::num::ParseFloatError) -> Self {
            Self::InvalidMetricsValue(err.to_string())
        }
    }

    impl From<serde_json::Error> for VllmMetricsError {
        fn from(err: serde_json::Error) -> Self {
            Self::InvalidMetricsResponse(err.to_string())
        }
    }

    impl From<prometheus_http_query::Error> for VllmMetricsError {
        fn from(err: prometheus_http_query::Error) -> Self {
            Self::FailedToCreateHttpClient(err.to_string())
        }
    }
}
