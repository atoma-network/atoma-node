use std::{
    collections::HashMap,
    num::ParseFloatError,
    sync::{LazyLock, Mutex},
};

use futures::stream::{FuturesUnordered, StreamExt};
use prometheus_http_query::Client;
use reqwest;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::constants::{
    IMAGE_GENERATION_LATENCY_QUERY, IMAGE_GENERATION_NUM_RUNNING_REQUESTS_QUERY,
    SGLANG_AVG_QUEUE_LATENCY_QUERY, SGLANG_RUNNING_REQUESTS_QUERY,
    SGLANG_TIME_PER_OUTPUT_TOKEN_QUERY, SGLANG_TIME_TO_FIRST_TOKEN_QUERY,
    SGLANG_WAITING_REQUESTS_QUERY, TEI_EMBEDDINGS_BATCH_SIZE_QUERY,
    TEI_EMBEDDINGS_BATCH_TOKENS_QUERY, TEI_EMBEDDINGS_INFERENCE_DURATION_QUERY,
    TEI_EMBEDDINGS_INPUT_LENGTH_QUERY, TEI_EMBEDDINGS_QUEUE_DURATION_QUERY,
    VLLM_GPU_CACHE_USAGE_PERC_QUERY, VLLM_RUNNING_REQUESTS_QUERY, VLLM_TIME_PER_OUTPUT_TOKEN_QUERY,
    VLLM_TIME_TO_FIRST_TOKEN_QUERY, VLLM_WAITING_REQUESTS_QUERY,
};

/// Metrics delta time, in seconds
pub const METRICS_DELTA_TIME: u64 = 30;

/// Metrics timeout, in seconds
const METRICS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// vLLM serving engine
const VLLM: &str = "vllm";

/// SGLANG serving engine
const SGLANG: &str = "sglang";

/// TEI serving engine
const TEI: &str = "tei";

/// Mistral-rs serving engine
const MISTRALRS: &str = "mistralrs";

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

/// A simple cache for vLLM queries.
static VLLM_QUERIES_CACHE: LazyLock<Mutex<ModelQueriesCache>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A simple cache for SGLANG queries.
static SGLANG_QUERIES_CACHE: LazyLock<Mutex<ModelQueriesCache>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A simple cache for TEI queries.
static TEI_QUERIES_CACHE: LazyLock<Mutex<ModelQueriesCache>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A simple cache for Mistral-rs queries.
static MISTRALRS_QUERIES_CACHE: LazyLock<Mutex<ModelQueriesCache>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A simple cache for model queries.
type ModelQueriesCache = HashMap<String, Vec<(String, String)>>;

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeMetrics {
    /// A map with key as the model name (e.g. "meta-llama/Llama-3.3-70B-Instruct")
    /// and value as the model metrics.
    pub model_metrics: HashMap<String, ModelMetrics>,
}

/// Structure to store the usage metrics for a specific deployed model
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelMetrics {
    ChatCompletions(ChatCompletionsMetrics),
    Embeddings(EmbeddingsMetrics),
    ImageGeneration(ImageGenerationMetrics),
}

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct ChatCompletionsMetrics {
    /// GPU KV cache usage percentage,
    /// computed as the average over the previous \Delta time
    pub gpu_kv_cache_usage_perc: f64,

    /// Time to first token (prefill phase), in seconds,
    /// computed as the percentile 95 over the previous \Delta time
    pub time_to_first_token: f64,

    /// Time per output token (excluding the first token generation)
    /// in seconds, computed as the percentile 95 over the previous \Delta time
    pub time_per_output_token: f64,

    /// Number of requests running on the model,
    /// counted over the previous \Delta time
    pub num_running_requests: u32,

    /// Number of requests waiting to be processed,
    /// counted over the previous \Delta time
    pub num_waiting_requests: u32,

    /// Average request queue latency, in seconds,
    /// computed as the percentile 95 over the previous
    pub avg_request_queue_latency: Option<f64>,
}

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct EmbeddingsMetrics {
    /// Embeddings queue duration, in seconds,
    /// computed as the percentile 95 over the previous \Delta time
    pub embeddings_queue_duration: f64,

    /// Embeddings inference duration, in seconds,
    /// computed as the percentile 95 over the previous \Delta time
    pub embeddings_inference_duration: f64,

    /// Embeddings input length, in tokens,
    /// computed as the percentile 95 over the previous \Delta time
    pub embeddings_input_length: f64,

    /// Embeddings batch size, in tokens,
    /// computed as the percentile 95 over the previous \Delta time
    pub embeddings_batch_size: f64,

    /// Embeddings batch tokens, in tokens,
    /// computed as the percentile 95 over the previous \Delta time
    pub embeddings_batch_tokens: f64,
}

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageGenerationMetrics {
    /// Image generation latency, in seconds,
    /// computed as the percentile 95 over the previous \Delta time
    pub image_generation_latency: f64,

    /// Number of requests running on the model,
    /// counted over the previous \Delta time
    pub num_running_requests: u32,
}

/// Trait for setting metrics values for a specific model
trait MetricsCollector {
    /// Set the metrics value
    fn set_metrics_value(&mut self, query: &str, value: f64);
}

/// A macro to implement `MetricsCollector` for a given type.
macro_rules! impl_metrics_collector {
    ($metrics_type:ty, { $($query:expr => $setter:expr),+ $(,)? }) => {
        impl MetricsCollector for $metrics_type {
            #[instrument(
                level = "trace",
                target = "metrics",
                skip_all,
                fields(query = %query, value = %value)
            )]
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            fn set_metrics_value(&mut self, query: &str, value: f64) {
                match query {
                    $(
                        $query => ($setter)(self, value),
                    )+
                    _ => {
                        tracing::error!("Unknown query: {}", query);
                    }
                }
            }
        }
    };
}

impl_metrics_collector!(ChatCompletionsMetrics, {
    // VLLM metrics
    VLLM_TIME_TO_FIRST_TOKEN_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.time_to_first_token = value,
    VLLM_TIME_PER_OUTPUT_TOKEN_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.time_per_output_token = value,
    VLLM_GPU_CACHE_USAGE_PERC_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.gpu_kv_cache_usage_perc = value,
    VLLM_RUNNING_REQUESTS_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.num_running_requests = u32::try_from(value as i64).unwrap_or(0),
    VLLM_WAITING_REQUESTS_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.num_waiting_requests = u32::try_from(value as i64).unwrap_or(0),
    // SGLANG metrics
    SGLANG_TIME_TO_FIRST_TOKEN_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.time_to_first_token = value,
    SGLANG_TIME_PER_OUTPUT_TOKEN_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.time_per_output_token = value,
    SGLANG_RUNNING_REQUESTS_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.num_running_requests = u32::try_from(value as i64).unwrap_or(0),
    SGLANG_WAITING_REQUESTS_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.num_waiting_requests = u32::try_from(value as i64).unwrap_or(0),
    SGLANG_AVG_QUEUE_LATENCY_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.avg_request_queue_latency = Some(value),
});

impl_metrics_collector!(EmbeddingsMetrics, {
    TEI_EMBEDDINGS_QUEUE_DURATION_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.embeddings_queue_duration = value,
    TEI_EMBEDDINGS_INFERENCE_DURATION_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.embeddings_inference_duration = value,
    TEI_EMBEDDINGS_INPUT_LENGTH_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.embeddings_input_length = value,
    TEI_EMBEDDINGS_BATCH_SIZE_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.embeddings_batch_size = value,
    TEI_EMBEDDINGS_BATCH_TOKENS_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.embeddings_batch_tokens = value,
});

impl_metrics_collector!(ImageGenerationMetrics, {
    IMAGE_GENERATION_LATENCY_QUERY => |s: &mut ImageGenerationMetrics, value: f64| s.image_generation_latency = value,
    IMAGE_GENERATION_NUM_RUNNING_REQUESTS_QUERY => |s: &mut ImageGenerationMetrics, value: f64| s.num_running_requests = u32::try_from(value as i64).unwrap_or(0),
});

/// Response structure for Prometheus metrics queries
///
/// This struct represents the response format from the Prometheus HTTP API,
/// which includes a status field and the actual metrics data.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricsResponse {
    /// The status of the query response ("success" or "error")
    pub status: String,

    /// The actual metrics data returned by Prometheus
    pub data: MetricsData,
}

/// Container for Prometheus query results
///
/// This struct holds the result type (usually "vector") and an array of metric results.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricsData {
    /// The type of result returned ("vector", "matrix", "scalar", or "string")
    #[serde(rename = "resultType")]
    pub result_type: String,

    /// Vector of individual metric results
    pub result: Vec<MetricsResult>,
}

/// Individual metric result from a Prometheus query
///
/// Each result contains metric labels, a timestamp, and the metric value.
/// The timestamp and value are returned as a 2-element array: [timestamp, value]
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricsResult {
    /// Labels associated with the metric (e.g., {"instance": "localhost:9090"})
    pub metric: HashMap<String, String>,

    /// A 2-element array containing [timestamp, value]
    /// The timestamp is a Unix timestamp in seconds
    /// The value is a string that can be parsed into a number
    pub value: (i64, String),
}

/// Collects metrics for a specific model by executing multiple Prometheus queries concurrently
///
/// This function fetches various metrics from a Prometheus endpoint for a given model. It executes
/// multiple queries in parallel using async/await and aggregates the results into a single metrics
/// structure.
///
/// # Type Parameters
///
/// * `T` - The type of metrics to collect. Must implement `Default`, `MetricsCollector`, and `Send`.
///         Common types are `ChatCompletionsMetrics`, `EmbeddingsMetrics`, and `ImageGenerationMetrics`.
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to collect metrics
/// * `queries` - A slice of tuples containing (query_name, query_string) pairs to execute
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * The collected metrics of type `T`
/// * A `NodeMetricsError` if any query fails or metrics collection encounters an error
///
/// # Errors
///
/// This function will return an error if:
/// * Any HTTP request to the Prometheus endpoint fails
/// * The response cannot be parsed as valid metrics data
/// * The query execution fails on the Prometheus side
///
/// # Example
///
/// ```rust,ignore
/// use crate::metrics::{ChatCompletionsMetrics, collect_metrics};
///
/// async fn example() -> Result<ChatCompletionsMetrics, NodeMetricsError> {
///     let model_name = "gpt-3.5-turbo";
///     let queries = get_vllm_metrics_queries(model_name);
///     let endpoint = "http://localhost:9090/api/v1/query";
///
///     let metrics = collect_metrics::<ChatCompletionsMetrics>(
///         model_name,
///         &queries,
///         endpoint
///     ).await?;
///     Ok(metrics)
/// }
/// ```
#[instrument(
    level = "info",
    target = "metrics",
    skip_all,
    fields(model_name = model_name)
)]
pub async fn collect_metrics<T: Default + MetricsCollector + Send>(
    model_name: &str,
    queries: &[(String, String)],
) -> Result<T, NodeMetricsError> {
    let mut metrics = T::default();
    let mut futures: FuturesUnordered<_> = queries
        .iter()
        .map(|(_, query)| get_metrics(&HTTP_CLIENT, query))
        .collect();

    while let Some(result) = futures.next().await {
        match result {
            Ok((query, metric_value)) => {
                metrics.set_metrics_value(&query, metric_value);
            }
            Err(e) => {
                tracing::error!("Failed to fetch metrics with error: {}", e);
                return Err(NodeMetricsError::VllmMetricsError(
                    VllmMetricsError::UnknownQuery(format!("Unknown query, with error: {e}")),
                ));
            }
        }
    }

    Ok(metrics)
}

/// Computes metrics for all models deployed on the node by querying their respective metrics endpoints
///
/// This function aggregates metrics from different serving engines (e.g., vLLM) for each deployed model.
/// It queries various performance metrics like cache usage, request counts, and latency measurements
/// through Prometheus endpoints.
///
/// # Arguments
///
/// * `metrics_endpoints` - A HashMap mapping model names to tuples of (serving_engine, metrics_endpoint_url)
///   where:
///   - serving_engine: The type of serving engine (e.g., "vllm")
///   - metrics_endpoint_url: The URL of the Prometheus metrics endpoint for that model
///
/// # Returns
///
/// * `Result<NodeMetrics, NodeMetricsError>` - A Result containing either:
///   - `NodeMetrics`: Collection of metrics for all models on the node
///   - `NodeMetricsError`: Error that occurred during metrics collection
///
/// # Errors
///
/// Returns `NodeMetricsError` if:
/// - Metrics collection fails for any model
/// - HTTP requests to metrics endpoints fail
/// - Response parsing fails
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
///
/// let mut endpoints = HashMap::new();
/// endpoints.insert(
///     "llama-7b".to_string(),
///     ("vllm".to_string(), "http://localhost:8000/metrics".to_string())
/// );
///
/// let metrics = compute_node_metrics(&endpoints).await?;
/// ```
#[instrument(
    level = "info",
    target = "metrics",
    skip_all,
    fields(metrics_endpoints = %metrics_endpoints.len())
)]
pub async fn compute_node_metrics(
    metrics_endpoints: &HashMap<String, (String, Vec<String>)>,
) -> Result<NodeMetrics, NodeMetricsError> {
    let futures =
        metrics_endpoints
            .iter()
            .map(|(model_name, (engine, jobs_for_model))| async move {
                match engine.as_str() {
                    VLLM => {
                        let queries = get_cached_vllm_metrics_queries(model_name, jobs_for_model);
                        let metrics =
                            collect_metrics::<ChatCompletionsMetrics>(model_name, &queries).await?;
                        Ok((
                            model_name.to_string(),
                            ModelMetrics::ChatCompletions(metrics),
                        ))
                    }
                    TEI => {
                        let queries = get_cached_tei_metrics_queries(model_name);
                        let metrics =
                            collect_metrics::<EmbeddingsMetrics>(model_name, &queries).await?;
                        Ok((model_name.to_string(), ModelMetrics::Embeddings(metrics)))
                    }
                    MISTRALRS => {
                        let queries = get_cached_image_generation_metrics_queries(model_name);
                        let metrics =
                            collect_metrics::<ImageGenerationMetrics>(model_name, &queries).await?;
                        Ok((
                            model_name.to_string(),
                            ModelMetrics::ImageGeneration(metrics),
                        ))
                    }
                    SGLANG => {
                        let queries = get_cached_sglang_metrics_queries(model_name, jobs_for_model);
                        let metrics =
                            collect_metrics::<ChatCompletionsMetrics>(model_name, &queries).await?;
                        Ok((
                            model_name.to_string(),
                            ModelMetrics::ChatCompletions(metrics),
                        ))
                    }
                    _ => Err(NodeMetricsError::VllmMetricsError(
                        VllmMetricsError::UnknownQuery(format!("Unknown serving engine: {engine}")),
                    )),
                }
            });
    let results = futures::future::try_join_all(futures).await?;
    let model_metrics = results.into_iter().collect();

    Ok(NodeMetrics { model_metrics })
}

/// Fetches metrics from a Prometheus endpoint for a specific query
///
/// This function makes an HTTP GET request to a Prometheus endpoint with a specific query
/// and returns the parsed metrics data. It includes a 1-second timeout for the request.
///
/// # Arguments
///
/// * `client` - A reqwest HTTP client for making the request
/// * `query` - The Prometheus query string to execute
/// * `endpoint` - The URL of the Prometheus endpoint
///
/// # Returns
///
/// Returns a `Result` containing either:
/// * `(String, MetricsData)` - A tuple of the query string and the parsed metrics data
/// * `NodeMetricsError` - An error that occurred during the request or parsing
///
/// # Errors
///
/// This function will return an error if:
/// * The HTTP request fails (connection issues, timeout, etc.)
/// * The response cannot be parsed as JSON
/// * The Prometheus query fails (status != "success")
///
/// # Example
///
/// ```rust,ignore
/// use reqwest;
///
/// let client = reqwest::Client::new();
/// let query = "up";
/// let endpoint = "http://localhost:9090/api/v1/query";
///
/// let (query, metrics) = get_metrics(&client, query, endpoint).await?;
/// ```
async fn get_metrics(client: &Client, query: &str) -> Result<(String, f64), NodeMetricsError> {
    let result = client.query(query).get().await?;
    let value = result
        .data()
        .as_vector()
        .ok_or_else(|| VllmMetricsError::NoMetricsFound(query.to_string()))
        .and_then(|vector| {
            vector
                .first()
                .ok_or_else(|| VllmMetricsError::NoMetricsFound(query.to_string()))
        })
        .and_then(|instant| {
            instant
                .sample()
                .value()
                .to_string()
                .parse::<f64>()
                .map_err(VllmMetricsError::ParseError)
        })?;
    Ok((query.to_string(), value))
}

/// Gets the cached vLLM metrics queries for a specific model
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to get the queries
/// * `jobs_for_model` - A vector of strings representing the jobs for which to get the queries
///
/// # Returns
///
/// Returns the cached queries for the given model
fn get_cached_vllm_metrics_queries(
    model_name: &str,
    jobs_for_model: &[String],
) -> Vec<(String, String)> {
    let mut cache = VLLM_QUERIES_CACHE.lock().unwrap();
    if let Some(queries) = cache.get(model_name) {
        return queries.clone();
    }
    let queries = get_vllm_metrics_queries(jobs_for_model);
    cache.insert(model_name.to_string(), queries.clone());
    queries
}

/// Gets the cached SGLANG metrics queries for a specific model
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to get the queries
///
/// # Returns
///
/// Returns the cached queries for the given model
///
/// # Example
///
/// ```rust,ignore
/// let model_name = "llama-7b";
/// let queries = get_cached_sglang_metrics_queries(model_name);
/// ```
fn get_cached_sglang_metrics_queries(
    model_name: &str,
    jobs_for_model: &[String],
) -> Vec<(String, String)> {
    let mut cache = SGLANG_QUERIES_CACHE.lock().unwrap();
    if let Some(queries) = cache.get(model_name) {
        return queries.clone();
    }
    let queries = get_sglang_metrics_queries(jobs_for_model);
    cache.insert(model_name.to_string(), queries.clone());
    queries
}

/// Generates Prometheus queries for vLLM metrics for a specific model
///
/// This function creates a vector of Prometheus query strings to fetch metrics related to
/// vLLM performance and resource usage. The queries include measurements for:
/// - Time to first token (prefill phase latency)
/// - Time per output token (excluding first token)
/// - GPU KV cache usage percentage
/// - Number of currently running requests
/// - Number of waiting requests
///
/// The metrics are averaged over a time window defined by `METRICS_DELTA_TIME`.
/// Each query includes an "or 0" fallback to handle cases where no data is available.
///
/// # Arguments
///
/// * `jobs` - A vector of strings representing the jobs for which to generate metrics queries
///
/// # Returns
///
/// Returns a vector of tuples, where each tuple contains:
/// - A query identifier string (defined in constants)
/// - The corresponding Prometheus query string
fn get_vllm_metrics_queries(jobs: &[String]) -> Vec<(String, String)> {
    let delta = format!("[{METRICS_DELTA_TIME}s:]");
    let jobs = jobs.join("|");

    vec![
        (VLLM_TIME_TO_FIRST_TOKEN_QUERY.to_string(),
         format!("histogram_quantile(0.90, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{{job=~\"{jobs}\"}}{delta})))")),
        (VLLM_TIME_PER_OUTPUT_TOKEN_QUERY.to_string(),
         format!("histogram_quantile(0.90, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket{{job=~\"{jobs}\"}}{delta})))")),
        (VLLM_GPU_CACHE_USAGE_PERC_QUERY.to_string(),
         format!("avg_over_time(avg(vllm:gpu_cache_usage_perc{{job=~\"{jobs}\"}}{delta}))")),
        (VLLM_RUNNING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(sum(vllm:num_requests_running{{job=~\"{jobs}\"}}{delta}))")),
        (VLLM_WAITING_REQUESTS_QUERY.to_string(),
         format!("histogram_quantile(0.90, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{{job=~\"{jobs}\"}}{delta})))")),
    ]
}

/// Generates Prometheus queries for SGLANG metrics for a specific model
///
/// This function creates a vector of Prometheus query strings to fetch metrics related to
/// SGLANG performance and resource usage. The queries include measurements for:
/// - Time to first token (prefill phase latency)
/// - Time per output token (excluding first token)
/// - Number of currently running requests
/// - Number of waiting requests
/// - Average request queue latency
///
/// The metrics are averaged over a time window defined by `METRICS_DELTA_TIME`.
/// Each query includes an "or 0" fallback to handle cases where no data is available.
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to generate metrics queries
///
/// # Returns
///
/// Returns a vector of tuples, where each tuple contains:
/// - A query identifier string (defined in constants)
/// - The corresponding Prometheus query string
///
/// # Example
///
/// ```rust,ignore
/// let model_name = "llama-7b";
/// let queries = get_vllm_metrics_queries(model_name);
/// // Returns queries like:
/// // rate(vllm:time_to_first_token_seconds_sum{model_name="llama-7b"}[30m]) /
/// // rate(vllm:time_to_first_token_seconds_count{model_name="llama-7b"}[30m])
fn get_sglang_metrics_queries(jobs: &[String]) -> Vec<(String, String)> {
    let delta = format!("[{METRICS_DELTA_TIME}s:]");
    let jobs = jobs.join("|");

    vec![
        (SGLANG_TIME_TO_FIRST_TOKEN_QUERY.to_string(),
         format!("histogram_quantile(0.90, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket{{job=~\"{jobs}\"}}{delta})))")),
        (SGLANG_TIME_PER_OUTPUT_TOKEN_QUERY.to_string(),
         format!("histogram_quantile(0.90, sum by (le) (rate(sglang:inter_token_latency_seconds_bucket{{job=~\"{jobs}\"}}{delta})))")),
        (SGLANG_RUNNING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(sum(sglang:num_running_reqs{{job=~\"{jobs}\"}}{delta}))")),
        (SGLANG_WAITING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(sum(sglang:num_queue_reqs{{job=~\"{jobs}\"}}{delta}))")),
        (SGLANG_AVG_QUEUE_LATENCY_QUERY.to_string(),
         format!("histogram_quantile(0.90, sum by (le) (rate(sglang:avg_request_queue_latency_bucket{{job=~\"{jobs}\"}}{delta})))")),
    ]
}

/// Gets the cached TEI metrics queries for a specific model
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to get the queries
///
/// # Returns
///
/// Returns the cached queries for the given model
fn get_cached_tei_metrics_queries(model_name: &str) -> Vec<(String, String)> {
    let mut cache = TEI_QUERIES_CACHE.lock().unwrap();
    if let Some(queries) = cache.get(model_name) {
        return queries.clone();
    }
    let queries = get_tei_metrics_queries();
    cache.insert(model_name.to_string(), queries.clone());
    queries
}

/// Generates Prometheus queries for TEI (Text Embeddings Interface) metrics for a specific model
///
/// This function creates a vector of Prometheus query strings to fetch metrics related to
/// text embeddings performance and usage. The queries include measurements for:
/// - Embeddings queue duration, in seconds,
///   computed as the percentile 95 over the previous \Delta time
/// - Embeddings inference duration, in seconds,
///   computed as the percentile 95 over the previous \Delta time
/// - Embeddings input length, in tokens,
///   computed as the percentile 95 over the previous \Delta time
/// - Embeddings batch size, in tokens,
///   computed as the percentile 95 over the previous \Delta time
/// - Embeddings batch tokens, in tokens,
///   computed as the percentile 95 over the previous \Delta time
///
/// The metrics are averaged over a time window defined by `METRICS_DELTA_TIME`.
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to generate metrics queries
///
/// # Returns
///
/// Returns a vector of tuples, where each tuple contains:
/// - A query identifier string (defined in constants)
/// - The corresponding Prometheus query string
///
/// # Example
///
/// ```rust,ignore
/// let model_name = "all-MiniLM-L6-v2";
/// let queries = get_tei_metrics_queries(model_name);
/// // Returns queries like:
/// // rate(atoma_text_embeddings_latency{model_name="all-MiniLM-L6-v2"}[30m]) /
/// // rate(atoma_text_embeddings_latency_count{model_name="all-MiniLM-L6-v2"}[30m])
/// ```
fn get_tei_metrics_queries() -> Vec<(String, String)> {
    let delta = format!("[{METRICS_DELTA_TIME}s:]");

    vec![
        (TEI_EMBEDDINGS_QUEUE_DURATION_QUERY.to_string(),
         format!("histogram_quantile(0.95, sum(rate(te_queue_duration_bucket{delta})) by (le)) or 0")),
        (TEI_EMBEDDINGS_INFERENCE_DURATION_QUERY.to_string(),
         format!("histogram_quantile(0.95, sum(rate(te_inference_duration_bucket{delta})) by (le)) or 0")),
        (TEI_EMBEDDINGS_INPUT_LENGTH_QUERY.to_string(),
         format!("histogram_quantile(0.95, sum(rate(te_request_input_length_bucket{delta})) by (le)) or 0")),
        (TEI_EMBEDDINGS_BATCH_SIZE_QUERY.to_string(),
         format!("histogram_quantile(0.95, sum(rate(te_batch_next_size_bucket{delta})) by (le)) or 0")),
        (TEI_EMBEDDINGS_BATCH_TOKENS_QUERY.to_string(),
         format!("histogram_quantile(0.95, sum(rate(te_batch_next_tokens_bucket{delta})) by (le)) or 0")),
    ]
}

/// Gets the cached Mistral-rs metrics queries for a specific model
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to get the queries
///
/// # Returns
///
/// Returns the cached queries for the given model
fn get_cached_image_generation_metrics_queries(model_name: &str) -> Vec<(String, String)> {
    let mut cache = MISTRALRS_QUERIES_CACHE.lock().unwrap();
    if let Some(queries) = cache.get(model_name) {
        return queries.clone();
    }
    let queries = get_image_generation_metrics_queries(model_name);
    cache.insert(model_name.to_string(), queries.clone());
    queries
}

/// Generates Prometheus queries for image generation metrics for a specific model
///
/// This function creates a vector of Prometheus query strings to fetch metrics related to
/// image generation performance and usage. The queries include measurements for:
/// - Image generation latency (average time to generate an image)
/// - Number of currently running requests
///
/// The metrics are averaged over a time window defined by `METRICS_DELTA_TIME`.
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to generate metrics queries
///
/// # Returns
///
/// Returns a vector of tuples, where each tuple contains:
/// - A query identifier string (defined in constants)
/// - The corresponding Prometheus query string
///
/// # Example
///
/// ```rust,ignore
/// let model_name = "stable-diffusion-xl";
/// let queries = get_image_generation_metrics_queries(model_name);
/// // Returns queries like:
/// // rate(mistral:image_generation_latency_seconds_sum{model_name="stable-diffusion-xl"}[30m]) /
/// // rate(mistral:image_generation_latency_seconds_count{model_name="stable-diffusion-xl"}[30m])
/// ```
fn get_image_generation_metrics_queries(model_name: &str) -> Vec<(String, String)> {
    let delta = format!("[{METRICS_DELTA_TIME}s:]");

    vec![
        (IMAGE_GENERATION_LATENCY_QUERY.to_string(),
         format!("rate(mistral:image_generation_latency_seconds_sum{{model_name=\"{model_name}\"}}{delta}) / rate(mistral:image_generation_latency_seconds_count{{model_name=\"{model_name}\"}}{delta}) or 0")),
        (IMAGE_GENERATION_NUM_RUNNING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(mistral:running_requests{{model_name=\"{model_name}\"}}{delta}) or 0")),
    ]
}

#[derive(Debug, Error)]
pub enum NodeMetricsError {
    #[error("Failed to fetch vLLM production metrics: {0}")]
    VllmMetricsError(#[from] VllmMetricsError),
    #[error("Failed to fetch SGLANG production metrics: {0}")]
    SglangMetricsError(#[from] SglangMetricsError),
    #[error("Request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("Prometheus error: {0}")]
    PrometheusError(#[from] prometheus_http_query::Error),
}

#[derive(Debug, Error)]
pub enum VllmMetricsError {
    #[error("Failed to parse Prometheus response: {0}")]
    ParseError(#[from] ParseFloatError),
    #[error("Unknown query: {0}")]
    UnknownQuery(String),
    #[error("Query failed with status: {0}")]
    QueryFailed(String),
    #[error("No metrics found for job: {0}")]
    NoMetricsFound(String),
}

#[derive(Debug, Error)]
pub enum SglangMetricsError {
    #[error("Failed to parse Prometheus response: {0}")]
    ParseError(#[from] ParseFloatError),
    #[error("Unknown query: {0}")]
    UnknownQuery(String),
    #[error("Query failed with status: {0}")]
    QueryFailed(String),
}
