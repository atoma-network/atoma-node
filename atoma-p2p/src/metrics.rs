use std::{collections::HashMap, num::ParseFloatError, sync::Mutex};

use futures::stream::{FuturesUnordered, StreamExt};
use once_cell::sync::Lazy;
use opentelemetry::{
    global,
    metrics::{Counter, Gauge, Histogram, Meter, UpDownCounter},
};
use reqwest;
use serde::{Deserialize, Serialize};
use sysinfo::Networks;
use thiserror::Error;
use tracing::instrument;

use crate::constants::{
    IMAGE_GENERATION_LATENCY_QUERY, IMAGE_GENERATION_NUM_RUNNING_REQUESTS_QUERY,
    TEI_EMBEDDINGS_LATENCY_QUERY, TEI_NUM_RUNNING_REQUESTS_QUERY, VLLM_CPU_CACHE_USAGE_PERC_QUERY,
    VLLM_GPU_CACHE_USAGE_PERC_QUERY, VLLM_RUNNING_REQUESTS_QUERY, VLLM_TIME_PER_OUTPUT_TOKEN_QUERY,
    VLLM_TIME_TO_FIRST_TOKEN_QUERY, VLLM_WAITING_REQUESTS_QUERY,
};

// Add global metrics
static GLOBAL_METER: Lazy<Meter> = Lazy::new(|| global::meter("atoma-node"));

/// Metrics delta time, in seconds
pub const METRICS_DELTA_TIME: std::time::Duration = std::time::Duration::from_secs(30);

/// Metrics timeout, in seconds
const METRICS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// vLLM serving engine
const VLLM: &str = "vllm";

/// TEI serving engine
const TEI: &str = "tei";

/// Mistral-rs serving engine
const MISTRALRS: &str = "mistralrs";

/// HTTP client for the node metrics queries
static HTTP_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .timeout(METRICS_TIMEOUT)
        .build()
        .expect("Failed to create HTTP client")
});

/// A simple cache for vLLM queries.
static VLLM_QUERIES_CACHE: Lazy<Mutex<ModelQueriesCache>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// A simple cache for TEI queries.
static TEI_QUERIES_CACHE: Lazy<Mutex<ModelQueriesCache>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// A simple cache for Mistral-rs queries.
static MISTRALRS_QUERIES_CACHE: Lazy<Mutex<ModelQueriesCache>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

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

    /// CPU KV cache usage percentage,
    /// computed as the average over the previous \Delta time
    pub cpu_kv_cache_usage_perc: f64,

    /// Time to first token (prefill phase), in seconds,
    /// computed as the percentille 95 over the previous \Delta time
    pub time_to_first_token: f64,

    /// Time per output token (excluding the first token generation)
    /// in seconds, computed as the percentille 95 over the previous \Delta time
    pub time_per_output_token: f64,

    /// Number of requests running on the model,
    /// counted over the previous \Delta time
    pub num_running_requests: u32,

    /// Number of requests waiting to be processed,
    /// counted over the previous \Delta time
    pub num_waiting_requests: u32,
}

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct EmbeddingsMetrics {
    /// Embeddings latency, in seconds,
    /// computed as the percentille 95 over the previous \Delta time
    pub embeddings_latency: f64,

    /// Number of requests running on the model,
    /// counted over the previous \Delta time
    pub num_running_requests: u32,
}

/// Structure to store the usage metrics for the node
///
/// This data is collected from the system and the GPU
/// to be sent across the p2p network, for efficient request routing.
#[derive(Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct ImageGenerationMetrics {
    /// Image generation latency, in seconds,
    /// computed as the percentille 95 over the previous \Delta time
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
    VLLM_TIME_TO_FIRST_TOKEN_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.time_to_first_token = value,
    VLLM_TIME_PER_OUTPUT_TOKEN_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.time_per_output_token = value,
    VLLM_GPU_CACHE_USAGE_PERC_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.gpu_kv_cache_usage_perc = value,
    VLLM_CPU_CACHE_USAGE_PERC_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.cpu_kv_cache_usage_perc = value,
    VLLM_RUNNING_REQUESTS_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.num_running_requests = u32::try_from(value as i64).unwrap_or(0),
    VLLM_WAITING_REQUESTS_QUERY => |s: &mut ChatCompletionsMetrics, value: f64| s.num_waiting_requests = u32::try_from(value as i64).unwrap_or(0),
});

impl_metrics_collector!(EmbeddingsMetrics, {
    TEI_EMBEDDINGS_LATENCY_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.embeddings_latency = value,
    TEI_NUM_RUNNING_REQUESTS_QUERY => |s: &mut EmbeddingsMetrics, value: f64| s.num_running_requests = u32::try_from(value as i64).unwrap_or(0),
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
/// * `endpoint` - The URL of the Prometheus metrics endpoint
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
    endpoint: &str,
) -> Result<T, NodeMetricsError> {
    let mut metrics = T::default();
    let mut futures: FuturesUnordered<_> = queries
        .iter()
        .map(|(_, query)| get_metrics(&HTTP_CLIENT, query, endpoint))
        .collect();

    while let Some(result) = futures.next().await {
        match result {
            Ok((query, data)) => {
                metrics.set_metrics_value(
                    &query,
                    data.result[0].value.1.parse::<f64>().map_err(|e| {
                        tracing::error!("Failed to parse metric value for {}: {}", query, e);
                        NodeMetricsError::VllmMetricsError(VllmMetricsError::ParseError(e))
                    })?,
                );
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
    metrics_endpoints: &HashMap<String, (String, String)>,
) -> Result<NodeMetrics, NodeMetricsError> {
    let futures = metrics_endpoints
        .iter()
        .map(|(model_name, (engine, endpoint))| async move {
            match engine.as_str() {
                VLLM => {
                    let queries = get_cached_vllm_metrics_queries(model_name);
                    let metrics =
                        collect_metrics::<ChatCompletionsMetrics>(model_name, &queries, endpoint)
                            .await?;
                    Ok((
                        model_name.to_string(),
                        ModelMetrics::ChatCompletions(metrics),
                    ))
                }
                TEI => {
                    let queries = get_cached_tei_metrics_queries(model_name);
                    let metrics =
                        collect_metrics::<EmbeddingsMetrics>(model_name, &queries, endpoint)
                            .await?;
                    Ok((model_name.to_string(), ModelMetrics::Embeddings(metrics)))
                }
                MISTRALRS => {
                    let queries = get_cached_image_generation_metrics_queries(model_name);
                    let metrics =
                        collect_metrics::<ImageGenerationMetrics>(model_name, &queries, endpoint)
                            .await?;
                    Ok((
                        model_name.to_string(),
                        ModelMetrics::ImageGeneration(metrics),
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
async fn get_metrics(
    client: &reqwest::Client,
    query: &str,
    endpoint: &str,
) -> Result<(String, MetricsData), NodeMetricsError> {
    let response: MetricsResponse = client
        .get(endpoint)
        .query(&[("query", query)])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok((query.to_string(), response.data))
}

/// Gets the cached vLLM metrics queries for a specific model
///
/// # Arguments
///
/// * `model_name` - The name of the model for which to get the queries
///
/// # Returns
///
/// Returns the cached queries for the given model
fn get_cached_vllm_metrics_queries(model_name: &str) -> Vec<(String, String)> {
    let mut cache = VLLM_QUERIES_CACHE.lock().unwrap();
    if let Some(queries) = cache.get(model_name) {
        return queries.clone();
    }
    let queries = get_vllm_metrics_queries(model_name);
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
/// - CPU KV cache usage percentage
/// - Number of currently running requests
/// - Number of waiting requests
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
/// ```
fn get_vllm_metrics_queries(model_name: &str) -> Vec<(String, String)> {
    let delta_minutes = METRICS_DELTA_TIME.as_secs_f64() / 60.;
    let delta = format!("[{delta_minutes}m]");

    vec![
        (VLLM_TIME_TO_FIRST_TOKEN_QUERY.to_string(),
         format!("(rate(vllm:time_to_first_token_seconds_sum{{model_name=\"{model_name}\"}}{delta}) / rate(vllm:time_to_first_token_seconds_count{{model_name=\"{model_name}\"}}{delta})) or 0")),
        (VLLM_TIME_PER_OUTPUT_TOKEN_QUERY.to_string(),
         format!("(rate(vllm:time_per_output_token_seconds_sum{{model_name=\"{model_name}\"}}{delta}) / rate(vllm:time_per_output_token_seconds_count{{model_name=\"{model_name}\"}}{delta})) or 0")),
        (VLLM_GPU_CACHE_USAGE_PERC_QUERY.to_string(),
         format!("avg_over_time(vllm:gpu_cache_usage_perc{{model_name=\"{model_name}\"}}{delta}) or 0")),
        (VLLM_CPU_CACHE_USAGE_PERC_QUERY.to_string(),
         format!("avg_over_time(vllm:cpu_cache_usage_perc{{model_name=\"{model_name}\"}}{delta}) or 0")),
        (VLLM_RUNNING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(vllm:running_requests{{model_name=\"{model_name}\"}}{delta}) or 0")),
        (VLLM_WAITING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(vllm:waiting_requests{{model_name=\"{model_name}\"}}{delta}) or 0")),
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
    let queries = get_tei_metrics_queries(model_name);
    cache.insert(model_name.to_string(), queries.clone());
    queries
}

/// Generates Prometheus queries for TEI (Text Embeddings Interface) metrics for a specific model
///
/// This function creates a vector of Prometheus query strings to fetch metrics related to
/// text embeddings performance and usage. The queries include measurements for:
/// - Embeddings latency (average time to generate embeddings)
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
/// let model_name = "all-MiniLM-L6-v2";
/// let queries = get_tei_metrics_queries(model_name);
/// // Returns queries like:
/// // rate(atoma_text_embeddings_latency{model_name="all-MiniLM-L6-v2"}[30m]) /
/// // rate(atoma_text_embeddings_latency_count{model_name="all-MiniLM-L6-v2"}[30m])
/// ```
fn get_tei_metrics_queries(model_name: &str) -> Vec<(String, String)> {
    let delta_minutes = METRICS_DELTA_TIME.as_secs_f64() / 60.0;
    let delta = format!("{delta_minutes}m");

    vec![
        (TEI_EMBEDDINGS_LATENCY_QUERY.to_string(),
         format!("rate(atoma_text_embeddings_latency{{model_name=\"{model_name}\"}}{delta}) / rate(atoma_text_embeddings_latency_count{{model_name=\"{model_name}\"}}{delta}) or 0")),
        (TEI_NUM_RUNNING_REQUESTS_QUERY.to_string(),
         format!("avg_over_time(atoma_text_embeddings_running_requests{{model_name=\"{model_name}\"}}{delta}) or 0")),
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
    let delta_minutes = METRICS_DELTA_TIME.as_secs_f64() / 60.0;
    let delta = format!("{delta_minutes}m");

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
    #[error("Request failed: {0}")]
    RequestError(#[from] reqwest::Error),
}

#[derive(Debug, Error)]
pub enum VllmMetricsError {
    #[error("Failed to parse Prometheus response: {0}")]
    ParseError(#[from] ParseFloatError),
    #[error("Unknown query: {0}")]
    UnknownQuery(String),
    #[error("Query failed with status: {0}")]
    QueryFailed(String),
}

/// Counter metric that tracks the total number of dial attempts.
///
/// This metric counts the number of dial attempts,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_dials_attempted`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: requests (count)
pub static TOTAL_DIALS_ATTEMPTED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_dials")
        .with_description("The total number of dials attempted")
        .with_unit("dials")
        .build()
});

/// Counter metric that tracks the total number of dial failures.
///
/// This metric counts the number of dial failures,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_dials_failed`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: dials (count)
pub static TOTAL_DIALS_FAILED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_dials_failed")
        .with_description("The total number of dials failed")
        .with_unit("dials")
        .build()
});

/// Gauge metric that tracks the total number of connections.
///
/// This metric counts the number of connections,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_connections`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: connections (count)
pub static TOTAL_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_connections")
        .with_description("The total number of connections")
        .with_unit("connections")
        .build()
});

/// Gauge metric that tracks the total number of peers connected.
///
/// This metric counts the number of peers connected,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_peers_connected`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: peers (count)
pub static PEERS_CONNECTED: Lazy<Gauge<i64>> = Lazy::new(|| {
    GLOBAL_METER
        .i64_gauge("peers_connected")
        .with_description("The number of peers connected")
        .with_unit("peers")
        .build()
});

/// Counter metric that tracks the total number of gossipsub subscriptions.
///
/// This metric counts the number of gossipsub subscriptions,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossipsub_subscriptions`
/// - Type: `UpDownCounter`
/// - Labels: `peer_id`
/// - Unit: `subscriptions` (count)
pub static TOTAL_GOSSIPSUB_SUBSCRIPTIONS: Lazy<UpDownCounter<i64>> = Lazy::new(|| {
    GLOBAL_METER
        .i64_up_down_counter("total_gossipsub_subscriptions")
        .with_description("The total number of gossipsub subscriptions")
        .with_unit("subscriptions")
        .build()
});

/// Counter metric that tracks the total number of invalid gossipsub messages received.
///
/// This metric counts the number of invalid gossipsub messages received,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_invalid_gossipsub_messages_received`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_INVALID_GOSSIPSUB_MESSAGES_RECEIVED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_invalid_gossipsub_messages_received")
        .with_description("The total number of invalid gossipsub messages received")
        .with_unit("messages")
        .build()
});

/// Counter metric that tracks the total number of gossipsub messages forwarded.
///
/// This metric counts the number of gossipsub messages forwarded,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossipsub_messages_forwarded`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_GOSSIPSUB_MESSAGES_FORWARDED: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_gossipsub_messages_forwarded")
        .with_description("The total number of gossipsub messages forwarded")
        .with_unit("messages")
        .build()
});

/// Counter metric that tracks the total number of gossipsub publishes.
///
/// This metric counts the number of gossipsub publishes,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossipsub_publishes`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_GOSSIPSUB_PUBLISHES: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_gossipsub_publishes")
        .with_description("The total number of gossipsub publishes")
        .with_unit("messages")
        .build()
});

/// Counter metric that tracks the total number of failed gossipsub messages.
///
/// This metric counts the number of failed gossipsub messages,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_failed_gossipsub_messages`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: messages (count)
pub static TOTAL_FAILED_GOSSIPSUB_PUBLISHES: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_failed_gossipsub_messages")
        .with_description("The total number of failed gossipsub messages")
        .with_unit("messages")
        .build()
});

/// Gauge metric that tracks the total number of incoming connections.
///
/// This metric counts the number of incoming connections,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_incoming_connections`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: connections (count)
pub static TOTAL_INCOMING_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_incoming_connections")
        .with_description("The total number of incoming connections")
        .with_unit("connections")
        .build()
});

/// Gauge metric that tracks the total number of outgoing connections.
///
/// This metric counts the number of outgoing connections,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_outgoing_connections`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: connections (count)
pub static TOTAL_OUTGOING_CONNECTIONS: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_outgoing_connections")
        .with_description("The total number of outgoing connections")
        .with_unit("connections")
        .build()
});

/// Counter metric that tracks the total number of mDNS discoveries.
///
/// This metric counts the number of mDNS discoveries,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_mdns_discoveries`
/// - Type: Counter
/// - Labels: `peer_id`
/// - Unit: discoveries (count)
pub static TOTAL_MDNS_DISCOVERIES: Lazy<Counter<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_counter("total_mdns_discoveries")
        .with_description("The total number of mDNS discoveries")
        .with_unit("discoveries")
        .build()
});

/// Gauge metric that tracks the total number of stream bandwidth.
///
/// This metric counts the number of stream bandwidth,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_stream_incoming_bandwidth`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: bytes (count)
pub static TOTAL_STREAM_INCOMING_BANDWIDTH: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_stream_bandwidth")
        .with_description("The total number of stream bandwidth")
        .with_unit("bytes")
        .build()
});

/// Gauge metric that tracks the total number of stream outgoing bandwidth.
///
/// This metric counts the number of stream outgoing bandwidth,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_stream_outgoing_bandwidth`
/// - Type: Gauge
/// - Labels: `peer_id`
/// - Unit: bytes (count)
pub static TOTAL_STREAM_OUTGOING_BANDWIDTH: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("total_stream_outgoing_bandwidth")
        .with_description("The total number of stream outgoing bandwidth")
        .with_unit("bytes")
        .build()
});

/// Histogram metric that tracks the histogram of gossip scores.
///
/// This metric counts the histogram of gossip scores,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_gossip_score_histogram`
/// - Type: Histogram
/// - Labels: `peer_id`
/// - Unit: score (count)
pub static GOSSIP_SCORE_HISTOGRAM: Lazy<Histogram<f64>> = Lazy::new(|| {
    GLOBAL_METER
        .f64_histogram("gossip_score_histogram")
        .with_description("The histogram of gossip scores")
        .with_unit("score")
        .build()
});

/// Gauge metric that tracks the size of the Kademlia routing table.
///
/// This metric counts the size of the Kademlia routing table,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
pub static KAD_ROUTING_TABLE_SIZE: Lazy<Gauge<u64>> = Lazy::new(|| {
    GLOBAL_METER
        .u64_gauge("kad_routing_table_size")
        .with_description("The size of the Kademlia routing table")
        .with_unit("size")
        .build()
});

/// Structure to store the network metrics.
///
/// This data is collected from the system
/// and is used to update the network metrics.
pub struct NetworkMetrics {
    networks: Networks,
    bytes_received: Gauge<u64>,
    bytes_transmitted: Gauge<u64>,
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        let networks = Networks::new_with_refreshed_list();

        let bytes_received = GLOBAL_METER
            .u64_gauge("total_bytes_received")
            .with_description("The total number of bytes received")
            .with_unit("bytes")
            .build();

        let bytes_transmitted = GLOBAL_METER
            .u64_gauge("total_bytes_transmitted")
            .with_description("The total number of bytes transmitted")
            .with_unit("bytes")
            .build();

        Self {
            networks,
            bytes_received,
            bytes_transmitted,
        }
    }
}

// Network metrics implementation
impl NetworkMetrics {
    pub fn update_metrics(&mut self) {
        self.networks.refresh(true);

        let total_received = self
            .networks
            .values()
            .map(sysinfo::NetworkData::total_received)
            .sum();

        let total_transmitted = self
            .networks
            .values()
            .map(sysinfo::NetworkData::total_transmitted)
            .sum();

        self.bytes_received.record(total_received, &[]); // Empty attributes array is fine since we're just recording a global metric
        self.bytes_transmitted.record(total_transmitted, &[]);
    }
}
