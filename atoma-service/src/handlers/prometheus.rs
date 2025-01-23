use once_cell::sync::Lazy;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};

const LATENCY_HISTOGRAM_BUCKETS: [f64; 15] = [
    0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

/// Counter metric that tracks the total number of chat completion requests.
///
/// This metric counts the number of incoming requests for chat completions,
/// broken down by model type. This helps monitor usage patterns and load
/// across different models.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_num_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static CHAT_COMPLETIONS_NUM_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_chat_completions_num_requests",
        "The number of incoming requests for chat completions tasks",
        &["model"]
    )
    .unwrap()
});

/// Counter metric that tracks the total number of image generation requests.
///
/// This metric counts the number of incoming requests for image generations,
/// broken down by model type. This helps monitor usage patterns and load
/// across different image generation models.
///
/// # Metric Details
/// - Name: `atoma_image_gen_num_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static IMAGE_GEN_NUM_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_image_gen_num_requests",
        "The number of incoming requests for image generation tasks",
        &["model"]
    )
    .unwrap()
});

/// Counter metric that tracks the total number of text embedding requests.
///
/// This metric counts the number of incoming requests for text embeddings,
/// broken down by model type. This helps monitor usage patterns and load
/// across different embedding models.
///
/// # Metric Details
/// - Name: `atoma_text_embs_num_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TEXT_EMBEDDINGS_NUM_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_text_embs_num_requests",
        "The number of incoming requests for text embeddings tasks",
        &["model"]
    )
    .unwrap()
});

/// Histogram metric that tracks the latency of chat completion token generation.
///
/// This metric measures the time taken to generate each token during chat completions,
/// broken down by model type. The histogram buckets range from 10ms to 10 minutes to
/// capture both fast and slow token generation scenarios.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_token_latency`
/// - Type: Histogram
/// - Labels: `model`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_LATENCY_METRICS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_token_latency",
        "The latency of chat completion generation in seconds",
        &["model"],
        LATENCY_HISTOGRAM_BUCKETS.to_vec(),
    )
    .unwrap()
});

/// Histogram metric that tracks the latency of image generation requests.
///
/// This metric measures the time taken to generate images, broken down by model type.
/// The histogram buckets range from 1ms to 10 minutes to capture both fast and slow
/// generation scenarios.
///
/// # Metric Details
/// - Name: `atoma_image_generation_latency`
/// - Type: Histogram
/// - Labels: `model`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static IMAGE_GEN_LATENCY_METRICS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_image_generation_latency",
        "The latency of image generation in seconds",
        &["model"],
        LATENCY_HISTOGRAM_BUCKETS.to_vec(),
    )
    .unwrap()
});

/// Histogram metric that tracks the latency of text embedding requests.
///
/// This metric measures the time taken to generate text embeddings, broken down by model type.
/// The histogram buckets range from 1ms to 10 minutes to capture both fast and slow
/// embedding generation scenarios.
///
/// # Metric Details
/// - Name: `atoma_text_embeddings_latency`
/// - Type: Histogram
/// - Labels: `model`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static TEXT_EMBEDDINGS_LATENCY_METRICS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_text_embeddings_latency",
        "The latency of text embeddings in seconds",
        &["model"],
        LATENCY_HISTOGRAM_BUCKETS.to_vec(),
    )
    .unwrap()
});

/// Histogram metric that tracks the total time spent in the decoding phase of chat completions.
///
/// This metric measures the complete duration of the decoding phase for chat completions,
/// broken down by model type. The histogram buckets range from 100ms to 10 minutes to
/// capture both typical and extended decoding scenarios.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_decoding_time`
/// - Type: Histogram
/// - Labels: `model`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_DECODING_TIME: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_decoding_time",
        "Time taken for the complete decoding phase  in seconds",
        &["model"],
        LATENCY_HISTOGRAM_BUCKETS.to_vec(),
    )
    .unwrap()
});

/// Histogram metric that tracks the time until the first token is generated in chat completions.
///
/// This metric measures the initial latency before token generation begins,
/// broken down by model type. The histogram buckets range from 0.1ms to 30 seconds to
/// capture both very fast and slow initial response times.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_time_to_first_token`
/// - Type: Histogram
/// - Labels: `model`
/// - Unit: seconds
/// - Buckets: [[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_time_to_first_token",
        "Time taken until first token is generated in seconds",
        &["model"],
        LATENCY_HISTOGRAM_BUCKETS.to_vec(),
    )
    .unwrap()
});

/// Histogram metric that tracks the time taken between each token generation phase in chat completions.
///
/// This metric measures the time taken between each token generation phase,
/// broken down by model type. The histogram buckets range from 0.1ms to 30 seconds to
/// capture both very fast and slow intra-token generation scenarios.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_intra_token_generation_time`
/// - Type: Histogram
/// - Labels: `model`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_INTER_TOKEN_GENERATION_TIME: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_intra_token_generation_time",
        "Time taken to stream between each token generation phase in seconds",
        &["model"],
        LATENCY_HISTOGRAM_BUCKETS.to_vec(),
    )
    .unwrap()
});

/// Counter metric that tracks the total number of input tokens processed in chat completions.
///
/// This metric counts the cumulative number of tokens in the input prompts,
/// broken down by model type. This helps monitor token usage and costs
/// across different models and client applications.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_input_tokens_metrics`
/// - Type: Counter
/// - Labels:
///   - `model`: The model used for completion
/// - Unit: tokens (count)
pub static CHAT_COMPLETIONS_INPUT_TOKENS_METRICS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_chat_completions_input_tokens_metrics",
        "Total number of input tokens processed",
        &["model"] // prompt,
    )
    .unwrap()
});

/// Counter metric that tracks the total number of output tokens generated in chat completions.
///
/// This metric counts the cumulative number of tokens generated in the completions,
/// broken down by model type. This helps monitor token usage and costs
/// across different models and client applications.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_output_tokens_metrics`
/// - Type: Counter
/// - Labels:
///   - `model`: The model used for completion
/// - Unit: tokens (count)
pub static CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_chat_completions_output_tokens_metrics",
        "Total number of output tokens processed",
        &["model"] // completion,
    )
    .unwrap()
});

/// Counter metrics that tracks the total number of successfully completed requests (including chat completions, image generation, and text embeddings)
///
/// # Metric Details
/// - Name: `atoma_total_completed_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_COMPLETED_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_total_completed_requests",
        "Total number of successfully completed requests",
        &["model"]
    )
    .unwrap()
});

/// Counter metric that tracks the total number of failed requests (including chat completions, image generation, and text embeddings)
///
/// # Metric Details
/// - Name: `atoma_total_failed_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_total_failed_requests",
        "Total number of failed requests",
        &["model"]
    )
    .unwrap()
});
