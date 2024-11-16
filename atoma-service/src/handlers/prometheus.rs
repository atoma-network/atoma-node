use once_cell::sync::Lazy;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};

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
/// - Buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_LATENCY_METRICS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_token_latency",
        "The latency of chat completion generation in seconds",
        &["model"],
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,],
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
/// - Buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_DECODING_TIME: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_decoding_time",
        "Time taken for the complete decoding phase  in seconds",
        &["model"],
        vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
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
/// - Buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
pub static CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "atoma_chat_completions_time_to_first_token",
        "Time taken until first token is generated in seconds",
        &["model"],
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )
    .unwrap()
});

/// Counter metric that tracks the total number of input tokens processed in chat completions.
///
/// This metric counts the cumulative number of tokens in the input prompts,
/// broken down by model type and stack ID. This helps monitor token usage and costs
/// across different models and client applications.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_input_tokens_metrics`
/// - Type: Counter
/// - Labels:
///   - `model`: The model used for completion
///   - `stack_id`: Identifier for the client application
/// - Unit: tokens (count)
pub static CHAT_COMPLETIONS_INPUT_TOKENS_METRICS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_chat_completions_input_tokens_metrics",
        "Total number of input tokens processed",
        &["model", "stack_id"] // prompt,
    )
    .unwrap()
});

pub static CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "atoma_chat_completions_output_tokens_metrics",
        "Total number of output tokens processed",
        &["model", "stack_id"] // completion,
    )
    .unwrap()
});
