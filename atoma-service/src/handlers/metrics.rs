use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter, UpDownCounter},
};
use std::sync::LazyLock;

// Add global metrics
static GLOBAL_METER: LazyLock<Meter> = LazyLock::new(|| global::meter("atoma-node"));

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
pub static CHAT_COMPLETIONS_NUM_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_chat_completions_num_requests")
        .with_description("The number of incoming requests for chat completions tasks")
        .with_unit("s")
        .build()
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
pub static CHAT_COMPLETIONS_DECODING_TIME: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    GLOBAL_METER
        .f64_histogram("atoma_chat_completions_decoding_time")
        .with_description("Time taken for the complete decoding phase in seconds")
        .with_unit("s")
        .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
        .build()
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
pub static CHAT_COMPLETIONS_INPUT_TOKENS_METRICS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_chat_completions_input_tokens_metrics")
        .with_description("Total number of input tokens processed")
        .with_unit("tokens")
        .build()
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
pub static CHAT_COMPLETIONS_INTER_TOKEN_GENERATION_TIME: LazyLock<Histogram<f64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .f64_histogram("atoma_chat_completions_intra_token_generation_time")
            .with_description("Time taken to stream between each token generation phase in seconds")
            .with_unit("s")
            .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
            .build()
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
pub static CHAT_COMPLETIONS_TIME_TO_FIRST_TOKEN: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    GLOBAL_METER
        .f64_histogram("atoma_chat_completions_time_to_first_token")
        .with_description("Time taken until first token is generated in seconds")
        .with_unit("s")
        .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
        .build()
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
pub static CHAT_COMPLETIONS_OUTPUT_TOKENS_METRICS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_chat_completions_output_tokens_metrics")
        .with_description("Total number of output tokens processed")
        .with_unit("tokens")
        .build()
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
pub static IMAGE_GEN_NUM_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_image_gen_num_requests")
        .with_description("The number of incoming requests for image generation tasks")
        .with_unit("requests")
        .build()
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
pub static TEXT_EMBEDDINGS_NUM_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_text_embs_num_requests")
        .with_description("The number of incoming requests for text embeddings tasks")
        .with_unit("requests")
        .build()
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
/// - Labels: `privacy_level`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_LATENCY_METRICS: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    GLOBAL_METER
        .f64_histogram("atoma_chat_completions_token_latency")
        .with_description("The latency of chat completion generation in seconds")
        .with_unit("s")
        .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
        .build()
});

/// Histogram metric that tracks the latency of chat completion streaming token generation.
///
/// This metric measures the time taken to generate each token during chat completions,
/// broken down by model type. The histogram buckets range from 10ms to 10 minutes to
/// capture both fast and slow token generation scenarios.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_token_latency`
/// - Type: Histogram
/// - Labels: `model`
/// - Labels: `privacy_level`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static CHAT_COMPLETIONS_STREAMING_LATENCY_METRICS: LazyLock<Histogram<f64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .f64_histogram("atoma_chat_completions_token_latency")
            .with_description("The latency of chat completion generation in seconds")
            .with_unit("s")
            .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
            .build()
    });

/// Counter metric that tracks the number of times the chat completions service is unavailable.
///
/// This metric counts the number of times the chat completions service is unavailable,
/// broken down by model type. This helps monitor the availability of the chat completions service.
///
/// # Metric Details
/// - Name: `atoma_chat_completions_service_unavailable`
/// - Description: The number of times the chat completions service is unavailable
/// - Unit: requests (count)
/// - Labels: `model`
/// - Type: Counter
pub static CHAT_COMPLETIONS_TOO_MANY_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_chat_completions_too_many_requests")
        .with_description("The number of times the chat completions service is unavailable")
        .with_unit("requests")
        .build()
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
/// - Labels: `privacy_level`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static IMAGE_GEN_LATENCY_METRICS: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    GLOBAL_METER
        .f64_histogram("atoma_image_generation_latency")
        .with_description("The latency of image generation in seconds")
        .with_unit("s")
        .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
        .build()
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
/// - Labels: `privacy_level`
/// - Unit: seconds
/// - Buckets: [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
pub static TEXT_EMBEDDINGS_LATENCY_METRICS: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    GLOBAL_METER
        .f64_histogram("atoma_text_embeddings_latency")
        .with_description("The latency of text embeddings in seconds")
        .with_unit("s")
        .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
        .build()
});

/// Counter metric that tracks the total number of tokens processed in chat completions.
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
pub static CHAT_COMPLETIONS_ESTIMATED_TOTAL_TOKENS: LazyLock<UpDownCounter<i64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .i64_up_down_counter("atoma_chat_completions_estimated_total_tokens")
            .with_description("The estimated total number of tokens processed")
            .with_unit("tokens")
            .build()
    });

/// Counter metrics that tracks the total number of successfully completed requests (including chat completions, image generation, and text embeddings)
///
/// # Metric Details
/// - Name: `atoma_total_completed_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_COMPLETED_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_completed_requests")
        .with_description("Total number of successfully completed requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of failed requests (including chat completions, image generation, and text embeddings)
///
/// # Metric Details
/// - Name: `atoma_total_failed_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_failed_requests")
        .with_description("Total number of failed requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of failed chat requests.
///
/// # Metric Details
/// - Name: `atoma_total_failed_chat_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_CHAT_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_failed_chat_requests")
        .with_description("Total number of failed chat requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of too many requests.
///
/// # Metric Details
/// - Name: `atoma_total_too_many_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_TOO_MANY_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_too_many_requests")
        .with_description("Total number of too many requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of unauthorized requests.
///
/// # Metric Details
/// - Name: `atoma_total_unauthorized_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_UNAUTHORIZED_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_unauthorized_requests")
        .with_description("Total number of unauthorized requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of too early requests.
///
/// # Metric Details
/// - Name: `atoma_total_too_early_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_TOO_EARLY_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_too_early_requests")
        .with_description("Total number of too early requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of locked requests.
///
/// # Metric Details
/// - Name: `atoma_total_locked_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_LOCKED_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_locked_requests")
        .with_description("Total number of locked requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of bad request requests.
///
/// # Metric Details
/// - Name: `atoma_total_bad_request_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_BAD_REQUEST_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_bad_request_requests")
        .with_description("Total number of bad request requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of confidential chat requests.
///
/// # Metric Details
/// - Name: `atoma_total_confidential_chat_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static CHAT_COMPLETIONS_CONFIDENTIAL_NUM_REQUESTS: LazyLock<Counter<u64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .u64_counter("atoma_total_confidential_chat_requests")
            .with_description("Total number of confidential chat requests")
            .with_unit("requests")
            .build()
    });

/// Counter metric that tracks the total number of failed confidential chat requests.
///
/// # Metric Details
/// - Name: `atoma_total_failed_confidential_chat_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_CHAT_CONFIDENTIAL_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_failed_confidential_chat_requests")
        .with_description("Total number of failed confidential chat requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of failed image generation requests.
///
/// # Metric Details
/// - Name: `atoma_total_failed_image_generation_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_IMAGE_GENERATION_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_failed_image_generation_requests")
        .with_description("Total number of failed image generation requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of failed image generation confidential requests.
///
/// # Metric Details
/// - Name: `atoma_total_failed_image_confidential_generation_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_IMAGE_CONFIDENTIAL_GENERATION_REQUESTS: LazyLock<Counter<u64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .u64_counter("atoma_total_failed_image_confidential_generation_requests")
            .with_description("Total number of failed image generation confidential requests")
            .with_unit("requests")
            .build()
    });

/// Counter metric that tracks the total number of image generation confidential requests.
///
/// # Metric Details
/// - Name: `atoma_image_generation_confidential_num_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static IMAGE_GEN_CONFIDENTIAL_NUM_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_image_generation_confidential_num_requests")
        .with_description("Total number of image generation confidential requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of failed text embedding requests.
///
/// # Metric Details
/// - Name: `atoma_total_failed_text_embedding_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_TEXT_EMBEDDING_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    GLOBAL_METER
        .u64_counter("atoma_total_failed_text_embedding_requests")
        .with_description("Total number of failed text embedding requests")
        .with_unit("requests")
        .build()
});

/// Counter metric that tracks the total number of text embedding confidential requests.
///
/// # Metric Details
/// - Name: `atoma_text_embeddings_confidential_num_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TEXT_EMBEDDINGS_CONFIDENTIAL_NUM_REQUESTS: LazyLock<Counter<u64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .u64_counter("atoma_text_embeddings_confidential_num_requests")
            .with_description("Total number of text embedding confidential requests")
            .with_unit("requests")
            .build()
    });

/// Counter metric that tracks the total number of failed text embedding confidential requests.
///
/// # Metric Details
/// - Name: `atoma_total_failed_text_embedding_confidential_requests`
/// - Type: Counter
/// - Labels: `model`
/// - Unit: requests (count)
pub static TOTAL_FAILED_TEXT_EMBEDDING_CONFIDENTIAL_REQUESTS: LazyLock<Counter<u64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .u64_counter("atoma_total_failed_text_embedding_confidential_requests")
            .with_description("Total number of failed text embedding confidential requests")
            .with_unit("requests")
            .build()
    });

/// Counter metric that tracks successful verify_permissions middleware time.
///
/// This metric measures the time taken by verify_permissions middleware to process requests,
/// broken down by model type. The histogram buckets range from 0.1ms to 30 seconds to
/// capture both very fast and slow verify_permissions middleware processing scenarios.
///
/// # Metric Details
/// - Name: `atoma_verify_permissions_middleware_time`
/// - Type: Histogram
/// - Labels: `model`
/// - Labels: `privacy_level`
/// - Unit: seconds
pub static VERIFY_PERMISSIONS_MIDDLEWARE_SUCCESSFUL_TIME: LazyLock<Histogram<f64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .f64_histogram("atoma_verify_permissions_middleware_successful_time")
            .with_description(
                "Time taken by verify_permissions middleware to process requests in seconds",
            )
            .with_unit("s")
            .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
            .build()
    });

/// Counter metric that tracks successful confidential_compute_middleware middleware time.
///
/// This metric measures the time taken by confidential_compute_middleware middleware to process requests,
/// broken down by model type. The histogram buckets range from 0.1ms to 30 seconds to
/// capture both very fast and slow confidential_compute_middleware middleware processing scenarios.
///
/// # Metric Details
/// - Name: `atoma_confidential_compute_middleware_middleware_time`
/// - Type: Histogram
/// - Labels: `model`
/// - Labels: `privacy_level`
/// - Unit: seconds
pub static CONFIDENTIAL_COMPUTE_MIDDLEWARE_SUCCESSFUL_TIME: LazyLock<Histogram<f64>> =
    LazyLock::new(|| {
        GLOBAL_METER
            .f64_histogram("atoma_confidential_compute_middleware_successful_time")
            .with_description(
                "Time taken by confidential_compute_middleware middleware to process requests in seconds",
            )
            .with_unit("s")
            .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
            .build()
    });

/// Counter metric that tracks successful signature_verification_middleware middleware time.
///
/// This metric measures the time taken by signature_verification_middleware middleware to process requests,
/// broken down by model type. The histogram buckets range from 0.1ms to 30 seconds to
/// capture both very fast and slow signature_verification_middleware middleware processing scenarios.
///
/// # Metric Details
/// - Name: `atoma_signature_verification_middleware_middleware_time`
/// - Type: Histogram
/// - Labels: `model`
/// - Labels: `privacy_level`
/// - Unit: seconds
pub static SIGNATURE_VERIFICATION_MIDDLEWARE_SUCCESSFUL_TIME: LazyLock<Histogram<f64>> =
    LazyLock::new(|| {
        GLOBAL_METER
        .f64_histogram("atoma_signature_verification_middleware_successful_time")
        .with_description(
            "Time taken by signature_verification_middleware middleware to process requests in seconds",
        )
        .with_unit("s")
        .with_boundaries(LATENCY_HISTOGRAM_BUCKETS.to_vec())
        .build()
    });
