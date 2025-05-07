/// Time to first token query
pub const VLLM_TIME_TO_FIRST_TOKEN_QUERY: &str = "vllm:time_to_first_token_seconds";

/// Time per output token query
pub const VLLM_TIME_PER_OUTPUT_TOKEN_QUERY: &str = "vllm:time_per_output_token_seconds";

/// GPU cache usage percentage query
pub const VLLM_GPU_CACHE_USAGE_PERC_QUERY: &str = "vllm:gpu_cache_usage_perc";

/// CPU cache usage percentage query
pub const VLLM_CPU_CACHE_USAGE_PERC_QUERY: &str = "vllm:cpu_cache_usage_perc";

/// Running requests query
pub const VLLM_RUNNING_REQUESTS_QUERY: &str = "vllm:running_requests";

/// Waiting requests query
pub const VLLM_WAITING_REQUESTS_QUERY: &str = "vllm:waiting_requests";

/// Embeddings queue duration query
pub const TEI_EMBEDDINGS_QUEUE_DURATION_QUERY: &str = "tei:embeddings_queue_duration";

/// Embeddings inference duration query
pub const TEI_EMBEDDINGS_INFERENCE_DURATION_QUERY: &str = "tei:embeddings_inference_duration";

/// Embeddings input length query
pub const TEI_EMBEDDINGS_INPUT_LENGTH_QUERY: &str = "tei:embeddings_input_length";

/// Embeddings batch size query
pub const TEI_EMBEDDINGS_BATCH_SIZE_QUERY: &str = "tei:embeddings_batch_size";

/// Embeddings batch tokens query
pub const TEI_EMBEDDINGS_BATCH_TOKENS_QUERY: &str = "tei:embeddings_batch_tokens";

/// Image generation latency query
pub const IMAGE_GENERATION_LATENCY_QUERY: &str = "mistral:image_generation_latency";

/// Number of running requests query
pub const IMAGE_GENERATION_NUM_RUNNING_REQUESTS_QUERY: &str = "mistral:running_requests";

/// One million, used for computing fiat amounts
pub const ONE_MILLION: u128 = 1_000_000;
