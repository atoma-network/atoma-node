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

/// Embeddings latency query
pub const TEI_EMBEDDINGS_LATENCY_QUERY: &str = "tei:embeddings_latency";

/// Number of running requests query
pub const TEI_NUM_RUNNING_REQUESTS_QUERY: &str = "tei:running_requests";

/// Image generation latency query
pub const IMAGE_GENERATION_LATENCY_QUERY: &str = "mistral:image_generation_latency";

/// Number of running requests query
pub const IMAGE_GENERATION_NUM_RUNNING_REQUESTS_QUERY: &str = "mistral:running_requests";
