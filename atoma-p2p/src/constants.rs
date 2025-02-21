pub const QUERIES: [(&str, &str); 8] = [
    ("chat_latency", "rate(atoma_chat_completions_token_latency_sum[5m]) / rate(atoma_chat_completions_token_latency_count[5m])"),
        ("first_token_time", "rate(atoma_chat_completions_time_to_first_token_sum[5m]) / rate(atoma_chat_completions_time_to_first_token_count[5m])"),
        ("inter_token_time", "rate(atoma_chat_completions_intra_token_generation_time_sum[5m]) / rate(atoma_chat_completions_intra_token_generation_time_count[5m])"),
        ("decoding_time", "rate(atoma_chat_completions_decoding_time_sum[5m]) / rate(atoma_chat_completions_decoding_time_count[5m])"),
        ("image_gen_latency", "rate(atoma_image_generation_latency_sum[5m]) / rate(atoma_image_generation_latency_count[5m])"),
        ("text_emb_latency", "rate(atoma_text_embeddings_latency_sum[5m]) / rate(atoma_text_embeddings_latency_count[5m])"),
        ("total_requests", "sum(increase(atoma_total_completed_requests[5m]))"),
        ("failed_requests", "sum(increase(atoma_total_failed_requests[5m]))")
    ];

pub const PROMETHEUS_URL: &str = "http://localhost:9090";

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
