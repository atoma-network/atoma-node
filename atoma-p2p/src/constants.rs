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
