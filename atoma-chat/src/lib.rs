pub mod service;
pub mod types;

use std::{collections::HashMap, sync::LazyLock};

pub static MAX_CONTEXT_WINDOW_SIZE: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    HashMap::from_iter([
        ("llama_v1".into(), 4096),
        ("llama_v2".into(), 4096),
        ("llama_solar_10_7b".into(), 4096),
        ("llama_tiny_llama_1_1b_chat".into(), 4096),
        ("llama3_8b".into(), 4096),
        ("llama3_instruct_8b".into(), 4096),
        ("llama3_70b".into(), 4096),
        ("llama31_8b".into(), 4096),
        ("llama31_instruct8b".into(), 4096),
        ("llama31_70b".into(), 4096),
        ("llama31_instruct70b".into(), 4096),
        ("llama31_405b".into(), 4096),
        ("llama31_instruct405b".into(), 4096),
        ("mamba_130m".into(), 4096),
        ("mamba_370m".into(), 4096),
        ("mamba_790m".into(), 4096),
        ("mamba_1-4b".into(), 4096),
        ("mamba_2-8b".into(), 4096),
        ("mistral_7bv01".into(), 4096),
        ("mistral_7bv02".into(), 4096),
        ("mistral_7b-instruct-v01".into(), 4096),
        ("mistral_7b-instruct-v02".into(), 4096),
        ("mixtral_8x7b-v01".into(), 4096),
        ("mixtral_8x7b-instruct-v01".into(), 4096),
        ("mixtral_8x22b-v01".into(), 4096),
        ("mixtral_8x22b-instruct-v01".into(), 4096),
        ("phi_3-mini".into(), 4096),
        ("qwen_w0.5b".into(), 4096),
        ("qwen_w1.8b".into(), 4096),
        ("qwen_w4b".into(), 4096),
        ("qwen_w7b".into(), 4096),
        ("qwen_w14b".into(), 4096),
        ("qwen_w72b".into(), 4096),
        ("qwen_moe_a2.7b".into(), 4096),
    ])
});
