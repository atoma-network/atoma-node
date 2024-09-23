pub mod service;
pub mod types;

use std::{collections::HashMap, sync::LazyLock};

pub static MAX_CONTEXT_WINDOW_SIZE: LazyLock<HashMap<String, usize>> =
    LazyLock::new(|| HashMap::from_iter([("llama-v3-70b-instruct".into(), 4096)]));
