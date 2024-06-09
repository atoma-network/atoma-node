use tracing::{info_span, Span};

/// `LlmEngine` - An LLM engine that receives requests and generates texts.
///
///    This is the main class for the atoma-vllm engine. It receives requests
///    from clients and generates texts from the LLM. It includes a tokenizer, a
///    language model (possibly distributed across multiple GPUs), and GPU memory
///    space allocated for intermediate states (aka KV cache). This class utilizes
///    iteration-level scheduling and efficient memory management to maximize the
///    serving throughput.
pub struct LlmEngine {
    /// Tracing span
    span: Span,
}

impl LlmEngine {
    /// Constructor
    pub fn new() -> Self {
        Self {
            span: info_span!("llm-engine"),
        }
    }
}
