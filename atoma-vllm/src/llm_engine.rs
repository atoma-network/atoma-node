use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{info_span, Span};

use crate::{
    config::{CacheConfig, SchedulerConfig},
    policy::FcfsPolicy,
    scheduler::{Scheduler, SchedulerError},
    types::GenerateRequest,
    validation::{Validation, ValidationError},
};

/// `LlmEngine` - An LLM engine that receives requests and generates texts.
///
///    This is the main class for the atoma-vllm engine. It receives requests
///    from clients and generates texts from the LLM. It includes a tokenizer, a
///    language model (possibly distributed across multiple GPUs), and GPU memory
///    space allocated for intermediate states (aka KV cache). This class utilizes
///    iteration-level scheduling and efficient memory management to maximize the
///    serving throughput.
pub struct LlmEngine {
    /// The scheduler, which handles CPU and GPU memory allocation
    scheduler: Scheduler<FcfsPolicy>,
    /// Request validator
    validation: Validation,
    /// Blockchain event requests receiver
    request_receiver: mpsc::UnboundedReceiver<GenerateRequest>,
    /// Tracing span
    span: Span,
}

impl LlmEngine {
    /// Constructor
    pub fn new(
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        validation: Validation,
        request_receiver: mpsc::UnboundedReceiver<GenerateRequest>,
    ) -> Result<Self, EngineError> {
        let scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)?;
        Ok(Self {
            scheduler,
            validation,
            request_receiver,
            span: info_span!("llm-engine"),
        })
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] ValidationError),
}
