use crate::serving::sequence::RequestMetrics;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use thiserror::Error;
use tracing::instrument;

pub type LogProb = f32;

/// `LlmEngine` - An asynchronous worker which is responsible for
/// scheduling new requests. It is also responsible
/// to communicate with the `ModelExecutor` service to send new requests
/// for continuously batched AI inference
pub struct LlmEngine {}

impl LlmEngine {
    /// Constructor
    pub fn new() -> Self {
        Self {}
    }

    // Main loop -
    ///     1. Listens to incoming requests and adds these to the underlying
    ///         `Scheduler`.
    ///     2. Awaits until new outputs are generated from the `ModelExecutor`
    ///         service. It then processes the outputs to update the associated
    ///         `SequenceGroup` states and re-schedules new requests.
    ///     3. Sends finished `SequenceGroup` outputs to the Atoma's client
    ///         service.
    #[instrument(skip(self))]
    pub async fn run(self) -> Result<(), EngineError> {
        loop {}
    }
}

/// `RequestOutput` - Output of running AI inference over a `SequenceGroup`
#[derive(Debug)]
pub struct GenerateRequestOutput {
    /// Request id
    pub request_id: String,
    /// The `String` prompt
    pub prompt: String,
    /// Inference outputs
    pub inference_outputs: Vec<InferenceOutput>,
    /// Prompt token ids
    pub prompt_token_ids: Vec<u32>,
    /// Is finished
    pub is_finished: bool,
    /// Metrics
    pub metrics: Arc<RwLock<RequestMetrics>>,
}

/// `InferenceOutput` - Output of running AI inference on a given sequence group
#[derive(Debug)]
pub struct InferenceOutput {
    /// The index of the output in the request
    pub index: usize,
    /// The generated output text
    pub output_text: String,
    /// The token ids of the generated output text
    pub token_ids: Vec<u32>,
    /// The cumulative log probability of the generated
    /// output text
    pub cumulative_logprob: f32,
    /// The log probabilities of the top probability words at each
    /// position if the logprobs are requested
    pub logprobs: Vec<HashMap<u32, LogProb>>,
    /// The reason why the sequence is finished
    pub finish_reason: Option<String>,
    /// The stop token id that caused the completion
    /// to stop, None if the completion finished for some other reason
    /// including encountering the eos token
    pub stop_reason: Option<u32>,
}

#[derive(Debug, Error)]
pub enum EngineError {}
