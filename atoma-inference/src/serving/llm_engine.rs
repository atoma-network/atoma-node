use futures::StreamExt;
use std::collections::HashMap;
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{error, info, info_span, instrument, Span};

use super::{
    model_executor::ModelThreadDispatcher,
    sequence::{
        ExecuteModelRequest, LogProb, RequestMetrics, Sequence, SequenceMetadata, SequenceOutput,
    },
};

/// `LlmEngine` - An asynchronous worker which is responsible for
/// scheduling new requests. It is also responsible
/// to communicate with the `ModelExecutor` service to send new requests
/// for continuously batched AI inference
pub struct LlmEngine {
    /// Atoma's client sender channel, to share newly AI
    /// generated outputs
    atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
    /// Dispatcher responsible to communicate with a
    /// model executor's  running thread, responsible
    /// for running prefill and decoding Inference
    /// to produce AI generated outputs
    model_thread_dispatcher: ModelThreadDispatcher,
    /// Receiver's channel responsible for receiving new
    /// requests from the running main `LlmService` instance
    request_receiver: UnboundedReceiver<Sequence>,
    /// Current scheduled `Sequence`'s metadata
    sequences_metadata: Vec<SequenceMetadata>,
    // /// Current `SchedulerOutputs`
    // scheduler_outputs: SchedulerOutputs,
    // /// A `Scheduler` instance
    // scheduler: Scheduler<FcfsPolicy>,
    /// Tokenizer for decoding sequences
    tokenizer: Tokenizer,
    /// Span
    span: Span,
}

impl LlmEngine {
    /// Constructor
    pub fn new(
        atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
        model_thread_dispatcher: ModelThreadDispatcher,
        request_receiver: UnboundedReceiver<Sequence>,
        // scheduler_outputs: SchedulerOutputs,
        // scheduler: Scheduler<FcfsPolicy>,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            atoma_client_sender,
            model_thread_dispatcher,
            request_receiver,
            sequences_metadata: vec![],
            // scheduler_outputs,
            // scheduler,
            tokenizer,
            span: info_span!("llm_engine"),
        }
    }

    // Main loop -
    ///     1. Listens to incoming requests and adds these to the underlying
    ///         `Scheduler`.
    ///     2. Awaits until new outputs are generated from the `ModelExecutor`
    ///         service. It then processes the outputs to update the associated
    ///         `Sequence` states and re-schedules new requests.
    ///     3. Sends finished `Sequence` outputs to the Atoma's client
    ///         service.
    #[instrument(skip(self))]
    pub async fn run(mut self) -> Result<(), EngineError> {
        loop {
            tokio::select! {
                Some(sequence_group) = self.request_receiver.recv() => {
                    // // 1. Adds the received `Sequence` to the `Scheduler` instance.
                    // self.scheduler.add_sequence_group(sequence_group);

                    // // 2. If the current `LlmInstance` doesn't have any on-going
                    // //    scheduled sequences, we wait some time and then
                    // //    schedule all the received requests so far.
                    // //    This includes the request added in 1.
                    // if self.sequences_metadata.is_empty() && self.scheduler_outputs.is_empty() {
                    //     self.step()?;
                    // }
                },
                Some(outputs) = self.model_thread_dispatcher.responses.next() => {
                    self.handle_outputs(outputs.map_err(EngineError::RecvError)).await?;
                }
                else => {
                    continue;
                }
            }
        }
    }

    /// Handles newly AI generated `SequenceOutput`'s
    #[instrument(skip_all)]
    async fn handle_outputs(
        &mut self,
        outputs: Result<Vec<SequenceOutput>, EngineError>,
    ) -> Result<(), EngineError> {
        match outputs {
            Ok(outputs) => {
                // // 1. Processes the newly AI generated outputs
                // let request_outputs = self.process_generated_outputs(outputs)?;

                // 2. Schedules new requests
                self.step()?;

                // // 3. After scheduling new requests to the `ModelExecutor`
                // //    we can send the finished outputs to the atoma client
                // //    service.
                // // NOTE: This is after scheduling new sequences above,
                // //    we do so to optimize GPU utilization. This is
                // //    supposed to be safe
                // if !request_outputs.is_empty() {
                //     self.atoma_client_sender.send(request_outputs)?;
                // }
            }
            Err(e) => {
                error!("Invalid generated outputs with error: {e}");
                // NOTE: In order to maintain the system live, we need to keep calling
                // the `self.step()` method, even in possible scenarios of failure.
                self.step()?;
            }
        }
        Ok(())
    }

    /// Main method of `LlmEngine`.
    ///
    /// 1. It is responsible for scheduling new
    ///     requests, via the associated `Scheduler`. Once scheduling is complete,
    ///
    /// 2. It sends a new `ExecuteModelRequest` to the `ModelExecutor`'s thread.
    #[instrument(skip_all)]
    pub fn step(&mut self) -> Result<(), EngineError> {
        // info!("`LlmEngine` new step..");
        // // 1. Schedule new requests
        // let (sequences_metadata, scheduler_outputs) = self.scheduler.schedule()?;

        // // 2. Update `self.sequences_metadata` and `scheduler_outputs`
        // self.sequences_metadata = sequences_metadata.clone();
        // self.scheduler_outputs = scheduler_outputs.clone();

        // // 3. If the scheduled data is empty, it means that
        // //     no new requests were received.
        // if scheduler_outputs.is_empty() {
        //     return Ok(());
        // }

        // let execute_model_request =
        //     ExecuteModelRequest::new(sequences_metadata, scheduler_outputs.running_queue_size);

        // // 4. Sends a new `ExecuteModelRequest` to the underlying `ModelExecutor`'s thread
        // self.model_thread_dispatcher.send(execute_model_request);

        Ok(())
    }
}

/// `RequestOutput` - Output of running AI inference over a `Sequence`
#[derive(Debug)]
pub struct GenerateRequestOutput {
    /// Request id
    pub request_id: u64,
    /// The `String` prompt
    pub prompt: String,
    /// Prompt token ids
    pub prompt_token_ids: Vec<u32>,
    /// Output text
    pub output_text: String,
    /// The token ids of the generated output text
    pub output_token_ids: Vec<u32>,
    /// The log probabilities of the top probability words at each
    /// position if the logprobs are requested
    pub logprobs: Vec<HashMap<u32, LogProb>>,
    /// The reason why the sequence is finished
    pub finish_reason: Option<String>,
    /// The stop token id that caused the completion
    /// to stop, None if the completion finished for some other reason
    /// including encountering the eos token
    pub stop_reason: Option<u32>,
    /// Is finished
    pub is_finished: bool,
    /// Metrics
    pub metrics: RequestMetrics,
}

impl GenerateRequestOutput {
    /// Creates a new `Self` instance from a `SequenceGroup`
    pub fn from_sequence(sequence: &Sequence) -> Self {
        Self {
            request_id: sequence.sequence_id,
            prompt: sequence.input.clone(),
            prompt_token_ids: sequence.sequence_data.prompt_token_ids.clone(),
            output_text: sequence.output_text.clone(),
            output_token_ids: sequence.sequence_data.output_token_ids.clone(),
            logprobs: sequence.output_logprobs.clone(),
            finish_reason: sequence.sequence_status.finished_reason(),
            is_finished: sequence.is_finished,
            stop_reason: sequence.stop_reason,
            metrics: sequence.metrics.clone(),
        }
    }
}

/// `InferenceOutput` - Output of running AI inference on a given sequence group
#[derive(Debug)]
pub struct InferenceOutput {
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
pub enum EngineError {
    #[error("Failed to receive outputs from model thread dispatcher: `{0}`")]
    RecvError(tokio::sync::oneshot::error::RecvError),
}
