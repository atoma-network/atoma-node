use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{Arc, RwLock},
    time::Instant,
};

use futures::StreamExt;
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{self, error::SendError};
use tracing::{error, info, info_span, instrument, Span};

use crate::{
    config::{CacheConfig, SchedulerConfig},
    model_executor::{ModelExecutor, ModelThreadDispatcher, ModelThreadError},
    policy::FcfsPolicy,
    scheduler::{ScheduledSequenceGroup, Scheduler, SchedulerError},
    sequence::{
        ExecuteModelRequest, LogProb, RequestMetrics, Sequence, SequenceError, SequenceGroup,
        SequenceGroupMetadata, SequenceGroupOutput, SequenceStatus,
    },
    tokenizer::DetokenizerRequest,
    types::GenerateRequest,
    validation::{ValidGenerateRequest, Validation, ValidationError},
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
    /// Model executor, responsible for running decoding steps to produce
    /// AI generated outputs
    model_thread_dispatcher: ModelThreadDispatcher,
    /// Blockchain event requests receiver
    request_receiver: mpsc::UnboundedReceiver<GenerateRequest>,
    /// Unbounded `mpsc` Sender channel, to send newly AI generated outputs
    /// to the Atoma's client service
    atoma_client_sender: mpsc::UnboundedSender<Vec<GenerateRequestOutput>>,
    /// Request counter
    request_counter: u64,
    /// Tokenizer for decoding sequences
    tokenizer: Tokenizer,
    /// End of sentence token, for the current model's tokenizer
    eos_token_id: u32,
    /// Tracing span
    span: Span,
}

impl LlmEngine {
    /// Constructor
    pub async fn new<M>(
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        validation: Validation,
        model_executor: M,
        tokenizer: Tokenizer,
        atoma_client_sender: mpsc::UnboundedSender<Vec<GenerateRequestOutput>>,
        request_receiver: mpsc::UnboundedReceiver<GenerateRequest>,
    ) -> Result<Self, EngineError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let eos_token_id = model_executor
            .eos_token_id()
            .ok_or(EngineError::MissingEosTokenId)?;
        let scheduler = Scheduler::new(cache_config, scheduler_config)?;
        let model_thread_dispatcher = ModelThreadDispatcher::start(model_executor)?;
        Ok(Self {
            scheduler,
            validation,
            model_thread_dispatcher,
            tokenizer,
            atoma_client_sender,
            request_receiver,
            request_counter: 0,
            eos_token_id,
            span: info_span!("llm-engine"),
        })
    }

    /// Processes newly arrived inputs
    #[instrument(skip(self))]
    async fn process_received_request(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, EngineError> {
        Ok(self.validation.validate(request).await?)
    }

    /// Add a newly arrived request to the `LlmEngine`'s request pool.
    ///
    /// The request is added to the request pool and will be processed
    /// by the scheduler as `engine.step()` is called. The exact scheduling
    /// policy is determined by the `Scheduler`.
    #[instrument(skip(self))]
    async fn add_request(&mut self, request: GenerateRequest) -> Result<(), EngineError> {
        let arrival_time = Instant::now();
        let request_id = request.request_id.clone();
        let valid_request = self.process_received_request(request).await?;

        let block_size = self.scheduler.cache_config.block_size;
        let sequence_id = self.request_counter;
        self.request_counter += 1;

        let sequence = Sequence::new(
            sequence_id,
            valid_request.inputs.clone(),
            valid_request.encoding.get_ids().to_vec(),
            block_size,
            valid_request.return_full_text,
        )?;
        let sequence_group = SequenceGroup::new(
            request_id,
            vec![sequence],
            arrival_time,
            valid_request.parameters.clone(),
            valid_request.stopping_parameters.clone(),
        )?;

        self.scheduler.add_sequence_group(sequence_group);
        Ok(())
    }

    /// Process AI generated outputs, outputs come in the form of a mapping
    /// from `request_id` -> `SequenceGroupOutput`
    #[instrument(skip(self))]
    fn process_model_outputs(
        &mut self,
        outputs: HashMap<String, SequenceGroupOutput>,
        scheduled_sequence_groups: Vec<ScheduledSequenceGroup>,
        ignored_sequence_groups: Vec<SequenceGroup>,
        sequence_groups_metadata: Vec<Arc<SequenceGroupMetadata>>,
    ) -> Result<Vec<GenerateRequestOutput>, EngineError> {
        let now = Instant::now();

        for (scheduled_sequence_group, sequence_group_metadata) in scheduled_sequence_groups
            .iter()
            .zip(sequence_groups_metadata.iter())
        {
            // update the number of computed tokens for scheduled `SequenceGroup`
            scheduled_sequence_group
                .scheduled_group
                .update_num_computed_tokens(scheduled_sequence_group.token_chunk_size)?;

            let sequence_group_id = &scheduled_sequence_group.scheduled_group.request_id;
            let sequence_group_output = if let Some(output) = outputs.get(sequence_group_id) {
                output
            } else {
                error!(
                    "Missing scheduled sequence group output for processing, with id = {}",
                    sequence_group_id
                );
                return Err(EngineError::MissingScheduleGroupOutput(
                    sequence_group_id.clone(),
                ));
            };

            let stopping_criteria_params =
                scheduled_sequence_group.scheduled_group.stopping_params();

            // TODO: we can process this concurrently
            for (sequence_id, sequence) in scheduled_sequence_group.scheduled_group.sequences.iter()
            {
                let sequence_output =
                    if let Some(output) = sequence_group_output.outputs.get(sequence_id) {
                        output
                    } else {
                        error!(
                            "Missing generated sequence output token for sequence with id = {}",
                            sequence_id
                        );
                        return Err(EngineError::MissingSequenceOutputToken(*sequence_id));
                    };
                {
                    // We need to add process the `Sequence` state. This includes:
                    //
                    // 1. Get the AI generated next output token id.
                    //
                    // 2. Update the `Sequence`'s output log-probabilities.
                    //
                    // 3. Update the `Sequence`'s `SequenceData` cumulative probabilities,
                    //    if we are in decoding phase.
                    //
                    // 4. Update the `Sequence`'s `SequenceData` output tokens,
                    //    if we are in decoding phase.
                    //
                    // 5. Decode the generated output token id.
                    //
                    // 6. Update the `Sequence`'s tokens vec.
                    //
                    // 7. Update the `output_text` with the newly generated token,
                    //    if in decoding phase.
                    //
                    // 8. Check if the last generated token is a stop token.
                    //    If so, update the `Sequence`'s `SequenceState` and
                    //    the `stop_reason`, as well.
                    //
                    // 9. Check if the current `Sequence` last generated token
                    //    id equals to the `eos_token_id`, in which case the
                    //    the `Sequence`'s status should become `FinishedStopped`.
                    //
                    //
                    // 10. Check if the `Sequence`'s length exceeds that of
                    //    `SchedulerConfig`'s.
                    //
                    // 11. Check if the `Sequence`'s output length exceeds that of
                    //     Request's `max_new_tokens`.
                    //
                    // NOTE: We do not need to update the `Sequence`'s `SequenceData`
                    // `num_computed_tokens` and `sequence_stage`, as this was already
                    // done above, at the `SequenceGroup` stage (using `update_num_computed_tokens`).
                    //

                    // 1.
                    let generated_token_id = sequence_output.output_token;

                    if sequence_group_metadata.do_sample {
                        // NOTE: this means we are in decoding phase.
                        // That said, we are generating new output tokens
                        // and these should be added to the `Sequence`'s
                        // state.

                        // 2. 3. 4.
                        {
                            sequence
                                .borrow_mut()
                                .add_token_id(generated_token_id, sequence_output.logprob)?;
                        }

                        // 5.
                        let generated_token = self
                            .tokenizer
                            .decode(&[generated_token_id], true)
                            .map_err(|e| EngineError::TokenizerError(e.to_string()))?;

                        // 6.
                        {
                            sequence.borrow_mut().tokens.push(generated_token)
                        }
                        // 7.
                        {
                            sequence.borrow_mut().output_text.push_str(&generated_token)
                        }
                        // 8.
                        if stopping_criteria_params
                            .stop_sequences
                            .contains(&generated_token)
                        {
                            info!("Current sequence with id = {sequence_id} has finished execution due to stopping token = {generated_token}");
                            {
                                sequence.borrow_mut().stop_reason = Some(generated_token_id)
                            }

                            sequence
                                .borrow()
                                .set_sequence_status(SequenceStatus::FinishedStopped)
                        }

                        // 9.
                        if self.eos_token_id == generated_token_id
                            && !stopping_criteria_params.ignore_eos_token
                        {
                            sequence
                                .borrow()
                                .set_sequence_status(SequenceStatus::FinishedStopped)
                        }

                        // 10.
                        if sequence.borrow().length()
                            > self.scheduler.scheduler_config.max_model_len()
                        {
                            sequence
                                .borrow()
                                .set_sequence_status(SequenceStatus::FinishedLengthCapped)
                        }

                        // 11.
                        if sequence.borrow().get_output_len()
                            > stopping_criteria_params.max_new_tokens as usize
                        {
                            sequence
                                .borrow()
                                .set_sequence_status(SequenceStatus::FinishedLengthCapped)
                        }
                    } else {
                        // NOTE: in this case, we are not sampling newly
                        // generated tokens. That is, we are in prefill
                        // phase (possibly while chunking). For this reason,
                        // we do not have to add tokens to the current
                        // `Sequence`'s state.

                        // 2.
                        sequence
                            .borrow_mut()
                            .output_logprobs
                            .push(sequence_output.logprob.clone());
                    }
                }
            }

            // Now that we have processed all `Sequence`'s in `SequenceGroup`.
            // We need to update the `SequenceGroup` state. This includes:
            //
            // 1. Check if all sequences are finished
            //
            // 2. add metrics
            //

            if scheduled_sequence_group.scheduled_group.is_finished() {
                // TODO: send the finished output to the atoma client
            }
            let arrival_time_histogram = metrics::histogram!("sequence-group-arrival-time");
            let metrics_guard = scheduled_sequence_group
                .scheduled_group
                .metrics
                .read()
                .unwrap();
            arrival_time_histogram.record(metrics_guard.arrival_time.elapsed().as_secs_f32());
            let last_token_time_histogram = metrics::histogram!("sequence-group-last-token-time");
            last_token_time_histogram.record(metrics_guard.last_token_time.elapsed().as_secs_f32());
        }

        // Free all finished sequence groups
        self.scheduler.free_finished_sequence();

        let mut request_outputs = Vec::new();
        for scheduled_sequence_group in scheduled_sequence_groups.iter() {
            scheduled_sequence_group
                .scheduled_group
                .maybe_set_first_scheduled_time(now);
            request_outputs.push(GenerateRequestOutput::from_sequence_group(
                &scheduled_sequence_group.scheduled_group,
            ));
        }
        for sequence_group in ignored_sequence_groups.iter() {
            sequence_group.maybe_set_first_scheduled_time(now);
            request_outputs.push(GenerateRequestOutput::from_sequence_group(&sequence_group));
        }

        Ok(request_outputs)
    }

    /// Runs the `LlmEngine` instance, it listens to new arriving requests and processes these
    #[instrument(skip(self))]
    pub async fn run(mut self) -> Result<(), EngineError> {
        loop {
            tokio::select! {
                    Some(request) = self.request_receiver.recv() => {
                        self.add_request(request).await?;
                    }
                    Some(resp) = self.model_thread_dispatcher.responses.next() => {
                        // REF[response_received]
                        match resp {
                            Ok(response) => {
                                // We have received a new LLM inference loop response,
                                // we now start the next model inference, we need:
                                //
                                // 1. Check if the response is non-empty.
                                //
                                // 2. If the response is not empty, we then need
                                //    to process the generated output
                                //
                                // 3. If the responses is not empty, we
                                //    send the output back to the Atoma output manager
                                //
                                // 4. Run a `self.step()` to run the next inference step
                                //    even if the sequence is empty, otherwise the system
                                //    can't make progress
                                //

                                // 1.
                                if !response.is_empty() {
                                    // 2.
                                    let outputs = self.process_model_outputs()?;

                                    // 3.
                                    self.atoma_client_sender.send(outputs);
                                } else {
                                    // The received response is empty, so we might just continue
                                    // with the next iteration of the loop
                                }

                                // 4.
                                self.step()?;
                        }
                        Err(e) => {
                            error!("Failed to generate model inference response, with error: {e}");
                            // NOTE: In order to maintain the system live, we need to keep calling
                            // the `self.step()` method, even in possible failure scenarios.
                            self.step();
                        }
                    }
                }
            }
        }
    }

    /// Main method of `LlmEngine`.
    ///
    /// 1. It is responsible for scheduling new
    ///     requests, via the associated `Scheduler`. Once scheduling is complete,
    ///
    /// 2. It sends a new `ExecuteModelRequest` to the `ModelExecutor`'s thread.
    ///
    /// 3. Once the execution is complete, the `self.model_thread_dispatcher.responses`
    ///     `FuturesUnordered` should be able to poll next `response`. This is executed
    ///     through the main loop in the `self.run()` main method.
    ///
    #[instrument(skip(self))]
    pub fn step(&mut self) -> Result<(), EngineError> {
        // 1. Schedule new requests
        let (scheduler_groups_metadata, scheduler_outputs) = self.scheduler.schedule()?;

        if !scheduler_outputs.is_empty() {
            let execute_model_request = ExecuteModelRequest::new(
                scheduler_groups_metadata,
                scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy,
                scheduler_outputs.running_queue_size,
            );

            // 2. Sends a new `ExecuteModelRequest` to the underlying `ModelExecutor`'s thread
            self.model_thread_dispatcher.send(execute_model_request);

            // 3. Is handled by the `self.run()` method in REF[response_received]
        } else {
            // TODO: check if we can improve the logic of sending empty requests/receiving empty responses
            // just to maintain the system live
            self.model_thread_dispatcher
                .send(ExecuteModelRequest::empty());
        }
        Ok(())
    }
}

fn update_sequence(sequence_id: u64, sequence: Rc<RefCell<Sequence>>) {
    
}
/// `RequestOutput` - Output of running AI inference over a `SequenceGroup`
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

impl GenerateRequestOutput {
    /// Creates a new `Self` instance from a `SequenceGroup`
    pub fn from_sequence_group(sequence_group: &SequenceGroup) -> Self {
        let mut sequences = sequence_group.sequences.values().collect::<Vec<_>>();

        let top_n_sequences = if sequences.len() == 1 {
            sequences
        } else {
            // Get top n sequences
            let n = sequence_group.next_token_chooser_params().n;
            sequences.sort_by(|s1, s2| {
                s1.borrow()
                    .cumulative_logprob()
                    .partial_cmp(&s2.borrow().cumulative_logprob())
                    .unwrap()
            });
            sequences[..n].to_vec()
        };

        let inference_outputs = top_n_sequences
            .iter()
            .enumerate()
            .map(|(i, s)| InferenceOutput {
                index: i,
                output_text: s.borrow().get_output_text(),
                token_ids: s.borrow().get_token_ids(),
                cumulative_logprob: s.borrow().cumulative_logprob(),
                logprobs: s.borrow().output_logprobs.clone(),
                finish_reason: s.borrow().get_sequence_status().finished_reason(),
                stop_reason: s.borrow().stop_reason.clone(),
            })
            .collect::<Vec<_>>();

        Self {
            request_id: sequence_group.request_id.clone(),
            inference_outputs,
            prompt: sequence_group.prompt(),
            prompt_token_ids: sequence_group.prompt_token_ids(),
            is_finished: sequence_group.is_finished(),
            metrics: sequence_group.metrics.clone(),
        }
    }
}

/// `InferenceOutput` - Output of running a
pub struct InferenceOutput {
    /// The index of the output in the request
    index: usize,
    /// The generated output text
    output_text: String,
    /// The token ids of the generated output text
    token_ids: Vec<u32>,
    /// The cumulative log probability of the generated
    /// output text
    cumulative_logprob: f32,
    /// The log probabilities of the top probability words at each
    /// position if the logprobs are requested
    logprobs: Vec<HashMap<u32, LogProb>>,
    /// The reason why the sequence is finished
    finish_reason: Option<String>,
    /// The stop token id that caused the completion
    /// to stop, None if the completion finished for some other reason
    /// including encountering the eos token
    stop_reason: Option<u32>,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] ValidationError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Invalid output length: `{0}`")]
    InvalidOutputLength(usize),
    #[error("Missing scheduled group output, id = `{0}`")]
    MissingScheduleGroupOutput(String),
    #[error("Missing sequence output token, id = `{0}`")]
    MissingSequenceOutputToken(u64),
    #[error("SendError: `{0}`")]
    SendError(#[from] SendError<DetokenizerRequest>),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(String),
    #[error("Model thread error: `{0}`")]
    ModelThreadError(#[from] ModelThreadError),
    #[error("Missing eos token id")]
    MissingEosTokenId,
}
