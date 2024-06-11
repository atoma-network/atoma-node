use std::{collections::HashMap,time::Instant};

use thiserror::Error;
use tokenizers::Token;
use tokio::sync::{mpsc::{self, error::SendError}, oneshot};
use tracing::{error, info_span, instrument, Span};

use crate::{
    config::{CacheConfig, SchedulerConfig},
    models::ModelExecutor,
    policy::FcfsPolicy,
    scheduler::{ScheduledSequenceGroup, Scheduler, SchedulerError},
    sequence::{
        Sequence, SequenceError, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput,
    },
    tokenizer::{DetokenizerRequest, TokenizerError, TokenizerRequest},
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
pub struct LlmEngine<M: ModelExecutor> {
    /// The scheduler, which handles CPU and GPU memory allocation
    scheduler: Scheduler<FcfsPolicy>,
    /// Request validator
    validation: Validation,
    /// Model executor, responsible for running decoding steps to produce
    /// AI generated outputs
    model_executor: M,
    /// Blockchain event requests receiver
    request_receiver: mpsc::UnboundedReceiver<GenerateRequest>,
    /// Request counter
    request_counter: u64,
    /// Tokenizer `mpsc` unbounded sender channel,
    tokenizer_sender: mpsc::UnboundedSender<DetokenizerRequest>,
    /// Tracing span
    span: Span,
}

impl<M: ModelExecutor> LlmEngine<M> {
    /// Constructor
    pub fn new(
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        validation: Validation,
        model_executor: M,
        request_receiver: mpsc::UnboundedReceiver<GenerateRequest>,
    ) -> Result<Self, EngineError> {
        let scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)?;
        Ok(Self {
            scheduler,
            validation,
            model_executor,
            request_receiver,
            request_counter: 0,
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
        )?;
        let sequence_group = SequenceGroup::new(
            request_id,
            vec![sequence],
            arrival_time,
            valid_request.parameters,
            valid_request.stopping_parameters,
        )?;

        self.scheduler.add_sequence_group(sequence_group);
        Ok(())
    }

    /// Process AI generated outputs, outputs come in the form of a mapping
    /// from `request_id` -> `SequenceGroupOutput`
    #[instrument(skip(self))]
    async fn process_model_outputs(
        &mut self,
        outputs: HashMap<String, SequenceGroupOutput>,
        scheduled_sequence_groups: Vec<ScheduledSequenceGroup>,
        ignored_sequence_groups: Vec<SequenceGroup>,
        sequence_group_metadata: Vec<SequenceGroupMetadata>,
    ) -> Result<(), EngineError> {
        let now = Instant::now();

        for (scheduled_sequence_group, scheduled_group_output) in
            scheduled_sequence_groups.iter_mut().zip(outputs.iter())
        {
            // update the number of computed tokens for scheduled `SequenceGroup`
            scheduled_sequence_group
                .scheduled_group
                .update_num_computed_tokens(scheduled_sequence_group.token_chunk_size);

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
                    let generated_token_id = sequence_output.output_token;
                    let (sender, receiver) = oneshot::channel();
                    let detokenizer_request = DetokenizerRequest {
                        token_id: generated_token_id,
                        sender,
                        span: Span::current(),
                     };
                    self.tokenizer_sender.send(detokenizer_request)?;
                    let generated_token = receiver.await.unwrap()?;

                    sequence
                        .borrow_mut()
                        .output_logprobs
                        .push(sequence_output.logprob.clone());

                    sequence.borrow_mut().output_text.push_str(&generated_token);
                    sequence.borrow_mut().prompt_token_ids.push(generated_token_id);
                    sequence.borrow_mut().tokens.push(generated_token);
                }
            }
        }

        // Free all finished sequence groups
        self.scheduler.free_finished_sequence();

        // let mut request_outputs = Vec::new();

        Ok(())
    }

    #[instrument(skip(self))]
    /// Updates a `SequenceGroup` log probabilities for a newly generated output
    fn process_prompt_logprob(
        &self,
        sequence_group: &mut SequenceGroup,
        outputs: &SequenceGroupOutput,
    ) -> Result<(), EngineError> {
        if outputs.len() != 1 {
            error!(
                "Sequence group output should be 1, but is {}",
                outputs.len()
            );
            return Err(EngineError::InvalidOutputLength(outputs.len()));
        }
        let output = &outputs[0];
        let prompt_logprobs = output.prompt_logprobs;

        Ok(())
    }
}

/// `OutputProcessor` - Responsible for processing AI generated
/// next token id
pub struct OutputProcessor {}

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
    TokenizerError(#[from] TokenizerError),
}
