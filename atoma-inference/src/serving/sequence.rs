use crate::serving::validation::{NextTokenChooserParameters, StoppingCriteriaParameters};
use candle::Tensor;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, Span};

/// `Sequence` - A sequence of tokens generated by the LLM
/// that can be parsed by the underlying LLM engine
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique identifier for the sequence
    pub sequence_id: u64,
    /// The text input string
    pub inputs: String,
    /// The tokenized input ids
    pub input_token_ids: Vec<u32>,
    /// The arrival time of the request,
    /// into the `LlmService`
    pub arrival_time: Instant,
    /// Return the full text, that is it
    /// prepends the input string to the generated
    /// output text
    pub return_full_text: bool,
    /// The LLM inference generation parameters
    pub parameters: NextTokenChooserParameters,
    /// The stopping parameters for the LLM inference
    pub stopping_parameters: StoppingCriteriaParameters,
    /// Tracing span
    pub span: Span,
}

impl Sequence {
    /// Constructor
    pub fn new(
        sequence_id: u64,
        inputs: String,
        input_token_ids: Vec<u32>,
        arrival_time: Instant,
        return_full_text: bool,
        parameters: NextTokenChooserParameters,
        stopping_parameters: StoppingCriteriaParameters,
    ) -> Self {
        Self {
            sequence_id,
            inputs,
            input_token_ids,
            arrival_time,
            return_full_text,
            parameters,
            stopping_parameters,
            span: info_span!("sequence"),
        }
    }
}

/// `SequenceMetadata` - Metadata for a sequence.
///
/// Args:
///     `request_id`: The ID of the request.
///     `is_prompt`: Whether the request is at prompt stage.
///     `computed_block_nums`: The block numbers that are already computed,
///          used in prefix caching.
///     `token_chunk_size`: The number of tokens to be processed. It corresponds
///         to the prompt token size, if the sequence is in prefill phase, otherwise
///         if in decoding phase, it is usually 1 (unless one does speculative decoding).
///     `multi_modal_data`: Multi modal data.
#[derive(Debug)]
pub struct SequenceMetadata {
    /// Request id
    pub request_id: String,
    /// Is prompt (bool)
    pub is_prompt: bool,
    /// Next token chooser parameters
    pub next_token_chooser_params: NextTokenChooserParameters,
    /// Stopping criteria parameters
    pub stopping_criteria_params: StoppingCriteriaParameters,
    /// Token chunk size
    pub token_chunk_size: usize,
    /// Sequence data
    pub sequence_data: SequenceData,
    /// Multi modal data
    pub multi_modal_data: Option<MultiModalData>,
}

impl SequenceMetadata {
    /// Constructor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        is_prompt: bool,
        sequence_data: SequenceData,
        next_token_chooser_params: NextTokenChooserParameters,
        stopping_criteria_params: StoppingCriteriaParameters,
        token_chunk_size: Option<usize>,
        multi_modal_data: Option<MultiModalData>,
    ) -> Self {
        let token_chunk_size = if let Some(size) = token_chunk_size {
            size
        } else if is_prompt {
            sequence_data.length()
        } else {
            1
        };

        Self {
            request_id,
            is_prompt,
            sequence_data,
            next_token_chooser_params,
            stopping_criteria_params,
            token_chunk_size,
            multi_modal_data,
        }
    }
}

/// `SequenceData` - data associated with a `Sequence`.
/// It includes the token ids of the prompt and the output,
/// the cumulative log probability, the number of computed tokens,
/// and the stage of the sequence.
///    `prompt_token_ids``: The token IDs of the prompt.
///    `output_token_ids``: The token IDs of the output. Set to an empty list if None
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceData {
    /// Prompt token ids, the token ids of the prompt
    /// computed in prefill stage.
    pub prompt_token_ids: Vec<u32>,
    /// Output generated token ids, the token ids
    /// of the output generated while in decode stage.
    pub output_token_ids: Vec<u32>,
    /// Cumulative log probability, for the whole
    /// generated sequence.
    pub cumulative_logprob: f32,
    /// Number of computed tokens, for the whole
    /// generated sequence.
    pub num_computed_tokens: usize,
    /// Stage of Sequence.
    pub stage: SequenceStage,
}

impl SequenceData {
    /// Constructor
    pub fn new(prompt_token_ids: Vec<u32>, output_token_ids: Vec<u32>) -> Self {
        Self {
            prompt_token_ids,
            output_token_ids,
            cumulative_logprob: 0.0,
            num_computed_tokens: 0,
            stage: SequenceStage::Prefill,
        }
    }

    /// Adds a new generated output token id
    #[instrument(skip(self))]
    pub fn add_token_id(&mut self, token_id: u32, logprob: f32) {
        info!("Adding token id to `SequenceData`..");
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
    }

    /// Get length of total number of token ids (including prompt token ids)
    pub fn length(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// Get prompt token ids length
    pub fn get_prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    /// Get output token ids length
    pub fn get_output_len(&self) -> usize {
        self.output_token_ids.len()
    }

    /// Get all token ids
    pub fn get_token_ids(&self) -> Vec<u32> {
        let mut output = Vec::with_capacity(self.length());
        output.extend(self.prompt_token_ids.iter());
        output.extend(self.output_token_ids.iter());
        output
    }

    /// Get prefix token ids
    pub fn get_prefix_token_ids(&self, num_tokens: usize) -> (Vec<u32>, Vec<u32>) {
        let prompt_len = self.get_prompt_len();
        if num_tokens > prompt_len {
            (
                self.prompt_token_ids.clone(),
                self.output_token_ids[..(num_tokens - prompt_len)].to_vec(),
            )
        } else {
            (self.prompt_token_ids[..num_tokens].to_vec(), vec![])
        }
    }

    /// Getter for `num_computed_tokens`. Return the number of tokens that are already computed.
    pub fn get_num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }

    /// Computes the number of 'uncomputed' tokens. In case of a
    /// preemption and recomputation
    pub fn get_num_uncomputed_tokens(&self) -> usize {
        // NOTE: we use `length()` which includes `prompt_len + output_len` instead
        // of `prompt_len` here. This is because during recompute we need to
        // prefill for both prompt and output.
        self.length() - self.get_num_computed_tokens()
    }

    /// Updates the number of computed tokens, so far
    #[instrument(skip(self))]
    pub fn update_num_computed_tokens(
        &mut self,
        num_new_computed_tokens: usize,
    ) -> Result<(), SequenceError> {
        info!(
            "Update number of computed tokens {} by {}",
            self.num_computed_tokens, num_new_computed_tokens
        );

        info!(
             "Parameters: self.num_computed_tokens = {}, self.length() = {}, num_new_computed_tokens = {}, self.get_num_uncomputed_tokens() = {}", 
             self.num_computed_tokens,
             self.length(),
             num_new_computed_tokens,
             self.get_num_uncomputed_tokens()
         );

        self.num_computed_tokens += num_new_computed_tokens;
        if self.num_computed_tokens <= self.length() {
            if self.get_num_uncomputed_tokens() == 0 {
                // Prompt tokens attention layers have been now computed, so sequence transits to decode stage
                self.stage = SequenceStage::Decode;
            }
            return Ok(());
        }
        error!("Failed to update number of computed tokens: self.num_computed_tokens = {} > self.length() = {}", self.num_computed_tokens, self.length());
        Err(SequenceError::InvalidNumberGeneratedTokens(
            self.num_computed_tokens,
            self.length(),
        ))
    }

    /// Reset state for recomputation.  It is supposed to be called when a sequence
    /// needs to be started from the beginning again (e.g., sequence is preempted).
    pub fn reset_state_for_recompute(&mut self) {
        self.num_computed_tokens = 0;
        self.stage = SequenceStage::Prefill
    }

    /// Get last generated token id
    pub fn get_last_token_id(&self) -> Option<u32> {
        if self.output_token_ids.is_empty() {
            return self.prompt_token_ids.last().copied();
        }
        self.output_token_ids.last().copied()
    }
}

/// State of a `Sequence`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceStage {
    Prefill,
    Decode,
}

/// `MultiModalData` - Used for multi-modal requests.
#[derive(Clone, Debug)]
pub struct MultiModalData {
    /// Type
    pub r#type: MultiModalType,
    /// Data
    pub data: Tensor,
}

/// `MultiModalType` - Multi-modal type
/// Currently, we support `Audio`, `Image`, and `Video` data types.
#[derive(Clone, Debug)]
pub enum MultiModalType {
    Audio,
    Image,
    Video,
}

/// Request metrics
#[derive(Clone, Debug)]
pub struct RequestMetrics {
    /// Time of request arrival to service
    pub arrival_time: Instant,
    /// Time to which last token was generated
    pub last_token_time: Instant,
    /// Time that request was first scheduled
    pub first_scheduled_time: Option<Instant>,
    /// Time to which first token was generated
    pub first_token_time: Option<Instant>,
    /// Duration of request in 'waiting' queue
    pub time_in_queue: Option<Duration>,
    /// Time to which generation finished for request
    pub finished_time: Option<Instant>,
}

#[derive(Debug, Error)]
pub enum SequenceError {
    #[error("Invalid number of generated tokens: {0}")]
    InvalidNumberGeneratedTokens(usize, usize),
}
