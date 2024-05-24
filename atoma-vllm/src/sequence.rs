use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    time::{Duration, Instant},
};

use candle::Tensor;
use thiserror::Error;
use tracing::{info_span, Span};

use crate::{block::LogicalTokenBlock, sampling_params::SamplingParams};

/// `LogProb` - log probabilities and token ranks.
#[derive(Clone)]
pub struct LogProb {
    /// Log probability
    logprob: f32,
    /// Token rank, in vocab
    rank: Option<u32>,
    /// Token decoding
    decoded_token: Option<String>,
}

impl LogProb {
    /// Constructor
    pub fn new(logprob: f32, rank: Option<u32>, decoded_token: Option<String>) -> Self {
        Self {
            logprob,
            rank,
            decoded_token,
        }
    }

    /// Getter for `logprob`
    pub fn logprob(&self) -> f32 {
        self.logprob
    }

    /// Getter for `rank`
    pub fn rank(&self) -> Option<u32> {
        self.rank
    }

    /// Getter for `decoded_token`
    pub fn decoded_token(&self) -> Option<String> {
        self.decoded_token
    }
}

/// `SequenceStatus` Status of a `Sequence`
#[derive(Clone)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    FinishedStopped,
    FinishedLengthCapped,
    FinishedAborted,
    FinishedIgnored,
}

impl SequenceStatus {
    /// Checks if `Sequence` is finished
    pub fn is_finished(&self) -> bool {
        match self {
            Self::FinishedAborted
            | Self::FinishedIgnored
            | Self::FinishedLengthCapped
            | Self::FinishedStopped => true,
            Self::Waiting | Self::Running | Self::Swapped => false,
        }
    }

    /// Finished reason
    pub fn finished_reason(&self) -> Option<String> {
        match self {
            Self::FinishedAborted => Some("aborted".into()),
            Self::FinishedIgnored => Some("ignored".into()),
            Self::FinishedLengthCapped => Some("length_capped".into()),
            Self::FinishedStopped => Some("stopped".into()),
            Self::Waiting | Self::Running | Self::Swapped => None,
        }
    }
}

/// State of a `Sequence`
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SequenceStage {
    Prefill,
    Decode,
}

/// Request metrics
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

/// `SequenceData` - data associated with a `Sequence`
///
/// Args:
///    `prompt_token_ids``: The token IDs of the prompt.
///    `output_token_ids``: The token IDs of the output. Set to an empty list if None.
#[derive(Clone)]
pub struct SequenceData {
    /// Prompt token ids
    prompt_token_ids: Vec<u32>,
    /// Output generated token ids
    output_token_ids: Vec<u32>,
    /// Cumulative log probability
    cumulative_logprob: f32,
    /// Number of computed tokens
    num_computed_tokens: usize,
    /// Stage of Sequence
    stage: SequenceStage,
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
    pub fn add_token_id(&mut self, token_id: u32, logprob: f32) {
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
        let mut output = Vec::with_capacity(self.get_prompt_len() + self.get_output_len());
        output.extend(self.prompt_token_ids.iter());
        output.extend(self.output_token_ids.iter());
        output
    }

    /// Get prefix token ids
    pub fn get_prefix_token_ids(&self, num_tokens: usize) -> (Vec<u32>, Vec<u32>) {
        let prompt_len = self.get_prompt_len();
        if num_tokens > prompt_len {
            return (
                self.prompt_token_ids.clone(),
                self.output_token_ids[..(num_tokens - prompt_len)].to_vec(),
            );
        } else {
            (self.prompt_token_ids[..num_tokens].to_vec(), vec![])
        }
    }

    /// Getter for `num_computed_tokens`. Return the number of prefill tokens that are already computed.
    pub fn get_num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }

    /// Computes the number of 'uncomputed' tokens
    pub fn get_num_uncomputed_tokens(&self) -> usize {
        // NOTE: we use `length()` which includes `prompt_len + output_len` instead
        // of `prompt_len` here. This is because during recompute we need to
        // prefill for both prompt and output.
        self.length() - self.get_num_computed_tokens()
    }

    /// Updates the number of computed tokens
    pub fn update_num_computed_tokens(
        &mut self,
        num_new_computed_tokens: usize,
    ) -> Result<(), SequenceError> {
        if self.num_computed_tokens + num_new_computed_tokens <= self.length() {
            self.num_computed_tokens += num_new_computed_tokens;
            if self.num_computed_tokens == self.length() {
                // All tokens have been already generated, so sequence transits to decode stage
                self.stage = SequenceStage::Decode;
            }
            return Ok(());
        }

        Err(SequenceError::InvalidNumberGeneratedTokens)
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

    /// Getter for `prompt_token_ids`
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.prompt_token_ids.clone()
    }

    /// Getter for `output_token_ids`
    pub fn output_token_ids(&self) -> Vec<u32> {
        self.output_token_ids.clone()
    }

    /// Getter for `stage`
    pub fn stage(&self) -> SequenceStage {
        self.stage
    }
}

/// `Sequence` - Stores the data, status, and block information of a sequence.
///
/// Args:
/// seq_id: The ID of the sequence.
/// prompt: The prompt of the sequence.
/// prompt_token_ids: The token IDs of the prompt.
/// block_size: The block size of the sequence. Should be the same as the
///    block size used by the block manager and cache engine.
///
/// Warn: Contrary to vLLM, we are not dealing with LoRA requests
#[derive(Clone)]
pub struct Sequence {
    /// Sequence Id,
    sequence_id: u64,
    /// End of sequence token
    eos_token_id: Option<u32>,
    /// Prompt
    prompt: String,
    /// Prompt associated token ids
    prompt_token_ids: Vec<u32>,
    /// Sequence data
    sequence_data: SequenceData,
    /// Block size
    block_size: usize,
    /// Logical token blocks
    logical_token_blocks: Vec<LogicalTokenBlock>,
    /// Prefix offset, used for incremental detokenization
    prefix_offset: usize,
    /// Read offset, used for incremental detokenization
    read_offset: usize,
    /// Output generated text
    output_text: String,
    /// List of all possible mappings from each generated output id to its `LogProb`
    output_logprobs: Vec<HashMap<u32, LogProb>>,
    /// Sequence status
    sequence_status: Option<SequenceStatus>,
    /// Generated tokens
    tokens: Vec<String>,
    /// Span
    span: Span,
}

impl Sequence {
    /// Constructor
    pub fn new(
        sequence_id: u64,
        eos_token_id: Option<u32>,
        prompt: String,
        prompt_token_ids: Vec<u32>,
        block_size: usize,
    ) -> Self {
        Self {
            sequence_id,
            eos_token_id,
            prompt,
            prompt_token_ids,
            sequence_data: SequenceData::new(prompt_token_ids, vec![]),
            logical_token_blocks: vec![],
            block_size,
            prefix_offset: 0,
            read_offset: 0,
            output_logprobs: vec![],
            output_text: String::new(),
            sequence_status: None,
            tokens: vec![],
            span: info_span!("sequence"),
        }
    }

    /// Get `output_text`
    pub fn get_output_text(&self) -> String {
        self.output_text.clone()
    }

    /// Computes the hash of a block given its logical index.
    pub fn hash_of_block(&self, logical_idx: usize) -> u64 {
        // TODO: This can produce incorrect hash when block size > prompt size
        let num_tokens = self.num_hashed_tokens_of_block(logical_idx);
        let hashed_tokens = self.sequence_data.get_prefix_token_ids(num_tokens);

        let mut hasher = DefaultHasher::new();
        hashed_tokens.hash(&mut hasher);

        hasher.finish()
    }

    /// Number of hashed tokens of a block
    pub fn num_hashed_tokens_of_block(&self, logical_idx: usize) -> usize {
        self.block_size * (logical_idx + 1)
    }

    /// Reset state for recomputation
    pub fn reset_state_for_recompute(&mut self) {
        self.sequence_data.reset_state_for_recompute()
    }

    /// Appends a new logical block to the `Sequence`
    fn append_logical_block(&mut self) {
        let block = LogicalTokenBlock::new(self.logical_token_blocks.len(), self.block_size);
        self.logical_token_blocks.push(block)
    }

    /// Appends tokens to last block in `Sequence`. If the last `logical_block` is full or `self.logical_token_blocks.is_empty()`,
    /// we allocate a new logical block to accommodate a new `logical_block`, for this purpose. If `token_ids.len()` exceeds
    /// the number of free slots in a `logical_block`, we allocate consecutively more `logical_blocks` to accommodate for the
    /// whole sequence of `token_ids`.
    fn append_tokens_to_blocks(&mut self, token_ids: &[u32]) {
        let mut cursor = 0;
        loop {
            if cursor >= token_ids.len() {
                break;
            }

            if self.logical_token_blocks.is_empty() {
                self.append_logical_block();
            }

            // DON'T PANIC: at this point in the logic, we already checked if `self.logical_token_blocks`
            let mut last_block = self.logical_token_blocks.last().unwrap();
            if last_block.is_full() {
                self.append_logical_block();
                last_block = self.logical_token_blocks.last().unwrap(); // DON'T PANIC: we are sure to have elements in `self.logical_token_blocks`
            }

            let num_empty_slots = last_block.get_num_empty_slots();
            last_block.append_tokens(&token_ids[cursor..(cursor + num_empty_slots)]);
            cursor += num_empty_slots;
        }
    }

    /// Appends a single token to the `Sequence`
    pub fn append_token_id(&mut self, token_id: u32, logprobs: HashMap<u32, LogProb>) {
        if logprobs.contains_key(&token_id) {
            self.append_tokens_to_blocks(&[token_id]);
            // DON'T PANIC: we have already verified that `token_id` is a valid key in `logprobs`
            let logprob = logprobs.get(&token_id).unwrap().logprob;
            self.sequence_data.add_token_id(token_id, logprob);
            self.output_logprobs.push(logprobs);
        }
    }

    /// Length of the underlying `Sequence`'s `SequenceData`
    pub fn length(&self) -> usize {
        self.sequence_data.length()
    }

    /// Length of the underlying `Sequence`'s `SequenceData` prompt length
    pub fn get_prompt_len(&self) -> usize {
        self.sequence_data.get_prompt_len()
    }

    /// Length of the output tokens in `SequenceData`
    pub fn get_output_len(&self) -> usize {
        self.sequence_data.get_output_len()
    }

    /// Getter for `SequenceData`'s prompt ids
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.sequence_data.prompt_token_ids()
    }

    /// Get `last_token_id` in `SequenceData`
    pub fn get_last_token_id(&self) -> Option<u32> {
        self.sequence_data.get_last_token_id()
    }

    /// Getter for `SequenceData`'s output token ids
    pub fn output_token_ids(&self) -> Vec<u32> {
        self.sequence_data.output_token_ids()
    }

    /// Get `SequenceData`'s token ids
    pub fn get_token_ids(&self) -> Vec<u32> {
        self.sequence_data.get_token_ids()
    }

    /// Get `SequenceData`'s cumulative log probabilities
    pub fn cumulative_logprob(&self) -> f32 {
        self.sequence_data.cumulative_logprob
    }

    /// Checks if a `Sequence` is finished
    pub fn is_finished(&self) -> bool {
        self.sequence_status
            .map(|s| s.is_finished())
            .unwrap_or(false)
    }

    /// Calculate the beam search score with length penalty.
    ///
    /// Adapted from
    ///
    /// https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    pub fn get_beam_search_score(
        &self,
        length_penalty: Option<f32>,
        mut sequence_length: Option<usize>,
        eos_token_id: Option<u32>,
    ) -> f32 {
        let length_penalty = length_penalty.unwrap_or(1.0);
        // NOTE: HF implementation does not count the EOS token
        // towards the length, we align with that here for testing.
        if sequence_length.is_none() {
            sequence_length = Some(self.length());
            if eos_token_id.is_some() && self.get_last_token_id() == eos_token_id {
                sequence_length = sequence_length.map(|l| l - 1);
            }
        }
        self.cumulative_logprob() / (sequence_length.unwrap_or_default() as f32 * length_penalty)
    }

    /// Fork a `Sequence`
    pub fn fork(&self, new_sequence_id: u64) -> Self {
        let mut new_seq = self.clone();
        new_seq.sequence_id = new_sequence_id;
        new_seq
    }

    /// Get the number of new tokens to be computed.
    ///
    /// Returns:
    /// The new number of tokens to be computed, i.e., 1 for decode, or
    /// the remaining prompt size for prefill.
    pub fn get_num_new_tokens(&self) -> usize {
        if self.sequence_data.stage == SequenceStage::Decode {
            return 1;
        }
        self.sequence_data.get_num_uncomputed_tokens()
    }

    /// Checks if we are in `Prefill` phase
    pub fn is_prefill(&self) -> bool {
        self.sequence_data.stage == SequenceStage::Prefill
    }
}

/// `SequenceGroupState` - Mutable state tied to a specific sequence group
pub struct SequenceGroupState {
    /// Generator used in seeded sampling
    pub generator: Option<usize>,
}

/// `MultiModalType` - The type of a multi-modal request
pub enum MultiModalType {
    Audio,
    Image,
    Video,
}

/// `MultiModalData` - Used for multi-modal requests.
///
/// Args:
///    type: The data type.
///    data: The actual data.
///    The required shape and semantic meaning of it depends on the vision
///    language config of the hosted model.
pub struct MultiModalData {
    /// Type
    pub r#type: MultiModalType,
    /// Data
    pub data: Tensor,
}

/// `Sequence` - A group of sequences that are generated from the same prompt.
///
/// Args:
/// request_id: The ID of the request.
/// seqs: The list of sequences.
/// sampling_params: The sampling parameters used to generate the outputs.
/// arrival_time: The arrival time of the request.
/// multi_modal_data: Multi modal data associated with the request.
/// embeddings: The embeddings vectors of the prompt of the sequence group
///    for an embedding model.
///
/// Warn: Our implementation does not consider LoRA and embeddings requests (contrary to vLLM).
pub struct SequenceGroup {
    /// Request Id
    request_id: String,
    /// Sequences
    sequences: Vec<Sequence>,
    /// Request's arrival time
    arrival_time: Instant,
    /// Multi modal data
    multi_model_data: Vec<MultiModalData>,
    /// Sampling parameters
    sampling_params: SamplingParams,
    /// State
    state: SequenceGroupState,
}

#[derive(Debug, Error)]
pub enum SequenceError {
    #[error("Invalid number of newly generated tokens for sequence")]
    InvalidNumberGeneratedTokens,
}
