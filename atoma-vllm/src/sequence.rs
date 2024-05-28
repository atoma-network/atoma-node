use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Index, IndexMut},
    rc::Rc,
    time::{Duration, Instant},
};

use candle::Tensor;
use thiserror::Error;
use tracing::{info_span, Span};

use crate::{block::LogicalTokenBlock, sampling_params::SamplingParams};

/// `LogProb` - log probabilities and token ranks.
#[derive(Clone, Debug, PartialEq)]
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
        self.decoded_token.clone()
    }
}

/// `SequenceStatus` Status of a `Sequence`
#[derive(Clone, PartialEq, Eq)]
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
            (
                self.prompt_token_ids.clone(),
                self.output_token_ids[..(num_tokens - prompt_len)].to_vec(),
            )
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
///    `seq_id`: The ID of the sequence.
///    `prompt`: The prompt of the sequence.
///    `prompt_token_ids`: The token IDs of the prompt.
///    `block_size`: The block size of the sequence. Should be the same as the
///        block size used by the block manager and cache engine.
///
/// Warn: Contrary to vLLM, we are not dealing with LoRA requests
#[derive(Clone)]
pub struct Sequence {
    /// Sequence Id,
    sequence_id: u64,
    /// End of sequence token
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    prefix_offset: usize,
    /// Read offset, used for incremental detokenization
    #[allow(dead_code)]
    read_offset: usize,
    /// Output generated text
    output_text: String,
    /// List of all possible mappings from each generated output id to its `LogProb`
    output_logprobs: Vec<HashMap<u32, LogProb>>,
    /// Sequence status
    sequence_status: Option<SequenceStatus>,
    /// Generated tokens
    #[allow(dead_code)]
    tokens: Vec<String>,
    /// Span
    #[allow(dead_code)]
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
            prompt_token_ids: prompt_token_ids.clone(),
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
            let mut last_block = self.logical_token_blocks.last_mut().unwrap();
            if last_block.is_full() {
                self.append_logical_block();
                last_block = self.logical_token_blocks.last_mut().unwrap(); // DON'T PANIC: we are sure to have elements in `self.logical_token_blocks`
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
            .as_ref()
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
///    `type`: The data type.
///    `data`: The actual data.
///    
/// The required shape and semantic meaning of it depends on the vision
/// language config of the hosted model.
pub struct MultiModalData {
    /// Type
    pub r#type: MultiModalType,
    /// Data
    pub data: Tensor,
}

/// `Sequence` - A group of sequences that are generated from the same prompt.
///
/// Args:
///    `request_id`: The ID of the request.
///    `seqs`: The list of sequences.
///    `sampling_params`: The sampling parameters used to generate the outputs.
///    `arrival_time`: The arrival time of the request.
///    `multi_modal_data`: Multi modal data associated with the request.
///    `embeddings`: The embeddings vectors of the prompt of the sequence group
///        for an embedding model.
///
/// Warn: Our implementation does not consider LoRA and embeddings requests (contrary to vLLM).
pub struct SequenceGroup {
    /// Request Id
    #[allow(dead_code)]
    request_id: String,
    /// Sequences
    sequences: HashMap<u64, Sequence>,
    /// Request's arrival time
    #[allow(dead_code)]
    arrival_time: Instant,
    /// Request metrics
    metrics: RequestMetrics,
    /// Multi modal data
    #[allow(dead_code)]
    multi_model_data: Vec<MultiModalData>,
    /// Prompt log probabilities
    #[allow(dead_code)]
    prompt_logprobs: Option<LogProb>,
    /// Sampling parameters
    #[allow(dead_code)]
    sampling_params: Option<SamplingParams>,
    /// State
    #[allow(dead_code)]
    state: SequenceGroupState,
}

impl SequenceGroup {
    /// Constructor
    pub fn new(
        request_id: String,
        sequences: Vec<Sequence>,
        arrival_time: Instant,
        prompt_logprobs: Option<LogProb>,
        sampling_params: Option<SamplingParams>,
        multi_model_data: Vec<MultiModalData>,
        state: SequenceGroupState,
    ) -> Self {
        Self {
            request_id,
            sequences: sequences.into_iter().map(|s| (s.sequence_id, s)).collect(),
            arrival_time,
            metrics: RequestMetrics {
                arrival_time,
                last_token_time: arrival_time,
                finished_time: None,
                first_scheduled_time: None,
                first_token_time: None,
                time_in_queue: None,
            },
            multi_model_data,
            prompt_logprobs,
            sampling_params,
            state,
        }
    }

    /// Prompt of the `SequenceGroup`, all the sequences should have the same prompt
    pub fn prompt(&self) -> String {
        self.sequences
            .iter()
            .next()
            .map(|(_, s)| s.prompt.clone())
            .unwrap_or_default()
    }

    /// Prompt token ids, all the sequences in the `SequenceGroup` should have the same prompt (thus same prompt token ids)
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.sequences
            .iter()
            .next()
            .map(|(_, s)| s.prompt_token_ids.clone())
            .unwrap_or_default()
    }

    /// Sets the last token time for Request level timings and outputs the latency between `now` and `last_token_time`
    pub fn get_last_latency(&mut self, now: Instant) -> Result<Duration, SequenceError> {
        if self.is_prefill() {
            return Err(SequenceError::WhileInPrefix);
        }
        let latency = now - self.metrics.last_token_time;
        self.metrics.last_token_time = now;
        Ok(latency)
    }

    /// Sets the first token time for Request level timings.
    #[allow(dead_code)]
    fn maybe_set_first_token_time(&mut self, time: Instant) {
        // NOTE: in a case where a sequence_group is swapped and
        // recomputed, the time between iterations is counted
        // in TPOT, rather than recalculating TTFT (since from the
        // POV of the user, there is simply a long generation delay.
        let initial_seq_len = self
            .sequences
            .iter()
            .next()
            .map(|(_, s)| s.get_output_len())
            .unwrap_or_default();
        if self.metrics.first_token_time.is_none() && initial_seq_len == 1 {
            self.metrics.first_token_time = Some(time);
        }
    }

    /// Sets the first scheduled time and time in queue for Request level timings.
    #[allow(dead_code)]
    fn maybe_set_first_scheduled_time(&mut self, time: Instant) {
        if self.metrics.first_scheduled_time.is_none() {
            self.metrics.first_scheduled_time = Some(time);
            self.metrics.time_in_queue = Some(time - self.metrics.arrival_time);
        }
    }

    /// Sets finished time
    #[allow(dead_code)]
    fn set_finished_time(&mut self, time: Instant) {
        self.metrics.finished_time = Some(time);
    }

    /// Gets the maximum number of sequences running in parallel, in the remaining lifetime of the request
    #[allow(dead_code)]
    fn get_max_num_running_seqs(&self) -> usize {
        if self.sampling_params.is_some() && self.sampling_params.as_ref().unwrap().use_beam_search
        {
            // For beam search, maximally there will always be `best_of` beam
            // candidates running in the future.
            return self.sampling_params.as_ref().unwrap().best_of;
        }
        if self.sampling_params.is_some()
            && self.sampling_params.as_ref().unwrap().best_of > self.get_num_sequences(None)
        {
            // At prompt stage, the sequence group is not yet filled up
            // and only have one sequence running. However, in the
            // generation stage, we will have `best_of` sequences running.
            return self.sampling_params.as_ref().unwrap().best_of;
        }
        // At sampling stages, return the number of actual sequences
        // that are not finished yet.
        self.num_unfinished_sequences()
    }

    /// Get sequences from `SequenceGroup`
    pub fn get_seqs(&self, status: Option<SequenceStatus>) -> Vec<Sequence> {
        match status {
            Some(ref status) => self
                .sequences
                .values()
                .filter_map(|seq| {
                    if seq.sequence_status.as_ref() == Some(status) {
                        Some(seq.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            None => self.sequences.values().cloned().collect(),
        }
    }

    /// Get a vector of unfinished sequences
    pub fn get_unfinished_sequences(&self) -> Vec<Sequence> {
        self.sequences
            .values()
            .filter(|s| !s.is_finished())
            .cloned()
            .collect()
    }

    /// Get a vector of finished sequences
    pub fn get_finished_sequences(&self) -> Vec<Sequence> {
        self.sequences
            .values()
            .filter(|s| s.is_finished())
            .cloned()
            .collect()
    }

    /// Updates the number of computed tokens
    pub fn update_num_computed_tokens(
        &mut self,
        num_new_computed_tokens: usize,
    ) -> Result<(), SequenceError> {
        for sequence in self.sequences.values_mut() {
            if !sequence.is_finished() {
                sequence
                    .sequence_data
                    .update_num_computed_tokens(num_new_computed_tokens)?;
            }
        }
        Ok(())
    }

    /// Get number of uncomputed tokens
    pub fn get_num_uncomputed_tokens(&self) -> usize {
        let mut num_uncomputed_tokens = 0;
        for sequence in self.sequences.values() {
            if !sequence.is_finished() {
                num_uncomputed_tokens += sequence.sequence_data.get_num_uncomputed_tokens();
            }
        }
        num_uncomputed_tokens
    }

    /// Number of sequences
    pub fn get_num_sequences(&self, status: Option<SequenceStatus>) -> usize {
        if status.is_none() {
            self.sequences.len()
        } else {
            self.get_seqs(status).len()
        }
    }

    pub fn num_unfinished_sequences(&self) -> usize {
        self.get_unfinished_sequences().len()
    }

    pub fn num_finished_sequences(&self) -> usize {
        self.get_finished_sequences().len()
    }

    /// Checks if it is in prefill phase, all sequences should either be or not in prefix phase, simultaneously
    pub fn is_prefill(&self) -> bool {
        self.sequences
            .iter()
            .next()
            .map(|(_, s)| s.is_prefill())
            .unwrap_or(false)
    }

    /// Finds a `Sequence` with a given `sequence_id`
    pub fn find(&self, sequence_id: u64) -> Option<Sequence> {
        self.sequences.get(&sequence_id).cloned()
    }

    /// Adds a new `Sequence` to the `SequenceGroup`
    pub fn add(&mut self, sequence: Sequence) {
        if self.sequences.contains_key(&sequence.sequence_id) {
            return;
        }
        self.sequences.insert(sequence.sequence_id, sequence);
    }

    /// Removes a `Sequence` from the `SequenceGroup`, as an idempotent
    pub fn remove(&mut self, sequence_id: u64) {
        self.sequences.remove(&sequence_id);
    }

    /// Checks if generation is finished for all `Sequence`'s in `SequenceGroup`
    pub fn is_finished(&self) -> bool {
        self.sequences.values().all(|s| s.is_finished())
    }
}

/// `SequenceGroupMetadata` - Metadata for a sequence group. Used to create `AttentionMetadata`
///
/// Args:
///     `request_id`: The ID of the request.
///     `is_prompt`: Whether the request is at prompt stage.
///     `sampling_params`: The sampling parameters used to generate the outputs.
///     `block_tables`: The block tables. (sequence id -> vector of physical block
///         numbers)
///     `do_sample`: True if sampling is required. Sampling is not required when
///          e.g., prefill is chunked, and the current iteration only computes
///          query tokens for prefill, we don't need sampling.
///     `token_chunk_size`: The number of tokens to be processed (per sequence).
///          None if chunking is not required.
///     `computed_block_nums`: The block numbers that are already computed,
///          used in prefix caching.
///     `state`: Internal state tied to this sequence group.
///     `multi_modal_data`: Multi modal data.
pub struct SequenceGroupMetadata {
    /// Request id
    #[allow(dead_code)]
    request_id: String,
    /// Is prompt (bool)
    #[allow(dead_code)]
    is_prompt: bool,
    /// Sampling parameters
    #[allow(dead_code)]
    sampling_params: SamplingParams,
    /// Block tables
    #[allow(dead_code)]
    block_tables: HashMap<u64, Vec<u64>>,
    /// Do sample (bool)
    #[allow(dead_code)]
    do_sample: bool,
    /// Token chunk size
    pub token_chunk_size: usize,
    /// Computed block numbers
    #[allow(dead_code)]
    computed_block_nums: Vec<u32>,
    /// Sequence data
    #[allow(dead_code)]
    sequence_data: HashMap<u64, SequenceData>,
    /// Internal state tied to this sequence group
    #[allow(dead_code)]
    state: SequenceGroupState,
    /// Optional multi model data
    #[allow(dead_code)]
    multi_modal_data: Option<MultiModalData>,
}

impl SequenceGroupMetadata {
    /// Constructor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        is_prompt: bool,
        sequence_data: HashMap<u64, SequenceData>,
        sampling_params: SamplingParams,
        block_tables: HashMap<u64, Vec<u64>>,
        do_sample: bool,
        token_chunk_size: Option<usize>,
        computed_block_nums: Vec<u32>,
        state: SequenceGroupState,
        multi_modal_data: Option<MultiModalData>,
    ) -> Self {
        let token_chunk_size = if let Some(size) = token_chunk_size {
            size
        } else if is_prompt {
            sequence_data
                .values()
                .next()
                .map(|s| s.length())
                .unwrap_or(0)
        } else {
            1
        };

        Self {
            request_id,
            is_prompt,
            sequence_data,
            sampling_params,
            block_tables,
            do_sample,
            token_chunk_size,
            computed_block_nums,
            state,
            multi_modal_data,
        }
    }
}

/// `SequenceOutput` - The model output associated with a sequence.
///
/// Args:
///     `parent_seq_id`: The ID of the parent sequence (for forking in beam
///         search).
///     `output_token`: The output token ID.
///     `logprobs`: The logprobs of the output token.
///         (Token id -> logP(x_i+1 | x_0, ..., x_i))
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceOutput {
    /// Parent sequence id
    pub parent_sequence_id: u64,
    /// Output token
    pub output_token: u32,
    /// Log probabilities
    pub logprob: HashMap<u64, LogProb>,
}

/// `CompletionSequenceGroupOutput` - The model output associated with a completion sequence group.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletionSequenceGroupOutput {
    pub samples: Vec<SequenceOutput>,
    pub prompt_logprobs: Vec<HashMap<u64, LogProb>>,
}

/// `SampledOutput` - For each sequence group, we generate a list of SequenceOutput object,
///     each of which contains one possible candidate for the next token.

/// This data structure implements methods, so it can be used like a list, but
///     also has optional fields for device tensors.
#[derive(Debug)]
pub struct SamplerOutput {
    /// Outputs
    pub outputs: Vec<CompletionSequenceGroupOutput>,
    /// Sampled token probabilities
    pub sampled_token_probs: Option<Tensor>,
    /// Log probabilities
    pub logprobs: Option<Tensor>,
    /// Sampled token ids
    pub sampled_token_ids: Option<Tensor>,
    /// Spec decoder worker metrics
    pub spec_decode_worker_metrics: Option<SpecDecodeWorkerMetrics>,
}

impl Index<usize> for SamplerOutput {
    type Output = CompletionSequenceGroupOutput;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.outputs[idx]
    }
}

impl IndexMut<usize> for SamplerOutput {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.outputs[idx]
    }
}

impl SamplerOutput {
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }
}

#[derive(Debug)]
/// `SpecDecoderWorkerMetrics`
pub struct SpecDecodeWorkerMetrics {
    /// The empirical acceptance rate of the proposal method on a per-token basis.
    /// This is useful for evaluating how well the proposal method aligns with the
    /// scoring method.
    pub draft_acceptance_rate: f32,

    /// The empirical efficiency, measured as the number of tokens emitted by the
    /// system divided by the number of tokens that could be emitted by the system
    /// if the proposal method were perfect.
    pub system_efficiency: f32,

    /// The number of speculative tokens produced by the proposal method.
    pub draft_tokens: i32,

    /// The number of tokens emitted by the entire system.
    pub emitted_tokens: i32,

    /// The number of tokens accepted by the scoring model and verification
    /// routine, e.g. Llama2-70B and lossless rejection sampling.
    ///
    /// NOTE: Any token accepted by the verification routine is considered
    /// accepted (regardless of if the speculative prefix is also accepted). The
    /// user will usually see less accepted tokens. This metric is helpful when
    /// evaluating alignment of the proposal method with the scoring model.
    pub accepted_tokens: i32,

    /// The number of speculative tokens per sequence.
    pub num_spec_tokens: i32,
}

/// `ExecuteModelRequest` - The model execution request
#[derive(Clone)]
pub struct ExecuteModelRequest {
    /// The sequence group metadata list
    #[allow(dead_code)]
    seq_group_metadata_list: Vec<Rc<SequenceGroupMetadata>>,
    /// Blocks to swap in. List of CPU -> GPU block number
    blocks_to_swap_in: Vec<(i32, i32)>,
    /// Blocks to swap out. List of GPU -> CPU block number
    blocks_to_swap_out: Vec<(i32, i32)>,
    /// Blocks to copy. Source to dest block
    blocks_to_copy: Vec<(i32, i32)>,
    /// The number of slots for lookahead decoding
    num_lookahead_slots: i32,
    /// The number of requests in the running queue
    running_queue_size: i32,
}

impl ExecuteModelRequest {
    /// Constructor
    pub fn new(
        seq_group_metadata_list: Vec<Rc<SequenceGroupMetadata>>,
        blocks_to_swap_in: Vec<(i32, i32)>,
        blocks_to_swap_out: Vec<(i32, i32)>,
        blocks_to_copy: Vec<(i32, i32)>,
        num_lookahead_slots: i32,
        running_queue_size: i32,
    ) -> Self {
        Self {
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            num_lookahead_slots,
            running_queue_size,
        }
    }

    /// Clone the request with a new sequence group metadata list.
    pub fn clone_with_seq_group_metadata_list(
        &self,
        seq_group_metadata_list: Vec<Rc<SequenceGroupMetadata>>,
    ) -> Self {
        Self {
            seq_group_metadata_list,
            blocks_to_swap_in: self.blocks_to_swap_in.clone(),
            blocks_to_swap_out: self.blocks_to_swap_out.clone(),
            blocks_to_copy: self.blocks_to_copy.clone(),
            num_lookahead_slots: self.num_lookahead_slots,
            running_queue_size: self.running_queue_size,
        }
    }
}

#[derive(Debug, Error)]
pub enum SequenceError {
    #[error("Invalid number of newly generated tokens for sequence")]
    InvalidNumberGeneratedTokens,
    #[error("Invalid last token generation while in prefix phase")]
    WhileInPrefix,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_outputs() -> Vec<CompletionSequenceGroupOutput> {
        (0..5)
            .map(|i| {
                let sequence_output = SequenceOutput {
                    parent_sequence_id: 0,
                    output_token: i,
                    logprob: HashMap::new(),
                };
                CompletionSequenceGroupOutput {
                    samples: vec![sequence_output],
                    prompt_logprobs: vec![],
                }
            })
            .collect()
    }

    fn sampler_output(sample_outputs: Vec<CompletionSequenceGroupOutput>) -> SamplerOutput {
        SamplerOutput {
            outputs: sample_outputs,
            sampled_token_ids: None,
            sampled_token_probs: None,
            logprobs: None,
            spec_decode_worker_metrics: None,
        }
    }

    /// Create a dummy prompt sequence and sequence group.
    fn create_dummy_prompt(
        request_id: u64,
        prompt_length: usize,
        block_size: Option<usize>,
        use_beam_search: bool,
        best_of: usize,
    ) -> (Sequence, SequenceGroup) {
        let block_size = block_size.unwrap_or(prompt_length);

        // Create dummy prompt sequence with tokens 0...block_size-1
        // and prompt "0 ... block_size".
        let prompt_tokens: Vec<u32> = (0..(prompt_length as u32)).collect();
        let prompt_str = prompt_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        let prompt = Sequence::new(
            request_id,
            Some(1000),
            prompt_str,
            prompt_tokens,
            block_size,
        );
        let seq_group = SequenceGroup::new(
            request_id.to_string(),
            vec![prompt.clone()],
            Instant::now(),
            None,
            Some(SamplingParams {
                use_beam_search,
                best_of,
                ..Default::default()
            }),
            vec![],
            SequenceGroupState {
                generator: Some(42),
            },
        );

        (prompt, seq_group)
    }

    #[test]
    fn test_sampler_output_initialization() {
        let sample_outputs = sample_outputs();
        let sampler_output = sampler_output(sample_outputs.clone());
        assert_eq!(sampler_output.outputs.len(), sample_outputs.len());
        assert!(sampler_output.logprobs.is_none());
        assert!(sampler_output.spec_decode_worker_metrics.is_none());
        assert!(sampler_output.sampled_token_ids.is_none());
        assert!(sampler_output.sampled_token_probs.is_none());
    }

    #[test]
    fn test_sampler_output_item() {
        let sample_outputs = sample_outputs();
        let mut sampler_output = sampler_output(sample_outputs.clone());
        assert_eq!(sampler_output[2], sample_outputs[2]);

        let sequence_output = SequenceOutput {
            parent_sequence_id: 0,
            output_token: 99,
            logprob: HashMap::new(),
        };
        let new_output = CompletionSequenceGroupOutput {
            samples: vec![sequence_output],
            prompt_logprobs: vec![],
        };
        sampler_output[2] = new_output.clone();
        assert_eq!(sampler_output[2], new_output)
    }

    #[test]
    fn test_sampler_output_len() {
        let sample_outputs = sample_outputs();
        let sampler_output = sampler_output(sample_outputs.clone());
        assert_eq!(sampler_output.len(), sample_outputs.len())
    }

    // #[test]
    // fn test_sampler_output_eq() {
    //     let sample_outputs = sample_outputs();
    //     let sampler_output1 = SamplerOutput {
    //         outputs: sample_outputs.clone(),
    //         sampled_token_ids: None,
    //         sampled_token_probs: None,
    //         logprobs: None,
    //         spec_decode_worker_metrics: None,
    //     };
    //     let sampler_output2 = SamplerOutput {
    //         outputs: sample_outputs.clone(),
    //         sampled_token_ids: None,
    //         sampled_token_probs: None,
    //         logprobs: None,
    //         spec_decode_worker_metrics: None,
    //     };
    //     let sampler_output3 = SamplerOutput {
    //         outputs: sample_outputs[..sample_outputs.len() - 1].to_vec(),
    //         sampled_token_ids: None,
    //         sampled_token_probs: None,
    //         logprobs: None,
    //         spec_decode_worker_metrics: None,
    //     };

    //     assert_eq!(sampler_output1, sampler_output2);
    //     assert_ne!(sampler_output1, sampler_output3)
    // }

    #[test]
    fn test_sequence_data_prefill() {
        let mut sequence_data = SequenceData::new(vec![1, 2, 3, 4], vec![]);
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 4);
        assert_eq!(sequence_data.get_num_computed_tokens(), 0);

        // advance by `2`
        sequence_data
            .update_num_computed_tokens(2)
            .expect("Failed to update");
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 2);
        assert_eq!(sequence_data.get_num_computed_tokens(), 2);

        // advance by `1`
        sequence_data
            .update_num_computed_tokens(1)
            .expect("Failed to update");
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 1);
        assert_eq!(sequence_data.get_num_computed_tokens(), 3);

        // append tokens and reset, simulating recompute
        sequence_data.add_token_id(1, 0.0);
        sequence_data.reset_state_for_recompute();
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 5);
        assert_eq!(sequence_data.get_num_computed_tokens(), 0);
    }

    #[test]
    fn test_sequence_group_stage() {
        let (_, mut seq_group) = create_dummy_prompt(1, 12, None, false, 5);
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(6)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(5)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(1)
            .expect("Failed to update");
        assert!(!seq_group.is_prefill());

        let seqs = seq_group.get_seqs(None);
        assert_eq!(seqs.len(), 1);

        seq_group
            .sequences
            .values_mut()
            .enumerate()
            .for_each(|(i, v)| {
                if i == 0 {
                    v.sequence_data.add_token_id(1, 0.0);
                }
            });
        seq_group
            .sequences
            .values_mut()
            .for_each(|v| v.reset_state_for_recompute());
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(5)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(7)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(1)
            .expect("Failed to update");
        assert!(!seq_group.is_prefill())
    }
}
