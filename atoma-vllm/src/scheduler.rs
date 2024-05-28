use std::{
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    time::{Duration, Instant},
};

use crate::{
    block_manager::{BlockSpaceManager, BlockSpaceManagerError},
    config::{CacheConfig, SchedulerConfig},
    policy::{FcfsPolicy, Policy},
    sequence::{Sequence, SequenceGroup, SequenceStatus},
};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, Span};

/// `SchedulingBudget` - The available slots for scheduling.
///
/// TODO: Right now, the budget is request_id-aware meaning it can ignore
///  budget update from the same request_id. It is because in normal scheduling
///  path, we update `Running` num_seqs ahead of time, meaning it could be
///  updated more than once when scheduling `Running` requests. Since this won't
///  happen if we only have chunked prefill scheduling, we can remove this
///  feature from the API when chunked prefill is enabled by default.
#[derive(Debug)]
struct SchedulingBudget {
    /// Token budget
    pub token_budget: usize,
    /// Maximum number of sequences
    pub max_num_sequences: usize,
    /// Set of request IDs that have updated num_batched_tokens.
    request_ids_num_batched_tokens: HashSet<String>,
    /// Set of request IDs that have updated num_curr_seqs.
    request_ids_num_curr_seqs: HashSet<String>,
    /// Number of batched tokens currently used.
    num_batched_tokens: usize,
    /// Number of current sequences.
    num_curr_seqs: usize,
    /// Tracing span
    pub span: Span,
}

impl SchedulingBudget {
    /// Creates a new `SchedulingBudget` with the specified token budget and maximum number of sequences.
    pub fn new(token_budget: usize, max_num_sequences: usize) -> Self {
        Self {
            token_budget,
            max_num_sequences,
            request_ids_num_batched_tokens: HashSet::new(),
            request_ids_num_curr_seqs: HashSet::new(),
            num_batched_tokens: 0,
            num_curr_seqs: 0,
            span: info_span!("scheduling-budget"),
        }
    }

    /// Checks if it is possible to schedule number of tokens
    #[instrument]
    pub fn can_schedule(
        &self,
        num_new_tokens: usize,
        num_new_sequences: usize,
    ) -> Result<bool, SchedulerError> {
        if num_new_sequences == 0 || num_new_tokens == 0 {
            error!("Empty scheduling, either `num_new_sequences` == 0 or `num_new_tokens` == 0");
            return Err(SchedulerError::EmptyScheduling);
        }

        Ok(
            (self.num_batched_tokens + num_new_tokens <= self.token_budget)
                && (self.num_curr_seqs + num_new_sequences <= self.max_num_sequences),
        )
    }

    /// Computes the remaining number of budget tokens
    pub fn remaining_budget_tokens(&self) -> usize {
        self.token_budget - self.num_batched_tokens
    }

    /// Adds number of batched tokens
    #[instrument]
    pub fn add_num_batched_tokens(&mut self, request_id: String, num_batched_tokens: usize) {
        info!("Adding number of batched tokens");
        // If request has already been batched, simply return
        if self.request_ids_num_batched_tokens.contains(&request_id) {
            return;
        }

        self.request_ids_num_batched_tokens.insert(request_id);
        self.num_batched_tokens += num_batched_tokens;
    }

    /// Subtracts number of batched tokens
    #[instrument]
    pub fn subtract_num_batched_tokens(&mut self, request_id: &str, num_batched_tokens: usize) {
        info!("Subtracting number of batched tokens..");
        // Only performs an action, if request with `request_id` has been already batched
        if self.request_ids_num_batched_tokens.contains(request_id) {
            self.request_ids_num_batched_tokens.remove(request_id);
            self.num_batched_tokens -= num_batched_tokens;
        }
    }

    /// Adds number sequences
    #[instrument]
    pub fn add_number_sequences(&mut self, request_id: String, num_current_sequences: usize) {
        info!("Adding number of sequences..");
        // If request has already been added, simply return
        if self.request_ids_num_curr_seqs.contains(&request_id) {
            return;
        }

        self.request_ids_num_curr_seqs.insert(request_id);
        self.num_curr_seqs += num_current_sequences;
    }

    /// Subtracts number sequences
    #[instrument]
    pub fn subtracts_number_sequences(&mut self, request_id: &str, num_current_sequences: usize) {
        info!("Subtracting number of sequences..");
        // Only performs an action, if request with `request_id` has been already added
        if self.request_ids_num_curr_seqs.contains(request_id) {
            self.request_ids_num_curr_seqs.remove(request_id);
            self.num_curr_seqs -= num_current_sequences;
        }
    }

    /// Number of batched tokens
    pub fn num_batched_tokens(&self) -> usize {
        self.num_batched_tokens
    }

    /// Number of current sequences
    pub fn num_current_sequences(&self) -> usize {
        self.num_curr_seqs
    }
}

/// `Scheduler` - Responsible for managing the schedule of incoming inference `SequenceGroup` requests
///
/// It handles processing multiple sequences, including tasks such as prefill (initial setup), decoding and swapping blocks from CPU <-> GPU.
/// It relies on `BlockSpaceManager` to efficiently allocate resources, schedule tasks, and handle preemption and swapping.
#[derive(Debug)]
pub struct Scheduler<P> {
    /// Cache configuration
    cache_config: CacheConfig,
    /// `Scheduler` configuration
    scheduler_config: SchedulerConfig,
    /// `BlockSpaceManager` to handle block resources efficiently
    block_manager: BlockSpaceManager,
    /// Waiting `SequenceGroup` queue
    waiting: VecDeque<SequenceGroup>,
    /// Running `SequenceGroup` queue
    running: VecDeque<SequenceGroup>,
    /// Swapped `SequenceGroup` queue
    swapped: VecDeque<SequenceGroup>,
    /// Time at previous scheduling step
    previous_time: Option<Instant>,
    /// Checks if we scheduled a prompt at previous steps
    previous_prompt: bool,
    /// Last prompt latency duration
    last_prompt_latency: Option<Duration>,
    /// Tracing span
    pub span: Span,
    /// Phantom data
    _phantom: PhantomData<P>,
}

impl<P> Scheduler<P> {
    /// Constructor
    pub fn new(
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, SchedulerError> {
        Ok(Self {
            block_manager: BlockSpaceManager::new(
                cache_config.block_size(),
                scheduler_config.device(),
                cache_config.num_cpu_blocks(),
                cache_config.num_gpu_blocks(),
                cache_config.sliding_window(),
            )?,
            cache_config,
            scheduler_config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped: VecDeque::new(),
            previous_time: None,
            previous_prompt: false,
            last_prompt_latency: None,
            span: info_span!("scheduler"),
            _phantom: PhantomData::default(),
        })
    }

    /// Number of new tokens, for each inference pass
    pub fn num_decoding_tokens_per_second(&self) -> usize {
        1
    }

    /// Aborts a sequence group with the given ID.
    ///
    /// Check if the sequence group with the given ID
    ///     is present in any of the state queue.
    /// If present, remove the sequence group from the state queue.
    /// Also, if any of the sequences in the sequence group is not finished,
    ///     free the sequence with status `FINISHED_ABORTED`.
    /// Otherwise, do nothing.
    pub fn abort_sequence_group(&mut self, request_id: String) -> Result<(), SchedulerError> {
        info!("Aborting sequence group..");

        if let Some(sequence_group) = self.waiting.iter().find(|s| s.request_id == request_id) {
            let sequences_ids = sequence_group
                .sequences
                .values()
                .filter_map(|s| {
                    if s.is_finished() {
                        Some(s.sequence_id())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            self.free_sequence(request_id.clone(), &sequences_ids, SequenceStatus::Waiting)?;
        }
        if let Some(sequence_group) = self.running.iter().find(|s| s.request_id == request_id) {
            let sequences_ids = sequence_group
                .sequences
                .values()
                .filter_map(|s| {
                    if s.is_finished() {
                        Some(s.sequence_id())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            self.free_sequence(request_id.clone(), &sequences_ids, SequenceStatus::Running)?;
        }
        if let Some(sequence_group) = self.swapped.iter().find(|s| s.request_id == request_id) {
            let sequences_ids = sequence_group
                .sequences
                .values()
                .filter_map(|s| {
                    if s.is_finished() {
                        Some(s.sequence_id())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            self.free_sequence(request_id, &sequences_ids, SequenceStatus::Swapped)?;
        }
        Ok(())
    }

    /// Frees blocks from a given `SequenceGroup`
    fn free_sequence(
        &mut self,
        request_id: String,
        sequences_ids: &[u64],
        sequence_status: SequenceStatus,
    ) -> Result<(), SchedulerError> {
        for sequence_id in sequences_ids {
            self.block_manager.free(*sequence_id)?
        }

        if sequence_status == SequenceStatus::Waiting {
            self.waiting.retain(|s| s.request_id != request_id);
        } else if sequence_status == SequenceStatus::Running {
            self.running.retain(|s| s.request_id != request_id);
        } else if sequence_status == SequenceStatus::Swapped {
            self.swapped.retain(|s| s.request_id != request_id);
        } else {
            unreachable!("Sequence status should only be one of values [Waiting, Running, Swapped]")
        }

        Ok(())
    }

    /// Gets number of unfinished sequences
    pub fn num_unfinished_sequeces(&self) -> usize {
        self.waiting.len() + self.running.len() + self.swapped.len()
    }
}

impl<P: Policy> Scheduler<P> {
    /// Has unfinished sequences
    pub fn has_unfinished_sequences(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }

    /// Schedule sequence groups that are running.
    ///
    /// Running queue should include decode and chunked prefill requests.
    ///
    /// Args:
    ///
    ///     running_queue: The queue that contains running requests (i.e.,
    ///         decodes). The given arguments are NOT in-place modified.
    ///     budget: The scheduling budget. The argument is in-place updated
    ///             when any decodes are preempted.
    ///     enable_chunking: If true, seq group can be chunked and only a
    ///             chunked number of tokens are scheduled  if
    ///             `budget.num_batched_tokens` has not enough capacity to schedule
    ///             all tokens.
    ///
    /// Returns:
    ///     A tuple of remaining running queue (should be always 0) after
    ///         scheduling and SchedulerRunningOutputs.
    fn schedule_running(
        &self,
        budget: &mut SchedulingBudget,
        running_queue: VecDeque<SequenceGroup>,
        enable_chunking: bool,
    ) {
        // Blocks that need to be swapped or copied before model execution
        let mut blocks_to_swap_out = HashMap::<usize, usize>::new();
        let mut blocks_to_copy = HashMap::<usize, usize>::new();

        let mut decode_seq_groups = Vec::<ScheduledSequenceGroup>::new();
        let mut prefill_seq_groups = Vec::<ScheduledSequenceGroup>::new();
        let mut preempted = Vec::<SequenceGroup>::new();
        let mut swapped_out = Vec::<SequenceGroup>::new();

        // Preemption happens only when there is no available slot
        // to keep all sequences groups in `Running` state.
        // In this case, the policy is responsible for deciding which sequence
        // groups should preempt next
        let now = Instant::now();
        let mut running_queue = P::sort_by_priority(now, &running_queue);

        while !running_queue.is_empty() {
            let sequence_group = running_queue.front().unwrap(); // DON'T PANIC: we have already checked that the `running_queue` is not empty
            let num_running_tokens = self.get_num_tokens(
                sequence_group,
                SequenceStatus::Running,
                enable_chunking,
                budget,
            );
        }
    }
}

impl<P> Scheduler<P> {
    /// Get the next new tokens to compute for a given sequence group
    /// that's in a given `status`.
    ///
    /// The API could chunk the number of tokens to compute based on `budget`
    /// if `enable_chunking` is true. If a sequence group has multiple
    /// sequences (e.g., running beam search), it means it is in the decoding
    /// phase, so chunking doesn't happen.
    ///
    /// Returns 0 if the new token cannot be computed due to token budget.
    fn get_num_tokens(
        &self,
        sequence_group: &SequenceGroup,
        sequence_status: SequenceStatus,
        enable_chunking: bool,
        budget: &mut SchedulingBudget,
    ) -> Result<usize, SchedulerError> {
        let mut num_new_tokens = 0;
        let mut num_sequences_in_status = 0;

        for (_, seq) in sequence_group.sequences.iter() {
            if seq.get_sequence_status() == sequence_status {
                num_new_tokens += seq.get_num_new_tokens();
                num_sequences_in_status += 1;
            }
        }

        if num_new_tokens == 0 {
            error!("No new tokens to be scheduled..");
            return Err(SchedulerError::ZeroNewTokensToSchedule);
        }

        // Chunk if a running request cannot fit in.
        // If the number of seqs > 1, it means it is doing beam search in a
        // decode phase. Do not chunk in that case.
        if enable_chunking && num_sequences_in_status == 1 {
            num_new_tokens = num_new_tokens.min(budget.remaining_budget_tokens());
        }

        Ok(num_new_tokens)
    }
}

/// A `SequenceGroup` that has been scheduled
struct ScheduledSequenceGroup {
    /// Sequence group
    scheduled_group: SequenceGroup,
    /// The total chunk size (number of tokens) to process for next iteration.
    /// 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    /// chunked, it can be smaller than that.
    token_chunk_size: usize,
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Block space manager error: `{0}`")]
    BlockSpaceManagerError(#[from] BlockSpaceManagerError),
    #[error("Empty scheduling")]
    EmptyScheduling,
    #[error("Zero number of new tokens to schedule")]
    ZeroNewTokensToSchedule,
}
