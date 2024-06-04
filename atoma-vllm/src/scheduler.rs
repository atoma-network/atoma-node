use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    marker::PhantomData,
    time::{Duration, Instant},
};

use crate::{
    block_manager::{AllocationStatus, BlockSpaceManager, BlockSpaceManagerError},
    config::{CacheConfig, SchedulerConfig},
    policy::Policy,
    sequence::{SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus},
};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, warn, Span};

/// Preemption modes.
///
/// 1. `Swapping`: Swap out the blocks of the preempted sequences to CPU memory
///     and swap them back in when the sequences are resumed.
/// 2. `Recomputation`: Discard the blocks of the preempted sequences and
///     recompute them when the sequences are resumed, treating the sequences as
///     new prompts.
#[derive(Debug, PartialEq, Eq)]
pub enum PreemptionMode {
    Swap,
    Recomputation,
}

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

/// `SchedulerRunningOutputs` - The requests that are scheduled from a running queue.
///
/// Could contain prefill (prefill that's chunked) or decodes. If there's not
/// enough memory, it can be preempted (for recompute) or swapped out.
pub struct SchedulerRunningOutputs {
    // Selected sequences that are running and in a decoding phase.
    decode_seq_groups: Vec<ScheduledSequenceGroup>,
    // Selected sequences that are running and in a prefill phase.
    // i.e., it means the prefill has been chunked.
    prefill_seq_groups: Vec<ScheduledSequenceGroup>,
    // The preempted sequences.
    preempted: Vec<SequenceGroup>,
    // Sequences that are swapped out.
    swapped_out: Vec<SequenceGroup>,
    // The blocks to swap out.
    blocks_to_swap_out: HashMap<u64, u64>,
    // The blocks to copy.
    blocks_to_copy: HashMap<u64, u64>,
}

impl SchedulerRunningOutputs {
    /// Create an empty `Self` instance
    fn create_empty() -> Self {
        Self {
            decode_seq_groups: vec![],
            prefill_seq_groups: vec![],
            preempted: vec![],
            swapped_out: vec![],
            blocks_to_swap_out: HashMap::new(),
            blocks_to_copy: HashMap::new(),
        }
    }
}

/// The requests that are scheduled from a swap queue.
///
/// Could contain prefill (prefill that's chunked) or decodes.
pub struct SchedulerSwappedInOutputs {
    /// Selected sequences that are going to be swapped in and is in a decoding phase.
    decode_seq_groups: Vec<ScheduledSequenceGroup>,
    /// Selected sequences that are going to be swapped in and in a prefill
    /// phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: Vec<ScheduledSequenceGroup>,
    /// The blocks to swap in.
    blocks_to_swap_in: HashMap<u64, u64>,
    /// The blocks to copy.
    blocks_to_copy: HashMap<u64, u64>,
    /// Infeasible sequence groups.
    infeasible_seq_groups: Vec<SequenceGroup>,
}

impl SchedulerSwappedInOutputs {
    /// Create an empty `Self` instance
    fn create_empty() -> Self {
        Self {
            decode_seq_groups: vec![],
            prefill_seq_groups: vec![],
            blocks_to_swap_in: HashMap::new(),
            blocks_to_copy: HashMap::new(),
            infeasible_seq_groups: vec![],
        }
    }
}

/// `SchedulerPrefillOutputs` - The requests that are scheduled from a waiting queue.
///
/// Could contain a fresh prefill requests or preempted requests that need
/// to be recomputed from scratch.
#[derive(Debug)]
pub struct SchedulerPrefillOutputs {
    /// Selected sequences for prefill
    sequence_groups: Vec<ScheduledSequenceGroup>,
    /// Ignored sequence groups.
    ignored_sequence_groups: Vec<SequenceGroup>,
}

impl SchedulerPrefillOutputs {
    /// Create an `empty` `Self` instance
    fn create_empty() -> Self {
        Self {
            sequence_groups: vec![],
            ignored_sequence_groups: vec![],
        }
    }
}

/// `SchedulerOutputs` - The scheduling decision made from a scheduler.
#[derive(Debug)]
pub struct SchedulerOutputs {
    // Scheduled sequence groups.
    scheduled_sequence_groups: Vec<ScheduledSequenceGroup>,
    // Number of prefill groups scheduled.
    number_prefill_groups: usize,
    // Total number of batched tokens.
    num_batched_tokens: usize,
    // Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: HashMap<u64, u64>,
    // Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: HashMap<u64, u64>,
    // Blocks to copy. Source to dest block.
    blocks_to_copy: HashMap<u64, u64>,
    // Ignored sequence groups
    ignored_seq_groups: Vec<SequenceGroup>,
    // The number of requests in the running queue
    running_queue_size: usize,
    // Number of preempted sequnce groups
    preempted: usize,
    // Tracing span
    span: Span,
}

impl SchedulerOutputs {
    /// Validate that `SchedulerOutputs` is well formed
    #[instrument]
    fn validate(&self) -> Result<(), SchedulerError> {
        if !self.blocks_to_swap_in.is_empty() && !self.blocks_to_swap_out.is_empty() {
            error!("Swap in and swap out should never happen at the same time.");
            return Err(SchedulerError::InvalidSchedulerOutput(
                "Swap in and swap out should never happen at the same time.".into(),
            ));
        }
        Ok(())
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
    previous_time: Instant,
    /// Checks if we scheduled a prompt at previous steps
    previous_prompt: bool,
    /// Last prompt latency duration
    last_prompt_latency: f32,
    /// Cumulative preemption
    num_cumulative_preemption: usize,
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
                cache_config.num_cpu_blocks(),
                cache_config.num_gpu_blocks(),
                cache_config.sliding_window(),
            )?,
            cache_config,
            scheduler_config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped: VecDeque::new(),
            previous_time: Instant::now(),
            previous_prompt: false,
            last_prompt_latency: 0.0,
            num_cumulative_preemption: 0,
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

        let mut queue_identifier = 'w';
        let waiting_length = self.waiting.len();
        let running_length = self.running.len();

        let mut sequence_ids_to_free = vec![];
        let mut index = 0;
        for sequence_group in self
            .waiting
            .iter()
            .chain(self.running.iter().chain(self.swapped.iter()))
        {
            if sequence_group.request_id == request_id {
                for sequence in sequence_group.sequences.values() {
                    let (sequence_id, is_finished) = {
                        (
                            sequence.borrow().sequence_id(),
                            sequence.borrow().is_finished(),
                        )
                    };
                    if is_finished {
                        continue;
                    }
                    sequence
                        .borrow_mut()
                        .set_sequence_status(SequenceStatus::FinishedAborted);
                    sequence_ids_to_free.push(sequence_id);
                }

                break;
            }
            index += 1;
        }

        for sequence_id in sequence_ids_to_free {
            self.free_sequence(sequence_id)?;
        }

        if index >= waiting_length && index < waiting_length + running_length {
            queue_identifier = 'r';
        } else if index > waiting_length + running_length {
            queue_identifier = 's';
        }

        if queue_identifier == 'w' {
            self.waiting.retain(|s| s.request_id != request_id);
        } else if queue_identifier == 'r' {
            self.running.retain(|s| s.request_id != request_id);
        } else {
            self.swapped.retain(|s| s.request_id == request_id);
        }

        Ok(())
    }

    /// Aborts multiple `SequenceGroup`'s at once
    pub fn abort_sequence_groups(
        &mut self,
        request_ids: impl Iterator<Item = String>,
    ) -> Result<(), SchedulerError> {
        for request_id in request_ids {
            self.abort_sequence_group(request_id)?;
        }

        Ok(())
    }

    /// Frees blocks from a given `SequenceGroup`
    fn free_sequences(
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

    /// Frees blocks from a given `Sequence` with `sequence_id`
    fn free_sequence(&mut self, sequence_id: u64) -> Result<(), SchedulerError> {
        Ok(self.block_manager.free(sequence_id)?)
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
    ///     running_queue - The queue that contains running requests (i.e.,
    ///         decodes). The given arguments are NOT in-place modified.
    ///     budget - The scheduling budget. The argument is in-place updated
    ///             when any decodes are preempted.
    ///     enable_chunking - If true, seq group can be chunked and only a
    ///             chunked number of tokens are scheduled  if
    ///             budget.num_batched_tokens has not enough capacity to schedule
    ///             all tokens.
    ///
    /// Returns:
    ///     A tuple of remaining running queue (should be always 0) after
    ///         scheduling and SchedulerRunningOutputs.
    fn schedule_running(
        &mut self,
        budget: &mut SchedulingBudget,
        running_queue: VecDeque<SequenceGroup>,
        enable_chunking: bool,
    ) -> Result<(VecDeque<SequenceGroup>, SchedulerRunningOutputs), SchedulerError> {
        info!("Schedule running..");
        // Blocks that need to be swapped or copied before model execution
        let mut blocks_to_swap_out = HashMap::<u64, u64>::new();
        let mut blocks_to_copy = HashMap::<u64, u64>::new();

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

        while let Some(mut sequence_group) = running_queue.pop_front() {
            let num_running_tokens = self.get_num_tokens(
                &sequence_group,
                SequenceStatus::Running,
                enable_chunking,
                budget,
            )?;

            // if no tokens are being processed, we break the loop
            if num_running_tokens == 0 {
                break;
            }

            loop {
                if !self.can_append_slots(&sequence_group) {
                    budget.subtract_num_batched_tokens(
                        &sequence_group.request_id,
                        num_running_tokens,
                    );
                    let num_running_sequences = sequence_group.get_max_num_running_seqs();
                    budget.subtracts_number_sequences(
                        &sequence_group.request_id,
                        num_running_sequences,
                    );

                    if !running_queue.is_empty() {
                        // Preempt the lowest-priority sequence groups first
                        // victim lies at the end of `runnning_queue`, as it is was last in, last out
                        let mut victim_sequence_group = running_queue.pop_back().unwrap(); // DON'T PANIC: already checked that `running_queue` is non-empty
                        let preempted_mode = self.preempt(
                            &mut victim_sequence_group,
                            &mut blocks_to_swap_out,
                            None,
                        )?;
                        if preempted_mode == PreemptionMode::Recomputation {
                            preempted.push(victim_sequence_group);
                        } else {
                            swapped_out.push(victim_sequence_group);
                        }
                    } else {
                        // No other sequence groups can be preempted.
                        // Preempt the current `SequenceGroup`
                        let preempted_mode =
                            self.preempt(&mut sequence_group, &mut blocks_to_swap_out, None)?;

                        if preempted_mode == PreemptionMode::Recomputation {
                            preempted.push(sequence_group.clone());
                        } else {
                            swapped_out.push(sequence_group.clone());
                        }

                        // As no other sequence groups can be preempted, we stop the loop
                        break;
                    }
                } else {
                    self.append_slots(&sequence_group, &mut blocks_to_copy)?;
                    let is_prefill = sequence_group.is_prefill();
                    if is_prefill {
                        // Prefill computation
                        prefill_seq_groups.push(ScheduledSequenceGroup {
                            scheduled_group: sequence_group.clone(),
                            token_chunk_size: num_running_tokens,
                        });
                    } else {
                        // Decoding computation (only decodes 1 token at a time)
                        decode_seq_groups.push(ScheduledSequenceGroup {
                            scheduled_group: sequence_group.clone(),
                            token_chunk_size: 1,
                        });
                    }
                    budget.add_num_batched_tokens(
                        sequence_group.request_id.clone(),
                        num_running_tokens,
                    );

                    // OPTIMIZATION: Note that `get_max_num_running_seqs` is
                    // expensive. For the default scheduling chase where
                    // `enable_chunking` is false, `num_seqs` are updated before running
                    // this method, so we don't have to update it again here.
                    if enable_chunking {
                        let num_running_seqs = sequence_group.get_max_num_running_seqs();
                        budget.add_number_sequences(
                            sequence_group.request_id.clone(),
                            num_running_seqs,
                        )
                    }
                    break;
                }
            }
        }

        let scheduler_running_outputs = SchedulerRunningOutputs {
            decode_seq_groups,
            prefill_seq_groups,
            preempted,
            swapped_out,
            blocks_to_swap_out,
            blocks_to_copy,
        };

        Ok((running_queue, scheduler_running_outputs))
    }

    /// Schedule sequence groups that are swapped out.
    ///
    /// It schedules swapped requests as long as it fits `budget`. The input arguments
    /// `budget` and are updated based on scheduled sequence_groups.
    ///
    /// Args:
    ///     swapped_queue: The queue that contains swapped out requests. The given arguments are NOT in-place modified.
    ///     budget: The scheduling budget. The argument is in-place updated
    ///         when any requests are swapped in.
    ///     policy: The sorting policy to sort swapped_queue.
    ///     enable_chunking: If true, seq group can be chunked and only a
    ///         chunked number of tokens are scheduled  if budget.num_batched_tokens has not enough capacity to schedule all tokens.
    ///
    /// Returns:
    ///     A tuple of remaining `swapped_queue` after scheduling and
    ///     SchedulerSwappedInOutputs.
    #[instrument]
    fn schedule_swapped(
        &mut self,
        budget: &mut SchedulingBudget,
        swapped_queue: VecDeque<SequenceGroup>,
        enable_chunking: bool,
    ) -> Result<(VecDeque<SequenceGroup>, SchedulerSwappedInOutputs), SchedulerError> {
        info!("Schedule swapped..");
        // Blocks that need to be swapped or copied before model execution.
        let mut blocks_to_swap_in = HashMap::<u64, u64>::new();
        let mut blocks_to_copy = HashMap::<u64, u64>::new();
        let mut decode_seq_groups = Vec::<ScheduledSequenceGroup>::new();
        let mut prefill_seq_groups = Vec::<ScheduledSequenceGroup>::new();

        let now = Instant::now();

        let mut swapped_queue = P::sort_by_priority(now, &swapped_queue);
        let mut infeasible_seq_groups = Vec::<SequenceGroup>::new();

        while !swapped_queue.is_empty() {
            let mut sequence_group = swapped_queue.pop_front().unwrap(); // DON'T PANIC: we are guaranteed that `swapped_queue` is non-empty at this point

            // If the sequence group cannot be swapped in, stop.
            let allocation_status = self.block_manager.can_swap_in(&sequence_group)?;
            if allocation_status == AllocationStatus::Later {
                break;
            } else if allocation_status == AllocationStatus::Never {
                warn!("Failing the request {} because there is not enough KV cache blocks to run the entire sequence..", 
                        sequence_group.request_id);
                for (_, sequence) in sequence_group.sequences.iter_mut() {
                    sequence
                        .borrow_mut()
                        .set_sequence_status(SequenceStatus::FinishedIgnored);
                }
                infeasible_seq_groups.push(sequence_group.clone());
            }

            // The total number of sequences in the RUNNING state should not
            // exceed the maximum number of sequences.
            let num_new_sequences = sequence_group.get_max_num_running_seqs();
            let num_new_tokens = self.get_num_tokens(
                &sequence_group,
                SequenceStatus::Swapped,
                enable_chunking,
                budget,
            )?;

            if num_new_tokens == 0 || !budget.can_schedule(num_new_tokens, num_new_sequences)? {
                info!("Either no new tokens to be swapped or no available budget to swap tokens");
                break;
            }

            self.swap_in(&mut sequence_group, &mut blocks_to_swap_in)?;
            self.append_slots(&sequence_group, &mut blocks_to_copy)?;
            let is_preffil = sequence_group.is_prefill();
            if is_preffil {
                prefill_seq_groups.push(ScheduledSequenceGroup {
                    scheduled_group: sequence_group.clone(),
                    token_chunk_size: num_new_tokens,
                })
            } else {
                decode_seq_groups.push(ScheduledSequenceGroup {
                    scheduled_group: sequence_group.clone(),
                    token_chunk_size: 1,
                })
            }

            budget.add_num_batched_tokens(sequence_group.request_id.clone(), num_new_tokens);
            budget.add_number_sequences(sequence_group.request_id.clone(), num_new_sequences);
        }

        Ok((
            swapped_queue,
            SchedulerSwappedInOutputs {
                decode_seq_groups,
                prefill_seq_groups,
                blocks_to_swap_in,
                blocks_to_copy,
                infeasible_seq_groups,
            },
        ))
    }

    /// Schedule sequence groups that are in prefill stage.
    ///
    /// Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
    /// as a new prefill (that starts from beginning -> most recently generated
    ///    tokens).
    ///
    /// Args:
    ///     waiting_queue: The queue that contains prefill requests.
    ///         The given arguments are NOT in-place modified.
    ///     budget: The scheduling budget. The argument is in-place updated
    ///         when any requests are scheduled.
    ///     enable_chunking: If True, seq group can be chunked and only a
    ///         chunked number of tokens are scheduled  if
    ///         budget.num_batched_tokens has not enough capacity to schedule
    ///         all tokens.
    ///
    /// Returns:
    ///     A tuple of remaining waiting_queue after scheduling and
    ///         SchedulerSwappedInOutputs,
    #[instrument]
    fn schedule_prefills(
        &mut self,
        mut waiting_queue: VecDeque<SequenceGroup>,
        budget: &mut SchedulingBudget,
        enable_chunking: bool,
    ) -> Result<(VecDeque<SequenceGroup>, SchedulerPrefillOutputs), SchedulerError> {
        info!("Schedulig prefills..");

        let mut ignored_sequence_groups = Vec::<SequenceGroup>::new();
        let mut sequence_groups = Vec::<ScheduledSequenceGroup>::new();

        // We don't sort `waiting_queue` because we assume it is sorted. We also require
        // ownership of `waiting_queue` so that we don't change it in place, in this method.

        while !waiting_queue.is_empty() && self.passed_delay(Instant::now()) {
            // DON'T PANIC: at this point, we are guaranteed that `waiting_queue` is non-empty
            let mut sequence_group = waiting_queue.pop_front().unwrap();

            // To be used below
            let can_allocate = self.block_manager.can_allocate(&sequence_group);
            let num_new_tokens = self.get_num_tokens(
                &sequence_group,
                SequenceStatus::Waiting,
                enable_chunking,
                budget,
            )?;

            let mut waiting_sequences = sequence_group
                .sequences
                .iter()
                .filter_map(|(_, s)| {
                    if s.borrow().get_sequence_status() == SequenceStatus::Waiting {
                        Some(s)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            if waiting_sequences.len() != 1 {
                error!("Waiting sequence group should have only one prompt sequence, it has {} for request = {}.", waiting_sequences.len(), sequence_group.request_id);
                return Err(SchedulerError::InvalidNumberWaitingSequence {
                    request_id: sequence_group.request_id.clone(),
                    num_sequences: waiting_sequences.len(),
                });
            }

            if !enable_chunking {
                // DON'T PANIC: by previous error check, we are guaranteed that `waiting_sequences` is non-empty
                let num_prompt_tokens = waiting_sequences.first().unwrap().borrow().length();
                if num_prompt_tokens != num_new_tokens {
                    error!("Invalid number of new tokens, got `{num_new_tokens}`, but it should be `{num_prompt_tokens}`");
                    return Err(SchedulerError::InvalidNumberOfNewTokens {
                        num_prompt_tokens,
                        num_new_tokens,
                    });
                }
            }

            let prompt_limit = self.get_prompt_limit();
            if num_new_tokens > prompt_limit {
                warn!(
                    "Input prompt ({} tokens) is too long and exceeds limits of {}",
                    num_new_tokens, prompt_limit
                );
                for (_, sequence) in sequence_group.sequences.iter_mut() {
                    sequence
                        .borrow_mut()
                        .set_sequence_status(SequenceStatus::FinishedIgnored)
                }
                ignored_sequence_groups.push(sequence_group.clone());
                continue;
            }

            // If the sequence cannot be allocated, just stop
            if can_allocate == AllocationStatus::Later {
                waiting_queue.push_front(sequence_group);
                break;
            } else if can_allocate == AllocationStatus::Never {
                warn!("Input prompt ({num_new_tokens} tokens) is too long and exceeds the capacity of `block_manager`");
                for sequence in waiting_sequences.iter_mut() {
                    sequence
                        .borrow_mut()
                        .set_sequence_status(SequenceStatus::FinishedIgnored);
                }
                ignored_sequence_groups.push(sequence_group.clone());
            }

            let num_new_sequences = sequence_group.get_max_num_running_seqs();
            if num_new_sequences == 0 || !budget.can_schedule(num_new_tokens, num_new_sequences)? {
                // Push the sequence group back to the `waiting_queue`
                waiting_queue.push_front(sequence_group);
                break;
            }

            // At this point, we can schedule this request
            self.allocate_and_set_running(&mut sequence_group)?;
            sequence_groups.push(ScheduledSequenceGroup {
                scheduled_group: sequence_group.clone(),
                token_chunk_size: num_new_tokens,
            });

            budget.add_num_batched_tokens(sequence_group.request_id.clone(), num_new_tokens);
            budget.add_number_sequences(sequence_group.request_id.clone(), num_new_sequences);
        }

        if sequence_groups.len() > 0 {
            self.previous_prompt = true;
        }

        Ok((
            waiting_queue,
            SchedulerPrefillOutputs {
                sequence_groups,
                ignored_sequence_groups,
            },
        ))
    }

    /// Schedule queued requests.
    ///
    /// The current policy is designed to optimize the throughput. First,
    /// it batches as many prefill requests as possible. And it schedules
    ///  decodes. If there's a pressure on GPU memory, decode requests can
    /// be swapped or preempted.
    #[instrument]
    fn schedule_default(&mut self) -> Result<SchedulerOutputs, SchedulerError> {
        info!("Scheduling default..");
        // Include running requests to the budget.
        let mut budget = SchedulingBudget::new(
            self.scheduler_config.max_num_batched_tokens(),
            self.scheduler_config.max_num_sequences(),
        );

        // Make sure we include num running seqs before scheduling prefill
        for sequence_group in self.running.iter() {
            budget.add_number_sequences(
                sequence_group.request_id.clone(),
                sequence_group.get_max_num_running_seqs(),
            );
        }

        let mut remaining_running = self.running.clone();
        let mut remaining_waiting = self.waiting.clone();
        let mut remaining_swapped = self.swapped.clone();

        let mut prefills = SchedulerPrefillOutputs::create_empty();
        let mut running_scheduled = SchedulerRunningOutputs::create_empty();
        let mut swapped_in = SchedulerSwappedInOutputs::create_empty();

        // If any requests are swapped, prioritized swapped requests
        if self.swapped.is_empty() {
            // NOTE: we don't mutate `self.waiting` in place, instead we clone the `waiting` queue
            (remaining_waiting, prefills) =
                self.schedule_prefills(remaining_waiting, &mut budget, false)?;
        }

        // Don't schedule decodes if prefills are scheduled.
        // NOTE: If `schedule_prefills` doesn't enable chunking, `self.running`
        // only contains decode requests, not chunked prefills.
        if prefills.sequence_groups.len() == 0 {
            // NOTE: we don't mutate `self.running` in place, instead we clone the `running` queue
            (remaining_running, running_scheduled) =
                self.schedule_running(&mut budget, remaining_running, false)?;

            // If any sequence group is preempted, do not swap in any sequence
            // group, because it means there's no slot for new running requests
            if running_scheduled.preempted.len() + running_scheduled.swapped_out.len() == 0 {
                (remaining_swapped, swapped_in) =
                    self.schedule_swapped(&mut budget, remaining_swapped, false)?
            }
        }

        if budget.num_batched_tokens > self.scheduler_config.max_num_batched_tokens() {
            error!("Number of budget batched tokens exceeds the configured number of max batched tokens");
            return Err(SchedulerError::InvalidNumberBudgetTokens(
                    "Number of budget batched tokens exceeds the configured number of max batched tokens".into()
                ));
        }

        if budget.num_current_sequences() > self.scheduler_config.max_num_sequences() {
            error!("Number of budget sequences exceed the configured number of max number of sequences");
            return Err(SchedulerError::InvalidNumberBudgetSequences(
                "Number of budget sequences exceed the configured number of max number of sequences".into()
            ));
        }

        // To be used later for method output
        let preempted = running_scheduled.preempted.len() + running_scheduled.swapped_out.len();

        // Update waiting requests
        self.waiting = remaining_waiting;
        // NOTE: need to reverse order of preempted sequence groups to preserve order once you push these
        // to the left on the `self.waiting` queue.
        // NOTE: Preempted running scheduled means there was not enough block space to be run on the
        // current inference loop, so these requests should have priority regarding newly received
        // requests.
        running_scheduled
            .preempted
            .iter()
            .rev()
            .for_each(|s| self.waiting.push_front(s.clone()));
        // Update new running requests
        self.running = remaining_running;
        // NOTE: newly prefill requests get appended first, then decoding ones
        self.running.extend(
            prefills
                .sequence_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            running_scheduled
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            swapped_in
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );

        // Update swapped requests
        self.swapped = remaining_swapped;
        self.swapped.extend(running_scheduled.swapped_out);

        // There should be no prefill from running queue because this policy
        // doesn't allow chunked prefills.
        if running_scheduled.prefill_seq_groups.len() != 0 {
            error!("Chunked prefills are not allowed for running schedules, there should be none but we received {}", running_scheduled.prefill_seq_groups.len());
            return Err(SchedulerError::ChunkedPrefillsNotAllowed(format!(
                "Chunked prefills are not allowed for running schedules, there should be none but we received {}",
                running_scheduled.prefill_seq_groups.len()
            )));
        }

        if swapped_in.prefill_seq_groups.len() != 0 {
            error!(
                "Chunked prefills are not allowed for swapped in schedules, there should be none but we received {}", running_scheduled.prefill_seq_groups.len()
            );
            return Err(SchedulerError::ChunkedPrefillsNotAllowed(
                format!("Chunked prefills are not allowed for swapped in schedules, there should be none but we received {}", swapped_in.prefill_seq_groups.len()),
            ));
        }

        let number_prefill_groups = prefills.sequence_groups.len();

        let scheduled_sequence_groups: Vec<ScheduledSequenceGroup> = prefills
            .sequence_groups
            .into_iter()
            .chain(
                running_scheduled
                    .decode_seq_groups
                    .into_iter()
                    .chain(swapped_in.decode_seq_groups.into_iter()),
            )
            .collect();

        let blocks_to_copy = running_scheduled
            .blocks_to_copy
            .into_iter()
            .chain(swapped_in.blocks_to_copy.into_iter())
            .collect();

        let ignored_seq_groups = prefills
            .ignored_sequence_groups
            .into_iter()
            .chain(swapped_in.infeasible_seq_groups.into_iter())
            .collect();

        // NOTE: `SchedulerOutputs` should only be used by read operations, as
        // contrary to original Python vllm implementation, `SchedulerOutputs` is passed
        // by value, and not by reference
        Ok(SchedulerOutputs {
            scheduled_sequence_groups,
            num_batched_tokens: budget.num_batched_tokens,
            number_prefill_groups,
            blocks_to_swap_in: swapped_in.blocks_to_swap_in,
            blocks_to_swap_out: running_scheduled.blocks_to_swap_out,
            blocks_to_copy,
            ignored_seq_groups,
            running_queue_size: self.running.len(),
            preempted,
            span: info_span!("scheduler-outputs"),
        })
    }

    /// Schedule queued requests.
    ///
    /// Chunked prefill allows to chunk prefill requests, batch them together
    /// with decode requests. This policy 1. schedule as many decoding requests
    /// as possible. 2. schedule chunked prefill requests that are not
    /// finished. 3. schedule swapped request. 4. schedule new prefill
    /// requests.
    ///
    /// The policy can sustain the high GPU utilization because it can put
    /// prefill and decodes requests to the same batch, while it improves
    /// inter token latency because decodes requests don't need to blocked
    /// by prefill requests.
    #[instrument]
    fn schedule_chunked_prefill(&mut self) -> Result<SchedulerOutputs, SchedulerError> {
        let mut budget = SchedulingBudget::new(
            self.scheduler_config.max_num_batched_tokens(),
            self.scheduler_config.max_num_sequences(),
        );

        let mut remaining_waiting = self.waiting.clone();
        let mut remaining_running = self.running.clone();
        let mut remaining_swapped = self.swapped.clone();

        let mut prefills = SchedulerPrefillOutputs::create_empty();
        let mut running_scheduled = SchedulerRunningOutputs::create_empty();
        let mut swapped_in = SchedulerSwappedInOutputs::create_empty();

        // Decoding should be always scheduled first by fcfs
        (remaining_running, running_scheduled) =
            self.schedule_running(&mut budget, remaining_running, true)?;

        // Schedule swapped out requests.
        // If preemption happens, it means we don't have space for swap in
        if running_scheduled.preempted.len() + running_scheduled.swapped_out.len() == 0 {
            (remaining_swapped, swapped_in) =
                self.schedule_swapped(&mut budget, remaining_swapped, true)?;
        }

        // Schedule new prefills.
        (remaining_waiting, prefills) =
            self.schedule_prefills(remaining_waiting, &mut budget, true)?;

        if budget.num_batched_tokens() > self.scheduler_config.max_num_batched_tokens() {
            error!("Number of budget batched tokens exceeds the configured number of max batched tokens");
            return Err(SchedulerError::InvalidNumberBudgetTokens(
                    "Number of budget batched tokens exceeds the configured number of max batched tokens".into()
                ));
        }

        if budget.num_current_sequences() > self.scheduler_config.max_num_sequences() {
            error!("Number of budget sequences exceed the configured number of max number of sequences");
            return Err(SchedulerError::InvalidNumberBudgetSequences(
                "Number of budget sequences exceed the configured number of max number of sequences".into()
            ));
        }

        // To be used later, on the output
        let preempted = running_scheduled.preempted.len() + running_scheduled.swapped_out.len();

        // Update waiting queue
        self.waiting = remaining_waiting;
        // NOTE: need to reverse order of preempted sequence groups to preserve order once you push these
        // to the left on the `self.waiting` queue.
        // NOTE: Preempted running scheduled means there was not enough block space to be run on the
        // current inference loop, so these requests should have priority regarding newly received
        // requests.
        running_scheduled
            .preempted
            .iter()
            .rev()
            .for_each(|s| self.waiting.push_front(s.clone()));

        // Update new running requests
        self.running = remaining_running;
        // NOTE: newly prefill requests get appended first, then decoding ones
        self.running.extend(
            prefills
                .sequence_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            running_scheduled
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            swapped_in
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            swapped_in
                .prefill_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );

        // Update swapped requests
        self.swapped = remaining_swapped;
        self.swapped.extend(running_scheduled.swapped_out);

        let number_prefill_groups = prefills.sequence_groups.len()
            + swapped_in.prefill_seq_groups.len()
            + running_scheduled.prefill_seq_groups.len();
        let scheduled_sequence_groups = prefills
            .sequence_groups
            .into_iter()
            .chain(
                running_scheduled.prefill_seq_groups.into_iter().chain(
                    swapped_in.prefill_seq_groups.into_iter().chain(
                        running_scheduled
                            .decode_seq_groups
                            .into_iter()
                            .chain(swapped_in.decode_seq_groups.into_iter()),
                    ),
                ),
            )
            .collect();
        let blocks_to_copy = running_scheduled
            .blocks_to_copy
            .into_iter()
            .chain(swapped_in.blocks_to_copy)
            .collect();
        let ignored_seq_groups = prefills.ignored_sequence_groups;

        Ok(SchedulerOutputs {
            scheduled_sequence_groups,
            num_batched_tokens: budget.num_batched_tokens(),
            number_prefill_groups,
            blocks_to_swap_in: swapped_in.blocks_to_swap_in,
            blocks_to_swap_out: running_scheduled.blocks_to_swap_out,
            blocks_to_copy,
            ignored_seq_groups,
            running_queue_size: self.running.len(),
            preempted,
            span: info_span!("scheduler-outputs"),
        })
    }

    /// Schedule queued requests.
    #[instrument]
    fn schedule_(&mut self) -> Result<SchedulerOutputs, SchedulerError> {
        if self.scheduler_config.enable_chunked_prefill() {
            self.schedule_chunked_prefill()
        } else {
            self.schedule_default()
        }
    }

    /// Schedule queued requests, in the form of `SequenceGroup`'s.
    /// This function calls the internal state of the `Scheduler`
    #[instrument]
    pub fn schedule(
        &mut self,
    ) -> Result<(Vec<SequenceGroupMetadata>, SchedulerOutputs), SchedulerError> {
        let scheduler_outputs = self.schedule_()?;
        let now = Instant::now();

        // Create input data structures
        let mut sequence_groups_metadata = Vec::new();
        for scheduled_sequence_group in scheduler_outputs.scheduled_sequence_groups.iter() {
            let mut sequence_group = scheduled_sequence_group.scheduled_group.clone();
            let token_chunk_size = scheduled_sequence_group.token_chunk_size;
            sequence_group.maybe_set_first_scheduled_time(now);

            // Mapping from sequence id to `SequenceData`
            let mut sequence_data = HashMap::<u64, SequenceData>::new();
            // Mapping from sequence id to `PhysicalBlock` number
            let mut block_tables = HashMap::<u64, Vec<u64>>::new();

            for sequence in sequence_group.sequences.iter().filter_map(|(_, s)| {
                if s.borrow().get_sequence_status() == SequenceStatus::Running {
                    Some(s)
                } else {
                    None
                }
            }) {
                let sequence_id = sequence.borrow().sequence_id();
                sequence_data.insert(sequence_id, sequence.borrow().sequence_data());
                if let Some(block_table_ids) = self.block_manager.get_block_table_ids(&sequence_id)
                {
                    block_tables.insert(sequence_id, block_table_ids);
                    self.block_manager
                        .access_all_blocks_in_sequence(&sequence_id, now)?;
                } else {
                    error!(
                        "Missing block table for sequence with id = {}",
                        sequence.borrow().sequence_id()
                    );
                }
            }

            let mut do_sample = true;
            if sequence_group.is_prefill() {
                if sequence_group.sequences.len() != 1 {
                    error!("Prefill mode has only one sequence");
                    return Err(SchedulerError::InvalidPrefillSequences(
                        "Prefill mode has only one sequence".into(),
                    ));
                }

                // DON'T PANIC: checked previously that `sequence_group.sequences.len() != 1`
                let sequence = sequence_group
                    .sequences
                    .iter()
                    .map(|(_, s)| s)
                    .next()
                    .unwrap();

                // In the next iteration, all prompt tokens are not computed.
                // It means the prefill is chunked, and we don't need sampling.
                // NOTE: We use get_len instead of get_prompt_len because when
                // a sequence is preempted, prefill includes previous generated
                // output tokens.
                if token_chunk_size + sequence.borrow().sequence_data().get_num_computed_tokens()
                    < sequence.borrow().sequence_data().length()
                {
                    do_sample = false;
                }
            }

            let multi_modal_data = if scheduler_outputs.number_prefill_groups > 0 {
                sequence_group.multi_modal_data()
            } else {
                None
            };

            // It assumes the scheduled_seq_groups is ordered by
            // prefill < decoding.
            let is_prompt = sequence_group.is_prefill();
            let sequence_group_metadata = SequenceGroupMetadata::new(
                sequence_group.request_id.clone(),
                is_prompt,
                sequence_data,
                sequence_group.sampling_params(),
                block_tables,
                do_sample,
                Some(token_chunk_size),
                sequence_group.state(),
                multi_modal_data,
            );
            sequence_groups_metadata.push(sequence_group_metadata);
        }

        // TODO: remove this code if not necessary
        // // Now that the batch has been created, we can assume all blocks in the
        // // batch will have been computed before the next scheduling invocation.
        // // This is because the engine assumes that a failure in model execution
        // //  will crash the vLLM instance / will not retry.
        // for scheduled_seq_group in scheduler_outputs.scheduled_sequence_groups.iter() {
        //     self.block_manager.mark_blocks_as_computed(
        //         scheduled_seq_group.scheduled_group)
        // }

        Ok((sequence_groups_metadata, scheduler_outputs))
    }
}

impl<P: Debug> Scheduler<P> {
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
            if seq.borrow().get_sequence_status() == sequence_status {
                num_new_tokens += seq.borrow().get_num_new_tokens();
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

    /// Determine whether or not we have enough space in the KV cache to
    /// continue generation of the sequence group.
    fn can_append_slots(&self, sequence_group: &SequenceGroup) -> bool {
        self.block_manager.can_append_slots(sequence_group)
    }

    /// Appends new slots to the sequences in the given sequence group.
    ///
    /// Args:
    /// `sequence_group`: The sequence group containing the
    /// sequences to append slots to.
    /// `blocks_to_copy`: Mapping of source block index to destination block index.
    ///     It is updated with the new source and destination block indices for the appended
    ///     slots.
    #[instrument]
    fn append_slots(
        &mut self,
        sequence_group: &SequenceGroup,
        blocks_to_copy: &mut HashMap<u64, u64>,
    ) -> Result<(), SchedulerError> {
        info!(
            "Appending slot to sequence group with id = {}",
            sequence_group.request_id
        );
        let running_sequences = sequence_group.sequences.iter().filter_map(|(_, s)| {
            if s.borrow().get_sequence_status() == SequenceStatus::Running {
                Some(s)
            } else {
                None
            }
        });
        for sequence in running_sequences {
            let cows = self.block_manager.append_slots(sequence.borrow())?;
            if let Some(cow) = cows {
                blocks_to_copy.insert(cow.0, cow.1);
            } else {
                warn!("No Copy on Write new blocks to append, for sequence with id = {} in sequence group with id = {}", 
                    sequence.borrow().sequence_id(), sequence_group.request_id);
            }
        }
        Ok(())
    }

    /// Adds new `SequenceGroup`'s to `waiting` queue
    pub fn add_sequence_group(&mut self, sequence_group: SequenceGroup) {
        self.waiting.push_back(sequence_group)
    }

    /// Allows for preemption of `SequenceGroup`
    #[instrument]
    fn preempt(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_out: &mut HashMap<u64, u64>,
        preemption_mode: Option<PreemptionMode>,
    ) -> Result<PreemptionMode, SchedulerError> {
        // If preemption mode is not specified, we determine the mode as follows:
        // We use recomputation by default since it incurs lower overhead than
        // swapping. However, when the sequence group has multiple sequences
        // (e.g., beam search), recomputation is not currently supported. In
        // such a case, we use swapping instead.
        // FIXME: This makes our scheduling policy a bit bizarre.
        // As swapped sequences are prioritized over waiting sequences,
        // sequence groups with multiple sequences are implicitly prioritized
        // over sequence groups with a single sequence.
        // TODO: Support recomputation for sequence groups with multiple
        // sequences. This may require a more sophisticated CUDA kernel.
        let preemption_mode = if preemption_mode.is_none() {
            if sequence_group.get_max_num_running_seqs() == 1 {
                PreemptionMode::Recomputation
            } else {
                PreemptionMode::Swap
            }
        } else {
            preemption_mode.unwrap()
        };

        if self.num_cumulative_preemption % 50 == 0 {
            warn!("Sequence group with id = {} is preempted by {:?} mode because there is not enough KV cache space. 
                    This can affect the end-to-end performance. Increase `gpu_memory_utilization` or `tensor_parallel_size` 
                    to provide more KV cache memory. `total_num_cumulative_preemption = {}` ", 
                    sequence_group.request_id, preemption_mode, self.num_cumulative_preemption + 1);
        }
        self.num_cumulative_preemption += 1;

        if preemption_mode == PreemptionMode::Recomputation {
            self.preempt_by_recompute(sequence_group)?;
        } else if preemption_mode == PreemptionMode::Swap {
            self.preempt_by_swap(sequence_group, blocks_to_swap_out)?;
        } else {
            unreachable!("Preemption mode not supported");
        }

        Ok(preemption_mode)
    }

    /// Preempts a `SequenceGroup` by `Recomputation` mode
    #[instrument]
    fn preempt_by_recompute(
        &mut self,
        sequence_group: &mut SequenceGroup,
    ) -> Result<(), SchedulerError> {
        info!(
            "Preemption by recomputation for sequence group with id = {}",
            sequence_group.request_id
        );
        let sequences = sequence_group
            .sequences
            .iter_mut()
            .filter_map(|(_, s)| {
                if s.borrow().get_sequence_status() == SequenceStatus::Running {
                    Some(s)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if sequences.len() != 1 {
            error!("Number of sequences in `SequenceGroup` for preempt by recompute should be 1, but is {}", sequences.len());
            return Err(SchedulerError::InvalidNumberSequencesForRecompute(
                sequences.len(),
            ));
        }

        for sequence in sequences {
            {
                sequence
                    .borrow_mut()
                    .set_sequence_status(SequenceStatus::Waiting);
            }
            {
                self.free_sequence(sequence.borrow().sequence_id())?;
            }
            {
                sequence.borrow_mut().reset_state_for_recompute();
            }
        }

        Ok(())
    }

    /// Preempts a `SequenceGroup` by `Swap` mode
    #[instrument]
    fn preempt_by_swap(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_out: &mut HashMap<u64, u64>,
    ) -> Result<(), SchedulerError> {
        info!(
            "Preemption by swap for sequence group with id = {}..",
            sequence_group.request_id
        );

        self.swap_out(sequence_group, blocks_to_swap_out)?;

        Ok(())
    }

    /// Swaps out GPU blocks to CPU blocks
    #[instrument]
    fn swap_out(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_out: &mut HashMap<u64, u64>,
    ) -> Result<(), SchedulerError> {
        info!(
            "Swapping out for sequence group with id = {}",
            sequence_group.request_id
        );

        if !self.block_manager.can_swap_out(sequence_group)? {
            error!("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.");
            return Err(SchedulerError::NotEnoughBlockSpaceForSwapOut);
        }

        let mapping = self.block_manager.swap_out(sequence_group)?;
        blocks_to_swap_out.extend(mapping.iter());
        sequence_group.sequences.iter_mut().for_each(|(_, s)| {
            let sequence_status = { s.borrow().get_sequence_status() };
            if sequence_status == SequenceStatus::Running {
                s.borrow_mut().set_sequence_status(SequenceStatus::Swapped)
            }
        });

        Ok(())
    }

    /// Swaps in CPU blocks to GPU blocks
    #[instrument]
    fn swap_in(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_in: &mut HashMap<u64, u64>,
    ) -> Result<(), SchedulerError> {
        let mapping = self.block_manager.swap_in(sequence_group)?;
        blocks_to_swap_in.extend(mapping.iter());
        sequence_group.sequences.iter_mut().for_each(|(_, s)| {
            let sequence_status = { s.borrow().get_sequence_status() };
            if sequence_status == SequenceStatus::Swapped {
                s.borrow_mut().set_sequence_status(SequenceStatus::Running)
            }
        });

        Ok(())
    }

    /// Computes if duration change has been greater than scheduled delay
    fn passed_delay(&mut self, now: Instant) -> bool {
        if self.previous_prompt {
            self.last_prompt_latency = (now - self.previous_time).as_secs_f32();
        }

        self.previous_time = now;
        self.previous_prompt = false;

        // Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor() > 0.0 && !self.waiting.is_empty() {
            // DON'T PANIC: at this point, we are guaranteed that `self.waiting` is non-empty
            let earliest_arrival_time =
                self.waiting.iter().map(|s| s.arrival_time()).min().unwrap();
            (now - earliest_arrival_time).as_secs_f32()
                > self.scheduler_config.delay_factor() * self.last_prompt_latency
                || self.running.is_empty()
        } else {
            true
        }
    }

    /// Get prompt limit
    fn get_prompt_limit(&self) -> usize {
        if self.scheduler_config.enable_chunked_prefill() {
            self.scheduler_config.max_model_len()
        } else {
            self.scheduler_config
                .max_model_len()
                .min(self.scheduler_config.max_num_batched_tokens())
        }
    }

    /// Allocates blocks to `SequenceGroup` and set sequences status to `Running`
    fn allocate_and_set_running(
        &mut self,
        sequence_group: &mut SequenceGroup,
    ) -> Result<(), SchedulerError> {
        self.block_manager.allocate(&sequence_group)?;
        sequence_group.sequences.iter_mut().for_each(|(_, s)| {
            let sequence_status = { s.borrow().get_sequence_status() };
            if sequence_status == SequenceStatus::Waiting {
                s.borrow_mut().set_sequence_status(SequenceStatus::Running)
            }
        });
        Ok(())
    }
}

/// A `SequenceGroup` that has been scheduled
#[derive(Debug)]
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
    #[error("Invalid number of sequences for recompute: `{0}`")]
    InvalidNumberSequencesForRecompute(usize),
    #[error("Not enough block space for swap out")]
    NotEnoughBlockSpaceForSwapOut,
    #[error("Invalid number of waiting sequences for request `{request_id}`: `{num_sequences}`")]
    InvalidNumberWaitingSequence {
        request_id: String,
        num_sequences: usize,
    },
    #[error("Invalid number of new tokens, got `{num_new_tokens}`, but it should be `{num_prompt_tokens}`")]
    InvalidNumberOfNewTokens {
        num_prompt_tokens: usize,
        num_new_tokens: usize,
    },
    #[error("Invalid scheduler output: `{0}`")]
    InvalidSchedulerOutput(String),
    #[error("Invalid number of budget tokens: `{0}`")]
    InvalidNumberBudgetTokens(String),
    #[error("Invalid number of sequences: `{0}`")]
    InvalidNumberBudgetSequences(String),
    #[error("Chunked prefills not allowed: `{0}`")]
    ChunkedPrefillsNotAllowed(String),
    #[error("Invalid prefill sequence: `{0}`")]
    InvalidPrefillSequences(String),
}

#[cfg(test)]
mod tests {
    use std::{collections::vec_deque, time};

    use super::*;
    use crate::{
        policy::FcfsPolicy,
        scheduler,
        sequence::{self, tests::create_dummy_prompt, LogProb},
    };

    fn get_sequence_groups(scheduler_outputs: &SchedulerOutputs) -> Vec<SequenceGroup> {
        scheduler_outputs
            .scheduled_sequence_groups
            .iter()
            .map(|s| s.scheduled_group.clone())
            .collect()
    }

    fn schedule_and_update_computed_tokens(
        scheduler: &mut Scheduler<FcfsPolicy>,
    ) -> (Vec<SequenceGroupMetadata>, SchedulerOutputs) {
        let (metadatas, mut outputs) = scheduler.schedule().expect("Failed to schedule");
        for (s, meta) in outputs
            .scheduled_sequence_groups
            .iter_mut()
            .zip(metadatas.iter())
        {
            s.scheduled_group
                .update_num_computed_tokens(meta.token_chunk_size)
                .expect("Failed to updated number of computed tokens");
        }
        (metadatas, outputs)
    }

    fn add_new_token(scheduler: &mut Scheduler<FcfsPolicy>, token_id: u32) {
        scheduler.running.iter_mut().for_each(|s| {
            for (_, sequence) in s.sequences.iter() {
                {
                    sequence.borrow_mut().add_token_id(
                        token_id,
                        HashMap::from_iter([(token_id, LogProb::new(0.5, None, None))]),
                    )
                }
            }
        });
    }

    fn add_new_token_to_output(out: &SchedulerOutputs, token_id: u32) {
        let sequence_groups = get_sequence_groups(out);
        for sequence_group in sequence_groups {
            for sequence in sequence_group.sequences.values() {
                sequence.borrow_mut().add_token_id(
                    token_id,
                    HashMap::from_iter([(token_id, LogProb::new(0.5, None, None))]),
                )
            }
        }
    }

    fn add_token_budget(
        budget: &mut SchedulingBudget,
        num_batched_tokens: usize,
        num_current_sequences: usize,
    ) {
        let (_, mock_seq_group) = create_dummy_prompt(10, 60, None, false, 1);
        budget.add_num_batched_tokens(mock_seq_group.request_id.clone(), num_batched_tokens);
        budget.add_number_sequences(mock_seq_group.request_id, num_current_sequences);
    }

    #[test]
    fn test_scheduler_add_sequence_group() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;
        const SWAP_SPACE: usize = 1;
        const CACHE_DTYPE: &str = "auto";

        const MAX_NUM_BATCHED_TOKENS: usize = 100;
        const MAX_NUM_SEQUENCES: usize = 64;
        const MAX_MODEL_LEN: usize = 1;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            CACHE_DTYPE.into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // adds multiple sequence groups to `Scheduler` instance
        let num_sequence_group: usize = 4;
        for i in 0..num_sequence_group {
            let (_, sequence_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, false, 1);
            scheduler.add_sequence_group(sequence_group);
            assert_eq!(scheduler.num_unfinished_sequeces(), i + 1);
        }
    }

    #[test]
    fn test_scheduler_abort_sequence_group() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;
        const SWAP_SPACE: usize = 1;
        const CACHE_DTYPE: &str = "auto";

        const MAX_NUM_BATCHED_TOKENS: usize = 100;
        const MAX_NUM_SEQUENCES: usize = 64;
        const MAX_MODEL_LEN: usize = 1;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            CACHE_DTYPE.into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // adds multiple sequence groups to `Scheduler` instance
        let num_sequence_group: usize = 4;
        let mut requests_ids = HashSet::new();
        for i in 0..num_sequence_group {
            let (_, sequence_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, false, 1);
            scheduler.add_sequence_group(sequence_group);
            requests_ids.insert(format!("{i}"));
        }

        assert_eq!(scheduler.num_unfinished_sequeces(), num_sequence_group);
        scheduler
            .abort_sequence_groups(requests_ids.into_iter())
            .expect("Failed to abort sequence groups");
        assert_eq!(scheduler.num_unfinished_sequeces(), 0);
    }

    #[test]
    fn test_scheduler_schedule_simple() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        const SWAP_SPACE: usize = 1;
        const CACHE_DTYPE: &str = "auto";

        const MAX_NUM_BATCHED_TOKENS: usize = 100;
        const MAX_NUM_SEQUENCES: usize = 4;
        const MAX_MODEL_LEN: usize = 16;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            CACHE_DTYPE.into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        let num_sequence_groups = 4;
        let mut running = vec![];

        for i in 0..num_sequence_groups {
            let (_, sequence_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, false, 1);
            scheduler.add_sequence_group(sequence_group.clone());
            running.push(sequence_group);
        }

        // Schedule sequence groups prompts.
        let num_tokens = BLOCK_SIZE * num_sequence_groups;
        let (sequence_groups_metadata, outputs) =
            schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&outputs);
        assert_eq!(
            sequence_groups
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>(),
            running
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>()
        );
        for sequence_group in sequence_groups.iter() {
            let running_sequence_group = running
                .iter()
                .find(|s| s.request_id == sequence_group.request_id)
                .unwrap()
                .clone();
            assert_eq!(
                running_sequence_group.sampling_params(),
                sequence_group.sampling_params()
            );
            assert_eq!(
                running_sequence_group
                    .sequences
                    .iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>(),
                sequence_group
                    .sequences
                    .iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
            );
            assert_eq!(running_sequence_group.state(), sequence_group.state())
        }
        assert_eq!(outputs.num_batched_tokens, num_tokens);
        assert!(
            outputs.blocks_to_copy.is_empty()
                && outputs.blocks_to_swap_in.is_empty()
                && outputs.blocks_to_swap_out.is_empty()
        );
        assert_eq!(sequence_groups_metadata.len(), num_sequence_groups);

        // add a new token for each running `SequenceGroup`'s internal `Sequence`
        add_new_token(&mut scheduler, 1);

        // Schedule seq groups generation.
        let (sequence_groups_metadata, outputs) =
            schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&outputs);
        assert_eq!(
            sequence_groups
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>(),
            running
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>()
        );
        for sequence_group in sequence_groups.iter() {
            let running_sequence_group = running
                .iter()
                .find(|s| s.request_id == sequence_group.request_id)
                .unwrap()
                .clone();
            assert_eq!(
                running_sequence_group.sampling_params(),
                sequence_group.sampling_params()
            );
            assert_eq!(
                running_sequence_group
                    .sequences
                    .iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>(),
                sequence_group
                    .sequences
                    .iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>()
            );
            assert_eq!(running_sequence_group.state(), sequence_group.state())
        }
        assert_eq!(outputs.num_batched_tokens, num_sequence_groups);
        assert!(
            outputs.blocks_to_copy.is_empty()
                && outputs.blocks_to_swap_in.is_empty()
                && outputs.blocks_to_swap_out.is_empty()
        );
        assert_eq!(sequence_groups_metadata.len(), num_sequence_groups);

        add_new_token(&mut scheduler, 1)
    }

    #[test]
    /// Verify running batched tokens are not applied to prefill requests.
    fn test_scheduler_prefill_prioritized() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 2;
        const NUM_GPU_BLOCKS: usize = 2;
        const SWAP_SPACE: usize = 1;
        const CACHE_DTYPE: &str = "auto";

        const MAX_NUM_BATCHED_TOKENS: usize = 30;
        const MAX_NUM_SEQUENCES: usize = 2;
        const MAX_MODEL_LEN: usize = 30;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            CACHE_DTYPE.into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // Add seq groups to scheduler
        let (_, sequence_group_a) = create_dummy_prompt(1, 1, None, false, 1);
        scheduler.add_sequence_group(sequence_group_a.clone());

        // Schedule seq groups prompts
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let out_sequence_groups = get_sequence_groups(&out);
        assert_eq!(out_sequence_groups.len(), 1);
        assert_eq!(
            out_sequence_groups[0].request_id,
            sequence_group_a.request_id
        );
        assert_eq!(out_sequence_groups[0].sequences.values().len(), 1);
        let sequence = out_sequence_groups[0]
            .sequences
            .values()
            .next()
            .unwrap()
            .borrow();
        let sequence_a = sequence_group_a.sequences.values().next().unwrap().borrow();

        assert_eq!(sequence.sequence_data(), sequence_a.sequence_data());
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_a.get_num_new_tokens()
        );
        assert_eq!(sequence.get_last_token_id(), sequence_a.get_last_token_id());
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_a.get_num_new_tokens()
        );
        assert_eq!(
            sequence.get_num_total_logical_token_blocks(),
            sequence_a.get_num_total_logical_token_blocks()
        );
        assert_eq!(sequence.get_token_ids(), sequence_a.get_token_ids());
        assert_eq!(
            sequence.get_sequence_status(),
            sequence_a.get_sequence_status()
        );

        // Add a new prefill request B
        let (_, sequence_group_b) = create_dummy_prompt(2, 30, None, false, 1);
        scheduler.add_sequence_group(sequence_group_b.clone());

        // Verify prefill requests are prioritized. Since max_batched_num_tokens
        // is 1, new prefill request has to be scheduled first.
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let out_sequence_groups = get_sequence_groups(&out);
        assert_eq!(out_sequence_groups.len(), 1);
        assert_eq!(
            out_sequence_groups[0].request_id,
            sequence_group_b.request_id
        );
        assert_eq!(out_sequence_groups[0].sequences.values().len(), 1);
        let sequence = out_sequence_groups[0]
            .sequences
            .values()
            .next()
            .unwrap()
            .borrow();
        let sequence_b = sequence_group_b.sequences.values().next().unwrap().borrow();

        assert_eq!(sequence.sequence_data(), sequence_b.sequence_data());
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_b.get_num_new_tokens()
        );
        assert_eq!(sequence.get_last_token_id(), sequence_b.get_last_token_id());
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_b.get_num_new_tokens()
        );
        assert_eq!(
            sequence.get_num_total_logical_token_blocks(),
            sequence_b.get_num_total_logical_token_blocks()
        );
        assert_eq!(sequence.get_token_ids(), sequence_b.get_token_ids());
        assert_eq!(
            sequence.get_sequence_status(),
            sequence_b.get_sequence_status()
        );
    }

    #[test]
    fn test_scheduler_schedule_preempt_abort() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 2;
        const NUM_GPU_BLOCKS: usize = 2;
        const SWAP_SPACE: usize = 1;
        const CACHE_DTYPE: &str = "auto";

        const MAX_NUM_BATCHED_TOKENS: usize = 64;
        const MAX_NUM_SEQUENCES: usize = 2;
        const MAX_MODEL_LEN: usize = 16;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            CACHE_DTYPE.into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // Add seq groups to scheduler
        let (_, sequence_group_a) = create_dummy_prompt(1, BLOCK_SIZE, None, false, 1);
        let (_, sequence_group_b) = create_dummy_prompt(2, BLOCK_SIZE, None, false, 1);

        scheduler.add_sequence_group(sequence_group_a.clone());
        scheduler.add_sequence_group(sequence_group_b.clone());

        // Schedule seq groups prompts
        let (sequence_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 2);
        assert_eq!(
            sequence_groups[0].request_id,
            sequence_group_a.request_id.clone()
        );
        assert_eq!(sequence_groups[0].sequences.values().len(), 1);

        {
            let sequence = sequence_groups[0]
                .sequences
                .values()
                .next()
                .unwrap()
                .borrow();
            let sequence_a = sequence_group_a.sequences.values().next().unwrap().borrow();

            assert_eq!(sequence.sequence_data(), sequence_a.sequence_data());
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_a.get_num_new_tokens()
            );
            assert_eq!(sequence.get_last_token_id(), sequence_a.get_last_token_id());
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_a.get_num_new_tokens()
            );
            assert_eq!(
                sequence.get_num_total_logical_token_blocks(),
                sequence_a.get_num_total_logical_token_blocks()
            );
            assert_eq!(sequence.get_token_ids(), sequence_a.get_token_ids());
            assert_eq!(
                sequence.get_sequence_status(),
                sequence_a.get_sequence_status()
            );

            assert_eq!(
                sequence_groups[1].request_id,
                sequence_group_b.request_id.clone()
            );
            assert_eq!(sequence_groups[1].sequences.values().len(), 1);
        }

        assert_eq!(
            sequence_groups[1].request_id,
            sequence_group_b.request_id.clone()
        );
        assert_eq!(sequence_groups[1].sequences.values().len(), 1);

        {
            let sequence = sequence_groups[1]
                .sequences
                .values()
                .next()
                .unwrap()
                .borrow();
            let sequence_b = sequence_group_b.sequences.values().next().unwrap().borrow();

            assert_eq!(sequence.sequence_data(), sequence_b.sequence_data());
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_b.get_num_new_tokens()
            );
            assert_eq!(sequence.get_last_token_id(), sequence_b.get_last_token_id());
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_b.get_num_new_tokens()
            );
            assert_eq!(
                sequence.get_num_total_logical_token_blocks(),
                sequence_b.get_num_total_logical_token_blocks()
            );
            assert_eq!(sequence.get_token_ids(), sequence_b.get_token_ids());
            assert_eq!(
                sequence.get_sequence_status(),
                sequence_b.get_sequence_status()
            );

            assert_eq!(out.num_batched_tokens, BLOCK_SIZE * 2);
            assert!(
                out.blocks_to_copy.is_empty()
                    && out.blocks_to_swap_in.is_empty()
                    && out.blocks_to_swap_out.is_empty()
            );
            assert_eq!(sequence_group_metadata.len(), 2);
            assert_eq!(scheduler.num_unfinished_sequeces(), 2);
        }

        // Append "generated" tokens, allowing the sequence to mark prompt tokens as
        // processed
        add_new_token(&mut scheduler, 1);

        // Schedule sequence groups generation and preempt sequence b
        let (sequence_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);

        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(out.preempted, 1);
        assert_eq!(sequence_group_metadata.len(), 1);
        assert_eq!(scheduler.num_unfinished_sequeces(), 2);
        assert_eq!(out.num_batched_tokens, 1);
        assert!(
            out.blocks_to_copy.is_empty()
                && out.blocks_to_swap_in.is_empty()
                && out.blocks_to_swap_out.is_empty()
        );
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.swapped.len(), 0);

        // Abort sequence a and reschedule sequence b with recomputation
        scheduler
            .abort_sequence_group(format!("1"))
            .expect("Failed to abort sequence group");
        let (sequence_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);

        let sequences_groups = get_sequence_groups(&out);

        assert_eq!(sequences_groups.len(), 1);
        assert_eq!(sequences_groups[0].request_id, sequence_group_b.request_id);
        assert_eq!(scheduler.waiting.len(), 0);
        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.swapped.len(), 0);

        assert_eq!(sequence_group_metadata.len(), 1);
    }

    #[test]
    fn test_scheduler_max_seqs() {
        const BLOCK_SIZE: usize = 4;
        const NUM_SEQ_GROUP: usize = 4;
        const MAX_SEQ_GROUP: usize = 2;
        const MAX_MODEL_LEN: usize = 16;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        let scheduler_config =
            SchedulerConfig::new(64, MAX_SEQ_GROUP, MAX_MODEL_LEN, 0.0, false, 0)
                .expect("Failed to get schedule config");
        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            1.0,
            1,
            "auto".into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate cache config");

        let mut scheduler =
            Scheduler::new(cache_config, scheduler_config).expect("Failed to generate scheduler");
        let mut all_sequence_groups = vec![];

        // Add sequence groups to the scheduler
        for i in 0..NUM_SEQ_GROUP {
            let (_, seq_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, false, 1);
            all_sequence_groups.push(seq_group);
        }

        // Append sequence group 1
        scheduler.add_sequence_group(all_sequence_groups[0].clone());

        // Schedule sequence group prompts
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(
            sequence_groups[0].request_id,
            all_sequence_groups[0].request_id
        );

        add_new_token_to_output(&out, 1);

        // Schedule sequence groups generation
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(
            sequence_groups[0].request_id,
            all_sequence_groups[0].request_id
        );

        // Append two more sequence groups
        scheduler.add_sequence_group(all_sequence_groups[1].clone());
        scheduler.add_sequence_group(all_sequence_groups[2].clone());

        // Schedule sequence group prompts
        // Only 1 sequence group should be scheduled since `max_seq_group` is 2
        // and one is prompting
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(
            sequence_groups[0].request_id,
            all_sequence_groups[1].request_id
        )
    }

    #[test]
    fn test_scheduler_delay_factor() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        let scheduler_config = SchedulerConfig::new(100, 64, 16, 0.5, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            1.0,
            1,
            "auto".into(),
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to get cache config");
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");

        // schedule first prompt
        let (_, sequence_group) = create_dummy_prompt(0, BLOCK_SIZE, None, false, 1);
        scheduler.add_sequence_group(sequence_group.clone());
        let (sequence_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert!(out.number_prefill_groups > 0);
        assert_eq!(sequence_group_meta[0].request_id(), "0".to_string());

        add_new_token_to_output(&out, 1);

        // wait for a second before scheduling next prompt
        std::thread::sleep(Duration::from_secs(1));

        let (_, sequence_group) = create_dummy_prompt(1, BLOCK_SIZE, None, false, 1);
        scheduler.add_sequence_group(sequence_group.clone());

        // second prompt should NOT be scheduled
        let (sequence_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert_eq!(out.number_prefill_groups, 0);
        assert_eq!(sequence_group_meta[0].request_id(), "0".to_string());

        add_new_token(&mut scheduler, 1);

        // wait for more than 0.5 seconds and try again
        std::thread::sleep(Duration::from_millis(600));
        let (sequence_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert!(out.number_prefill_groups > 0);
        assert_eq!(sequence_group_meta[0].request_id(), "1".to_string());

        add_new_token(&mut scheduler, 1);
    }

    #[test]
    fn test_swapped_out_prioritized() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, "auto".into(), None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");

        // best_of = 2 * 3 == 6 sequences
        for i in 0..3 {
            let (_, sequence_group) = create_dummy_prompt(i, 60, None, false, 2);
            scheduler.add_sequence_group(sequence_group);
        }

        let (seq_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);

        // prefill is scheduled now
        assert_eq!(out.scheduled_sequence_groups.len(), 3);
        add_new_token_to_output(&out, 1);

        // The last request should be swapped out
        let (seq_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert_eq!(out.scheduled_sequence_groups.len(), 2);
        assert_eq!(out.num_batched_tokens, 2);
        assert!(!out.blocks_to_swap_out.is_empty());
        assert!(out.blocks_to_swap_in.is_empty());

        // Add 1 more task. Swap should be prioritized over prefill
        let (_, sequence_group) = create_dummy_prompt(2, 60, None, false, 2);
        scheduler.add_sequence_group(sequence_group);
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        add_new_token_to_output(&out, 1);

        // assert_eq!(out.scheduled_sequence_groups.len(), 1);
        // assert_eq!(out.num_batched_tokens, 3);
        // assert!(out.blocks_to_swap_in.is_empty());
        // assert!(out.blocks_to_swap_out.is_empty())
    }

    #[test]
    /// Test prompt longer than max_prompt_len is aborted
    fn test_prefill_schedule_max_prompt_len() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 30, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, "auto".into(), None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");

        let (_, seq_group) = create_dummy_prompt(0, 60, None, false, 1);
        let waiting = VecDeque::from_iter([seq_group]);
        let mut budget = SchedulingBudget::new(10000, 10000);

        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting, &mut budget, false)
            .expect("Failed to schedule prefills");

        assert_eq!(output.ignored_sequence_groups.len(), 1);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 0);
    }

    #[test]
    /// Test token budget respected.
    fn test_prefill_schedule_token_budget() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config =
            CacheConfig::new(BLOCK_SIZE, 1.0, 1, "auto".to_string(), None, None, 8, 8)
                .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut waiting: VecDeque<SequenceGroup> = VecDeque::new();
        let mut budget = SchedulingBudget::new(0, 10_000);

        for i in 0..2 {
            let (_, sequence_group) = create_dummy_prompt(i, 60, None, false, 1);
            waiting.push_back(sequence_group);
        }

        // 0 token budget == nothing is scheduled
        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting.clone(), &mut budget, false)
            .expect("Failed to run schedule prefills");
        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 2);

        // 60 token budget == 1 request scheduled
        let mut budget = SchedulingBudget::new(60, 10_000);
        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting, &mut budget, false)
            .expect("Failed to run schedule prefills");
        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 1);
        assert_eq!(budget.num_batched_tokens, 60);
        assert_eq!(budget.num_curr_seqs, 1);
        assert_eq!(remaining_waiting.len(), 1);

        // Test when current_batched_tokens is respected
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");
        let mut waiting = VecDeque::new();
        let mut budget = SchedulingBudget::new(60, 10_000);
        add_token_budget(&mut budget, 30, 0);
        let (_, sequence_group) = create_dummy_prompt(1, 60, None, false, 1);
        // Cannot schedule a prompt that doesn't fit the budget.
        waiting.push_back(sequence_group);
        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting.clone(), &mut budget, false)
            .expect("Failed to schedule prefills");

        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 30);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 1);

        let mut budget = SchedulingBudget::new(90, 10_000);
        add_token_budget(&mut budget, 30, 0);

        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting, &mut budget, false)
            .expect("Failed to schedule prefills");
        assert_eq!(output.sequence_groups.len(), 1);
        assert_eq!(budget.num_batched_tokens, 90);
        assert_eq!(budget.num_curr_seqs, 1);
        assert_eq!(remaining_waiting.len(), 0);
    }
}
