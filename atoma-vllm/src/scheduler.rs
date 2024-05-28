use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use crate::{
    block_manager::{BlockSpaceManager, BlockSpaceManagerError},
    config::{CacheConfig, SchedulerConfig},
    sequence::SequenceGroup,
};
use thiserror::Error;

/// `Scheduler` - Responsible for managing the schedule of incoming inference `SequenceGroup` requests
///
/// It handles processing multiple sequences, including tasks such as prefill (initial setup), decoding and swapping blocks from CPU <-> GPU.
/// It relies on `BlockSpaceManager` to efficiently allocate resources, schedule tasks, and handle preemption and swapping.
pub struct Scheduler {
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
}

impl Scheduler {
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
        })
    }
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Block space manager error: `{0}`")]
    BlockSpaceManagerError(#[from] BlockSpaceManagerError),
}
