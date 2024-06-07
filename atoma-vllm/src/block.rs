use std::{
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    time::Instant,
};

use thiserror::Error;
use tracing::{error, info_span, instrument, Span};

use crate::traits::{BlockReadLock, BlockWriteLock};

/// `BlockTable` corresponds to a mapping between logical and physical KV blocks of each request. Each block table entry
/// records the corresponding physical blocks of a logical block and the number of filled positions.
pub type BlockTable = Vec<SyncPhysicalTokenBlock>;

/// Block device (either CPU or GPU)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockDevice {
    Cpu,
    Gpu,
}

/// A block that stores a contiguous chunk of tokens from left to right. Logical blocks are used to represent the states of the corresponding
/// physical blocks in the KV cache (allocated on the GPU).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTokenBlock {
    /// Block number
    block_number: usize,
    /// Block size
    block_size: usize,
    /// Token IDs, of maximum size `block_size`
    token_ids: Vec<u32>,
    /// Number of tokens already allocated
    num_tokens: usize,
    /// Tracing span
    span: Span,
}

impl LogicalTokenBlock {
    /// Constructor
    pub fn new(block_number: usize, block_size: usize) -> Self {
        Self {
            block_number,
            block_size,
            token_ids: Vec::with_capacity(block_size),
            num_tokens: 0,
            span: info_span!("block"),
        }
    }

    /// Getter for `block_number`
    pub fn block_number(&self) -> usize {
        self.block_number
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Checks if `token_ids` is empty
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Check if `token_ids` is full
    pub fn is_full(&self) -> bool {
        self.num_tokens == self.block_size
    }

    /// Get the number of additional token ids that can be added to the current `LogicalTokenBlock`
    pub fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.num_tokens
    }

    /// Appends a new set of token ids, if there are enough empty slots in the current `LogicalTokenBlock`
    #[instrument]
    pub fn append_tokens(&mut self, token_ids: &[u32]) -> Result<(), BlockError> {
        if token_ids.len() <= self.get_num_empty_slots() {
            self.token_ids.extend(token_ids);
            self.num_tokens += token_ids.len();
            return Ok(());
        }
        error!("Not enough space for allocation");
        Err(BlockError::AllocationError(
            "Not enough space for allocation".into(),
        ))
    }

    /// Getter for `token_ids`
    pub fn get_token_ids(&self) -> Vec<u32> {
        self.token_ids.clone()
    }

    /// Getter for last element in `token_ids`
    pub fn get_last_token_id(&self) -> Option<u32> {
        self.token_ids.last().cloned()
    }
}

/// A physical block structure. It represents a contiguous memory KV cache block, usually allocated on the 'physical' GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalTokenBlock {
    /// Block number
    block_number: u64,
    /// Block size (representing number of KV vectors for contiguous input tokens)
    block_size: usize,
    /// Block has been computed
    computed: bool,
    /// Device to which the physical block is allocated
    device: BlockDevice,
    /// Last instant in which the block has been accessed
    last_accessed: Option<Instant>,
    /// Number of hashed tokens
    num_hashed_tokens: usize,
    /// Reference counter, used for CoW operations involved in more involved decoding techniques (e.g. ParallelSampling)
    ref_count: usize,
}

impl PhysicalTokenBlock {
    /// Constructor
    pub fn new(block_number: u64, block_size: usize, device: BlockDevice) -> Self {
        Self {
            block_number,
            block_size,
            computed: false,
            device,
            last_accessed: None,
            num_hashed_tokens: 0,
            ref_count: 0,
        }
    }

    /// Getter for `block_number`
    pub fn block_number(&self) -> u64 {
        self.block_number
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Getter for `computed`
    pub fn computed(&self) -> bool {
        self.computed
    }

    /// Set `computed`
    pub fn set_computed(&mut self, value: bool) {
        self.computed = value
    }

    /// Getter for `device`
    pub fn device(&self) -> BlockDevice {
        self.device.clone()
    }

    /// Getter for `num_hashed_tokens`
    pub fn num_hashed_tokens(&self) -> usize {
        self.num_hashed_tokens
    }

    /// Getter for `last_accessed`
    pub fn last_accessed(&self) -> Option<Instant> {
        self.last_accessed
    }

    /// Sets `last_accessed`
    pub fn set_last_accessed(&mut self, instant: Instant) {
        self.last_accessed = Some(instant)
    }

    /// Getter for `ref_count`
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Set `num_hashed_tokens`
    pub fn update_num_hashed_tokens(&mut self, num_hashed_tokens: usize) {
        self.num_hashed_tokens = num_hashed_tokens
    }

    /// Set `computed` to false
    pub fn not_computed(&mut self) {
        self.computed = false;
    }

    /// Increments the `ref_count` variable by +1
    pub fn increment_ref_count(&mut self) {
        self.ref_count += 1;
    }

    /// Sets the `ref_count` by `value`
    pub fn set_ref_count_by(&mut self, value: usize) {
        self.ref_count = value;
    }

    /// Decreases the `ref_count` variable by -1, if possible
    pub fn decrease_ref_count(&mut self) -> Result<(), BlockError> {
        if self.ref_count > 0 {
            self.ref_count -= 1;
            Ok(())
        } else {
            error!("Reference counter is already zero, trying to dereference once more which should not be possible..");
            Err(BlockError::ReferenceCountError)
        }
    }
}

/// Sync and Send shared access physical block
pub type SyncPhysicalTokenBlock = Arc<RwLock<PhysicalTokenBlock>>;

impl BlockReadLock for SyncPhysicalTokenBlock {
    fn read_lock(&self) -> Result<RwLockReadGuard<PhysicalTokenBlock>, BlockError> {
        self.read()
            .map_err(|e| BlockError::PoisonError(e.to_string()))
    }
}

impl BlockWriteLock for SyncPhysicalTokenBlock {
    fn write_lock(&self) -> Result<RwLockWriteGuard<PhysicalTokenBlock>, BlockError> {
        self.write()
            .map_err(|e| BlockError::PoisonError(e.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum BlockError {
    #[error("Poison error: `{0}`")]
    PoisonError(String),
    #[error("Allocation error: `{0}`Not enough space for allocation")]
    AllocationError(String),
    #[error("Reference counter error, it cannot be negative")]
    ReferenceCountError,
}
