use std::time::Instant;

use candle::Device;

/// `BlockTable` corresponds to a mapping between logical and physical KV blocks of each request. Each block table entry
/// records the corresponding physical blocks of a logical block and the number of filled positions.
pub type BlockTable = Vec<PhysicalTokenBlock>;

/// A block that stores a contiguous chunk of tokens from left to right. Logical blocks are used to represent the states of the corresponding
/// physical blocks in the KV cache (allocated on the GPU).
#[derive(Clone, Debug)]
pub struct LogicalTokenBlock {
    /// Block number
    block_number: usize,
    /// Block size
    block_size: usize,
    /// Token IDs, of maximum size `block_size`
    token_ids: Vec<u32>,
    /// Number of tokens already allocated
    num_tokens: usize,
}

impl LogicalTokenBlock {
    /// Constructor
    pub fn new(block_number: usize, block_size: usize) -> Self {
        Self {
            block_number,
            block_size,
            token_ids: Vec::with_capacity(block_size),
            num_tokens: 0,
        }
    }

    /// Getter for `block_number`
    pub fn block_number(&self) -> usize {
        self.block_number
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_number
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

    /// Appends a new token id, if there are enough empty slots in the current `LogicalTokenBlock`
    pub fn append_tokens(&mut self, token_id: u32) {
        if !self.is_full() {
            self.token_ids.push(token_id);
            self.num_tokens += 1;
        }
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
#[derive(Clone, Debug)]
pub struct PhysicalTokenBlock {
    /// Block id
    block_hash: u64,
    /// Block number
    block_number: usize,
    /// Block size (representing number of KV vectors for contiguous input tokens)
    block_size: usize,
    /// Block has been computed
    computed: bool,
    /// Device to which the physical block is allocated
    device: Device,
    /// Last instant in which the block has been accessed
    last_accessed: Option<Instant>,
    /// Number of hashed tokens
    num_hashed_tokens: usize,
    /// Reference counter, used for CoW operations involved in more involved decoding techniques (e.g. ParallelSampling)
    ref_count: usize,
}

impl PhysicalTokenBlock {
    /// Constructor
    pub fn new(block_hash: u64, block_number: usize, block_size: usize, device: Device) -> Self {
        Self {
            block_hash,
            block_number,
            block_size,
            computed: false,
            device,
            last_accessed: None,
            num_hashed_tokens: 0,
            ref_count: 0,
        }
    }

    /// Getter for `block_id`
    pub fn block_hash(&self) -> u64 {
        self.block_hash
    }

    /// Getter for `block_number`
    pub fn block_number(&self) -> usize {
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

    /// Getter for `device`
    pub fn device(&self) -> Device {
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

    /// Getter for `ref_count`
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Set `block_hash`
    pub fn update_block_hash(&mut self, block_hash: u64) {
        self.block_hash = block_hash
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

    /// Decreases the `ref_count` variable by -1, if possible
    pub fn decrease_ref_count(&mut self) {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
    }
}

impl PartialEq for PhysicalTokenBlock {
    fn eq(&self, other: &Self) -> bool {
        self.block_hash == other.block_hash
            && self.block_number == other.block_number
            && self.block_size == other.block_size
            && self.device.same_device(&other.device)
            && self.ref_count == other.ref_count
    }
}
