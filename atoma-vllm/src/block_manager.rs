use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockWriteGuard},
};

use crate::{
    block::{
        BlockError, BlockTable, DerefRead, DerefWrite, PhysicalTokenBlock, SyncPhysicalTokenBlock,
    },
    block_allocator::{BlockAllocator, BlockAllocatorError},
    sequence::{Sequence, SequenceGroup, SequenceStatus},
};
use candle::Device;
use thiserror::Error;
use tracing::{error, info, info_span, instrument, warn, Span};

/// `AllocationStatus` - keeps a state of status of possible block allocation
#[derive(Debug)]
pub enum AllocationStatus {
    /// Ok: seq_group can be allocated now.
    Ok,
    /// Later: `seq_group` cannot be allocated.
    /// The capacity of allocator is larger than seq_group required.
    Later,
    /// Never: `seq_group` can never be allocated.
    /// The `seq_group` is too large to allocated in GPU.
    Never,
    /// Nothing: there is no `Sequence` in `seq_group` awaiting to be allocated
    Nothing,
}

/// `BlockSpaceManager` - Manages the mapping between logical and physical token blocks.
#[derive(Debug)]
pub struct BlockSpaceManager {
    /// Block size
    block_size: usize,
    /// Block tables, mapping: `seq_id` -> `BlockTable`
    block_tables: HashMap<u64, BlockTable>,
    /// Total umber of CPU blocks
    num_cpu_blocks: usize,
    /// Total number of GPU blocks
    num_gpu_blocks: usize,
    /// CPU allocator
    cpu_allocator: BlockAllocator,
    /// GPU allocator
    gpu_allocator: BlockAllocator,
    /// Watermark
    watermark: f32,
    /// Watermark blocks
    watermark_blocks: u32,
    /// Block sliding window
    block_sliding_window: Option<usize>,
    /// Tracing span
    pub span: Span,
}

impl BlockSpaceManager {
    /// Constructor
    pub fn new(
        block_size: usize,
        device: usize,
        num_cpu_blocks: usize,
        num_gpu_blocks: usize,
        watermark: f32,
        sliding_window: Option<usize>,
    ) -> Result<Self, BlockSpaceManagerError> {
        let block_sliding_window = sliding_window.map(|sw| sw.div_ceil(block_size));

        let watermark_blocks = (watermark * num_gpu_blocks as f32).round() as u32;

        let (cpu_allocator, gpu_allocator): (BlockAllocator, BlockAllocator) = (
            BlockAllocator::new(block_size, Device::Cpu, num_cpu_blocks),
            BlockAllocator::new(block_size, Device::new_cuda(device)?, num_gpu_blocks),
        );

        let span = info_span!("block-space-manager");

        Ok(Self {
            block_size,
            block_tables: HashMap::new(),
            cpu_allocator: cpu_allocator,
            gpu_allocator: gpu_allocator,
            num_cpu_blocks,
            num_gpu_blocks,
            block_sliding_window,
            span,
            watermark,
            watermark_blocks,
        })
    }

    /// Checks if it is possible to allocate enough blocks for current
    /// `seq_group`, with output an `AllocationStatus`
    #[instrument]
    pub fn can_allocate(&self, seq_group: SequenceGroup) -> AllocationStatus {
        let num_required_blocks =
            seq_group.get_num_total_logical_token_blocks(SequenceStatus::Waiting);
        if let Some(mut num_required_blocks) = num_required_blocks {
            let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();

            if let Some(block_sliding_window) = self.block_sliding_window {
                num_required_blocks = num_required_blocks.min(block_sliding_window);
            }

            if num_free_gpu_blocks >= num_required_blocks {
                AllocationStatus::Ok
            } else if self.num_gpu_blocks < num_required_blocks {
                AllocationStatus::Never
            } else {
                AllocationStatus::Later
            }
        } else {
            // No `Sequence` awaiting to be allocated
            info!("No `Sequence` awaiting to be allocated in `SequenceGroup`");
            return AllocationStatus::Nothing;
        }
    }

    /// Allocates a new `SequenceGroup`
    #[instrument]
    pub fn allocate(&mut self, seq_group: SequenceGroup) -> Result<(), BlockSpaceManagerError> {
        if let Some(sequence) = seq_group.get_first_sequence(Some(SequenceStatus::Waiting)) {
            let num_logical_blocks_to_allocate = sequence.get_num_total_logical_token_blocks();
            let mut block_table: Vec<SyncPhysicalTokenBlock> =
                Vec::with_capacity(num_logical_blocks_to_allocate);

            for logical_idx in 0..num_logical_blocks_to_allocate {
                let block = if self
                    .block_sliding_window
                    .map(|bsw| logical_idx >= bsw)
                    .unwrap_or(false)
                {
                    let block_sliding_window = self.block_sliding_window.unwrap(); // DON'T PANIC: already verified that `self.block_sliding_window` is not None
                    let block = block_table.get(logical_idx % block_sliding_window).unwrap();
                    // TODO: I don't think this code is necessary
                    {
                        let mut block_guard = match block.deref_write() {
                            Ok(v) => v,
                            Err(e) => {
                                error!(
                                    "Failed to acquire lock for sequence_group with id = {}",
                                    seq_group.request_id
                                );
                                return Err(BlockSpaceManagerError::BlockError(e));
                            }
                        };
                        block_guard.increment_ref_count_by(
                            seq_group.get_num_sequences(Some(SequenceStatus::Waiting)),
                        );
                    }
                    block.clone()
                } else {
                    let block = self.gpu_allocator.allocate()?;
                    {
                        let mut block_guard = match block.deref_write() {
                            Ok(v) => v,
                            Err(e) => {
                                error!(
                                    "Failed to acquire lock for sequence_group with id = {}",
                                    seq_group.request_id
                                );
                                return Err(BlockSpaceManagerError::BlockError(e));
                            }
                        };
                        block_guard.increment_ref_count_by(
                            seq_group.get_num_sequences(Some(SequenceStatus::Waiting)),
                        );
                    }
                    block
                };
                block_table.push(block);
            }

            // Assign the block table for each sequence.
            for seq_id in seq_group.get_sequences_ids(Some(SequenceStatus::Waiting)) {
                self.block_tables.insert(seq_id, block_table.clone());
            }
        }

        Ok(())
    }

    /// Checks if we can append new slots
    pub fn can_append_slots(&self, seq_group: SequenceGroup) -> bool {
        // HEURISTIC: if there is at least one free block
        // for each sequence, we can append
        let num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks();
        let num_seqs = seq_group.get_num_sequences(Some(SequenceStatus::Running));
        num_seqs <= num_free_gpu_blocks
    }

    /// Checks if the last block is already full
    #[instrument]
    fn is_last_block_full(&self, sequence: &Sequence) -> bool {
        let token_ids_len = sequence.length();
        token_ids_len > 0 && (token_ids_len % sequence.block_size() == 0)
    }

    /// Allocates a new physical slot for a new token
    #[instrument]
    pub fn append_slots(
        &mut self,
        sequence: Sequence,
    ) -> Result<Option<(usize, usize)>, BlockSpaceManagerError> {
        let num_total_logical_token_blocks = sequence.get_num_total_logical_token_blocks();
        if num_total_logical_token_blocks == 0 {
            error!("Total number of logical token blocks is zero, sequences should not be empty");
            return Err(BlockSpaceManagerError::EmptySequence);
        }
        if let Some(block_table) = self.block_tables.get_mut(&sequence.sequence_id()) {
            // If we need to allocate a new physical block
            if block_table.len() < num_total_logical_token_blocks {
                if block_table.len() != num_total_logical_token_blocks - 1 {
                    error!(
                        "Can only allocate one physical block at the time, requested = {} blocks",
                        num_total_logical_token_blocks - block_table.len()
                    );
                    return Err(BlockSpaceManagerError::AppendSlotError(
                        "Can only allocate one physical block at the time".into(),
                    ));
                }

                if self.block_sliding_window.is_some()
                    && block_table.len() >= self.block_sliding_window.unwrap()
                {
                    // Block table has more than `block_sliding_window` blocks, so we might as well
                    // reuse a block prior to beginning of `block_table.len() - block_sliding_window`
                    //
                    // DON'T PANIC: `self.block_sliding_window` is not `None` and
                    // `block_table.len() % self.block_sliding_window.unwrap() <= block_table.len()`, forcibly
                    block_table.push(
                        block_table
                            .get(block_table.len() % self.block_sliding_window.unwrap())
                            .unwrap()
                            .clone(),
                    );
                } else {
                    // In this case, the sequence already has a new logical block to be appended
                    // we need to allocate a new physical block
                    let new_block = self.gpu_allocator.allocate()?;
                    block_table.push(new_block);

                    return Ok(None);
                }

                // We need to append the new token to the last block
                let last_block = block_table.last_mut().unwrap(); // DON'T PANIC: at this point we are sure that `block_table` is non-empty
                {
                    let guard = last_block.deref_read()?;
                    if !guard.device().is_cuda() {
                        error!("Invalid device, it should be a `Cuda` device");
                        return Err(BlockSpaceManagerError::InvalidDevice);
                    }

                    if guard.ref_count() == 1 {
                        return Ok(None);
                    }
                }

                // At this point, the block is shared with other sequences, so we perform Copy on Write (CoW)
                // CoW: Allocate a new block and copy the tokens
                let new_block = self.gpu_allocator.allocate()?;
                self.gpu_allocator.free(last_block.clone())?;
                *last_block = new_block;
            }
        }
        Ok(None)
    }

    /// Fork a `Sequence`. It never allocates new physical blocks, therefore this method is safe from OOM
    /// NOTE: we are cloning shared references to `PhysicalBlocks` from the parent to child sequence
    #[instrument]
    pub fn fork(
        &mut self,
        parent_sequence: Sequence,
        child_sequence: Sequence,
    ) -> Result<(), BlockSpaceManagerError> {
        info!(
            "Forking current parent sequence with id = {}",
            parent_sequence.sequence_id()
        );
        if !self
            .block_tables
            .contains_key(&parent_sequence.sequence_id())
        {
            return Err(BlockSpaceManagerError::MissingSequence);
        }

        // // DON'T PANIC: already checked to not be `None`
        let source_block_table = self
            .block_tables
            .get(&parent_sequence.sequence_id())
            .unwrap()
            .clone();

        self.block_tables
            .insert(child_sequence.sequence_id(), source_block_table.clone());

        // When using a sliding window, blocks will be eventually reused.
        // In this case the block tables will contain repeated blocks.
        // When forking, we must make sure that each block's `ref_count`
        // is only incremented by one, so we deduplicate them
        let block_ids = vec![];
        for block in source_block_table.iter() {
            let mut guard = block.deref_write()?;
            if !block_ids.contains(&guard.block_hash()) {
                guard.increment_ref_count();
            }
        }
        Ok(())
    }

    /// Get allocated physical blocks to each `Sequence` of `SequenceGroup`
    fn get_physical_blocks(
        &self,
        seq_group: &SequenceGroup,
    ) -> Result<Vec<SyncPhysicalTokenBlock>, BlockSpaceManagerError> {
        // NOTE: we assume that physical blocks are only shared across `Sequence`'s of the
        // same `SequenceGroup`
        let mut output = Vec::new();
        let mut block_ids = Vec::new();
        for sequence in seq_group.get_unfinished_sequences() {
            if let Some(blocks) = self.block_tables.get(&sequence.sequence_id()) {
                for block in blocks {
                    {
                        let block_id = block.deref_read()?.block_hash();
                        if !block_ids.contains(&block_id) {
                            block_ids.push(block_id);
                            output.push(block.clone());
                        }
                    }
                }
            }
        }
        Ok(output)
    }

    /// Checks if can swap in logical with physical token blocks
    #[instrument]
    pub fn can_swap_in(
        &self,
        seq_group: SequenceGroup,
    ) -> Result<AllocationStatus, BlockSpaceManagerError> {
        info!(
            "Can swap in, for sequence group with id = {}",
            seq_group.request_id
        );
        let blocks = self.get_physical_blocks(&seq_group)?;
        let num_swapped_sequences = seq_group.get_num_sequences(Some(SequenceStatus::Swapped));
        let num_free_blocks = self.gpu_allocator.get_num_free_blocks();
        // NOTE: Conservatively we assume that every sequence will allocate
        // at least one block free block right after the swap-in
        // NOTE: it should match the logic in `can_append_slot`
        let num_required_blocks = blocks.len() + num_swapped_sequences;
        if self.gpu_allocator.get_num_total_blocks() < num_required_blocks {
            Ok(AllocationStatus::Never)
        } else if num_free_blocks >= num_required_blocks {
            Ok(AllocationStatus::Ok)
        } else {
            Ok(AllocationStatus::Later)
        }
    }

    /// Swaps in CPU with GPU blocks
    #[instrument]
    pub fn swap_in(
        &mut self,
        seq_group: SequenceGroup,
    ) -> Result<HashMap<u64, u64>, BlockSpaceManagerError> {
        info!(
            "Swapping in CPU to GPU blocks, for sequence group with id = {}",
            seq_group.request_id
        );
        // CPU (physical) block -> GPU (physical) block
        let mut mapping = HashMap::new();
        for sequence in seq_group.get_seqs(Some(SequenceStatus::Swapped)) {
            let mut new_block_table: BlockTable = Vec::new();
            if let Some(block_table) = self.block_tables.get(&sequence.sequence_id()) {
                for cpu_block in block_table {
                    let cpu_block_id = { cpu_block.deref_read()?.block_hash() };
                    let gpu_block = if let Entry::Vacant(e) = mapping.entry(cpu_block_id) {
                        // Create a new block
                        let gpu_block = self.gpu_allocator.allocate()?;
                        e.insert(gpu_block.clone());
                        gpu_block
                    } else {
                        // Reuse a block
                        // DON'T PANIC: already checked that `cpu_block_id` lies in `mapping.keys()`
                        let gpu_block = mapping.get(&cpu_block_id).unwrap();
                        // Increase the `ref_count` of `gpu_block`
                        {
                            gpu_block.deref_write()?.increment_ref_count();
                        }
                        gpu_block.clone()
                    };
                    new_block_table.push(gpu_block);
                    // Free the CPU block that was allocated into the GPU
                    self.cpu_allocator.free(cpu_block.clone())?;
                }
            }
            self.block_tables
                .insert(sequence.sequence_id(), new_block_table);
        }

        let mut block_number_mapping = HashMap::with_capacity(mapping.len());
        for (cpu_block_id, gpu_block) in mapping.iter() {
            let gpu_block_id = { gpu_block.deref_read()?.block_hash() };
            block_number_mapping.insert(*cpu_block_id, gpu_block_id);
        }
        Ok(block_number_mapping)
    }

    /// Can swap out from GPU to CPU blocks
    #[instrument]
    pub fn can_swap_out(&self, seq_group: SequenceGroup) -> Result<bool, BlockSpaceManagerError> {
        info!(
            "Can swap out, for sequence group with id = {}",
            seq_group.request_id
        );
        let blocks = self.get_physical_blocks(&seq_group)?;
        Ok(blocks.len() <= self.cpu_allocator.get_num_free_blocks())
    }

    /// Swaps out GPU to CPU blocks
    #[instrument]
    pub fn swap_out(&mut self, seq_group: SequenceGroup) -> Result<HashMap<u64, u64>, BlockSpaceManagerError> {
        info!("Swap out GPU to CPU blocks, for sequence group with id = {}", seq_group.request_id);
        // GPU (physical) block -> CPU (physical) block
        let mut mapping = HashMap::new();
        for sequence in seq_group.get_seqs(Some(SequenceStatus::Running)).iter() { 
            let mut new_block_table: BlockTable = Vec::new();
            if let Some(block_table) = self.block_tables.get(&sequence.sequence_id()) { 
                for gpu_block in block_table { 
                    let gpu_block_id = {gpu_block.deref_read()?.block_hash()};
                    let cpu_block = if let Entry::Vacant(e) = mapping.entry(gpu_block_id) {
                        // Create a new block
                        let cpu_block = self.cpu_allocator.allocate()?;
                        e.insert(cpu_block.clone());
                        cpu_block
                    } else { 
                        // Reuse a block
                        // DON'T PANIC: already checked that `cpu_block_id` lies in `mapping.keys()`
                        let cpu_block = mapping.get(&gpu_block_id).unwrap();
                        // Increase the `ref_count` of `gpu_block`
                        {
                            cpu_block.deref_write()?.increment_ref_count();
                        }
                        cpu_block.clone()
                    };
                    new_block_table.push(cpu_block);
                    // Free the CPU block that was allocated into the GPU
                    self.gpu_allocator.free(gpu_block.clone())?;
                }
                self.block_tables
                .insert(sequence.sequence_id(), new_block_table);
            }
        }

        let mut block_number_mapping = HashMap::with_capacity(mapping.len());
        for (gpu_block_id, cpu_block) in mapping.iter() {
            let cpu_block_id = { cpu_block.deref_read()?.block_hash() };
            block_number_mapping.insert(*gpu_block_id, cpu_block_id);
        }
        Ok(block_number_mapping)
    }
}

#[derive(Debug, Error)]
pub enum BlockSpaceManagerError {
    #[error("Sliding window is not allowed with prefix caching enabled")]
    SlidingWindowDisabledWithCaching,
    #[error("Candle error: `{0}`")]
    CandleError(#[from] candle::Error),
    #[error("Block allocator error: `{0}`")]
    BlockAllocatorError(#[from] BlockAllocatorError),
    #[error("Poison write error: `{0}`")]
    PoisonError(String),
    #[error("Method not supported: `{0}`")]
    MethodNotSupported(String),
    #[error("Invalid reference count: `{0}`, it should be 1")]
    InvalidRefCount(usize),
    #[error("Append slot error: `{0}`")]
    AppendSlotError(String),
    #[error("Empty `Sequence`")]
    EmptySequence,
    #[error("Invalid `Device`")]
    InvalidDevice,
    #[error("Block error: `{0}`")]
    BlockError(#[from] BlockError),
    #[error("Missing sequence from block table")]
    MissingSequence,
}
