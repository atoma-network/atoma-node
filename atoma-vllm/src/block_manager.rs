use std::{
    cell::Ref,
    collections::{hash_map::Entry, HashMap},
    time::Instant,
};

use crate::{
    block::{BlockDevice, BlockError, BlockTable, SyncPhysicalTokenBlock},
    block_allocator::{BlockAllocator, BlockAllocatorError},
    sequence::{Sequence, SequenceGroup, SequenceStatus},
    traits::{DerefRead, DerefWrite},
};

use candle::utils::{cuda_is_available, metal_is_available};

use thiserror::Error;
use tracing::{error, info, info_span, instrument, warn, Span};

/// `AllocationStatus` - keeps a state of status of possible block allocation
#[derive(Clone, Debug, PartialEq, Eq)]
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
    #[allow(dead_code)]
    num_cpu_blocks: usize,
    /// Total number of GPU blocks
    num_gpu_blocks: usize,
    /// CPU allocator
    cpu_allocator: BlockAllocator,
    /// GPU allocator
    pub(crate) gpu_allocator: BlockAllocator,
    /// Block sliding window
    block_sliding_window: Option<usize>,
    /// Tracing span
    pub span: Span,
}

impl BlockSpaceManager {
    /// Constructor
    pub fn new(
        block_size: usize,
        num_cpu_blocks: usize,
        num_gpu_blocks: usize,
        sliding_window: Option<usize>,
    ) -> Result<Self, BlockSpaceManagerError> {
        let block_sliding_window = sliding_window.map(|sw| sw.div_ceil(block_size));

        let span = info_span!("block-space-manager");

        let (cpu_allocator, gpu_allocator): (BlockAllocator, BlockAllocator) =
            if cuda_is_available() || metal_is_available() {
                (
                    BlockAllocator::new(block_size, BlockDevice::Cpu, num_cpu_blocks),
                    BlockAllocator::new(block_size, BlockDevice::Gpu, num_gpu_blocks),
                )
            } else {
                error!("Unrecognized GPU");
                // TODO: we maintain this for test purposes, but we should error
                (
                    BlockAllocator::new(block_size, BlockDevice::Cpu, num_cpu_blocks),
                    BlockAllocator::new(block_size, BlockDevice::Gpu, num_gpu_blocks),
                )
            };

        Ok(Self {
            block_size,
            block_tables: HashMap::new(),
            cpu_allocator,
            gpu_allocator,
            num_cpu_blocks,
            num_gpu_blocks,
            block_sliding_window,
            span,
        })
    }

    /// Checks if it is possible to allocate enough blocks for current
    /// `seq_group`, with output an `AllocationStatus`
    #[instrument]
    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> AllocationStatus {
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
    ///
    /// WARN: The way the implementation works will FAIL if we try to `allocate` for the `SequenceGroup`
    /// as we are creating a new empty `block_table`, every time.
    #[instrument]
    pub fn allocate(&mut self, seq_group: &SequenceGroup) -> Result<(), BlockSpaceManagerError> {
        if let Some(sequence) = seq_group.get_first_sequence(Some(SequenceStatus::Waiting)) {
            let num_logical_blocks_to_allocate =
                sequence.borrow().get_num_total_logical_token_blocks();
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
                        block_guard.set_ref_count_by(
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
                        block_guard.set_ref_count_by(
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
    pub fn can_append_slots(&self, seq_group: &SequenceGroup) -> bool {
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
        sequence: Ref<Sequence>,
    ) -> Result<Option<(u64, u64)>, BlockSpaceManagerError> {
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
            }

            // We need to append the new token to the last block
            let last_block = block_table.last_mut().unwrap(); // DON'T PANIC: at this point we are sure that `block_table` is non-empty
            {
                let guard = last_block.deref_read()?;
                // if !guard.device().is_cuda() {
                //     error!("Invalid device, it should be a `Cuda` device");
                //     return Err(BlockSpaceManagerError::InvalidDevice);
                // }

                if guard.ref_count() == 1 {
                    return Ok(None);
                }
            }

            // At this point, the block is shared with other sequences, so we perform Copy on Write (CoW)
            // CoW: Allocate a new block and copy the tokens
            let new_block = self.gpu_allocator.allocate()?;
            self.gpu_allocator.free(last_block.clone())?;
            let (last_block_number, new_block_number) = {
                (
                    last_block.deref_read()?.block_number(),
                    new_block.deref_read()?.block_number(),
                )
            };
            *last_block = new_block;
            return Ok(Some((last_block_number, new_block_number)));
        }

        Ok(None)
    }

    /// Fork a `Sequence`. It never allocates new physical blocks, therefore this method is safe from OOM
    /// NOTE: we are cloning shared references to `PhysicalBlocks` from the parent to child sequence
    #[instrument]
    pub fn fork(
        &mut self,
        parent_sequence: Ref<Sequence>,
        child_sequence: Ref<Sequence>,
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
        let mut block_ids = vec![];
        for block in source_block_table.iter() {
            let mut guard = block.deref_write()?;
            if !block_ids.contains(&guard.block_number()) {
                guard.increment_ref_count();
            }
            block_ids.push(guard.block_number());
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
            if let Some(blocks) = self.block_tables.get(&sequence.borrow().sequence_id()) {
                for block in blocks {
                    {
                        let block_id = block.deref_read()?.block_number();
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
        seq_group: &SequenceGroup,
    ) -> Result<AllocationStatus, BlockSpaceManagerError> {
        info!(
            "Can swap in, for sequence group with id = {}",
            seq_group.request_id
        );
        let blocks = self.get_physical_blocks(seq_group)?;
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
        seq_group: &mut SequenceGroup,
    ) -> Result<HashMap<u64, u64>, BlockSpaceManagerError> {
        info!(
            "Swapping in CPU to GPU blocks, for sequence group with id = {}",
            seq_group.request_id
        );
        // CPU (physical) block -> GPU (physical) block
        let mut mapping = HashMap::new();
        for sequence_id in seq_group
            .get_sequences_ids(Some(SequenceStatus::Swapped))
            .iter()
        {
            let mut new_block_table: BlockTable = Vec::new();
            if let Some(block_table) = self.block_tables.get(sequence_id) {
                for cpu_block in block_table {
                    let cpu_block_id = { cpu_block.deref_read()?.block_number() };
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
                self.block_tables.insert(*sequence_id, new_block_table);
            }
            // NOTE: we update the status of the `Sequence` right after the previous check, and not on the scheduler logic
            let sequence = seq_group.get_sequence_from_id(*sequence_id).unwrap(); // DON'T PANIC: we already checked that `SequenceGroup` contains `Sequence` with `sequence_id`
            {
                sequence
                    .borrow_mut()
                    .set_sequence_status(SequenceStatus::Running);
            }
        }

        let mut block_number_mapping = HashMap::with_capacity(mapping.len());
        for (cpu_block_id, gpu_block) in mapping.iter() {
            let gpu_block_id = { gpu_block.deref_read()?.block_number() };
            block_number_mapping.insert(*cpu_block_id, gpu_block_id);
        }
        Ok(block_number_mapping)
    }

    /// Can swap out from GPU to CPU blocks
    #[instrument]
    pub fn can_swap_out(&self, seq_group: &SequenceGroup) -> Result<bool, BlockSpaceManagerError> {
        info!(
            "Can swap out, for sequence group with id = {}",
            seq_group.request_id
        );
        let blocks = self.get_physical_blocks(seq_group)?;
        Ok(blocks.len() <= self.cpu_allocator.get_num_free_blocks())
    }

    /// Swaps out GPU to CPU blocks
    #[instrument]
    pub fn swap_out(
        &mut self,
        seq_group: &mut SequenceGroup,
    ) -> Result<HashMap<u64, u64>, BlockSpaceManagerError> {
        info!(
            "Swap out GPU to CPU blocks, for sequence group with id = {}",
            seq_group.request_id
        );
        // GPU (physical) block -> CPU (physical) block
        let mut mapping = HashMap::new();
        for sequence_id in seq_group
            .get_sequences_ids(Some(SequenceStatus::Running))
            .iter()
        {
            let mut new_block_table: BlockTable = Vec::new();
            if let Some(block_table) = self.block_tables.get(sequence_id) {
                for gpu_block in block_table {
                    let gpu_block_id = { gpu_block.deref_read()?.block_number() };
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
                self.block_tables.insert(*sequence_id, new_block_table);
            }
            // NOTE: we update the status of the `Sequence` right after the previous check, and not on the scheduler logic
            let sequence = seq_group.get_sequence_from_id(*sequence_id).unwrap(); // DON'T PANIC: we already checked that `SequenceGroup` contains `Sequence` with `sequence_id`
            {
                sequence
                    .borrow_mut()
                    .set_sequence_status(SequenceStatus::Swapped);
            }
        }

        let mut block_number_mapping = HashMap::with_capacity(mapping.len());
        for (gpu_block_id, cpu_block) in mapping.iter() {
            let cpu_block_id = { cpu_block.deref_read()?.block_number() };
            block_number_mapping.insert(*gpu_block_id, cpu_block_id);
        }
        Ok(block_number_mapping)
    }

    /// Free block table
    fn free_block_table(&mut self, block_table: &BlockTable) -> Result<(), BlockSpaceManagerError> {
        // when using a sliding window, each seq will only use up
        // to `self.block_sliding_window` blocks. When freeing
        // the block table, we must make sure to not free blocks more
        // than once. If no sliding window is used, there is no block
        // reuse in the block table, so we must free all blocks.
        let blocks_to_free = if let Some(block_sliding_window) = self.block_sliding_window {
            block_table[block_sliding_window..].to_vec()
        } else {
            block_table.clone()
        };

        let mut block_ids = Vec::new();

        for block in blocks_to_free {
            let block_device = {
                let block_guard = block.deref_read()?;
                let block_id = block_guard.block_number();
                if block_ids.contains(&block_id) {
                    continue;
                } else {
                    block_ids.push(block_id)
                }
                block_guard.device()
            };
            if block_device == BlockDevice::Cpu {
                self.cpu_allocator.free(block)?;
            } else {
                self.gpu_allocator.free(block)?;
            }
        }

        Ok(())
    }

    /// Frees blocks for `Sequence`
    #[instrument]
    pub fn free(&mut self, sequence_id: u64) -> Result<(), BlockSpaceManagerError> {
        info!("Freeing blocks for sequence with id = {}", sequence_id);

        if !self.block_tables.contains_key(&sequence_id) {
            // NOTE: Either `Sequence`'s blocks have been freed already, or haven't been scheduled yet
            info!(
                "Sequence's blocks already freed or haven't been scheduled yet, sequence's id = {}",
                sequence_id
            );
            // Idempotent, we don't error
            return Ok(());
        }

        // DON'T PANIC: already checked that `sequence_id` is present in `self.block_tables`
        let block_table = self.block_tables.get(&sequence_id).unwrap().clone();
        self.free_block_table(&block_table)?;

        self.block_tables.remove(&sequence_id);

        Ok(())
    }

    /// Reset's all block tables
    #[instrument]
    pub fn reset(&mut self) -> Result<(), BlockSpaceManagerError> {
        info!("Resetting all block tables..");
        let block_tables = self.block_tables.clone();
        for (_, bt) in block_tables.iter() {
            self.free_block_table(bt)?;
        }
        self.block_tables.clear();
        Ok(())
    }

    /// Gets `Sequence`'s `BlockTable`
    pub fn get_block_table(&self, sequence: Ref<Sequence>) -> Option<BlockTable> {
        self.block_tables.get(&sequence.sequence_id()).cloned()
    }

    /// Gets `BlockId` for each `Sequence` in `BlockTable`
    pub fn get_block_table_ids(&self, sequence_id: &u64) -> Option<Vec<u64>> {
        self.block_tables.get(sequence_id).and_then(|bt| {
            bt.iter()
                .map(|b| b.deref_read().map(|ok| ok.block_number()))
                .collect::<Result<Vec<_>, _>>()
                .ok()
        })
    }

    /// Gets number of free gpu blocks
    pub fn get_number_of_free_gpu_blocks(&self) -> usize {
        self.gpu_allocator.get_num_free_blocks()
    }

    /// Gets number of free cpu blocks
    pub fn get_number_of_free_cpu_blocks(&self) -> usize {
        self.cpu_allocator.get_num_free_blocks()
    }

    /// Accesses all blocks in a given `Sequence`
    pub fn access_all_blocks_in_sequence(
        &self,
        sequence_id: &u64,
        access_time: Instant,
    ) -> Result<(), BlockSpaceManagerError> {
        if let Some(block_table) = self.block_tables.get(sequence_id) {
            for block in block_table {
                {
                    block.deref_write()?.set_last_accessed(access_time)
                }
            }
        }

        Ok(())
    }

    /// Computes full blocks in `Sequence`
    #[instrument]
    pub fn compute_full_blocks_in_sequence(
        &self,
        sequence: Sequence,
    ) -> Result<(), BlockSpaceManagerError> {
        info!(
            "Computes full blocks in sequence, for sequence_id = {}",
            sequence.sequence_id()
        );
        if let Some(block_table) = self.block_tables.get(&sequence.sequence_id()) {
            let max_full_block_plus_one = sequence.length() / self.block_size;
            if max_full_block_plus_one == 0 {
                return Ok(());
            }
            let max_full_block = max_full_block_plus_one - 1; // DON'T PANIC: already checked that `max_full_block_plus_one >= 1`
            for i in (0..max_full_block).rev() {
                if let Some(block) = block_table.get(i) {
                    {
                        let mut block_guard = block.deref_write()?;
                        if block_guard.computed() {
                            break;
                        } else {
                            block_guard.set_computed(true)
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Gets all computed blocks
    #[instrument]
    pub fn gets_all_computed_blocks(
        &self,
        sequence: Sequence,
    ) -> Result<Vec<u64>, BlockSpaceManagerError> {
        info!(
            "Getting all computed blocks for sequence with id = {}",
            sequence.sequence_id()
        );
        if let Some(block_table) = self.block_tables.get(&sequence.sequence_id()) {
            // NOTE We exclude the last block to avoid the case where the entire
            // prompt is cached. This would cause erroneous behavior in model
            // runner
            let mut output = Vec::new();
            for block in block_table[..block_table.len() - 1].iter() {
                {
                    let block_guard = block.deref_read()?;
                    if block_guard.computed() {
                        output.push(block_guard.block_number());
                    }
                }
            }
            return Ok(output);
        }
        Ok(vec![])
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
    #[error("Unrecognized GPU")]
    UnrecognizedGpu,
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        sampling_params::SamplingParams,
        sequence::{tests::create_dummy_prompt, LogProb, SequenceGroupState},
    };

    use super::*;

    #[test]
    fn test_allocate() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;
        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        // Allocate same `SequenceGroup` to all available GPU blocks
        for i in 0..NUM_GPU_BLOCKS {
            let (_, seq_group) =
                create_dummy_prompt(i as u64, BLOCK_SIZE, Some(BLOCK_SIZE), false, 1);
            assert_eq!(block_manager.can_allocate(&seq_group), AllocationStatus::Ok);
            block_manager
                .allocate(&seq_group)
                .expect("Failed to allocate");
        }

        // We can't allocate further blocks, as all available blocks have been already allocated
        let (_, seq_group) = create_dummy_prompt(
            NUM_GPU_BLOCKS as u64,
            BLOCK_SIZE,
            Some(BLOCK_SIZE),
            false,
            1,
        );
        assert_eq!(
            block_manager.can_allocate(&seq_group),
            AllocationStatus::Later
        );
    }

    #[test]
    fn test_append_slot_single_seq() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;

        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        // Allocate single seq to gpu block.
        let (prompt, seq_group) = create_dummy_prompt(
            NUM_GPU_BLOCKS as u64,
            BLOCK_SIZE,
            Some(BLOCK_SIZE),
            false,
            1,
        );

        block_manager
            .allocate(&seq_group)
            .expect("Failed to allocate block to `SequenceGroup`");

        // Nothing to append. `Sequence` has no new logical blocks
        assert!(block_manager.can_append_slots(&seq_group));
        let before_num_free_blocks = block_manager.get_number_of_free_gpu_blocks();
        assert!(block_manager
            .append_slots(prompt.try_borrow().unwrap())
            .expect("Failed to append slot")
            .is_none());
        let after_num_free_blocks = block_manager.get_number_of_free_gpu_blocks();
        assert_eq!(before_num_free_blocks, after_num_free_blocks);

        // Add `block_size` number of new tokens and append slot
        for i in 0..BLOCK_SIZE {
            let token_id = i + BLOCK_SIZE + 1;
            let sequence_id = { prompt.try_borrow().unwrap().sequence_id() };
            seq_group.add_token_id_to_seq(
                sequence_id,
                token_id as u32,
                HashMap::from_iter([(token_id as u32, LogProb::new(0.0, None, None))]),
            );
        }

        // We need to access the `Sequence` after being mutated above by adding the token_ids,
        // as `prompt` only contains tokens [0, 1, 2, 3] and not the remaining
        let sequence = seq_group
            .get_sequence_from_id(prompt.try_borrow().unwrap().sequence_id())
            .unwrap();

        assert!(block_manager.can_append_slots(&seq_group));
        let before_num_free_blocks = block_manager.get_number_of_free_gpu_blocks();
        assert!(block_manager
            .append_slots(sequence.try_borrow().unwrap())
            .expect("Failed to append slot")
            .is_none());
        let after_num_free_blocks = block_manager.get_number_of_free_gpu_blocks();
        assert_eq!(before_num_free_blocks, after_num_free_blocks + 1)
    }

    #[test]
    fn test_append_slot_with_cow() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;

        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        // Allocates `prompt` to GPU block. There will be one single slot left in the block
        let prompt = Sequence::new(0, None, "one two three".into(), vec![1, 2, 3], BLOCK_SIZE);

        // Fork the `Sequence` (increase `ref_count` by one) so that CoW will be required when we append a new `token_id`
        let child = prompt.fork(2);

        // Allocate space for `SequenceGroup`
        let seq_group = SequenceGroup::new(
            0.to_string(),
            vec![prompt.clone(), child.clone()],
            Instant::now(),
            None,
            SamplingParams {
                ..Default::default()
            },
            None,
            SequenceGroupState {
                generator: Some(42),
            },
        )
        .expect("Failed to construct a new `SequenceGroup`");

        block_manager
            .allocate(&seq_group)
            .expect("Failed to allocate sequence group");

        // Fork and append a new token id, we expect CoW to be scheduled
        let token_id = 4;
        seq_group.add_token_id_to_seq(
            2, // child sequence id
            token_id,
            HashMap::from_iter([(token_id, LogProb::new(0.0, None, None))]),
        );

        // We need to access the `Sequence` after being mutated above by adding the token_ids,
        // as `child` only contains tokens `[1, 2, 3]` and not the `4`
        let parent_sequence = seq_group
            .get_sequence_from_id(prompt.sequence_id())
            .unwrap();
        let child_sequence = seq_group.get_sequence_from_id(child.sequence_id()).unwrap();
        block_manager
            .fork(
                parent_sequence.try_borrow().unwrap(),
                child_sequence.try_borrow().unwrap(),
            )
            .expect("Block manager failed to fork `Sequence`s");

        assert!(block_manager.can_append_slots(&seq_group));
        let before_num_free_blocks = block_manager.get_number_of_free_gpu_blocks();
        let cows = block_manager
            .append_slots(child_sequence.try_borrow().unwrap())
            .expect("Failed to append slots to `child_sequence`");
        assert_eq!(cows, Some((3, 2)));

        let after_num_free_blocks = block_manager.get_number_of_free_gpu_blocks();
        assert_eq!(before_num_free_blocks, after_num_free_blocks + 1);
    }

    #[test]
    fn test_fork() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;

        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        let (prompt, seq_group) =
            create_dummy_prompt(1, BLOCK_SIZE - 1, Some(BLOCK_SIZE), false, 1);

        block_manager
            .allocate(&seq_group)
            .expect("Failed to allocated `SequenceGroup`");

        // Fork prompt and copy block tables
        let child = { Rc::new(RefCell::new(prompt.try_borrow().unwrap().fork(2))) };
        // we can use both `prompt` and `child`, as we haven't mutated `SeqGroup` internally
        block_manager
            .fork(prompt.try_borrow().unwrap(), child.try_borrow().unwrap())
            .expect("Failed to fork prompt `Sequence`");
        let prompt_block_table = block_manager
            .get_block_table(prompt.try_borrow().unwrap())
            .expect("Failed to get block table for `prompt`");
        let child_block_table = block_manager
            .get_block_table(child.try_borrow().unwrap())
            .expect("Failed to get block table for `child`");
        assert_eq!(prompt_block_table.len(), 1);
        assert!(prompt_block_table
            .iter()
            .zip(child_block_table)
            .all(|(pb, cb)| {
                pb.deref_read().unwrap().block_number() == cb.deref_read().unwrap().block_number()
                    && pb.deref_read().unwrap().block_size()
                        == cb.deref_read().unwrap().block_size()
                    && pb.deref_read().unwrap().computed() == cb.deref_read().unwrap().computed()
                    && pb.deref_read().unwrap().ref_count() == cb.deref_read().unwrap().ref_count()
                    && pb.deref_read().unwrap().last_accessed()
                        == cb.deref_read().unwrap().last_accessed()
                    && pb.deref_read().unwrap().num_hashed_tokens()
                        == cb.deref_read().unwrap().num_hashed_tokens()
            }));

        let token_id = 4;
        // Append token to `child` `Sequence`. Block is shared so Copy on Write occurs
        {
            child.borrow_mut().add_token_id(
                token_id,
                HashMap::from_iter([(token_id, LogProb::new(0.0, None, None))]),
            );
        }

        block_manager
            .append_slots(child.try_borrow().unwrap())
            .expect("Failed to append slots to `child` sequence");

        let new_prompt_block_table = block_manager
            .get_block_table(prompt.try_borrow().unwrap())
            .expect("Failed to get block table for `prompt`");
        let new_child_block_table = block_manager
            .get_block_table(child.try_borrow().unwrap())
            .expect("Failed to get block table for `child`");

        assert!(new_prompt_block_table
            .iter()
            .zip(new_child_block_table)
            .all(|(pb, cb)| {
                pb.deref_read().unwrap().block_number() != cb.deref_read().unwrap().block_number()
            }));
    }

    #[test]
    fn test_swap() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;

        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        let (prompt, seq_group) =
            create_dummy_prompt(1, BLOCK_SIZE - 1, Some(BLOCK_SIZE), false, 1);
        block_manager
            .allocate(&seq_group)
            .expect("Failed to allocate sequence group");

        // Emulate a forward pass by appending a single token.
        // The block manager then knows how many unprocessed
        // tokens will be written in the next forward pass
        let token_id = 0;
        let prompt = seq_group
            .get_sequence_from_id(prompt.try_borrow().unwrap().sequence_id())
            .unwrap();
        {
            prompt
                .borrow_mut()
                .set_sequence_status(SequenceStatus::Running);
        }
        {
            prompt.borrow_mut().add_token_id(
                token_id,
                HashMap::from_iter([(token_id, LogProb::new(0.0, None, None))]),
            );
        }

        // make sure we don't incur double mutable access to seq_group
        let prompt = prompt.clone();
        let mut seq_group = seq_group.clone();

        // Swap `seq_group` from GPU -> CPU
        let gpu_blocks_ids = block_manager
            .get_block_table_ids(&prompt.try_borrow().unwrap().sequence_id())
            .expect("Failed to get block ids from block table for `prompt`");
        assert!(block_manager
            .can_swap_out(&seq_group)
            .expect("Failed to run `can_swap_out`"));

        let before_cpu_blocks = block_manager.get_number_of_free_cpu_blocks();
        let before_gpu_blocks = block_manager.get_number_of_free_gpu_blocks();
        let mapping = block_manager
            .swap_out(&mut seq_group)
            .expect("Failed to `swap_out`");

        assert!(mapping
            .keys()
            .zip(gpu_blocks_ids.clone())
            .all(|(m, b)| { *m == b }));

        let after_cpu_blocks = block_manager.get_number_of_free_cpu_blocks();
        let after_gpu_blocks = block_manager.get_number_of_free_gpu_blocks();

        assert_eq!(before_cpu_blocks, after_cpu_blocks + gpu_blocks_ids.len());
        assert_eq!(before_gpu_blocks + gpu_blocks_ids.len(), after_gpu_blocks);

        let prompt = seq_group
            .get_sequence_from_id(prompt.try_borrow().unwrap().sequence_id())
            .unwrap();
        assert_eq!(
            prompt.try_borrow().unwrap().get_sequence_status(),
            SequenceStatus::Swapped
        );

        // Now swap sequence group from CPU -> GPU
        let cpu_blocks_ids = block_manager
            .get_block_table_ids(&prompt.try_borrow().unwrap().sequence_id())
            .expect("Failed to get block ids from block table for `prompt`");
        assert_eq!(
            block_manager
                .can_swap_in(&seq_group)
                .expect("failed to run `swap_in`"),
            AllocationStatus::Ok
        );

        let before_cpu_blocks = block_manager.get_number_of_free_cpu_blocks();
        let before_gpu_blocks = block_manager.get_number_of_free_gpu_blocks();
        let mapping = block_manager
            .swap_in(&mut seq_group)
            .expect("Failed to `swap_out`");

        assert!(mapping
            .keys()
            .zip(cpu_blocks_ids.clone())
            .all(|(m, b)| { *m == b }));

        let after_cpu_blocks = block_manager.get_number_of_free_cpu_blocks();
        let after_gpu_blocks = block_manager.get_number_of_free_gpu_blocks();

        assert_eq!(before_cpu_blocks + cpu_blocks_ids.len(), after_cpu_blocks);
        assert_eq!(before_gpu_blocks, after_gpu_blocks + cpu_blocks_ids.len());
    }

    #[test]
    fn test_free() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;

        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        let (prompt, seq_group) =
            create_dummy_prompt(1, BLOCK_SIZE - 1, Some(BLOCK_SIZE), false, 1);
        block_manager
            .allocate(&seq_group)
            .expect("Failed to allocate sequence group");

        // Free allocated sequence
        let _prompt_blocks = block_manager
            .get_block_table_ids(&prompt.try_borrow().unwrap().sequence_id())
            .expect("Failed to get block table ides")
            .len();
        let _before_blocks = block_manager.get_number_of_free_gpu_blocks();
        block_manager
            .free(prompt.try_borrow().unwrap().sequence_id())
            .expect("Failed to free blocks for `prompt`");
        let _after_blocks = block_manager.get_number_of_free_gpu_blocks();

        // Assert that block table for freed sequence is deleted
        assert!(block_manager
            .get_block_table_ids(&prompt.try_borrow().unwrap().sequence_id())
            .is_none())
    }

    #[test]
    fn test_reset() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;

        let mut block_manager =
            BlockSpaceManager::new(BLOCK_SIZE, NUM_CPU_BLOCKS, NUM_GPU_BLOCKS, None)
                .expect("Failed to create a `BlockSpaceManager`");

        // Allocate same seq group on all available gpu blocks
        let original_blocks = block_manager.get_number_of_free_gpu_blocks();
        for i in 0..NUM_GPU_BLOCKS {
            let (_, seq_group) =
                create_dummy_prompt(i as u64, BLOCK_SIZE, Some(BLOCK_SIZE), false, 1);
            block_manager
                .allocate(&seq_group)
                .unwrap_or_else(|_| panic!("Failed to allocate sequence group, index = {i}"));
        }

        assert_eq!(block_manager.get_number_of_free_gpu_blocks(), 0);
        // Resetting block manager frees all allocated blocks
        block_manager.reset().expect("Failed to reset");
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            original_blocks
        );
    }

    #[test]
    /// Tests that memory allocation and deallocation is handled
    /// correctly with multiple sequences that exceed the sliding
    /// window's capacity.
    fn test_sliding_window_multi_seq() {
        const BLOCK_SIZE: usize = 1;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        const SLIDING_WINDOW: usize = 2;

        let mut block_manager = BlockSpaceManager::new(
            BLOCK_SIZE,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
            Some(SLIDING_WINDOW),
        )
        .expect("Failed to create a `BlockSpaceManager`");
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS
        );

        let parent = Sequence::new(
            1,
            None,
            "one two three".to_string(),
            vec![1, 2, 3],
            BLOCK_SIZE,
        );
        let seq_group = SequenceGroup::new(
            "1".into(),
            vec![parent.clone()],
            Instant::now(),
            None,
            SamplingParams {
                ..Default::default()
            },
            None,
            SequenceGroupState {
                generator: Some(42),
            },
        )
        .expect("Failed to get `SequenceGroup`");
        let parent = seq_group.sequences.values().next().unwrap().clone();
        block_manager
            .allocate(&seq_group)
            .expect("Failed to allocated to sequence group");

        // assert the number of blocks allocated is correct
        // the parent seq has len 3, but since sliding_window is 2,
        // we will use at most 2 blocks
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS - SLIDING_WINDOW
        );

        // Fork prompt and copy block tables.
        let child = { Rc::new(RefCell::new(parent.borrow_mut().fork(2))) };
        block_manager
            .fork(parent.try_borrow().unwrap(), child.try_borrow().unwrap())
            .expect("Failed to fork");

        // assert the number of blocks allocated is correct
        // forking does not increase memory consumption
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS - SLIDING_WINDOW
        );

        // assert both parent and child share all blocks
        assert_eq!(
            block_manager.get_block_table_ids(&parent.try_borrow().unwrap().sequence_id()),
            block_manager.get_block_table_ids(&child.try_borrow().unwrap().sequence_id())
        );

        let token_id = 4;
        // Append token to child. Block is shared so copy on write occurs.
        {
            child.borrow_mut().add_token_id(
                token_id,
                HashMap::from_iter([(token_id, LogProb::new(0.0, None, None))]),
            );
        }
        block_manager
            .append_slots(child.try_borrow().unwrap())
            .expect("Failed to append slots");

        // assert the number of blocks allocated is correct
        // we will use now one block more. Each seq will use 2 blocks,
        // but only one can be shared
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS - SLIDING_WINDOW - 1
        );

        let token_id = 5;
        {
            parent.borrow_mut().add_token_id(
                token_id,
                HashMap::from_iter([(token_id, LogProb::new(0.0, None, None))]),
            );
        }
        block_manager
            .append_slots(parent.try_borrow().unwrap())
            .expect("Failed to append slots");

        // assert the number of blocks allocated is correct
        // no change, because both sequences are still just sharing one block
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS - SLIDING_WINDOW - 1
        );

        let block_table_parent = block_manager
            .get_block_table_ids(&parent.try_borrow().unwrap().sequence_id())
            .expect("Failed to get parent block table");
        let block_table_child = block_manager
            .get_block_table_ids(&child.try_borrow().unwrap().sequence_id())
            .expect("Failed to get child block_table");

        assert!(block_table_parent
            .iter()
            .zip(block_table_child.iter())
            .any(|(p, c)| p != c));

        // assert both blocks are sharing the second-last block
        assert_eq!(
            block_table_parent[block_table_parent.len() - 2],
            block_table_child[block_table_child.len() - 2]
        );

        // now let's clean up...
        block_manager
            .free(parent.try_borrow().unwrap().sequence_id())
            .expect("Failed to free block manager");

        // assert the number of blocks allocated is correct
        // We have freed one seq, reducing the ref count of two blocks by one.
        // One of the two was only used by the parent seq, so this is now free.
        // The child seq still consumes sliding_window blocks
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS - SLIDING_WINDOW
        );

        // free all blocks
        block_manager
            .free(child.try_borrow().unwrap().sequence_id())
            .expect("Failed to free block manager");

        // assert all blocks are free now
        assert_eq!(
            block_manager.get_number_of_free_gpu_blocks(),
            NUM_GPU_BLOCKS
        );
    }
}
