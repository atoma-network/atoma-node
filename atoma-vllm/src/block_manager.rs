use std::{
    collections::HashMap,
    sync::{Arc, PoisonError, RwLock},
};

use crate::{
    block::{BlockTable, PhysicalTokenBlock},
    block_allocator::{
        BlockAllocator, BlockAllocatorError, CachedBlockAllocator, UncachedBlockAllocator,
    },
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
    cpu_allocator: Box<dyn BlockAllocator>,
    /// GPU allocator
    gpu_allocator: Box<dyn BlockAllocator>,
    /// Watermark
    watermark: f32,
    /// Watermark blocks
    watermark_blocks: u32,
    /// Block sliding window
    block_sliding_window: Option<usize>,
    /// Enable caching
    enable_caching: bool,
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
        enable_caching: bool,
    ) -> Result<Self, BlockSpaceManagerError> {
        if enable_caching && sliding_window.is_some() {
            return Err(BlockSpaceManagerError::SlidingWindowDisabledWithCaching);
        }

        let block_sliding_window = sliding_window.map(|sw| sw.div_ceil(block_size));

        let watermark_blocks = (watermark * num_gpu_blocks as f32).round() as u32;
        let span = info_span!("block-space-manager");

        let (cpu_allocator, gpu_allocator): (Box<dyn BlockAllocator>, Box<dyn BlockAllocator>) =
            if enable_caching {
                info!("Block space manager uses cached block allocator");
                (
                    Box::new(CachedBlockAllocator::new(
                        block_size,
                        Device::Cpu,
                        num_cpu_blocks,
                    )),
                    Box::new(CachedBlockAllocator::new(
                        block_size,
                        Device::new_cuda(device)?,
                        num_gpu_blocks,
                    )),
                )
            } else {
                info!("Block space managar uses uncached block allocator");
                (
                    Box::new(UncachedBlockAllocator::new(
                        block_size,
                        Device::Cpu,
                        num_cpu_blocks,
                    )),
                    Box::new(UncachedBlockAllocator::new(
                        block_size,
                        Device::new_cuda(device)?,
                        num_gpu_blocks,
                    )),
                )
            };

        Ok(Self {
            block_size,
            block_tables: HashMap::new(),
            cpu_allocator: cpu_allocator,
            gpu_allocator: gpu_allocator,
            num_cpu_blocks,
            num_gpu_blocks,
            block_sliding_window,
            enable_caching,
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
            let mut block_table: Vec<Arc<RwLock<PhysicalTokenBlock>>> =
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
                        let mut block_guard = match block.write() {
                            Ok(v) => v,
                            Err(e) => {
                                error!(
                                    "Failed to acquire lock for sequence_group with id = {}",
                                    seq_group.request_id
                                );
                                return Err(BlockSpaceManagerError::PoisonError(e.to_string()));
                            }
                        };
                        block_guard.increment_ref_count_by(
                            seq_group.get_num_sequences(Some(SequenceStatus::Waiting)),
                        );
                    }
                    block.clone()
                } else if self.enable_caching {
                    let block_hash = sequence.hash_of_block(logical_idx);
                    let num_hashed_tokens = sequence.num_hashed_tokens_of_block(logical_idx);
                    let block = self
                        .gpu_allocator
                        .allocate(Some(block_hash), Some(num_hashed_tokens))?;
                    block
                } else {
                    let block = self.gpu_allocator.allocate(None, None)?;
                    {
                        let mut block_guard = match block.write() {
                            Ok(v) => v,
                            Err(e) => {
                                error!(
                                    "Failed to acquire lock for sequence_group with id = {}",
                                    seq_group.request_id
                                );
                                return Err(BlockSpaceManagerError::PoisonError(e.to_string()));
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

    /// Promotes last block
    #[instrument]
    fn promote_last_block(
        &mut self,
        sequence: Sequence,
        last_block: Arc<RwLock<PhysicalTokenBlock>>,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockSpaceManagerError> {
        if !self.enable_caching {
            error!("`promote_last_block` method only supported with `enable_caching`");
            return Err(BlockSpaceManagerError::MethodNotSupported(
                "promote_last_block".into(),
            ));
        }

        // Computes a new hash for the block, so that it can be shared by other sequences
        let new_hash = sequence.hash_of_block(sequence.get_num_total_logical_token_blocks() - 1);

        // is the `new_hash` is already in cache table, then free `last_block` and return the cache version
        if self.gpu_allocator.contains_block(new_hash) {
            info!("Gpu allocator already contains a block with `{new_hash}`, freeing last block..");
            self.gpu_allocator.free(last_block)?;
            Ok(self.gpu_allocator.allocate(Some(new_hash), None)?)
        } else {
            self.gpu_allocator
                .update_hash(new_hash, last_block.clone())?;
            Ok(last_block)
        }
    }

    /// Checks if the last block is already full
    #[instrument]
    fn is_last_block_full(&self, sequence: &Sequence) -> bool {
        let token_ids_len = sequence.length();
        token_ids_len > 0 && (token_ids_len % sequence.block_size() == 0)
    }

    /// Try promote last block
    #[instrument]
    fn maybe_promote_last_block(
        &mut self,
        sequence: Sequence,
        last_block: Arc<RwLock<PhysicalTokenBlock>>,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockSpaceManagerError> {
        if self.is_last_block_full(&sequence) {
            self.promote_last_block(sequence, last_block)
        } else {
            Ok(last_block)
        }
    }

    /// Allocates a new physical block,
    #[instrument]
    fn allocate_last_block(
        &mut self,
        sequence: Sequence,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockSpaceManagerError> {
        if !self.enable_caching {
            return Ok(self.gpu_allocator.allocate(None, None)?);
        }

        let mut block_hash = None;
        let logical_idx = sequence.length() - 1;

        // None if the last block is not full. Otherwise, we set it to the
        // content hash of `last_block`
        if self.is_last_block_full(&sequence) {
            block_hash = Some(sequence.hash_of_block(logical_idx));
        }

        let num_hashed_tokens = sequence.num_hashed_tokens_of_block(logical_idx);

        // `num_hashed_tokens` is used to compute future hashes
        // (e.g. in the hashing function, it is used to ask the sequence for
        // prefix tokens)
        let new_block = self
            .gpu_allocator
            .allocate(block_hash, Some(num_hashed_tokens))?;

        // The `block_hash` being `None` means that `last_block` was not full.
        // In that case, `reference_count` should be 1
        if block_hash.is_none() {
            let block_guard = new_block
                .read()
                .map_err(|e| BlockSpaceManagerError::PoisonError(e.to_string()))?;
            let block_ref_count = block_guard.ref_count();
            if block_ref_count != 1 {
                return Err(BlockSpaceManagerError::InvalidRefCount(block_ref_count));
            }
        }

        Ok(new_block)
    }

    /// Allocates a new physical slot for a new token
    pub fn append_slots(
        &mut self,
        sequence: Sequence,
    ) -> Result<Option<(usize, usize)>, BlockSpaceManagerError> {
        let num_total_logical_token_blocks = sequence.get_num_total_logical_token_blocks();
        if let Some(block_table) = self.block_tables.get_mut(&sequence.sequence_id()) {
            // If we need to allocate a new physical block
            if block_table.len() < num_total_logical_token_blocks {
                if block_table.len() != num_total_logical_token_blocks - 1 {
                    return Err(BlockSpaceManagerError::AppendSlotError(
                        "Can only allocate one physical block at the time".into(),
                    ));
                }

                if self.block_sliding_window.is_some()
                    && block_table.len() >= self.block_sliding_window.unwrap()
                {
                    // Reuse a block
                    // DON'T PANIC: `self.block_sliding_window` is not `None` and `block_table` length
                    block_table.push(
                        block_table
                            .get(block_table.len() % self.block_sliding_window.unwrap())
                            .unwrap().clone(),
                    );
                }
            }
        }
        Ok(None)
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
}
