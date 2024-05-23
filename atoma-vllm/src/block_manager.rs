use std::collections::HashMap;

use crate::{
    block::BlockTable,
    block_allocator::{BlockAllocator, CachedBlockAllocator, UncachedBlockAllocator},
};
use candle::Device;
use thiserror::Error;
use tracing::{info_span, Span};

/// `AllocationStatus` - enum that keeps a state of status of possible block allocation
pub enum AllocationStatus {
    /// Ok: seq_group can be allocated now.
    Ok,
    /// Later: `seq_group` cannot be allocated.
    /// The capacity of allocator is larger than seq_group required.
    Later,
    /// Never: `seq_group` can never be allocated.
    /// The `seq_group` is too large to allocated in GPU.
    Never,
}

/// `BlockSpaceManager` - Manages the mapping between logical and physical token blocks.
pub struct BlockSpaceManager {
    /// Block size
    block_size: usize,
    /// Block tables, mapping: `seq_id` -> `BlockTable`
    block_tables: HashMap<u64, BlockTable>,
    /// Number of CPU blocks
    num_cpu_blocks: usize,
    /// Number of GPU blocks
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
    block_sliding_window: Option<u64>,
    /// Sliding window
    sliding_window: Option<u64>,
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
        sliding_window: Option<u64>,
        enable_caching: bool,
    ) -> Result<Self, BlockSpaceManagerError> {
        if enable_caching && sliding_window.is_some() {
            return Err(BlockSpaceManagerError::SlidingWindowDisabledWithCaching);
        }

        let block_sliding_window = sliding_window.map(|sw| sw / block_size as u64);

        let watermark_blocks = (watermark * num_gpu_blocks as f32).round() as u32;

        let (cpu_allocator, gpu_allocator): (Box<dyn BlockAllocator>, Box<dyn BlockAllocator>) =
            if enable_caching {
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
            sliding_window,
            enable_caching,
            span: info_span!("block-space-manager"),
            watermark,
            watermark_blocks,
        })
    }

    /// Checks if it is possible to allocate enough blocks for current
    /// `seq_group`, with output an `AllocationStatus`
    fn can_allocate(&self, seq_group: SequenceGroup) -> AllocationStatus {
        // FUTURE: In this implementation, we assume that all sequences in `SequenceGroup` share the
        // same prompt. This might not be the case for preempted sequences.
        let seq = seq_group.get_sequences();
    }
}

#[derive(Debug, Error)]
pub enum BlockSpaceManagerError {
    #[error("Sliding window is not allowed with prefix caching enabled")]
    SlidingWindowDisabledWithCaching,
    #[error("Candle error: `{0}`")]
    CandleError(#[from] candle::Error),
}
