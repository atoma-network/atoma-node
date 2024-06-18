use std::sync::{Arc, RwLock};

use thiserror::Error;
use tracing::{error, info_span, instrument, Span};

use crate::{
    block::{BlockDevice, BlockError, BlockTable, PhysicalTokenBlock, SyncPhysicalTokenBlock},
    types::{ReadLock, WriteLock},
};

/// `UncachedBlockAllocator` Manages free physical token blocks for a device, without cache.
///
/// The allocator maintains a list of free blocks and allocates a block when
/// requested. When a block is freed, its reference count is decremented. If
/// the reference count becomes zero, the block is added back to the free list.
#[allow(dead_code)]
#[derive(Debug)]
pub struct BlockAllocator {
    /// Block size
    #[allow(dead_code)]
    block_size: usize,
    /// Device
    device: BlockDevice,
    /// Number of blocks
    num_blocks: usize,
    /// Free blocks available
    pub(crate) free_blocks: BlockTable,
    /// Tracing span
    pub span: Span,
}

impl BlockAllocator {
    /// Constructor
    pub fn new(block_size: usize, device: BlockDevice, num_blocks: usize) -> Self {
        let free_blocks = (0..(num_blocks as u64))
            .map(|i| {
                Arc::new(RwLock::new(PhysicalTokenBlock::new(
                    i,
                    block_size,
                    device.clone(),
                )))
            })
            .collect();

        Self {
            block_size,
            device,
            num_blocks,
            free_blocks,
            span: info_span!("uncached-block-allocator"),
        }
    }
}

impl BlockAllocator {
    #[instrument]
    /// Allocates a new physical block
    pub fn allocate(&mut self) -> Result<SyncPhysicalTokenBlock, BlockAllocatorError> {
        if let Some(block) = self.free_blocks.pop() {
            block.write_lock()?.increment_ref_count();
            Ok(block)
        } else {
            error!("Out of memory, no available free blocks!");
            Err(BlockAllocatorError::OutOfMemory)
        }
    }

    /// Frees a given (already allocated) block
    #[instrument]
    pub fn free(&mut self, block: SyncPhysicalTokenBlock) -> Result<(), BlockAllocatorError> {
        {
            let block_guard = block.read_lock()?;
            let block_ref_count = block_guard.ref_count();
            let block_number = block_guard.block_number();
            if block_ref_count == 0 {
                error!(
                    "Double free! {} is already freed.",
                    block_guard.block_number()
                );
                return Err(BlockAllocatorError::CannotDoubleFree(block_number));
            }
        }

        let block_clone = block.clone();
        let mut block_write_guard = block_clone.write_lock()?;
        block_write_guard.decrease_ref_count()?;

        if block_write_guard.ref_count() == 0 {
            self.free_blocks.push(block);
        }

        Ok(())
    }

    /// Gets number of free blocks
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Gets total number of blocks
    pub fn get_num_total_blocks(&self) -> usize {
        self.num_blocks
    }
}

#[derive(Debug, Error)]
pub enum BlockAllocatorError {
    #[error("Cannot allocate further blocks, `cached_blocks` is full")]
    CannotAllocateNewBlock,
    #[error("Block already allocated in cached")]
    BlockAlreadyAllocated,
    #[error("Block already in use")]
    BlockAlreadyInUse,
    #[error("Cannot free unused block, with block_number = `{0}`")]
    CannotDoubleFree(u64),
    #[error("Block not found, with block_number = `{0}`")]
    BlockNotFound(u64),
    #[error("Failed to acquire read lock: `{0}`")]
    PoisonError(String),
    #[error("Out of memory error")]
    OutOfMemory,
    #[error("Block error: `{0}`")]
    BlockError(#[from] BlockError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator_allocate() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;

        let mut cpu_allocator = BlockAllocator::new(BLOCK_SIZE, BlockDevice::Cpu, NUM_CPU_BLOCKS);

        // Allocate all available CPU blocks
        let mut num_free_blocks = NUM_CPU_BLOCKS;
        assert_eq!(cpu_allocator.get_num_free_blocks(), num_free_blocks);
        for _ in 0..NUM_CPU_BLOCKS {
            let block = cpu_allocator.allocate().expect("Failed to allocate block");
            num_free_blocks -= 1;

            let block_id = block.read_lock().unwrap().block_number();
            // Allocated block is not part of free blocks, anymore
            assert!(cpu_allocator.free_blocks.iter().all(|block| block
                .read()
                .unwrap()
                .block_number()
                != block_id));
            assert_eq!(cpu_allocator.get_num_free_blocks(), num_free_blocks);
        }

        cpu_allocator
            .allocate()
            .err()
            .unwrap()
            .to_string()
            .contains("Out of memory error");
    }

    #[test]
    fn test_block_allocator_free() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;

        let mut cpu_allocator = BlockAllocator::new(BLOCK_SIZE, BlockDevice::Cpu, NUM_CPU_BLOCKS);

        // Allocate all available CPU blocks
        let mut blocks = Vec::with_capacity(NUM_CPU_BLOCKS);
        for _ in 0..NUM_CPU_BLOCKS {
            let block = cpu_allocator.allocate().expect("Failed to allocate block");

            blocks.push(block.clone());
            let block_guard = block.read().unwrap();

            assert!(!cpu_allocator
                .free_blocks
                .iter()
                .map(|block| block.read().unwrap().block_number())
                .collect::<Vec<_>>()
                .contains(&block_guard.block_number()));
        }

        // Free all the allocated cpu blocks
        let mut num_free_blocks = 0;
        assert_eq!(cpu_allocator.get_num_free_blocks(), num_free_blocks);
        for block in blocks {
            cpu_allocator
                .free(block.clone())
                .expect("Failed to free block");
            num_free_blocks += 1;

            let block_clone = block.clone();
            let block_guard = block_clone.read().unwrap();

            assert!(cpu_allocator
                .free_blocks
                .iter()
                .map(|block| { block.read().unwrap().block_number() })
                .collect::<Vec<_>>()
                .contains(&block_guard.block_number()));
            assert_eq!(cpu_allocator.get_num_free_blocks(), num_free_blocks);

            // Trying to free same block again should fail
            assert!(cpu_allocator
                .free(block)
                .err()
                .unwrap()
                .to_string()
                .contains(""));
        }
    }
}
