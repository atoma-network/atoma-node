use std::collections::HashMap;

use candle::Device;
use thiserror::Error;
use tracing::{error, info, info_span, Span};

use crate::{
    block::PhysicalTokenBlock,
    evictor::{Evictor, EvictorError, LRUEvictor},
};

pub trait BlockAllocator {
    fn allocate_block(
        &mut self,
        block_hash: u64,
        num_hashed_tokens: usize,
    ) -> Result<PhysicalTokenBlock, BlockAllocatorError>;
    fn allocate(
        &mut self,
        block_hash: Option<u64>,
        num_hashed_tokens: usize,
    ) -> Result<PhysicalTokenBlock, BlockAllocatorError>;
    fn free(&mut self, block: &mut PhysicalTokenBlock) -> Result<(), BlockAllocatorError>;
    fn get_num_free_blocks(&self) -> usize;
    fn get_num_total_blocks(&self) -> usize;
    fn contains_block(&self, block_hash: u64) -> bool;
    fn update_hash(&self, block_hash: u64, block: PhysicalTokenBlock);
}

/// `BlockCachedAllocator` manages free physical token blocks for a device.
///
/// The allocator maintains a list of free blocks and allocates a new block whenever requested.
/// When a block is freed, its reference counter field is decremented. Once the reference counter
/// equals zero, the block is added back to the free list.
pub struct BlockCacheAllocator {
    /// Block size
    block_size: usize,
    /// Counter, used to identify blocks
    counter: u64,
    /// Device
    device: Device,
    /// Total number of blocks
    num_blocks: usize,
    /// Current number of allocated blocks
    current_num_blocks: usize,
    /// Cached blocks, indexed by `block_hash`
    cached_blocks: HashMap<u64, PhysicalTokenBlock>,
    /// Evictor
    evictor: LRUEvictor,
    /// Span that lives as long as the current instance
    pub span: Span,
}

impl BlockCacheAllocator {
    /// Constructor
    pub fn new(block_size: usize, device: Device, num_blocks: usize) -> Self {
        Self {
            block_size,
            counter: 0,
            device,
            num_blocks,
            current_num_blocks: 0,
            cached_blocks: HashMap::new(),
            evictor: LRUEvictor::new(),
            span: info_span!("block-cached-allocator"),
        }
    }
}

impl BlockAllocator for BlockCacheAllocator {
    /// Allocates a new block to the cache
    fn allocate_block(
        &mut self,
        block_hash: u64,
        num_hashed_tokens: usize,
    ) -> Result<PhysicalTokenBlock, BlockAllocatorError> {
        let _enter_span = self.span.enter();
        if self.current_num_blocks == self.num_blocks {
            info!("Current number blocks equals total number of blocks, evicting a block..");
            let mut evicted_block = self.evictor.evict()?;
            evicted_block.update_block_hash(block_hash);
            evicted_block.update_num_hashed_tokens(num_hashed_tokens);
            return Ok(evicted_block);
        }

        let block = PhysicalTokenBlock::new(
            block_hash,
            self.current_num_blocks,
            self.block_size,
            self.device.clone(),
        );

        // update the number of already allocated blocks
        self.current_num_blocks += 1;

        Ok(block)
    }

    fn allocate(
        &mut self,
        mut block_hash: Option<u64>,
        num_hashed_tokens: usize,
    ) -> Result<PhysicalTokenBlock, BlockAllocatorError> {
        let span = &self.span;
        let _ = span.enter();
        if block_hash.is_none() {
            self.counter += 1;
            block_hash = Some(self.counter);
        }
        // DON'T PANIC: already enforced that `block_hash` is not `None`
        let block_hash = block_hash.unwrap();
        if self.evictor.contains(block_hash) {
            if self.cached_blocks.contains_key(&block_hash) {
                info!("Block with block_hash = {block_hash} has already been allocated in cache");
                return Err(BlockAllocatorError::BlockAlreadyAllocated);
            }

            // DON'T PANIC: checked that `self.evictor` contains `block_hash`
            let mut block = self.evictor.remove(block_hash).unwrap();
            if block.ref_count() != 0 {
                info!(
                    "Block with block_hash = {block_hash} already in use, ref_count = {}",
                    block.ref_count()
                );
                return Err(BlockAllocatorError::BlockAlreadyInUse);
            }

            debug_assert!(block.block_hash() == block_hash);
            block.increment_ref_count();
            self.cached_blocks.insert(block_hash, block.clone());
            return Ok(block);
        }

        if !self.cached_blocks.contains_key(&block_hash) {
            let allocated_block = self.allocate_block(block_hash, num_hashed_tokens)?;
            self.cached_blocks.insert(block_hash, allocated_block);
        }

        // DON'T PANIC: if original `block_hash` was not a key in `self.cached_blocks`, we inserted nonetheless
        let block = self.cached_blocks.get_mut(&block_hash).unwrap();
        block.increment_ref_count();

        Ok(block.clone())
    }

    /// Checks if current allocator contains block with `block_hash`
    fn contains_block(&self, block_hash: u64) -> bool {
        self.evictor.contains(block_hash) || self.cached_blocks.contains_key(&block_hash)
    }

    /// Free a new physical block
    fn free(&mut self, block: &mut PhysicalTokenBlock) -> Result<(), BlockAllocatorError> {
        let _enter_span = self.span.enter();
        if block.ref_count() == 0 {
            error!(
                "Double free! Block with block_hash = {}, is already freed.",
                block.block_hash()
            );
            return Err(BlockAllocatorError::CannotDoubleFree(block.block_hash()));
        }

        block.decrease_ref_count();

        if block.ref_count() == 0 {
            if !self.evictor.contains(block.block_hash()) {
                self.evictor.add(block.clone());

                // Remove the block from the cached_blocks
                self.cached_blocks.remove(&block.block_hash());
            } else {
                error!(
                    "Double free! Block with block_hash = {}, is already freed.",
                    block.block_hash()
                );
                // Block already exists in the evictor
                return Err(BlockAllocatorError::CannotDoubleFree(block.block_hash()));
            }
        }

        Ok(())
    }

    /// Get number of blocks
    fn get_num_free_blocks(&self) -> usize {
        return self.num_blocks - self.current_num_blocks + self.evictor.num_blocks();
    }

    /// Getter for `num_blocks`
    fn get_num_total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Updates block hash
    fn update_hash(&self, block_hash: u64, block: PhysicalTokenBlock) {}
}

#[derive(Debug, Error)]
pub enum BlockAllocatorError {
    #[error("Cannot allocate further blocks, `cached_blocks` is full")]
    CannotAllocateNewBlock,
    #[error("Evictor error: `{0}`")]
    EvictorError(#[from] EvictorError),
    #[error("Block already allocated in cached")]
    BlockAlreadyAllocated,
    #[error("Block already in use")]
    BlockAlreadyInUse,
    #[error("Cannot free unused block, with block_hash = `{0}`")]
    CannotDoubleFree(u64),
}
