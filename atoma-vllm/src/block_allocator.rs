use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use candle::Device;
use thiserror::Error;
use tracing::{error, info, info_span, warn, Span};

use crate::{
    block::{BlockTable, PhysicalTokenBlock},
    evictor::{Evictor, EvictorError, LRUEvictor},
};

pub trait BlockAllocator {
    fn allocate_block(
        &mut self,
        _block_hash: u64,
        _num_hashed_tokens: usize,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockAllocatorError> {
        panic!("Not implemented")
    }
    fn allocate(
        &mut self,
        block_hash: Option<u64>,
        num_hashed_tokens: Option<usize>,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockAllocatorError>;
    fn free(&mut self, block: Arc<RwLock<PhysicalTokenBlock>>) -> Result<(), BlockAllocatorError>;
    fn get_num_free_blocks(&self) -> usize;
    fn get_num_total_blocks(&self) -> usize;
    fn contains_block(&self, _block_hash: u64) -> bool {
        panic!("Not implemented")
    }
    fn update_hash(
        &mut self,
        _block_hash: u64,
        _block: &mut PhysicalTokenBlock,
    ) -> Result<(), BlockAllocatorError> {
        panic!("Not implemented")
    }
}

/// `CachedBlockManager` manages free physical token blocks for a device.
///
/// The allocator maintains a list of free blocks and allocates a new block whenever requested.
/// When a block is freed, its reference counter field is decremented. Once the reference counter
/// equals zero, the block is added back to the free list.
pub struct CachedBlockAllocator {
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
    cached_blocks: HashMap<u64, Arc<RwLock<PhysicalTokenBlock>>>,
    /// Evictor
    evictor: LRUEvictor,
    /// Span that lives as long as the current instance
    pub span: Span,
}

impl CachedBlockAllocator {
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
            span: info_span!("cached-block-allocator"),
        }
    }
}

impl BlockAllocator for CachedBlockAllocator {
    /// Allocates a new block
    fn allocate_block(
        &mut self,
        block_hash: u64,
        num_hashed_tokens: usize,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockAllocatorError> {
        let _enter_span = self.span.enter();
        if self.current_num_blocks == self.num_blocks {
            info!("Current number blocks equals total number of blocks, evicting a block..");
            let mut evicted_block = self.evictor.evict()?;
            evicted_block.update_block_hash(block_hash);
            evicted_block.update_num_hashed_tokens(num_hashed_tokens);
            return Ok(Arc::new(RwLock::new(evicted_block)));
        }

        let block = PhysicalTokenBlock::new(
            block_hash,
            self.current_num_blocks,
            self.block_size,
            self.device.clone(),
        );

        // update the number of already allocated blocks
        self.current_num_blocks += 1;

        Ok(Arc::new(RwLock::new(block)))
    }

    /// Allocates a new block to the cache
    fn allocate(
        &mut self,
        mut block_hash: Option<u64>,
        num_hashed_tokens: Option<usize>,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockAllocatorError> {
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
                error!("Block with block_hash = {block_hash} has already been allocated in cache");
                return Err(BlockAllocatorError::BlockAlreadyAllocated);
            }

            // DON'T PANIC: checked that `self.evictor` contains `block_hash`
            let mut block = self.evictor.remove(block_hash).unwrap();
            if block.ref_count() != 0 {
                error!(
                    "Block with block_hash = {block_hash} already in use, ref_count = {}",
                    block.ref_count()
                );
                return Err(BlockAllocatorError::BlockAlreadyInUse);
            }

            debug_assert!(block.block_hash() == block_hash);

            block.increment_ref_count();
            let block = Arc::new(RwLock::new(block));
            self.cached_blocks.insert(block_hash, block.clone());

            return Ok(block);
        }

        #[allow(clippy::map_entry)]
        if !self.cached_blocks.contains_key(&block_hash) {
            let allocated_block =
                self.allocate_block(block_hash, num_hashed_tokens.unwrap_or_default())?;
            self.cached_blocks.insert(block_hash, allocated_block);
        }

        // DON'T PANIC: if original `block_hash` was not a key in `self.cached_blocks`, we inserted nonetheless
        let block = self.cached_blocks.get(&block_hash).unwrap();

        loop {
            if let Ok(mut guard) = block.write() {
                guard.increment_ref_count();
                break;
            }
        }

        Ok(block.clone())
    }

    /// Checks if current allocator contains block with `block_hash`
    fn contains_block(&self, block_hash: u64) -> bool {
        self.evictor.contains(block_hash) || self.cached_blocks.contains_key(&block_hash)
    }

    /// Free a new physical block
    fn free(&mut self, block: Arc<RwLock<PhysicalTokenBlock>>) -> Result<(), BlockAllocatorError> {
        let _enter_span = self.span.enter();
        let block_clone = block.clone();
        let (block_hash, block_ref_count) = {
            let guard = block_clone
                .read()
                .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?;
            (guard.block_hash(), guard.ref_count())
        };
        if block_ref_count == 0 {
            error!(
                "Double free! Block with block_hash = {}, is already freed.",
                block_hash
            );
            return Err(BlockAllocatorError::CannotDoubleFree(block_hash));
        }

        let block = {
            block
                .write()
                .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?
                .decrease_ref_count();

            block
        };

        let block_ref_count = {
            let guard = block
                .read()
                .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?;
            guard.ref_count()
        };

        if block_ref_count == 0 {
            if !self.evictor.contains(block_hash) {
                {
                    self.evictor.add(
                        block
                            .read()
                            .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?
                            .clone(),
                    );
                }

                // Remove the block from the cached_blocks
                self.cached_blocks.remove(&block_hash);
                self.current_num_blocks -= 1;
            } else {
                error!(
                    "Double free! Block with block_hash = {}, is already freed.",
                    block_hash
                );
                // Block already exists in the evictor
                return Err(BlockAllocatorError::CannotDoubleFree(block_hash));
            }
        }

        Ok(())
    }

    /// Get number of free blocks
    fn get_num_free_blocks(&self) -> usize {
        self.num_blocks - self.current_num_blocks
    }

    /// Getter for `num_blocks`
    fn get_num_total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Updates block hash
    fn update_hash(
        &mut self,
        block_hash: u64,
        block: &mut PhysicalTokenBlock,
    ) -> Result<(), BlockAllocatorError> {
        let enter_span = &self.span;
        let _ = enter_span.enter();

        if !self.contains_block(block_hash) {
            warn!("Block with block_hash {block_hash} not found..");
            return Err(BlockAllocatorError::BlockNotFound(block_hash));
        }

        let old_hash = block.block_hash();
        block.update_block_hash(block_hash);

        if let Some(block) = self.cached_blocks.remove(&old_hash) {
            info!("Updating block with new block hash in cache..");
            self.cached_blocks.insert(block_hash, block.clone());
            Ok(())
        } else {
            warn!("Block with old block hash {old_hash} not found in cache..");
            Err(BlockAllocatorError::BlockNotFound(block.block_hash()))
        }
    }
}

/// `UncachedBlockAllocator` Manages free physical token blocks for a device, without cache.
///
/// The allocator maintains a list of free blocks and allocates a block when
/// requested. When a block is freed, its reference count is decremented. If
/// the reference count becomes zero, the block is added back to the free list.
#[allow(dead_code)]
pub struct UncachedBlockAllocator {
    /// Block size
    block_size: usize,
    /// Device
    device: Device,
    /// Number of blocks
    num_blocks: usize,
    /// Free blocks available
    free_blocks: BlockTable,
    /// Tracing span
    pub span: Span,
}

impl UncachedBlockAllocator {
    /// Constructor
    pub fn new(block_size: usize, device: Device, num_blocks: usize) -> Self {
        let free_blocks = (0..num_blocks)
            .map(|i| {
                Arc::new(RwLock::new(PhysicalTokenBlock::new(
                    0,
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

impl BlockAllocator for UncachedBlockAllocator {
    fn allocate(
        &mut self,
        _block_hash: Option<u64>,
        _num_hashed_tokens: Option<usize>,
    ) -> Result<Arc<RwLock<PhysicalTokenBlock>>, BlockAllocatorError> {
        let _span = self.span.enter();
        if let Some(block) = self.free_blocks.pop() {
            block
                .write()
                .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?
                .increment_ref_count();
            Ok(block)
        } else {
            error!("Out of memory, no available free blocks!");
            Err(BlockAllocatorError::OutOfMemory)
        }
    }

    fn free(&mut self, block: Arc<RwLock<PhysicalTokenBlock>>) -> Result<(), BlockAllocatorError> {
        let _span = self.span.enter();
        {
            let block_guard = block
                .read()
                .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?;
            let block_ref_count = block_guard.ref_count();
            let block_number = block_guard.block_number();
            if block_ref_count == 0 {
                error!(
                    "Double free! {} is already freed.",
                    block_guard.block_number()
                );
                return Err(BlockAllocatorError::CannotDoubleFree(block_number as u64));
            }
        }

        let block_clone = block.clone();
        let mut block_write_guard = block_clone
            .write()
            .map_err(|e| BlockAllocatorError::PoisonError(e.to_string()))?;
        block_write_guard.decrease_ref_count();

        if block_write_guard.ref_count() == 0 {
            self.free_blocks.push(block);
        }

        Ok(())
    }

    fn get_num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    fn get_num_total_blocks(&self) -> usize {
        self.num_blocks
    }
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
    #[error("Block not found, with block_hash = `{0}`")]
    BlockNotFound(u64),
    #[error("Failed to acquire read lock: `{0}`")]
    PoisonError(String),
    #[error("Out of memory error")]
    OutOfMemory,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_block_allocator() {
        const BLOCK_SIZE: usize = 16;
        const NUM_BLOCKS: usize = 16;

        let block_hash = 1;
        let mut block_allocator = CachedBlockAllocator::new(BLOCK_SIZE, Device::Cpu, NUM_BLOCKS);

        // Allocate two `PhysicalTokenBlock` with same `block_hash` and check that these are allocated
        let first_block = block_allocator
            .allocate(Some(block_hash), None)
            .expect("Failed to allocate new block");
        let second_block = block_allocator
            .allocate(Some(block_hash), None)
            .expect("Failed to allocate new block");

        // Check equality between each field
        assert_eq!(*first_block.read().unwrap(), *second_block.read().unwrap());
        assert_eq!(second_block.read().unwrap().ref_count(), 2);

        // Free the `first_block` and check that `second_block` reference counter has decreased by 1
        block_allocator
            .free(first_block)
            .expect("Failed to free `first_block`");
        assert_eq!(second_block.read().unwrap().ref_count(), 1);

        // Reallocate the first block and confirm that we get the same block back
        let first_block = block_allocator
            .allocate(Some(block_hash), None)
            .expect("Failed to allocate new block");
        assert_eq!(*first_block.read().unwrap(), *second_block.read().unwrap());
        assert_eq!(second_block.read().unwrap().ref_count(), 2);
    }

    #[test]
    fn test_cache_block_eviction() {
        const BLOCK_SIZE: usize = 16;
        const NUM_BLOCKS: usize = 16;

        let mut block_allocator = CachedBlockAllocator::new(BLOCK_SIZE, Device::Cpu, NUM_BLOCKS);
        let mut blocks = vec![];

        // Allocate multiple blocks
        for i in 0..(NUM_BLOCKS as u64) {
            blocks.push(
                block_allocator
                    .allocate(Some(i), None)
                    .expect("Failed to allocate block"),
            );
        }

        // Free all blocks
        for block in blocks.iter() {
            block_allocator
                .free(block.clone())
                .expect("Failed to free block");
        }

        // Check that evicted blocks number is correct
        assert_eq!(block_allocator.current_num_blocks, 0);
        assert_eq!(
            block_allocator.get_num_free_blocks(),
            block_allocator.get_num_total_blocks()
        );

        // allocate a new block
        let new_block_hash = NUM_BLOCKS as u64;
        let new_block = block_allocator
            .allocate(Some(new_block_hash), None)
            .expect("Failed to allocate block");

        assert_eq!(new_block.read().unwrap().ref_count(), 1);
        assert_eq!(new_block.read().unwrap().block_number(), 0);
        assert_eq!(new_block.read().unwrap().block_size(), BLOCK_SIZE);
        assert_eq!(new_block.read().unwrap().last_accessed(), None);
        assert!(new_block.read().unwrap().device().is_cpu());
        assert_eq!(new_block.read().unwrap().num_hashed_tokens(), 0);
        assert_eq!(new_block.read().unwrap().block_hash(), new_block_hash);

        // Reallocate the second block in blocks, to remove it from the free list
        let realloc_block_hash = 1u64;
        let realloc_block = block_allocator
            .allocate(Some(realloc_block_hash), None)
            .expect("Failed to allocate block");

        assert_eq!(realloc_block.read().unwrap().ref_count(), 1);
        assert_eq!(realloc_block.read().unwrap().block_number(), 1);
        assert_eq!(
            realloc_block.read().unwrap().block_number(),
            blocks[realloc_block_hash as usize]
                .read()
                .unwrap()
                .block_number()
        );
        assert_eq!(
            realloc_block.read().unwrap().block_size(),
            blocks[realloc_block_hash as usize]
                .read()
                .unwrap()
                .block_size()
        );
        assert_eq!(
            realloc_block.read().unwrap().last_accessed(),
            blocks[realloc_block_hash as usize]
                .read()
                .unwrap()
                .last_accessed()
        );
        assert!(
            realloc_block.read().unwrap().device().is_cpu()
                && blocks[realloc_block_hash as usize]
                    .read()
                    .unwrap()
                    .device()
                    .is_cpu()
        );
        assert_eq!(
            realloc_block.read().unwrap().num_hashed_tokens(),
            blocks[realloc_block_hash as usize]
                .read()
                .unwrap()
                .num_hashed_tokens()
        );
        assert_eq!(
            realloc_block.read().unwrap().block_hash(),
            realloc_block_hash
        );

        // Allocate a new block and confirm that it's not the realloc_block,
        // since the realloc_block shouldn't be in the free list
        let new_block_hash = BLOCK_SIZE as u64 + 1;
        let new_block = block_allocator
            .allocate(Some(new_block_hash), None)
            .expect("Failed to allocate block");
        // assert (realloc_block != new_block)

        assert_ne!(*new_block.read().unwrap(), *realloc_block.read().unwrap());
        assert_eq!(new_block.read().unwrap().block_number(), 1);
    }

    #[test]
    fn test_uncache_block_allocator_allocate() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;

        let mut cpu_allocator =
            UncachedBlockAllocator::new(BLOCK_SIZE, Device::Cpu, NUM_CPU_BLOCKS);

        // Allocate all available CPU blocks
        let mut num_free_blocks = NUM_CPU_BLOCKS;
        assert_eq!(cpu_allocator.get_num_free_blocks(), num_free_blocks);
        for _ in 0..NUM_CPU_BLOCKS {
            let block = cpu_allocator
                .allocate(None, None)
                .expect("Failed to allocate block");
            num_free_blocks -= 1;

            // Allocated block is not part of free blocks, anymore
            assert!(!cpu_allocator
                .free_blocks
                .iter()
                .map(|block| block.read().unwrap().block_number())
                .collect::<Vec<_>>()
                .contains(&block.read().unwrap().block_number()));
            assert_eq!(cpu_allocator.get_num_free_blocks(), num_free_blocks);
        }

        cpu_allocator
            .allocate(None, None)
            .err()
            .unwrap()
            .to_string()
            .contains("Out of memory error");
    }

    #[test]
    fn test_uncached_block_allocator_free() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 4;

        let mut cpu_allocator =
            UncachedBlockAllocator::new(BLOCK_SIZE, Device::Cpu, NUM_CPU_BLOCKS);

        // Allocate all available CPU blocks
        let mut blocks = Vec::with_capacity(NUM_CPU_BLOCKS);
        for _ in 0..NUM_CPU_BLOCKS {
            let block = cpu_allocator
                .allocate(None, None)
                .expect("Failed to allocate block");

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
