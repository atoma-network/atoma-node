use indexmap::IndexMap;
use thiserror::Error;

use crate::block::PhysicalTokenBlock;

pub trait Evictor {
    fn contains(&self, block_hash: u64) -> bool;
    fn evict(&mut self) -> Result<PhysicalTokenBlock, EvictorError>;
    fn add(&mut self, block: PhysicalTokenBlock);
    fn remove(&mut self, block_hash: u64) -> Option<PhysicalTokenBlock>;
    fn num_blocks(&self) -> usize;
}

/// The `LRUEvictor` struct evicts cached blocks based on their last_accessed timestamp, which represents the last time the block was accessed.
/// If there are multiple blocks with the same `last_accessed` time, the block with the largest `num_hashed_tokens` will be evicted.
/// If multiple blocks have the same `last_accessed` time and the highest `num_hashed_tokens` value, one of them will be chosen arbitrarily.
pub struct LRUEvictor {
    free_table: IndexMap<u64, PhysicalTokenBlock>,
}

impl LRUEvictor {
    /// Constructor
    pub fn new() -> Self {
        Self {
            free_table: IndexMap::new(),
        }
    }
}

impl Evictor for LRUEvictor {
    /// Checks if `LRUEvictor` contains a block for the corresponding `block_hash`
    fn contains(&self, block_hash: u64) -> bool {
        self.free_table.contains_key(&block_hash)
    }

    /// Evicts a cached block, based on the eviction policy
    fn evict(&mut self) -> Result<PhysicalTokenBlock, EvictorError> {
        if self.free_table.is_empty() {
            return Err(EvictorError::EmptyFreeTable);
        }

        // Step 1: Find the block to evict
        let mut evicted_block_key = None;
        let mut evicted_block: Option<PhysicalTokenBlock> = None;

        // The blocks with the lowest `last_accessed` should be placed consecutively
        // at the start of `free_table`. Loop through all these blocks to
        // find the one with maximum number of hashed tokens.
        for (key, block) in &self.free_table {
            if let Some(current_evicted_block) = &evicted_block {
                if current_evicted_block.last_accessed() < block.last_accessed() {
                    break;
                }
                if current_evicted_block.num_hashed_tokens() < block.num_hashed_tokens() {
                    evicted_block = Some(block.clone());
                    evicted_block_key = Some(*key);
                }
            } else {
                evicted_block = Some(block.clone());
                evicted_block_key = Some(*key);
            }
        }

        // Step 2: Remove the block from the free table
        if let Some(key) = evicted_block_key {
            let mut evicted_block = self.free_table.shift_remove(&key).unwrap(); // DON'T PANIC: we already checked that `free_table` is not empty
            evicted_block.not_computed();
            return Ok(evicted_block);
        }

        Err(EvictorError::EmptyFreeTable)
    }

    /// Adds a new block to `free_table`
    fn add(&mut self, block: PhysicalTokenBlock) {
        self.free_table.insert(block.block_hash(), block);
    }

    /// Removes, if possible, a block with `block_hash`
    fn remove(&mut self, block_hash: u64) -> Option<PhysicalTokenBlock> {
        self.free_table.shift_remove(&block_hash)
    }

    /// Gets the number of blocks in `free_table`
    fn num_blocks(&self) -> usize {
        self.free_table.len()
    }
}

#[derive(Debug, Error)]
pub enum EvictorError {
    #[error("Free table is empty")]
    EmptyFreeTable,
}
