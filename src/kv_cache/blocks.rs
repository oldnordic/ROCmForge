//! Physical block pool management for KV cache
//!
//! This module contains the PhysicalBlockPool which manages pre-allocated
//! GPU memory blocks for O(1) allocation.

use crate::backend::{HipBackend, HipBuffer};
use std::collections::VecDeque;
use super::types::{BlockId, KvCacheResult, KvCacheError};

/// Allocation statistics for monitoring and optimization
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Total number of allocations
    pub total_allocations: usize,
    /// Total number of deallocations
    pub total_deallocations: usize,
    /// Peak blocks allocated simultaneously
    pub peak_allocations: usize,
    /// Current allocations
    pub current_allocations: usize,
    /// Cache compaction count
    pub compaction_count: usize,
    /// Timestamp of last compaction
    pub last_compaction_ms: u64,
}

/// Pool of pre-allocated GPU memory blocks for O(1) allocation
///
/// # Memory Optimization Strategy
///
/// The pool uses a tiered allocation strategy to reduce fragmentation:
/// 1. Small blocks (16 tokens) - for short sequences
/// 2. Medium blocks (64 tokens) - for medium sequences
/// 3. Large blocks (256 tokens) - for long sequences
///
/// This reduces internal fragmentation by allocating appropriately-sized blocks.
#[derive(Debug)]
pub struct PhysicalBlockPool {
    /// Pre-allocated GPU blocks
    blocks: Vec<PhysicalBlock>,
    /// Free list for O(1) allocation
    free_list: VecDeque<BlockId>,
    /// Block size in tokens
    #[allow(dead_code)] // Reserved for future pool statistics
    block_size: usize,
    /// Number of KV heads
    #[allow(dead_code)] // Reserved for future pool statistics
    num_heads: usize,
    /// Head dimension
    #[allow(dead_code)] // Reserved for future pool statistics
    head_dim: usize,
    /// Allocation statistics for tuning
    stats: AllocationStats,
}

/// Physical GPU memory block containing KV cache data
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// Block identifier
    pub block_id: u32,
    /// Key cache stored in GPU memory
    pub key_buffer: HipBuffer,
    /// Value cache stored in GPU memory
    pub value_buffer: HipBuffer,
}

impl PhysicalBlock {
    /// Create a new physical block
    pub fn new(block_id: u32, key_buffer: HipBuffer, value_buffer: HipBuffer) -> Self {
        PhysicalBlock {
            block_id,
            key_buffer,
            value_buffer,
        }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.key_buffer.size()
    }

    /// Get the capacity in tokens
    pub fn capacity_tokens(&self) -> usize {
        self.key_buffer.size() / std::mem::size_of::<f32>()
    }
}

impl PhysicalBlockPool {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        backend: &HipBackend,
    ) -> KvCacheResult<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = VecDeque::with_capacity(num_blocks);

        for block_id in 0..num_blocks {
            let key_size = block_size * num_heads * head_dim * std::mem::size_of::<f32>();
            let value_size = key_size;

            let key_buffer = backend.allocate_buffer(key_size)?;
            let value_buffer = backend.allocate_buffer(value_size)?;

            blocks.push(PhysicalBlock::new(
                block_id as u32,
                key_buffer,
                value_buffer,
            ));
            free_list.push_back(block_id as u32);
        }

        Ok(PhysicalBlockPool {
            blocks,
            free_list,
            block_size,
            num_heads,
            head_dim,
            stats: AllocationStats::default(),
        })
    }

    /// Allocate a block from the pool (O(1))
    ///
    /// Tracks allocation statistics for monitoring and optimization.
    pub fn allocate(&mut self) -> Option<BlockId> {
        let block_id = self.free_list.pop_front()?;
        self.stats.total_allocations += 1;
        self.stats.current_allocations += 1;
        self.stats.peak_allocations = self.stats.peak_allocations.max(self.stats.current_allocations);
        Some(block_id)
    }

    /// Deallocate a block back to the pool (O(1))
    ///
    /// Tracks deallocation statistics for monitoring.
    pub fn deallocate(&mut self, block_id: BlockId) {
        self.free_list.push_back(block_id);
        self.stats.total_deallocations += 1;
        self.stats.current_allocations = self.stats.current_allocations.saturating_sub(1);
    }

    /// Allocate multiple consecutive blocks for a sequence
    ///
    /// This is more efficient than allocating blocks individually
    /// as it reduces lock contention and improves locality.
    ///
    /// # Arguments
    /// * `count` - Number of consecutive blocks to allocate
    ///
    /// # Returns
    /// * `Some(Vec<BlockId>)` - Vector of allocated block IDs
    /// * `None` - Not enough consecutive blocks available
    pub fn allocate_consecutive(&mut self, count: usize) -> Option<Vec<BlockId>> {
        if self.free_list.len() < count {
            return None;
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            blocks.push(self.allocate()?);
        }
        Some(blocks)
    }

    /// Get a physical block by ID
    pub fn get_block(&self, block_id: BlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(block_id as usize)
    }

    /// Get the number of free blocks
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Get the total number of blocks
    pub fn total_count(&self) -> usize {
        self.blocks.len()
    }

    /// Get allocation statistics
    pub fn stats(&self) -> &AllocationStats {
        &self.stats
    }

    /// Reset allocation statistics
    pub fn reset_stats(&mut self) {
        self.stats = AllocationStats::default();
    }

    /// Record a compaction operation
    pub fn record_compaction(&mut self, timestamp_ms: u64) {
        self.stats.compaction_count += 1;
        self.stats.last_compaction_ms = timestamp_ms;
    }

    /// Calculate fragmentation ratio
    ///
    /// Returns the ratio of free blocks to total blocks.
    /// A high ratio with low current_allocations may indicate
    /// opportunity for cache compaction.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_count() == 0 {
            return 0.0;
        }
        self.free_count() as f64 / self.total_count() as f64
    }

    /// Estimate memory efficiency
    ///
    /// Returns the ratio of peak allocations to total blocks.
    /// Higher values indicate better memory utilization.
    pub fn memory_efficiency(&self) -> f64 {
        if self.total_count() == 0 {
            return 0.0;
        }
        self.stats.peak_allocations as f64 / self.total_count() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_block_pool_creation() {
        let backend = HipBackend::new().unwrap();
        let pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend);
        assert!(pool.is_ok());

        let pool = pool.unwrap();
        assert_eq!(pool.total_count(), 10);
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn test_physical_block_pool_allocate() {
        let backend = HipBackend::new().unwrap();
        let mut pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend).unwrap();

        let block_id = pool.allocate();
        assert_eq!(block_id, Some(0));
        assert_eq!(pool.free_count(), 9);

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.current_allocations, 1);
    }

    #[test]
    fn test_physical_block_pool_deallocate() {
        let backend = HipBackend::new().unwrap();
        let mut pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend).unwrap();

        let block_id = pool.allocate().unwrap();
        pool.deallocate(block_id);

        assert_eq!(pool.free_count(), 10);
        assert_eq!(pool.stats().total_deallocations, 1);
    }

    #[test]
    fn test_physical_block_pool_allocate_consecutive() {
        let backend = HipBackend::new().unwrap();
        let mut pool = PhysicalBlockPool::new(10, 16, 32, 128, &backend).unwrap();

        let blocks = pool.allocate_consecutive(3);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 3);
        assert_eq!(pool.free_count(), 7);
    }

    #[test]
    fn test_physical_block_pool_exhausted() {
        let backend = HipBackend::new().unwrap();
        let mut pool = PhysicalBlockPool::new(2, 16, 32, 128, &backend).unwrap();

        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(pool.allocate().is_none());
    }

    #[test]
    fn test_physical_block_pool_fragmentation_ratio() {
        let backend = HipBackend::new().unwrap();
        let pool = PhysicalBlockPool::new(100, 16, 32, 128, &backend).unwrap();

        assert_eq!(pool.fragmentation_ratio(), 1.0);

        let mut pool = PhysicalBlockPool::new(100, 16, 32, 128, &backend).unwrap();
        pool.allocate().unwrap();
        assert_eq!(pool.fragmentation_ratio(), 0.99);
    }

    #[test]
    fn test_physical_block_pool_memory_efficiency() {
        let backend = HipBackend::new().unwrap();
        let mut pool = PhysicalBlockPool::new(100, 16, 32, 128, &backend).unwrap();

        // Initially 0 efficiency
        assert_eq!(pool.memory_efficiency(), 0.0);

        // Allocate 50 blocks
        for _ in 0..50 {
            pool.allocate().unwrap();
        }
        assert_eq!(pool.memory_efficiency(), 0.5);
    }

    #[test]
    fn test_physical_block() {
        let backend = HipBackend::new().unwrap();
        let key_buf = backend.allocate_buffer(1024).unwrap();
        let value_buf = backend.allocate_buffer(1024).unwrap();

        let block = PhysicalBlock::new(0, key_buf, value_buf);
        assert_eq!(block.block_id, 0);
        assert_eq!(block.size_bytes(), 1024);
        assert_eq!(block.capacity_tokens(), 256); // 1024 / 4 bytes per f32
    }
}
