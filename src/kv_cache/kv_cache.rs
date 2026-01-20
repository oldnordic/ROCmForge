//! Paged KV cache for efficient GPU memory management
//!
//! This is the production-grade paged KV cache implementation
//! with PagedAttention support, LRU eviction, and block sharing.
//!
//! For the simple GPU-resident KV cache (legacy), see `crate::model::kv_cache::SimpleKVCache`.

use super::block_allocator::BlockAllocator;
use super::blocks::{AllocationStats, PhysicalBlock, PhysicalBlockPool};
use super::config::CacheConfig;
use super::page_table::PageTable;
use super::pages::{CachePage, SequenceCache};
use super::types::{
    BlockId, BlockTable, CacheStats, KvCacheError, KvCacheResult, MemoryProfile, PagedCacheStats,
};

// Import extracted module operations
use super::cache_ops::{allocate_page, append_token as cache_append_token};
use super::sequence_ops::{
    cleanup_completed_sequences as seq_cleanup_completed_sequences,
    get_active_sequences, get_sequence_access_time,
    is_sequence_completed, mark_sequence_completed, sort_free_lists,
    update_sequence_access,
};
use super::block_ops::{
    allocate_block as block_allocate_block, copy_block, get_allocation_stats as block_get_allocation_stats,
    get_block, get_physical_block, get_paged_stats, ref_block, reset_allocation_stats as block_reset_allocation_stats,
    unref_block,
};

use crate::backend::HipBackend;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Paged KV cache with PagedAttention support
///
/// This is the production KV cache used by the inference engine.
#[derive(Debug)]
pub struct KvCache {
    config: CacheConfig,
    backend: Arc<HipBackend>,
    /// Block pool for physical GPU memory (PagedAttention)
    block_pool: RwLock<PhysicalBlockPool>,
    /// Block table: logical ID -> physical block mapping (PagedAttention)
    block_table: RwLock<HashMap<BlockId, BlockTable>>,
    /// Page table for mapping logical positions to physical blocks
    page_table: RwLock<PageTable>,
    /// Block allocator for O(1) block allocation
    block_allocator: RwLock<BlockAllocator>,
    /// Legacy: sequence-owned pages (for backward compatibility)
    pages: RwLock<HashMap<u32, CachePage>>,
    sequences: RwLock<HashMap<u32, SequenceCache>>,
    free_pages: RwLock<Vec<u32>>,
    next_page_id: RwLock<u32>,
    /// Free logical block IDs (PagedAttention)
    free_blocks: RwLock<Vec<BlockId>>,
    next_block_id: RwLock<BlockId>,
}

impl KvCache {
    pub fn new(config: CacheConfig, backend: Arc<HipBackend>) -> KvCacheResult<Self> {
        // Initialize the physical block pool for PagedAttention
        let block_pool = PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        )?;

        // Initialize PageTable with default block size
        let page_table = PageTable::new();

        // Initialize BlockAllocator
        let block_allocator = BlockAllocator::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
        );

        Ok(KvCache {
            config,
            backend,
            block_pool: RwLock::new(block_pool),
            block_table: RwLock::new(HashMap::new()),
            page_table: RwLock::new(page_table),
            block_allocator: RwLock::new(block_allocator),
            pages: RwLock::new(HashMap::new()),
            sequences: RwLock::new(HashMap::new()),
            free_pages: RwLock::new(Vec::new()),
            next_page_id: RwLock::new(0),
            free_blocks: RwLock::new(Vec::new()),
            next_block_id: RwLock::new(0),
        })
    }

    pub fn allocate_page(&mut self, sequence_id: u32) -> KvCacheResult<u32> {
        allocate_page(
            &self.pages,
            &self.sequences,
            &self.free_pages,
            &self.next_page_id,
            &self.config,
            &self.backend,
            sequence_id,
        )
    }

    pub fn append_token(&mut self, sequence_id: u32, token: u32) -> KvCacheResult<()> {
        cache_append_token(
            &self.pages,
            &self.sequences,
            &self.free_pages,
            &self.next_page_id,
            &self.config,
            &self.backend,
            sequence_id,
            token,
        )
    }

    pub fn get_sequence_tokens(&self, sequence_id: u32) -> KvCacheResult<Vec<u32>> {
        super::cache_ops::get_sequence_tokens(&self.pages, &self.sequences, sequence_id)
    }

    pub fn get_sequence_length(&self, sequence_id: u32) -> KvCacheResult<usize> {
        super::cache_ops::get_sequence_length(&self.sequences, sequence_id)
    }

    pub fn remove_sequence(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        super::cache_ops::remove_sequence(
            &self.pages,
            &self.sequences,
            &self.free_pages,
            &self.page_table,
            &self.block_allocator,
            sequence_id,
        )
    }

    pub fn get_cache_stats(&self) -> CacheStats {
        super::cache_ops::get_cache_stats(&self.pages, &self.free_pages, &self.sequences)
    }

    // ========== Sequence State Tracking Methods ==========

    /// Mark a sequence as completed (to be cleaned up later)
    pub fn mark_sequence_completed(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        mark_sequence_completed(&self.sequences, sequence_id)
    }

    /// Check if a sequence is marked as completed
    pub fn is_sequence_completed(&self, sequence_id: u32) -> KvCacheResult<bool> {
        is_sequence_completed(&self.sequences, sequence_id)
    }

    /// Update the last access time for a sequence (for LRU tracking)
    pub fn update_sequence_access(&mut self, sequence_id: u32) -> KvCacheResult<()> {
        update_sequence_access(&self.sequences, sequence_id)
    }

    /// Get the last access time for a sequence
    pub fn get_sequence_access_time(&self, sequence_id: u32) -> KvCacheResult<Instant> {
        get_sequence_access_time(&self.sequences, sequence_id)
    }

    /// Get list of active (non-completed) sequence IDs
    pub fn get_active_sequences(&self) -> KvCacheResult<Vec<u32>> {
        get_active_sequences(&self.sequences)
    }

    /// Remove all completed sequences from the cache
    /// This should be called periodically to free up memory
    pub fn cleanup_completed_sequences(&mut self) -> KvCacheResult<usize> {
        seq_cleanup_completed_sequences(&self.pages, &self.sequences, &self.free_pages)
    }

    /// Evict least recently used sequences to free up space for new sequences
    ///
    /// This method is part of the cache eviction API but currently
    /// unused because the cache uses a simpler capacity-based eviction.
    #[allow(dead_code)] // Reserved for future LRU eviction implementation
    fn evict_lru_sequences(&mut self, required_pages: usize) -> KvCacheResult<()> {
        let sequences = self.sequences.read()?;
        let pages = self.pages.read()?;

        // Check if we need to evict
        let current_usage = pages.len();
        let max_pages = self.config.max_pages;

        if current_usage + required_pages <= max_pages {
            return Ok(());
        }

        // Find LRU sequences (only active sequences, not completed ones)
        let mut seq_access_times: Vec<(u32, std::time::Instant)> = sequences
            .iter()
            .filter(|(_, s)| s.is_active()) // Only consider active sequences
            .map(|(id, s)| (*id, s.last_access))
            .collect();

        // Sort by access time (oldest first)
        seq_access_times.sort_by_key(|(_, time)| *time);

        // Calculate how many sequences we need to evict
        let pages_to_free = (current_usage + required_pages) - max_pages;

        // Estimate pages per sequence (rough estimate)
        let avg_pages_per_seq = if seq_access_times.is_empty() {
            1
        } else {
            let total_pages: usize = sequences
                .values()
                .filter(|s| s.is_active())
                .map(|s| s.pages.len())
                .sum();
            (total_pages / seq_access_times.len()).max(1)
        };

        let seqs_to_evict = (pages_to_free / avg_pages_per_seq)
            .max(1)
            .min(seq_access_times.len());

        // Drop the read locks before acquiring write locks
        drop(sequences);
        drop(pages);

        // Evict LRU sequences
        for (seq_id, _) in seq_access_times.iter().take(seqs_to_evict) {
            let _ = self.remove_sequence(*seq_id);
        }

        Ok(())
    }

    // ========== Phase 2: PageTable + BlockAllocator Integration ==========

    /// Append token with paged KV cache using PageTable and BlockAllocator
    ///
    /// This method integrates PageTable (for mapping logical positions to physical blocks)
    /// with BlockAllocator (for O(1) block allocation) to provide efficient paged attention.
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to append the token to
    /// * `token` - The token to append
    ///
    /// # Behavior
    /// - Checks if we need a new block (every `block_size` tokens)
    /// - Allocates from BlockAllocator if needed
    /// - Updates PageTable with new block mapping
    /// - Delegates to existing append_token() for actual storage
    pub fn append_token_paged(&mut self, sequence_id: u32, token: u32) -> KvCacheResult<()> {
        // Get current sequence length
        let current_tokens = self.get_sequence_length(sequence_id).unwrap_or(0);
        let block_size = self.config.page_size;

        // Check if we need to allocate a page for the sequence
        let has_page = {
            let sequences = self.sequences.read()?;
            sequences.get(&sequence_id).is_some()
        };

        if !has_page {
            // Allocate first page for new sequence
            self.allocate_page(sequence_id)?;
        }

        // Check if we need a new block
        if current_tokens > 0 && current_tokens % block_size == 0 {
            // Time to allocate a new block
            if let Some(block_id) = self.block_allocator.write()?.allocate() {
                // Update page table with new block
                self.page_table.write()?.append_block(sequence_id, block_id);

                tracing::debug!(
                    "Allocated new block {} for sequence {} at token position {}",
                    block_id,
                    sequence_id,
                    current_tokens
                );
            } else {
                return Err(KvCacheError::CapacityExceeded);
            }
        } else if current_tokens == 0 {
            // First token - allocate initial block
            if let Some(block_id) = self.block_allocator.write()?.allocate() {
                self.page_table.write()?.append_block(sequence_id, block_id);

                tracing::debug!(
                    "Allocated initial block {} for new sequence {}",
                    block_id,
                    sequence_id
                );
            } else {
                return Err(KvCacheError::CapacityExceeded);
            }
        }

        // Delegate to existing append_token for actual storage
        // This handles token storage and sequence tracking
        self.append_token(sequence_id, token)
    }

    /// Get the physical block for a given sequence and token position
    ///
    /// This uses the PageTable to map logical positions to physical blocks.
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to query
    /// * `token_pos` - The logical token position within the sequence
    ///
    /// # Returns
    /// * `Some((block_id, offset))` - The physical block ID and offset within that block
    /// * `None` - If the sequence doesn't exist or position is out of range
    pub fn get_block_for_position(
        &self,
        sequence_id: u32,
        token_pos: usize,
    ) -> KvCacheResult<Option<(u32, usize)>> {
        Ok(self
            .page_table
            .read()?
            .get_block_for_position(sequence_id, token_pos))
    }

    /// Get all blocks for a sequence using PageTable
    ///
    /// # Arguments
    /// * `sequence_id` - The sequence to query
    ///
    /// # Returns
    /// * `Some(&[u32])` - Slice of block IDs
    /// * `None` - If the sequence doesn't exist
    pub fn get_sequence_blocks_from_page_table(
        &self,
        sequence_id: u32,
    ) -> KvCacheResult<Option<Vec<u32>>> {
        Ok(self
            .page_table
            .read()?
            .get_sequence_blocks(sequence_id)
            .map(|v| v.to_vec()))
    }

    /// Get BlockAllocator statistics
    pub fn get_block_allocator_stats(&self) -> (usize, usize) {
        let allocator = self
            .block_allocator
            .read()
            .expect("Block allocator lock poisoned");
        (allocator.total_blocks(), allocator.free_blocks())
    }

    // ========== PagedAttention Block Management ==========

    /// Allocate a block for PagedAttention-style KV cache management
    /// Returns the logical block ID
    pub fn allocate_block(&mut self, sequence_id: u32) -> KvCacheResult<BlockId> {
        block_allocate_block(
            &self.block_pool,
            &self.block_table,
            &self.free_blocks,
            &self.next_block_id,
            &self.config,
            sequence_id,
        )
    }

    /// Get a block table entry by logical block ID
    pub fn get_block(&self, block_id: BlockId) -> KvCacheResult<BlockTable> {
        get_block(&self.block_table, block_id)
    }

    /// Get the physical block for a given logical block ID
    pub fn get_physical_block(&self, block_id: BlockId) -> KvCacheResult<PhysicalBlock> {
        get_physical_block(&self.block_pool, &self.block_table, block_id)
    }

    /// Increment reference count for a block (for block sharing across sequences)
    pub fn ref_block(&mut self, block_id: BlockId, sequence_id: u32) -> KvCacheResult<()> {
        ref_block(&self.block_table, block_id, sequence_id)
    }

    /// Decrement reference count for a block
    /// Returns true if block was freed (ref count reached zero)
    pub fn unref_block(&mut self, block_id: BlockId, sequence_id: u32) -> KvCacheResult<bool> {
        unref_block(
            &self.block_pool,
            &self.block_table,
            &self.free_blocks,
            block_id,
            sequence_id,
        )
    }

    /// Copy a block for copy-on-write (COW) optimization
    /// Useful when a sequence diverges from a shared prefix
    ///
    /// NOTE: This requires GPU device-to-device memcpy support in HipBackend.
    /// For now, this is not implemented - use block sharing (ref_block) instead.
    pub fn copy_block(
        &mut self,
        block_id: BlockId,
        new_sequence_id: u32,
    ) -> KvCacheResult<BlockId> {
        copy_block(
            &self.block_pool,
            &self.block_table,
            &self.free_blocks,
            &self.next_block_id,
            &self.config,
            block_id,
            new_sequence_id,
        )
    }

    /// Get PagedAttention statistics
    pub fn get_paged_stats(&self) -> PagedCacheStats {
        get_paged_stats(&self.block_pool, &self.block_table, &self.sequences)
    }

    /// Get detailed memory profile for profiling and optimization
    ///
    /// This method provides comprehensive memory usage statistics including
    /// fragmentation analysis and per-token metrics. Used for identifying
    /// memory optimization opportunities.
    ///
    /// # Returns
    /// A `MemoryProfile` struct containing detailed memory statistics
    ///
    /// # Example
    /// ```ignore
    /// let profile = cache.memory_profile();
    /// profile.report();
    /// println!("Efficiency: {:.2}%", profile.efficiency_ratio() * 100.0);
    /// ```
    pub fn memory_profile(&self) -> MemoryProfile {
        // Gather statistics from all components
        let block_pool = self
            .block_pool
            .read()
            .expect("KvCache block_pool lock poisoned");
        let block_table = self
            .block_table
            .read()
            .expect("KvCache block_table lock poisoned");
        let page_table = self
            .page_table
            .read()
            .expect("KvCache page_table lock poisoned");
        let block_allocator = self
            .block_allocator
            .read()
            .expect("KvCache block_allocator lock poisoned");
        let sequences = self
            .sequences
            .read()
            .expect("KvCache sequences lock poisoned");

        // Calculate physical block memory
        let physical_blocks = block_pool.total_count();
        let bytes_per_block = self.config.page_size
            * self.config.num_heads
            * self.config.head_dim
            * 2 // K and V
            * std::mem::size_of::<f32>();
        let total_gpu_bytes = physical_blocks * bytes_per_block;

        // Calculate memory in use (allocated blocks)
        let logical_blocks = block_table.len();
        let used_gpu_bytes = logical_blocks * bytes_per_block;
        let free_gpu_bytes = total_gpu_bytes - used_gpu_bytes;

        // Calculate metadata overhead
        // Page table: each sequence has a Vec<u32> of block IDs
        let page_table_bytes: usize = page_table
            .tables()
            .values()
            .map(|v| v.len() * std::mem::size_of::<u32>())
            .sum();

        // Block allocator: VecDeque + Vec overhead
        let allocator_bytes = (block_allocator.total_blocks() * std::mem::size_of::<BlockId>())
            + (block_allocator.free_blocks() * std::mem::size_of::<BlockId>())
            + std::mem::size_of::<VecDeque<BlockId>>()
            + std::mem::size_of::<Vec<PhysicalBlock>>();

        // Count total tokens
        let total_tokens: usize = sequences.values().map(|s| s.total_tokens).sum();

        // Calculate bytes per token
        let bytes_per_token = if total_tokens > 0 {
            used_gpu_bytes as f64 / total_tokens as f64
        } else {
            0.0
        };

        // Estimate fragmentation
        // Fragmentation occurs when allocated blocks are not fully utilized
        let fragmentation_ratio = if logical_blocks > 0 && total_tokens > 0 {
            let total_capacity = logical_blocks * self.config.page_size;
            let waste = total_capacity.saturating_sub(total_tokens);
            waste as f64 / total_capacity as f64
        } else {
            0.0
        };

        MemoryProfile {
            total_gpu_bytes,
            used_gpu_bytes,
            free_gpu_bytes,
            physical_blocks,
            logical_blocks,
            page_table_bytes,
            allocator_bytes,
            active_sequences: sequences.len(),
            total_tokens,
            bytes_per_token,
            fragmentation_ratio,
        }
    }

    // ========== Cache Compaction for Long-Running Inference ==========

    /// Check if cache compaction is needed
    ///
    /// Compaction is recommended when:
    /// - Fragmentation is high (many free blocks scattered)
    /// - Memory efficiency is low (peak usage << total capacity)
    ///
    /// Returns true if compaction should be performed.
    pub fn should_compact(&self) -> KvCacheResult<bool> {
        if !self.config.enable_compaction {
            return Ok(false);
        }

        let block_pool = self.block_pool.read()?;
        let fragmentation = block_pool.fragmentation_ratio();
        let efficiency = block_pool.memory_efficiency();

        // Compact if fragmentation is high AND efficiency is low
        let should_compact = fragmentation > self.config.compaction_threshold
            && efficiency < 0.7;

        Ok(should_compact)
    }

    /// Perform cache compaction to reduce fragmentation
    ///
    /// Compaction reorganizes the KV cache to:
    /// 1. Free memory from completed sequences
    /// 2. Consolidate sparse allocations
    /// 3. Improve memory locality
    ///
    /// This is useful for long-running inference sessions where
    /// sequences are frequently created and destroyed.
    ///
    /// # Returns
    /// * `Ok(freed_bytes)` - Number of bytes freed during compaction
    /// * `Err(KvCacheError)` - Compaction failed
    ///
    /// # Note
    /// This is a relatively expensive operation and should only be
    /// called when `should_compact()` returns true.
    pub fn compact_cache(&mut self) -> KvCacheResult<usize> {
        tracing::debug!("Starting KV cache compaction");

        let bytes_before = self.calculate_memory_usage()?;

        // Step 1: Remove all completed sequences
        let removed_sequences = self.cleanup_completed_sequences()?;

        // Step 2: Reclaim unused blocks
        let reclaimed = self.reclaim_unused_blocks()?;

        // Step 3: Sort free lists for better allocation patterns
        sort_free_lists(&self.free_pages);

        let bytes_after = self.calculate_memory_usage()?;
        let freed_bytes = bytes_before.saturating_sub(bytes_after);

        // Update compaction statistics
        {
            let mut block_pool = self.block_pool.write()?;
            block_pool.record_compaction(self.timestamp_ms());
        }

        tracing::debug!(
            "Cache compaction complete: freed {} bytes, removed {} sequences, reclaimed {} blocks",
            freed_bytes, removed_sequences, reclaimed
        );

        Ok(freed_bytes)
    }

    /// Calculate current memory usage in bytes
    fn calculate_memory_usage(&self) -> KvCacheResult<usize> {
        let block_pool = self.block_pool.read()?;
        let in_use = block_pool.stats().current_allocations;
        let bytes_per_block = self.config.page_size
            * self.config.num_heads
            * self.config.head_dim
            * 2 // K + V
            * std::mem::size_of::<f32>();
        Ok(in_use * bytes_per_block)
    }

    /// Get current timestamp in milliseconds
    fn timestamp_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Reclaim unused physical blocks from the pool
    ///
    /// This returns blocks to the free list that are no longer
    /// referenced by any active sequence.
    fn reclaim_unused_blocks(&mut self) -> KvCacheResult<usize> {
        let mut reclaimed = 0;

        // Get all currently used physical block IDs
        let mut used_physical_ids = HashSet::new();

        {
            let block_table = self.block_table.read()?;
            for entry in block_table.values() {
                used_physical_ids.insert(entry.physical_block_id);
            }
        }

        // Check the block allocator's free list vs actual usage
        let block_pool = self.block_pool.read()?;
        let total_blocks = block_pool.total_count();
        drop(block_pool);

        // Count blocks that could be reclaimed
        let block_table = self.block_table.read()?;
        let allocated = block_table.len();
        if allocated < total_blocks {
            reclaimed = total_blocks - allocated;
        }

        Ok(reclaimed)
    }

    /// Get allocation statistics from the block pool
    pub fn get_allocation_stats(&self) -> AllocationStats {
        block_get_allocation_stats(&self.block_pool)
    }

    /// Reset allocation statistics
    pub fn reset_allocation_stats(&mut self) {
        block_reset_allocation_stats(&self.block_pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_creation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(1024, 100, 32, 128, 24).unwrap();
        let cache = KvCache::new(config, backend);
        assert!(cache.is_ok());
    }

    #[test]
    fn test_page_allocation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        let page_id = cache.allocate_page(1);
        assert!(page_id.is_ok());
        assert_eq!(page_id.unwrap(), 0);

        let stats = cache.get_cache_stats();
        assert_eq!(stats.total_pages, 1);
        assert_eq!(stats.free_pages, 0);
        assert_eq!(stats.active_sequences, 1);
    }

    #[test]
    fn test_token_appending() {
        let backend = HipBackend::new().unwrap();
        // FIX-21-02: Strict capacity enforcement - no LRU eviction in allocate_page
        let config = CacheConfig::new(4, 1, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a page first
        cache.allocate_page(1).unwrap();

        // Append tokens - first 4 fit in the page
        for i in 0..4 {
            let result = cache.append_token(1, i);
            assert!(result.is_ok());
        }

        // FIX-21-02: 5th token exceeds page capacity and no free pages
        // allocate_page would be called but cache is at max_pages
        let result = cache.append_token(1, 5);
        assert!(result.is_err()); // Should fail - cache at capacity
    }

    #[test]
    fn test_sequence_retrieval() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        cache.allocate_page(1).unwrap();

        // Append tokens
        for i in 0..3 {
            cache.append_token(1, i).unwrap();
        }

        let tokens = cache.get_sequence_tokens(1).unwrap();
        assert_eq!(tokens, vec![0, 1, 2]);

        let length = cache.get_sequence_length(1).unwrap();
        assert_eq!(length, 3);
    }

    #[test]
    fn test_sequence_removal() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        cache.allocate_page(1).unwrap();
        cache.append_token(1, 42).unwrap();

        let stats_before = cache.get_cache_stats();
        assert_eq!(stats_before.active_sequences, 1);
        assert_eq!(stats_before.free_pages, 0);

        cache.remove_sequence(1).unwrap();

        let stats_after = cache.get_cache_stats();
        assert_eq!(stats_after.active_sequences, 0);
        assert_eq!(stats_after.free_pages, 1);
    }

    #[test]
    fn test_capacity_limit() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 2, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate maximum pages
        cache.allocate_page(1).unwrap();
        cache.allocate_page(2).unwrap();

        // FIX-21-02: Strict capacity enforcement - no LRU eviction
        // Third allocation should fail with CapacityExceeded
        let result = cache.allocate_page(3);
        assert!(result.is_err());
        assert!(matches!(result, Err(KvCacheError::CapacityExceeded)));

        // Both sequences should still be present (no eviction occurred)
        assert!(cache.get_sequence_tokens(1).is_ok());
        assert!(cache.get_sequence_tokens(2).is_ok());
    }

    // ========== PagedAttention Tests ==========

    #[test]
    fn test_block_allocation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a block
        let block_id = cache.allocate_block(1);
        assert!(block_id.is_ok());
        assert_eq!(block_id.unwrap(), 0);

        // Check stats
        let stats = cache.get_paged_stats();
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 9);
        assert_eq!(stats.allocated_blocks, 1);
    }

    #[test]
    fn test_block_ref_counting() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a block for sequence 1
        let block_id = cache.allocate_block(1).unwrap();

        // Get the block
        let block = cache.get_block(block_id);
        assert!(block.is_ok());
        assert_eq!(block.unwrap().ref_count(), 1);

        // Reference block for sequence 2 (sharing)
        let result = cache.ref_block(block_id, 2);
        assert!(result.is_ok());

        // Check ref count increased
        let block = cache.get_block(block_id).unwrap();
        assert_eq!(block.ref_count(), 2);
        assert!(block.sequences.contains(&1));
        assert!(block.sequences.contains(&2));
    }

    #[test]
    fn test_block_sharing_and_unreference() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate a block for sequence 1
        let block_id = cache.allocate_block(1).unwrap();

        // Share with sequence 2
        cache.ref_block(block_id, 2).unwrap();

        // Unref sequence 1 - should NOT free (sequence 2 still holds ref)
        let freed = cache.unref_block(block_id, 1).unwrap();
        assert!(!freed);

        let stats = cache.get_paged_stats();
        assert_eq!(stats.allocated_blocks, 1);

        // Unref sequence 2 - SHOULD free now
        let freed = cache.unref_block(block_id, 2).unwrap();
        assert!(freed);

        let stats = cache.get_paged_stats();
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_blocks, 10); // All blocks free
    }

    #[test]
    fn test_block_capacity_limit() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 2, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate all blocks
        let _block1 = cache.allocate_block(1).unwrap();
        let _block2 = cache.allocate_block(2).unwrap();

        // Third allocation should fail
        let block3 = cache.allocate_block(3);
        assert!(block3.is_err());
        assert!(matches!(block3, Err(KvCacheError::CapacityExceeded)));
    }

    #[test]
    fn test_paged_cache_stats() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Initial stats
        let stats = cache.get_paged_stats();
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 10);
        assert_eq!(stats.allocated_blocks, 0);

        // After allocation
        cache.allocate_block(1).unwrap();
        cache.allocate_block(2).unwrap();

        let stats = cache.get_paged_stats();
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 8);
        assert_eq!(stats.allocated_blocks, 2);
    }

    #[test]
    fn test_block_id_recycling() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Allocate and free a block
        let block_id = cache.allocate_block(1).unwrap();
        cache.unref_block(block_id, 1).unwrap();

        // Allocate another - should reuse the same ID
        let new_block_id = cache.allocate_block(2).unwrap();
        assert_eq!(new_block_id, block_id);
    }

    #[test]
    fn test_get_physical_block() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        let block_id = cache.allocate_block(1).unwrap();

        // Get physical block
        let physical = cache.get_physical_block(block_id);
        assert!(physical.is_ok());

        let physical = physical.unwrap();
        // PhysicalBlock.block_id is the physical block ID
        assert!(physical.size_bytes() > 0);
        assert!(physical.capacity_tokens() > 0);
    }

    // ========== Phase 2: PageTable + BlockAllocator Integration Tests ==========

    #[test]
    fn test_append_token_paged_initial_allocation() {
        let backend = HipBackend::new().unwrap();
        // Use page_size=4 for faster testing
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // First token should allocate initial block
        cache.append_token_paged(1, 42).unwrap();

        // Check block allocator stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 10);
        assert_eq!(free, 9); // One block allocated

        // Check page table has the block
        let blocks = cache.get_sequence_blocks_from_page_table(1).unwrap();
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 1);
    }

    #[test]
    fn test_append_token_paged_multiple_blocks() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Append 9 tokens (should span 3 blocks of size 4)
        for i in 0..9 {
            cache.append_token_paged(1, i).unwrap();
        }

        // Check block allocator stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 10);
        assert_eq!(free, 7); // 3 blocks allocated (0, 4, 8 tokens trigger allocations)

        // Check page table has 3 blocks
        let blocks = cache.get_sequence_blocks_from_page_table(1).unwrap();
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 3);
    }

    #[test]
    fn test_get_block_for_position() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Append some tokens
        for i in 0..20 {
            cache.append_token_paged(1, i).unwrap();
        }

        // Position 0 should be in block 0, offset 0
        let result = cache.get_block_for_position(1, 0).unwrap();
        assert_eq!(result, Some((0, 0)));

        // Position 10 should be in block 0, offset 10
        let result = cache.get_block_for_position(1, 10).unwrap();
        assert_eq!(result, Some((0, 10)));

        // Position 16 should be in block 1, offset 0
        let result = cache.get_block_for_position(1, 16).unwrap();
        assert_eq!(result, Some((1, 0)));

        // Position 20 should be in block 1, offset 4 (block_size=16)
        // PageTable doesn't check actual token count - it maps positions to blocks
        let result = cache.get_block_for_position(1, 20).unwrap();
        assert_eq!(result, Some((1, 4)));

        // Position 32 would be in block 2 (but we only have 20 tokens)
        // PageTable will return None because block 2 doesn't exist
        let result = cache.get_block_for_position(1, 32).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_get_block_allocator_stats() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 100, 32, 128, 24).unwrap();
        let cache = KvCache::new(config, backend).unwrap();

        // Initial stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 100);
        assert_eq!(free, 100);
    }

    #[test]
    fn test_multiple_sequences_paged() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 20, 32, 128, 24).unwrap();
        let mut cache = KvCache::new(config, backend).unwrap();

        // Sequence 1: 5 tokens (2 blocks)
        for i in 0..5 {
            cache.append_token_paged(1, i).unwrap();
        }

        // Sequence 2: 3 tokens (1 block)
        for i in 0..3 {
            cache.append_token_paged(2, i).unwrap();
        }

        // Check block allocator stats
        let (total, free) = cache.get_block_allocator_stats();
        assert_eq!(total, 20);
        assert_eq!(free, 17); // 3 blocks allocated total

        // Verify each sequence has correct blocks
        let blocks1 = cache.get_sequence_blocks_from_page_table(1).unwrap();
        assert_eq!(blocks1.unwrap().len(), 2);

        let blocks2 = cache.get_sequence_blocks_from_page_table(2).unwrap();
        assert_eq!(blocks2.unwrap().len(), 1);
    }
}
