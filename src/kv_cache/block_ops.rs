//! PagedAttention block management operations
//!
//! This module contains operations for allocating, referencing, and managing
//! physical blocks for PagedAttention-style KV cache management.

use super::blocks::{PhysicalBlock, PhysicalBlockPool};
use super::config::CacheConfig;
use super::types::{BlockId, BlockTable, KvCacheError, KvCacheResult, PagedCacheStats};
use std::collections::HashMap;
use std::sync::RwLock;

/// Allocate a block for PagedAttention-style KV cache management
///
/// Returns the logical block ID.
pub fn allocate_block(
    block_pool: &RwLock<PhysicalBlockPool>,
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    free_blocks: &RwLock<Vec<BlockId>>,
    next_block_id: &RwLock<BlockId>,
    _config: &CacheConfig,
    sequence_id: u32,
) -> KvCacheResult<BlockId> {
    // Allocate physical block from pool
    let physical_block_id = block_pool
        .write()?
        .allocate()
        .ok_or(KvCacheError::CapacityExceeded)?;

    // Get or create logical block ID
    let logical_block_id = if let Some(free_id) = free_blocks.write()?.pop() {
        free_id
    } else {
        let mut next_id = next_block_id.write()?;
        let id = *next_id;
        *next_id += 1;
        id
    };

    // Create block table entry
    let mut table_entry = BlockTable::new(logical_block_id, physical_block_id);
    table_entry.add_sequence(sequence_id);

    // Register in block table
    block_table.write()?.insert(logical_block_id, table_entry);

    Ok(logical_block_id)
}

/// Get a block table entry by logical block ID
pub fn get_block(
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    block_id: BlockId,
) -> KvCacheResult<BlockTable> {
    let block_table_guard = block_table.read()?;
    block_table_guard
        .get(&block_id)
        .cloned()
        .ok_or(KvCacheError::PageNotFound(block_id))
}

/// Get the physical block for a given logical block ID
pub fn get_physical_block(
    block_pool: &RwLock<PhysicalBlockPool>,
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    block_id: BlockId,
) -> KvCacheResult<PhysicalBlock> {
    let table_entry = get_block(block_table, block_id)?;
    let block_pool_guard = block_pool.read()?;
    block_pool_guard
        .get_block(table_entry.physical_block_id)
        .cloned()
        .ok_or(KvCacheError::PageNotFound(block_id))
}

/// Increment reference count for a block (for block sharing across sequences)
pub fn ref_block(
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    block_id: BlockId,
    sequence_id: u32,
) -> KvCacheResult<()> {
    let mut block_table_guard = block_table.write()?;
    let block = block_table_guard
        .get_mut(&block_id)
        .ok_or(KvCacheError::PageNotFound(block_id))?;

    block.add_sequence(sequence_id);
    block.incr_ref();

    Ok(())
}

/// Decrement reference count for a block
///
/// Returns true if block was freed (ref count reached zero).
pub fn unref_block(
    block_pool: &RwLock<PhysicalBlockPool>,
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    free_blocks: &RwLock<Vec<BlockId>>,
    block_id: BlockId,
    sequence_id: u32,
) -> KvCacheResult<bool> {
    let mut physical_block_id = None;
    let mut should_free = false;

    {
        let mut block_table_guard = block_table.write()?;
        let block = block_table_guard
            .get_mut(&block_id)
            .ok_or(KvCacheError::PageNotFound(block_id))?;

        block.remove_sequence(sequence_id);
        let prev_count = block.decr_ref();

        if prev_count == 0 {
            should_free = true;
            physical_block_id = Some(block.physical_block_id);
        }
    }

    if should_free {
        // Remove from block table
        block_table.write()?.remove(&block_id);
        // Return physical block to pool
        if let Some(phys_id) = physical_block_id {
            block_pool.write()?.deallocate(phys_id);
        }
        // Recycle logical block ID
        free_blocks.write()?.push(block_id);

        Ok(true)
    } else {
        Ok(false)
    }
}

/// Copy a block for copy-on-write (COW) optimization
///
/// Useful when a sequence diverges from a shared prefix.
///
/// NOTE: This requires GPU device-to-device memcpy support in HipBackend.
/// For now, this is not implemented - use block sharing (ref_block) instead.
pub fn copy_block(
    block_pool: &RwLock<PhysicalBlockPool>,
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    free_blocks: &RwLock<Vec<BlockId>>,
    next_block_id: &RwLock<BlockId>,
    config: &CacheConfig,
    _block_id: BlockId,
    new_sequence_id: u32,
) -> KvCacheResult<BlockId> {
    // COW block copying requires HipBackend::memcpy_device_to_device()
    // which is not yet implemented. For now, allocate a fresh block.
    // The caller should use ref_block() for sharing instead.
    allocate_block(
        block_pool,
        block_table,
        free_blocks,
        next_block_id,
        config,
        new_sequence_id,
    )
}

/// Get PagedAttention statistics
pub fn get_paged_stats(
    block_pool: &RwLock<PhysicalBlockPool>,
    block_table: &RwLock<HashMap<BlockId, BlockTable>>,
    sequences: &RwLock<HashMap<u32, super::pages::SequenceCache>>,
) -> PagedCacheStats {
    // Safe: These locks should never be poisoned in normal operation
    let block_pool_guard = block_pool
        .read()
        .expect("KvCache block_pool lock poisoned");
    let block_table_guard = block_table
        .read()
        .expect("KvCache block_table lock poisoned");
    let sequences_guard = sequences
        .read()
        .expect("KvCache sequences lock poisoned");

    PagedCacheStats {
        total_blocks: block_pool_guard.total_count(),
        free_blocks: block_pool_guard.free_count(),
        allocated_blocks: block_table_guard.len(),
        active_sequences: sequences_guard.len(),
    }
}

/// Get allocation statistics from the block pool
pub fn get_allocation_stats(
    block_pool: &RwLock<PhysicalBlockPool>,
) -> super::blocks::AllocationStats {
    block_pool
        .read()
        .expect("Block pool lock poisoned")
        .stats()
        .clone()
}

/// Reset allocation statistics
pub fn reset_allocation_stats(block_pool: &RwLock<PhysicalBlockPool>) {
    block_pool
        .write()
        .expect("Block pool lock poisoned")
        .reset_stats();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::backend::hip_backend::HipBackend;

    #[test]
    fn test_block_allocation() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let block_pool = RwLock::new(PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        ).unwrap());

        let block_table = RwLock::new(HashMap::new());
        let free_blocks = RwLock::new(Vec::new());
        let next_block_id = RwLock::new(0);

        // Allocate a block
        let block_id = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            1,
        )
        .unwrap();

        assert_eq!(block_id, 0);

        // Check stats
        let stats = get_paged_stats(&block_pool, &block_table, &RwLock::new(std::collections::HashMap::new()));
        assert_eq!(stats.total_blocks, 10);
        assert_eq!(stats.free_blocks, 9);
        assert_eq!(stats.allocated_blocks, 1);
    }

    #[test]
    fn test_block_ref_counting() {
        let backend = Arc::new(HipBackend::new().unwrap());
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let block_pool = RwLock::new(PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        ).unwrap());

        let block_table = RwLock::new(HashMap::new());
        let free_blocks = RwLock::new(Vec::new());
        let next_block_id = RwLock::new(0);

        // Allocate a block for sequence 1
        let block_id = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            1,
        )
        .unwrap();

        // Get the block
        let block = get_block(&block_table, block_id).unwrap();
        assert_eq!(block.ref_count(), 1);

        // Reference block for sequence 2 (sharing)
        let result = ref_block(&block_table, block_id, 2);
        assert!(result.is_ok());

        // Check ref count increased
        let block = get_block(&block_table, block_id).unwrap();
        assert_eq!(block.ref_count(), 2);
        assert!(block.sequences.contains(&1));
        assert!(block.sequences.contains(&2));
    }

    #[test]
    fn test_block_sharing_and_unreference() {
        let backend = Arc::new(HipBackend::new().unwrap());
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let block_pool = RwLock::new(PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        ).unwrap());

        let block_table = RwLock::new(HashMap::new());
        let free_blocks = RwLock::new(Vec::new());
        let next_block_id = RwLock::new(0);

        // Allocate a block for sequence 1
        let block_id = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            1,
        )
        .unwrap();

        // Share with sequence 2
        ref_block(&block_table, block_id, 2).unwrap();

        // Unref sequence 1 - should NOT free (sequence 2 still holds ref)
        let freed = unref_block(&block_pool, &block_table, &free_blocks, block_id, 1).unwrap();
        assert!(!freed);

        let stats = get_paged_stats(&block_pool, &block_table, &RwLock::new(std::collections::HashMap::new()));
        assert_eq!(stats.allocated_blocks, 1);

        // Unref sequence 2 - SHOULD free now
        let freed = unref_block(&block_pool, &block_table, &free_blocks, block_id, 2).unwrap();
        assert!(freed);

        let stats = get_paged_stats(&block_pool, &block_table, &RwLock::new(std::collections::HashMap::new()));
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_blocks, 10); // All blocks free
    }

    #[test]
    fn test_block_capacity_limit() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(16, 2, 32, 128, 24).unwrap();
        let block_pool = RwLock::new(PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        ).unwrap());

        let block_table = RwLock::new(HashMap::new());
        let free_blocks = RwLock::new(Vec::new());
        let next_block_id = RwLock::new(0);

        // Allocate all blocks
        let _block1 = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            1,
        )
        .unwrap();
        let _block2 = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            2,
        )
        .unwrap();

        // Third allocation should fail
        let block3 = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            3,
        );
        assert!(block3.is_err());
        assert!(matches!(block3, Err(KvCacheError::CapacityExceeded)));
    }

    #[test]
    fn test_block_id_recycling() {
        let backend = Arc::new(HipBackend::new().unwrap());
        let config = CacheConfig::new(16, 10, 32, 128, 24).unwrap();
        let block_pool = RwLock::new(PhysicalBlockPool::new(
            config.max_pages,
            config.page_size,
            config.num_heads,
            config.head_dim,
            &backend,
        ).unwrap());

        let block_table = RwLock::new(HashMap::new());
        let free_blocks = RwLock::new(Vec::new());
        let next_block_id = RwLock::new(0);

        // Allocate and free a block
        let block_id = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            1,
        )
        .unwrap();
        unref_block(&block_pool, &block_table, &free_blocks, block_id, 1).unwrap();

        // Allocate another - should reuse the same ID
        let new_block_id = allocate_block(
            &block_pool,
            &block_table,
            &free_blocks,
            &next_block_id,
            &config,
            2,
        )
        .unwrap();
        assert_eq!(new_block_id, block_id);
    }
}
