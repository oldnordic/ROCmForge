//! Phase 5: PagedAttention Integration Tests (TDD)
//!
//! These tests verify that the ExecutionPlan properly integrates with
//! PagedAttention for efficient non-contiguous KV cache access.

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use crate::backend::hip_backend::{DeviceTensor, HipBackend, HipError};
    use crate::kv_cache::kv_cache::{CacheConfig, KvCache};
    use crate::loader::TensorShape;
    use std::sync::Arc;

    /// Helper function to create a test backend
    fn create_test_backend() -> Arc<HipBackend> {
        HipBackend::new().expect("Failed to create HIP backend")
    }

    /// Helper function to create a test KV cache with paged blocks
    fn create_paged_kv_cache() -> (Arc<HipBackend>, KvCache, u32) {
        let backend = create_test_backend();
        let config = CacheConfig::new(16, 10, 32, 128, 24).expect("Invalid config");
        let mut cache = KvCache::new(config, Arc::clone(&backend)).expect("Failed to create cache");

        let sequence_id = 1u32;

        // Append tokens using paged allocation (allocates blocks)
        for i in 0..20 {
            cache
                .append_token_paged(sequence_id, i)
                .expect("Failed to append token");
        }

        (backend, cache, sequence_id)
    }

    // Test 1: Verify PageTable has blocks after paged token appends
    #[test]
    fn test_page_table_has_blocks_after_paged_append() {
        let (_backend, cache, sequence_id) = create_paged_kv_cache();

        // Verify PageTable has blocks for this sequence
        let blocks = cache
            .get_sequence_blocks_from_page_table(sequence_id)
            .expect("Failed to get blocks from page table");

        assert!(
            blocks.is_some(),
            "PageTable should have blocks for sequence with paged tokens"
        );

        let blocks = blocks.unwrap();
        assert!(
            !blocks.is_empty(),
            "PageTable should have at least one block"
        );

        println!(
            "PageTable has {} blocks for sequence {}",
            blocks.len(),
            sequence_id
        );
    }

    // Test 2: Verify block allocation from BlockAllocator
    #[test]
    fn test_block_allocator_allocates_blocks() {
        let (_backend, cache, _sequence_id) = create_paged_kv_cache();

        let (total, free) = cache.get_block_allocator_stats();

        assert!(total > 0, "BlockAllocator should have total blocks");

        assert!(
            free < total,
            "BlockAllocator should have allocated some blocks (free < total)"
        );

        println!("BlockAllocator: {} total, {} free", total, free);
    }

    // Test 3: Verify get_block_for_position returns valid mappings
    #[test]
    fn test_get_block_for_position() {
        let (_backend, cache, sequence_id) = create_paged_kv_cache();

        // Test positions 0-19 (we appended 20 tokens with block_size=16)
        for pos in 0..20 {
            let result = cache.get_block_for_position(sequence_id, pos);

            assert!(
                result.is_ok(),
                "get_block_for_position should succeed for position {}: {:?}",
                pos,
                result
            );

            let block_info = result.unwrap();
            assert!(
                block_info.is_some(),
                "Position {} should map to a block",
                pos
            );

            let (block_id, offset) = block_info.unwrap();
            println!("Position {} -> block {} offset {}", pos, block_id, offset);

            // With block_size=16:
            // Positions 0-15 -> block 0
            // Positions 16-19 -> block 1
            let expected_block = if pos < 16 { 0 } else { 1 };
            let expected_offset = if pos < 16 { pos } else { pos - 16 };

            assert_eq!(
                block_id, expected_block,
                "Position {} should map to block {}",
                pos, expected_block
            );
            assert_eq!(
                offset, expected_offset,
                "Position {} should have offset {}",
                pos, expected_offset
            );
        }
    }

    // Test 4: Verify multiple blocks are allocated for long sequences
    #[test]
    fn test_multiple_blocks_for_long_sequence() {
        let backend = create_test_backend();
        let config = CacheConfig::new(4, 10, 32, 128, 24).expect("Invalid config");
        let mut cache = KvCache::new(config, Arc::clone(&backend)).expect("Failed to create cache");

        let sequence_id = 1u32;

        // Append 12 tokens with block_size=4 (should allocate 3 blocks)
        for i in 0..12 {
            cache
                .append_token_paged(sequence_id, i)
                .expect("Failed to append token");
        }

        let blocks = cache
            .get_sequence_blocks_from_page_table(sequence_id)
            .expect("Failed to get blocks")
            .expect("Should have blocks");

        assert_eq!(
            blocks.len(),
            3,
            "Sequence of 12 tokens with block_size=4 should have 3 blocks"
        );

        println!("Sequence has {} blocks: {:?}", blocks.len(), blocks);
    }

    // Test 5: Verify block ID mappings are consistent
    #[test]
    fn test_block_id_mappings_consistent() {
        let (_backend, cache, sequence_id) = create_paged_kv_cache();

        let blocks = cache
            .get_sequence_blocks_from_page_table(sequence_id)
            .expect("Failed to get blocks")
            .expect("Should have blocks");

        // Verify block IDs are sequential
        for (i, &block_id) in blocks.iter().enumerate() {
            assert_eq!(block_id as usize, i, "Block {} should have ID {}", i, i);
        }

        println!("Block IDs are consistently numbered: {:?}", blocks);
    }

    // Test 6: Verify fallback when PageTable is empty
    #[test]
    fn test_fallback_when_page_table_empty() {
        let backend = create_test_backend();
        let config = CacheConfig::new(16, 10, 32, 128, 24).expect("Invalid config");
        let cache = KvCache::new(config, Arc::clone(&backend)).expect("Failed to create cache");

        // Create a sequence without using paged append
        let sequence_id = 999u32; // Non-existent sequence

        let blocks = cache
            .get_sequence_blocks_from_page_table(sequence_id)
            .expect("Failed to get blocks");

        assert!(
            blocks.is_none(),
            "Non-existent sequence should have no blocks in PageTable"
        );

        println!("Non-existent sequence has no blocks (correct for fallback)");
    }

    // Test 7: Verify block reference counting works
    #[test]
    fn test_block_reference_counting() {
        let backend = create_test_backend();
        let config = CacheConfig::new(16, 10, 32, 128, 24).expect("Invalid config");
        let mut cache = KvCache::new(config, Arc::clone(&backend)).expect("Failed to create cache");

        // Allocate a block
        let block_id = cache.allocate_block(1).expect("Failed to allocate block");

        // Check initial ref count
        let block = cache.get_block(block_id).expect("Failed to get block");
        assert_eq!(
            block.ref_count(),
            1,
            "Newly allocated block should have ref_count=1"
        );

        // Add another sequence to the block (sharing)
        cache.ref_block(block_id, 2).expect("Failed to ref block");

        let block = cache.get_block(block_id).expect("Failed to get block");
        assert_eq!(
            block.ref_count(),
            2,
            "Block should have ref_count=2 after sharing"
        );

        // Unref first sequence - should NOT free
        let freed = cache.unref_block(block_id, 1).expect("Failed to unref");
        assert!(!freed, "Block should not be freed (ref_count=1)");

        // Unref second sequence - SHOULD free
        let freed = cache.unref_block(block_id, 2).expect("Failed to unref");
        assert!(freed, "Block should be freed (ref_count=0)");

        println!("Block reference counting works correctly");
    }
}
