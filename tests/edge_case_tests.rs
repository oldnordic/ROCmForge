//! Edge Case Tests for ROCmForge
//!
//! This module contains tests for edge cases and boundary conditions
//! that may not be covered by the main test suite.
//!
//! Following TDD principles, these tests document expected behavior
//! for edge cases even if some may currently fail.

use rocmforge::backend::gpu_test_common::GPU_FIXTURE;
use serial_test::serial;
use rocmforge::kv_cache::{CacheConfig, KvCache, KvCacheError};
use anyhow::Context;

// ============================================================================
// KV Cache Edge Cases
// ============================================================================

#[test]
#[serial]
fn test_kv_cache_empty_initial_state() -> anyhow::Result<()> {
    // Test KV cache with no prior cached keys/values
    // Edge case: Cache should be properly initialized with zero state
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend().clone();
    let config = CacheConfig::new(1024, 100, 32, 128, 24).context("TODO: add error context")?;
    let cache = KvCache::new(config, backend).context("TODO: add error context")?;

    // Cache should be empty initially
    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 0, "Cache should start with no pages");
    assert_eq!(
        stats.active_sequences, 0,
        "Cache should start with no active sequences"
    );
    assert_eq!(stats.total_tokens, 0, "Cache should start with no tokens");
    Ok(())
}

#[test]
#[serial]
fn test_kv_cache_single_token() -> anyhow::Result<()> {
    // Test KV cache with single token (minimum meaningful operation)
    // Edge case: Smallest non-zero allocation
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend().clone();
    let config = CacheConfig::new(1024, 10, 4, 32, 2).context("TODO: add error context")?;
    let mut cache = KvCache::new(config, backend).context("TODO: add error context")?;

    // Allocate page for single token
    let page_id = cache.allocate_page(1);
    assert!(page_id.is_ok(), "Should allocate page for single token");
    assert_eq!(page_id.context("TODO: add error context")?, 0, "First page should have ID 0");

    let stats = cache.get_cache_stats();
    assert_eq!(
        stats.total_pages, 1,
        "Should have 1 page after single allocation"
    );
    assert_eq!(stats.active_sequences, 1, "Should have 1 active sequence");
    Ok(())
}

#[test]
#[serial]
fn test_kv_cache_eviction_at_capacity() -> anyhow::Result<()> {
    // Test KV cache behavior when reaching max capacity
    // Edge case: Boundary between acceptable and over-capacity
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend().clone();
    let page_size = 4;
    let max_pages = 3; // Small cache for testing
    let config = CacheConfig::new(page_size, max_pages, 4, 32, 2).context("TODO: add error context")?;
    let mut cache = KvCache::new(config, backend).context("TODO: add error context")?;

    // Fill cache to capacity
    for seq_id in 1..=max_pages {
        let page_id = cache.allocate_page(seq_id as u32);
        assert!(
            page_id.is_ok(),
            "Should allocate page for sequence {}",
            seq_id
        );
    }

    let stats = cache.get_cache_stats();
    assert_eq!(
        stats.total_pages, max_pages as usize,
        "Cache should be at capacity"
    );
    assert_eq!(
        stats.active_sequences, max_pages,
        "Should have {} active sequences",
        max_pages
    );

    // Try to allocate one more page - should fail with clear error
    let result = cache.allocate_page((max_pages + 1) as u32);
    assert!(result.is_err(), "Should fail when cache is full");

    match result {
        Err(KvCacheError::CapacityExceeded) => {
            // Expected error type
        }
        Err(e) => {
            panic!("Expected CapacityExceeded error, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Should not succeed when cache is full");
        }
    }
    Ok(())
}

#[test]
#[serial]
fn test_kv_cache_cross_sequence_isolation() -> anyhow::Result<()> {
    // Test that different sequences are properly isolated in the cache
    // Edge case: Multiple concurrent sequences should not interfere
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend().clone();
    let config = CacheConfig::new(1024, 100, 4, 32, 2).context("TODO: add error context")?;
    let mut cache = KvCache::new(config, backend).context("TODO: add error context")?;

    // Allocate pages for different sequences
    let page1 = cache.allocate_page(1).context("TODO: add error context")?;
    let page2 = cache.allocate_page(2).context("TODO: add error context")?;
    let page3 = cache.allocate_page(1).context("TODO: add error context")?; // Another page for sequence 1

    // Verify page IDs are unique
    assert_ne!(
        page1, page2,
        "Different sequences should get different pages"
    );
    assert_ne!(page2, page3, "Each allocation should get unique page ID");

    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 3, "Should have 3 total pages");
    assert_eq!(stats.active_sequences, 2, "Should have 2 active sequences");
    Ok(())
}

#[test]
#[serial]
fn test_kv_cache_sequence_reuse() -> anyhow::Result<()> {
    // Test that allocating for same sequence increases page count
    // Edge case: Same sequence ID with multiple allocations
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend().clone();
    let config = CacheConfig::new(1024, 100, 4, 32, 2).context("TODO: add error context")?;
    let mut cache = KvCache::new(config, backend).context("TODO: add error context")?;

    // Allocate multiple pages for same sequence
    let page1 = cache.allocate_page(1).context("TODO: add error context")?;
    let page2 = cache.allocate_page(1).context("TODO: add error context")?;
    let page3 = cache.allocate_page(1).context("TODO: add error context")?;

    // Verify pages are sequential
    assert_eq!(page1, 0, "First page should be 0");
    assert_eq!(page2, 1, "Second page should be 1");
    assert_eq!(page3, 2, "Third page should be 2");

    let stats = cache.get_cache_stats();
    assert_eq!(stats.total_pages, 3, "Should have 3 pages");
    assert_eq!(
        stats.active_sequences, 1,
        "Should still have 1 active sequence"
    );
    Ok(())
}

// ============================================================================
// Configuration Validation Edge Cases
// ============================================================================

#[test]
#[serial]
fn test_cache_config_zero_page_size() {
    // Test edge case: page_size = 0 (invalid)
    let config = CacheConfig::new(0, 100, 32, 128, 24);
    assert!(config.is_err(), "Zero page size should be invalid");

    match config {
        Err(KvCacheError::InvalidConfiguration) => {
            // Expected
        }
        Err(e) => {
            panic!("Expected InvalidConfiguration, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Should not accept zero page size");
        }
    }
}

#[test]
#[serial]
fn test_cache_config_zero_max_pages() {
    // Test edge case: max_pages = 0 (invalid)
    let config = CacheConfig::new(1024, 0, 32, 128, 24);
    assert!(config.is_err(), "Zero max pages should be invalid");

    match config {
        Err(KvCacheError::InvalidConfiguration) => {
            // Expected
        }
        Err(e) => {
            panic!("Expected InvalidConfiguration, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Should not accept zero max pages");
        }
    }
}

#[test]
#[serial]
fn test_cache_config_zero_heads() {
    // Test edge case: num_heads = 0 (invalid)
    let config = CacheConfig::new(1024, 100, 0, 128, 24);
    assert!(config.is_err(), "Zero heads should be invalid");

    match config {
        Err(KvCacheError::InvalidConfiguration) => {
            // Expected
        }
        Err(e) => {
            panic!("Expected InvalidConfiguration, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Should not accept zero heads");
        }
    }
}

#[test]
#[serial]
fn test_cache_config_zero_head_dim() {
    // Test edge case: head_dim = 0 (invalid)
    let config = CacheConfig::new(1024, 100, 32, 0, 24);
    assert!(config.is_err(), "Zero head dimension should be invalid");

    match config {
        Err(KvCacheError::InvalidConfiguration) => {
            // Expected
        }
        Err(e) => {
            panic!("Expected InvalidConfiguration, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Should not accept zero head dimension");
        }
    }
}

#[test]
#[serial]
fn test_cache_config_zero_layers() {
    // Test edge case: num_layers = 0 (invalid)
    let config = CacheConfig::new(1024, 100, 32, 128, 0);
    assert!(config.is_err(), "Zero layers should be invalid");

    match config {
        Err(KvCacheError::InvalidConfiguration) => {
            // Expected
        }
        Err(e) => {
            panic!("Expected InvalidConfiguration, got: {:?}", e);
        }
        Ok(_) => {
            panic!("Should not accept zero layers");
        }
    }
}

#[test]
#[serial]
fn test_cache_config_minimum_valid_values() -> anyhow::Result<()> {
    // Test edge case: All minimum valid values (1)
    // This tests the lower boundary of valid configurations
    let config = CacheConfig::new(1, 1, 1, 1, 1);
    assert!(
        config.is_ok(),
        "Minimum valid configuration should be accepted"
    );

    let config = config.context("TODO: add error context")?;
    assert_eq!(config.page_size, 1);
    assert_eq!(config.max_pages, 1);
    assert_eq!(config.num_heads, 1);
    assert_eq!(config.head_dim, 1);
    assert_eq!(config.num_layers, 1);
    Ok(())
}

#[test]
#[serial]
fn test_cache_config_large_values() -> anyhow::Result<()> {
    // Test edge case: Large but still valid values
    // This tests the upper boundary before resource limits
    let config = CacheConfig::new(65536, 10000, 128, 256, 100);
    assert!(config.is_ok(), "Large configuration should be accepted");

    let config = config.context("TODO: add error context")?;
    assert_eq!(config.page_size, 65536);
    assert_eq!(config.max_pages, 10000);
    assert_eq!(config.num_heads, 128);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.num_layers, 100);
    Ok(())
}
