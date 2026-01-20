//! KV cache operations - insertion, retrieval, and management
//!
//! This module contains the core cache operations for token insertion,
//! sequence retrieval, and basic cache management.

use super::config::CacheConfig;
use super::pages::{CachePage, SequenceCache};
use super::types::{CacheStats, KvCacheError, KvCacheResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::backend::HipBackend;

/// Allocate a new page for a sequence
///
/// This operation either reuses a free page or allocates a new one
/// if capacity is available.
pub fn allocate_page(
    pages: &RwLock<HashMap<u32, CachePage>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    free_pages: &RwLock<Vec<u32>>,
    next_page_id: &RwLock<u32>,
    config: &CacheConfig,
    backend: &Arc<HipBackend>,
    sequence_id: u32,
) -> KvCacheResult<u32> {
    // Try to reuse a free page first
    let page_id = if let Some(free_id) = free_pages.write()?.pop() {
        // Reuse existing free page
        free_id
    } else {
        // No free pages - check if we can allocate a new page
        let current_pages = pages.read()?.len();
        if current_pages >= config.max_pages {
            return Err(KvCacheError::CapacityExceeded);
        }
        // Allocate new page ID
        let mut next_id = next_page_id.write()?;
        let id = *next_id;
        *next_id += 1;
        id
    };

    let page = CachePage::new(page_id, sequence_id, backend, config)?;
    pages.write()?.insert(page_id, page);

    // Update sequence cache
    let mut sequences_guard = sequences.write()?;
    let sequence = sequences_guard
        .entry(sequence_id)
        .or_insert_with(|| SequenceCache::new(sequence_id));
    sequence.add_page(page_id);

    Ok(page_id)
}

/// Append a token to a sequence
///
/// This handles automatic page allocation when the current page is full.
pub fn append_token(
    pages: &RwLock<HashMap<u32, CachePage>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    free_pages: &RwLock<Vec<u32>>,
    next_page_id: &RwLock<u32>,
    config: &CacheConfig,
    backend: &Arc<HipBackend>,
    sequence_id: u32,
    token: u32,
) -> KvCacheResult<()> {
    // Check if sequence is completed before appending
    {
        let sequences_guard = sequences.read()?;
        if let Some(sequence) = sequences_guard.get(&sequence_id) {
            if sequence.is_completed {
                return Err(KvCacheError::InvalidSequenceId(sequence_id));
            }
        }
    }

    let last_page_id = {
        let sequences_guard = sequences.read()?;
        let sequence = sequences_guard
            .get(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
        sequence
            .get_last_page()
            .ok_or(KvCacheError::PageNotFound(sequence_id))?
    };

    let can_append = {
        let pages_guard = pages.read()?;
        let page = pages_guard
            .get(&last_page_id)
            .ok_or(KvCacheError::PageNotFound(last_page_id))?;
        page.can_append(token)
    };

    if can_append {
        {
            let mut pages_guard = pages.write()?;
            let page = pages_guard
                .get_mut(&last_page_id)
                .ok_or(KvCacheError::PageNotFound(last_page_id))?;
            page.append_token(token)?;
        }
        let mut sequences_guard = sequences.write()?;
        let sequence = sequences_guard
            .get_mut(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
        sequence.total_tokens += 1;
        sequence.update_access(); // Update access time on append
    } else {
        // Allocate new page
        let new_page_id = allocate_page(
            pages,
            sequences,
            free_pages,
            next_page_id,
            config,
            backend,
            sequence_id,
        )?;
        {
            let mut pages_guard = pages.write()?;
            // Safe: new_page_id was just allocated above
            let new_page = pages_guard
                .get_mut(&new_page_id)
                .ok_or(KvCacheError::PageNotFound(new_page_id))?;
            new_page.append_token(token)?;
        }
        let mut sequences_guard = sequences.write()?;
        let sequence = sequences_guard
            .get_mut(&sequence_id)
            .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;
        sequence.total_tokens += 1;
    }

    Ok(())
}

/// Get all tokens for a sequence
pub fn get_sequence_tokens(
    pages: &RwLock<HashMap<u32, CachePage>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    sequence_id: u32,
) -> KvCacheResult<Vec<u32>> {
    let sequences_guard = sequences.read()?;
    let sequence = sequences_guard
        .get(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    let mut tokens = Vec::with_capacity(sequence.total_tokens);
    let sequence_pages = sequence.pages.clone();

    drop(sequences_guard); // Release sequences lock before acquiring pages lock

    let pages_guard = pages.read()?;
    for page_id in &sequence_pages {
        let page = pages_guard
            .get(page_id)
            .ok_or(KvCacheError::PageNotFound(*page_id))?;
        tokens.extend_from_slice(&page.tokens);
    }

    Ok(tokens)
}

/// Get the length of a sequence
pub fn get_sequence_length(
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    sequence_id: u32,
) -> KvCacheResult<usize> {
    let sequences_guard = sequences.read()?;
    let sequence = sequences_guard
        .get(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    Ok(sequence.total_tokens)
}

/// Remove a sequence and free its pages
///
/// NOTE: This function requires page_table and block_allocator for
/// deallocating blocks from the paged system. Use with caution.
pub fn remove_sequence(
    pages: &RwLock<HashMap<u32, CachePage>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    free_pages: &RwLock<Vec<u32>>,
    page_table: &RwLock<super::PageTable>,
    block_allocator: &RwLock<super::BlockAllocator>,
    sequence_id: u32,
) -> KvCacheResult<()> {
    let sequence = sequences
        .write()?
        .remove(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    // Free pages from GPU memory
    let mut pages_guard = pages.write()?;
    let mut free_pages_guard = free_pages.write()?;

    for page_id in sequence.pages {
        if pages_guard.remove(&page_id).is_some() {
            free_pages_guard.push(page_id);
        }
    }

    // Also deallocate blocks from paged system
    // Get blocks from page table before removing sequence
    let blocks_to_deallocate = page_table
        .read()?
        .get_sequence_blocks(sequence_id)
        .map(|v| v.to_vec());

    if let Some(blocks) = blocks_to_deallocate {
        let mut allocator = block_allocator.write()?;
        for &block_id in &blocks {
            allocator.deallocate(block_id);
        }
    }

    // Remove from page table
    page_table.write()?.remove_sequence(sequence_id);

    Ok(())
}

/// Get cache statistics
pub fn get_cache_stats(
    pages: &RwLock<HashMap<u32, CachePage>>,
    free_pages: &RwLock<Vec<u32>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
) -> CacheStats {
    // Safe: These locks should never be poisoned in normal operation
    // If they are, it indicates a serious bug and we want to know about it
    let pages_guard = pages.read().expect("KvCache pages lock poisoned");
    let free_pages_guard = free_pages
        .read()
        .expect("KvCache free_pages lock poisoned");
    let sequences_guard = sequences
        .read()
        .expect("KvCache sequences lock poisoned");

    CacheStats {
        total_pages: pages_guard.len(),
        free_pages: free_pages_guard.len(),
        active_sequences: sequences_guard.len(),
        total_tokens: sequences_guard.values().map(|s| s.total_tokens).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_page_first_page() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();

        let pages = RwLock::new(HashMap::new());
        let sequences = RwLock::new(HashMap::new());
        let free_pages = RwLock::new(Vec::new());
        let next_page_id = RwLock::new(0);

        let page_id = allocate_page(
            &pages,
            &sequences,
            &free_pages,
            &next_page_id,
            &config,
            &backend,
            1,
        )
        .unwrap();

        assert_eq!(page_id, 0);
        assert_eq!(pages.read().unwrap().len(), 1);
        assert_eq!(sequences.read().unwrap().len(), 1);
    }

    #[test]
    fn test_allocate_page_reuse_free() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();

        let pages = RwLock::new(HashMap::new());
        let sequences = RwLock::new(HashMap::new());
        let mut free_pages_vec = Vec::new();
        free_pages_vec.push(5); // Pre-populate with a free page
        let free_pages = RwLock::new(free_pages_vec);
        let next_page_id = RwLock::new(0);

        let page_id = allocate_page(
            &pages,
            &sequences,
            &free_pages,
            &next_page_id,
            &config,
            &backend,
            1,
        )
        .unwrap();

        // Should reuse the free page
        assert_eq!(page_id, 5);
    }

    #[test]
    fn test_allocate_page_capacity_exceeded() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(2, 2, 32, 128, 24).unwrap();

        let pages = RwLock::new(HashMap::new());
        let sequences = RwLock::new(HashMap::new());
        let free_pages = RwLock::new(Vec::new());
        let next_page_id = RwLock::new(0);

        // Allocate max pages
        allocate_page(
            &pages,
            &sequences,
            &free_pages,
            &next_page_id,
            &config,
            &backend,
            1,
        )
        .unwrap();
        allocate_page(
            &pages,
            &sequences,
            &free_pages,
            &next_page_id,
            &config,
            &backend,
            2,
        )
        .unwrap();

        // Third allocation should fail
        let result = allocate_page(
            &pages,
            &sequences,
            &free_pages,
            &next_page_id,
            &config,
            &backend,
            3,
        );

        assert!(matches!(result, Err(KvCacheError::CapacityExceeded)));
    }
}
