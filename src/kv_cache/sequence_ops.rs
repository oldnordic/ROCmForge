//! Sequence state tracking and LRU eviction operations
//!
//! This module contains operations for managing sequence lifecycle,
//! tracking access times for LRU eviction, and cleanup of completed sequences.

use super::config::CacheConfig;
use super::page_table::PageTable;
use super::pages::{CachePage, SequenceCache};
use super::block_allocator::BlockAllocator;
use super::types::{KvCacheError, KvCacheResult};
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;
use std::time::Instant;

/// Mark a sequence as completed (to be cleaned up later)
pub fn mark_sequence_completed(
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    sequence_id: u32,
) -> KvCacheResult<()> {
    let mut sequences_guard = sequences.write()?;
    let sequence = sequences_guard
        .get_mut(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    sequence.mark_completed();
    Ok(())
}

/// Check if a sequence is marked as completed
pub fn is_sequence_completed(
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    sequence_id: u32,
) -> KvCacheResult<bool> {
    let sequences_guard = sequences.read()?;
    let sequence = sequences_guard
        .get(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    Ok(sequence.is_completed)
}

/// Update the last access time for a sequence (for LRU tracking)
pub fn update_sequence_access(
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    sequence_id: u32,
) -> KvCacheResult<()> {
    let mut sequences_guard = sequences.write()?;
    let sequence = sequences_guard
        .get_mut(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    sequence.update_access();
    Ok(())
}

/// Get the last access time for a sequence
pub fn get_sequence_access_time(
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    sequence_id: u32,
) -> KvCacheResult<Instant> {
    let sequences_guard = sequences.read()?;
    let sequence = sequences_guard
        .get(&sequence_id)
        .ok_or(KvCacheError::InvalidSequenceId(sequence_id))?;

    Ok(sequence.last_access)
}

/// Get list of active (non-completed) sequence IDs
pub fn get_active_sequences(
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
) -> KvCacheResult<Vec<u32>> {
    let sequences_guard = sequences.read()?;

    let active: Vec<u32> = sequences_guard
        .values()
        .filter(|s| s.is_active())
        .map(|s| s.sequence_id)
        .collect();

    Ok(active)
}

/// Remove all completed sequences from the cache
///
/// This should be called periodically to free up memory.
/// Returns the number of sequences removed.
pub fn cleanup_completed_sequences(
    pages: &RwLock<HashMap<u32, CachePage>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    free_pages: &RwLock<Vec<u32>>,
) -> KvCacheResult<usize> {
    let mut sequences_guard = sequences.write()?;

    // Find completed sequence IDs
    let completed_ids: Vec<u32> = sequences_guard
        .iter()
        .filter(|(_, s)| s.is_completed)
        .map(|(id, _)| *id)
        .collect();

    let mut removed_count = 0;

    for seq_id in completed_ids {
        // Remove from sequences map
        if let Some(sequence) = sequences_guard.remove(&seq_id) {
            // Free pages from GPU memory
            let mut pages_guard = pages.write()?;
            let mut free_pages_guard = free_pages.write()?;

            for page_id in sequence.pages {
                if pages_guard.remove(&page_id).is_some() {
                    free_pages_guard.push(page_id);
                }
            }

            removed_count += 1;
        }
    }

    Ok(removed_count)
}

/// Evict least recently used sequences to free up space for new sequences
///
/// This is called automatically when capacity is exceeded during allocation.
/// Only active (non-completed) sequences are considered for eviction.
pub fn evict_lru_sequences(
    pages: &RwLock<HashMap<u32, CachePage>>,
    sequences: &RwLock<HashMap<u32, SequenceCache>>,
    config: &CacheConfig,
    required_pages: usize,
    remove_fn: &mut impl FnMut(u32) -> KvCacheResult<()>,
) -> KvCacheResult<()> {
    let sequences_guard = sequences.read()?;

    // Check if we need to evict
    let current_usage = pages.read()?.len();
    let max_pages = config.max_pages;

    if current_usage + required_pages <= max_pages {
        return Ok(());
    }

    // Find LRU sequences (only active sequences, not completed ones)
    let mut seq_access_times: Vec<(u32, Instant)> = sequences_guard
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
        let total_pages: usize = sequences_guard
            .values()
            .filter(|s| s.is_active())
            .map(|s| s.pages.len())
            .sum();
        (total_pages / seq_access_times.len()).max(1)
    };

    let seqs_to_evict = (pages_to_free / avg_pages_per_seq)
        .max(1)
        .min(seq_access_times.len());

    // Drop the read lock before acquiring write locks
    drop(sequences_guard);

    // Evict LRU sequences
    for (seq_id, _) in seq_access_times.iter().take(seqs_to_evict) {
        let _ = remove_fn(*seq_id);
    }

    Ok(())
}

/// Sort free lists to improve allocation patterns
///
/// Sorting free lists can improve cache locality by allocating
/// contiguous blocks when possible.
pub fn sort_free_lists(free_pages: &RwLock<Vec<u32>>) {
    let mut free_pages_guard = free_pages.write().expect("Free pages lock poisoned");
    free_pages_guard.sort();
    free_pages_guard.dedup(); // Remove duplicates if any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HipBackend;

    #[test]
    fn test_mark_and_check_completed() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();

        let sequences = RwLock::new(HashMap::new());
        let seq_id = 1;

        // Create a sequence
        sequences
            .write()
            .unwrap()
            .insert(seq_id, SequenceCache::new(seq_id));

        // Initially not completed
        assert!(!is_sequence_completed(&sequences, seq_id).unwrap());

        // Mark as completed
        mark_sequence_completed(&sequences, seq_id).unwrap();

        // Now should be completed
        assert!(is_sequence_completed(&sequences, seq_id).unwrap());
    }

    #[test]
    fn test_update_and_get_access_time() {
        let sequences = RwLock::new(HashMap::new());
        let seq_id = 1;

        sequences
            .write()
            .unwrap()
            .insert(seq_id, SequenceCache::new(seq_id));

        let time1 = get_sequence_access_time(&sequences, seq_id).unwrap();

        // Update access time
        std::thread::sleep(std::time::Duration::from_millis(10));
        update_sequence_access(&sequences, seq_id).unwrap();

        let time2 = get_sequence_access_time(&sequences, seq_id).unwrap();

        // Time should have advanced
        assert!(time2 > time1);
    }

    #[test]
    fn test_get_active_sequences() {
        let sequences = RwLock::new(HashMap::new());

        // Add sequences
        sequences
            .write()
            .unwrap()
            .insert(1, SequenceCache::new(1));
        sequences
            .write()
            .unwrap()
            .insert(2, SequenceCache::new(2));
        sequences
            .write()
            .unwrap()
            .insert(3, SequenceCache::new(3));

        // Mark one as completed
        mark_sequence_completed(&sequences, 2).unwrap();

        // Should return only active sequences
        let active = get_active_sequences(&sequences).unwrap();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&1));
        assert!(active.contains(&3));
        assert!(!active.contains(&2));
    }

    #[test]
    fn test_cleanup_completed_sequences() {
        let backend = HipBackend::new().unwrap();
        let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();

        let pages = RwLock::new(HashMap::new());
        let sequences = RwLock::new(HashMap::new());
        let free_pages = RwLock::new(Vec::new());

        // Create a page and sequence
        let page = CachePage::new(0, 1, &backend, &config).unwrap();
        pages.write().unwrap().insert(0, page);

        let mut sequence = SequenceCache::new(1);
        sequence.add_page(0);
        sequences.write().unwrap().insert(1, sequence);

        // Mark as completed
        mark_sequence_completed(&sequences, 1).unwrap();

        // Cleanup should remove the sequence and free the page
        let removed = cleanup_completed_sequences(&pages, &sequences, &free_pages).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(sequences.read().unwrap().len(), 0);
        assert_eq!(free_pages.read().unwrap().len(), 1);
        assert_eq!(pages.read().unwrap().len(), 0);
    }
}
