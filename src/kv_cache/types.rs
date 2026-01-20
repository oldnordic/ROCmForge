//! Core types for KV cache
//!
//! This module contains error types, result types, and core data structures
//! used throughout the KV cache implementation.

use crate::backend::HipBuffer;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KvCacheError {
    #[error("Cache capacity exceeded")]
    CapacityExceeded,
    #[error("Invalid sequence ID: {0}")]
    InvalidSequenceId(u32),
    #[error("Page not found for sequence: {0}")]
    PageNotFound(u32),
    #[error("GPU memory error: {0}")]
    GpuError(#[from] crate::backend::HipError),
    #[error("Invalid cache configuration")]
    InvalidConfiguration,
    #[error("Internal lock poisoned - this indicates a bug: {0}")]
    LockPoisoned(String),
}

impl<T> From<std::sync::PoisonError<T>> for KvCacheError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        KvCacheError::LockPoisoned(format!("Lock poisoned: {}", err))
    }
}

pub type KvCacheResult<T> = Result<T, KvCacheError>;

/// Block identifier type for PagedAttention
pub type BlockId = u32;

/// Block table entry for PagedAttention
/// Maps logical block IDs to physical GPU memory blocks
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical block ID
    pub block_id: BlockId,
    /// Physical block ID (index into block pool)
    pub physical_block_id: u32,
    /// Reference count for sharing across sequences
    pub ref_count: Arc<AtomicUsize>,
    /// Sequence IDs using this block (for sharing)
    pub sequences: HashSet<u32>,
}

impl BlockTable {
    /// Create a new BlockTable entry
    pub fn new(block_id: BlockId, physical_block_id: u32) -> Self {
        BlockTable {
            block_id,
            physical_block_id,
            ref_count: Arc::new(AtomicUsize::new(1)),
            sequences: HashSet::new(),
        }
    }

    /// Add a sequence to this block
    pub fn add_sequence(&mut self, sequence_id: u32) {
        self.sequences.insert(sequence_id);
    }

    /// Remove a sequence from this block
    pub fn remove_sequence(&mut self, sequence_id: u32) -> bool {
        self.sequences.remove(&sequence_id)
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Increment reference count
    pub fn incr_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement reference count, returns previous count
    pub fn decr_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) - 1
    }
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

/// PagedAttention-specific cache statistics
#[derive(Debug, Clone)]
pub struct PagedCacheStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub active_sequences: usize,
}

/// Basic cache statistics for KV cache monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_pages: usize,
    pub free_pages: usize,
    pub active_sequences: usize,
    pub total_tokens: usize,
}

/// Detailed memory usage statistics for KV cache profiling
///
/// This provides comprehensive memory tracking for profiling and optimization.
/// Used by benchmarks and telemetry to understand memory patterns.
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Total GPU memory allocated for KV cache (bytes)
    pub total_gpu_bytes: usize,
    /// Memory currently in use by active sequences (bytes)
    pub used_gpu_bytes: usize,
    /// Memory available for new allocations (bytes)
    pub free_gpu_bytes: usize,
    /// Number of physical blocks allocated
    pub physical_blocks: usize,
    /// Number of logical blocks in use
    pub logical_blocks: usize,
    /// Page table memory overhead (bytes)
    pub page_table_bytes: usize,
    /// Block allocator metadata overhead (bytes)
    pub allocator_bytes: usize,
    /// Number of active sequences
    pub active_sequences: usize,
    /// Total tokens stored across all sequences
    pub total_tokens: usize,
    /// Memory per token (bytes/token)
    pub bytes_per_token: f64,
    /// Fragmentation ratio (0-1, higher = more fragmented)
    pub fragmentation_ratio: f64,
}

impl MemoryProfile {
    /// Format bytes as human readable
    pub fn format_bytes(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = 1024 * 1024;
        const GB: usize = 1024 * 1024 * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Print memory profile report
    pub fn report(&self) {
        println!("\n  KV Cache Memory Profile:");
        println!("    Total GPU memory:   {} ({} blocks)",
                 Self::format_bytes(self.total_gpu_bytes), self.physical_blocks);
        println!("    Used memory:        {}", Self::format_bytes(self.used_gpu_bytes));
        println!("    Free memory:        {}", Self::format_bytes(self.free_gpu_bytes));
        println!("    Page table overhead:{}", Self::format_bytes(self.page_table_bytes));
        println!("    Allocator overhead: {}", Self::format_bytes(self.allocator_bytes));
        println!("    Active sequences:   {}", self.active_sequences);
        println!("    Total tokens:       {}", self.total_tokens);
        println!("    Bytes per token:    {:.2}", self.bytes_per_token);
        println!("    Fragmentation:      {:.2}%", self.fragmentation_ratio * 100.0);

        // Calculate metadata overhead percentage
        let metadata_bytes = self.page_table_bytes + self.allocator_bytes;
        let overhead_pct = if self.total_gpu_bytes > 0 {
            (metadata_bytes as f64 / self.total_gpu_bytes as f64) * 100.0
        } else {
            0.0
        };
        println!("    Metadata overhead:  {} ({:.3}%)",
                 Self::format_bytes(metadata_bytes), overhead_pct);
    }

    /// Calculate memory efficiency (used / total)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.total_gpu_bytes > 0 {
            self.used_gpu_bytes as f64 / self.total_gpu_bytes as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_creation() {
        let bt = BlockTable::new(0, 100);
        assert_eq!(bt.block_id, 0);
        assert_eq!(bt.physical_block_id, 100);
        assert_eq!(bt.ref_count(), 1);
        assert!(bt.sequences.is_empty());
    }

    #[test]
    fn test_block_table_ref_counting() {
        let bt = BlockTable::new(0, 100);
        assert_eq!(bt.ref_count(), 1);

        bt.incr_ref();
        assert_eq!(bt.ref_count(), 2);

        bt.decr_ref();
        assert_eq!(bt.ref_count(), 1);
    }

    #[test]
    fn test_block_table_sequences() {
        let mut bt = BlockTable::new(0, 100);
        bt.add_sequence(1);
        bt.add_sequence(2);

        assert!(bt.sequences.contains(&1));
        assert!(bt.sequences.contains(&2));
        assert_eq!(bt.sequences.len(), 2);

        assert!(bt.remove_sequence(1));
        assert!(!bt.sequences.contains(&1));
        assert!(bt.sequences.contains(&2));
        assert_eq!(bt.sequences.len(), 1);
    }

    #[test]
    fn test_memory_profile_format_bytes() {
        assert_eq!(MemoryProfile::format_bytes(500), "500 B");
        assert_eq!(MemoryProfile::format_bytes(2048), "2.00 KB");
        assert_eq!(MemoryProfile::format_bytes(2_000_000), "1.91 MB");
        assert_eq!(MemoryProfile::format_bytes(2_000_000_000), "1.86 GB");
    }
}
