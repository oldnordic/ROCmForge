//! Tensor allocator for efficient buffer reuse.
//!
//! Inspired by llama.cpp's ggml_allocr, this allocator tracks and reuses
//! tensor buffers to reduce GPU allocations during graph execution.
//!
//! # Strategy
//!
//! - Maintain pools of free buffers by size
//! - When allocating, try to reuse a buffer of the same size
//! - When freeing, return buffer to the appropriate pool
//! - Reset clears all pools for fresh graph execution

use std::collections::HashMap;

use crate::backend::HipBuffer;

/// Free block in the allocator pool.
#[derive(Debug)]
struct FreeBlock {
    /// Buffer that can be reused
    buffer: HipBuffer,
    /// Size in bytes
    #[allow(dead_code)] // Reserved for future allocator statistics tracking
    size: usize,
}

/// Tensor allocator for efficient buffer reuse.
///
/// Maintains pools of free buffers grouped by size. When allocating,
/// it tries to find a buffer of the exact same size. When freeing,
/// buffers are returned to the appropriate pool for reuse.
#[derive(Debug)]
pub struct TensorAllocator {
    /// Free buffer pools indexed by size (in bytes)
    free_pools: HashMap<usize, Vec<FreeBlock>>,
    /// Total allocated bytes (for stats)
    total_allocated: usize,
    /// Total reused bytes (for stats)
    total_reused: usize,
    /// Maximum number of buffers to keep per size pool
    max_pool_size: usize,
}

impl Default for TensorAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorAllocator {
    /// Create a new tensor allocator.
    pub fn new() -> Self {
        Self {
            free_pools: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            max_pool_size: 16, // Keep up to 16 buffers of each size
        }
    }

    /// Set the maximum number of buffers to keep per size pool.
    pub fn with_max_pool_size(mut self, max: usize) -> Self {
        self.max_pool_size = max;
        self
    }

    /// Allocate a buffer of the given size.
    ///
    /// First tries to find a reusable buffer of the same size.
    /// If none exists, calls the allocate_fn to create a new buffer.
    ///
    /// # Parameters
    /// - `size`: Required size in bytes
    /// - `allocate_fn`: Function to create a new buffer if no reusable one exists
    ///
    /// # Returns
    /// The allocated buffer, or error if allocation fails
    pub fn allocate<F>(&mut self, size: usize, allocate_fn: F) -> Result<HipBuffer, String>
    where
        F: FnOnce(usize) -> Result<HipBuffer, String>,
    {
        // Try to find a reusable buffer of the exact size
        if let Some(pool) = self.free_pools.get_mut(&size) {
            if let Some(free_block) = pool.pop() {
                self.total_reused += size;
                return Ok(free_block.buffer);
            }
        }

        // No reusable buffer found, allocate new
        self.total_allocated += size;
        allocate_fn(size)
    }

    /// Free a buffer back to the pool for reuse.
    ///
    /// # Parameters
    /// - `buffer`: The buffer to return
    /// - `size`: Size of the buffer in bytes
    pub fn free(&mut self, buffer: HipBuffer, size: usize) {
        let pool = self.free_pools.entry(size).or_insert_with(Vec::new);

        // Only keep if we haven't exceeded the max pool size
        if pool.len() < self.max_pool_size {
            pool.push(FreeBlock { buffer, size });
        }
        // Otherwise, drop the buffer (it gets deallocated)
    }

    /// Reset the allocator, clearing all free pools.
    ///
    /// This should be called between graph executions to start fresh.
    pub fn reset(&mut self) {
        self.free_pools.clear();
        self.total_allocated = 0;
        self.total_reused = 0;
    }

    /// Get the total number of bytes allocated (including reused).
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get the total number of bytes reused from the pool.
    pub fn total_reused(&self) -> usize {
        self.total_reused
    }

    /// Get the number of buffers currently in the free pools.
    pub fn pooled_buffer_count(&self) -> usize {
        self.free_pools.values().map(|v| v.len()).sum()
    }

    /// Get the total size of all buffers in the free pools.
    pub fn pooled_bytes(&self) -> usize {
        self.free_pools
            .iter()
            .flat_map(|(size, blocks)| blocks.iter().map(move |_| *size))
            .sum()
    }

    /// Get statistics about the allocator's performance.
    pub fn stats(&self) -> AllocatorStats {
        let reuse_rate = if self.total_allocated > 0 {
            (self.total_reused as f64 / (self.total_allocated + self.total_reused) as f64)
                * 100.0
        } else {
            0.0
        };

        AllocatorStats {
            total_allocated: self.total_allocated,
            total_reused: self.total_reused,
            pooled_buffer_count: self.pooled_buffer_count(),
            pooled_bytes: self.pooled_bytes(),
            reuse_rate_percent: reuse_rate,
        }
    }
}

/// Statistics about allocator performance.
#[derive(Debug, Clone, Copy)]
pub struct AllocatorStats {
    /// Total bytes allocated (not counting reuse)
    pub total_allocated: usize,
    /// Total bytes reused from pool
    pub total_reused: usize,
    /// Number of buffers currently in free pools
    pub pooled_buffer_count: usize,
    /// Total size of all pooled buffers
    pub pooled_bytes: usize,
    /// Percentage of allocations that were reused
    pub reuse_rate_percent: f64,
}

impl std::fmt::Display for AllocatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AllocatorStats: allocated={}KB, reused={}KB ({}%), pooled={} buffers ({}KB)",
            self.total_allocated / 1024,
            self.total_reused / 1024,
            self.reuse_rate_percent as u32,
            self.pooled_buffer_count,
            self.pooled_bytes / 1024
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // For tests, we can't easily test with real HipBuffer since it requires HIP
    // So we'll just test the stats tracking

    #[test]
    fn test_allocator_stats() {
        let alloc = TensorAllocator::new();
        let stats = alloc.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.total_reused, 0);
        assert_eq!(stats.pooled_buffer_count, 0);
        assert_eq!(stats.pooled_bytes, 0);
        assert_eq!(stats.reuse_rate_percent, 0.0);
    }

    #[test]
    fn test_allocator_reset() {
        let mut alloc = TensorAllocator::new();
        alloc.total_allocated = 1000;
        alloc.total_reused = 500;
        alloc.free_pools.insert(100, vec![]);
        alloc.free_pools.insert(200, vec![]);

        alloc.reset();

        assert_eq!(alloc.total_allocated, 0);
        assert_eq!(alloc.total_reused, 0);
        assert_eq!(alloc.pooled_buffer_count(), 0);
    }

    #[test]
    fn test_allocator_max_pool_size() {
        let alloc = TensorAllocator::new().with_max_pool_size(2);
        assert_eq!(alloc.max_pool_size, 2);
    }

    #[test]
    fn test_stats_display() {
        let stats = AllocatorStats {
            total_allocated: 1024,
            total_reused: 512,
            pooled_buffer_count: 4,
            pooled_bytes: 2048,
            reuse_rate_percent: 33.33,
        };
        let display = format!("{}", stats);
        assert!(display.contains("1KB")); // allocated
        assert!(display.contains("4 buffers")); // count
    }
}
