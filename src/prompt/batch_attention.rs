//! Batch-optimized attention computation for prompt processing
//!
//! During prompt processing, all tokens are known upfront, enabling
//! specialized optimizations that aren't possible during autoregressive generation.

use crate::prompt::profiling::{AttentionPattern, estimate_attention_memory};

impl AttentionPattern {
    /// Detect attention pattern from model configuration and heuristics
    pub fn detect(
        seq_len: usize,
        use_causal: bool,
        _layer_idx: usize,
        _total_layers: usize,
    ) -> Self {
        if use_causal {
            // Most LLMs use causal attention
            AttentionPattern::Causal
        } else {
            // Non-causal is typically dense
            AttentionPattern::Dense
        }
    }

    /// Get optimal block size for kernel execution
    pub fn optimal_block_size(&self, seq_len: usize, head_dim: usize) -> usize {
        match self {
            AttentionPattern::Dense => {
                // Dense attention benefits from larger blocks
                if seq_len <= 512 {
                    256
                } else if seq_len <= 1024 {
                    512
                } else {
                    1024
                }
            }
            AttentionPattern::SlidingWindow { window_size } => {
                // Match block size to window
                (*window_size).min(256)
            }
            AttentionPattern::Causal => {
                // Causal can use standard block size
                256
            }
            AttentionPattern::Sparse => 128,
            AttentionPattern::Mixed => 256,
        }
    }

    /// Estimate if tiling is beneficial
    pub fn should_tile(&self, seq_len: usize, head_dim: usize) -> bool {
        match self {
            AttentionPattern::Dense => seq_len * head_dim > 65536, // 256x256 threshold
            AttentionPattern::SlidingWindow { .. } => false, // Window attention already tiles
            AttentionPattern::Causal => seq_len > 512,
            AttentionPattern::Sparse => false,
            AttentionPattern::Mixed => seq_len > 512,
        }
    }
}

/// Batch attention optimizer configuration
#[derive(Debug, Clone)]
pub struct BatchAttentionConfig {
    /// Enable parallel attention computation across heads
    pub parallel_heads: bool,

    /// Enable memory-efficient tiling for long sequences
    pub enable_tiling: bool,

    /// Tile size for memory-efficient attention
    pub tile_size: usize,

    /// Use fused QKV projection kernel
    pub use_fused_qkv: bool,

    /// Batch multiple layers in single kernel launch
    pub batch_layers: bool,

    /// Number of layers to batch at once
    pub layer_batch_size: usize,
}

impl Default for BatchAttentionConfig {
    fn default() -> Self {
        BatchAttentionConfig {
            parallel_heads: true,
            enable_tiling: true,
            tile_size: 512,
            use_fused_qkv: true,
            batch_layers: false,
            layer_batch_size: 4,
        }
    }
}

impl BatchAttentionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parallel_heads(mut self, enabled: bool) -> Self {
        self.parallel_heads = enabled;
        self
    }

    pub fn with_tiling(mut self, enabled: bool) -> Self {
        self.enable_tiling = enabled;
        self
    }

    pub fn with_tile_size(mut self, size: usize) -> Self {
        self.tile_size = size;
        self
    }

    pub fn with_fused_qkv(mut self, enabled: bool) -> Self {
        self.use_fused_qkv = enabled;
        self
    }

    pub fn with_layer_batching(mut self, enabled: bool, batch_size: usize) -> Self {
        self.batch_layers = enabled;
        self.layer_batch_size = batch_size;
        self
    }
}

/// Optimizer for batch attention computation
pub struct BatchAttentionOptimizer {
    config: BatchAttentionConfig,
    pattern: AttentionPattern,
}

impl BatchAttentionOptimizer {
    /// Create a new optimizer with the given pattern
    pub fn new(pattern: AttentionPattern) -> Self {
        BatchAttentionOptimizer {
            config: BatchAttentionConfig::default(),
            pattern,
        }
    }

    /// Create with custom configuration
    pub fn with_config(pattern: AttentionPattern, config: BatchAttentionConfig) -> Self {
        BatchAttentionOptimizer { config, pattern }
    }

    /// Calculate optimal kernel launch parameters
    pub fn kernel_params(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> KernelLaunchParams {
        let block_size = self.pattern.optimal_block_size(seq_len, head_dim);
        let should_tile = self.pattern.should_tile(seq_len, head_dim) && self.config.enable_tiling;
        let tile_size = if should_tile {
            self.config.tile_size.min(seq_len)
        } else {
            seq_len
        };

        // Calculate grid dimensions
        let grid_x = (seq_len + block_size - 1) / block_size;
        let grid_y = num_heads;

        // Shared memory per block (for tiling)
        let shared_mem_bytes = if should_tile {
            // QK^T tile + softmax partials
            (tile_size * tile_size * std::mem::size_of::<f32>()) + (block_size * std::mem::size_of::<f32>())
        } else {
            0
        };

        KernelLaunchParams {
            block_size,
            grid_x,
            grid_y,
            shared_mem_bytes,
            use_tiling: should_tile,
            tile_size,
        }
    }

    /// Estimate memory usage for batched attention
    pub fn estimate_memory(&self, seq_len: usize, num_heads: usize, head_dim: usize) -> usize {
        let base_memory = estimate_attention_memory(seq_len, num_heads, head_dim);

        // Add overhead for tiling if enabled
        let tiling_overhead = if self.config.enable_tiling && self.pattern.should_tile(seq_len, head_dim) {
            let tile_size = self.config.tile_size.min(seq_len);
            tile_size * tile_size * std::mem::size_of::<f32>()
        } else {
            0
        };

        base_memory + tiling_overhead
    }

    /// Estimate speedup from batch optimizations
    pub fn estimated_speedup(&self, seq_len: usize) -> f32 {
        let mut speedup = 1.0;

        // Parallel heads
        if self.config.parallel_heads {
            speedup *= 1.5;
        }

        // Pattern-specific speedup
        speedup *= self.pattern.batch_speedup_factor(seq_len);

        // Tiling benefit for long sequences
        if self.config.enable_tiling && seq_len > 1024 {
            speedup *= 1.2;
        }

        // Fused QKV
        if self.config.use_fused_qkv {
            speedup *= 1.3;
        }

        // Layer batching
        if self.config.batch_layers {
            speedup *= 1.1;
        }

        speedup
    }

    /// Check if attention computation should be chunked
    pub fn should_chunk(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        available_memory_mb: usize,
    ) -> bool {
        let required_memory = self.estimate_memory(seq_len, num_heads, head_dim);
        let available_bytes = available_memory_mb * 1024 * 1024;

        required_memory > available_bytes
    }

    /// Calculate optimal chunk size for memory-constrained processing
    pub fn calculate_chunk_size(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        available_memory_mb: usize,
    ) -> usize {
        let available_bytes = available_memory_mb * 1024 * 1024;
        let per_token_memory = estimate_attention_memory(1, num_heads, head_dim);

        // Binary search for optimal chunk size
        let mut low = 1;
        let mut high = seq_len;

        while low < high {
            let mid = (low + high + 1) / 2;
            let memory = self.estimate_memory(mid, num_heads, head_dim);

            if memory <= available_bytes {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        low
    }
}

impl Default for BatchAttentionOptimizer {
    fn default() -> Self {
        Self::new(AttentionPattern::Causal)
    }
}

/// Kernel launch parameters for optimized attention
#[derive(Debug, Clone)]
pub struct KernelLaunchParams {
    /// Block size (threads per block)
    pub block_size: usize,

    /// Grid X dimension
    pub grid_x: usize,

    /// Grid Y dimension
    pub grid_y: usize,

    /// Shared memory per block in bytes
    pub shared_mem_bytes: usize,

    /// Whether tiling is enabled
    pub use_tiling: bool,

    /// Tile size (if tiling enabled)
    pub tile_size: usize,
}

impl KernelLaunchParams {
    /// Total number of thread blocks
    pub fn total_blocks(&self) -> usize {
        self.grid_x * self.grid_y
    }

    /// Total number of threads
    pub fn total_threads(&self) -> usize {
        self.total_blocks() * self.block_size
    }

    /// Check if parameters are valid
    pub fn is_valid(&self) -> bool {
        self.block_size > 0
            && self.grid_x > 0
            && self.grid_y > 0
            && self.tile_size <= self.block_size * 2
    }
}

/// Helper for determining optimal RoPE batch size
pub struct RopeBatchOptimizer {
    /// Precomputed cos/sin cache size
    cache_size: usize,

    /// Batch cos/sin computation
    batch_cos_sin: bool,
}

impl RopeBatchOptimizer {
    pub fn new(cache_size: usize) -> Self {
        RopeBatchOptimizer {
            cache_size,
            batch_cos_sin: true,
        }
    }

    /// Calculate optimal batch size for RoPE computation
    pub fn optimal_batch_size(&self, seq_len: usize) -> usize {
        if seq_len <= self.cache_size {
            // Can process entire sequence at once
            seq_len
        } else {
            // Batch in cache-sized chunks
            self.cache_size
        }
    }

    /// Check if cos/sin values should be precomputed
    pub fn should_precompute(&self, seq_len: usize) -> bool {
        self.batch_cos_sin && seq_len <= self.cache_size * 4
    }

    /// Calculate memory for precomputed cos/sin tables
    pub fn cos_sin_memory(&self, seq_len: usize, head_dim: usize) -> usize {
        let half_dim = head_dim / 2;
        seq_len * half_dim * 2 * std::mem::size_of::<f32>()
    }
}

impl Default for RopeBatchOptimizer {
    fn default() -> Self {
        Self::new(2048)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_pattern_detect() {
        let pattern = AttentionPattern::detect(512, true, 0, 32);
        assert_eq!(pattern, AttentionPattern::Causal);

        let pattern = AttentionPattern::detect(512, false, 0, 32);
        assert_eq!(pattern, AttentionPattern::Dense);
    }

    #[test]
    fn test_batch_speedup_factor() {
        let causal = AttentionPattern::Causal;
        assert_eq!(causal.batch_speedup_factor(512), 2.5);

        let dense = AttentionPattern::Dense;
        // Dense at 512: base=2.0, scaling=sqrt(512/512)=1.0, so speedup=2.0
        assert!(dense.batch_speedup_factor(512) >= 2.0);

        // For longer sequences, scaling > 1.0
        assert!(dense.batch_speedup_factor(1024) > dense.batch_speedup_factor(512));
    }

    #[test]
    fn test_optimal_block_size() {
        let causal = AttentionPattern::Causal;
        assert_eq!(causal.optimal_block_size(512, 128), 256);

        let dense = AttentionPattern::Dense;
        assert_eq!(dense.optimal_block_size(512, 128), 256);
        // For 2048: seq_len > 1024, so returns 1024
        assert_eq!(dense.optimal_block_size(2048, 128), 1024);
    }

    #[test]
    fn test_kernel_params() {
        let optimizer = BatchAttentionOptimizer::new(AttentionPattern::Causal);
        let params = optimizer.kernel_params(512, 32, 128);

        assert_eq!(params.block_size, 256);
        assert!(params.grid_x > 0);
        assert_eq!(params.grid_y, 32);
        assert!(params.is_valid());
    }

    #[test]
    fn test_estimated_speedup() {
        let config = BatchAttentionConfig::new()
            .with_parallel_heads(true)
            .with_fused_qkv(true);

        let optimizer = BatchAttentionOptimizer::with_config(AttentionPattern::Causal, config);
        let speedup = optimizer.estimated_speedup(512);

        assert!(speedup > 1.0);
    }

    #[test]
    fn test_should_chunk() {
        let optimizer = BatchAttentionOptimizer::new(AttentionPattern::Causal);

        // Should not chunk for small sequence with plenty of memory
        assert!(!optimizer.should_chunk(512, 32, 128, 1024));

        // Should chunk for large sequence with limited memory
        assert!(optimizer.should_chunk(4096, 32, 128, 10));
    }

    #[test]
    fn test_calculate_chunk_size() {
        let optimizer = BatchAttentionOptimizer::new(AttentionPattern::Causal);

        // With limited memory, should return small chunks
        let chunk_size = optimizer.calculate_chunk_size(4096, 32, 128, 10);
        assert!(chunk_size < 4096);
        assert!(chunk_size > 0);
    }

    #[test]
    fn test_rope_batch_optimizer() {
        let optimizer = RopeBatchOptimizer::new(2048);

        assert_eq!(optimizer.optimal_batch_size(512), 512);
        assert_eq!(optimizer.optimal_batch_size(4096), 2048);
        assert!(optimizer.should_precompute(2048));
    }

    #[test]
    fn test_cos_sin_memory() {
        let optimizer = RopeBatchOptimizer::new(2048);
        let memory = optimizer.cos_sin_memory(512, 128);

        assert_eq!(memory, 512 * 64 * 2 * 4); // seq_len * half_dim * 2 * 4 bytes
    }
}
