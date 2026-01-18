//! Prompt processing optimization module
//!
//! This module provides specialized optimizations for the prompt processing (encoding)
//! phase of LLM inference. Prompt processing differs from token generation in that:
//!
//! 1. All tokens are known upfront (no KV cache buildup during processing)
//! 2. Attention can be computed in parallel (batch processing)
//! 3. Memory can be pre-allocated for the entire prompt
//! 4. Early exit opportunities exist for cached prefixes
//!
//! # Architecture
//!
//! The module is organized into several components:
//!
//! - [`profiling`] - Performance profiling and bottleneck identification
//! - [`chunking`] - Prompt chunking strategies for memory optimization
//! - [`cache`] - Prefix cache for reusing computed states
//! - [`batch_attention`] - Batch-optimized attention computation

pub mod profiling;
pub mod chunking;
pub mod cache;
pub mod batch_attention;

pub use profiling::{PromptProfiler, PromptProfile, AttentionPattern};
pub use chunking::{PromptChunker, ChunkStrategy};
pub use cache::{PrefixCache, CachedPrefix};
pub use batch_attention::{BatchAttentionOptimizer};

use std::time::Duration;

/// Configuration for prompt processing optimizations
#[derive(Debug, Clone)]
pub struct PromptOptimizationConfig {
    /// Maximum prompt length to apply optimizations
    pub max_prompt_len: usize,

    /// Chunk size for prompt chunking (0 = no chunking)
    pub chunk_size: usize,

    /// Enable prefix caching
    pub enable_prefix_cache: bool,

    /// Minimum prefix length to cache
    pub min_cache_prefix: usize,

    /// Enable batch-optimized attention
    pub enable_batch_attention: bool,

    /// Profile prompt processing performance
    pub enable_profiling: bool,
}

impl Default for PromptOptimizationConfig {
    fn default() -> Self {
        Self {
            max_prompt_len: 4096,
            chunk_size: 512,
            enable_prefix_cache: true,
            min_cache_prefix: 32,
            enable_batch_attention: true,
            enable_profiling: false,
        }
    }
}

impl PromptOptimizationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_prompt_len(mut self, len: usize) -> Self {
        self.max_prompt_len = len;
        self
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_prefix_cache(mut self, enable: bool) -> Self {
        self.enable_prefix_cache = enable;
        self
    }

    pub fn with_batch_attention(mut self, enable: bool) -> Self {
        self.enable_batch_attention = enable;
        self
    }

    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }
}

/// Result of optimized prompt processing
#[derive(Debug, Clone)]
pub struct PromptProcessingResult {
    /// Number of tokens processed
    pub tokens_processed: usize,

    /// Total processing time
    pub total_time: Duration,

    /// Time spent on attention computation
    pub attention_time: Duration,

    /// Time spent on RoPE computation
    pub rope_time: Duration,

    /// Time spent on KV cache writes
    pub kv_write_time: Duration,

    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f32,

    /// Memory allocated for KV cache (in bytes)
    pub kv_memory_bytes: usize,

    /// Detailed profile (if profiling enabled)
    pub profile: Option<PromptProfile>,
}

impl PromptProcessingResult {
    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs > 0.0 {
            self.tokens_processed as f64 / secs
        } else {
            0.0
        }
    }

    /// Calculate time to first token (TTFT) in milliseconds
    pub fn ttft_ms(&self) -> f64 {
        self.total_time.as_millis() as f64
    }

    /// Calculate memory per token in bytes
    pub fn memory_per_token(&self) -> usize {
        if self.tokens_processed > 0 {
            self.kv_memory_bytes / self.tokens_processed
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PromptOptimizationConfig::default();
        assert_eq!(config.max_prompt_len, 4096);
        assert_eq!(config.chunk_size, 512);
        assert!(config.enable_prefix_cache);
        assert!(config.enable_batch_attention);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_config_builder() {
        let config = PromptOptimizationConfig::new()
            .with_max_prompt_len(2048)
            .with_chunk_size(256)
            .with_prefix_cache(false)
            .with_batch_attention(true)
            .with_profiling(true);

        assert_eq!(config.max_prompt_len, 2048);
        assert_eq!(config.chunk_size, 256);
        assert!(!config.enable_prefix_cache);
        assert!(config.enable_batch_attention);
        assert!(config.enable_profiling);
    }

    #[test]
    fn test_prompt_processing_result_metrics() {
        let result = PromptProcessingResult {
            tokens_processed: 512,
            total_time: Duration::from_millis(100),
            attention_time: Duration::from_millis(60),
            rope_time: Duration::from_millis(20),
            kv_write_time: Duration::from_millis(20),
            cache_hit_rate: 0.5,
            kv_memory_bytes: 512 * 1024, // 512KB
            profile: None,
        };

        // Tokens per second
        assert!((result.tokens_per_second() - 5120.0).abs() < 1.0);

        // TTFT
        assert!((result.ttft_ms() - 100.0).abs() < 0.1);

        // Memory per token
        assert_eq!(result.memory_per_token(), 1024);
    }
}
