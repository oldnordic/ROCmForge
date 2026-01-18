//! Prompt processing profiling utilities
//!
//! Provides detailed performance profiling for the prompt processing phase,
//! identifying bottlenecks in attention, RoPE, and KV cache operations.

use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Detailed profile of prompt processing performance
#[derive(Debug, Clone)]
pub struct PromptProfile {
    /// Total number of tokens processed
    pub token_count: usize,

    /// Number of layers processed
    pub layer_count: usize,

    /// Per-layer timing breakdown
    pub layer_times: Vec<LayerTiming>,

    /// Aggregated operation times
    pub operation_times: OperationTiming,

    /// Memory allocation statistics
    pub memory_stats: MemoryStats,

    /// Attention pattern analysis
    pub attention_patterns: AttentionPatternAnalysis,
}

/// Timing breakdown for a single layer
#[derive(Debug, Clone)]
pub struct LayerTiming {
    /// Layer index
    pub layer_idx: usize,

    /// Total time for this layer
    pub total_time: Duration,

    /// Time spent on attention (QKV projection + attention computation)
    pub attention_time: Duration,

    /// Time spent on RoPE application
    pub rope_time: Duration,

    /// Time spent on MLP/feed-forward
    pub mlp_time: Duration,

    /// Time spent on layer normalization
    pub layernorm_time: Duration,

    /// Time spent on residual connections
    pub residual_time: Duration,
}

/// Aggregated timing across all operations
#[derive(Debug, Clone)]
pub struct OperationTiming {
    /// Total time across all layers
    pub total_time: Duration,

    /// Time spent computing QK^T (query-key dot product)
    pub qk_computation: Duration,

    /// Time spent computing softmax
    pub softmax_time: Duration,

    /// Time spent computing weighted value (attention_weights * V)
    pub weighted_value_time: Duration,

    /// Time spent on QKV projection
    pub qkv_projection: Duration,

    /// Time spent on output projection
    pub output_projection: Duration,

    /// Time spent writing to KV cache
    pub kv_cache_write: Duration,
}

/// Memory statistics for prompt processing
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,

    /// Memory used for KV cache
    pub kv_cache_bytes: usize,

    /// Memory used for attention weights (QK^T matrix)
    pub attention_weights_bytes: usize,

    /// Memory used for intermediate activations
    pub activation_bytes: usize,

    /// Number of memory allocations
    pub allocation_count: usize,

    /// Number of memory deallocations
    pub deallocation_count: usize,
}

/// Analysis of attention computation patterns
#[derive(Debug, Clone)]
pub struct AttentionPatternAnalysis {
    /// Pattern type detected
    pub pattern_type: AttentionPattern,

    /// Average attention window size (for sparse patterns)
    pub avg_window_size: Option<usize>,

    /// Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    pub sparsity: f32,

    /// Cache-friendliness score (0.0 - 1.0, higher is better)
    pub cache_friendliness: f32,

    /// Estimated speedup from batch-optimized attention
    pub estimated_batch_speedup: f32,
}

/// Detected attention computation pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPattern {
    /// Dense attention - all tokens attend to all tokens
    Dense,

    /// Sliding window - tokens attend to local window
    SlidingWindow { window_size: usize },

    /// Causal only - standard autoregressive mask
    Causal,

    /// Sparse with specific pattern (e.g., ALiBi)
    Sparse,

    /// Mixed pattern - different layers use different patterns
    Mixed,
}

impl AttentionPattern {
    /// Estimate speedup from batch processing
    pub fn batch_speedup_factor(&self, seq_len: usize) -> f32 {
        match self {
            AttentionPattern::Dense => {
                // Dense attention benefits significantly from batching
                // Speedup scales with sequence length
                let base = 2.0;
                let scaling = (seq_len as f32 / 512.0).sqrt().min(3.0);
                base * scaling
            }
            AttentionPattern::SlidingWindow { window_size } => {
                // Window attention benefits from batching within window
                let window_ratio = (*window_size as f32 / seq_len as f32).sqrt();
                1.5 + window_ratio
            }
            AttentionPattern::Causal => {
                // Causal attention can be batched efficiently
                2.5
            }
            AttentionPattern::Sparse => {
                // Sparse attention may not benefit as much
                1.5
            }
            AttentionPattern::Mixed => {
                // Mixed patterns get moderate benefit
                2.0
            }
        }
    }

    /// Estimate memory reduction from chunking
    pub fn chunking_memory_reduction(&self, chunk_size: usize, original_len: usize) -> f32 {
        if original_len <= chunk_size {
            return 0.0;
        }

        let num_chunks = (original_len + chunk_size - 1) / chunk_size;
        match self {
            AttentionPattern::Dense => {
                // Dense attention scales O(n^2), chunking gives significant savings
                let original_mem = (original_len * original_len) as f32;
                let chunked_mem = (num_chunks * chunk_size * chunk_size) as f32;
                1.0 - (chunked_mem / original_mem)
            }
            AttentionPattern::SlidingWindow { .. } => {
                // Window attention is already efficient, less benefit from chunking
                0.1
            }
            AttentionPattern::Causal => {
                // Causal can benefit from chunking with careful management
                0.3
            }
            AttentionPattern::Sparse => {
                // Sparse patterns see moderate benefit
                0.2
            }
            AttentionPattern::Mixed => {
                // Mixed patterns get moderate benefit
                0.25
            }
        }
    }
}

/// Real-time profiler for prompt processing
pub struct PromptProfiler {
    /// Profile data being collected
    profile: PromptProfile,

    /// Currently active timers
    timers: HashMap<String, Instant>,

    /// Memory usage tracking
    memory_tracking: bool,

    /// Enable profiling (can be disabled at runtime)
    enabled: bool,
}

impl PromptProfiler {
    /// Create a new profiler for a given token and layer count
    pub fn new(token_count: usize, layer_count: usize) -> Self {
        PromptProfiler {
            profile: PromptProfile {
                token_count,
                layer_count,
                layer_times: Vec::with_capacity(layer_count),
                operation_times: OperationTiming {
                    total_time: Duration::ZERO,
                    qk_computation: Duration::ZERO,
                    softmax_time: Duration::ZERO,
                    weighted_value_time: Duration::ZERO,
                    qkv_projection: Duration::ZERO,
                    output_projection: Duration::ZERO,
                    kv_cache_write: Duration::ZERO,
                },
                memory_stats: MemoryStats {
                    peak_memory_bytes: 0,
                    kv_cache_bytes: 0,
                    attention_weights_bytes: 0,
                    activation_bytes: 0,
                    allocation_count: 0,
                    deallocation_count: 0,
                },
                attention_patterns: AttentionPatternAnalysis {
                    pattern_type: AttentionPattern::Causal,
                    avg_window_size: None,
                    sparsity: 0.0,
                    cache_friendliness: 0.5,
                    estimated_batch_speedup: 2.0,
                },
            },
            timers: HashMap::new(),
            memory_tracking: false,
            enabled: true,
        }
    }

    /// Enable or disable profiling at runtime
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Start timing a named operation
    pub fn start(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        self.timers.insert(name.to_string(), Instant::now());
    }

    /// Stop timing a named operation and record the duration
    pub fn stop(&mut self, name: &str) -> Duration {
        if !self.enabled {
            return Duration::ZERO;
        }
        if let Some(start) = self.timers.remove(name) {
            let elapsed = start.elapsed();
            self.record_operation(name, elapsed);
            elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Record an operation's duration
    fn record_operation(&mut self, name: &str, duration: Duration) {
        match name {
            "qk_computation" => self.profile.operation_times.qk_computation += duration,
            "softmax" => self.profile.operation_times.softmax_time += duration,
            "weighted_value" => self.profile.operation_times.weighted_value_time += duration,
            "qkv_projection" => self.profile.operation_times.qkv_projection += duration,
            "output_projection" => self.profile.operation_times.output_projection += duration,
            "kv_cache_write" => self.profile.operation_times.kv_cache_write += duration,
            "layer_norm" => {
                if let Some(last) = self.profile.layer_times.last_mut() {
                    last.layernorm_time += duration;
                }
            }
            "rope" => {
                if let Some(last) = self.profile.layer_times.last_mut() {
                    last.rope_time += duration;
                }
            }
            "mlp" => {
                if let Some(last) = self.profile.layer_times.last_mut() {
                    last.mlp_time += duration;
                }
            }
            "residual" => {
                if let Some(last) = self.profile.layer_times.last_mut() {
                    last.residual_time += duration;
                }
            }
            _ => {}
        }
    }

    /// Start profiling a new layer
    pub fn start_layer(&mut self, layer_idx: usize) {
        if !self.enabled {
            return;
        }
        self.timers.insert(format!("layer_{}", layer_idx), Instant::now());
        self.profile.layer_times.push(LayerTiming {
            layer_idx,
            total_time: Duration::ZERO,
            attention_time: Duration::ZERO,
            rope_time: Duration::ZERO,
            mlp_time: Duration::ZERO,
            layernorm_time: Duration::ZERO,
            residual_time: Duration::ZERO,
        });
    }

    /// End profiling for a layer
    pub fn end_layer(&mut self, layer_idx: usize) -> Duration {
        if !self.enabled {
            return Duration::ZERO;
        }
        if let Some(start) = self.timers.remove(&format!("layer_{}", layer_idx)) {
            let elapsed = start.elapsed();
            if let Some(layer) = self.profile.layer_times.last_mut() {
                layer.total_time = elapsed;
            }
            self.profile.operation_times.total_time += elapsed;
            elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Record attention timing for current layer
    pub fn record_attention(&mut self, duration: Duration) {
        if !self.enabled {
            return;
        }
        if let Some(layer) = self.profile.layer_times.last_mut() {
            layer.attention_time += duration;
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&mut self, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.profile.memory_stats.allocation_count += 1;
        self.profile.memory_stats.peak_memory_bytes =
            self.profile.memory_stats.peak_memory_bytes.saturating_add(bytes);
    }

    /// Record KV cache memory usage
    pub fn record_kv_cache_memory(&mut self, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.profile.memory_stats.kv_cache_bytes = bytes;
    }

    /// Analyze detected attention pattern
    pub fn analyze_attention_pattern(&mut self, pattern: AttentionPattern) {
        if !self.enabled {
            return;
        }
        self.profile.attention_patterns.pattern_type = pattern;
        self.profile.attention_patterns.estimated_batch_speedup =
            pattern.batch_speedup_factor(self.profile.token_count);
    }

    /// Get the complete profile
    pub fn finish(self) -> PromptProfile {
        self.profile
    }

    /// Get a snapshot of the current profile (for debugging)
    pub fn snapshot(&self) -> &PromptProfile {
        &self.profile
    }
}

impl Default for PromptProfiler {
    fn default() -> Self {
        Self::new(512, 32)
    }
}

/// Compute estimated memory requirements for attention computation
pub fn estimate_attention_memory(seq_len: usize, num_heads: usize, head_dim: usize) -> usize {
    // QK^T attention matrix: [seq_len, seq_len]
    let qk_matrix = seq_len * seq_len * std::mem::size_of::<f32>();

    // Q, K, V tensors: [seq_len, num_heads, head_dim]
    let qkv_tensor = 3 * seq_len * num_heads * head_dim * std::mem::size_of::<f32>();

    // Output tensor: [seq_len, num_heads, head_dim]
    let output_tensor = seq_len * num_heads * head_dim * std::mem::size_of::<f32>();

    qk_matrix + qkv_tensor + output_tensor
}

/// Compute estimated KV cache memory per token
pub fn estimate_kv_cache_per_token(num_layers: usize, num_heads: usize, head_dim: usize) -> usize {
    // Each layer stores K and V for each token
    // K and V each: [num_heads, head_dim]
    let per_layer_kv = 2 * num_heads * head_dim * std::mem::size_of::<f32>();
    num_layers * per_layer_kv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = PromptProfiler::new(128, 4);
        profiler.start("test_operation");
        std::thread::sleep(Duration::from_millis(10));
        let duration = profiler.stop("test_operation");
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_layers() {
        let mut profiler = PromptProfiler::new(128, 4);

        profiler.start_layer(0);
        profiler.record_attention(Duration::from_millis(5));
        profiler.start("rope");
        profiler.stop("rope");
        profiler.end_layer(0);

        profiler.start_layer(1);
        profiler.record_attention(Duration::from_millis(6));
        profiler.end_layer(1);

        let profile = profiler.finish();
        assert_eq!(profile.layer_times.len(), 2);
        assert_eq!(profile.layer_times[0].layer_idx, 0);
        assert_eq!(profile.layer_times[1].layer_idx, 1);
        assert_eq!(profile.layer_times[0].attention_time, Duration::from_millis(5));
    }

    #[test]
    fn test_attention_pattern_speedup() {
        let dense = AttentionPattern::Dense;
        let speedup_512 = dense.batch_speedup_factor(512);
        let speedup_1024 = dense.batch_speedup_factor(1024);

        // Dense at 512: base=2.0, scaling=sqrt(512/512)=1.0, so speedup=2.0
        assert!(speedup_512 >= 2.0);
        assert!(speedup_1024 > speedup_512); // Longer sequences benefit more
    }

    #[test]
    fn test_memory_estimation() {
        let attn_mem = estimate_attention_memory(512, 32, 128);
        let kv_per_token = estimate_kv_cache_per_token(32, 32, 128);

        // Attention matrix dominates: 512*512*4 = ~1MB
        assert!(attn_mem > 1_000_000);

        // KV per token: 32 layers * 2 * 32 heads * 128 dim * 4 bytes
        assert!(kv_per_token > 1_000_000); // ~1MB per token
    }

    #[test]
    fn test_profiler_disabled() {
        let mut profiler = PromptProfiler::new(128, 4);
        profiler.set_enabled(false);

        profiler.start("test");
        profiler.stop("test");
        profiler.start_layer(0);
        profiler.end_layer(0);

        let profile = profiler.finish();
        assert_eq!(profile.layer_times.len(), 0);
        assert_eq!(profile.operation_times.total_time, Duration::ZERO);
    }
}
