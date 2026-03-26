//! Prefill batch configuration derived from CPU capabilities.

use crate::config::ModelConfig;
use super::caps::CpuCapabilities;

/// Derived prefill batching configuration.
///
/// Contains optimal batch size and core count for prefill
/// based on detected hardware capabilities and model dimensions.
pub struct BatchConfig {
    /// Maximum tokens to process in one batch.
    ///
    /// This value is chosen so that the working set for a batch
    /// fits within L3 cache (80% of L3, to leave room for OS/other processes).
    pub max_tokens_per_batch: usize,
    /// Number of cores to use for parallelism.
    ///
    /// Always uses physical cores only, not hyperthreading.
    /// Hyperthreading typically does not improve pure compute performance
    /// and can increase contention.
    pub num_cores: usize,
}

impl BatchConfig {
    /// Derive batch config from CPU capabilities and model configuration.
    ///
    /// Uses L3 cache size to determine maximum tokens per batch
    /// and physical core count for parallelism.
    pub fn from_capabilities(
        caps: &CpuCapabilities,
        config: &ModelConfig,
    ) -> Self {
        let mem_per_tok = memory_per_token(config);

        // Determine max tokens per batch based on L3 cache
        let max_batch = if caps.has_l3_cache() {
            // Use 80% of L3 to leave room for OS/other processes
            let usable_l3 = caps.l3_cache_bytes * 8 / 10;
            usable_l3 / mem_per_tok
        } else {
            // Fallback: no L3 info, assume 4MB per batch
            (4 * 1024 * 1024) / mem_per_tok
        };

        // Clamp to sensible range: at least 1, at most 256
        let max_tokens_per_batch = max_batch.max(1).min(256);

        // Use physical cores only for compute
        let num_cores = caps.compute_cores();

        Self {
            max_tokens_per_batch,
            num_cores,
        }
    }
}

/// Compute memory footprint per token during prefill.
///
/// This estimates the memory required to process a single token
/// in the prefill phase, considering:
/// - Activation buffers (reused across layers)
/// - KV cache writes (per layer, K and V for this position)
///
/// The calculation is based on the ModelConfig dimensions and
/// the buffer layout used in `cpu_prefill_forward()`.
pub fn memory_per_token(config: &ModelConfig) -> usize {
    let h = config.hidden_size;
    let q = config.num_heads * config.head_dim;
    let kv = config.num_kv_heads * config.head_dim;
    let ff = config.intermediate_size;

    // Activation buffers (reused across layers, f32 each)
    // From CpuPrefillScratch in src/cpu/prefill.rs:
    // - hidden: h
    // - normed: h
    // - q: q
    // - k: kv
    // - v: kv
    // - attn_out: q
    // - layer_out: h
    // - gate: ff
    // - swiglu: ff
    let activations = (h * 3 + q * 2 + kv * 2 + ff * 2) * std::mem::size_of::<f32>();

    // KV cache writes (per layer, K and V for this position)
    // Each layer writes K and V of size kv each
    let kv_write_per_layer = kv * 2 * std::mem::size_of::<f32>();
    let kv_write_total = kv_write_per_layer * config.num_layers;

    activations + kv_write_total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 24,
            hidden_size: 896,
            num_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 4864,
            vocab_size: 151936,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_neox: true,
            use_attention_bias: true,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "qwen2".to_string(),
        }
    }

    fn make_test_caps() -> CpuCapabilities {
        use crate::hardware::SimdFeatures;
        CpuCapabilities {
            physical_cores: 8,
            logical_cpus: 16,
            l3_cache_bytes: 96 * 1024 * 1024,
            l2_cache_bytes: 2 * 1024 * 1024,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            simd: SimdFeatures::default(),
        }
    }

    #[test]
    fn memory_per_token_is_positive() {
        let config = make_test_config();
        let bytes = memory_per_token(&config);

        assert!(bytes > 0);
    }

    #[test]
    fn memory_per_token_matches_scratch_buffers() {
        let config = make_test_config();
        let bytes = memory_per_token(&config);

        // Manually calculate based on CpuPrefillScratch structure
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let f32_size = std::mem::size_of::<f32>();

        let activations = (h * 3 + q * 2 + kv * 2 + ff * 2) * f32_size;
        let kv_write = kv * 2 * f32_size * config.num_layers;

        assert_eq!(bytes, activations + kv_write);
    }

    #[test]
    fn batch_config_clamps_to_sensible_range() {
        let caps = make_test_caps();
        let config = make_test_config();
        let batch = BatchConfig::from_capabilities(&caps, &config);

        // Should be at least 1
        assert!(batch.max_tokens_per_batch >= 1);
        // Should be at most 256
        assert!(batch.max_tokens_per_batch <= 256);
        // Should use physical cores only
        assert_eq!(batch.num_cores, 8);
    }

    #[test]
    fn batch_config_with_no_l3_uses_fallback() {
        let mut caps = make_test_caps();
        caps.l3_cache_bytes = 0; // No L3 info

        let config = make_test_config();
        let batch = BatchConfig::from_capabilities(&caps, &config);

        // Should still have a valid batch size (fallback 4MB)
        assert!(batch.max_tokens_per_batch >= 1);
    }

    #[test]
    fn batch_config_derives_from_actual_l3() {
        let caps = make_test_caps();
        let config = make_test_config();

        let mem_per_tok = memory_per_token(&config);
        let expected_max = (caps.l3_cache_bytes * 8 / 10) / mem_per_tok;

        let batch = BatchConfig::from_capabilities(&caps, &config);

        // Should be based on 80% of L3, clamped to 256
        let clamped_expected = expected_max.min(256).max(1);
        assert_eq!(batch.max_tokens_per_batch, clamped_expected);
    }

    #[test]
    fn batch_config_uses_physical_cores() {
        let mut caps = make_test_caps();
        caps.logical_cpus = 32; // 16 hyperthreads per core
        caps.physical_cores = 2;  // Only 2 actual cores

        let config = make_test_config();
        let batch = BatchConfig::from_capabilities(&caps, &config);

        // Should use physical cores, not logical
        assert_eq!(batch.num_cores, 2);
    }

    #[test]
    fn memory_per_token_scales_with_model_size() {
        let base_config = make_test_config();
        let base_bytes = memory_per_token(&base_config);

        // Double the hidden size
        let mut larger_config = base_config.clone();
        larger_config.hidden_size *= 2;
        larger_config.num_heads *= 2; // Maintain head_dim

        let larger_bytes = memory_per_token(&larger_config);

        // Memory per token should roughly double
        // (not exactly double due to other dimensions)
        assert!(larger_bytes > base_bytes);
    }
}
