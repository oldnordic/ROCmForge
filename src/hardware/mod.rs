//! CPU hardware detection and batch configuration.
//!
//! This module provides automatic detection of CPU capabilities
//! and derivation of optimal prefill batch sizes.
//!
//! ## Usage
//!
//! ```no_run
//! use rocmforge::hardware::{CpuCapabilities, BatchConfig, detect, derive_batch_config};
//!
//! // Detect hardware capabilities
//! let caps = detect().unwrap();
//!
//! // Derive batch config from hardware + model config
//! let batch_config = derive_batch_config(&caps, &model_config);
//!
//! println!("Physical cores: {}", caps.physical_cores);
//! println!("L3 cache: {} MB", caps.l3_cache_mb());
//! println!("Max tokens/batch: {}", batch_config.max_tokens_per_batch);
//! ```

mod error;
mod caps;
mod config;

pub use error::HardwareError;
pub use caps::{CpuCapabilities, SimdFeatures};
pub use config::{BatchConfig, memory_per_token};

/// Detect CPU hardware capabilities.
///
/// This is a convenience function that calls [`CpuCapabilities::detect()`].
/// Returns detected CPU topology, cache sizes, and memory.
pub fn detect() -> Result<CpuCapabilities, HardwareError> {
    CpuCapabilities::detect()
}

/// Derive prefill batch configuration from capabilities.
///
/// This is a convenience function that calls [`BatchConfig::from_capabilities()`].
/// Returns optimal batch size and core count based on L3 cache.
pub fn derive_batch_config(
    caps: &CpuCapabilities,
    config: &crate::config::ModelConfig,
) -> BatchConfig {
    BatchConfig::from_capabilities(caps, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_convenience_function_works() {
        let caps = detect().unwrap();
        assert!(caps.physical_cores > 0);
    }

    #[test]
    fn derive_batch_config_convenience_works() {
        let caps = CpuCapabilities {
            physical_cores: 8,
            logical_cpus: 16,
            l3_cache_bytes: 96 * 1024 * 1024,
            l2_cache_bytes: 2 * 1024 * 1024,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            simd: SimdFeatures::default(),
        };

        let config = crate::config::ModelConfig {
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
        };

        let batch = derive_batch_config(&caps, &config);

        assert!(batch.max_tokens_per_batch >= 1);
        assert_eq!(batch.num_cores, 8);
    }
}
