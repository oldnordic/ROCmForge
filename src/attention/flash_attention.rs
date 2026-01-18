//! Flash Attention backend implementation
//!
//! This module provides a flash attention backend that implements the
//! BackendImplementation trait. Flash attention is an IO-aware exact attention
//! algorithm that reduces memory bandwidth and improves performance through
//! kernel fusion.
//!
//! Based on research from Phase 06-01, flash attention provides:
//! - 2-4x speedup for typical inference workloads
//! - Reduced memory bandwidth (no attention matrix materialization)
//! - Single kernel launch vs 5+ separate operations

use crate::attention::{
    backend_registry::{BackendImplementation, AttentionConfig, AttentionBackendResult, KvCacheLayout},
    cpu,
};

#[cfg(feature = "rocm")]
use crate::backend::{HipBuffer, HipBlasHandle};
#[cfg(feature = "rocm")]
use crate::backend::hip_backend::synchronize_device;

/// Maximum head dimension supported by flash attention kernels
/// Due to register storage limits in the kernels
const MAX_FLASH_HEAD_DIM: usize = 128;

/// Maximum sequence length supported by flash attention kernels
/// Due to shared memory constraints
const MAX_FLASH_SEQ_LEN: usize = 2048;

/// Flash Attention backend implementation
///
/// This backend uses fused flash attention kernels when available,
/// providing significant performance improvements over traditional
/// multi-kernel attention computation.
pub struct FlashAttentionBackend {
    #[cfg(feature = "rocm")]
    _handle: HipBlasHandle,
}

impl FlashAttentionBackend {
    /// Create a new FlashAttention backend
    ///
    /// Returns an error if the ROCm feature is not enabled.
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "rocm")]
        {
            let handle = HipBlasHandle::new()
                .map_err(|e| format!("Failed to create HIP BLAS handle: {}", e))?;
            Ok(FlashAttentionBackend { _handle: handle })
        }

        #[cfg(not(feature = "rocm"))]
        {
            Err("FlashAttention backend requires 'rocm' feature".to_string())
        }
    }

    /// Check if flash attention can be used for the given configuration
    pub fn can_use_flash_attention(config: &AttentionConfig) -> bool {
        cfg!(feature = "rocm")
            && config.head_dim <= MAX_FLASH_HEAD_DIM
            && config.seq_len <= config.max_sequence_length
            && config.max_sequence_length <= MAX_FLASH_SEQ_LEN
    }

    /// Check if flash attention can be used with the given mask
    pub fn supports_mask(config: &AttentionConfig, has_mask: bool) -> bool {
        if !has_mask {
            return true; // No mask is always supported
        }

        // Causal masking uses dedicated kernel (supported)
        // Custom masks require generic kernel (not yet supported)
        config.is_causal
    }
}

impl Default for FlashAttentionBackend {
    fn default() -> Self {
        #[cfg(feature = "rocm")]
        {
            Self::new().expect("Failed to create FlashAttention backend")
        }

        #[cfg(not(feature = "rocm"))]
        {
            panic!("FlashAttention backend requires 'rocm' feature");
        }
    }
}

impl BackendImplementation for FlashAttentionBackend {
    fn name(&self) -> &str {
        "flash_attention"
    }

    fn supports(&self, config: &AttentionConfig) -> bool {
        // Flash attention is supported when:
        // - ROCm feature is enabled
        // - Sequence length is within bounds
        // - Head dimension is compatible with shared memory
        Self::can_use_flash_attention(config)
    }

    fn required_kv_layout(&self) -> Option<KvCacheLayout> {
        // Flash attention works best with FlashAttention-optimized layout
        Some(KvCacheLayout::FlashAttention)
    }

    fn forward(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>> {
        // Validate config
        config.validate()
            .map_err(|e| crate::attention::backend_registry::AttentionBackendError::NotSupported(e))?;

        // For now, we use the CPU implementation with a note that
        // GPU kernel integration will happen in 06-03
        // This allows the backend to be registered and tested first
        #[cfg(feature = "rocm")]
        {
            // TODO: In 06-03, replace this with actual GPU kernel calls
            // For now, use CPU as reference implementation
            let result = cpu::CpuBackend::forward(config.dim, q, k, v, mask, config.dropout)
                .map_err(|e| crate::attention::backend_registry::AttentionBackendError::OperationFailed(
                    format!("CPU fallback failed: {}", e)
                ))?;

            // Note: This is a placeholder. The actual flash kernels will be
            // integrated in phase 06-03 (Flash attention kernel integration)
            Ok(result)
        }

        #[cfg(not(feature = "rocm"))]
        {
            Err(crate::attention::backend_registry::AttentionBackendError::NotSupported(
                "FlashAttention requires 'rocm' feature".to_string()
            ))
        }
    }
}

#[cfg(feature = "rocm")]
impl FlashAttentionBackend {
    /// Forward pass using flash attention causal kernel
    ///
    /// This will be implemented in phase 06-03 with actual kernel calls.
    fn forward_causal(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        _mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>> {
        // Placeholder for 06-03 implementation
        // Will call flash_attention_causal_gpu_kernel from kernels.rs
        cpu::CpuBackend::forward(config.dim, q, k, v, None, config.dropout)
            .map_err(|e| crate::attention::backend_registry::AttentionBackendError::OperationFailed(e.to_string()))
    }

    /// Forward pass using flash attention non-causal kernel
    ///
    /// This will be implemented in phase 06-03 with actual kernel calls.
    fn forward_nocausal(
        &self,
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        _mask: Option<&[f32]>,
    ) -> AttentionBackendResult<Vec<f32>> {
        // Placeholder for 06-03 implementation
        // Will call flash_attention_nocausal_gpu_kernel from kernels.rs
        cpu::CpuBackend::forward(config.dim, q, k, v, None, config.dropout)
            .map_err(|e| crate::attention::backend_registry::AttentionBackendError::OperationFailed(e.to_string()))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "rocm")]
    fn test_flash_attention_backend_creation() {
        let backend = FlashAttentionBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    #[cfg(not(feature = "rocm"))]
    fn test_flash_attention_backend_fails_without_rocm() {
        let result = FlashAttentionBackend::new();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("rocm"));
    }

    #[test]
    fn test_can_use_flash_attention_valid_config() {
        let config = AttentionConfig::new(512, 8, 64)
            .with_max_sequence_length(1024);

        assert!(FlashAttentionBackend::can_use_flash_attention(&config));
    }

    #[test]
    fn test_can_use_flash_attention_head_dim_too_large() {
        let config = AttentionConfig::new(512, 4, 129) // head_dim > 128
            .with_max_sequence_length(1024);

        assert!(!FlashAttentionBackend::can_use_flash_attention(&config));
    }

    #[test]
    fn test_can_use_flash_attention_seq_len_too_large() {
        let config = AttentionConfig::new(512, 8, 64)
            .with_max_sequence_length(2049); // > 2048

        assert!(!FlashAttentionBackend::can_use_flash_attention(&config));
    }

    #[test]
    fn test_supports_mask_causal() {
        let config = AttentionConfig::new(512, 8, 64)
            .with_causal(true);

        assert!(FlashAttentionBackend::supports_mask(&config, true));
    }

    #[test]
    fn test_supports_mask_no_mask() {
        let config = AttentionConfig::new(512, 8, 64)
            .with_causal(false);

        assert!(FlashAttentionBackend::supports_mask(&config, false));
    }

    #[test]
    fn test_supports_mask_custom_mask_not_supported() {
        // Custom masks (non-causal) are not yet supported
        let config = AttentionConfig::new(512, 8, 64)
            .with_causal(false);

        assert!(!FlashAttentionBackend::supports_mask(&config, true));
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_backend_name() {
        let backend = FlashAttentionBackend::new().unwrap();
        assert_eq!(backend.name(), "flash_attention");
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_backend_supports_valid_config() {
        let backend = FlashAttentionBackend::new().unwrap();
        let config = AttentionConfig::new(512, 8, 64)
            .with_max_sequence_length(1024);

        assert!(backend.supports(&config));
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_backend_does_not_support_invalid_config() {
        let backend = FlashAttentionBackend::new().unwrap();
        let config = AttentionConfig::new(512, 4, 129) // head_dim too large
            .with_max_sequence_length(1024);

        assert!(!backend.supports(&config));
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_backend_required_kv_layout() {
        let backend = FlashAttentionBackend::new().unwrap();
        assert_eq!(
            backend.required_kv_layout(),
            Some(KvCacheLayout::FlashAttention)
        );
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_backend_forward_simple() {
        let backend = FlashAttentionBackend::new().unwrap();
        let config = AttentionConfig::new(16, 4, 4)
            .with_max_sequence_length(16);

        // Simple test data: batch=1, seq=16, dim=16
        // Total size = 1 * 16 * 16 = 256
        let q = vec![0.1f32; 256];
        let k = vec![0.2f32; 256];
        let v = vec![0.3f32; 256];

        let result = backend.forward(&config, &q, &k, &v, None);
        assert!(result.is_ok(), "Forward failed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(output.len(), q.len());
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_backend_forward_with_mask() {
        let backend = FlashAttentionBackend::new().unwrap();
        let config = AttentionConfig::new(16, 4, 4)
            .with_causal(true)
            .with_max_sequence_length(16);

        let q = vec![0.1f32; 256];
        let k = vec![0.2f32; 256];
        let v = vec![0.3f32; 256];
        let mask = vec![0.0f32; 16 * 16]; // seq_len x seq_len

        let result = backend.forward(&config, &q, &k, &v, Some(&mask));
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), q.len());
    }
}
