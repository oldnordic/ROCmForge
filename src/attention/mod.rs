//! Attention mechanism for ROCmForge
//! Implements Scaled Dot-Product Attention

pub mod backend;
pub mod backend_registry;
pub mod compare;
pub mod compute;
pub mod cpu;
pub mod flash_attention;
pub mod gpu;
pub mod kernels;
pub mod mask;
pub mod multi_query;
pub mod paged_kernel;
pub mod rope;
pub mod softmax;

// Phase 1 kernel tests (CPU vs GPU comparison)
#[cfg(test)]
mod kernel_tests;

// Phase 2 RoPE GPU tests
#[cfg(test)]
mod rope_gpu_tests;

// Phase 3 FlashAttention tests
#[cfg(test)]
mod flash_attention_tests;

// Phase 3a.1 QK^T matmul tests (divide & conquer)
#[cfg(test)]
mod qkt_matmul_tests;

// Phase 3a.3.1 Softmax with explicit layout tests
#[cfg(test)]
mod softmax_explicit_tests;

// Phase 3a.4 Weighted × V matmul tests
#[cfg(test)]
mod weighted_matmul_tests;

// Phase 3a.5 Fused non-causal FlashAttention tests
#[cfg(test)]
mod flash_nocausal_tests;

// Phase 3b.1 Causal mask tests (standalone, before fusion)
#[cfg(test)]
mod causal_mask_tests;

// Phase 3b.2 Fused causal FlashAttention tests
#[cfg(test)]
mod flash_causal_tests;

// Phase 3: Paged Attention tests
#[cfg(test)]
mod paged_tests;

// Phase 19.2: MQA KV Replication Kernel Integration tests
#[cfg(test)]
mod mqa_kernel_tests;

use crate::backend::{DeviceTensor, HipBackend};
pub use backend::AttentionBackend;
// Re-export backend registry types for public API
// Note: The pluggable backend trait is now named BackendImplementation
// to distinguish it from the AttentionBackend enum used for simple selection
pub use backend_registry::{
    AttentionBackendError, AttentionBackendRegistry, AttentionBackendResult, AttentionConfig,
    BackendImplementation, KvCacheLayout,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Dimension error: {0}")]
    DimensionError(String),
    #[error("GPU memory allocation failed: {0}")]
    MemoryAllocation(String),
    #[error("GPU memory copy failed: {0}")]
    MemoryCopy(String),
    #[error("GPU operation failed: {0}")]
    GpuOperation(String),
    #[error("Handle/resource creation failed: {0}")]
    HandleCreation(String),
    #[error("GPU synchronization failed: {0}")]
    Synchronization(String),
}

pub type AttentionResult<T> = Result<T, AttentionError>;

pub struct Attention {
    pub dim: usize,
    pub backend: AttentionBackend,
}

impl Attention {
    pub fn new(dim: usize) -> Self {
        Attention {
            dim,
            backend: AttentionBackend::default(),
        }
    }

    pub fn with_backend(dim: usize, backend: AttentionBackend) -> Self {
        Attention { dim, backend }
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        dropout: Option<f32>,
    ) -> AttentionResult<Vec<f32>> {
        println!(
            "DEBUG: Attention::forward called with backend: {:?}",
            self.backend
        );
        println!(
            "DEBUG: Input lengths - q: {}, k: {}, v: {}",
            q.len(),
            k.len(),
            v.len()
        );
        match self.backend {
            AttentionBackend::Cpu => {
                println!("DEBUG: Using CPU backend");
                cpu::CpuBackend::forward(self.dim, q, k, v, mask, dropout)
            }
            AttentionBackend::Gpu => {
                println!("DEBUG: Using GPU backend");
                gpu::GpuBackend::forward(self.dim, q, k, v, mask, dropout)
            }
        }
    }

    /// Forward pass with DeviceTensor inputs for zero-copy GPU computation
    pub fn forward_device(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        dropout: Option<f32>,
    ) -> AttentionResult<DeviceTensor> {
        match self.backend {
            AttentionBackend::Cpu => {
                // Fallback to CPU implementation by copying to host
                let backend = HipBackend::new().map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to create HIP backend: {}", e))
                })?;

                let mut q_host = vec![0.0f32; q.len()];
                backend.copy_from_device_safe(q.buffer(), &mut q_host).map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to copy Q to host: {}", e))
                })?;
                let mut k_host = vec![0.0f32; k.len()];
                backend.copy_from_device_safe(k.buffer(), &mut k_host).map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to copy K to host: {}", e))
                })?;
                let mut v_host = vec![0.0f32; v.len()];
                backend.copy_from_device_safe(v.buffer(), &mut v_host).map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to copy V to host: {}", e))
                })?;
                let mask_host = mask
                    .map(|m| {
                        let mut data = vec![0.0f32; m.len()];
                        backend.copy_from_device_safe(m.buffer(), &mut data).map_err(|e| {
                            AttentionError::DimensionError(format!(
                                "Failed to copy mask to host: {}",
                                e
                            ))
                        })?;
                        Ok(data)
                    })
                    .transpose()?;

                let output = cpu::CpuBackend::forward(
                    self.dim,
                    &q_host,
                    &k_host,
                    &v_host,
                    mask_host.as_deref(),
                    dropout,
                )?;
                println!(
                    "DEBUG: CPU returned output with {} elements: {:?}",
                    output.len(),
                    output
                );

                // Create output DeviceTensor with same shape as input V tensor
                let output_shape = v.shape().clone(); // Use same shape as V tensor
                println!(
                    "DEBUG: CPU returned output with {} elements: {:?}",
                    output.len(),
                    output
                );
                println!("DEBUG: About to call DeviceTensor::from_host_vec...");
                let result = DeviceTensor::from_host_vec(&backend, output, output_shape);
                match result {
                    Ok(tensor) => {
                        println!(
                            "DEBUG: Created output tensor: len() = {}, size() = {}, shape = {:?}",
                            tensor.len(),
                            tensor.size(),
                            tensor.shape()
                        );
                        Ok(tensor)
                    }
                    Err(e) => {
                        println!("DEBUG: Failed to create output tensor: {}", e);
                        Err(AttentionError::DimensionError(format!(
                            "Failed to create output tensor: {}",
                            e
                        )))
                    }
                }
            }
            AttentionBackend::Gpu => {
                gpu::GpuBackend::forward_device(self.dim, q, k, v, mask, dropout)
            }
        }
    }
}
