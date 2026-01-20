//! GPU kernel functions for attention operations.
//!
//! This module provides Rust wrappers for HIP kernels that implement
//! core attention operations on GPU.

#![allow(non_snake_case)] // Kernel parameter names follow HIP conventions

// Private sub-modules for kernel organization
// kernels_basic and kernels_flash are submodules of kernels_cache
// to access private KernelCache fields
mod kernels_cache;

// Re-export all kernel functions (only available with rocm feature)
#[cfg(feature = "rocm")]
pub use kernels_cache::kernels_basic::{
    mask_gpu_kernel,
    position_embeddings_gpu_kernel,
    rope_gpu_kernel,
    scale_gpu_kernel,
    softmax_gpu_kernel,
};
#[cfg(feature = "rocm")]
pub use kernels_cache::kernels_flash::{
    causal_mask_gpu_kernel,
    flash_attention_causal_gpu_kernel,
    flash_attention_gpu_kernel,
    flash_attention_nocausal_gpu_kernel,
    mqa_kv_replicate_gpu_kernel,
    qkt_matmul_gpu_kernel,
    qkt_matmul_gpu_kernel_scaled,
    weighted_matmul_gpu_kernel,
};
