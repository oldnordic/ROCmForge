//! Attention kernel implementations
//!
//! Provides CPU and GPU kernels for attention operations:
//! - Causal mask generation and application
//! - Softmax computation with numerical stability
//! - QK^T matrix multiplication for attention scores
//! - Attention-weighted value aggregation
//! - FlashAttention-style fused operations (causal and non-causal)
//! - Rotary Positional Embeddings (RoPE)

pub mod mask;
pub mod matmul;
pub mod flash;
pub mod rope;
pub mod softmax;

// Re-export all public functions for convenience

// Mask exports
pub use mask::{
    apply_mask_in_place,
    apply_mask_with_fallback,
    create_causal_mask,
};

// Matmul exports
pub use matmul::{
    qkt_matmul_cpu,
    qkt_matmul_with_fallback,
    weighted_matmul_cpu,
    weighted_matmul_with_fallback,
};

// FlashAttention exports
pub use flash::{
    can_use_flash_attention,
    create_causal_mask as flash_create_causal_mask,
    supports_causal_mask,
    FlashAttentionBackend,
    MAX_FLASH_HEAD_DIM,
    MAX_FLASH_SEQ_LEN,
};

// RoPE exports
pub use rope::{
    rope_with_fallback,
    Rope,
    RopeConfig,
};

// Softmax exports
pub use softmax::{
    softmax_in_place,
    softmax_in_place_cpu,
};

// ROCm-specific exports (always available)
pub use mask::{
    hip_mask_op,
    mask_gpu_kernel,
};

pub use matmul::{
    qkt_matmul_gpu_kernel,
    qkt_matmul_gpu_kernel_scaled,
    weighted_matmul_gpu_kernel,
};

pub use flash::{
    causal_mask_gpu_kernel,
    flash_attention_causal_gpu_kernel,
    flash_attention_gpu_kernel,
    flash_attention_nocausal_gpu_kernel,
};

pub use rope::{
    hip_rope_op,
    rope_gpu_kernel,
};

pub use softmax::softmax_with_fallback;
