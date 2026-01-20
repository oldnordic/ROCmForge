//! Operations module for GPU computations

pub mod attention;
pub mod qkv;

// Re-export for backward compatibility
pub use attention::{
    AttentionSoftmax, CausalMaskOp, HipAttentionKernels, QkMatmul, WeightedMatmul,
};

// Backward compatibility: re-export attention_gpu module as alias
// This allows existing code using `crate::ops::attention_gpu` to continue working
pub use attention as attention_gpu;
