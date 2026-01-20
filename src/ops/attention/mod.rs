//! GPU Attention Operations Layer
//!
//! High-level attention operations for model execution using GPU kernels.
//! Provides orchestration of the full attention pipeline.

pub mod kernels;
pub mod softmax;

#[cfg(feature = "rocm")]
pub mod hiprtc;

// Re-export public API
pub use kernels::HipAttentionKernels;
pub use softmax::{AttentionSoftmax, CausalMaskOp, QkMatmul, WeightedMatmul};
