//! Element-wise kernel implementations
//!
//! Provides CPU and GPU kernels for element-wise operations:
//! - RMSNorm: Root Mean Square Layer Normalization
//! - SwiGLU: Swish-Gated Linear Unit activation
//! - Scale: Tensor scaling operations
//! - Add, mul, and other basic arithmetic

pub mod rms_norm;
pub mod scale;
pub mod swiglu;

// Public exports for element-wise operations
pub use rms_norm::rms_norm;
pub use scale::{add, scale};
pub use swiglu::swiglu;
