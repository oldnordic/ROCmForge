//! HIP SwiGLU op using existing SwiGLU kernel.
//!
//! This module now re-exports from the kernels::element module.

// Re-export from kernels::element for backward compatibility
pub use crate::kernels::element::swiglu;
