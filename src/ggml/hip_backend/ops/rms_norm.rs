//! HIP RMSNorm op using existing RMSNorm kernel.
//!
//! This module now re-exports from the kernels::element module.

// Re-export from kernels::element for backward compatibility
pub use crate::kernels::element::rms_norm;
