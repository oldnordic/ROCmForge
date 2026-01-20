//! Re-export of matmul from kernels::matmul for backward compatibility
//!
//! This module re-exports the matmul function from the centralized
//! kernels::matmul module to maintain backward compatibility with existing code.

pub use crate::kernels::matmul::matmul;
