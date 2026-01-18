//! ROCm/HIP backend module

#[cfg(feature = "simd")]
pub mod cpu;
pub mod gpu_executor;
pub mod gpu_test_common;
pub mod hip_backend;
pub mod hip_blas;
pub mod scratch;

#[cfg(feature = "simd")]
pub use cpu::*;
pub use gpu_executor::*;
#[cfg(test)]
pub use gpu_test_common::*;
pub use hip_backend::*;
pub use hip_blas::*;
pub use scratch::*;
