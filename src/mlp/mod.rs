//! MLP (Multi-Layer Perceptron) operations for ROCmForge
//!
//! Implements GPU kernels for MLP components:
//! - SwiGLU activation
//! - RMSNorm normalization

#[cfg(feature = "rocm")]
pub mod kernels;

// Phase 4.1: SwiGLU GPU kernel tests
#[cfg(test)]
#[cfg(feature = "rocm")]
mod swiglu_tests;

// Phase 4.2: RMSNorm GPU kernel tests
#[cfg(test)]
#[cfg(feature = "rocm")]
mod rms_norm_tests;

// Phase 4.3: GPU path regression tests (no host round-trip)
#[cfg(test)]
#[cfg(feature = "rocm")]
mod gpu_path_regression_tests;
