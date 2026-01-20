//! MLP (Multi-Layer Perceptron) operations for ROCmForge
//!
//! Implements GPU kernels for MLP components:
//! - SwiGLU activation
//! - RMSNorm normalization

pub mod kernels;

// Phase 4.1: SwiGLU GPU kernel tests
#[cfg(test)]
mod swiglu_tests;

// Phase 4.2: RMSNorm GPU kernel tests
#[cfg(test)]
mod rms_norm_tests;

// Phase 4.3: GPU path regression tests (no host round-trip)
#[cfg(test)]
mod gpu_path_regression_tests;
