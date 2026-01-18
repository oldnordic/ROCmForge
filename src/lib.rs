//! ROCmForge - AMD GPU Inference Engine
//!
//! A high-performance inference engine for Large Language Models
//! specifically designed for AMD GPUs using ROCm and HIP.

#![cfg_attr(feature = "simd", feature(portable_simd))] // For std::simd CPU backend (Rust 1.82+)

#![allow(clippy::too_many_arguments)] // Many FFI functions and kernel launches need many args
#![allow(clippy::manual_slice_size_calculation)] // Common in GPU kernel code
#![allow(clippy::needless_range_loop)] // Clearer for GPU operations
#![allow(clippy::collapsible_else_if)] // Sometimes clearer for control flow
#![allow(clippy::collapsible_if)] // Sometimes clearer for control flow
#![allow(clippy::bool_comparison)] // Sometimes clearer for intent
#![allow(clippy::let_and_return)] // Sometimes clearer for debugging
#![allow(clippy::clone_on_copy)] // Sometimes needed for API clarity
#![allow(clippy::type_complexity)] // Complex types are common in ML
#![allow(clippy::missing_safety_doc)] // FFI bindings documented at module level
#![allow(clippy::bool_to_int_with_if)] // Sometimes clearer for intent
#![allow(clippy::if_same_then_else)] // Sometimes clearer for future expansion
#![allow(clippy::redundant_clone)] // Sometimes needed for API compatibility
#![allow(clippy::manual_memcpy)] // GPU memory operations often manual

pub mod attention;
pub mod backend;
pub mod engine;
pub mod error;
pub mod http;
pub mod ggml;
pub mod kv_cache;
pub mod loader;
pub mod mlp;
pub mod model;
pub mod models;
pub mod ops;
pub mod profiling;
pub mod prompt;
pub mod sampler;
pub mod scheduler;
pub mod tensor;
pub mod tokenizer;

#[cfg(test)]
mod hip_backend_debug_tests;
#[cfg(test)]
mod hip_isolation_test;

pub use attention::Attention;
pub use backend::HipBackend;
pub use engine::InferenceEngine;
pub use error::{ErrorCategory, ForgeResult, RocmForgeError};
pub use kv_cache::KvCache;
pub use profiling::{KernelTimer, ScopedTimer};
pub use sampler::Sampler;
pub use scheduler::Scheduler;
pub use tensor::Tensor;

// Public test utilities for integration testing
#[cfg(test)]
pub use backend::gpu_test_common::*;

#[cfg(test)]
mod library_tests {
    use super::*;

    #[test]
    fn test_library_imports() {
        // Basic smoke test to ensure all modules compile
        assert!(true);
    }
}
