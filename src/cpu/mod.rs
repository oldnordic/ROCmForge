//! CPU inference backend — pure Rust with no GPU dependencies.

//!
//! Architecture:
//! - `CpuModelWeights` — holds quantized weights from GGUF
//! - `CpuLayerWeights` — per-layer attention + FFN weights
//! - `CpuKvCache` — key-value cache for autoregressive decoding
//! - `CpuForwardScratch` — reusable scratch buffers for inference
//! - All ops are pure f32 operations (no SIMD by default)
//!
//! Design principles:
//! - TDD: write failing tests first, then implement
//! - <1K LOC per file
//! - Metadata-driven: all shapes from ModelConfig, no model-specific code
//! - Explicit device selection: CPU path has no GPU fallback

pub mod features;
pub mod simd;
pub mod kernels;
pub mod error;
pub mod transpose;
pub mod weights;
pub mod cache;
pub mod ops;
pub mod quant;
pub mod forward;
pub mod prefill;
pub mod sampler;

// Re-export hardware for convenience
pub use crate::hardware::{CpuCapabilities, BatchConfig, detect, derive_batch_config};

// Re-export CPU features
pub use self::features::{CpuFeatures, KernelPreference};
// Re-export SIMD kernel system
pub use self::simd::{SimdKernels, SimdActivations};
// Re-export Q4_K kernels
pub use self::kernels::gemm_q4k_q8::{gemv_q4_k_q8_k_dispatch, gemm_q4_k_q8_k_dispatch_gemm};

#[cfg(test)]
mod ops_tests;

use crate::loader::{GgmlType, LoadError};

// ── Error ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CpuError {
    UnsupportedWeightType(GgmlType),
    DimensionMismatch(&'static str),
    InvalidOperation(String),
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuError::UnsupportedWeightType(t) => {
                write!(f, "unsupported weight type for CPU backend: {:?}", t)
            }
            CpuError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
            CpuError::InvalidOperation(msg) => write!(f, "invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for CpuError {}

impl From<LoadError> for CpuError {
    fn from(e: LoadError) -> Self {
        CpuError::InvalidOperation(e.to_string())
    }
}
