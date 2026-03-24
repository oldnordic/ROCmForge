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

pub mod error;
pub mod weights;
pub mod cache;
pub mod ops;
pub mod quant;
pub mod forward;
pub mod prefill;
pub mod sampler;

#[cfg(test)]
mod ops_tests;

use crate::config::ModelConfig;
use crate::loader::{GgmlType, GgufFile, LoadError};

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
