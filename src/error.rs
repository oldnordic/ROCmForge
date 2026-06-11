use std::fmt;

use crate::cpu::CpuError;
use crate::gpu::error::GpuError;
use crate::loader::LoadError;
use crate::cpu::weights::WeightError;

/// The unified error type for the rocmforge inference engine.
#[derive(Debug)]
pub enum RocmForgeError {
    /// An error occurred during GPU operations (kernels, VRAM, HIP API).
    Gpu(GpuError),
    /// An error occurred during CPU operations (kernels, threading, memory).
    Cpu(CpuError),
    /// An error occurred while loading weights or parsing model structures.
    Weight(WeightError),
    /// An error occurred while parsing the initial model file format.
    Load(LoadError),
    /// A generic or application-level orchestration error.
    /// Used during the transition away from String errors.
    Generic(String),
}

impl fmt::Display for RocmForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RocmForgeError::Gpu(e) => write!(f, "GPU error: {}", e),
            RocmForgeError::Cpu(e) => write!(f, "CPU error: {}", e),
            RocmForgeError::Weight(e) => write!(f, "Weight error: {}", e),
            RocmForgeError::Load(e) => write!(f, "Load error: {}", e),
            RocmForgeError::Generic(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for RocmForgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RocmForgeError::Gpu(e) => Some(e),
            RocmForgeError::Cpu(e) => Some(e),
            RocmForgeError::Weight(e) => Some(e),
            RocmForgeError::Load(e) => Some(e),
            RocmForgeError::Generic(_) => None,
        }
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<GpuError> for RocmForgeError {
    fn from(err: GpuError) -> Self {
        RocmForgeError::Gpu(err)
    }
}

impl From<CpuError> for RocmForgeError {
    fn from(err: CpuError) -> Self {
        RocmForgeError::Cpu(err)
    }
}

impl From<WeightError> for RocmForgeError {
    fn from(err: WeightError) -> Self {
        RocmForgeError::Weight(err)
    }
}

impl From<LoadError> for RocmForgeError {
    fn from(err: LoadError) -> Self {
        RocmForgeError::Load(err)
    }
}

impl From<String> for RocmForgeError {
    fn from(msg: String) -> Self {
        RocmForgeError::Generic(msg)
    }
}

impl From<&str> for RocmForgeError {
    fn from(msg: &str) -> Self {
        RocmForgeError::Generic(msg.to_string())
    }
}

pub type RocmForgeResult<T> = Result<T, RocmForgeError>;
