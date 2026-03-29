//! GPU error types with safe error handling.
//!
//! Design principle: Never panic, always return errors gracefully.
//! All GPU errors should be recoverable (CPU fallback).

use crate::loader::GgmlType;

/// Errors that can occur during GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// HIP runtime not available (rocm不是 not installed or incompatible version)
    HipNotAvailable,

    /// No AMD GPU detected on system
    NoDeviceFound,

    /// HIP API call failed with error code and description
    HipApiError { code: i32, description: String },

    /// Out of GPU memory (VRAM exhausted)
    OutOfMemory {
        /// Requested allocation size in bytes
        requested: usize,
        /// Available VRAM in bytes
        available: usize,
    },

    /// Kernel launch failed (compilation or execution error)
    KernelLaunchFailed { kernel: String },

    /// Invalid device ID (requested device doesn't exist)
    InvalidDevice { device_id: i32 },

    /// Weight transfer from CPU to GPU failed
    WeightTransferFailed { layer: usize },

    /// KV cache allocation failed
    CacheAllocationFailed { reason: String },

    /// GPU backend does not implement this GGUF weight format yet
    UnsupportedWeightType { tensor: String, wtype: GgmlType },

    /// Matrix weight metadata is invalid for GPU inference
    InvalidWeightLayout {
        tensor: String,
        dims: Vec<u64>,
        reason: String,
    },

    /// Model does not fit in available VRAM
    ModelTooLarge { required: usize, available: usize },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::HipNotAvailable => {
                write!(f, "HIP runtime not available (is ROCm installed?)")
            }
            GpuError::NoDeviceFound => {
                write!(f, "No AMD GPU detected")
            }
            GpuError::HipApiError { code, description } => {
                write!(f, "HIP API error (code {}): {}", code, description)
            }
            GpuError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of GPU memory: requested {} MB, available {} MB",
                    requested / (1024 * 1024),
                    available / (1024 * 1024)
                )
            }
            GpuError::KernelLaunchFailed { kernel } => {
                write!(f, "Kernel launch failed: {}", kernel)
            }
            GpuError::InvalidDevice { device_id } => {
                write!(f, "Invalid GPU device ID: {}", device_id)
            }
            GpuError::WeightTransferFailed { layer } => {
                write!(f, "Failed to transfer weights for layer {}", layer)
            }
            GpuError::CacheAllocationFailed { reason } => {
                write!(f, "Failed to allocate KV cache: {}", reason)
            }
            GpuError::UnsupportedWeightType { tensor, wtype } => {
                write!(f, "unsupported GPU weight type for {}: {}", tensor, wtype)
            }
            GpuError::InvalidWeightLayout {
                tensor,
                dims,
                reason,
            } => {
                write!(
                    f,
                    "invalid GPU weight layout for {} {:?}: {}",
                    tensor, dims, reason
                )
            }
            GpuError::ModelTooLarge {
                required,
                available,
            } => {
                write!(
                    f,
                    "model too large for GPU: requires {} MB, available {} MB",
                    required / (1024 * 1024),
                    available / (1024 * 1024)
                )
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// Result type alias for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_hip_not_available() {
        let e = GpuError::HipNotAvailable;
        assert!(e.to_string().contains("HIP"));
        assert!(e.to_string().contains("ROCm"));
    }

    #[test]
    fn display_no_device_found() {
        let e = GpuError::NoDeviceFound;
        assert!(e.to_string().contains("No AMD GPU"));
    }

    #[test]
    fn display_out_of_memory() {
        let e = GpuError::OutOfMemory {
            requested: 1024 * 1024 * 1024, // 1 GB
            available: 512 * 1024 * 1024,  // 512 MB
        };
        let s = e.to_string();
        assert!(s.contains("1024 MB"));
        assert!(s.contains("512 MB"));
    }

    #[test]
    fn display_kernel_launch_failed() {
        let e = GpuError::KernelLaunchFailed {
            kernel: "gemm_q4_0".to_string(),
        };
        assert!(e.to_string().contains("gemm_q4_0"));
    }

    #[test]
    fn gpu_result_works() {
        let ok: GpuResult<()> = Ok(());
        assert!(ok.is_ok());

        let err: GpuResult<()> = Err(GpuError::HipNotAvailable);
        assert!(err.is_err());
    }

    #[test]
    fn display_unsupported_weight_type() {
        let e = GpuError::UnsupportedWeightType {
            tensor: "blk.0.attn_q.weight".to_string(),
            wtype: GgmlType::Q6_K,
        };
        let s = e.to_string();
        assert!(s.contains("blk.0.attn_q.weight"));
        assert!(s.contains("Q6_K"));
    }

    #[test]
    fn display_invalid_weight_layout() {
        let e = GpuError::InvalidWeightLayout {
            tensor: "output.weight".to_string(),
            dims: vec![32000],
            reason: "matrix weights must have at least 2 dimensions".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("output.weight"));
        assert!(s.contains("32000"));
    }
}
