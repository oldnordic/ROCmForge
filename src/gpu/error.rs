//! GPU error types with safe error handling.
//!
//! Design principle: Never panic, always return errors gracefully.
//! All GPU errors should be recoverable (CPU fallback).

use std::fmt;

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
            GpuError::OutOfMemory { requested, available } => {
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
            requested: 1024 * 1024 * 1024,  // 1 GB
            available: 512 * 1024 * 1024,    // 512 MB
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
}
