//! RoPE (Rotary Position Embedding) kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use std::os::raw::c_int;

/// RoPE in-place transformation for a single token.
///
/// Rotates each pair (x[2i], x[2i+1]) by position angle:
///   x[2i]   = x[2i] * cos(pos * theta_i) - x[2i+1] * sin(pos * theta_i)
///   x[2i+1] = x[2i] * sin(pos * theta_i) + x[2i+1] * cos(pos * theta_i)
///
/// # Arguments
/// * `x` - GPU pointer to input/output tensor [hidden_size] (modified in-place)
/// * `pos` - Position index for this token
/// * `dim` - Hidden size (must be even)
/// * `theta_base` - Base frequency (typically 10000.0)
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - Memory pointer must be a valid GPU pointer
/// - Bounds are validated on CPU before kernel launch
pub fn rope(
    x: *mut f32,
    pos: usize,
    dim: usize,
    theta_base: f32,
) -> GpuResult<()> {
    if dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RoPE: dim cannot be zero".to_string(),
        });
    }

    if dim % 2 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("RoPE: dim {} must be even", dim),
        });
    }

    let result = unsafe {
        gpu_rope(x, pos as c_int, dim as c_int, theta_base)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "rope".to_string(),
        });
    }

    Ok(())
}

/// Batched RoPE for prefill (multiple tokens).
///
/// # Arguments
/// * `x` - GPU pointer to input/output [seq_len, hidden_size] (modified in-place)
/// * `start_pos` - Starting position for this batch
/// * `dim` - Hidden size (must be even)
/// * `theta_base` - Base frequency (typically 10000.0)
/// * `seq_len` - Number of tokens in this batch
pub fn rope_batched(
    x: *mut f32,
    start_pos: usize,
    dim: usize,
    theta_base: f32,
    seq_len: usize,
) -> GpuResult<()> {
    if dim == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RoPE batched: dim and seq_len cannot be zero".to_string(),
        });
    }

    if dim % 2 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("RoPE batched: dim {} must be even", dim),
        });
    }

    let result = unsafe {
        gpu_rope_batched(x, start_pos as c_int, dim as c_int, theta_base, seq_len as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "rope_batched".to_string(),
        });
    }

    Ok(())
}

// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn gpu_rope(
        x: *mut f32,
        pos: c_int,
        dim: c_int,
        theta_base: f32,
    ) -> c_int;

    fn gpu_rope_batched(
        x: *mut f32,
        start_pos: c_int,
        dim: c_int,
        theta_base: f32,
        seq_len: c_int,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rejects_zero_dim() {
        let result = rope(
            std::ptr::null_mut(),
            0,
            0,
            10000.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn rope_rejects_odd_dim() {
        let result = rope(
            std::ptr::null_mut(),
            0,
            127,
            10000.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn rope_batched_rejects_zero_seq_len() {
        let result = rope_batched(
            std::ptr::null_mut(),
            0,
            128,
            10000.0,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn rope_batched_rejects_odd_dim() {
        let result = rope_batched(
            std::ptr::null_mut(),
            0,
            127,
            10000.0,
            10,
        );
        assert!(result.is_err());
    }
}
