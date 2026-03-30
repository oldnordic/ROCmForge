//! RMS norm kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

/// RMS normalization: out = x * rsqrt(mean(x^2) + eps) * weight
///
/// # Arguments
/// * `x` - GPU pointer to input tensor [hidden_size]
/// * `weight` - GPU pointer to scale tensor [hidden_size]
/// * `out` - GPU pointer to output tensor [hidden_size]
/// * `n` - Number of elements (hidden_size)
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn rms_norm(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
) -> GpuResult<()> {
    rms_norm_on_stream(x, weight, out, n, eps, hipStream_t::null())
}

/// RMS normalization on an explicit HIP stream.
pub fn rms_norm_on_stream(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RMS norm: n cannot be zero".to_string(),
        });
    }

    // Load and call kernel
    let result = unsafe { gpu_rms_norm(x, weight, out, n as c_int, eps, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rms_norm kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Specialized RMS normalization using wavefront shuffles (Vulkan-style).
pub fn rms_norm_vulkan_style(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RMS norm: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_rms_norm_vulkan_style(x, weight, out, n as c_int, eps, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rms_norm_vulkan_style kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched RMS norm for prefill
///
/// # Arguments
/// * `x` - GPU pointer to input [seq_len, hidden_size]
/// * `weight` - GPU pointer to scale [hidden_size]
/// * `out` - GPU pointer to output [seq_len, hidden_size]
/// * `n` - Hidden size
/// * `eps` - Epsilon for numerical stability
/// * `seq_len` - Number of sequences
pub fn rms_norm_batched(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    seq_len: usize,
) -> GpuResult<()> {
    if n == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RMS norm batched: n and seq_len cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_rms_norm_batched(x, weight, out, n as c_int, eps, seq_len as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rms_norm_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn gpu_rms_norm(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: c_int,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_rms_norm_vulkan_style(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: c_int,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_rms_norm_batched(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: c_int,
        eps: f32,
        seq_len: c_int,
    ) -> hipError_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_rejects_zero_n() {
        let result = rms_norm(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            1e-5,
        );
        assert!(result.is_err());
    }

    #[test]
    fn rms_norm_batched_rejects_zero_seq_len() {
        let result = rms_norm_batched(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            128,
            1e-5,
            0,
        );
        assert!(result.is_err());
    }
}
