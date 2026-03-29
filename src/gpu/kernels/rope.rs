//! RoPE (Rotary Position Embedding) kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipError_t;
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
pub fn rope(x: *mut f32, pos: usize, dim: usize, theta_base: f32) -> GpuResult<()> {
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

    let result = unsafe { gpu_rope(x, pos as c_int, dim as c_int, theta_base) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rope kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Multi-head RoPE for decode.
///
/// Applies RoPE within each attention head, supporting both classic and NeoX
/// split-half pairing styles.
pub fn rope_heads(
    x: *mut f32,
    pos: usize,
    num_heads: usize,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
) -> GpuResult<()> {
    if num_heads == 0 || head_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RoPE heads: num_heads and head_dim cannot be zero".to_string(),
        });
    }

    if head_dim % 2 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("RoPE heads: head_dim {} must be even", head_dim),
        });
    }

    let result = unsafe {
        gpu_rope_heads(
            x,
            pos as c_int,
            num_heads as c_int,
            head_dim as c_int,
            theta_base,
            neox as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rope_heads kernel failed: {:?}", result),
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
        gpu_rope_batched(
            x,
            start_pos as c_int,
            dim as c_int,
            theta_base,
            seq_len as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rope_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Batched multi-head RoPE for prefill.
pub fn rope_heads_batched(
    x: *mut f32,
    start_pos: usize,
    num_heads: usize,
    head_dim: usize,
    theta_base: f32,
    seq_len: usize,
    neox: bool,
) -> GpuResult<()> {
    if num_heads == 0 || head_dim == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "RoPE heads batched: num_heads, head_dim, and seq_len cannot be zero"
                .to_string(),
        });
    }

    if head_dim % 2 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("RoPE heads batched: head_dim {} must be even", head_dim),
        });
    }

    let result = unsafe {
        gpu_rope_heads_batched(
            x,
            start_pos as c_int,
            num_heads as c_int,
            head_dim as c_int,
            theta_base,
            seq_len as c_int,
            neox as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("rope_heads_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn gpu_rope(x: *mut f32, pos: c_int, dim: c_int, theta_base: f32) -> hipError_t;

    fn gpu_rope_heads(
        x: *mut f32,
        pos: c_int,
        num_heads: c_int,
        head_dim: c_int,
        theta_base: f32,
        neox: c_int,
    ) -> hipError_t;

    fn gpu_rope_batched(
        x: *mut f32,
        start_pos: c_int,
        dim: c_int,
        theta_base: f32,
        seq_len: c_int,
    ) -> hipError_t;

    fn gpu_rope_heads_batched(
        x: *mut f32,
        start_pos: c_int,
        num_heads: c_int,
        head_dim: c_int,
        theta_base: f32,
        seq_len: c_int,
        neox: c_int,
    ) -> hipError_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rejects_zero_dim() {
        let result = rope(std::ptr::null_mut(), 0, 0, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn rope_rejects_odd_dim() {
        let result = rope(std::ptr::null_mut(), 0, 127, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn rope_heads_rejects_zero_num_heads() {
        let result = rope_heads(std::ptr::null_mut(), 0, 0, 128, 10000.0, true);
        assert!(result.is_err());
    }

    #[test]
    fn rope_batched_rejects_zero_seq_len() {
        let result = rope_batched(std::ptr::null_mut(), 0, 128, 10000.0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn rope_batched_rejects_odd_dim() {
        let result = rope_batched(std::ptr::null_mut(), 0, 127, 10000.0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn rope_heads_batched_rejects_zero_seq_len() {
        let result = rope_heads_batched(std::ptr::null_mut(), 0, 8, 128, 10000.0, 0, false);
        assert!(result.is_err());
    }
}
