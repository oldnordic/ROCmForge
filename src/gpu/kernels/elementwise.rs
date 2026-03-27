//! Element-wise kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use std::os::raw::c_int;

/// Element-wise add: out = x + y
pub fn add(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise add: n cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_add(x, y, out, n as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "add".to_string(),
        });
    }

    Ok(())
}

/// Element-wise multiply: out = x * y
pub fn mul(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise mul: n cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_mul(x, y, out, n as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "mul".to_string(),
        });
    }

    Ok(())
}

/// Element-wise scale: out = x * scale
pub fn scale(
    x: *const f32,
    out: *mut f32,
    scale: f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise scale: n cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_scale(x, out, scale, n as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "scale".to_string(),
        });
    }

    Ok(())
}

/// GELU activation: out = gelu(x)
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(
    x: *const f32,
    out: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise gelu: n cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_gelu(x, out, n as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "gelu".to_string(),
        });
    }

    Ok(())
}

/// SiLU activation: out = x / (1 + exp(-x))
pub fn silu(
    x: *const f32,
    out: *mut f32,
    n: usize,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise silu: n cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_silu(x, out, n as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "silu".to_string(),
        });
    }

    Ok(())
}

/// Batched element-wise add for prefill: out[s, i] = x[s, i] + y[i]
/// where x is [seq_len, n] and y is [n] (broadcast)
pub fn add_batched(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if n == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise add_batched: n and seq_len cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_add_batched(x, y, out, n as c_int, seq_len as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "add_batched".to_string(),
        });
    }

    Ok(())
}

/// Batched element-wise multiply for prefill: out[s, i] = x[s, i] * y[i]
pub fn mul_batched(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if n == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise mul_batched: n and seq_len cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_mul_batched(x, y, out, n as c_int, seq_len as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "mul_batched".to_string(),
        });
    }

    Ok(())
}

// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn gpu_add(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;

    fn gpu_mul(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;

    fn gpu_scale(
        x: *const f32,
        out: *mut f32,
        scale: f32,
        n: c_int,
    ) -> c_int;

    fn gpu_gelu(
        x: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;

    fn gpu_silu(
        x: *const f32,
        out: *mut f32,
        n: c_int,
    ) -> c_int;

    fn gpu_add_batched(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
        seq_len: c_int,
    ) -> c_int;

    fn gpu_mul_batched(
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: c_int,
        seq_len: c_int,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_rejects_zero_n() {
        let result = add(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn gelu_rejects_zero_n() {
        let result = gelu(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn add_batched_rejects_zero_seq_len() {
        let result = add_batched(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            128,
            0,
        );
        assert!(result.is_err());
    }
}
