use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

extern "C" {
    fn gpu_add(x: *const f32, y: *const f32, out: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_mul_on_stream(x: *const f32, y: *const f32, out: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_weighted_add_on_stream(src: *const f32, dst: *mut f32, weight: f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_dot_f16_f32_on_stream(weights: *const u8, input: *const f32, output: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_scale(x: *const f32, out: *mut f32, scale: f32, n: c_int) -> hipError_t;
    fn gpu_gelu_on_stream(x: *const f32, out: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_silu_on_stream(x: *const f32, out: *mut f32, n: c_int, stream: hipStream_t) -> hipError_t;
    fn gpu_add_batched(x: *const f32, y: *const f32, out: *mut f32, n: c_int, seq_len: c_int) -> hipError_t;
    fn gpu_mul_batched(x: *const f32, y: *const f32, out: *mut f32, n: c_int, seq_len: c_int) -> hipError_t;
}

/// Element-wise add: out = x + y
pub fn add(x: *const f32, y: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    add_on_stream(x, y, out, n, hipStream_t::null())
}

/// Element-wise add on an explicit HIP stream.
pub fn add_on_stream(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise add: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_add(x, y, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("add kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Element-wise multiply: out = x * y
pub fn mul(x: *const f32, y: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    mul_on_stream(x, y, out, n, hipStream_t::null())
}

/// Element-wise multiply on an explicit HIP stream.
pub fn mul_on_stream(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise mul: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_mul_on_stream(x, y, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("mul kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Element-wise weighted add: dst[i] += src[i] * weight
pub fn weighted_add_on_stream(
    src: *const f32,
    dst: *mut f32,
    weight: f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise weighted_add: n cannot be zero".to_string(),
        });
    }
    if src.is_null() || dst.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise weighted_add: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe { gpu_weighted_add_on_stream(src, dst, weight, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("weighted_add kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Dot one F16 vector with an F32 vector, writing a single F32 scalar.
pub fn dot_f16_f32_on_stream(
    weights: *const u8,
    input: *const f32,
    output: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dot_f16_f32: n cannot be zero".to_string(),
        });
    }
    if weights.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dot_f16_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe { gpu_dot_f16_f32_on_stream(weights, input, output, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("dot_f16_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Element-wise scale: out = x * scale
pub fn scale(x: *const f32, out: *mut f32, scale: f32, n: usize) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise scale: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_scale(x, out, scale, n as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("scale kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// GELU activation: out = gelu(x)
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    gelu_on_stream(x, out, n, hipStream_t::null())
}

/// GELU activation on an explicit HIP stream.
pub fn gelu_on_stream(
    x: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise gelu: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_gelu_on_stream(x, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gelu kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// SiLU activation: out = x / (1 + exp(-x))
pub fn silu(x: *const f32, out: *mut f32, n: usize) -> GpuResult<()> {
    silu_on_stream(x, out, n, hipStream_t::null())
}

/// SiLU activation on an explicit HIP stream.
pub fn silu_on_stream(
    x: *const f32,
    out: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Elementwise silu: n cannot be zero".to_string(),
        });
    }

    let result = unsafe { gpu_silu_on_stream(x, out, n as c_int, stream) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("silu kernel failed: {:?}", result),
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

    let result = unsafe { gpu_add_batched(x, y, out, n as c_int, seq_len as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("add_batched kernel failed: {:?}", result),
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

    let result = unsafe { gpu_mul_batched(x, y, out, n as c_int, seq_len as c_int) };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("mul_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}
