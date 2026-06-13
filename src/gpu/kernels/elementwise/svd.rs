use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use crate::gpu::GpuBuffer;
use std::os::raw::c_int;

extern "C" {
    fn gpu_gemv_f32_k(
        d_V: *const f32,
        d_x: *const f32,
        d_t: *mut f32,
        in_dim: c_int,
        k: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_gemv_f32_add_k(
        d_U: *const f32,
        d_t: *const f32,
        d_y: *mut f32,
        out_dim: c_int,
        k: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

/// Dispatches the low-rank SVD correction y_out = U_k * (V_k * x) and adds it to the output.
///
/// Runs asynchronously on the given stream.
pub fn dispatch_svd_correction(
    stream: hipStream_t,
    u_buf: &GpuBuffer,
    v_buf: &GpuBuffer,
    k: u32,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    temp_vector: *mut f32, // scratch workspace of size k in GPU memory
) -> GpuResult<()> {
    if k == 0 || in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    if input.is_null() || output.is_null() || temp_vector.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dispatch_svd_correction: pointers must be non-null".to_string(),
        });
    }

    // 1. Compute intermediate t = V_k * x
    let res1 = unsafe {
        gpu_gemv_f32_k(
            v_buf.as_ptr() as *const f32,
            input,
            temp_vector,
            in_dim as c_int,
            k as c_int,
            stream,
        )
    };
    if res1 != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res1 as i32,
            description: format!(
                "dispatch_svd_correction (V_k * x) kernel failed: {:?}",
                res1
            ),
        });
    }

    // 2. Compute y_out += U_k * t
    let res2 = unsafe {
        gpu_gemv_f32_add_k(
            u_buf.as_ptr() as *const f32,
            temp_vector,
            output,
            out_dim as c_int,
            k as c_int,
            stream,
        )
    };
    if res2 != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res2 as i32,
            description: format!(
                "dispatch_svd_correction (U_k * t) kernel failed: {:?}",
                res2
            ),
        });
    }

    Ok(())
}

/// General dense F32 matrix-vector multiplication: output = weights * input.
/// Note: weights is of size out_dim x in_dim.
pub fn dispatch_gemv_f32_on_stream(
    weights: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    if weights.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dispatch_gemv_f32_on_stream: pointers must be non-null".to_string(),
        });
    }
    let res = unsafe {
        gpu_gemv_f32_k(
            weights,
            input,
            output,
            in_dim as c_int,
            out_dim as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: format!("gpu_gemv_f32_k kernel failed: {:?}", res),
        });
    }
    Ok(())
}

/// General dense F32 matrix-vector multiplication with accumulation: output += weights * input.
/// Note: weights is of size out_dim x in_dim.
pub fn dispatch_gemv_f32_add_on_stream(
    weights: *const f32,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if in_dim == 0 || out_dim == 0 {
        return Ok(());
    }
    if weights.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "dispatch_gemv_f32_add_on_stream: pointers must be non-null".to_string(),
        });
    }
    let res = unsafe {
        gpu_gemv_f32_add_k(
            weights,
            input,
            output,
            out_dim as c_int,
            in_dim as c_int,
            stream,
        )
    };
    if res != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: res as i32,
            description: format!("gpu_gemv_f32_add_k kernel failed: {:?}", res),
        });
    }
    Ok(())
}
