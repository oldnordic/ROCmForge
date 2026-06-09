//! GEMM kernels for all quantization types.

use super::super::super::device::GpuDevice;
use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::{c_int, c_void};

unsafe extern "C" {
    fn gemm_q4_0_f32_launch(
        weights_q4_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q4_1_f32_launch(
        weights_q4_1: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q8_0_f32_launch(
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q4_k_f32_launch(
        weights_q4_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q5_k_f32_launch(
        weights_q5_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemm_q6_k_f32_launch(
        weights_q6_k: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        batch_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

// ── GEMM Kernels ──────────────────────────────────────────────────────────────────

pub fn gemm_q4_0_f32(
    weights_q4_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_0_f32_launch(
            weights_q4_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q4_1_f32(
    weights_q4_1: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_1_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_1.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_1_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_1_f32_launch(
            weights_q4_1,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_1_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q8_0_f32(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q8_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q8_0_f32_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q8_0_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q8_0_f32_on_stream(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q8_0.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q8_0_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q8_0_f32_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q8_0_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q4_k_f32(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_k_f32_launch(
            weights_q4_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q4_k_f32_on_stream(
    weights_q4_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q4_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q4_k_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q4_k_f32_launch(
            weights_q4_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q4_k_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_k_f32(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q5_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_k_f32_launch(
            weights_q5_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q5_k_f32_on_stream(
    weights_q5_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q5_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q5_k_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q5_k_f32_launch(
            weights_q5_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q5_k_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q6_k_f32(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q6_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q6_k_f32_launch(
            weights_q6_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q6_k_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn gemm_q6_k_f32_on_stream(
    weights_q6_k: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    batch_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 || batch_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32_on_stream: dimensions cannot be zero".to_string(),
        });
    }

    if weights_q6_k.is_null() || input.is_null() || output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gemm_q6_k_f32_on_stream: pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gemm_q6_k_f32_launch(
            weights_q6_k,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            batch_size as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("gemm_q6_k_f32_on_stream kernel failed: {:?}", result),
        });
    }

    Ok(())
}
