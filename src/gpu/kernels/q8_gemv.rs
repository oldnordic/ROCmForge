//! Q8_0 GEMV kernel wrappers.
//!
//! The generic Q8_0 GEMV path remains available for smaller projections, while
//! the LM-head path can use a dedicated launcher tuned for very wide vocab
//! projections.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

fn validate_q8_0_gemv_args(
    kernel_name: &str,
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    if n_rows == 0 || ncols_dst == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("{kernel_name}: n_rows and ncols_dst cannot be zero"),
        });
    }

    if n_rows % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("{kernel_name}: n_rows must be multiple of 32, got {n_rows}"),
        });
    }

    if weights_q8_0.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("{kernel_name}: weights_q8_0 pointer is null"),
        });
    }

    if input.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("{kernel_name}: input pointer is null"),
        });
    }

    if output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("{kernel_name}: output pointer is null"),
        });
    }

    Ok(())
}

fn map_launch_result(kernel_name: &str, result: hipError_t) -> GpuResult<()> {
    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("{kernel_name} kernel failed: {result:?}"),
        });
    }

    Ok(())
}

pub fn gemv_q8_0_f32(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q8_0_f32_on_stream(
        weights_q8_0,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q8_0_f32_on_stream(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_q8_0_gemv_args(
        "gemv_q8_0_f32",
        weights_q8_0,
        input,
        output,
        n_rows,
        ncols_dst,
    )?;

    let result = unsafe {
        gemv_q8_0_f32_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    map_launch_result("gemv_q8_0_f32", result)
}

/// Q8_0 × f32 GEMV specialized for metadata-selected LM heads.
///
/// This uses a dedicated launcher so wide vocab projections can use a different
/// kernel geometry without changing the generic Q8_0 path.
pub fn gemv_q8_0_f32_lm_head(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
) -> GpuResult<()> {
    gemv_q8_0_f32_lm_head_on_stream(
        weights_q8_0,
        input,
        output,
        n_rows,
        ncols_dst,
        hipStream_t::null(),
    )
}

pub fn gemv_q8_0_f32_lm_head_on_stream(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_q8_0_gemv_args(
        "gemv_q8_0_f32_lm_head",
        weights_q8_0,
        input,
        output,
        n_rows,
        ncols_dst,
    )?;

    let result = unsafe {
        gemv_q8_0_f32_lm_head_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            stream,
        )
    };

    map_launch_result("gemv_q8_0_f32_lm_head", result)
}

pub fn gemv_q8_0_f32_lm_head_on_stream_variant(
    weights_q8_0: *const u8,
    input: *const f32,
    output: *mut f32,
    n_rows: usize,
    ncols_dst: usize,
    variant: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_q8_0_gemv_args(
        "gemv_q8_0_f32_lm_head_variant",
        weights_q8_0,
        input,
        output,
        n_rows,
        ncols_dst,
    )?;

    let result = unsafe {
        gemv_q8_0_f32_lm_head_variant_launch(
            weights_q8_0,
            input,
            output,
            n_rows as c_int,
            ncols_dst as c_int,
            variant,
            stream,
        )
    };

    map_launch_result("gemv_q8_0_f32_lm_head_variant", result)
}

unsafe extern "C" {
    fn gemv_q8_0_f32_launch(
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q8_0_f32_lm_head_launch(
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_q8_0_f32_lm_head_variant_launch(
        weights_q8_0: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: c_int,
        ncols_dst: c_int,
        variant: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}
