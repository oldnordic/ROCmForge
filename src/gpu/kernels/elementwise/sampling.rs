use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

extern "C" {
    fn gpu_argmax_f32_on_stream(
        input: *const f32,
        partial_values: *mut f32,
        partial_indices: *mut i32,
        output_index: *mut i32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

/// Argmax reduction over logits: returns the index of the maximum value.
///
/// Uses reusable GPU workspace to avoid per-token allocations.
pub fn argmax_f32(
    input: *const f32,
    partial_values: *mut f32,
    partial_indices: *mut i32,
    output_index: *mut i32,
    n: usize,
) -> GpuResult<()> {
    argmax_f32_on_stream(
        input,
        partial_values,
        partial_indices,
        output_index,
        n,
        hipStream_t::null(),
    )
}

/// Argmax reduction over logits on an explicit HIP stream.
///
/// Uses reusable GPU workspace to avoid per-token allocations.
pub fn argmax_f32_on_stream(
    input: *const f32,
    partial_values: *mut f32,
    partial_indices: *mut i32,
    output_index: *mut i32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "argmax_f32: n cannot be zero".to_string(),
        });
    }
    if input.is_null()
        || partial_values.is_null()
        || partial_indices.is_null()
        || output_index.is_null()
    {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "argmax_f32: all pointers must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gpu_argmax_f32_on_stream(
            input,
            partial_values,
            partial_indices,
            output_index,
            n as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("argmax_f32 kernel failed: {:?}", result),
        });
    }

    Ok(())
}
