//! Shortconv (causal gated depthwise convolution) kernels.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;

extern "C" {
    fn gpu_shortconv_f32(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        state: *mut f32,
        h: i32,
        L: i32,
        stream: hipStream_t,
    ) -> crate::gpu::ffi::hipError_t;

    fn gpu_shortconv_sequence_f32(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        state: *mut f32,
        h: i32,
        L: i32,
        seq_len: i32,
        stream: hipStream_t,
    ) -> crate::gpu::ffi::hipError_t;
}

pub fn dispatch_shortconv(
    output: *mut f32,
    input: *const f32,
    weight: *const f32,
    state: *mut f32,
    h: usize,
    l_cache: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if output.is_null() || input.is_null() || weight.is_null() || state.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "shortconv: null pointer".to_string(),
        });
    }

    let err = unsafe {
        gpu_shortconv_f32(
            output,
            input,
            weight,
            state,
            h as i32,
            l_cache as i32,
            stream,
        )
    };
    if err != crate::gpu::ffi::hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: err as i32,
            description: "shortconv kernel failed".to_string(),
        });
    }
    Ok(())
}

pub fn dispatch_shortconv_sequence(
    output: *mut f32,
    input: *const f32,
    weight: *const f32,
    state: *mut f32,
    h: usize,
    l_cache: usize,
    seq_len: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if output.is_null() || input.is_null() || weight.is_null() || state.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "shortconv_sequence: null pointer".to_string(),
        });
    }

    let err = unsafe {
        gpu_shortconv_sequence_f32(
            output,
            input,
            weight,
            state,
            h as i32,
            l_cache as i32,
            seq_len as i32,
            stream,
        )
    };
    if err != crate::gpu::ffi::hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: err as i32,
            description: "shortconv_sequence kernel failed".to_string(),
        });
    }
    Ok(())
}
