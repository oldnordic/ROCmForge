use super::ffi::*;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

pub fn reconstruct_kv_cache_prefix_sum(
    k_cache: *mut f32,
    v_cache: *mut f32,
    start_pos: usize,
    seq_len: usize,
    kv_lora_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_reconstruct_kv_cache_prefix_sum(
            k_cache,
            v_cache,
            std::ptr::null(), // d_pos not provided by caller
            start_pos as c_int,
            seq_len as c_int,
            kv_lora_dim as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_reconstruct_prefix_sum kernel failed: {:?}", result),
        });
    }

    Ok(())
}
