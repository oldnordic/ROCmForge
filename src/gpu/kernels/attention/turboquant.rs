use super::ffi::*;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

pub fn kv_write_compressed(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    d_pos: *const c_int,
    num_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    kv_lora_dim: usize,
    kv_frame_codec_enabled: bool,
    w_down_k: *const f32,
    w_down_v: *const f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_kv_write_compressed(
            k_cache as *mut u8,
            v_cache as *mut u8,
            k,
            v,
            d_pos,
            num_kv_heads as c_int,
            head_dim as c_int,
            theta_base,
            if neox { 1 } else { 0 },
            kv_lora_dim as c_int,
            if kv_frame_codec_enabled { 1 } else { 0 },
            w_down_k,
            w_down_v,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_compressed kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn kv_write_batched_compressed(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    start_pos: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_lora_dim: usize,
    kv_frame_codec_enabled: bool,
    w_down_k: *const f32,
    w_down_v: *const f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_kv_write_batched_compressed(
            k_cache as *mut u8,
            v_cache as *mut u8,
            k,
            v,
            start_pos as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            seq_len as c_int,
            kv_lora_dim as c_int,
            if kv_frame_codec_enabled { 1 } else { 0 },
            w_down_k,
            w_down_v,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_batched_compressed kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn kv_write_turboquant(
    k_cache: *mut u8,
    v_cache: *mut u8,
    k: *const f32,
    v: *const f32,
    d_pos: *const c_int,
    num_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    kv_lora_dim: usize,
    centroids: *const f32,
    qjl_scale: f32,
    w_down_k: *const f32,
    w_down_v: *const f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_kv_write_turboquant(
            k_cache,
            v_cache,
            k,
            v,
            d_pos,
            num_kv_heads as c_int,
            head_dim as c_int,
            theta_base,
            if neox { 1 } else { 0 },
            kv_lora_dim as c_int,
            centroids,
            qjl_scale,
            w_down_k,
            w_down_v,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_turboquant kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn flash_attn_decode_turboquant(
    out: *mut f32,
    q: *const f32,
    k_cache: *const u8,
    v_cache: *const u8,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_lora_dim: usize,
    centroids: *const f32,
    qjl_scale: f32,
    w_up_k: *const f32,
    w_up_v: *const f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_flash_attn_decode_turboquant(
            out,
            q,
            k_cache,
            v_cache,
            seq_len as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            scale,
            kv_lora_dim as c_int,
            centroids,
            qjl_scale,
            w_up_k,
            w_up_v,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "gpu_flash_attn_decode_turboquant kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}
