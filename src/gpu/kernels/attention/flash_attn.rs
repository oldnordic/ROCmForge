use super::ffi::*;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

pub fn flash_attn_decode_strided_multi_head(
    d_out: *mut f32,
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_lora_dim: usize,
    adastate_anchors_enabled: bool,
    w_up_k: *const f32,
    w_up_v: *const f32,
) -> GpuResult<()> {
    flash_attn_decode_strided_multi_head_on_stream(
        d_out,
        d_q,
        d_k_cache,
        d_v_cache,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        kv_lora_dim,
        adastate_anchors_enabled,
        w_up_k,
        w_up_v,
        hipStream_t::null(),
    )
}

pub fn flash_attn_decode_strided_multi_head_on_stream(
    d_out: *mut f32,
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_lora_dim: usize,
    adastate_anchors_enabled: bool,
    w_up_k: *const f32,
    w_up_v: *const f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_flash_attn_decode_strided_multi_head(
            d_out,
            d_q,
            d_k_cache,
            d_v_cache,
            seq_len as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            scale,
            kv_lora_dim as c_int,
            if adastate_anchors_enabled { 1 } else { 0 },
            w_up_k,
            w_up_v,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("flash_attn_decode kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn flash_attn_decode_strided_multi_head_from_state_on_stream(
    d_out: *mut f32,
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    d_seq_len: *const c_int,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    kv_lora_dim: usize,
    adastate_anchors_enabled: bool,
    w_up_k: *const f32,
    w_up_v: *const f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_flash_attn_decode_strided_multi_head_state(
            d_out,
            d_q,
            d_k_cache,
            d_v_cache,
            d_seq_len,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            scale,
            kv_lora_dim as c_int,
            if adastate_anchors_enabled { 1 } else { 0 },
            w_up_k,
            w_up_v,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("flash_attn_decode_state kernel failed: {:?}", result),
        });
    }

    Ok(())
}

pub fn flash_attn_decode(
    d_out: *mut f32,
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> GpuResult<()> {
    flash_attn_decode_strided_multi_head(
        d_out,
        d_q,
        d_k_cache,
        d_v_cache,
        seq_len,
        1,
        1,
        head_dim,
        scale,
        0,
        false,
        std::ptr::null(),
        std::ptr::null(),
    )
}

pub fn flash_attn_decode_strided(
    d_out: *mut f32,
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    seq_len: usize,
    head_dim: usize,
    kv_size: usize,
    head_offset: usize,
    scale: f32,
) -> GpuResult<()> {
    let d_k_offset = unsafe { d_k_cache.add(head_offset) };
    let d_v_offset = unsafe { d_v_cache.add(head_offset) };
    let num_kv_heads = kv_size / head_dim;
    flash_attn_decode_strided_multi_head(
        d_out,
        d_q,
        d_k_offset,
        d_v_offset,
        seq_len,
        1,
        num_kv_heads,
        head_dim,
        scale,
        0,
        false,
        std::ptr::null(),
        std::ptr::null(),
    )
}

pub fn flash_attn_prefill_strided(
    d_out: *mut f32,
    d_q: *const f32,
    d_k: *const f32,
    d_v: *const f32,
    seq_len: usize,
    head_dim: usize,
    out_stride: usize,
    q_stride: usize,
    kv_stride: usize,
    out_head_offset: usize,
    q_head_offset: usize,
    kv_head_offset: usize,
    scale: f32,
    kv_lora_dim: usize,
    w_up_k: *const f32,
    w_up_v: *const f32,
) -> GpuResult<()> {
    let result = unsafe {
        gpu_flash_attn_prefill_strided(
            d_out,
            d_q,
            d_k,
            d_v,
            seq_len as c_int,
            head_dim as c_int,
            out_stride as c_int,
            q_stride as c_int,
            kv_stride as c_int,
            out_head_offset as c_int,
            q_head_offset as c_int,
            kv_head_offset as c_int,
            scale,
            kv_lora_dim as c_int,
            w_up_k,
            w_up_v,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("flash_attn_prefill kernel failed: {:?}", result),
        });
    }

    Ok(())
}
