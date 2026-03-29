//! Flash attention kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::{c_float, c_int, c_void};

unsafe extern "C" {
    fn gpu_kv_write(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
    ) -> hipError_t;

    fn gpu_kv_write_batched(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        start_pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
        seq_len: c_int,
    ) -> hipError_t;

    fn gpu_flash_attn_prefill_strided(
        d_out: *mut f32,
        d_q: *const f32,
        d_k: *const f32,
        d_v: *const f32,
        seq_len: c_int,
        head_dim: c_int,
        out_stride: c_int,
        q_stride: c_int,
        kv_stride: c_int,
        out_head_offset: c_int,
        q_head_offset: c_int,
        kv_head_offset: c_int,
        scale: c_float,
    ) -> hipError_t;

    fn gpu_flash_attn_decode_strided_multi_head(
        d_out: *mut f32,
        d_q: *const f32,
        d_k_cache: *const f32,
        d_v_cache: *const f32,
        seq_len: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        stream: hipStream_t,
    ) -> hipError_t;
}

/// Write K/V to cache.
pub fn kv_write(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    kv_size: usize,
    max_seq: usize,
) -> GpuResult<()> {
    if kv_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV write: kv_size cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_kv_write(
            k_cache,
            v_cache,
            k,
            v,
            pos as c_int,
            kv_size as c_int,
            max_seq as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Write K/V to cache (batched).
pub fn kv_write_batched(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    start_pos: usize,
    kv_size: usize,
    max_seq: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if kv_size == 0 || seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV write batched: kv_size and seq_len cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_kv_write_batched(
            k_cache,
            v_cache,
            k,
            v,
            start_pos as c_int,
            kv_size as c_int,
            max_seq as c_int,
            seq_len as c_int,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_batched kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Fused multi-head attention decode.
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
) -> GpuResult<()> {
    if seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "flash_attn_decode: seq_len cannot be zero".to_string(),
        });
    }

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
            hipStream_t::null(),
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("flash_attn_decode_multi_head kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Flash attention prefill (strided).
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
) -> GpuResult<()> {
    if seq_len == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "flash_attn_prefill: seq_len cannot be zero".to_string(),
        });
    }

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
