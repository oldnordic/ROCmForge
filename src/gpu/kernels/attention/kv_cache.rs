use super::ffi::*;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi::{hipError_t, hipStream_t};
use crate::gpu::GpuKvCache;
use std::os::raw::c_int;

pub fn kv_write(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    kv_size: usize,
    max_seq: usize,
) -> GpuResult<()> {
    kv_write_on_stream(
        k_cache,
        v_cache,
        k,
        v,
        pos,
        kv_size,
        max_seq,
        hipStream_t::null(),
    )
}

/// Write K/V to cache on an explicit HIP stream.
pub fn kv_write_on_stream(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    kv_size: usize,
    max_seq: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if kv_size == 0 {
        return Ok(());
    }

    if pos >= max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("KV write out of bounds: pos {} >= max_seq {}", pos, max_seq),
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
            stream,
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

/// Write K/V to cache with RoPE applied on-the-fly.
pub fn kv_write_rope_on_stream(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    if pos >= max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("KV write out of bounds: pos {} >= max_seq {}", pos, max_seq),
        });
    }

    let result = unsafe {
        gpu_kv_write_rope(
            k_cache,
            v_cache,
            k,
            v,
            pos as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            theta_base,
            if neox { 1 } else { 0 },
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_rope kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Hybrid version that reads next pos from GPU state buffer.
pub fn kv_write_from_state_on_stream(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    d_pos: *const c_int,
    kv_size: usize,
    max_seq: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if kv_size == 0 {
        return Ok(());
    }

    let result = unsafe {
        gpu_kv_write_state(
            k_cache,
            v_cache,
            k,
            v,
            d_pos,
            kv_size as c_int,
            max_seq as c_int,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_state kernel failed: {:?}", result),
        });
    }

    Ok(())
}

/// Hybrid version that applies RoPE and reads next pos from GPU state buffer.
pub fn kv_write_rope_from_state_on_stream(
    kv: &mut GpuKvCache,
    layer_idx: usize,
    k: *const f32,
    v: *const f32,
    d_pos: *const c_int,
    num_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<()> {
    let k_cache = kv.k_ptr(layer_idx)? as *mut f32;
    let v_cache = kv.v_ptr(layer_idx)? as *mut f32;
    let max_seq_len = kv.max_seq_len;

    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let result = unsafe {
        gpu_kv_write_rope_state(
            k_cache,
            v_cache,
            k,
            v,
            d_pos,
            num_kv_heads as c_int,
            head_dim as c_int,
            theta_base,
            if neox { 1 } else { 0 },
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("kv_write_rope_state kernel failed: {:?}", result),
        });
    }

    Ok(())
}

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
