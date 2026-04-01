//! Flash attention kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use crate::gpu::GpuKvCache;
use std::os::raw::{c_float, c_int};

unsafe extern "C" {
    fn gpu_kv_write(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_kv_write_state(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        d_pos: *const c_int,
        kv_size: c_int,
        max_seq: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_kv_write_rope(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        pos: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        theta_base: f32,
        neox: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gpu_kv_write_rope_state(
        d_k_cache: *mut f32,
        d_v_cache: *mut f32,
        d_k: *const f32,
        d_v: *const f32,
        d_pos: *const c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        theta_base: c_float,
        neox: c_int,
        stream: hipStream_t,
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

    fn gpu_flash_attn_decode_strided_multi_head_state(
        d_out: *mut f32,
        d_q: *const f32,
        d_k_cache: *const f32,
        d_v_cache: *const f32,
        d_seq_len: *const c_int,
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

/// Fused KV cache write and RoPE application.
pub fn kv_write_rope_on_stream(
    kv: &mut GpuKvCache,
    layer_idx: usize,
    d_k: *const f32,
    d_v: *const f32,
    pos: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<()> {
    if pos >= kv.max_seq_len {
        return Err(GpuError::InvalidSequencePosition {
            pos,
            max: kv.max_seq_len,
        });
    }

    let result = unsafe {
        gpu_kv_write_rope(
            kv.k_ptr(layer_idx)? as *mut f32,
            kv.v_ptr(layer_idx)? as *mut f32,
            d_k,
            d_v,
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

/// Write K/V to cache using a device-resident position scalar.
pub fn kv_write_from_state_on_stream(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos_ptr: *const i32,
    kv_size: usize,
    max_seq: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if kv_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV write: kv_size cannot be zero".to_string(),
        });
    }
    if pos_ptr.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV write: pos_ptr must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gpu_kv_write_state(
            k_cache,
            v_cache,
            k,
            v,
            pos_ptr,
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

/// Apply RoPE to K and write rotated K plus V into the KV cache.
pub fn kv_write_rope_from_state_on_stream(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos_ptr: *const i32,
    num_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV rope write: num_kv_heads and head_dim cannot be zero".to_string(),
        });
    }
    if head_dim % 2 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("KV rope write: head_dim {} must be even", head_dim),
        });
    }
    if pos_ptr.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV rope write: pos_ptr must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gpu_kv_write_rope_state(
            k_cache,
            v_cache,
            k,
            v,
            pos_ptr,
            num_kv_heads as c_int,
            head_dim as c_int,
            theta_base,
            neox as c_int,
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
        hipStream_t::null(),
    )
}

/// Fused multi-head attention decode on an explicit HIP stream.
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
    stream: hipStream_t,
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
            stream,
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

/// Fused multi-head attention decode using a device-resident sequence-length scalar.
pub fn flash_attn_decode_strided_multi_head_from_state_on_stream(
    d_out: *mut f32,
    d_q: *const f32,
    d_k_cache: *const f32,
    d_v_cache: *const f32,
    seq_len_ptr: *const i32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if seq_len_ptr.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "flash_attn_decode: seq_len_ptr must be non-null".to_string(),
        });
    }

    let result = unsafe {
        gpu_flash_attn_decode_strided_multi_head_state(
            d_out,
            d_q,
            d_k_cache,
            d_v_cache,
            seq_len_ptr,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            scale,
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!(
                "flash_attn_decode_multi_head_state kernel failed: {:?}",
                result
            ),
        });
    }

    Ok(())
}

/// Fused attention decode (single head alias for tests).
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
        d_out, d_q, d_k_cache, d_v_cache, seq_len, 1, 1, head_dim, scale,
    )
}

/// Fused attention decode strided (single head alias for tests).
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
    // Offset pointers by head_offset to simulate selecting a specific head
    let d_k_offset = unsafe { d_k_cache.add(head_offset) };
    let d_v_offset = unsafe { d_v_cache.add(head_offset) };

    // Calculate num_kv_heads based on kv_size
    let num_kv_heads = kv_size / head_dim;

    flash_attn_decode_strided_multi_head(
        d_out,
        d_q,
        d_k_offset,
        d_v_offset,
        seq_len,
        1, // single output head
        num_kv_heads,
        head_dim,
        scale,
    )
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
