//! Attention kernel wrappers.
//!
//! Safety-first: bounds checked before kernel launch.

use super::super::error::{GpuError, GpuResult};
use std::os::raw::c_int;

/// Write K/V to cache at a single position.
///
/// # Arguments
/// * `k_cache` - GPU pointer to key cache [max_seq * kv_size]
/// * `v_cache` - GPU pointer to value cache [max_seq * kv_size]
/// * `k` - GPU pointer to key to write [kv_size]
/// * `v` - GPU pointer to value to write [kv_size]
/// * `pos` - Position to write at
/// * `kv_size` - Size per position (num_kv_heads * head_dim)
/// * `max_seq` - Maximum sequence length (for bounds checking)
pub fn kv_write(
    k_cache: *mut f32,
    v_cache: *mut f32,
    k: *const f32,
    v: *const f32,
    pos: usize,
    kv_size: usize,
    max_seq: usize,
) -> GpuResult<()> {
    if pos >= max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!("KV write position {} exceeds max_seq {}", pos, max_seq),
        });
    }

    if kv_size == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "KV write: kv_size cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_kv_write(k_cache, v_cache, k, v, pos as c_int, kv_size as c_int, max_seq as c_int)
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "kv_write".to_string(),
        });
    }

    Ok(())
}

/// Batched KV write for prefill.
///
/// # Arguments
/// * `k_cache` - GPU pointer to key cache [max_seq * kv_size]
/// * `v_cache` - GPU pointer to value cache [max_seq * kv_size]
/// * `k` - GPU pointer to keys to write [seq_len * kv_size]
/// * `v` - GPU pointer to values to write [seq_len * kv_size]
/// * `start_pos` - Starting position
/// * `kv_size` - Size per position
/// * `max_seq` - Maximum sequence length
/// * `seq_len` - Number of positions to write
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
    if start_pos + seq_len > max_seq {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "KV batch write range [{}..{}) exceeds max_seq {}",
                start_pos,
                start_pos + seq_len,
                max_seq
            ),
        });
    }

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

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "kv_write_batched".to_string(),
        });
    }

    Ok(())
}

/// Flash attention for single-token decode.
///
/// Computes attention for a single query token against all cached K/V.
/// Uses online softmax for numerical stability.
///
/// # Arguments
/// * `out` - GPU pointer to output [head_dim]
/// * `q` - GPU pointer to query [head_dim]
/// * `k_cache` - GPU pointer to key cache [seq_len * head_dim]
/// * `v_cache` - GPU pointer to value cache [seq_len * head_dim]
/// * `seq_len` - Number of cached positions
/// * `head_dim` - Head dimension
/// * `scale` - Attention scale (typically 1.0 / sqrt(head_dim as f32))
pub fn flash_attn_decode(
    out: *mut f32,
    q: *const f32,
    k_cache: *const f32,
    v_cache: *const f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> GpuResult<()> {
    if seq_len == 0 || head_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Flash attn decode: seq_len and head_dim cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_flash_attn_decode(
            out,
            q,
            k_cache,
            v_cache,
            seq_len as c_int,
            head_dim as c_int,
            scale,
        )
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "flash_attn_decode".to_string(),
        });
    }

    Ok(())
}

/// Causal flash attention for prefill.
///
/// Each token in the batch attends to all previous tokens (including itself).
///
/// # Arguments
/// * `out` - GPU pointer to output [seq_len * head_dim]
/// * `q` - GPU pointer to queries [seq_len * head_dim]
/// * `k` - GPU pointer to keys [seq_len * head_dim]
/// * `v` - GPU pointer to values [seq_len * head_dim]
/// * `seq_len` - Number of tokens
/// * `head_dim` - Head dimension
/// * `scale` - Attention scale
pub fn flash_attn_prefill(
    out: *mut f32,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> GpuResult<()> {
    if seq_len == 0 || head_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "Flash attn prefill: seq_len and head_dim cannot be zero".to_string(),
        });
    }

    let result = unsafe {
        gpu_flash_attn_prefill(
            out,
            q,
            k,
            v,
            seq_len as c_int,
            head_dim as c_int,
            scale,
        )
    };

    if result != 0 {
        return Err(GpuError::KernelLaunchFailed {
            kernel: "flash_attn_prefill".to_string(),
        });
    }

    Ok(())
}

// FFI declarations - will be linked from compiled HIP kernels
unsafe extern "C" {
    fn gpu_kv_write(
        k_cache: *mut f32,
        v_cache: *mut f32,
        k: *const f32,
        v: *const f32,
        pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
    ) -> c_int;

    fn gpu_kv_write_batched(
        k_cache: *mut f32,
        v_cache: *mut f32,
        k: *const f32,
        v: *const f32,
        start_pos: c_int,
        kv_size: c_int,
        max_seq: c_int,
        seq_len: c_int,
    ) -> c_int;

    fn gpu_flash_attn_decode(
        out: *mut f32,
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        seq_len: c_int,
        head_dim: c_int,
        scale: f32,
    ) -> c_int;

    fn gpu_flash_attn_prefill(
        out: *mut f32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        seq_len: c_int,
        head_dim: c_int,
        scale: f32,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_write_rejects_out_of_bounds() {
        let result = kv_write(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            100,
            128,
            100,  // max_seq == pos, should fail
        );
        assert!(result.is_err());
    }

    #[test]
    fn kv_write_batched_rejects_out_of_bounds() {
        let result = kv_write_batched(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            90,
            128,
            100,
            20,  // 90 + 20 = 110 > 100, should fail
        );
        assert!(result.is_err());
    }

    #[test]
    fn flash_attn_decode_rejects_zero_seq_len() {
        let result = flash_attn_decode(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            128,
            0.0883883,  // 1 / sqrt(128)
        );
        assert!(result.is_err());
    }
}
