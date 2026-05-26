//! GQA-aware QKV fusion kernels
//!
//! Fusion of QKV projection + RoPE for Grouped Query Attention (GQA).
//! Reduces kernel launches from 4 (Q,K,V,RoPE) → 1 per layer.

use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_int;

/// Fused QKV projection + RoPE for GQA (Q4_0 quantization)
///
/// Combines 4 operations into 1 kernel launch:
/// - Q projection (from quantized weights)
/// - K projection (from quantized weights)
/// - V projection (from quantized weights)
/// - RoPE application (to Q and K)
///
/// # GQA Grouping
/// For n_heads=14, n_kv_heads=2:
/// - KV head 0 serves query heads 0-6 (7 query heads)
/// - KV head 1 serves query heads 7-13 (7 query heads)
///
/// # Arguments
/// * `w_q` - Quantized query weights [n_heads * hidden_size * sizeof_q4_0]
/// * `w_k` - Quantized key weights [n_kv_heads * hidden_size * sizeof_q4_0]
/// * `w_v` - Quantized value weights [n_kv_heads * hidden_size * sizeof_q4_0]
/// * `input` - Input hidden states [hidden_size]
/// * `out_q` - Output query states (RoPE'd) [n_heads * head_dim]
/// * `out_k` - Output key states (RoPE'd) [n_kv_heads * head_dim]
/// * `out_v` - Output value states [n_kv_heads * head_dim]
/// * `pos` - Current position for RoPE
/// * `n_heads` - Number of query heads
/// * `n_kv_heads` - Number of KV heads
/// * `head_dim` - Dimension per head
/// * `rope_theta` - RoPE theta (10000.0 for Qwen2)
/// * `rope_neox` - Whether to use Neox-style RoPE
/// * `stream` - HIP stream
///
/// # Returns
/// Ok(()) on success, Err if kernel launch fails
///
/// # Safety
/// - All memory pointers must be valid GPU pointers
/// - Bounds are validated on CPU before kernel launch
pub fn fused_qkv_rope_q4_0_gqa_on_stream(
    device: &GpuDevice,
    w_q: *const u8,
    w_k: *const u8,
    w_v: *const u8,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rope_neox: bool,
    stream: hipStream_t,
) -> GpuResult<()> {
    if input.is_null() || out_q.is_null() || out_k.is_null() || out_v.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "fused_qkv_rope_q4_0_gqa: null pointer".to_string(),
        });
    }

    if n_kv_heads == 0 || n_heads == 0 || head_dim == 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "fused_qkv_rope_q4_0_gqa: invalid dimensions n_heads={} n_kv_heads={} head_dim={}",
                n_heads, n_kv_heads, head_dim
            ),
        });
    }

    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "fused_qkv_rope_q4_0_gqa: n_heads must be divisible by n_kv_heads ({} % {} != 0)",
                n_heads, n_kv_heads
            ),
        });
    }

    let result = unsafe {
        fused_qkv_rope_q4_0_gqa_launch(
            w_q as *const _,
            w_k as *const _,
            w_v as *const _,
            input,
            out_q,
            out_k,
            out_v,
            pos as c_int,
            n_heads as c_int,
            n_kv_heads as c_int,
            head_dim as c_int,
            rope_theta,
            rope_neox,
            (n_heads * head_dim) as c_int, // hidden_size
            stream,
        )
    };

    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError {
            code: result as i32,
            description: format!("fused_qkv_rope_q4_0_gqa kernel failed: {:?}", result),
        });
    }

    Ok(())
}

// External FFI declaration
extern "C" {
    fn fused_qkv_rope_q4_0_gqa_launch(
        w_q: *const u8,
        w_k: *const u8,
        w_v: *const u8,
        input: *const f32,
        out_q: *mut f32,
        out_k: *mut f32,
        out_v: *mut f32,
        pos: c_int,
        n_heads: c_int,
        n_kv_heads: c_int,
        head_dim: c_int,
        rope_theta: f32,
        rope_neox: bool,
        hidden_size: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}
