//! GPU kernel wrappers organized by functionality.
//!
//! Safety-first design:
//! - All kernels validate bounds before launching
//! - All hipError_t return values checked
//! - Never panic, always return GpuError

pub mod attention;
pub mod elementwise;
pub mod mpo;
pub mod norm;
pub mod q8_decode;
pub mod q8_gemv;
pub mod quant;
pub mod quant_gqa;
pub mod rope;
pub mod sparse_csr;
pub mod ssm;

/// Check if a pointer is aligned to a given boundary.
pub(crate) fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    (ptr as usize).is_multiple_of(alignment)
}

pub use attention::{
    flash_attn_decode, flash_attn_decode_strided, flash_attn_decode_strided_multi_head,
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_prefill_strided, kv_write,
    kv_write_batched, kv_write_batched_compressed, kv_write_compressed,
    kv_write_from_state_on_stream, kv_write_on_stream, kv_write_rope_from_state_on_stream,
    kv_write_rope_on_stream, reconstruct_kv_cache_prefix_sum,
};
pub use elementwise::{
    add, add_batched, add_on_stream, argmax_f32, argmax_f32_on_stream, dot_f16_f32_on_stream,
    embed_q4_0_batch, embed_q4_0_token, embed_q8_0_batch, embed_q8_0_token, gelu,
    increment_decode_state_on_stream, mul, mul_batched, mul_on_stream, scale, silu, silu_on_stream,
    weighted_add_on_stream, zero_fill,
};
pub use norm::{rms_norm, rms_norm_batched, rms_norm_on_stream, rms_norm_vulkan_style};
pub use q8_decode::{
    gemv_gate_up_q4_0_q8_0_on_stream, gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant,
    gemv_gate_up_swiglu_q4_0_q8_0_on_stream, gemv_q4_0_f32_q8_inline_residual_on_stream,
    gemv_q4_0_f32_q8_inline_residual_on_stream_variant, gemv_q4_0_q8_0_on_stream,
    gemv_q4_0_q8_0_residual_on_stream, q8_0_workspace_bytes, quantize_q8_0_on_stream,
};
pub use q8_gemv::{
    gemv_q8_0_f32, gemv_q8_0_f32_lm_head, gemv_q8_0_f32_lm_head_on_stream,
    gemv_q8_0_f32_lm_head_on_stream_variant, gemv_q8_0_f32_on_stream,
};
pub use quant::{
    dequantize_q4_0, dequantize_q4_0_batched, dequantize_q4_1, dequantize_q4_1_batched,
    dequantize_q4_k, dequantize_q4_k_batched, dequantize_q5_k, dequantize_q5_k_batched,
    dequantize_q6_k, dequantize_q6_k_batched, dequantize_q8_0, dequantize_q8_0_batched,
    finalize_q4_0_metrics, finalize_q4_1_metrics, finalize_q4_k_metrics, finalize_q5_k_metrics,
    finalize_q8_0_metrics, gemm_q4_0_f32, gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32,
    gemm_q6_k_f32, gemm_q8_0_f32, gemv_gate_up_q4_0_f32, gemv_gate_up_q4_0_f32_on_stream,
    gemv_gate_up_swiglu_q4_0_f32, gemv_gate_up_swiglu_q4_0_f32_on_stream,
    gemv_norm_qkv_rope_kvwrite_q4_0_f32_dp4a_on_stream, gemv_q4_0_f32, gemv_q4_0_f32_on_stream,
    gemv_q4_0_f32_on_stream_unchecked, gemv_q4_0_f32_residual_on_stream,
    gemv_q4_0_f32_residual_on_stream_unchecked, gemv_q4_0_f32_vulkan_style,
    gemv_q4_0_f32_wave32_on_stream_unchecked, gemv_q4_0_f32_wave32_residual_on_stream_unchecked,
    gemv_q4_1_f32, gemv_q4_1_f32_on_stream, gemv_q4_1_f32_on_stream_unchecked,
    gemv_q4_1_f32_residual_on_stream, gemv_q4_1_f32_residual_on_stream_unchecked,
    gemv_q4_1_f32_residual_on_stream_variant_unchecked, gemv_q4_1_f32_wave32_on_stream_unchecked,
    gemv_q4_1_f32_wave32_residual_on_stream_unchecked,
    /* DISABLED: gemv_q4_k_f32 and gemv_q4_k_f32_on_stream not available */
    /* gemv_q4_k_f32, gemv_q4_k_f32_on_stream, */
    gemv_q4_k_f32_vulkan_style,
    /* DISABLED: gemv_q5_k_f32 and gemv_q5_k_f32_on_stream not available */
    /* gemv_q5_k_f32, gemv_q5_k_f32_on_stream, */
    gemv_q6_k_f32, gemv_q6_k_f32_on_stream, gemv_qkv_q4_0_f32, gemv_qkv_q4_0_f32_on_stream,
    gemv_qkv_q4_0_f32_on_stream_variant, quantize_q4_0, quantize_q4_1, quantize_q4_k,
    quantize_q5_k, quantize_q6_k, quantize_q8_0, verify_q4_0_accuracy, verify_q4_1_accuracy,
    verify_q4_k_accuracy, verify_q5_k_accuracy, verify_q6_k_accuracy, verify_q8_0_accuracy,
};
pub use quant_gqa::fused_qkv_rope_q4_0_gqa_on_stream;
pub use rope::{
    rope, rope_batched, rope_heads, rope_heads_batched, rope_heads_from_state_on_stream,
    rope_heads_on_stream,
};
// Re-export batched kernels for prefill
pub use mpo::dispatch_mpo_apply_f32;
pub use quant::batched::{
    batched_fused_gate_up_q4_0_f32, batched_gemm_q4_0_f32, batched_gemm_q4_1_f32,
    wmma_matmul_q4_0_f32,
};
pub use sparse_csr::dispatch_sparse_csr_gemv_f32;
pub use ssm::{
    dispatch_batched_conv1d_silu, dispatch_batched_fused_qk_l2_norm_scale,
    dispatch_batched_fused_sigmoid_alpha_gate, dispatch_batched_gated_delta_net,
    dispatch_batched_gated_norm, dispatch_conv1d_silu, dispatch_fused_qk_l2_norm_scale,
    dispatch_fused_sigmoid_alpha_gate, dispatch_gated_delta_net, dispatch_gated_norm,
    dispatch_repeat_interleave_qk,
};
