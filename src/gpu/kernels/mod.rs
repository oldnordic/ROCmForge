//! GPU kernel wrappers organized by functionality.
//!
//! Safety-first design:
//! - All kernels validate bounds before launching
//! - All hipError_t return values checked
//! - Never panic, always return GpuError

pub mod attention;
pub mod elementwise;
pub mod norm;
pub mod quant;
pub mod rope;

pub use attention::{
    flash_attn_decode_strided_multi_head,
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_prefill_strided, kv_write,
    kv_write_batched, kv_write_from_state_on_stream, kv_write_on_stream,
    kv_write_rope_from_state_on_stream,
};
pub use elementwise::{
    add, add_batched, add_on_stream, argmax_f32, argmax_f32_on_stream, embed_q8_0_batch,
    embed_q8_0_token, gelu, mul, mul_batched, scale, silu, zero_fill,
};
pub use norm::{rms_norm, rms_norm_batched, rms_norm_on_stream};
pub use quant::{
    dequantize_q4_0, dequantize_q4_0_batched, dequantize_q4_1, dequantize_q4_1_batched,
    dequantize_q4_k, dequantize_q4_k_batched, dequantize_q5_k, dequantize_q5_k_batched,
    dequantize_q8_0, dequantize_q8_0_batched, finalize_q4_0_metrics, finalize_q4_1_metrics,
    finalize_q4_k_metrics, finalize_q5_k_metrics, finalize_q8_0_metrics, gemm_q4_0_f32,
    gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q8_0_f32, gemv_gate_up_swiglu_q4_0_f32,
    gemv_gate_up_swiglu_q4_0_f32_on_stream, gemv_q4_0_f32, gemv_q4_0_f32_on_stream,
    gemv_q4_0_f32_residual_on_stream, gemv_q4_1_f32, gemv_q4_1_f32_on_stream, gemv_q4_k_f32,
    gemv_q4_k_f32_on_stream, gemv_q5_k_f32, gemv_q5_k_f32_on_stream, gemv_q8_0_f32,
    gemv_q8_0_f32_lm_head, gemv_q8_0_f32_lm_head_on_stream, gemv_q8_0_f32_on_stream,
    gemv_qkv_q4_0_f32, gemv_qkv_q4_0_f32_on_stream, quantize_q4_0, quantize_q4_1, quantize_q4_k,
    quantize_q5_k, quantize_q8_0, verify_q4_0_accuracy, verify_q4_1_accuracy, verify_q4_k_accuracy,
    verify_q5_k_accuracy, verify_q8_0_accuracy,
};
pub use rope::{
    rope, rope_batched, rope_heads, rope_heads_batched, rope_heads_from_state_on_stream,
    rope_heads_on_stream,
};
