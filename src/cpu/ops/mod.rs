//! CPU primitive operations for inference.

pub mod activation;
pub mod arithmetic;
pub mod attention;
pub mod avx2;
pub mod gemm;
pub mod gemv;
pub mod norm;
pub mod rope;

pub use activation::{argmax, online_softmax_update, silu, silu_fuse, softmax};
pub use arithmetic::{add_bias, add_bias_batched, residual_add, residual_add_batched};
pub use attention::{flash_attn_decode, flash_attn_prefill};
pub use avx2::{
    dot_f32_avx2, dot_q4_0_block_avx2, dot_q4_0_q8_0_block_avx2, dot_q4_1_q8_0_block_avx2,
};
pub use gemm::{
    dispatch_gemm, dispatch_gemm_transposed, gemm_f32, gemm_q3_k_fallback, gemm_q4_0,
    gemm_q4_0_transposed_gemm, gemm_q4_1, gemm_q4_1_transposed_gemm, gemm_q5_0,
    gemm_q5_0_transposed, gemm_q5_k_fallback, gemm_q6_k_fallback, gemm_q8_0,
    gemm_q8_0_transposed_gemm,
};
pub use gemv::{
    dispatch_gemv, dispatch_gemv_transposed, gemv_f32, gemv_q4_0, gemv_q4_0_q8_0,
    gemv_q4_0_transposed, gemv_q4_1_q8_0, gemv_q4_1_transposed, gemv_q5_0, gemv_q5_k, gemv_q6_k,
    gemv_q8_0, gemv_q8_0_transposed,
};
pub use norm::{rms_norm, rms_norm_batch};
pub use rope::{rope, rope_batch, rope_with_pos};
