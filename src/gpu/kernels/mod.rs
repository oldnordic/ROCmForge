//! GPU kernel wrappers organized by functionality.
//!
//! Safety-first design:
//! - All kernels validate bounds before launching
//! - All hipError_t return values checked
//! - Never panic, always return GpuError

pub mod norm;
pub mod rope;
pub mod elementwise;
pub mod attention;
pub mod quant;

pub use norm::{rms_norm, rms_norm_batched};
pub use rope::{rope, rope_batched};
pub use elementwise::{add, mul, scale, gelu, silu, add_batched, mul_batched, zero_fill};
pub use attention::{kv_write, kv_write_batched, flash_attn_decode, flash_attn_prefill};
pub use quant::{
    quantize_q4_k,
    dequantize_q4_k,
    dequantize_q4_k_batched,
    verify_q4_k_accuracy,
    finalize_q4_k_metrics,
    quantize_q5_k,
    dequantize_q5_k,
    dequantize_q5_k_batched,
    verify_q5_k_accuracy,
    finalize_q5_k_metrics,
    quantize_q8_0,
    dequantize_q8_0,
    dequantize_q8_0_batched,
    verify_q8_0_accuracy,
    finalize_q8_0_metrics,
    gemv_q8_0_f32,
    gemv_q4_k_f32,
    gemv_q5_k_f32,
};
