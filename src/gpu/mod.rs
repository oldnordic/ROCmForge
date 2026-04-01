//! AMD GPU inference backend with HIP.
//!
//! Safety-first design:
//! - All HIP API calls wrapped with error handling
//! - Never panic, always return GpuError
//! - CPU fallback when GPU unavailable

pub mod arch;
pub mod cache;
pub mod detect;
pub mod device;
pub mod dynamic_loader;
pub mod error;
pub mod ffi;
pub mod forward;
pub mod graph;
pub mod kernels;
pub mod ops;
pub mod quant;
pub mod quant_wrapper;
pub mod weights;

pub use arch::GpuArchitecture;
pub use cache::{GpuForwardScratch, GpuKvCache, GpuPrefillScratch};
pub use detect::GpuCapabilities;
pub use device::GpuDevice;
pub use dynamic_loader::{library_info, DynamicLibrary, LibraryInfo};
pub use error::{GpuError, GpuResult};
pub use forward::{
    decode_stage_profile_snapshot, gpu_embed_token_hybrid, gpu_full_forward_hybrid,
    gpu_layer_forward_hybrid, gpu_prefill_forward_hybrid, gpu_prefill_layer_forward_hybrid,
    reset_decode_stage_profile, GpuDecodeStageProfileSnapshot, GpuLogitsMode,
};
pub use graph::{CapturedDecodeGraph, DecodeGraphKey, HipGraph, HipGraphExec};
pub use kernels::{
    add, add_batched, argmax_f32, dequantize_q4_0, dequantize_q4_0_batched, dequantize_q4_1,
    dequantize_q4_1_batched, dequantize_q4_k, dequantize_q4_k_batched, dequantize_q5_k,
    dequantize_q5_k_batched, dequantize_q8_0, dequantize_q8_0_batched, embed_q8_0_batch,
    embed_q8_0_token, finalize_q4_0_metrics, finalize_q4_1_metrics, finalize_q4_k_metrics,
    finalize_q5_k_metrics, finalize_q8_0_metrics, flash_attn_decode, flash_attn_decode_strided,
    flash_attn_decode_strided_multi_head, flash_attn_prefill_strided, gelu, gemm_q4_0_f32,
    gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q8_0_f32, gemv_gate_up_swiglu_q4_0_f32,
    gemv_q4_0_f32, gemv_q4_1_f32, gemv_q4_k_f32, gemv_q5_k_f32, gemv_q8_0_f32,
    gemv_q8_0_f32_lm_head, gemv_qkv_q4_0_f32, kv_write, kv_write_batched, mul, mul_batched,
    quantize_q4_0, quantize_q4_1, quantize_q4_k, quantize_q5_k, quantize_q8_0, rms_norm,
    rms_norm_batched, rope, rope_batched, rope_heads, rope_heads_batched, scale, silu,
    verify_q4_0_accuracy, verify_q4_1_accuracy, verify_q4_k_accuracy, verify_q5_k_accuracy,
    verify_q8_0_accuracy, zero_fill,
};
pub use ops::{
    gpu_dispatch_fused_gate_up, gpu_dispatch_fused_qkv, gpu_dispatch_gemm, gpu_dispatch_gemv,
};
pub use quant::{
    Q4KBlock, Q4_0Block, Q4_1Block, Q5KBlock, Q8_0Block, K_SCALE_SIZE, Q4_0_BLOCK_SIZE,
    Q4_1_BLOCK_SIZE, Q4_K_BLOCK_SIZE, Q5_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE, Q8_0_MAX, QK4_0, QK4_1,
    QK8_0, QK_K,
};
pub use quant_wrapper::GpuQuant;
pub use weights::{GpuBuffer, GpuLayerWeights, GpuModelWeights, TensorRole, WeightMeta};

/// Detect AMD GPU capabilities (safe wrapper).
pub fn detect() -> Option<GpuCapabilities> {
    GpuCapabilities::detect()
}
