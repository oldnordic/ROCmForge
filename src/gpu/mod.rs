//! AMD GPU inference backend with HIP.
//!
//! Safety-first design:
//! - All HIP API calls wrapped with error handling
//! - Never panic, always return GpuError
//! - CPU fallback when GPU unavailable

pub mod arch;
pub mod cache;
mod decode_graph_keys;
pub mod decode_profile;
pub mod detect;
pub mod device;
pub mod dynamic_loader;
pub mod error;
pub mod ffi;
pub mod forward;
pub mod graph;
pub mod kernels;
pub mod launch_autotune;
pub mod ops;
pub mod quant;
pub mod quant_wrapper;
pub mod safety;
pub mod weights;

pub use arch::GpuArchitecture;
pub use cache::{GpuForwardScratch, GpuKvCache, GpuPrefillScratch};
pub use decode_profile::{
    decode_stage_profile_snapshot, reset_decode_stage_profile, GpuDecodeStageProfileSnapshot,
};
pub use detect::GpuCapabilities;
pub use device::GpuDevice;
pub use dynamic_loader::{library_info, DynamicLibrary, LibraryInfo};
pub use error::{GpuError, GpuResult};
pub use forward::{
    gpu_embed_token_hybrid, gpu_full_forward_hybrid, gpu_layer_forward_hybrid,
    gpu_prefill_forward_hybrid, gpu_prefill_layer_forward_hybrid, GpuLogitsMode,
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
pub use launch_autotune::{
    launch_autotune_enabled, refresh_launch_autotune_state, select_gate_up_swiglu_q8_variant,
    select_lm_head_q8_variant, select_q4_0_q8_residual_variant, select_q4_1_residual_variant,
    select_qkv_variant, select_variant, OpType, ShapeKey, VariantId,
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
pub use safety::{
    decode_graph_enabled, experimental_ffn_fastpath_enabled, experimental_gpu_kernels_enabled,
    experimental_q8_activation_fastpath_enabled, gpu_safe_mode_enabled,
    real_model_gpu_tests_enabled, refresh_runtime_env_flags, run_experimental_gpu_tests_enabled,
    run_gpu_benches_enabled, DISABLE_DECODE_GRAPH_ENV, ENABLE_DECODE_GRAPH_ENV,
    ENABLE_EXPERIMENTAL_FFN_FASTPATH_ENV, ENABLE_EXPERIMENTAL_GPU_KERNELS_ENV,
    ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH_ENV, ENABLE_LAUNCH_AUTOTUNE_ENV, GPU_SAFE_MODE_ENV,
    RUN_EXPERIMENTAL_GPU_TESTS_ENV, RUN_GPU_BENCHES_ENV, RUN_REAL_MODEL_GPU_TESTS_ENV,
};
pub use weights::{GpuBuffer, GpuLayerWeights, GpuModelWeights, TensorRole, WeightMeta};

/// Detect AMD GPU capabilities (safe wrapper).
pub fn detect() -> Option<GpuCapabilities> {
    GpuCapabilities::detect()
}
