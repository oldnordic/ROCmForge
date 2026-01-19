//! HIP backend ops for ggml IR.

pub mod accumulate;
pub mod add_scale;
pub mod batch_quantized;
pub mod copy;
pub mod fused_ops;
pub mod get_rows;
pub mod matmul;
pub mod mask;
pub mod q4_0_dequant;
pub mod q4_k_dequant;
pub mod q6_k_dequant;
pub mod quantized_matmul;
pub mod rms_norm;
pub mod rope;
pub mod softmax;
pub mod swiglu;
pub mod split_qkv;

// Public exports for batch quantized operations
pub use batch_quantized::{QuantFormat, QuantizedMatmulOp, BatchQuantizedMatmul, BatchMatmulResult};

#[cfg(feature = "rocm")]
pub use batch_quantized::{AsyncKernelLauncher, AsyncHandle};

// Public exports for Q4_0 dequantization operations
pub use q4_0_dequant::{
    dequantize_q4_0_with_fallback,
    dequantize_q4_0_cpu,
};

// GPU-only exports (require ROCm feature)
#[cfg(feature = "rocm")]
pub use q4_0_dequant::{
    dequantize_q4_0_kernel_cached,
    get_or_init_q4_0_dequant_cache,
    dequantize_q4_0_cpu_upload,
};

// Public exports for Q4_K dequantization operations
pub use q4_k_dequant::{
    dequantize_q4_k_with_fallback,
    dequantize_q4_k_gpu_kernel,
    get_or_init_q4_k_dequant_cache,
    dequantize_q4_k_cpu,
};

// Public exports for Q6_K dequantization operations
pub use q6_k_dequant::{
    dequantize_q6_k_with_fallback,
    dequantize_q6_k_gpu_kernel,
    get_or_init_q6_k_dequant_cache,
    dequantize_q6_k_cpu,
};
