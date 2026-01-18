//! HIP backend ops for ggml IR.

pub mod accumulate;
pub mod add_scale;
pub mod copy;
pub mod get_rows;
pub mod matmul;
pub mod mask;
pub mod q4_0_dequant;
pub mod quantized_matmul;
pub mod rms_norm;
pub mod rope;
pub mod softmax;
pub mod swiglu;
pub mod split_qkv;
