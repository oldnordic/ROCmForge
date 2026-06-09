//! GPU math dispatch for model weights.
//!
//! This layer validates GGUF metadata before calling the raw HIP kernels.

mod fastpath;
mod gate_up;
mod gemm;
mod gemv;
mod gemv_residual;
mod norm;
mod qkv;

pub use gate_up::{gpu_dispatch_fused_gate_up, gpu_dispatch_fused_gate_up_on_stream};
pub use gemm::gpu_dispatch_gemm;
pub use gemv::{
    gpu_dispatch_gemv, gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_ptr_on_stream,
    gpu_dispatch_gemv_svd_on_stream, gpu_dispatch_gemv_with_fallback_on_stream,
    gpu_dispatch_mpo_apply_on_stream, gpu_dispatch_sparse_csr_gemv_on_stream,
};
pub use gemv_residual::gpu_dispatch_gemv_residual_on_stream;
pub use norm::gpu_dispatch_rms_norm;
pub use qkv::{
    gpu_dispatch_fused_qkv, gpu_dispatch_fused_qkv_gqa_on_stream, gpu_dispatch_fused_qkv_on_stream,
};

use super::error::{GpuError, GpuResult};
use super::weights::{TensorRole, WeightMeta};
use crate::loader::GgmlType;

pub(crate) fn supports_gemv_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q2_K
            | GgmlType::Q3_K
    )
}

fn is_lm_head_role(role: TensorRole) -> bool {
    matches!(role, TensorRole::LmHead | TensorRole::TiedLmHead)
}

fn config_num_heads(q_size: usize, h: usize) -> usize {
    if h.is_multiple_of(128) {
        q_size / 128
    } else {
        q_size / 64
    }
}

fn validate_gemv_layout(meta: &WeightMeta, out_dim: usize, in_dim: usize) -> GpuResult<()> {
    if meta.dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: "weight metadata must describe a 2D matrix".to_string(),
        });
    }

    if (meta.dims[0] as usize == in_dim && meta.dims[1] as usize == out_dim)
        || (meta.dims[0] as usize == out_dim && meta.dims[1] as usize == in_dim)
    {
        Ok(())
    } else {
        Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "shape mismatch: matrix is {:?}, but vector is [{}] and output is [{}]",
                meta.dims, in_dim, out_dim
            ),
        })
    }
}
