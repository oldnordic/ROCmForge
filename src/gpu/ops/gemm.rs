use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::weights::{GpuBuffer, WeightMeta};
use crate::gpu::kernels::batched_gemm_q4_0_f32;
use crate::loader::GgmlType;

use super::gemv::gpu_dispatch_gemv;
use super::supports_gemv_type;

pub fn gpu_dispatch_gemm(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if seq_len == 1 && supports_gemv_type(meta.wtype) {
        return gpu_dispatch_gemv(device, weights, meta, input, output, out_dim, in_dim);
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q4_0 {
        return batched_gemm_q4_0_f32(
            weights.as_ptr(),
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    Err(GpuError::UnsupportedOperation {
        operation: format!("gpu_dispatch_gemm for seq_len={}", seq_len),
        reason: "GEMM kernels not yet implemented. Use GEMV (seq_len=1) for now.".to_string(),
    })
}
