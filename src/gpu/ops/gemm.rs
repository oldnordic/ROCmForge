use super::super::device::GpuDevice;
use super::super::error::{GpuError, GpuResult};
use super::super::weights::{GpuBuffer, WeightMeta};
use crate::gpu::kernel_dispatch_profile::record_gemm_dispatch;
use crate::gpu::kernels::{
    batched_gemm_q4_0_f32, batched_gemm_q4_1_f32, gemm_q4_k_f32_on_stream, gemm_q5_0_f32_on_stream,
    gemm_q5_1_f32_on_stream, gemm_q5_k_f32_on_stream, gemm_q6_k_f32_on_stream,
    gemm_q8_0_f32_on_stream,
};
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
        record_gemm_dispatch("Q4_0", "batched");
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

    if seq_len > 1 && meta.wtype == GgmlType::Q4_1 {
        record_gemm_dispatch("Q4_1", "batched");
        return batched_gemm_q4_1_f32(
            weights.as_ptr(),
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q4_K {
        record_gemm_dispatch("Q4_K", "on_stream");
        return gemm_q4_k_f32_on_stream(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q5_K {
        record_gemm_dispatch("Q5_K", "on_stream");
        return gemm_q5_k_f32_on_stream(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q6_K {
        record_gemm_dispatch("Q6_K", "on_stream");
        return gemm_q6_k_f32_on_stream(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q8_0 {
        record_gemm_dispatch("Q8_0", "on_stream");
        return gemm_q8_0_f32_on_stream(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q5_0 {
        record_gemm_dispatch("Q5_0", "on_stream");
        return gemm_q5_0_f32_on_stream(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    if seq_len > 1 && meta.wtype == GgmlType::Q5_1 {
        record_gemm_dispatch("Q5_1", "on_stream");
        return gemm_q5_1_f32_on_stream(
            weights.as_ptr() as *const u8,
            input,
            output,
            in_dim,
            out_dim,
            seq_len,
            device.stream(),
        );
    }

    Err(GpuError::UnsupportedOperation {
        operation: format!("gpu_dispatch_gemm for {:?} seq_len={}", meta.wtype, seq_len),
        reason: "GEMM kernel not implemented for this quant type. Use GEMV (seq_len=1) for now."
            .to_string(),
    })
}
