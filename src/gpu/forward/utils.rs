use crate::aligned::AlignedVec;
use crate::config::ModelConfig;
use crate::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use crate::cpu::quant::load_f16_scale;
use crate::gpu::decode_profile::decode_stage_profiling_enabled;
use crate::gpu::device::GpuDevice;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernels::elementwise::add_on_stream;
use crate::gpu::safety::decode_graph_disabled_override_requested;
use crate::gpu::weights::{GpuBuffer, GpuModelWeights, WeightMeta};

pub(super) fn bytes_of_f32_slice(src: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

pub(super) fn bytes_of_i32_slice(src: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

pub(super) fn bytes_of_f32_slice_mut(dst: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, std::mem::size_of_val(dst))
    }
}

pub(super) fn upload_f32(dst: &mut GpuBuffer, src: &[f32]) -> GpuResult<()> {
    dst.copy_from_host(bytes_of_f32_slice(src))
}

pub(super) fn upload_i32(dst: &mut GpuBuffer, src: &[i32]) -> GpuResult<()> {
    dst.copy_from_host(bytes_of_i32_slice(src))
}

pub(super) fn download_f32(src: &GpuBuffer, dst: &mut [f32]) -> GpuResult<()> {
    src.copy_to_host(bytes_of_f32_slice_mut(dst))
}

pub(super) fn cpu_fallback_error(op: &str, err: impl std::fmt::Display) -> GpuError {
    GpuError::HipApiError {
        code: -1,
        description: format!("{} CPU fallback failed: {}", op, err),
    }
}

pub(super) fn decode_graph_disabled(gpu_weights: &GpuModelWeights) -> bool {
    decode_stage_profiling_enabled()
        || decode_graph_disabled_override_requested()
        // Sparse / MPO LM heads cannot be captured in a HIP graph
        // because the dispatch functions perform dynamic indexing.
        || gpu_weights.lm_head.as_dense().is_none()
        || gpu_weights.has_unsupported_gpu_gemv_weights()
}

pub(super) fn lm_head_is_dense(gpu_weights: &GpuModelWeights) -> bool {
    gpu_weights.lm_head.as_dense().is_some()
}

pub(super) fn validate_token_embedding_layout(
    meta: &WeightMeta,
    config: &ModelConfig,
) -> GpuResult<()> {
    if meta.dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "token_emb".to_string(),
            dims: meta.dims.clone(),
            reason: "embedding weights must have at least 2 dimensions".to_string(),
        });
    }

    let hidden_size = meta.dims[0] as usize;
    let vocab_size = meta.dims[1] as usize;
    if hidden_size != config.hidden_size || vocab_size != config.vocab_size {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "token_emb".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "expected [{}, {}], got [{}, {}]",
                config.hidden_size, config.vocab_size, hidden_size, vocab_size
            ),
        });
    }
    if meta.needs_transpose {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "token_emb".to_string(),
            dims: meta.dims.clone(),
            reason: "token embeddings must not require transpose for GPU lookup".to_string(),
        });
    }

    Ok(())
}

pub(super) fn residual_add_inplace(
    device: &GpuDevice,
    hidden: &GpuBuffer,
    residual: &GpuBuffer,
    len: usize,
) -> GpuResult<()> {
    add_on_stream(
        hidden.as_ptr() as *const f32,
        residual.as_ptr() as *const f32,
        hidden.as_ptr() as *mut f32,
        len,
        device.stream(),
    )
}

/// Controls how the hybrid GPU path handles final logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuLogitsMode {
    /// Skip final norm and output projection entirely.
    Skip,
    /// Download full logits to host memory.
    DownloadToHost,
    /// Keep logits on GPU and return the greedy token via GPU argmax.
    GreedyArgmax,
}

pub(super) fn cpu_fallback_gemv(
    op: &str,
    weights: &[u8],
    meta: &crate::cpu::weights::WeightMeta,
    input_gpu: &GpuBuffer,
    input_host: &mut [f32],
    output_host: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    q8_scratch: &mut [u8],
) -> GpuResult<()> {
    download_f32(input_gpu, &mut input_host[..in_dim])?;
    cpu_dispatch_gemv(
        weights,
        meta,
        &input_host[..in_dim],
        &mut output_host[..out_dim],
        out_dim,
        in_dim,
        Some(q8_scratch),
    )
    .map_err(|e| cpu_fallback_error(op, e))?;

    Ok(())
}

pub(super) fn cpu_fallback_gemv_and_upload(
    op: &str,
    weights: &[u8],
    meta: &crate::cpu::weights::WeightMeta,
    input_gpu: &GpuBuffer,
    input_host: &mut AlignedVec<f32>,
    output_host: &mut AlignedVec<f32>,
    output_gpu: &mut GpuBuffer,
    out_dim: usize,
    in_dim: usize,
    q8_scratch: &mut [u8],
) -> GpuResult<()> {
    ensure_size(input_host, in_dim);
    ensure_size(output_host, out_dim);
    cpu_fallback_gemv(
        op,
        weights,
        meta,
        input_gpu,
        input_host,
        output_host,
        out_dim,
        in_dim,
        q8_scratch,
    )?;
    upload_f32(output_gpu, &output_host[..out_dim])?;
    Ok(())
}

pub(super) fn ensure_size(v: &mut AlignedVec<f32>, size: usize) {
    if v.len() < size {
        v.resize(size, 0.0);
    }
}
