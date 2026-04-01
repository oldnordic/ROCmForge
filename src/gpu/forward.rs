//! Hybrid GPU decode forward path.
//!
//! This is the smallest end-to-end GPU runner for the current kernel set:
//! most layer math stays on HIP, while unsupported embedding and logits paths
//! may still use the existing CPU implementation.

use super::cache::{GpuForwardScratch, GpuKvCache, GpuPrefillScratch};
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi;
use super::graph::{DecodeGraphKey, DecodeGraphScope, HipGraph};
use super::kernels::attention::{
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_prefill_strided,
    kv_write_rope_from_state_on_stream, kv_write_rope_on_stream,
};
use super::kernels::elementwise::{
    add, add_batched, add_on_stream, argmax_f32, argmax_f32_on_stream, embed_q8_0_batch,
    embed_q8_0_token, silu,
};
use super::kernels::norm::{rms_norm, rms_norm_batched, rms_norm_on_stream};
use super::kernels::rope::{
    rope_heads_batched, rope_heads_from_state_on_stream, rope_heads_on_stream,
};
use super::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_fused_qkv_on_stream, gpu_dispatch_gemm,
    gpu_dispatch_gemv, gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_residual_on_stream,
    gpu_dispatch_rms_norm,
};
use super::weights::{GpuBuffer, GpuLayerWeights, GpuModelWeights, WeightMeta};
use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::forward::cpu_embed_token;
use crate::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use crate::cpu::weights::CpuModelWeights;
use crate::loader::GgmlType;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn bytes_of_f32_slice(src: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

fn bytes_of_i32_slice(src: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

fn bytes_of_f32_slice_mut(dst: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, std::mem::size_of_val(dst))
    }
}

fn upload_f32(dst: &mut GpuBuffer, src: &[f32]) -> GpuResult<()> {
    dst.copy_from_host(bytes_of_f32_slice(src))
}

fn upload_i32(dst: &mut GpuBuffer, src: &[i32]) -> GpuResult<()> {
    dst.copy_from_host(bytes_of_i32_slice(src))
}

fn download_f32(src: &GpuBuffer, dst: &mut [f32]) -> GpuResult<()> {
    src.copy_to_host(bytes_of_f32_slice_mut(dst))
}

fn cpu_fallback_error(op: &str, err: impl std::fmt::Display) -> GpuError {
    GpuError::HipApiError {
        code: -1,
        description: format!("{} CPU fallback failed: {}", op, err),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDecodeStageProfileSnapshot {
    pub layer_invocations: u64,
    pub tail_invocations: u64,
    pub attn_norm_ns: u128,
    pub qkv_ns: u128,
    pub q_rope_ns: u128,
    pub k_rope_ns: u128,
    pub kv_write_ns: u128,
    pub attention_ns: u128,
    pub attn_proj_ns: u128,
    pub attn_residual_ns: u128,
    pub ffn_norm_ns: u128,
    pub gate_up_ns: u128,
    pub ffn_down_ns: u128,
    pub ffn_residual_ns: u128,
    pub logits_norm_ns: u128,
    pub logits_proj_ns: u128,
    pub argmax_ns: u128,
}

#[derive(Clone, Copy, Debug)]
enum DecodeStage {
    AttnNorm,
    Qkv,
    QRope,
    KRope,
    KvWrite,
    Attention,
    AttnProj,
    AttnResidual,
    FfnNorm,
    GateUp,
    FfnDown,
    FfnResidual,
    LogitsNorm,
    LogitsProj,
    Argmax,
}

fn decode_stage_profile_store() -> &'static Mutex<GpuDecodeStageProfileSnapshot> {
    static STORE: OnceLock<Mutex<GpuDecodeStageProfileSnapshot>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(GpuDecodeStageProfileSnapshot::default()))
}

fn decode_stage_profiling_enabled() -> bool {
    std::env::var_os("ROCMFORGE_PROFILE_DECODE_STAGES").is_some()
}

fn decode_graph_disabled() -> bool {
    decode_stage_profiling_enabled() || std::env::var_os("ROCMFORGE_DISABLE_DECODE_GRAPH").is_some()
}

fn record_decode_stage(stage: DecodeStage, elapsed_ns: u128) {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    match stage {
        DecodeStage::AttnNorm => guard.attn_norm_ns += elapsed_ns,
        DecodeStage::Qkv => guard.qkv_ns += elapsed_ns,
        DecodeStage::QRope => guard.q_rope_ns += elapsed_ns,
        DecodeStage::KRope => guard.k_rope_ns += elapsed_ns,
        DecodeStage::KvWrite => guard.kv_write_ns += elapsed_ns,
        DecodeStage::Attention => guard.attention_ns += elapsed_ns,
        DecodeStage::AttnProj => guard.attn_proj_ns += elapsed_ns,
        DecodeStage::AttnResidual => guard.attn_residual_ns += elapsed_ns,
        DecodeStage::FfnNorm => guard.ffn_norm_ns += elapsed_ns,
        DecodeStage::GateUp => guard.gate_up_ns += elapsed_ns,
        DecodeStage::FfnDown => guard.ffn_down_ns += elapsed_ns,
        DecodeStage::FfnResidual => guard.ffn_residual_ns += elapsed_ns,
        DecodeStage::LogitsNorm => guard.logits_norm_ns += elapsed_ns,
        DecodeStage::LogitsProj => guard.logits_proj_ns += elapsed_ns,
        DecodeStage::Argmax => guard.argmax_ns += elapsed_ns,
    }
}

fn profile_decode_stage<T>(
    device: &GpuDevice,
    stage: DecodeStage,
    op: impl FnOnce() -> GpuResult<T>,
) -> GpuResult<T> {
    if !decode_stage_profiling_enabled() {
        return op();
    }

    let start = Instant::now();
    let result = op()?;
    device.synchronize()?;
    record_decode_stage(stage, start.elapsed().as_nanos());
    Ok(result)
}

pub fn reset_decode_stage_profile() {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    *guard = GpuDecodeStageProfileSnapshot::default();
}

pub fn decode_stage_profile_snapshot() -> GpuDecodeStageProfileSnapshot {
    *decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

fn validate_token_embedding_layout(meta: &WeightMeta, config: &ModelConfig) -> GpuResult<()> {
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

fn residual_add_inplace(
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

fn cpu_fallback_gemv(
    op: &str,
    weights: &[u8],
    meta: &crate::cpu::weights::WeightMeta,
    input_gpu: &GpuBuffer,
    input_host: &mut [f32],
    output_host: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    q8_scratch: &mut Vec<u8>,
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
    .map_err(|e| cpu_fallback_error(op, e))
}

fn gpu_launch_greedy_argmax(scratch: &mut GpuForwardScratch, vocab_size: usize) -> GpuResult<()> {
    argmax_f32(
        scratch.logits_ptr(),
        scratch.argmax_partial_values_mut_ptr(),
        scratch.argmax_partial_indices_mut_ptr(),
        scratch.argmax_result_index_mut_ptr(),
        vocab_size,
    )
}

fn gpu_read_greedy_argmax_result(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    _vocab_size: usize,
) -> GpuResult<()> {
    unsafe {
        ffi::hip_memcpy_d2h_async(
            scratch.argmax_result_index.as_ptr(),
            scratch.argmax_result_device.as_ptr(),
            std::mem::size_of::<i32>(),
            device.stream(),
        )?;
    }

    Ok(())
}

fn gpu_greedy_argmax_token(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    vocab_size: usize,
) -> GpuResult<u32> {
    gpu_launch_greedy_argmax(scratch, vocab_size)?;
    gpu_read_greedy_argmax_result(device, scratch, vocab_size)?;
    device.synchronize()?;
    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
    Ok(index as u32)
}

fn gpu_launch_greedy_logits_tail_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let v = config.vocab_size;
    if decode_stage_profiling_enabled() {
        let mut guard = decode_stage_profile_store()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.tail_invocations += 1;
    }
    profile_decode_stage(device, DecodeStage::LogitsNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_weights.output_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::LogitsProj, || {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_weights.lm_head,
            &gpu_weights.lm_head_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.logits.as_ptr() as *mut f32,
            v,
            h,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::Argmax, || {
        argmax_f32_on_stream(
            scratch.logits_ptr(),
            scratch.argmax_partial_values_mut_ptr(),
            scratch.argmax_partial_indices_mut_ptr(),
            scratch.argmax_result_index_mut_ptr(),
            v,
            device.stream(),
        )
    })
}

fn gpu_greedy_logits_graph_key(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    config: &ModelConfig,
) -> DecodeGraphKey {
    DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        config,
        GpuLogitsMode::GreedyArgmax,
        gpu_weights.output_norm.as_ptr() as usize,
        gpu_weights.lm_head.as_ptr() as usize,
        gpu_weights.lm_head_meta.wtype,
        gpu_weights.lm_head_meta.role,
    )
    .with_decode_scope(DecodeGraphScope::GreedyTail)
}

fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn gpu_layer_weights_binding_tag(layer: &GpuLayerWeights) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, layer.attn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.attn_q.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_q_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_k.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_k_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_v.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_v_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_o.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_gate.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_up.as_ptr() as usize);
    mix_binding_tag(tag, layer.ffn_down.as_ptr() as usize)
}

fn gpu_model_weights_binding_tag(gpu_weights: &GpuModelWeights) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, gpu_weights.output_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, gpu_weights.lm_head.as_ptr() as usize);
    for layer in &gpu_weights.layers {
        tag ^= gpu_layer_weights_binding_tag(layer);
    }
    tag
}

fn gpu_kv_binding_tag(kv: &GpuKvCache) -> GpuResult<u64> {
    let mut tag = 0u64;
    for layer_idx in 0..kv.num_layers {
        tag = mix_binding_tag(tag, kv.k_ptr(layer_idx)? as usize);
        tag = mix_binding_tag(tag, kv.v_ptr(layer_idx)? as usize);
    }
    Ok(tag)
}

fn gpu_full_decode_graph_key(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &GpuKvCache,
    config: &ModelConfig,
) -> GpuResult<DecodeGraphKey> {
    Ok(DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        config,
        GpuLogitsMode::GreedyArgmax,
        gpu_weights.output_norm.as_ptr() as usize,
        gpu_weights.lm_head.as_ptr() as usize,
        gpu_weights.lm_head_meta.wtype,
        gpu_weights.lm_head_meta.role,
    )
    .with_decode_scope(DecodeGraphScope::FullGreedyDecode)
    .with_layer_weights_binding_tag(gpu_model_weights_binding_tag(gpu_weights))
    .with_kv_binding_tag(gpu_kv_binding_tag(kv)?))
}

fn gpu_capture_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<super::graph::CapturedDecodeGraph> {
    let key = gpu_greedy_logits_graph_key(device, gpu_weights, config);
    device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
    let capture_result =
        gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config);
    let end_capture_result = device.end_capture();

    match capture_result {
        Ok(()) => {
            let graph = HipGraph::from_raw(end_capture_result?);
            super::graph::CapturedDecodeGraph::from_captured_graph(graph, key)
        }
        Err(err) => {
            let _ = end_capture_result;
            Err(err)
        }
    }
}

fn gpu_greedy_logits_tail_token(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<u32> {
    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)?;
    gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
    device.synchronize()?;
    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
    Ok(index as u32)
}

fn gpu_try_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<u32> {
    if decode_graph_disabled() {
        return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config);
    }

    let key = gpu_greedy_logits_graph_key(device, gpu_weights, config);
    if !scratch.has_decode_graph_for(key) {
        scratch.clear_decode_graph();
        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config),
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config);
        }

        match gpu_capture_greedy_decode_graph(device, gpu_weights, scratch, config) {
            Ok(graph) => {
                scratch.replace_decode_graph(graph);
            }
            Err(err @ GpuError::InvalidWeightLayout { .. })
            | Err(err @ GpuError::UnsupportedWeightType { .. }) => return Err(err),
            Err(_) => return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config),
        }
    }

    if let Some(graph) = scratch.decode_graph() {
        if graph.launch(device.stream()).is_ok() {
            gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
            device.synchronize()?;
            let index = scratch.argmax_result_index.as_slice::<i32>()[0];
            return Ok(index as u32);
        }
        scratch.clear_decode_graph();
    }

    gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config)
}

fn gpu_attention_decode(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let seq_len = pos + 1;
    let head_dim = config.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    flash_attn_decode_strided_multi_head_on_stream(
        out_base,
        q_base,
        k_cache,
        v_cache,
        seq_len,
        config.num_heads,
        config.num_kv_heads,
        head_dim,
        scale,
        device.stream(),
    )
}

fn gpu_attention_decode_from_state(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let head_dim = config.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const f32;
    let v_cache = kv.v_ptr(layer_idx)? as *const f32;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    flash_attn_decode_strided_multi_head_from_state_on_stream(
        out_base,
        q_base,
        k_cache,
        v_cache,
        scratch.decode_seq_len_ptr(),
        config.num_heads,
        config.num_kv_heads,
        head_dim,
        scale,
        device.stream(),
    )
}

fn gpu_project_rows(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    seq_len: usize,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    gpu_dispatch_gemm(
        device, weights, meta, input, output, out_dim, in_dim, seq_len,
    )
}

fn gpu_rms_norm_rows(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
) -> GpuResult<()> {
    rms_norm_batched(input, weight, output, hidden_size, eps, seq_len)
}

fn gpu_rope_rows(
    buffer: *mut f32,
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
    neox: bool,
) -> GpuResult<()> {
    rope_heads_batched(buffer, start_pos, num_heads, head_dim, theta, seq_len, neox)
}

fn gpu_attention_prefill(
    scratch: &mut GpuPrefillScratch,
    seq_len: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let kv_group = config.num_heads / config.num_kv_heads;
    let scale = 1.0f32 / (config.head_dim as f32).sqrt();

    for head in 0..config.num_heads {
        let kv_head = head / kv_group;
        let q_offset = head * config.head_dim;
        let kv_offset = kv_head * config.head_dim;

        flash_attn_prefill_strided(
            scratch.attn_out.as_ptr() as *mut f32,
            scratch.q.as_ptr() as *const f32,
            scratch.k.as_ptr() as *const f32,
            scratch.v.as_ptr() as *const f32,
            seq_len,
            config.head_dim,
            q_size,
            q_size,
            kv_size,
            q_offset,
            q_offset,
            kv_offset,
            scale,
        )?;
    }

    Ok(())
}

fn gpu_embed_tokens_hybrid(
    tokens: &[u32],
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    scratch: &mut GpuPrefillScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    match gpu_weights.token_emb_meta.wtype {
        GgmlType::Q8_0 => {
            validate_token_embedding_layout(&gpu_weights.token_emb_meta, config)?;
            let token_ids: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
            if token_ids
                .iter()
                .any(|&token| token < 0 || token as usize >= config.vocab_size)
            {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: "prefill token id out of vocab range".to_string(),
                });
            }
            upload_i32(&mut scratch.token_ids, &token_ids)?;
            embed_q8_0_batch(
                gpu_weights.token_emb.as_ptr(),
                scratch.token_ids.as_ptr() as *const i32,
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                tokens.len(),
            )
        }
        _ => {
            let mut host_hidden = vec![0.0f32; tokens.len() * h];
            for (row, &token_id) in tokens.iter().enumerate() {
                cpu_embed_token(
                    token_id,
                    cpu_weights,
                    &mut host_hidden[row * h..(row + 1) * h],
                    config,
                );
            }
            upload_f32(&mut scratch.hidden, &host_hidden)
        }
    }
}

/// Hybrid single-layer prefill step over a full prompt batch.
pub fn gpu_prefill_layer_forward_hybrid(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuPrefillScratch,
    layer_idx: usize,
    start_pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let seq_len = scratch.seq_len;
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    gpu_rms_norm_rows(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        seq_len,
        h,
        eps,
    )?;

    gpu_project_rows(
        device,
        &gpu_layer.attn_q,
        &gpu_layer.attn_q_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.q.as_ptr() as *mut f32,
        seq_len,
        q_size,
        h,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.attn_k,
        &gpu_layer.attn_k_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.k.as_ptr() as *mut f32,
        seq_len,
        kv_size,
        h,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.attn_v,
        &gpu_layer.attn_v_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.v.as_ptr() as *mut f32,
        seq_len,
        kv_size,
        h,
    )?;

    if let Some(bq) = &gpu_layer.attn_q_bias {
        add_batched(
            scratch.q.as_ptr() as *const f32,
            bq.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32,
            q_size,
            seq_len,
        )?;
    }
    if let Some(bk) = &gpu_layer.attn_k_bias {
        add_batched(
            scratch.k.as_ptr() as *const f32,
            bk.as_ptr() as *const f32,
            scratch.k.as_ptr() as *mut f32,
            kv_size,
            seq_len,
        )?;
    }
    if let Some(bv) = &gpu_layer.attn_v_bias {
        add_batched(
            scratch.v.as_ptr() as *const f32,
            bv.as_ptr() as *const f32,
            scratch.v.as_ptr() as *mut f32,
            kv_size,
            seq_len,
        )?;
    }

    gpu_rope_rows(
        scratch.q.as_ptr() as *mut f32,
        start_pos,
        seq_len,
        config.num_heads,
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
    )?;
    gpu_rope_rows(
        scratch.k.as_ptr() as *mut f32,
        start_pos,
        seq_len,
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
    )?;

    kv.write_batched(
        layer_idx,
        start_pos,
        seq_len,
        scratch.k.as_ptr() as *const f32,
        scratch.v.as_ptr() as *const f32,
    )?;

    gpu_attention_prefill(scratch, seq_len, config)?;

    gpu_project_rows(
        device,
        &gpu_layer.attn_o,
        &gpu_layer.attn_o_meta,
        scratch.attn_out.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        seq_len,
        h,
        q_size,
    )?;
    add(
        scratch.hidden.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        seq_len * h,
    )?;

    gpu_rms_norm_rows(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        seq_len,
        h,
        eps,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.ffn_gate,
        &gpu_layer.ffn_gate_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        seq_len,
        ff_size,
        h,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.ffn_up,
        &gpu_layer.ffn_up_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *mut f32,
        seq_len,
        ff_size,
        h,
    )?;

    silu(
        scratch.gate.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        seq_len * ff_size,
    )?;
    super::kernels::mul(
        scratch.gate.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *mut f32,
        seq_len * ff_size,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        seq_len,
        h,
        ff_size,
    )?;
    add(
        scratch.hidden.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        seq_len * h,
    )?;

    Ok(())
}

/// Full batched prompt prefill using GPU kernels plus targeted CPU fallbacks.
pub fn gpu_prefill_forward_hybrid(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    prefill: &mut GpuPrefillScratch,
    decode_scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    tokens: &[u32],
    start_pos: usize,
    config: &ModelConfig,
    logits_mode: GpuLogitsMode,
) -> GpuResult<Option<u32>> {
    if tokens.is_empty() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_prefill_forward_hybrid: empty prompt".to_string(),
        });
    }
    if start_pos != 0 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_prefill_forward_hybrid".to_string(),
            dims: vec![],
            reason: "batched GPU prefill currently supports start_pos == 0 only".to_string(),
        });
    }
    if start_pos + tokens.len() > kv.max_seq_len {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_prefill_forward_hybrid: prompt length {} exceeds kv cache {}",
                start_pos + tokens.len(),
                kv.max_seq_len
            ),
        });
    }
    if prefill.seq_len != tokens.len() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_prefill_forward_hybrid: scratch seq_len {} does not match tokens {}",
                prefill.seq_len,
                tokens.len()
            ),
        });
    }

    gpu_embed_tokens_hybrid(tokens, gpu_weights, cpu_weights, prefill, config)?;

    for layer_idx in 0..config.num_layers {
        gpu_prefill_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            prefill,
            layer_idx,
            start_pos,
            config,
        )?;
    }

    if matches!(logits_mode, GpuLogitsMode::Skip) {
        return Ok(None);
    }

    let h = config.hidden_size;
    let v = config.vocab_size;
    let last_hidden =
        unsafe { (prefill.hidden.as_ptr() as *const f32).add((tokens.len() - 1) * h) };
    rms_norm(
        last_hidden,
        gpu_weights.output_norm.as_ptr() as *const f32,
        decode_scratch.normed.as_ptr() as *mut f32,
        h,
        config.rms_norm_eps,
    )?;

    match gpu_dispatch_gemv(
        device,
        &gpu_weights.lm_head,
        &gpu_weights.lm_head_meta,
        decode_scratch.normed.as_ptr() as *const f32,
        decode_scratch.logits.as_ptr() as *mut f32,
        v,
        h,
    ) {
        Ok(()) => match logits_mode {
            GpuLogitsMode::DownloadToHost => {
                download_f32(&decode_scratch.logits, &mut host_scratch.logits[..v])?;
                Ok(None)
            }
            GpuLogitsMode::GreedyArgmax => {
                gpu_greedy_argmax_token(device, decode_scratch, v).map(Some)
            }
            GpuLogitsMode::Skip => Ok(None),
        },
        Err(GpuError::InvalidWeightLayout { .. }) | Err(GpuError::UnsupportedWeightType { .. }) => {
            cpu_fallback_gemv(
                "lm_head",
                &cpu_weights.lm_head,
                &cpu_weights.lm_head_meta,
                &decode_scratch.normed,
                &mut host_scratch.normed,
                &mut host_scratch.logits,
                v,
                h,
                &mut host_scratch.q8_scratch,
            )?;
            match logits_mode {
                GpuLogitsMode::DownloadToHost => Ok(None),
                GpuLogitsMode::GreedyArgmax => Ok(Some(crate::cpu::sampler::cpu_sample_greedy(
                    &host_scratch.logits[..v],
                ))),
                GpuLogitsMode::Skip => Ok(None),
            }
        }
        Err(err) => Err(err),
    }
}

fn gpu_layer_forward_from_state_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        device.stream(),
    )?;

    gpu_dispatch_fused_qkv_on_stream(
        device,
        &gpu_layer.attn_q,
        &gpu_layer.attn_q_meta,
        gpu_layer.attn_q_bias.as_ref(),
        &gpu_layer.attn_k,
        &gpu_layer.attn_k_meta,
        gpu_layer.attn_k_bias.as_ref(),
        &gpu_layer.attn_v,
        &gpu_layer.attn_v_meta,
        gpu_layer.attn_v_bias.as_ref(),
        scratch.normed.as_ptr() as *const f32,
        scratch.q.as_ptr() as *mut f32,
        scratch.k.as_ptr() as *mut f32,
        scratch.v.as_ptr() as *mut f32,
        q_size,
        kv_size,
        h,
        device.stream(),
    )?;

    rope_heads_from_state_on_stream(
        scratch.q.as_ptr() as *mut f32,
        scratch.decode_pos_ptr(),
        config.num_heads,
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
        device.stream(),
    )?;
    kv_write_rope_from_state_on_stream(
        kv.k_ptr(layer_idx)?,
        kv.v_ptr(layer_idx)?,
        scratch.k.as_ptr() as *const f32,
        scratch.v.as_ptr() as *const f32,
        scratch.decode_pos_ptr(),
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
        device.stream(),
    )?;

    gpu_attention_decode_from_state(device, scratch, kv, layer_idx, config)?;

    let attn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &gpu_layer.attn_o,
        &gpu_layer.attn_o_meta,
        scratch.attn_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        h,
        q_size,
        device.stream(),
    )?;
    if !attn_residual_fused {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            q_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    gpu_dispatch_rms_norm(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        h,
        eps,
        device.stream(),
    )?;
    gpu_dispatch_fused_gate_up_on_stream(
        device,
        &gpu_layer.ffn_gate,
        &gpu_layer.ffn_gate_meta,
        &gpu_layer.ffn_up,
        &gpu_layer.ffn_up_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )?;

    let ffn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )?;
    if !ffn_residual_fused {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    Ok(())
}

fn gpu_launch_full_greedy_decode_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    for layer_idx in 0..config.num_layers {
        gpu_layer_forward_from_state_on_stream(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            config,
        )?;
    }

    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)
}

fn gpu_capture_full_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<super::graph::CapturedDecodeGraph> {
    let key = gpu_full_decode_graph_key(device, gpu_weights, kv, config)?;
    scratch.upload_decode_state(0, 1, device.stream())?;
    device.synchronize()?;
    device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
    let capture_result =
        gpu_launch_full_greedy_decode_on_stream(device, gpu_weights, kv, scratch, config);
    let end_capture_result = device.end_capture();

    match capture_result {
        Ok(()) => {
            let graph = HipGraph::from_raw(end_capture_result?);
            super::graph::CapturedDecodeGraph::from_captured_graph(graph, key)
        }
        Err(err) => {
            let _ = end_capture_result;
            Err(err)
        }
    }
}

fn gpu_try_full_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<Option<u32>> {
    if decode_graph_disabled() {
        return Ok(None);
    }

    let key = gpu_full_decode_graph_key(device, gpu_weights, kv, config)?;
    if !scratch.has_decode_graph_for(key) {
        // Try updating existing graph if it matches top-level key
        // (This part is simplified for prototype, true llama.cpp logic
        // would keep the executable graph alive and just patch pointers)

        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => return Ok(None),
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            return Ok(None);
        }

        // Capture a new graph temporarily to see if we can update the executable one
        scratch.upload_decode_state(0, 1, device.stream())?;
        device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
        let capture_res =
            gpu_launch_full_greedy_decode_on_stream(device, gpu_weights, kv, scratch, config);
        let raw_graph = device.end_capture()?;
        capture_res?;

        let new_graph = HipGraph::from_raw(raw_graph);
        if scratch.try_update_decode_graph(&new_graph)? {
            // Update successful! No need to instantiate a new Executable graph.
        } else {
            // Topology changed or no existing graph, instantiate new one
            let captured = super::graph::CapturedDecodeGraph::from_captured_graph(new_graph, key)?;
            scratch.replace_decode_graph(captured);
        }
    }

    scratch.upload_decode_state(pos, pos + 1, device.stream())?;
    if let Some(graph) = scratch.decode_graph() {
        if graph.launch(device.stream()).is_ok() {
            // Wait for completion and read result safely
            gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
            device.synchronize()?;
            let token = scratch.argmax_result_index.as_slice::<i32>()[0];
            return Ok(Some(token as u32));
        }
        scratch.clear_decode_graph();
    }

    Ok(None)
}

/// Hybrid single-layer decode step used by the CLI path and GPU integration tests.
pub fn gpu_layer_forward_hybrid(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    if decode_stage_profiling_enabled() {
        let mut guard = decode_stage_profile_store()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.layer_invocations += 1;
    }

    profile_decode_stage(device, DecodeStage::AttnNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )
    })?;

    profile_decode_stage(device, DecodeStage::Qkv, || {
        gpu_dispatch_fused_qkv_on_stream(
            device,
            &gpu_layer.attn_q,
            &gpu_layer.attn_q_meta,
            gpu_layer.attn_q_bias.as_ref(),
            &gpu_layer.attn_k,
            &gpu_layer.attn_k_meta,
            gpu_layer.attn_k_bias.as_ref(),
            &gpu_layer.attn_v,
            &gpu_layer.attn_v_meta,
            gpu_layer.attn_v_bias.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32,
            scratch.k.as_ptr() as *mut f32,
            scratch.v.as_ptr() as *mut f32,
            q_size,
            kv_size,
            h,
            device.stream(),
        )
    })?;

    profile_decode_stage(device, DecodeStage::QRope, || {
        rope_heads_on_stream(
            scratch.q.as_ptr() as *mut f32,
            pos,
            config.num_heads,
            config.head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )
    })?;

    profile_decode_stage(device, DecodeStage::KvWrite, || {
        kv_write_rope_on_stream(
            kv,
            layer_idx,
            scratch.k.as_ptr() as *mut f32,
            scratch.v.as_ptr() as *mut f32,
            pos,
            config.num_kv_heads,
            config.head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )
    })?;

    profile_decode_stage(device, DecodeStage::Attention, || {
        gpu_attention_decode(device, scratch, kv, layer_idx, pos, config)
    })?;

    let attn_residual_fused = profile_decode_stage(device, DecodeStage::AttnProj, || {
        gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            q_size,
            device.stream(),
        )
    })?;
    if !attn_residual_fused {
        profile_decode_stage(device, DecodeStage::AttnProj, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                scratch.attn_out.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                q_size,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::AttnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    profile_decode_stage(device, DecodeStage::FfnNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::GateUp, || {
        gpu_dispatch_fused_gate_up_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )
    })?;

    let ffn_residual_fused = profile_decode_stage(device, DecodeStage::FfnDown, || {
        gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )
    })?;
    if !ffn_residual_fused {
        profile_decode_stage(device, DecodeStage::FfnDown, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                ff_size,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::FfnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    Ok(())
}

/// Embed one token, preferring a native GPU path and falling back to CPU upload.
pub fn gpu_embed_token_hybrid(
    device: &GpuDevice,
    token_id: u32,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    match gpu_weights.token_emb_meta.wtype {
        GgmlType::Q8_0 => {
            validate_token_embedding_layout(&gpu_weights.token_emb_meta, config)?;
            embed_q8_0_token(
                gpu_weights.token_emb.as_ptr(),
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                token_id,
            )
        }
        _ => {
            // CPU embed into pinned buffer
            cpu_embed_token(
                token_id,
                cpu_weights,
                &mut scratch.input_hidden_pinned.as_slice_mut::<f32>()[..h],
                config,
            );
            // Async upload
            unsafe {
                ffi::hip_memcpy_h2d_async(
                    scratch.hidden.as_ptr(),
                    scratch.input_hidden_pinned.as_ptr(),
                    h * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
            Ok(())
        }
    }
}

/// Full decode forward pass using GPU kernels plus targeted CPU fallbacks.
pub fn gpu_full_forward_hybrid(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
    logits_mode: GpuLogitsMode,
) -> GpuResult<Option<u32>> {
    if matches!(logits_mode, GpuLogitsMode::GreedyArgmax) {
        if let Some(token) =
            gpu_try_full_greedy_decode_graph(device, gpu_weights, kv, scratch, pos, config)?
        {
            return Ok(Some(token));
        }
    }

    for layer_idx in 0..config.num_layers {
        gpu_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            pos,
            config,
        )?;
    }

    if matches!(logits_mode, GpuLogitsMode::Skip) {
        return Ok(None);
    }

    let h = config.hidden_size;
    let v = config.vocab_size;
    let gpu_result = match logits_mode {
        GpuLogitsMode::DownloadToHost => {
            gpu_dispatch_rms_norm(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_weights.output_norm.as_ptr() as *const f32,
                scratch.normed.as_ptr() as *mut f32,
                h,
                config.rms_norm_eps,
                device.stream(),
            )?;
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_weights.lm_head,
                &gpu_weights.lm_head_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.logits.as_ptr() as *mut f32,
                v,
                h,
                device.stream(),
            )?;
            download_f32(&scratch.logits, &mut host_scratch.logits[..v])?;
            Ok(None)
        }
        GpuLogitsMode::GreedyArgmax => {
            gpu_try_greedy_decode_graph(device, gpu_weights, scratch, config).map(Some)
        }
        GpuLogitsMode::Skip => Ok(None),
    };

    match gpu_result {
        Ok(result) => Ok(result),
        Err(GpuError::InvalidWeightLayout { .. }) | Err(GpuError::UnsupportedWeightType { .. }) => {
            gpu_dispatch_rms_norm(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_weights.output_norm.as_ptr() as *const f32,
                scratch.normed.as_ptr() as *mut f32,
                h,
                config.rms_norm_eps,
                device.stream(),
            )?;
            cpu_fallback_gemv(
                "lm_head",
                &cpu_weights.lm_head,
                &cpu_weights.lm_head_meta,
                &scratch.normed,
                &mut host_scratch.normed,
                &mut host_scratch.logits,
                v,
                h,
                &mut host_scratch.q8_scratch,
            )?;
            match logits_mode {
                GpuLogitsMode::DownloadToHost => Ok(None),
                GpuLogitsMode::GreedyArgmax => Ok(Some(crate::cpu::sampler::cpu_sample_greedy(
                    &host_scratch.logits[..v],
                ))),
                GpuLogitsMode::Skip => Ok(None),
            }
        }
        Err(err) => Err(err),
    }
}
