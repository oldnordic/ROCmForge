//! Prefill-only GPU forward path (quarantined 2026-04-19).
//!
//! This module contains hybrid GPU prefill functions that are not part of the
//! current decode hot path. These functions are preserved for potential future
//! prefill optimization work but are not used by the live decode spine.
//!
//! Moved from src/gpu/forward.rs during GPU decode surface cleanup.

use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::forward::cpu_embed_token;
use crate::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use crate::cpu::weights::CpuModelWeights;
use crate::loader::GgmlType;

use super::cache::{GpuKvCache, GpuPrefillScratch};
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::kernels::attention::{flash_attn_prefill_strided, kv_write_batched};
use super::kernels::elementwise::{add, add_batched, embed_q8_0_batch, silu};
use super::kernels::norm::{rms_norm, rms_norm_batched};
use super::kernels::rope::rope_heads_batched;
use super::ops::{gpu_dispatch_gemm, gpu_dispatch_gemv, gpu_dispatch_rms_norm};
use super::weights::{GpuBuffer, GpuLayerWeights, GpuModelWeights, WeightMeta};

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

fn cpu_fallback_error(op: &str, err: impl std::fmt::Display) -> GpuError {
    GpuError::HipApiError {
        code: -1,
        description: format!("{} CPU fallback failed: {}", op, err),
    }
}

fn download_f32(src: &GpuBuffer, dst: &mut [f32]) -> GpuResult<()> {
    src.copy_to_host(unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, std::mem::size_of_val(dst))
    })
}

fn upload_f32(dst: &mut GpuBuffer, src: &[f32]) -> GpuResult<()> {
    dst.copy_from_host(unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src))
    })
}

fn upload_i32(dst: &mut GpuBuffer, src: &[i32]) -> GpuResult<()> {
    dst.copy_from_host(unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src))
    })
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

/// Full batched prompt prefill using GPU kernels plus targeted CPU fallbacks.
pub fn gpu_prefill_forward_hybrid(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    prefill: &mut GpuPrefillScratch,
    decode_scratch: &mut super::cache::GpuForwardScratch,
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
                super::forward::gpu_greedy_argmax_token(device, decode_scratch, v).map(Some)
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
