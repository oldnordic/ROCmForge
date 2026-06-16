#![allow(
    clippy::all,
    unused_parens,
    unused_variables,
    clippy::needless_range_loop,
    unused_imports
)]
//! CPU single-token decode forward pass.
//!
//! Implements the autoregressive decode path: one token through all transformer layers.
//! Uses KV cache for efficient attention computation.

use std::io::Write;

use super::cache::{CpuForwardScratch, CpuKvCache};
use super::graph::{CpuExecutionContext, DirectContext};
use super::ops::{
    dispatch_gemv, flash_attn_decode, gelu_inplace, residual_add, rms_norm, rope_partial, silu_fuse,
};
use super::weights::{CpuLayerWeights, CpuModelWeights, CpuMoeWeights};
use super::CpuError;
use crate::config::ModelConfig;
use crate::loader::GgmlType;
// WeightMeta used internally via re-export; no direct import needed here.

// ── Layer forward ────────────────────────────────────────────────────────────────

/// Apply RMS norm to Q/K/V heads.
///
/// Q and K use the provided per-head weight tensors. V is normalized without a
/// learned scale (used by Gemma4's scaleless value norm).
fn apply_qk_norm(
    q: &mut [f32],
    k: &mut [f32],
    v: Option<&mut [f32]>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    num_heads: usize,
    num_kv_heads: usize,
    q_head_dim: usize,
    kv_head_dim: usize,
    eps: f32,
) {
    if let Some(norm) = q_norm {
        for h in 0..num_heads {
            let start = h * q_head_dim;
            let end = start + q_head_dim;
            let slice = &mut q[start..end];
            let mut inv_rms = 0.0f32;
            for v in slice.iter() {
                inv_rms += v * v;
            }
            inv_rms = (inv_rms / q_head_dim as f32 + eps).sqrt().recip();
            for (i, v) in slice.iter_mut().enumerate() {
                *v = *v * inv_rms * norm[i];
            }
        }
    }
    if let Some(norm) = k_norm {
        for h in 0..num_kv_heads {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;
            let slice = &mut k[start..end];
            let mut inv_rms = 0.0f32;
            for v in slice.iter() {
                inv_rms += v * v;
            }
            inv_rms = (inv_rms / kv_head_dim as f32 + eps).sqrt().recip();
            for (i, v) in slice.iter_mut().enumerate() {
                *v = *v * inv_rms * norm[i];
            }
        }
    }
    if let Some(v) = v {
        for h in 0..num_kv_heads {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;
            let slice = &mut v[start..end];
            let mut inv_rms = 0.0f32;
            for x in slice.iter() {
                inv_rms += x * x;
            }
            inv_rms = (inv_rms / kv_head_dim as f32 + eps).sqrt().recip();
            for x in slice.iter_mut() {
                *x *= inv_rms;
            }
        }
    }
}

/// Single-token shortconv forward (depthwise causal conv1d with double gating).
fn shortconv_forward(
    normed: &[f32],
    weights: &CpuLayerWeights,
    kv: &mut CpuKvCache,
    shortconv_bcx: &mut [f32],
    shortconv_tmp: &mut [f32],
    layer_out: &mut [f32],
    layer: usize,
    config: &ModelConfig,
) -> Result<(), CpuError> {
    let h = config.hidden_size;
    let l_cache = config.shortconv_l_cache.unwrap_or(3);
    let d_conv = l_cache.saturating_sub(1);
    let sc = weights.shortconv.as_ref().ok_or_else(|| {
        CpuError::InvalidOperation(format!("shortconv weights missing for layer {}", layer))
    })?;

    // 1. in_proj: [h] → [3h]
    dispatch_gemv(
        &sc.in_proj,
        &sc.in_proj_meta,
        normed,
        shortconv_bcx,
        3 * h,
        h,
        None,
    )?;

    // 2. Split into B, C, x
    let (b, rest) = shortconv_bcx[..3 * h].split_at(h);
    let (c, x) = rest.split_at(h);

    // 3. Bx = B ⊙ x (elementwise)
    for i in 0..h {
        shortconv_tmp[i] = b[i] * x[i];
    }

    // 4. Read conv state, append Bx, compute causal conv1d
    let conv_state = &mut kv.conv_state[layer];
    if d_conv > 0 {
        let mut concat = vec![0.0f32; l_cache * h];
        for i in 0..d_conv {
            for j in 0..h {
                concat[i * h + j] = conv_state[i * h + j];
            }
        }
        for j in 0..h {
            concat[d_conv * h + j] = shortconv_tmp[j];
        }

        // 5. Depthwise conv1d: conv_out[j] = Σ_k concat[k][j] * kernel[k][j]
        let conv_f32 = crate::cpu::weights::try_as_f32_slice(&sc.conv)
            .ok_or_else(|| CpuError::InvalidOperation("shortconv conv not f32".to_string()))?;
        for j in 0..h {
            let mut acc = 0.0f32;
            for k in 0..l_cache {
                acc += concat[k * h + j] * conv_f32[k * h + j];
            }
            shortconv_tmp[j] = acc;
        }

        // 6. Update conv state: shift left, append Bx
        for i in 0..d_conv.saturating_sub(1) {
            for j in 0..h {
                conv_state[i * h + j] = conv_state[(i + 1) * h + j];
            }
        }
        if d_conv > 0 {
            for j in 0..h {
                conv_state[(d_conv - 1) * h + j] = shortconv_tmp[j];
            }
        }
    }

    // 7. Second gate: y = C ⊙ conv_out
    for i in 0..h {
        shortconv_tmp[i] *= c[i];
    }

    // 8. out_proj: [h] → [h]
    dispatch_gemv(
        &sc.out_proj,
        &sc.out_proj_meta,
        shortconv_tmp,
        layer_out,
        h,
        h,
        None,
    )?;

    Ok(())
}

/// Decode-time MoE FFN: top-k expert selection + weighted sum.
fn moe_forward_decode<C: CpuExecutionContext>(
    ctx: &mut C,
    normed: &[f32],
    moe: &CpuMoeWeights,
    gate: &mut [f32],
    swiglu: &mut [f32],
    tmp: &mut [f32],
    layer_out: &mut [f32],
    config: &ModelConfig,
    mut q8_scratch: Option<&mut [u8]>,
) -> Result<(), CpuError> {
    let h = config.hidden_size;
    let num_experts = moe.num_experts;
    let ff_size = moe.ff_size;
    let top_k = config.num_experts_per_tok.unwrap_or(4).min(num_experts);

    // 1. Router: gate_inp * normed → logits
    let mut router_meta = moe.gate_inp_meta.clone();
    if router_meta.dims.len() == 2 {
        router_meta.dims.truncate(1);
    }
    ctx.execute_gemv(
        &moe.gate_inp,
        &router_meta,
        normed,
        gate,
        num_experts,
        h,
        q8_scratch.as_deref_mut(),
    )?;

    // 2. Optional bias
    if let Some(ref bias) = moe.exp_probs_b_bias {
        for i in 0..num_experts {
            gate[i] += bias[i];
        }
    }

    // 3. Softmax + top-k
    let mut max_logit = f32::NEG_INFINITY;
    for i in 0..num_experts {
        if gate[i] > max_logit {
            max_logit = gate[i];
        }
    }
    let mut sum_exp = 0.0f32;
    for i in 0..num_experts {
        gate[i] = (gate[i] - max_logit).exp();
        sum_exp += gate[i];
    }
    if sum_exp > 0.0 {
        for i in 0..num_experts {
            gate[i] /= sum_exp;
        }
    }

    let mut indexed: Vec<(usize, f32)> = gate[..num_experts].iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(top_k);

    // 4. Run selected experts and accumulate
    layer_out.fill(0.0);

    let gate_stride = moe.gate_exps_meta.wtype.bytes_for_elements(h * ff_size);
    let up_stride = moe.up_exps_meta.wtype.bytes_for_elements(h * ff_size);
    let down_stride = moe.down_exps_meta.wtype.bytes_for_elements(ff_size * h);

    for (expert_idx, weight) in indexed {
        let mut gate_meta_2d = moe.gate_exps_meta.clone();
        let mut up_meta_2d = moe.up_exps_meta.clone();
        let mut down_meta_2d = moe.down_exps_meta.clone();
        if gate_meta_2d.dims.len() == 3 {
            gate_meta_2d.dims.truncate(2);
        }
        if up_meta_2d.dims.len() == 3 {
            up_meta_2d.dims.truncate(2);
        }
        if down_meta_2d.dims.len() == 3 {
            down_meta_2d.dims.truncate(2);
        }

        let gate_off = expert_idx * gate_stride;
        let up_off = expert_idx * up_stride;
        let down_off = expert_idx * down_stride;

        let gate_slice = &moe.gate_exps[gate_off..gate_off + gate_stride];
        let up_slice = &moe.up_exps[up_off..up_off + up_stride];
        let down_slice = &moe.down_exps[down_off..down_off + down_stride];

        ctx.execute_gemv(
            gate_slice,
            &gate_meta_2d,
            normed,
            gate,
            ff_size,
            h,
            q8_scratch.as_deref_mut(),
        )?;
        ctx.execute_gemv(
            up_slice,
            &up_meta_2d,
            normed,
            swiglu,
            ff_size,
            h,
            q8_scratch.as_deref_mut(),
        )?;
        ctx.execute_silu(gate, swiglu);
        ctx.execute_gemv(
            down_slice,
            &down_meta_2d,
            swiglu,
            tmp,
            h,
            ff_size,
            q8_scratch.as_deref_mut(),
        )?;

        for i in 0..h {
            layer_out[i] += weight * tmp[i];
        }
    }

    let scale = config.expert_weights_scale;
    if scale != 1.0 {
        for v in layer_out[..h].iter_mut() {
            *v *= scale;
        }
    }

    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "function has many parameters by design"
)]
/// Forward pass through a single transformer layer.
///
/// Architecture: RMSNorm → Attention/Shortconv → Residual → RMSNorm → FFN → Residual
pub fn cpu_layer_forward(
    hidden: &mut [f32],
    weights: &CpuLayerWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    layer: usize,
    pos: usize,
    rope_sin: &[f32],
    rope_cos: &[f32],
    config: &ModelConfig,
    debug: bool,
) -> Result<(), CpuError> {
    let mut ctx = DirectContext;
    cpu_layer_forward_with_ctx(
        &mut ctx, hidden, weights, kv, scratch, layer, pos, rope_sin, rope_cos, config, debug,
    )
}

#[allow(clippy::too_many_arguments)]
/// Internal forward pass using abstract context.
pub fn cpu_layer_forward_with_ctx<C: CpuExecutionContext>(
    ctx: &mut C,
    hidden: &mut [f32],
    weights: &CpuLayerWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    layer: usize,
    pos: usize,
    rope_sin: &[f32],
    rope_cos: &[f32],
    config: &ModelConfig,
    debug: bool,
) -> Result<(), CpuError> {
    // Legacy qwen35 SSM still rejected; everything else now supported.
    if config.architecture == "qwen35" || weights.ssm.is_some() {
        return Err(CpuError::InvalidOperation(
            "qwen35 hybrid SSM CPU forward is not implemented yet".to_string(),
        ));
    }

    let is_gemma4 = config.architecture == "gemma4";
    let h = config.hidden_size;
    let q_size = config.q_size(layer);
    let kv_size = config.kv_size(layer);
    let head_dim = config.head_dim_for_layer(layer);
    let kv_head_dim = config.kv_head_dim_for_layer(layer);
    let ff_size = config.intermediate_size_for_layer(layer);
    let eps = config.rms_norm_eps;
    let sliding_window = config.sliding_window_for_layer(layer);

    // 1. Attention RMS norm
    ctx.execute_rms_norm(hidden, &weights.attn_norm, &mut scratch.normed, eps);

    if debug && layer == 0 {
        let norm_mean: f32 = scratch.normed.iter().copied().sum::<f32>() / h as f32;
        let norm_std: f32 = ((scratch.normed.iter().map(|x| x * x).sum::<f32>() / h as f32)
            - norm_mean * norm_mean)
            .sqrt();
        eprintln!(
            "[Layer {} after norm] mean={:.4} std={:.4}",
            layer, norm_mean, norm_std
        );
    }

    // 2. Attention or Shortconv
    if weights.is_attention_layer {
        let first_shared = config.first_kv_shared_layer_idx();
        let is_kv_shared = is_gemma4 && layer >= first_shared;
        let stores_shared_kv = is_gemma4 && layer < first_shared && config.stores_shared_kv(layer);

        // Query projection
        if weights.attn_qkv.is_none() {
            ctx.execute_gemv(
                &weights.attn_q,
                &weights.attn_q_meta,
                &scratch.normed,
                &mut scratch.q[..q_size],
                q_size,
                h,
                Some(&mut scratch.q8_scratch),
            )?;
            if let Some(bq) = &weights.attn_q_bias {
                super::ops::add_bias(&mut scratch.q[..q_size], bq);
            }
        }

        // Key/value projection (skipped for Gemma4 shared-KV layers)
        if !is_kv_shared {
            if let (Some(ref qkv_w), Some(ref qkv_m)) = (&weights.attn_qkv, &weights.attn_qkv_meta)
            {
                // Fused QKV path (non-Gemma4)
                let qkv_total = q_size + 2 * kv_size;
                ctx.execute_gemv(
                    qkv_w,
                    qkv_m,
                    &scratch.normed,
                    &mut scratch.qkv[..qkv_total],
                    qkv_total,
                    h,
                    Some(&mut scratch.q8_scratch),
                )?;
                scratch.q[..q_size].copy_from_slice(&scratch.qkv[..q_size]);
                scratch.k[..kv_size].copy_from_slice(&scratch.qkv[q_size..q_size + kv_size]);
                scratch.v[..kv_size].copy_from_slice(&scratch.qkv[q_size + kv_size..qkv_total]);
            } else {
                let normed = &*scratch.normed;
                ctx.execute_gemv(
                    &weights.attn_k,
                    &weights.attn_k_meta,
                    normed,
                    &mut scratch.k[..kv_size],
                    kv_size,
                    h,
                    Some(&mut scratch.q8_scratch),
                )?;
                ctx.execute_gemv(
                    &weights.attn_v,
                    &weights.attn_v_meta,
                    normed,
                    &mut scratch.v[..kv_size],
                    kv_size,
                    h,
                    Some(&mut scratch.q8_scratch),
                )?;
            }
            if let Some(bk) = &weights.attn_k_bias {
                super::ops::add_bias(&mut scratch.k[..kv_size], bk);
            }
            if let Some(bv) = &weights.attn_v_bias {
                super::ops::add_bias(&mut scratch.v[..kv_size], bv);
            }
        }

        // QK-Norm and RoPE
        if is_kv_shared {
            // Shared layers only normalize and rotate the query; K/V are reused.
            if let Some(q_norm) = weights.attn_q_norm.as_deref() {
                apply_qk_norm(
                    &mut scratch.q[..q_size],
                    &mut [],
                    None,
                    Some(q_norm),
                    None,
                    config.num_heads,
                    config.num_kv_heads,
                    head_dim,
                    kv_head_dim,
                    eps,
                );
            }
            if is_gemma4 {
                let partial_factor = config.rope_partial_factor_for_layer(layer);
                let rotated_dims = (head_dim as f32 * partial_factor) as usize;
                let rotated_half = rotated_dims / 2;
                let freq = config.rope_freq_for_layer(layer);
                for i in 0..rotated_half {
                    let angle = pos as f32 * freq[i];
                    let (s, c) = angle.sin_cos();
                    scratch.rope_sin[i] = s;
                    scratch.rope_cos[i] = c;
                }
                rope_partial(
                    &mut scratch.q[..q_size],
                    config.num_heads,
                    head_dim,
                    rotated_dims,
                    &scratch.rope_sin[..rotated_half],
                    &scratch.rope_cos[..rotated_half],
                    config.rope_neox,
                );
            } else {
                ctx.execute_rope(
                    &mut scratch.q[..q_size],
                    config.num_heads,
                    head_dim,
                    rope_sin,
                    rope_cos,
                    config.rope_neox,
                );
            }
        } else {
            // Non-shared layers: normalize and rotate both Q and K.
            if weights.attn_q_norm.is_some() || weights.attn_k_norm.is_some() {
                apply_qk_norm(
                    &mut scratch.q[..q_size],
                    &mut scratch.k[..kv_size],
                    Some(&mut scratch.v[..kv_size]),
                    weights.attn_q_norm.as_deref(),
                    weights.attn_k_norm.as_deref(),
                    config.num_heads,
                    config.num_kv_heads,
                    head_dim,
                    kv_head_dim,
                    eps,
                );
            }

            if is_gemma4 {
                let partial_factor = config.rope_partial_factor_for_layer(layer);
                let rotated_dims = (head_dim as f32 * partial_factor) as usize;
                let kv_rotated_dims = (kv_head_dim as f32 * partial_factor) as usize;
                let rotated_half = rotated_dims / 2;
                let kv_rotated_half = kv_rotated_dims / 2;
                let freq = config.rope_freq_for_layer(layer);
                for i in 0..rotated_half.max(kv_rotated_half) {
                    let angle = pos as f32 * freq[i];
                    let (s, c) = angle.sin_cos();
                    scratch.rope_sin[i] = s;
                    scratch.rope_cos[i] = c;
                }
                let rope_sin = &scratch.rope_sin[..rotated_half];
                let rope_cos = &scratch.rope_cos[..rotated_half];
                let kv_rope_sin = &scratch.rope_sin[..kv_rotated_half];
                let kv_rope_cos = &scratch.rope_cos[..kv_rotated_half];
                rope_partial(
                    &mut scratch.q[..q_size],
                    config.num_heads,
                    head_dim,
                    rotated_dims,
                    rope_sin,
                    rope_cos,
                    config.rope_neox,
                );
                rope_partial(
                    &mut scratch.k[..kv_size],
                    config.num_kv_heads,
                    kv_head_dim,
                    kv_rotated_dims,
                    kv_rope_sin,
                    kv_rope_cos,
                    config.rope_neox,
                );
            } else {
                ctx.execute_rope(
                    &mut scratch.q[..q_size],
                    config.num_heads,
                    head_dim,
                    rope_sin,
                    rope_cos,
                    config.rope_neox,
                );
                ctx.execute_rope(
                    &mut scratch.k[..kv_size],
                    config.num_kv_heads,
                    kv_head_dim,
                    rope_sin,
                    rope_cos,
                    config.rope_neox,
                );
            }

            // Write K, V cache
            kv.write_k(layer, pos, &scratch.k[..kv_size]);
            kv.write_v(layer, pos, &scratch.v[..kv_size]);
            if stores_shared_kv {
                let ty = config.layer_type_for_layer(layer);
                kv.write_shared_k(ty, pos, &scratch.k[..kv_size]);
                kv.write_shared_v(ty, pos, &scratch.v[..kv_size]);
            }
        }

        // Flash attention
        let seq_len = pos + 1;
        if is_gemma4 {
            let logit_cap = config.attention_logit_cap.unwrap_or(0.0);
            let (k_cache, v_cache) = if is_kv_shared {
                let ty = config.layer_type_for_layer(layer);
                kv.shared_kv(ty).ok_or_else(|| {
                    CpuError::InvalidOperation(format!(
                        "missing shared KV state for layer type {}",
                        ty
                    ))
                })?
            } else {
                (kv.k_buf(layer), kv.v_buf(layer))
            };
            flash_attn_decode(
                &scratch.q[..q_size],
                k_cache,
                v_cache,
                &mut scratch.attn_out[..q_size],
                seq_len,
                config.num_heads,
                config.num_kv_heads,
                head_dim,
                sliding_window,
                logit_cap,
                config.attention_scale,
            );
        } else {
            ctx.execute_attention(
                &scratch.q[..q_size],
                kv.k_buf(layer),
                kv.v_buf(layer),
                &mut scratch.attn_out[..q_size],
                seq_len,
                config.num_heads,
                config.num_kv_heads,
                head_dim,
                kv.max_seq_len,
            );
        }

        // Output projection
        ctx.execute_gemv(
            &weights.attn_o,
            &weights.attn_o_meta,
            &scratch.attn_out[..q_size],
            &mut scratch.layer_out[..h],
            h,
            q_size,
            Some(&mut scratch.q8_scratch),
        )?;
    } else {
        // Shortconv path
        shortconv_forward(
            &scratch.normed,
            weights,
            kv,
            &mut scratch.shortconv_bcx,
            &mut scratch.shortconv_tmp,
            &mut scratch.layer_out[..h],
            layer,
            config,
        )?;
    }

    // 3. Optional post-attention norm (Gemma4) and residual
    if let Some(ref norm) = weights.post_attention_norm {
        scratch.shortconv_tmp[..h].copy_from_slice(&scratch.layer_out[..h]);
        rms_norm(
            &scratch.shortconv_tmp[..h],
            norm,
            &mut scratch.layer_out[..h],
            eps,
        );
    }
    ctx.execute_residual_add(hidden, &scratch.layer_out[..h]);

    // 4. FFN RMS norm
    ctx.execute_rms_norm(hidden, &weights.ffn_norm, &mut scratch.normed, eps);

    // 5. FFN (dense or MoE)
    if let Some(ref moe) = weights.moe {
        moe_forward_decode(
            ctx,
            &scratch.normed,
            moe,
            &mut scratch.gate,
            &mut scratch.swiglu,
            &mut scratch.shortconv_tmp,
            &mut scratch.layer_out[..h],
            config,
            Some(&mut scratch.q8_scratch),
        )?;
    } else {
        let normed = &*scratch.normed;
        let swiglu = &mut scratch.swiglu[..ff_size];
        if let (Some(ref gate_w), Some(ref gate_m)) = (&weights.ffn_gate, &weights.ffn_gate_meta) {
            let gate = &mut scratch.gate[..ff_size];
            ctx.execute_gemv(
                gate_w,
                gate_m,
                normed,
                gate,
                ff_size,
                h,
                Some(&mut scratch.q8_scratch),
            )?;
            ctx.execute_gemv(
                &weights.ffn_up,
                &weights.ffn_up_meta,
                normed,
                swiglu,
                ff_size,
                h,
                Some(&mut scratch.q8_scratch),
            )?;
            if config.use_gelu_swiglu {
                gelu_inplace(gate);
                for i in 0..ff_size {
                    swiglu[i] *= gate[i];
                }
            } else {
                ctx.execute_silu(gate, swiglu);
            }
        } else {
            ctx.execute_gemv(
                &weights.ffn_up,
                &weights.ffn_up_meta,
                normed,
                swiglu,
                ff_size,
                h,
                Some(&mut scratch.q8_scratch),
            )?;
            gelu_inplace(swiglu);
        }
        ctx.execute_gemv(
            &weights.ffn_down,
            &weights.ffn_down_meta,
            &scratch.swiglu[..ff_size],
            &mut scratch.layer_out[..h],
            h,
            ff_size,
            Some(&mut scratch.q8_scratch),
        )?;
    }

    // 6. Optional post-ffw norm (Gemma4) and residual
    if let Some(ref norm) = weights.post_ffw_norm {
        scratch.shortconv_tmp[..h].copy_from_slice(&scratch.layer_out[..h]);
        rms_norm(
            &scratch.shortconv_tmp[..h],
            norm,
            &mut scratch.layer_out[..h],
            eps,
        );
    }
    ctx.execute_residual_add(hidden, &scratch.layer_out[..h]);

    // 7. Per-Layer Embedding (PLE) branch for Gemma4
    if is_gemma4 && config.hidden_size_per_layer_input > 0 {
        if let (Some((ref gate_w, ref gate_m)), Some((ref proj_w, ref proj_m))) =
            (&weights.inp_gate, &weights.proj)
        {
            let ple_dim = config.hidden_size_per_layer_input;
            let ple_offset = layer * ple_dim;
            let ple_slice = &scratch.ple_input[ple_offset..ple_offset + ple_dim];

            // inp_gate: hidden -> ple_dim, then GELU
            let gate = &mut scratch.gate[..ple_dim];
            ctx.execute_gemv(
                gate_w,
                gate_m,
                hidden,
                gate,
                ple_dim,
                h,
                Some(&mut scratch.q8_scratch),
            )?;
            gelu_inplace(gate);
            for i in 0..ple_dim {
                gate[i] *= ple_slice[i];
            }

            // proj: ple_dim -> hidden
            ctx.execute_gemv(
                proj_w,
                proj_m,
                gate,
                &mut scratch.layer_out[..h],
                h,
                ple_dim,
                Some(&mut scratch.q8_scratch),
            )?;

            if let Some(ref norm) = weights.post_norm {
                scratch.shortconv_tmp[..h].copy_from_slice(&scratch.layer_out[..h]);
                rms_norm(
                    &scratch.shortconv_tmp[..h],
                    norm,
                    &mut scratch.layer_out[..h],
                    eps,
                );
            }

            for i in 0..h {
                hidden[i] += scratch.layer_out[i];
            }
        }
    }

    // 8. Optional layer output scale (Gemma4)
    if let Some(scale) = weights.layer_output_scale {
        for v in hidden.iter_mut() {
            *v *= scale;
        }
    }

    Ok(())
}

// ── Full forward pass ────────────────────────────────────────────────────────────

/// Complete forward pass through all transformer layers.
///
/// After this function, `scratch.logits` contains the output logits.
pub fn cpu_full_forward(
    hidden: &mut [f32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
) -> Result<(), CpuError> {
    // Debug: input hidden statistics
    let debug = std::env::var("ROCMFORGE_DEBUG").is_ok();
    if debug {
        let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
        let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32)
            - mean * mean)
            .sqrt();
        eprintln!(
            "[Forward input] pos={} mean={:.4} std={:.4}",
            pos, mean, std
        );
    }

    // Optional hidden-state dump for layer-by-layer debugging.
    if let Ok(spec) = std::env::var("ROCMFORGE_DUMP_HIDDEN") {
        if let Some((p, path)) = spec.split_once(':') {
            if p.parse::<usize>().ok() == Some(pos) {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            hidden.as_ptr() as *const u8,
                            hidden.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    let _ = f.write_all(bytes);
                }
            }
        }
    }

    // Precompute RoPE sin/cos for this position once, reuse across all layers.
    // For models with per-layer head dimensions (e.g. Gemma4), size the table for
    // the largest rotated dimension seen across layers, not the global head_dim.
    let max_rotated_half = (0..config.num_layers)
        .map(|layer| {
            let head_dim = config.head_dim_for_layer(layer);
            let factor = config.rope_partial_factor_for_layer(layer);
            ((head_dim as f32 * factor) as usize / 2).max(1)
        })
        .max()
        .unwrap_or(config.head_dim / 2);
    let rope_freq_len = config.rope_freq.len();
    // Zero any trailing entries that may be unused on this model.
    scratch.rope_sin[..max_rotated_half].fill(0.0);
    scratch.rope_cos[..max_rotated_half].fill(0.0);
    for i in 0..max_rotated_half.min(rope_freq_len) {
        let angle = pos as f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        scratch.rope_sin[i] = s;
        scratch.rope_cos[i] = c;
    }

    // SAFETY: rope_sin/rope_cos are only read in cpu_layer_forward and are never
    // mutated by it. We use raw-pointer slices to avoid a borrow-checker conflict
    // between the immutable borrows into scratch.rope_sin/rope_cos and the mutable
    // borrow of scratch passed to cpu_layer_forward.
    let rope_sin =
        unsafe { std::slice::from_raw_parts(scratch.rope_sin.as_ptr(), max_rotated_half) };
    let rope_cos =
        unsafe { std::slice::from_raw_parts(scratch.rope_cos.as_ptr(), max_rotated_half) };

    // Process all transformer layers
    for layer_idx in 0..config.num_layers {
        cpu_layer_forward(
            hidden,
            weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            pos,
            rope_sin,
            rope_cos,
            config,
            debug,
        )?;

        // Debug: show hidden state after each layer
        if debug {
            let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
            let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32)
                - mean * mean)
                .sqrt();
            eprintln!(
                "[After layer {}] mean={:.4} std={:.4}",
                layer_idx, mean, std
            );
        }

        // Optional hidden-state dump for layer-by-layer debugging.
        if let Ok(spec) = std::env::var("ROCMFORGE_DUMP_HIDDEN") {
            if let Some((p, path)) = spec.split_once(':') {
                if p.parse::<usize>().ok() == Some(pos) {
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                hidden.as_ptr() as *const u8,
                                hidden.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        let _ = f.write_all(bytes);
                    }
                }
            }
        }
    }

    // Debug: show hidden state before final norm
    if debug {
        let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
        let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32)
            - mean * mean)
            .sqrt();
        let min: f32 = hidden.iter().copied().fold(f32::INFINITY, f32::min);
        let max: f32 = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[Before final norm] mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
            mean, std, min, max
        );
    }

    // Final RMS norm
    rms_norm(
        hidden,
        &weights.output_norm,
        &mut scratch.normed,
        config.rms_norm_eps,
    );

    // Debug: show normed output
    if debug {
        let mean: f32 = scratch.normed.iter().copied().sum::<f32>() / scratch.normed.len() as f32;
        let std: f32 = ((scratch.normed.iter().map(|x| x * x).sum::<f32>()
            / scratch.normed.len() as f32)
            - mean * mean)
            .sqrt();
        eprintln!("[After final norm] mean={:.4} std={:.4}", mean, std);
    }

    // LM head: project to vocabulary
    let h = config.hidden_size;
    let v = config.vocab_size;
    if debug {
        eprintln!(
            "[LM head] type={:?} h={} v={} tied={} transpose={}",
            weights.lm_head_meta.wtype,
            h,
            v,
            weights.lm_head_tied,
            weights.lm_head_meta.needs_transpose
        );
    }
    // Use metadata to automatically select correct kernel (regular or transposed)
    super::ops::dispatch_gemv(
        &weights.lm_head,
        &weights.lm_head_meta,
        &scratch.normed,
        &mut scratch.logits,
        v, // vocab_size (out_dim)
        h, // hidden_size (in_dim)
        Some(&mut scratch.q8_scratch),
    )?;

    // Final logit softcapping (Gemma4)
    if let Some(cap) = config.final_logit_softcapping {
        let inv_cap = 1.0 / cap;
        for logit in scratch.logits.iter_mut() {
            *logit = cap * (*logit * inv_cap).tanh();
        }
    }

    // Debug: show logits statistics
    if debug {
        let mean: f32 = scratch.logits.iter().copied().sum::<f32>() / scratch.logits.len() as f32;
        let std: f32 = ((scratch.logits.iter().map(|x| x * x).sum::<f32>()
            / scratch.logits.len() as f32)
            - mean * mean)
            .sqrt();
        let min: f32 = scratch.logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max: f32 = scratch
            .logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[After LM head] mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
            mean, std, min, max
        );
    }

    Ok(())
}

/// Full transformer decode forward pass with a pluggable execution context.
///
/// This is identical to `cpu_full_forward` but records every captured op
/// through `ctx`. It is the hook used to turn a real inference session into a
/// `GraphMap` when `ctx` is a `CaptureContext`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_full_forward_with_ctx<C: CpuExecutionContext>(
    ctx: &mut C,
    hidden: &mut [f32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
) -> Result<(), CpuError> {
    let debug = std::env::var("ROCMFORGE_DEBUG").is_ok();
    if debug {
        let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
        let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32)
            - mean * mean)
            .sqrt();
        eprintln!(
            "[Forward input] pos={} mean={:.4} std={:.4}",
            pos, mean, std
        );
    }

    let max_rotated_half = (0..config.num_layers)
        .map(|layer| {
            let head_dim = config.head_dim_for_layer(layer);
            let factor = config.rope_partial_factor_for_layer(layer);
            ((head_dim as f32 * factor) as usize / 2).max(1)
        })
        .max()
        .unwrap_or(config.head_dim / 2);
    let rope_freq_len = config.rope_freq.len();
    scratch.rope_sin[..max_rotated_half].fill(0.0);
    scratch.rope_cos[..max_rotated_half].fill(0.0);
    for i in 0..max_rotated_half.min(rope_freq_len) {
        let angle = pos as f32 * config.rope_freq[i];
        let (s, c) = angle.sin_cos();
        scratch.rope_sin[i] = s;
        scratch.rope_cos[i] = c;
    }

    let rope_sin =
        unsafe { std::slice::from_raw_parts(scratch.rope_sin.as_ptr(), max_rotated_half) };
    let rope_cos =
        unsafe { std::slice::from_raw_parts(scratch.rope_cos.as_ptr(), max_rotated_half) };

    for layer_idx in 0..config.num_layers {
        cpu_layer_forward_with_ctx(
            ctx,
            hidden,
            weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            pos,
            rope_sin,
            rope_cos,
            config,
            debug,
        )?;

        if debug && layer_idx < 2 {
            let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
            let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32)
                - mean * mean)
                .sqrt();
            eprintln!(
                "[After layer {}] mean={:.4} std={:.4}",
                layer_idx, mean, std
            );
        }
    }

    ctx.execute_rms_norm(
        hidden,
        &weights.output_norm,
        &mut scratch.normed,
        config.rms_norm_eps,
    );

    if debug {
        let mean: f32 = scratch.normed.iter().copied().sum::<f32>() / scratch.normed.len() as f32;
        let std: f32 = ((scratch.normed.iter().map(|x| x * x).sum::<f32>()
            / scratch.normed.len() as f32)
            - mean * mean)
            .sqrt();
        eprintln!("[After final norm] mean={:.4} std={:.4}", mean, std);
    }

    let h = config.hidden_size;
    let v = config.vocab_size;
    if debug {
        eprintln!(
            "[LM head] type={:?} h={} v={} tied={} transpose={}",
            weights.lm_head_meta.wtype,
            h,
            v,
            weights.lm_head_tied,
            weights.lm_head_meta.needs_transpose
        );
    }
    ctx.execute_gemv(
        &weights.lm_head,
        &weights.lm_head_meta,
        &scratch.normed,
        &mut scratch.logits,
        v,
        h,
        Some(&mut scratch.q8_scratch),
    )?;

    // Final logit softcapping (Gemma4)
    if let Some(cap) = config.final_logit_softcapping {
        let inv_cap = 1.0 / cap;
        for logit in scratch.logits.iter_mut() {
            *logit = cap * (*logit * inv_cap).tanh();
        }
    }

    if debug {
        let mean: f32 = scratch.logits.iter().copied().sum::<f32>() / scratch.logits.len() as f32;
        let std: f32 = ((scratch.logits.iter().map(|x| x * x).sum::<f32>()
            / scratch.logits.len() as f32)
            - mean * mean)
            .sqrt();
        let min: f32 = scratch.logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max: f32 = scratch
            .logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[After LM head] mean={:.4} std={:.4} range=[{:.4}, {:.4}]",
            mean, std, min, max
        );
    }

    Ok(())
}

// ── Prefill wrappers ────────────────────────────────────────────────────────────────

/// Convenience wrapper for prompt prefill that populates `scratch.logits` for sampling.
///
/// The `hidden` buffer is provided for API compatibility but is immediately overwritten
/// by `cpu_embed_token` on the first decode step, so its contents after this call are unused.
///
/// Automatically selects parallel or sequential processing based on prompt length
/// (parallel is used when the prompt spans ≥2 batches).
pub fn cpu_prefill(
    _hidden: &mut [f32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    prompt_tokens: &[u32],
    config: &ModelConfig,
) -> Result<(), CpuError> {
    let batch_config = match crate::hardware::detect() {
        Ok(caps) => crate::hardware::derive_batch_config(&caps, config),
        Err(_) => crate::hardware::BatchConfig {
            max_tokens_per_batch: prompt_tokens.len().clamp(1, 256),
            num_cores: rayon::current_num_threads(),
        },
    };
    super::prefill::cpu_prefill_forward_parallel(
        prompt_tokens,
        weights,
        kv,
        scratch,
        0,
        config,
        &batch_config,
    )
}

// ── Per-Layer Embeddings (PLE) ───────────────────────────────────────────────────

/// Compute Gemma4 Per-Layer Embedding (PLE) inputs for the current token.
///
/// This implements `Gemma4TextModel.project_per_layer_inputs`:
///   per_layer_token_embd lookup (scaled by sqrt(ple_dim))
///   + main_embedding projected through per_layer_model_proj (scaled by 1/sqrt(hidden_size))
///   RMSNorm per layer with per_layer_proj_norm
///   * 1/sqrt(2)
///
/// Result is written to `scratch.ple_input` as `[num_layers * ple_dim]`.
pub fn cpu_compute_ple_inputs(
    token_id: u32,
    hidden: &[f32],
    weights: &CpuModelWeights,
    scratch: &mut CpuForwardScratch,
    config: &ModelConfig,
) {
    if config.architecture != "gemma4" || config.hidden_size_per_layer_input == 0 {
        return;
    }

    let ple_dim = config.hidden_size_per_layer_input;
    let ple_total = config.num_layers * ple_dim;
    let h = config.hidden_size;
    let eps = config.rms_norm_eps;

    let (ple_emb_data, ple_emb_meta) = match &weights.per_layer_token_emb {
        Some(x) => x,
        None => return,
    };
    let (ple_proj_data, ple_proj_meta) = match &weights.per_layer_model_proj {
        Some(x) => x,
        None => return,
    };
    let ple_norm = match &weights.per_layer_proj_norm {
        Some(x) => x,
        None => return,
    };

    // 1. Lookup per-layer token embedding into ple_input and scale by sqrt(ple_dim).
    match ple_emb_meta.wtype {
        GgmlType::F32 => {
            if let Some(emb) = super::weights::try_as_f32_slice(ple_emb_data) {
                super::quant::embed_f32(
                    token_id as usize,
                    emb,
                    &mut scratch.ple_input[..ple_total],
                );
            } else {
                let start = token_id as usize * ple_total * 4;
                let bytes = &ple_emb_data[start..start + ple_total * 4];
                for i in 0..ple_total {
                    scratch.ple_input[i] = f32::from_le_bytes([
                        bytes[i * 4],
                        bytes[i * 4 + 1],
                        bytes[i * 4 + 2],
                        bytes[i * 4 + 3],
                    ]);
                }
            }
        }
        GgmlType::F16 => {
            let start_idx = token_id as usize * ple_total;
            let emb = &ple_emb_data[start_idx * 2..(start_idx + ple_total) * 2];
            for i in 0..ple_total {
                let bits = u16::from_le_bytes([emb[i * 2], emb[i * 2 + 1]]);
                scratch.ple_input[i] = half::f16::from_bits(bits).to_f32();
            }
        }
        GgmlType::BF16 => {
            let start_idx = token_id as usize * ple_total;
            let emb = &ple_emb_data[start_idx * 2..(start_idx + ple_total) * 2];
            for i in 0..ple_total {
                let bits = u16::from_le_bytes([emb[i * 2], emb[i * 2 + 1]]);
                scratch.ple_input[i] = half::bf16::from_bits(bits).to_f32();
            }
        }
        GgmlType::Q4_0 => {
            super::quant::embed_q4_0(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q4_1 => {
            super::quant::embed_q4_1(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q4_K => {
            super::quant::embed_q4_k(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q5_0 => {
            super::quant::embed_q5_0(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q5_K => {
            super::quant::embed_q5_k(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q6_K => {
            super::quant::embed_q6_k(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q8_0 => {
            super::quant::embed_q8_0(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q3_K => {
            super::quant::embed_q3_k(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        GgmlType::Q2_K => {
            super::quant::embed_q2_k(
                token_id as usize,
                ple_emb_data,
                &mut scratch.ple_input[..ple_total],
                ple_total,
            );
        }
        _ => {
            panic!(
                "Unsupported per-layer token embedding type: {:?}",
                ple_emb_meta.wtype
            );
        }
    }

    let ple_scale = (ple_dim as f32).sqrt();
    for v in scratch.ple_input[..ple_total].iter_mut() {
        *v *= ple_scale;
    }

    // 2. Project main hidden state through per_layer_model_proj -> ple_proj and
    //    apply Gemma4's per-layer projection scale (1/sqrt(hidden_size)).
    dispatch_gemv(
        ple_proj_data,
        ple_proj_meta,
        hidden,
        &mut scratch.ple_proj[..ple_total],
        ple_total,
        h,
        Some(&mut scratch.q8_scratch),
    )
    .expect("per_layer_model_proj GEMV failed");

    let hidden_scale = 1.0 / (h as f32).sqrt();
    for v in scratch.ple_proj[..ple_total].iter_mut() {
        *v *= hidden_scale;
    }

    // 3. RMSNorm the projection per layer, add per-layer token embeddings, then
    //    scale the combined result by 1/sqrt(2) to match Gemma4's
    //    project_per_layer_inputs.
    let inv_sqrt2 = 1.0 / (2.0f32).sqrt();
    let mut norm_tmp = vec![0.0f32; ple_dim];
    for layer in 0..config.num_layers {
        let off = layer * ple_dim;
        rms_norm(
            &scratch.ple_proj[off..off + ple_dim],
            ple_norm,
            &mut norm_tmp,
            eps,
        );
        for i in 0..ple_dim {
            let combined = norm_tmp[i] + scratch.ple_input[off + i];
            scratch.ple_input[off + i] = combined * inv_sqrt2;
        }
    }
}

// ── Token embedding ──────────────────────────────────────────────────────────────

/// Embed a single token into hidden state.
///
/// Looks up the token embedding and stores it in `hidden`.
/// Dispatches based on embedding quantization type (F32, Q4_0, etc.)
/// If `scratch` is provided, also computes Gemma4 PLE inputs.
pub fn cpu_embed_token(
    token_id: u32,
    weights: &CpuModelWeights,
    hidden: &mut [f32],
    config: &ModelConfig,
    scratch: Option<&mut CpuForwardScratch>,
) {
    let h = config.hidden_size;
    match weights.token_emb_meta.wtype {
        GgmlType::F32 => {
            if let Some(emb) = super::weights::try_as_f32_slice(&weights.token_emb) {
                super::quant::embed_f32(token_id as usize, emb, &mut hidden[..h]);
            } else {
                let start = token_id as usize * h * 4;
                let bytes = &weights.token_emb[start..start + h * 4];
                for i in 0..h {
                    hidden[i] = f32::from_le_bytes([
                        bytes[i * 4],
                        bytes[i * 4 + 1],
                        bytes[i * 4 + 2],
                        bytes[i * 4 + 3],
                    ]);
                }
            }
        }
        GgmlType::F16 => {
            let start_idx = token_id as usize * h;
            let emb = &weights.token_emb[start_idx * 2..(start_idx + h) * 2];

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
                unsafe {
                    embed_f16_avx2_f16c(emb, &mut hidden[..h]);
                }
            } else {
                for i in 0..h {
                    let bits = u16::from_le_bytes([emb[i * 2], emb[i * 2 + 1]]);
                    let val = half::f16::from_bits(bits).to_f32();
                    hidden[i] = val;
                }
            }
        }
        GgmlType::Q4_0 => {
            super::quant::embed_q4_0(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q4_1 => {
            super::quant::embed_q4_1(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q4_K => {
            super::quant::embed_q4_k(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q5_0 => {
            super::quant::embed_q5_0(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q6_K => {
            super::quant::embed_q6_k(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q8_0 => {
            super::quant::embed_q8_0(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q3_K => {
            super::quant::embed_q3_k(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q2_K => {
            super::quant::embed_q2_k(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        _ => {
            let msg = format!(
                "Unsupported embedding type: {:?}",
                weights.token_emb_meta.wtype
            );
            std::panic::panic_any(msg);
        }
    }

    // Gemma-style embedding scaling (sqrt(hidden_size) for gemma4, 1.0 otherwise).
    let scale = config.embedding_scale;
    if scale != 1.0 {
        for v in hidden[..h].iter_mut() {
            *v *= scale;
        }
    }

    if let Some(scratch) = scratch {
        cpu_compute_ple_inputs(token_id, &hidden[..h], weights, scratch, config);
    }
}

/// AVX2+F16C vectorized F16->F32 embedding copy.
/// Processes 8 f16 values (16 bytes) per iteration.
#[cfg(target_arch = "x86_64")]
unsafe fn embed_f16_avx2_f16c(emb: &[u8], hidden: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = hidden.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 16;
        let v16 = _mm_loadu_si128(emb.as_ptr().add(offset) as *const __m128i);
        let v32 = _mm256_cvtph_ps(v16);
        _mm256_storeu_ps(hidden.as_mut_ptr().add(i * 8), v32);
    }
    // Scalar tail
    for (i, h) in hidden.iter_mut().enumerate().take(n).skip(chunks * 8) {
        let offset = i * 2;
        let bits = u16::from_le_bytes([emb[offset], emb[offset + 1]]);
        *h = half::f16::from_bits(bits).to_f32();
    }
}
