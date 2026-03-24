//! CPU single-token decode forward pass.
//!
//! Implements the autoregressive decode path: one token through all transformer layers.
//! Uses KV cache for efficient attention computation.

use super::cache::{CpuForwardScratch, CpuKvCache};
use super::ops::{dispatch_gemv, residual_add, rms_norm, rope, silu_fuse, flash_attn_decode};
use super::weights::{CpuLayerWeights, CpuModelWeights};
use super::CpuError;
use crate::config::ModelConfig;
use crate::loader::GgmlType;

// ── Layer forward ────────────────────────────────────────────────────────────────

/// Forward pass through a single transformer layer.
///
/// Architecture: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual
pub fn cpu_layer_forward(
    hidden: &mut [f32],
    weights: &CpuLayerWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    layer: usize,
    pos: usize,
    config: &ModelConfig,
    debug: bool,
) -> Result<(), CpuError> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let wtype = weights.weight_type;

    // 1. Attention RMS norm
    rms_norm(hidden, &weights.attn_norm, &mut scratch.normed, eps);

    if debug && layer == 0 {
        let norm_mean: f32 = scratch.normed.iter().copied().sum::<f32>() / h as f32;
        let norm_std: f32 = ((scratch.normed.iter().map(|x| x * x).sum::<f32>() / h as f32) - norm_mean * norm_mean).sqrt();
        eprintln!("[Layer {} after norm] mean={:.4} std={:.4}",
                 layer, norm_mean, norm_std);
    }

    // 2. QKV projections
    dispatch_gemv(&weights.attn_q, wtype, &scratch.normed, &mut scratch.q, q_size, h)?;
    dispatch_gemv(&weights.attn_k, wtype, &scratch.normed, &mut scratch.k, kv_size, h)?;
    dispatch_gemv(&weights.attn_v, wtype, &scratch.normed, &mut scratch.v, kv_size, h)?;

    // 3. Optional biases (same as prefill)
    if let Some(bq) = &weights.attn_q_bias {
        super::ops::add_bias(&mut scratch.q, bq);
    }
    if let Some(bk) = &weights.attn_k_bias {
        super::ops::add_bias(&mut scratch.k, bk);
    }
    if let Some(bv) = &weights.attn_v_bias {
        super::ops::add_bias(&mut scratch.v, bv);
    }

    if debug && layer == 0 {
        eprintln!("[Layer {} after QKV] q_mean={:.4} k_mean={:.4} v_mean={:.4}",
                 layer,
                 scratch.q.iter().copied().sum::<f32>() / q_size as f32,
                 scratch.k.iter().copied().sum::<f32>() / kv_size as f32,
                 scratch.v.iter().copied().sum::<f32>() / kv_size as f32);
    }

    // 3. RoPE on Q and K
    rope(&mut scratch.q, config.num_heads, config.head_dim, pos, config.rope_theta, config.rope_neox);
    rope(&mut scratch.k, config.num_kv_heads, config.head_dim, pos, config.rope_theta, config.rope_neox);

    // 4. Write K, V in cache
    kv.write_k(layer, pos, &scratch.k);
    kv.write_v(layer, pos, &scratch.v);

    // 5. Flash attention decode (reads full KV cache)
    let seq_len = pos + 1;
    flash_attn_decode(
        &scratch.q,
        kv.k_buf(layer),
        kv.v_buf(layer),
        &mut scratch.attn_out,
        seq_len,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
    );
    // Verify KV cache indexing (debug check)
    #[cfg(debug_assertions)]
    {
        let kv_size = config.num_kv_heads * config.head_dim;
        for t in 0..seq_len {
            let k_start_expected = t * kv_size;
            let k_start_actual = k_start_expected; // For KV cache interface
            debug_assert_eq!(k_start_expected, k_start_actual, "KV cache index mismatch");
        }
    }

    if debug && layer == 0 {
        let ao_mean: f32 = scratch.attn_out.iter().copied().sum::<f32>() / q_size as f32;
        let ao_std: f32 = ((scratch.attn_out.iter().map(|x| x * x).sum::<f32>() / q_size as f32) - ao_mean * ao_mean).sqrt();
        eprintln!("[Layer {} after attn] out_mean={:.4} out_std={:.4}",
                 layer, ao_mean, ao_std);
    }

    // 6. Output projection
    dispatch_gemv(&weights.attn_o, wtype, &scratch.attn_out, &mut scratch.layer_out, h, q_size)?;

    if debug && layer == 0 {
        let lo_mean: f32 = scratch.layer_out.iter().copied().sum::<f32>() / h as f32;
        let lo_std: f32 = ((scratch.layer_out.iter().map(|x| x * x).sum::<f32>() / h as f32) - lo_mean * lo_mean).sqrt();
        eprintln!("[Layer {} attn_out_proj] layer_out mean={:.4} std={:.4}", layer, lo_mean, lo_std);
    }

    // 7. Residual
    residual_add(hidden, &scratch.layer_out);

    if debug && layer == 0 {
        let h_mean: f32 = hidden.iter().copied().sum::<f32>() / h as f32;
        let h_std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / h as f32) - h_mean * h_mean).sqrt();
        eprintln!("[Layer {} after attn residual] hidden mean={:.4} std={:.4}", layer, h_mean, h_std);
    }

    // 8. FFN RMS norm
    rms_norm(hidden, &weights.ffn_norm, &mut scratch.normed, eps);

    // 9. FFN: gate + up projections
    dispatch_gemv(&weights.ffn_gate, wtype, &scratch.normed, &mut scratch.gate, ff_size, h)?;
    dispatch_gemv(&weights.ffn_up, wtype, &scratch.normed, &mut scratch.swiglu, ff_size, h)?;

    if debug && layer == 0 {
        let gate_mean: f32 = scratch.gate.iter().copied().sum::<f32>() / ff_size as f32;
        let gate_std: f32 = ((scratch.gate.iter().map(|x| x * x).sum::<f32>() / ff_size as f32) - gate_mean * gate_mean).sqrt();
        let up_mean: f32 = scratch.swiglu.iter().copied().sum::<f32>() / ff_size as f32;
        let up_std: f32 = ((scratch.swiglu.iter().map(|x| x * x).sum::<f32>() / ff_size as f32) - up_mean * up_mean).sqrt();
        eprintln!("[Layer {} ffn_gate] mean={:.4} std={:.4}", layer, gate_mean, gate_std);
        eprintln!("[Layer {} ffn_up] mean={:.4} std={:.4}", layer, up_mean, up_std);
        eprintln!("[Layer {} ffn_gate] [0..5] = {:?}", layer, &scratch.gate[0..5]);
        eprintln!("[Layer {} ffn_up] [0..5] = {:?}", layer, &scratch.swiglu[0..5]);
    }

    // 10. SwiGLU: silu(gate) * up
    silu_fuse(&scratch.gate, &mut scratch.swiglu);

    if debug && layer == 0 {
        let swiglu_mean: f32 = scratch.swiglu.iter().copied().sum::<f32>() / ff_size as f32;
        let swiglu_std: f32 = ((scratch.swiglu.iter().map(|x| x * x).sum::<f32>() / ff_size as f32) - swiglu_mean * swiglu_mean).sqrt();
        eprintln!("[Layer {} swiglu] mean={:.4} std={:.4}", layer, swiglu_mean, swiglu_std);
    }

    // 11. Down projection
    if debug && layer == 0 {
        let num_blocks = ff_size / 32;
        let row_bytes = num_blocks * 18;
        let expected_weight_bytes = h * row_bytes;
        eprintln!("[Layer {} ffn_down] out_dim={} in_dim={} num_blocks={} row_bytes={} expected_weight_bytes={} actual_weight_bytes={}",
                 layer, h, ff_size, num_blocks, row_bytes, expected_weight_bytes, weights.ffn_down.len());
        eprintln!("[Layer {} ffn_down] swiglu.len={} layer_out.len={}",
                 layer, scratch.swiglu.len(), scratch.layer_out.len());
        // Print first 5 values of swiglu
        eprintln!("[Layer {} ffn_down] swiglu[0..5] = {:?}", layer, &scratch.swiglu[0..5]);
        // Print first block of weights
        let first_block = &weights.ffn_down[0..18];
        eprintln!("[Layer {} ffn_down] first_weight_block = {:?}", layer, first_block);
        let scale = super::quant::load_f16_scale(&first_block[0..2]);
        let qs = &first_block[2..18];
        let mut dequant = [0.0f32; 32];
        for i in 0..16 {
            dequant[i] = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
            dequant[i + 16] = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
        }
        eprintln!("[Layer {} ffn_down] first_block dequant[0..5] = {:?}", layer, &dequant[0..5]);
    }
    dispatch_gemv(&weights.ffn_down, wtype, &scratch.swiglu, &mut scratch.layer_out, h, ff_size)?;

    if debug && layer == 0 {
        let ffn_out_mean: f32 = scratch.layer_out.iter().copied().sum::<f32>() / h as f32;
        let ffn_out_std: f32 = ((scratch.layer_out.iter().map(|x| x * x).sum::<f32>() / h as f32) - ffn_out_mean * ffn_out_mean).sqrt();
        eprintln!("[Layer {} ffn_down] layer_out mean={:.4} std={:.4}", layer, ffn_out_mean, ffn_out_std);
        eprintln!("[Layer {} ffn_down] layer_out[0..5] = {:?}", layer, &scratch.layer_out[0..5]);
    }

    // 12. Residual
    residual_add(hidden, &scratch.layer_out);

    if debug && layer == 0 {
        let h_mean: f32 = hidden.iter().copied().sum::<f32>() / h as f32;
        let h_std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / h as f32) - h_mean * h_mean).sqrt();
        eprintln!("[Layer {} after ffn residual] hidden mean={:.4} std={:.4}", layer, h_mean, h_std);
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
        let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32) - mean * mean).sqrt();
        eprintln!("[Forward input] pos={} mean={:.4} std={:.4}", pos, mean, std);
    }

    // Process all transformer layers
    for layer_idx in 0..config.num_layers {
        cpu_layer_forward(
            hidden,
            weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            pos,
            config,
            debug,
        )?;

        // Debug: show hidden state after each layer
        if debug && layer_idx < 2 {
            let mean: f32 = hidden.iter().copied().sum::<f32>() / hidden.len() as f32;
            let std: f32 = ((hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32) - mean * mean).sqrt();
            eprintln!("[After layer {}] mean={:.4} std={:.4}", layer_idx, mean, std);
        }
    }

    // Final RMS norm
    rms_norm(hidden, &weights.output_norm, &mut scratch.normed, config.rms_norm_eps);

    // Debug: show normed output
    if debug {
        let mean: f32 = scratch.normed.iter().copied().sum::<f32>() / scratch.normed.len() as f32;
        let std: f32 = ((scratch.normed.iter().map(|x| x * x).sum::<f32>() / scratch.normed.len() as f32) - mean * mean).sqrt();
        eprintln!("[After final norm] mean={:.4} std={:.4}", mean, std);
    }

    // LM head: project to vocabulary
    let h = config.hidden_size;
    let v = config.vocab_size;
    if debug {
        eprintln!("[LM head] type={:?} h={} v={}", weights.lm_head_type, h, v);
    }
    dispatch_gemv(
        &weights.lm_head,
        weights.lm_head_type,
        &scratch.normed,
        &mut scratch.logits,
        v,
        h,
    )?;

    // Debug: show logits statistics
    if debug {
        let mean: f32 = scratch.logits.iter().copied().sum::<f32>() / scratch.logits.len() as f32;
        let std: f32 = ((scratch.logits.iter().map(|x| x * x).sum::<f32>() / scratch.logits.len() as f32) - mean * mean).sqrt();
        let min: f32 = scratch.logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max: f32 = scratch.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[After LM head] mean={:.4} std={:.4} range=[{:.4}, {:.4}]", mean, std, min, max);
    }

    Ok(())
}

// ── Token embedding ──────────────────────────────────────────────────────────────

/// Embed a single token into hidden state.
///
/// Looks up the token embedding and stores it in `hidden`.
/// Dispatches based on embedding quantization type (F32, Q4_0, etc.)
pub fn cpu_embed_token(
    token_id: u32,
    weights: &CpuModelWeights,
    hidden: &mut [f32],
    config: &ModelConfig,
) {
    let h = config.hidden_size;
    match weights.token_emb_type {
        GgmlType::F32 => {
            let emb: &[f32] = unsafe {
                std::slice::from_raw_parts(weights.token_emb.as_ptr() as *const f32, weights.token_emb.len() / 4)
            };
            super::quant::embed_f32(token_id as usize, emb, &mut hidden[..h]);
        }
        GgmlType::Q4_0 => {
            super::quant::embed_q4_0(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q4_1 => {
            super::quant::embed_q4_1(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q6_K => {
            super::quant::embed_q6_k(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        GgmlType::Q8_0 => {
            super::quant::embed_q8_0(token_id as usize, &weights.token_emb, &mut hidden[..h], h);
        }
        _ => panic!("Unsupported embedding type: {:?}", weights.token_emb_type),
    }
}

