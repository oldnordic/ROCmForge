//! CPU batched prefill — processes entire prompt in one pass.
//!
//! Uses GEMM (batched matrix multiply) for QKV projections instead of GEMV.
//! Populates KV cache for subsequent decode steps.

use super::cache::{CpuForwardScratch, CpuKvCache};
use super::ops::{add_bias_batched, dispatch_gemm, flash_attn_prefill, rms_norm, silu_fuse};
use super::quant::{embed_f32_batch, embed_q4_0_batch};
use super::weights::{CpuLayerWeights, CpuModelWeights};
use super::CpuError;
use crate::config::ModelConfig;
use crate::loader::GgmlType;

// ── PrefillScratch ────────────────────────────────────────────────────────────

/// Scratch buffers for batched prefill.
///
/// All buffers are sized for [seq_len, dim] layout.
struct CpuPrefillScratch {
    /// Hidden states [seq_len * hidden_size]
    hidden: Vec<f32>,
    /// Normalized hidden [seq_len * hidden_size]
    normed: Vec<f32>,
    /// Query [seq_len * q_size]
    q: Vec<f32>,
    /// Key [seq_len * kv_size]
    k: Vec<f32>,
    /// Value [seq_len * kv_size]
    v: Vec<f32>,
    /// Attention output [seq_len * q_size]
    attn_out: Vec<f32>,
    /// Layer output [seq_len * hidden_size]
    layer_out: Vec<f32>,
    /// FFN gate [seq_len * intermediate_size]
    gate: Vec<f32>,
    /// FFN SwiGLU [seq_len * intermediate_size]
    swiglu: Vec<f32>,
}

impl CpuPrefillScratch {
    fn new(config: &ModelConfig, seq_len: usize) -> Self {
        let n = seq_len;
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        Self {
            hidden: vec![0.0; n * h],
            normed: vec![0.0; n * h],
            q: vec![0.0; n * q],
            k: vec![0.0; n * kv],
            v: vec![0.0; n * kv],
            attn_out: vec![0.0; n * q],
            layer_out: vec![0.0; n * h],
            gate: vec![0.0; n * ff],
            swiglu: vec![0.0; n * ff],
        }
    }
}

// ── prefill_layer_forward ─────────────────────────────────────────────────────

fn prefill_layer_forward(
    ps: &mut CpuPrefillScratch,
    weights: &CpuLayerWeights,
    kv: &mut CpuKvCache,
    layer: usize,
    start_pos: usize,
    config: &ModelConfig,
    seq_len: usize,
) -> Result<(), CpuError> {
    let h = config.hidden_size;
    let q_s = config.num_heads * config.head_dim;
    let kv_s = config.num_kv_heads * config.head_dim;
    let ff = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let wtype = weights.weight_type;

    // 1. RMS norm each row: normed[s, :] = rms_norm(hidden[s, :], attn_norm)
    for s in 0..seq_len {
        let xr = &ps.hidden[s * h..(s + 1) * h];
        let or = &mut ps.normed[s * h..(s + 1) * h];
        rms_norm(xr, &weights.attn_norm, or, eps);
    }

    // 2. QKV GEMM
    dispatch_gemm(
        &weights.attn_q,
        wtype,
        &ps.normed,
        &mut ps.q,
        q_s,
        h,
    )?;
    dispatch_gemm(
        &weights.attn_k,
        wtype,
        &ps.normed,
        &mut ps.k,
        kv_s,
        h,
    )?;
    dispatch_gemm(
        &weights.attn_v,
        wtype,
        &ps.normed,
        &mut ps.v,
        kv_s,
        h,
    )?;

    // 3. Optional biases
    if let Some(bq) = &weights.attn_q_bias {
        add_bias_batched(&mut ps.q, bq, q_s, seq_len);
    }
    if let Some(bk) = &weights.attn_k_bias {
        add_bias_batched(&mut ps.k, bk, kv_s, seq_len);
    }
    if let Some(bv) = &weights.attn_v_bias {
        add_bias_batched(&mut ps.v, bv, kv_s, seq_len);
    }

    // 4. RoPE batched
    let row_len = config.num_heads * config.head_dim;
    for s in 0..seq_len {
        let qr = &mut ps.q[s * row_len..(s + 1) * row_len];
        super::ops::rope(
            qr,
            config.num_heads,
            config.head_dim,
            start_pos + s,
            config.rope_theta,
            config.rope_neox,
        );
    }
    let kv_row_len = config.num_kv_heads * config.head_dim;
    for s in 0..seq_len {
        let kr = &mut ps.k[s * kv_row_len..(s + 1) * kv_row_len];
        super::ops::rope(
            kr,
            config.num_kv_heads,
            config.head_dim,
            start_pos + s,
            config.rope_theta,
            config.rope_neox,
        );
    }

    // 5. KV write batched
    for s in 0..seq_len {
        let pos = start_pos + s;
        let k_row = &ps.k[s * kv_s..(s + 1) * kv_s];
        let v_row = &ps.v[s * kv_s..(s + 1) * kv_s];
        kv.write_k(layer, pos, k_row);
        kv.write_v(layer, pos, v_row);
    }

    // 6. Causal flash attention (prefill: Q, K, V are all [seq_len, ...])
    flash_attn_prefill(
        &ps.q,
        &ps.k,
        &ps.v,
        &mut ps.attn_out,
        seq_len,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
    );

    // 7. O projection GEMM
    dispatch_gemm(
        &weights.attn_o,
        wtype,
        &ps.attn_out,
        &mut ps.layer_out,
        h,
        q_s,
    )?;

    // 8. Residual: hidden += layer_out
    for i in 0..seq_len * h {
        ps.hidden[i] += ps.layer_out[i];
    }

    // 9. Post-attention RMS norm batched
    for s in 0..seq_len {
        let xr = &ps.hidden[s * h..(s + 1) * h];
        let or = &mut ps.normed[s * h..(s + 1) * h];
        rms_norm(xr, &weights.ffn_norm, or, eps);
    }

    // 10. MLP: gate + up projections
    dispatch_gemm(
        &weights.ffn_gate,
        wtype,
        &ps.normed,
        &mut ps.gate,
        ff,
        h,
    )?;
    dispatch_gemm(
        &weights.ffn_up,
        wtype,
        &ps.normed,
        &mut ps.swiglu,
        ff,
        h,
    )?;

    // 11. SwiGLU: silu(gate) * up
    silu_fuse(&ps.gate, &mut ps.swiglu);

    // 12. Down projection GEMM
    dispatch_gemm(
        &weights.ffn_down,
        wtype,
        &ps.swiglu,
        &mut ps.layer_out,
        h,
        ff,
    )?;

    // 13. Residual: hidden += layer_out
    for i in 0..seq_len * h {
        ps.hidden[i] += ps.layer_out[i];
    }

    Ok(())
}

// ── cpu_prefill_forward ───────────────────────────────────────────────────────

/// Process the entire prompt in one batched CPU pass.
///
/// Fills `scratch.logits` with the next-token prediction from the last prompt position.
/// KV cache populated for positions start_pos..start_pos+tokens.len().
///
/// # Arguments
/// * `tokens` - Prompt token IDs
/// * `weights` - Model weights
/// * `kv` - KV cache (will be populated)
/// * `scratch` - Scratch buffers for final output
/// * `start_pos` - Starting position in KV cache (0 for new conversation)
/// * `config` - Model configuration
pub fn cpu_prefill_forward(
    tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    start_pos: usize,
    config: &ModelConfig,
) -> Result<(), CpuError> {
    let seq_len = tokens.len();
    assert!(seq_len > 0, "cpu_prefill_forward: empty token list");
    assert!(
        start_pos + seq_len <= kv.max_seq_len,
        "prompt longer than KV cache"
    );

    let mut ps = CpuPrefillScratch::new(config, seq_len);
    let h = config.hidden_size;

    // 1. Gather embeddings
    match weights.token_emb_type {
        GgmlType::F32 => {
            let wf: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    weights.token_emb.as_ptr() as *const f32,
                    weights.token_emb.len() / 4,
                )
            };
            embed_f32_batch(tokens, wf, &mut ps.hidden, h);
        }
        GgmlType::Q4_0 => {
            embed_q4_0_batch(tokens, &weights.token_emb, &mut ps.hidden, h);
        }
        other => {
            return Err(CpuError::UnsupportedWeightType(other));
        }
    }

    // 2. All layers
    for layer_idx in 0..config.num_layers {
        prefill_layer_forward(
            &mut ps,
            weights.layer(layer_idx),
            kv,
            layer_idx,
            start_pos,
            config,
            seq_len,
        )?;
    }

    // 3. Extract last row of hidden
    let last_row_start = (seq_len - 1) * h;
    let last_row = &ps.hidden[last_row_start..last_row_start + h];

    // 4. Final RMS norm on last row
    rms_norm(last_row, &weights.output_norm, &mut scratch.normed, config.rms_norm_eps);

    // 5. LM head GEMV (use decode path for single token)
    let v = config.vocab_size;
    super::forward::dispatch_gemv(
        &weights.lm_head,
        weights.lm_head_type,
        &scratch.normed,
        &mut scratch.logits,
        v,
        h,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 2,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 100,
            max_seq_len: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_neox: true,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "qwen2".to_string(),
        }
    }

    #[test]
    fn prefill_scratch_sizes() {
        let config = make_test_config();
        let ps = CpuPrefillScratch::new(&config, 4);
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        assert_eq!(ps.hidden.len(), 4 * h);
        assert_eq!(ps.normed.len(), 4 * h);
        assert_eq!(ps.q.len(), 4 * q);
        assert_eq!(ps.k.len(), 4 * kv);
        assert_eq!(ps.v.len(), 4 * kv);
        assert_eq!(ps.attn_out.len(), 4 * q);
        assert_eq!(ps.layer_out.len(), 4 * h);
        assert_eq!(ps.gate.len(), 4 * ff);
        assert_eq!(ps.swiglu.len(), 4 * ff);
    }
}
