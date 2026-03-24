//! CPU single-token decode forward pass.
//!
//! Implements the autoregressive decode path: one token through all transformer layers.
//! Uses KV cache for efficient attention computation.

use super::cache::{CpuForwardScratch, CpuKvCache};
use super::ops::{residual_add, rms_norm, rope, silu_fuse, flash_attn_decode};
use super::weights::{CpuLayerWeights, CpuModelWeights};
use super::CpuError;
use crate::config::ModelConfig;
use crate::loader::GgmlType;

// ── GEMV dispatch ────────────────────────────────────────────────────────────────

/// Dispatch GEMV based on weight type.
///
/// Computes: y = W * x (matrix-vector multiply)
fn dispatch_gemv(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] = unsafe {
                std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4)
            };
            gemv_f32(wf, x, y);
        }
        GgmlType::Q4_0 => gemv_q4_0(w, x, y, out_dim, in_dim),
        GgmlType::Q8_0 => gemv_q8_0(w, x, y, out_dim, in_dim),
        other => return Err(CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// F32 GEMV: y[row] = dot(W[row, :], x)
fn gemv_f32(w: &[f32], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    for (row, out) in y.iter_mut().enumerate() {
        let row_w = &w[row * in_dim..(row + 1) * in_dim];
        *out = row_w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
}

/// Q4_0 GEMV: dequantize on-the-fly
fn gemv_q4_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    const Q4_BLOCK_ELEMS: usize = 32;
    const Q4_BLOCK_BYTES: usize = 18;

    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    for row in 0..out_dim {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..];
            let scale = super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];

            for i in 0..16 {
                let lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                let hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                acc += lo * xb[i] + hi * xb[i + 16];
            }
        }
        y[row] = acc;
    }
}

/// Q8_0 GEMV: dequantize on-the-fly
fn gemv_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    const Q8_BLOCK_ELEMS: usize = 32;
    const Q8_BLOCK_BYTES: usize = 34;

    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q8_BLOCK_BYTES;

    for row in 0..out_dim {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &row_w[b * Q8_BLOCK_BYTES..];
            let scale = super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];

            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }
        y[row] = acc;
    }
}

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
) -> Result<(), CpuError> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let wtype = weights.weight_type;

    // 1. Attention RMS norm
    rms_norm(hidden, &weights.attn_norm, &mut scratch.normed, eps);

    // 2. QKV projections
    dispatch_gemv(&weights.attn_q, wtype, &scratch.normed, &mut scratch.q, q_size, h)?;
    dispatch_gemv(&weights.attn_k, wtype, &scratch.normed, &mut scratch.k, kv_size, h)?;
    dispatch_gemv(&weights.attn_v, wtype, &scratch.normed, &mut scratch.v, kv_size, h)?;

    // 3. RoPE on Q and K
    rope(&mut scratch.q, config.num_heads, config.head_dim, pos, config.rope_theta, config.rope_neox);
    rope(&mut scratch.k, config.num_kv_heads, config.head_dim, pos, config.rope_theta, config.rope_neox);

    // 4. Write K, V to cache
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

    // 6. Output projection
    dispatch_gemv(&weights.attn_o, wtype, &scratch.attn_out, &mut scratch.layer_out, h, q_size)?;

    // 7. Residual
    residual_add(hidden, &scratch.layer_out);

    // 8. FFN RMS norm
    rms_norm(hidden, &weights.ffn_norm, &mut scratch.normed, eps);

    // 9. FFN: gate + up projections
    dispatch_gemv(&weights.ffn_gate, wtype, &scratch.normed, &mut scratch.gate, ff_size, h)?;
    dispatch_gemv(&weights.ffn_up, wtype, &scratch.normed, &mut scratch.swiglu, ff_size, h)?;

    // 10. SwiGLU: silu(gate) * up
    silu_fuse(&scratch.gate, &mut scratch.swiglu);

    // 11. Down projection
    dispatch_gemv(&weights.ffn_down, wtype, &scratch.swiglu, &mut scratch.layer_out, h, ff_size)?;

    // 12. Residual
    residual_add(hidden, &scratch.layer_out);

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
        )?;
    }

    // Final RMS norm
    rms_norm(hidden, &weights.output_norm, &mut scratch.normed, config.rms_norm_eps);

    // LM head: project to vocabulary
    let h = config.hidden_size;
    let v = config.vocab_size;
    dispatch_gemv(
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
    fn gemv_f32_correct() {
        // 2x3 matrix times 3-vector
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows, 3 cols
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];

        gemv_f32(&w, &x, &mut y);

        // Row 0: 1*1 + 2*2 + 3*3 = 14
        // Row 1: 4*1 + 5*2 + 6*3 = 32
        assert!((y[0] - 14.0).abs() < 1e-5);
        assert!((y[1] - 32.0).abs() < 1e-5);
    }
}
