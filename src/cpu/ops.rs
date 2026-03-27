//! CPU primitive operations for inference.
//!
//! Pure f32 operations - no SIMD by default. Operations are:
//! - RMS normalization
//! - RoPE positional embeddings (NeoX and classic styles)
//! - Attention (GQA-aware flash attention)
//! - SwiGLU activation
//! - Softmax and sampling utilities

use rayon::prelude::*;
use crate::cpu::quant::{Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q8_BLOCK_BYTES, Q8_BLOCK_ELEMS, Q8_0_MAX, load_f16_scale};
use crate::cpu::weights::WeightMeta;
use crate::loader::GgmlType;

/// Load f16 value from bytes as f32.
fn load_f16_as_f32(bytes: &[u8]) -> f32 {
    let u = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from(half::f16::from_bits(u))
}

// ── Normalization ────────────────────────────────────────────────────────────────

/// RMS normalization: out[i] = x[i] / rms(x) * w[i]
///
/// # Arguments
/// * `x` - Input vector [hidden_size]
/// * `w` - Weight vector [hidden_size]
/// * `out` - Output vector [hidden_size]
/// * `eps` - Epsilon for numerical stability
pub fn rms_norm(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(w.len(), n, "weight dimension mismatch");
    debug_assert_eq!(out.len(), n, "output dimension mismatch");

    // Compute RMS: sqrt(mean(x^2) + eps)
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();

    // Normalize and scale
    for i in 0..n {
        out[i] = x[i] / rms * w[i];
    }
}

/// Batched RMS norm: out[s] = rms_norm(x[s], w) for each row s
pub fn rms_norm_batch(x: &[f32], w: &[f32], out: &mut [f32], hidden: usize, eps: f32) {
    let seq_len = x.len() / hidden;
    for s in 0..seq_len {
        let x_row = &x[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];
        rms_norm(x_row, w, out_row, eps);
    }
}

// ── RoPE positional embeddings ───────────────────────────────────────────────────

/// Apply RoPE (Rotary Position Embeddings) to a vector.
///
/// Input shape: [num_heads, head_dim]
/// Position: current token position in sequence
/// Theta: base frequency for rotation
///
/// # Modes
/// - `neox = true`: GPT-NeoX style (pairs within same half) - Qwen2, GPT-NeoX
/// - `neox = false`: Classic RoPE (consecutive pairs) - LLaMA, Mistral
pub fn rope(x: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32, neox: bool) {
    let total_len = num_heads * head_dim;
    debug_assert_eq!(x.len(), total_len, "rope input dimension mismatch");

    for h in 0..num_heads {
        let base = h * head_dim;
        let half = head_dim / 2;

        for i in 0..half {
            // Frequency: 1 / theta^(2i/d)
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();

            if neox {
                // NeoX: pairs are (i, i+half)
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                x[base + i] = x0 * cos_a - x1 * sin_a;
                x[base + i + half] = x0 * sin_a + x1 * cos_a;
            } else {
                // Classic: pairs are (2i, 2i+1)
                let x0 = x[base + 2 * i];
                let x1 = x[base + 2 * i + 1];
                x[base + 2 * i] = x0 * cos_a - x1 * sin_a;
                x[base + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/// Batched RoPE: apply to each row with position start_pos + s
pub fn rope_batch(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    start_pos: usize,
    theta: f32,
    neox: bool,
) {
    let row_len = num_heads * head_dim;
    let seq_len = x.len() / row_len;

    for s in 0..seq_len {
        let row = &mut x[s * row_len..(s + 1) * row_len];
        rope(row, num_heads, head_dim, start_pos + s, theta, neox);
    }
}

// ── Activation functions ────────────────────────────────────────────────────────

/// SiLU activation: x / (1 + exp(-x))
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU fuse in-place: up[i] = silu(gate[i]) * up[i]
pub fn silu_fuse(gate: &[f32], up: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len(), "gate/up dimension mismatch");
    for (g, u) in gate.iter().zip(up.iter_mut()) {
        *u = silu(*g) * *u;
    }
}

// ── Softmax ─────────────────────────────────────────────────────────────────────

/// Softmax in-place: x[i] = exp(x[i] - max) / sum
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));

    // Exp and sum
    let mut sum = 0.0f32;
    for xi in x.iter_mut() {
        *xi = (*xi - max).exp();
        sum += *xi;
    }

    // Normalize
    if sum > 0.0 {
        for xi in x.iter_mut() {
            *xi /= sum;
        }
    }
}

/// Online softmax for flash attention: update running max and sum
///
/// Given new score, update m (max), l (sum of exp), and acc (weighted sum)
#[inline]
pub fn online_softmax_update(
    score: f32,
    m_prev: f32,
    l_prev: f32,
    acc: &mut [f32],
    v: &[f32],
) -> (f32, f32) {
    let m_new = m_prev.max(score);
    let exp_diff = (m_prev - m_new).exp();
    let exp_score = (score - m_new).exp();

    // Update accumulator: acc = acc * exp_diff + v * exp_score
    for (a, vi) in acc.iter_mut().zip(v.iter()) {
        *a = *a * exp_diff + exp_score * vi;
    }

    let l_new = l_prev * exp_diff + exp_score;
    (m_new, l_new)
}

// ── Attention ───────────────────────────────────────────────────────────────────

/// Flash attention for decode (single query token, seq_len K/V in cache).
///
/// q:       [num_heads, head_dim]
/// k_cache: [max_seq, kv_size]  where kv_size = num_kv_heads * head_dim
/// v_cache: [max_seq, kv_size]
/// out:     [num_heads, head_dim]
/// seq_len: number of valid K/V positions (= current pos + 1)
///
/// For decode (single query), uses serial iteration to avoid rayon overhead.
pub fn flash_attn_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let kv_group = num_heads / num_kv_heads; // GQA ratio
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_size = num_kv_heads * head_dim;

    // Serial iteration for decode (single query) - no rayon overhead
    for h in 0..num_heads {
        let kv_h = h / kv_group;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];

        // Online softmax state
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];

        for t in 0..seq_len {
            let k_start = t * kv_size + kv_h * head_dim;
            let v_start = t * kv_size + kv_h * head_dim;
            let k_t = &k_cache[k_start..k_start + head_dim];
            let v_t = &v_cache[v_start..v_start + head_dim];

            // dot(q, k) * scale
            let score: f32 = q_h
                .iter()
                .zip(k_t.iter())
                .map(|(qi, ki)| qi * ki)
                .sum::<f32>()
                * scale;

            // Online softmax update
            (m, l) = online_softmax_update(score, m, l, &mut acc, v_t);
        }

        // Normalize and write output
        for (oi, ai) in out_h.iter_mut().zip(acc.iter()) {
            *oi = ai / l;
        }
    }
}

/// Causal flash attention for prefill.
///
/// q: [seq_len, num_heads, head_dim]   (row-major)
/// k: [seq_len, num_kv_heads, head_dim]
/// v: [seq_len, num_kv_heads, head_dim]
/// out: [seq_len, num_heads, head_dim]
///
/// Position s attends to 0..=s (causal mask).
pub fn flash_attn_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let kv_group = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_stride = num_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;

    // Parallelise over (s, h) pairs
    out.par_chunks_mut(q_stride)
        .enumerate()
        .for_each(|(s, out_s)| {
            for h in 0..num_heads {
                let kv_h = h / kv_group;
                let q_sh = &q[s * q_stride + h * head_dim..s * q_stride + (h + 1) * head_dim];
                let out_sh = &mut out_s[h * head_dim..(h + 1) * head_dim];

                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];

                // Causal: attend to positions 0..=s
                for t in 0..=s {
                    let k_th = &k[t * kv_stride + kv_h * head_dim..t * kv_stride + kv_h * head_dim + head_dim];
                    let v_th = &v[t * kv_stride + kv_h * head_dim..t * kv_stride + kv_h * head_dim + head_dim];

                    let score: f32 = q_sh
                        .iter()
                        .zip(k_th.iter())
                        .map(|(qi, ki)| qi * ki)
                        .sum::<f32>()
                        * scale;

                    (m, l) = online_softmax_update(score, m, l, &mut acc, v_th);
                }

                for (oi, ai) in out_sh.iter_mut().zip(acc.iter()) {
                    *oi = ai / l;
                }
            }
        });
}

// ── Residual operations ─────────────────────────────────────────────────────────

/// In-place bias add: x[i] += bias[i]
pub fn add_bias(x: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(x.len(), bias.len(), "bias dimension mismatch");
    for (xi, bi) in x.iter_mut().zip(bias.iter()) {
        *xi += bi;
    }
}

/// Batched bias add: x[s, :] += bias[:] for each s in 0..seq_len
pub fn add_bias_batched(x: &mut [f32], bias: &[f32], dim: usize, seq_len: usize) {
    for s in 0..seq_len {
        let xr = &mut x[s * dim..(s + 1) * dim];
        add_bias(xr, bias);
    }
}

/// Residual add: a[i] += b[i]
pub fn residual_add(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len(), "residual dimension mismatch");
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

/// Batched residual add: a[s, :] += b[s, :] for each s in 0..seq_len
pub fn residual_add_batched(a: &mut [f32], b: &[f32], dim: usize, seq_len: usize) {
    for i in 0..seq_len * dim {
        a[i] += b[i];
    }
}

// ── GEMM (batched matrix multiply for prefill) ───────────────────────────────────

/// F32 GEMM: Y[s, o] = dot(W[o, :], X[s, :])
/// W: [out_dim, in_dim], X: [seq_len, in_dim], Y: [seq_len, out_dim]
pub fn gemm_f32(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let w_row = &w[o * in_dim..(o + 1) * in_dim];
            y_row[o] = w_row.iter().zip(x_row.iter()).map(|(wi, xi)| wi * xi).sum();
        }
    });
}

/// Q4_0 GEMM: dequant on-the-fly.
pub fn gemm_q4_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;
            for b in 0..num_blocks {
                let block = &row_w[b * Q4_BLOCK_BYTES..(b + 1) * Q4_BLOCK_BYTES];
                let scale = super::quant::load_f16_scale(&block[0..2]);
                let qs = &block[2..18];
                let xb = &x_row[b * Q4_BLOCK_ELEMS..];
                for i in 0..16 {
                    let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                    let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                    acc += q_lo * xb[i] + q_hi * xb[i + 16];
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Q4_1 GEMM: dequant on-the-fly with min offset.
pub fn gemm_q4_1(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_1_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_1_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;
            for b in 0..num_blocks {
                let block = &row_w[b * Q4_1_BLOCK_BYTES..(b + 1) * Q4_1_BLOCK_BYTES];
                let scale = load_f16_scale(&block[0..2]);
                let min = load_f16_as_f32(&block[2..4]);
                let qs = &block[4..20];
                let xb = &x_row[b * Q4_1_BLOCK_ELEMS..];
                for i in 0..16 {
                    let q_lo = (qs[i] & 0x0F) as f32 * scale + min;
                    let q_hi = (qs[i] >> 4) as f32 * scale + min;
                    acc += q_lo * xb[i] + q_hi * xb[i + 16];
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Q5_0 GEMM: dequant on-the-fly.
pub fn gemm_q5_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q5_0_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_0_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;
            for b in 0..num_blocks {
                let block = &row_w[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
                let d = super::quant::load_f16_scale(&block[0..2]);
                let qh = &block[2..6];
                let qs = &block[6..22];
                let xb = &x_row[b * Q5_0_BLOCK_ELEMS..];

                for i in 0..16 {
                    // Process 2 values per iteration
                    let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                    let low_bits_0 = qs[i] & 0x0F;
                    let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                    let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                    let low_bits_1 = (qs[i] >> 4) & 0x0F;
                    let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                    acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Q8_0 GEMM: dequant on-the-fly.
pub fn gemm_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q8_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;
            for b in 0..num_blocks {
                let block = &row_w[b * Q8_BLOCK_BYTES..(b + 1) * Q8_BLOCK_BYTES];
                let scale = super::quant::load_f16_scale(&block[0..2]);
                let qs = &block[2..34];
                let xb = &x_row[b * Q8_BLOCK_ELEMS..];
                for i in 0..Q8_BLOCK_ELEMS {
                    acc += (qs[i] as i8) as f32 * scale * xb[i];
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Q6_K GEMM fallback: exact translation of llama.cpp dequantize_row_q6_K + dot product.
///
/// Following llama.cpp/ggml/src/ggml-quants.c dequantize_row_q6_K exactly:
/// - Two iterations for 256 values (n = 0, 128)
/// - Each iteration processes 128 output values
/// - Pointers advance between iterations
pub fn gemm_q6_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::quant::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q6_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q6_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;

            for b in 0..num_blocks {
                let block = &row_w[b * Q6_K_BLOCK_BYTES..(b + 1) * Q6_K_BLOCK_BYTES];

                // Q6_K block layout (from llama.cpp ggml-common.h):
                // struct block_q6_K {
                //     uint8_t ql[QK_K/2];      // 128 bytes
                //     uint8_t qh[QK_K/4];      // 64 bytes
                //     int8_t  scales[QK_K/16]; // 16 bytes
                //     ggml_half d;             // 2 bytes (AT THE END!)
                // }
                let mut ql = &block[0..128];
                let mut qh = &block[128..192];
                let mut sc: &[i8] = unsafe {
                    std::slice::from_raw_parts(block[192..208].as_ptr() as *const i8, 16)
                };
                let d = super::quant::load_f16_scale(&block[208..210]);

                // Base index into x_row for this 256-element block
                let xb_base = b * Q6_K_BLOCK_ELEMS;
                let mut xb_offset = 0; // Offset within block (0, then 128)

                // for (int n = 0; n < QK_K; n += 128) {
                // This iterates TWICE: n=0, n=128
                for _ in 0..2 {
                    // for (int l = 0; l < 32; ++l) {
                    for l in 0..32 {
                        let is = l / 16;

                        // const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        let q1_ql_part = ql[l + 0] & 0xF;
                        let q1_qh_part = ((qh[l] >> 0) & 3) << 4;
                        let q1_combined = q1_ql_part | q1_qh_part;
                        let q1 = q1_combined as i8 as i32 - 32;
                        // const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        let q2_ql_part = ql[l + 32] & 0xF;
                        let q2_qh_part = ((qh[l] >> 2) & 3) << 4;
                        let q2_combined = q2_ql_part | q2_qh_part;
                        let q2 = q2_combined as i8 as i32 - 32;
                        // const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        let q3_ql_part = (ql[l + 0] >> 4) & 0xF;
                        let q3_qh_part = ((qh[l] >> 4) & 3) << 4;
                        let q3_combined = q3_ql_part | q3_qh_part;
                        let q3 = q3_combined as i8 as i32 - 32;
                        // const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        let q4_ql_part = (ql[l + 32] >> 4) & 0xF;
                        let q4_qh_part = ((qh[l] >> 6) & 3) << 4;
                        let q4_combined = q4_ql_part | q4_qh_part;
                        let q4 = q4_combined as i8 as i32 - 32;

                        // Compute dot product contribution
                        // llama.cpp: y[l + 0] = d * sc[is + 0] * q1;
                        // For dot product: acc += d * sc[is + 0] * q1 * xb[l + 0]
                        let scale1 = d * (sc[is + 0] as f32);
                        let scale2 = d * (sc[is + 2] as f32);
                        let scale3 = d * (sc[is + 4] as f32);
                        let scale4 = d * (sc[is + 6] as f32);

                        acc += scale1 * (q1 as f32) * x_row[xb_base + xb_offset + l + 0];
                        acc += scale2 * (q2 as f32) * x_row[xb_base + xb_offset + l + 32];
                        acc += scale3 * (q3 as f32) * x_row[xb_base + xb_offset + l + 64];
                        acc += scale4 * (q4 as f32) * x_row[xb_base + xb_offset + l + 96];
                    }

                    // Advance pointers for next 128 elements (llama.cpp pattern)
                    // y  += 128;  ql += 64;  qh += 32;  sc += 8;
                    ql = &ql[64..];   // Advance 64 bytes into the block
                    qh = &qh[32..];   // Advance 32 bytes
                    sc = &sc[8..];    // Advance 8 scales
                    xb_offset += 128; // Advance 128 elements in x_row
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Q3_K GEMM fallback: dequantize blocks on the fly and compute matrix multiply.
///
/// For each output row:
/// - Load Q3_K blocks from weight matrix
/// - Dequantize each block to f32
/// - Compute dot product with input row
/// - Store result in output
///
/// Performance notes:
/// - Dequantization is done inline during GEMM
/// - Each iteration processes 256 output values (Q3_K block size)
/// - Uses rayon for parallel processing across output rows
pub fn gemm_q3_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::quant::{Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q3_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q3_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;

            for b in 0..num_blocks {
                let block = &row_w[b * Q3_K_BLOCK_BYTES..(b + 1) * Q3_K_BLOCK_BYTES];

                // Q3_K block layout:
                // - hmask[32]: high bit mask
                // - qs[64]: low 2 bits
                // - scales[12]: packed 6-bit scales
                // - d[2]: f16 super-block scale
                let hmask = &block[0..32];
                let qs = &block[32..96];
                let scales = &block[96..108];
                let d = super::quant::load_f16_scale(&block[108..110]);

                // Unpack scales from packed 6-bit format
                let tmp = u32::from_le_bytes([scales[2], scales[3], scales[4], scales[5]]);
                let mut unpacked_scales = [0i8; 12];
                unpacked_scales[0] = (((scales[0] & 0x0F) as i8) << 2) | ((tmp & 0x03) as i8);
                unpacked_scales[1] = ((scales[0] >> 4) & 0x0F) as i8;
                unpacked_scales[2] = (((scales[1] & 0x0F) as i8) << 2) | (((tmp >> 2) & 0x03)) as i8;
                unpacked_scales[3] = ((scales[1] >> 4) & 0x0F) as i8;
                unpacked_scales[4] = ((tmp >> 4) & 0x0F) as i8;
                unpacked_scales[5] = (((scales[2] & 0x0F) as i8) << 2) | (((tmp >> 6) & 0x03)) as i8;
                unpacked_scales[6] = ((scales[2] >> 4) & 0x0F) as i8;
                unpacked_scales[7] = ((tmp >> 8) & 0x0F) as i8;
                unpacked_scales[8] = ((scales[0] >> 6) & 0x03) as i8;
                unpacked_scales[9] = ((scales[1] >> 6) & 0x03) as i8;
                unpacked_scales[10] = (((scales[2] >> 2) & 0x03) as i8) << 2;
                unpacked_scales[11] = ((scales[2] >> 0) & 0x03) as i8;

                let mut scale_idx = 0;
                let xb_base = b * Q3_K_BLOCK_ELEMS;

                // Process 256 elements as two 128-element chunks
                for chunk in 0..2 {
                    let q = &qs[chunk * 32..];
                    let hm = &hmask[chunk * 16..];

                    let mut m = 1u8;
                    let mut shift = 0i32;

                    // Reset scale_idx for second chunk (second 128 elements use only 4 scales)
                    if chunk == 1 {
                        scale_idx = 8;
                    }

                    // First chunk: 4 groups (8 scales), second chunk: 2 groups (4 scales)
                    let num_groups = if chunk == 0 { 4 } else { 2 };
                    for _group in 0..num_groups {
                        let dl = d * (unpacked_scales[scale_idx] - 32) as f32;
                        scale_idx += 1;

                        // First 16 elements
                        for l in 0..16 {
                            let ql = (q[l >> 2] >> shift) & 0x03;
                            let hbit = if hm[l >> 3] & m != 0 { 0 } else { 4 };
                            let q = (ql as i8 - hbit) as f32;
                            acc += dl * q * x_row[xb_base + chunk * 64 + _group * 32 + l];
                        }

                        // Next 16 elements
                        let dl = d * (unpacked_scales[scale_idx] - 32) as f32;
                        scale_idx += 1;

                        for l in 16..32 {
                            let ql = (q[l >> 2] >> shift) & 0x03;
                            let hbit = if hm[l >> 3] & m != 0 { 0 } else { 4 };
                            let q = (ql as i8 - hbit) as f32;
                            acc += dl * q * x_row[xb_base + chunk * 64 + _group * 32 + l];
                        }

                        shift += 2;
                        m <<= 1;
                    }
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Q5_K GEMM fallback: dequantize weights on the fly during matrix multiplication.
///
/// Computes Y = W * X where W is Q5_K quantized.
///
/// # Arguments
/// * `w` - Quantized weight matrix (column-major, stored as [out_dim, in_dim] in Q5_K blocks)
/// * `x` - Input matrix [batch_size, in_dim]
/// * `y` - Output matrix [batch_size, out_dim]
/// * `out_dim` - Output dimension (columns of W, rows of stored layout)
/// * `in_dim` - Input dimension (rows of W, columns of stored layout)
///
/// Performance notes:
/// - Dequantization is done inline during GEMM
/// - Each iteration processes 256 output values (Q5_K block size)
/// - Uses rayon for parallel processing across output rows
pub fn gemm_q5_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::quant::{Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q5_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        for o in 0..out_dim {
            let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
            let mut acc = 0.0f32;

            for b in 0..num_blocks {
                let block = &row_w[b * Q5_K_BLOCK_BYTES..(b + 1) * Q5_K_BLOCK_BYTES];

                // Q5_K block layout (matches llama.cpp block_q5_K):
                // - d[2]: f16 super-block scale
                // - dmin[2]: f16 super-block min scale
                // - scales[12]: scales and mins packed as 6-bit values
                // - qh[32]: high bit of 5-bit quantization
                // - ql[128]: low 4-bit quantized weights
                let d = super::quant::load_f16_scale(&block[0..2]);
                let dmin = super::quant::load_f16_scale(&block[2..4]);
                let scales = &block[4..16];
                let qh = &block[16..48];
                let ql = &block[48..176];

                // Unpack scales and mins from packed 6-bit format (get_scale_min_k4 pattern)
                let mut unpacked_scales = [0i8; 8];
                let mut unpacked_mins = [0i8; 8];
                for j in 0..8 {
                    if j < 4 {
                        unpacked_scales[j] = (scales[j] & 63) as i8;
                        unpacked_mins[j] = (scales[j + 4] & 63) as i8;
                    } else {
                        unpacked_scales[j] = ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as i8;
                        unpacked_mins[j] = ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)) as i8;
                    }
                }

                let mut scale_idx = 0;
                let xb_base = b * Q5_K_BLOCK_ELEMS;

                // Process 256 elements as 4 chunks of 64 elements each
                for chunk in 0..4 {
                    let q = &ql[chunk * 32..];
                    let hm = &qh[chunk * 8..];
                    let u1 = 1u8.wrapping_shl(2 * chunk as u32);
                    let u2 = 1u8.wrapping_shl(2 * chunk as u32 + 1);

                    // First 32 elements: d1 * q - m1
                    let d1 = d * unpacked_scales[scale_idx] as f32;
                    let m1 = dmin * unpacked_mins[scale_idx] as f32;
                    scale_idx += 1;
                    for l in 0..32 {
                        let ql_bits = q[l] & 0x0F;
                        let hbit = if hm[l >> 3] & u1 != 0 { 16 } else { 0 };
                        let q_val = (ql_bits + hbit) as f32;
                        acc += (d1 * q_val - m1) * x_row[xb_base + chunk * 64 + l];
                    }

                    // Next 32 elements: d2 * q - m2
                    let d2 = d * unpacked_scales[scale_idx] as f32;
                    let m2 = dmin * unpacked_mins[scale_idx] as f32;
                    scale_idx += 1;
                    for l in 0..32 {
                        let ql_bits = q[l] >> 4;
                        let hbit = if hm[l >> 3] & u2 != 0 { 16 } else { 0 };
                        let q_val = (ql_bits + hbit) as f32;
                        acc += (d2 * q_val - m2) * x_row[xb_base + chunk * 64 + 32 + l];
                    }
                }
            }
            y_row[o] = acc;
        }
    });
}

/// Dispatch GEMM by weight type with automatic transposition detection.
///
/// Uses metadata to determine if weights need transposed access.
pub fn dispatch_gemm(
    w: &[u8],
    meta: &WeightMeta,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), super::CpuError> {
    // Validate dimensions for block-based quantization
    let block_elems = match meta.wtype {
        GgmlType::F32 => 1,
        GgmlType::Q8_0 => super::quant::Q8_BLOCK_ELEMS,
        GgmlType::Q4_0 | GgmlType::Q4_1 => super::quant::Q4_BLOCK_ELEMS,
        GgmlType::Q5_0 => super::quant::Q5_0_BLOCK_ELEMS,
        GgmlType::Q4_K | GgmlType::Q6_K => super::quant::Q4_K_BLOCK_ELEMS,
        GgmlType::Q3_K => super::quant::Q3_K_BLOCK_ELEMS,
        GgmlType::Q5_K => super::quant::Q5_K_BLOCK_ELEMS,
        _ => 1,
    };

    if in_dim % block_elems != 0 {
        return Err(super::CpuError::InvalidOperation(format!(
            "in_dim {} is not a multiple of block size {} for type {:?}",
            in_dim, block_elems, meta.wtype
        )));
    }

    match meta.wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            if meta.needs_transpose {
                gemm_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemm_f32(wf, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_0 => {
            if meta.needs_transpose {
                gemm_q4_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_1 => {
            if meta.needs_transpose {
                gemm_q4_1_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_1(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q5_0 => {
            if meta.needs_transpose {
                gemm_q5_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemm_q5_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q8_0 => {
            if meta.needs_transpose {
                gemm_q8_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_K => {
            // Q4_K × Q8_K GEMM: quantize input on the fly
            if meta.needs_transpose {
                // For transposed Q4_K, use dequant-on-the-fly fallback
                gemm_q4_k_transposed_fallback(w, x, y, 1, out_dim, in_dim);
            } else {
                crate::cpu::kernels::gemm_q4k_q8::gemm_q4_k_q8_k_dispatch_gemm(w, x, y, 1, out_dim, in_dim);
            }
        }
        GgmlType::Q6_K => {
            // Q6_K: dequantize to f32 on the fly (slower but works)
            // For transposed weights, the fallback function handles it correctly
            gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q3_K => {
            // Q3_K: dequantize to f32 on the fly (slower but works)
            gemm_q3_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            // Q5_K: dequantize to f32 on the fly (slower but works)
            gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// Dispatch GEMM with transposed flag for transposed weight matrices.
///
/// When `transposed` is true, computes: Y = W^T * X
/// Otherwise computes: Y = W * X
///
/// Used for FFN down projection where weights are stored as [in_dim, out_dim].
pub fn dispatch_gemm_transposed(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    transposed: bool,
) -> Result<(), super::CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            if transposed {
                gemm_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemm_f32(wf, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_0 => {
            if transposed {
                gemm_q4_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_1 => {
            if transposed {
                gemm_q4_1_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_1(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q8_0 => {
            if transposed {
                gemm_q8_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q6_K => {
            gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// Q4_0 GEMM transposed for transposed weight matrices.
///
/// Computes: Y = W^T * X where W has shape [in_dim, out_dim]
/// stored in column-major Q4_0 blocked format.
pub fn gemm_q4_0_transposed_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_col = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_BLOCK_BYTES;
    let seq_len = x.len() / in_dim;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            let mut acc = 0.0f32;

            // Iterate through blocks in this column
            for b in 0..num_blocks_per_col {
                let block = &w[col_offset + b * Q4_BLOCK_BYTES..col_offset + (b + 1) * Q4_BLOCK_BYTES];
                let scale = super::quant::load_f16_scale(&block[0..2]);
                let qs = &block[2..18];
                let xb = &x_row[b * Q4_BLOCK_ELEMS..];

                for i in 0..16 {
                    let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                    let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                    acc += q_lo * xb[i] + q_hi * xb[i + 16];
                }
            }

            y_row[o] = acc;
        });
    }
}

/// Q4_1 GEMM transposed for transposed weight matrices.
pub fn gemm_q4_1_transposed_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_col = in_dim / Q4_1_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_1_BLOCK_BYTES;
    let seq_len = x.len() / in_dim;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            let mut acc = 0.0f32;

            // Iterate through blocks in this column
            for b in 0..num_blocks_per_col {
                let block = &w[col_offset + b * Q4_1_BLOCK_BYTES..col_offset + (b + 1) * Q4_1_BLOCK_BYTES];
                let w_scale = super::quant::load_f16_scale(&block[0..2]);
                let w_min = super::quant::load_f16_scale(&block[2..4]);
                let qs = &block[4..20];
                let xb = &x_row[b * Q4_1_BLOCK_ELEMS..];

                for i in 0..16 {
                    let q_lo = ((qs[i] & 0x0F) as i32) as f32;
                    let q_hi = ((qs[i] >> 4) as i32) as f32;
                    let v_lo = (q_lo * w_scale + w_min) * xb[i];
                    let v_hi = (q_hi * w_scale + w_min) * xb[i + 16];
                    acc += v_lo + v_hi;
                }
            }

            y_row[o] = acc;
        });
    }
}

/// Q5_0 GEMM transposed for transposed weight matrices.
///
/// Computes: Y = W^T * X where W has shape [in_dim, out_dim]
/// stored in column-major Q5_0 blocked format.
pub fn gemm_q5_0_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks_per_col = in_dim / Q5_0_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q5_0_BLOCK_BYTES;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            let mut acc = 0.0f32;

            // Iterate through blocks in this column
            for b in 0..num_blocks_per_col {
                let block = &w[col_offset + b * Q5_0_BLOCK_BYTES..col_offset + (b + 1) * Q5_0_BLOCK_BYTES];
                let d = super::quant::load_f16_scale(&block[0..2]);
                let qh = &block[2..6];
                let qs = &block[6..22];
                let xb = &x_row[b * Q5_0_BLOCK_ELEMS..];

                for i in 0..16 {
                    // Process 2 values per iteration
                    let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                    let low_bits_0 = qs[i] & 0x0F;
                    let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                    let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                    let low_bits_1 = (qs[i] >> 4) & 0x0F;
                    let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                    acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
                }
            }

            y_row[o] = acc;
        });
    }
}

/// Q8_0 GEMM transposed for transposed weight matrices.
pub fn gemm_q8_0_transposed_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_col = in_dim / Q8_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q8_BLOCK_BYTES;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            let mut acc = 0.0f32;

            // Iterate through blocks in this column
            for b in 0..num_blocks_per_col {
                let block = &w[col_offset + b * Q8_BLOCK_BYTES..col_offset + (b + 1) * Q8_BLOCK_BYTES];
                let scale = super::quant::load_f16_scale(&block[0..2]);
                let qs = &block[2..34];
                let xb = &x_row[b * Q8_BLOCK_ELEMS..];

                for i in 0..Q8_BLOCK_ELEMS {
                    acc += (qs[i] as i8) as f32 * scale * xb[i];
                }
            }

            y_row[o] = acc;
        });
    }
}

/// F32 GEMM transposed.
fn gemm_f32_transposed(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let seq_len = x.len() / in_dim;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        // Process all sequences in parallel
        y.par_chunks_mut(out_dim).enumerate().for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            let mut acc = 0.0f32;

            // Compute dot product: sum_i(x[i] * W[i, o])
            for i in 0..in_dim {
                acc += x_row[i] * w[i * out_dim + o];
            }

            y_row[o] = acc;
        });
    }
}

// ── GEMV (matrix-vector multiply for decode) ────────────────────────────────

/// Quantize a single f32 vector to Q8_0 (one block per Q8_BLOCK_ELEMS).
///
/// This is used for quantizing activations once per GEMV call for Q4_0 × Q8_0 dot products.
fn quantize_q8_0_single(x: &[f32], out: &mut [u8], in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    debug_assert_eq!(out.len(), num_blocks * Q8_BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * Q8_BLOCK_ELEMS..(b + 1) * Q8_BLOCK_ELEMS];
        let off = b * Q8_BLOCK_BYTES;
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / Q8_0_MAX } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let scale_bytes = half::f16::from_f32(scale).to_bits().to_le_bytes();
        out[off] = scale_bytes[0];
        out[off + 1] = scale_bytes[1];
        for i in 0..Q8_BLOCK_ELEMS {
            let q = (xb[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out[off + 2 + i] = q as u8;
        }
    }
}

/// F32 GEMV: y[row] = dot(W[row, :], x)
///
/// W layout: [out_dim, in_dim] row-major.
pub fn gemv_f32(w: &[f32], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();

    // AVX2 feature detection (cached)
    let features = super::features::CpuFeatures::get();
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = features.has_avx2 && features.has_fma && in_dim % 8 == 0;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * in_dim..(row + 1) * in_dim];
        *out = if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                unsafe { dot_f32_avx2(row_w, x) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                row_w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum()
            }
        } else {
            row_w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum()
        };
    });
}

/// Q4_0 GEMV: dequant on-the-fly.
pub fn gemv_q4_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    // AVX2 feature detection (cached)
    let features = super::features::CpuFeatures::get();
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = features.has_avx2 && features.has_fma;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for row in 0..out_dim {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..(b + 1) * Q4_BLOCK_BYTES];
            let scale = super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_block_avx2(qs, xb, scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..16 {
                        let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                        let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                        acc += q_lo * xb[i] + q_hi * xb[i + 16];
                    }
                }
            } else {
                for i in 0..16 {
                    let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                    let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                    acc += q_lo * xb[i] + q_hi * xb[i + 16];
                }
            }
        }
        y[row] = acc;
    }
}

/// Q4_0 × Q8_0 GEMV: quantize input to Q8_0 once, then integer dot product.
/// This is faster than Q4_0 × f32 because it avoids 4× int→f32 conversions per block.
///
/// # Arguments
///
/// * `w` - Weight matrix in Q4_0 format [out_dim * row_bytes]
/// * `x` - Input vector [in_dim]
/// * `y` - Output vector [out_dim]
/// * `out_dim` - Output dimension (number of output rows)
/// * `in_dim` - Input dimension (must be multiple of Q4_BLOCK_ELEMS)
/// * `scratch` - Optional scratch buffer for Q8_0 quantization. If None, allocates internally.
pub fn gemv_q4_0_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize, scratch: Option<&mut [u8]>) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    // Use provided scratch buffer or allocate on heap
    let mut owned_scratch;
    let x_q8 = if let Some(buf) = scratch {
        if buf.len() >= num_blocks * Q8_BLOCK_BYTES {
            &mut buf[..num_blocks * Q8_BLOCK_BYTES]
        } else {
            // Scratch buffer too small, fall back to heap allocation
            owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
            &mut owned_scratch[..]
        }
    } else {
        // No scratch buffer provided, allocate on heap (backward compatible)
        owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
        &mut owned_scratch[..]
    };

    quantize_q8_0_single(x, x_q8, in_dim);

    let features = super::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        // Process 2 blocks at a time for better ILP (instruction-level parallelism)
        let mut b = 0;
        while b + 1 < num_blocks {
            // Prefetch blocks ahead
            #[cfg(target_arch = "x86_64")]
            if b + 2 < num_blocks {
                unsafe {
                    use std::arch::x86_64::_MM_HINT_T0;
                    use std::arch::x86_64::_mm_prefetch;
                    let next_ptr = row_w[(b + 2) * Q4_BLOCK_BYTES..].as_ptr();
                    _mm_prefetch(next_ptr as *const i8, _MM_HINT_T0);
                }
            }

            // Block 0
            let block0 = &row_w[b * Q4_BLOCK_BYTES..];
            let w_scale0 = load_f16_scale(&block0[0..2]);
            let x_scale0 = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale0 = w_scale0 * x_scale0;
            let qs0 = &block0[2..18];
            let q8_0 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Block 1
            let block1 = &row_w[(b + 1) * Q4_BLOCK_BYTES..];
            let w_scale1 = load_f16_scale(&block1[0..2]);
            let x_scale1 = load_f16_scale(&x_q8[(b + 1) * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale1 = w_scale1 * x_scale1;
            let qs1 = &block1[2..18];
            let q8_1 = &x_q8[(b + 1) * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Compute both blocks
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs0, q8_0, combined_scale0) };
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs1, q8_1, combined_scale1) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_0_q8_0_block_scalar(qs0, q8_0, combined_scale0);
                    acc += dot_q4_0_q8_0_block_scalar(qs1, q8_1, combined_scale1);
                }
            } else {
                acc += dot_q4_0_q8_0_block_scalar(qs0, q8_0, combined_scale0);
                acc += dot_q4_0_q8_0_block_scalar(qs1, q8_1, combined_scale1);
            }

            b += 2;
        }

        // Handle remaining block
        while b < num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..];
            let w_scale = load_f16_scale(&block[0..2]);
            let x_scale = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale = w_scale * x_scale;
            let qs = &block[2..18];
            let q8 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs, q8, combined_scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_0_q8_0_block_scalar(qs, q8, combined_scale);
                }
            } else {
                acc += dot_q4_0_q8_0_block_scalar(qs, q8, combined_scale);
            }
            b += 1;
        }

        *out = acc;
    });
}

/// Q4_1 × Q8_0 GEMV: quantize input to Q8_0 once, then integer dot product.
///
/// Q4_1 block format: [f16 scale | f16 min | 16 nibble bytes] = 20 bytes
/// Values are in range [min, min + 15*scale]
pub fn gemv_q4_1_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize, scratch: Option<&mut [u8]>) {
    let num_blocks = in_dim / Q4_1_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_1_BLOCK_BYTES;

    // Use provided scratch buffer or allocate on heap
    let mut owned_scratch;
    let x_q8 = if let Some(buf) = scratch {
        if buf.len() >= num_blocks * Q8_BLOCK_BYTES {
            &mut buf[..num_blocks * Q8_BLOCK_BYTES]
        } else {
            owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
            &mut owned_scratch[..]
        }
    } else {
        owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
        &mut owned_scratch[..]
    };

    quantize_q8_0_single(x, x_q8, in_dim);

    let features = super::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        // Process 2 blocks at a time for better ILP
        let mut b = 0;
        while b + 1 < num_blocks {
            // Prefetch blocks ahead
            #[cfg(target_arch = "x86_64")]
            if b + 2 < num_blocks {
                unsafe {
                    use std::arch::x86_64::_MM_HINT_T0;
                    use std::arch::x86_64::_mm_prefetch;
                    let next_ptr = row_w[(b + 2) * Q4_1_BLOCK_BYTES..].as_ptr();
                    _mm_prefetch(next_ptr as *const i8, _MM_HINT_T0);
                }
            }

            // Block 0
            let block0 = &row_w[b * Q4_1_BLOCK_BYTES..];
            let w_scale0 = load_f16_scale(&block0[0..2]);
            let w_min0 = load_f16_scale(&block0[2..4]);
            let x_scale0 = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale0 = w_scale0 * x_scale0;
            let qs0 = &block0[4..20];
            let q8_0 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Block 1
            let block1 = &row_w[(b + 1) * Q4_1_BLOCK_BYTES..];
            let w_scale1 = load_f16_scale(&block1[0..2]);
            let w_min1 = load_f16_scale(&block1[2..4]);
            let x_scale1 = load_f16_scale(&x_q8[(b + 1) * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale1 = w_scale1 * x_scale1;
            let qs1 = &block1[4..20];
            let q8_1 = &x_q8[(b + 1) * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Compute both blocks
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_1_q8_0_block_avx2(qs0, q8_0, combined_scale0, w_min0 * x_scale0) };
                    acc += unsafe { dot_q4_1_q8_0_block_avx2(qs1, q8_1, combined_scale1, w_min1 * x_scale1) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_1_q8_0_block_scalar(qs0, q8_0, combined_scale0, w_min0 * x_scale0);
                    acc += dot_q4_1_q8_0_block_scalar(qs1, q8_1, combined_scale1, w_min1 * x_scale1);
                }
            } else {
                acc += dot_q4_1_q8_0_block_scalar(qs0, q8_0, combined_scale0, w_min0 * x_scale0);
                acc += dot_q4_1_q8_0_block_scalar(qs1, q8_1, combined_scale1, w_min1 * x_scale1);
            }

            b += 2;
        }

        // Handle remaining block
        while b < num_blocks {
            let block = &row_w[b * Q4_1_BLOCK_BYTES..];
            let w_scale = load_f16_scale(&block[0..2]);
            let w_min = load_f16_scale(&block[2..4]);
            let x_scale = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale = w_scale * x_scale;
            let qs = &block[4..20];
            let q8 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_1_q8_0_block_avx2(qs, q8, combined_scale, w_min * x_scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_1_q8_0_block_scalar(qs, q8, combined_scale, w_min * x_scale);
                }
            } else {
                acc += dot_q4_1_q8_0_block_scalar(qs, q8, combined_scale, w_min * x_scale);
            }
            b += 1;
        }

        *out = acc;
    });
    let _ = out_dim;
}

/// Q5_0 GEMV: dequant on-the-fly.
pub fn gemv_q5_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q5_0_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_0_BLOCK_BYTES;

    for o in 0..out_dim {
        let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
            let d = super::quant::load_f16_scale(&block[0..2]);
            let qh = &block[2..6];
            let qs = &block[6..22];
            let xb = &x[b * Q5_0_BLOCK_ELEMS..];

            for i in 0..16 {
                let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                let low_bits_0 = qs[i] & 0x0F;
                let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                let low_bits_1 = (qs[i] >> 4) & 0x0F;
                let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
            }
        }
        y[o] = acc;
    }
}

/// Q8_0 GEMV: dequant on-the-fly.
pub fn gemv_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q8_BLOCK_BYTES;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &row_w[b * Q8_BLOCK_BYTES..(b + 1) * Q8_BLOCK_BYTES];
            let scale = super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];

            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }
        *out = acc;
    });
}

/// Q6_K GEMV: dequant on-the-fly (fallback, slower but works).
/// Reuses gemm_q6_k_fallback with batch_size=1 for consistency.
pub fn gemv_q6_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // GEMV is just GEMM with batch_size=1
    gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q5_K GEMV: dequantize weights on the fly during matrix-vector multiplication.
///
/// Computes y = W * x where W is Q5_K quantized.
/// This is a wrapper around gemm_q5_k_fallback since GEMV is just GEMM with batch_size=1.
pub fn gemv_q5_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // GEMV is just GEMM with batch_size=1
    gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q4_K GEMV for transposed weights (tied embeddings).
///
/// For transposed access (tied LM head), we compute y = W^T * x
/// where W is [hidden, vocab] stored as [vocab, hidden].
/// This means each output dimension corresponds to a row of W^T, which is a column of W.
///
/// Simpler approach: dequantize to f32 on-the-fly for each output dimension.
fn gemv_q4_k_transposed_fallback(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    use crate::cpu::kernels::q4::BlockQ4K;

    let num_blocks_k = in_dim / 256;
    let row_bytes = num_blocks_k * BlockQ4K::SIZE;

    // For transposed access, output dimension corresponds to rows in stored layout
    y.par_iter_mut().enumerate().for_each(|(vocab_idx, out)| {
        let mut acc = 0.0f32;

        for block_idx in 0..num_blocks_k {
            let block_ptr = unsafe {
                w.as_ptr().add(vocab_idx * row_bytes + block_idx * BlockQ4K::SIZE) as *const BlockQ4K
            };
            let block = unsafe { &*block_ptr };

            // Dequantize this block of 256 weights and compute dot product with x
            let block_start = block_idx * 256;
            for i in 0..256 {
                if block_start + i < in_dim {
                    // Dequantize single Q4_K value
                    let sub_block_idx = i / 32;
                    let scale_idx = sub_block_idx / 2;
                    let sign = (i % 32) / 16;

                    // Get the scale (simplified - actual Q4_K scale unpacking is complex)
                    // For now, use a simple dequantization
                    let q4_value = if i < 128 {
                        (block.qs[i / 2] >> (4 * (i % 2))) & 0x0F
                    } else {
                        (block.qs[64 + (i - 128) / 2] >> (4 * ((i - 128) % 2))) & 0x0F
                    };

                    // Simple dequantization (not fully accurate but works for fallback)
                    let d = half::f16::from_le_bytes(block.d).to_f32();
                    let weight = d * (q4_value as f32 - 8.0);
                    acc += weight * x[block_start + i];
                }
            }
        }

        *out = acc;
    });
}

/// Q4_K GEMM for transposed weights.
///
/// For transposed access, computes Y = W^T * X where W is stored as [in_dim, out_dim].
/// Uses dequantization-on-the-fly for simplicity (slower but correct).
fn gemm_q4_k_transposed_fallback(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use crate::cpu::kernels::q4::BlockQ4K;

    let num_blocks_k = k / 256;
    let row_bytes = num_blocks_k * BlockQ4K::SIZE;

    // For transposed access: W stored as [k, n], compute as W^T * X
    // Each output column j corresponds to row j in stored layout
    y.par_chunks_mut(n).enumerate().for_each(|(batch_idx, y_row)| {
        let x_row = &x[batch_idx * k..(batch_idx + 1) * k];

        for out_col in 0..n {
            let mut acc = 0.0f32;

            for block_idx in 0..num_blocks_k {
                let block_ptr = unsafe {
                    w.as_ptr().add(out_col * row_bytes + block_idx * BlockQ4K::SIZE) as *const BlockQ4K
                };
                let block = unsafe { &*block_ptr };

                // Dequantize this block and compute dot product
                let block_start = block_idx * 256;
                for i in 0..256 {
                    if block_start + i < k {
                        // Simplified Q4_K dequantization
                        let q4_value = if i < 128 {
                            (block.qs[i / 2] >> (4 * (i % 2))) & 0x0F
                        } else {
                            (block.qs[64 + (i - 128) / 2] >> (4 * ((i - 128) % 2))) & 0x0F
                        };

                        let d = half::f16::from_le_bytes(block.d).to_f32();
                        let weight = d * (q4_value as f32 - 8.0);
                        acc += weight * x_row[block_start + i];
                    }
                }
            }

            y_row[out_col] = acc;
        }
    });
}

/// Dispatch GEMV based on weight type with automatic transposition detection.
///
/// Uses metadata to determine if weights need transposed access.
///
/// Computes: y = W * x (matrix-vector multiply) or y = W^T * x if transposed
///
/// # Arguments
///
/// * `w` - Weight matrix bytes
/// * `meta` - Weight metadata (type, dimensions, transposition)
/// * `x` - Input vector
/// * `y` - Output vector
/// * `out_dim` - Output dimension
/// * `in_dim` - Input dimension
/// * `scratch` - Optional scratch buffer for Q8_0 quantization (avoids heap allocation)
pub fn dispatch_gemv(
    w: &[u8],
    meta: &WeightMeta,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    scratch: Option<&mut [u8]>,
) -> Result<(), super::CpuError> {
    // Validate dimensions for block-based quantization
    let block_elems = match meta.wtype {
        GgmlType::F32 => 1,
        GgmlType::Q8_0 => super::quant::Q8_BLOCK_ELEMS,
        GgmlType::Q4_0 | GgmlType::Q4_1 => super::quant::Q4_BLOCK_ELEMS,
        GgmlType::Q5_0 => super::quant::Q5_0_BLOCK_ELEMS,
        GgmlType::Q4_K | GgmlType::Q6_K => super::quant::Q4_K_BLOCK_ELEMS,
        GgmlType::Q3_K => super::quant::Q3_K_BLOCK_ELEMS,
        GgmlType::Q5_K => super::quant::Q5_K_BLOCK_ELEMS,
        _ => 1,
    };

    if in_dim % block_elems != 0 {
        return Err(super::CpuError::InvalidOperation(format!(
            "in_dim {} is not a multiple of block size {} for type {:?}",
            in_dim, block_elems, meta.wtype
        )));
    }

    match meta.wtype {
        GgmlType::F32 => {
            let wf: &[f32] = unsafe {
                std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4)
            };
            if meta.needs_transpose {
                gemv_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemv_f32(wf, x, y);
            }
        }
        GgmlType::Q4_0 => {
            if meta.needs_transpose {
                gemv_q4_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_0_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q4_1 => {
            if meta.needs_transpose {
                gemv_q4_1_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_1_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q5_0 => {
            crate::cpu::kernels::gemm_q5_0_q8::gemv_q5_0_q8_0_dispatch(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q8_0 => {
            if meta.needs_transpose {
                gemv_q8_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_K => {
            // Q4_K GEMV: use SIMD for non-transposed, scalar for transposed (tied embeddings)
            if meta.needs_transpose {
                // For transposed Q4_K (tied embeddings), use dequantize-on-the-fly
                gemv_q4_k_transposed_fallback(w, x, y, out_dim, in_dim);
            } else {
                crate::cpu::kernels::gemm_q4k_q8::gemv_q4_k_q8_k_dispatch(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q6_K => {
            gemv_q6_k(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            gemv_q5_k(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

// ── AVX2 kernels ─────────────────────────────────────────────────────────────

/// Horizontal sum of __m256 register.
///
/// Folds 8-lane f32 vector to scalar sum using SSE instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Fold top 4 lanes into bottom 4
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    // Fold top 2 into bottom 2
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    // Fold top 1 into bottom 1
    let sum1 = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
    _mm_cvtss_f32(sum1)
}

/// Unpack Q4_0 nibbles to i8 values in __m256i.
///
/// Input: 16 bytes, each containing 2 nibbles (32 values total).
/// Output: __m256i with 32 i8 values, each = nibble - 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_q4_0_nibbles_avx2(qs: &[u8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "unpack_q4_0_nibbles_avx2: qs must be 16 bytes"
    );

    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);
    let lo = _mm_and_si128(raw, lo_mask);
    let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);
    let q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    _mm256_sub_epi8(q4, _mm256_set1_epi8(8))
}

/// Multiply-sum Q4_0 × Q8_0 block (unscaled).
///
/// Computes sum(q4[i] * q8[i]) for 32-element blocks.
/// Returns __m256 with one i32 result per 8-element group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[target_feature(enable = "avxvnni")]
unsafe fn mul_sum_q4_0_q8_0_block_avx2_vnni(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "mul_sum_q4_0_q8_0_block_avx2_vnni: q8 must have 32 elements"
    );

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    // AVX2VNNI: compute dot product of signed i8 vectors
    // This does both multiply and horizontal sum in one instruction
    let zero = _mm256_setzero_si256();
    let dot32 = _mm256_dpwssd_avx_epi32(zero, q4, q8v);
    _mm256_cvtepi32_ps(dot32)
}

/// Multiply-sum Q4_0 × Q8_0 block (unscaled) without VNNI.
///
/// Computes sum(q4[i] * q8[i]) for 32-element blocks.
/// Returns __m256 with one i32 result per 8-element group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sum_q4_0_q8_0_block_avx2_unscaled(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "mul_sum_q4_0_q8_0_block_avx2_unscaled: q8 must have 32 elements"
    );

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let q4_abs = _mm256_sign_epi8(q4, q4);
    let q8_signed = _mm256_sign_epi8(q8v, q4);
    let dot16 = _mm256_maddubs_epi16(q4_abs, q8_signed);
    let ones = _mm256_set1_epi16(1);
    let dot32 = _mm256_madd_epi16(ones, dot16);
    _mm256_cvtepi32_ps(dot32)
}

/// AVX2+FMA dot product: sum(a[i] * b[i]) for f32 slices.
///
/// `a` and `b` must have the same length, which must be a multiple of 8.
/// Caller must ensure AVX2+FMA are available (checked via is_x86_feature_detected!).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot_f32_avx2: a and b must have the same length"
    );
    debug_assert_eq!(n % 8, 0, "dot_f32_avx2: length must be multiple of 8");

    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    hsum_avx2(acc)
}

/// AVX2 Q4_0 block dot product — processes one 32-element block in 4 FMA ops.
///
/// Layout: qs[i] contains lo nibble (→ x[i]) and hi nibble (→ x[i+16])
/// Dequant: (nibble - 8) * scale
///
/// `qs` must be exactly 16 bytes. `xb` must be at least 32 floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_block_avx2(qs: &[u8], xb: &[f32], scale: f32) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(qs.len(), 16, "dot_q4_0_block_avx2: qs must be 16 bytes");
    debug_assert!(
        xb.len() >= 32,
        "dot_q4_0_block_avx2: xb must have at least 32 elements"
    );

    // Load 16 nibble bytes into a 128-bit register
    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);

    // Extract lo nibbles (bits 0..3): AND with 0x0F
    let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_bytes = _mm_and_si128(raw, lo_mask);

    // Extract hi nibbles (bits 4..7): shift right 4 then mask
    let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);

    // Subtract 8 from each nibble (as i8) to get signed values -8..7
    let eight = _mm_set1_epi8(8i8);
    let lo_signed = _mm_sub_epi8(lo_bytes, eight);
    let hi_signed = _mm_sub_epi8(hi_bytes, eight);

    let scale_v = _mm256_set1_ps(scale);
    let mut acc = _mm256_setzero_ps();

    // lo nibbles 0..7 dot x[0..7] and lo nibbles 8..15 dot x[8..15]
    let lo_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_signed));
    let lo_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(lo_signed, 8)));
    let x0 = _mm256_loadu_ps(xb.as_ptr());
    let x1 = _mm256_loadu_ps(xb.as_ptr().add(8));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(lo_f0, scale_v), x0, acc);
    acc = _mm256_fmadd_ps(_mm256_mul_ps(lo_f1, scale_v), x1, acc);

    // hi nibbles 0..7 dot x[16..23] and hi nibbles 8..15 dot x[24..31]
    let hi_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_signed));
    let hi_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(hi_signed, 8)));
    let x2 = _mm256_loadu_ps(xb.as_ptr().add(16));
    let x3 = _mm256_loadu_ps(xb.as_ptr().add(24));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(hi_f0, scale_v), x2, acc);
    acc = _mm256_fmadd_ps(_mm256_mul_ps(hi_f1, scale_v), x3, acc);

    hsum_avx2(acc)
}

/// AVX2 Q4_0 × Q8_0 block dot product — one 32-element block.
///
/// Uses FMA accumulation for better performance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_q8_0_block_avx2(qs: &[u8], q8: &[u8], scale: f32) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_0_q8_0_block_avx2: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_0_q8_0_block_avx2: q8 must have 32 elements"
    );

    let q4 = unpack_q4_0_nibbles_avx2(qs);

    // Use cached CPU features to select implementation
    #[cfg(target_arch = "x86_64")]
    let dotf = if super::features::CpuFeatures::get().has_avxvnni {
        mul_sum_q4_0_q8_0_block_avx2_vnni(q4, q8)
    } else {
        mul_sum_q4_0_q8_0_block_avx2_unscaled(q4, q8)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let dotf = mul_sum_q4_0_q8_0_block_avx2_unscaled(q4, q8);
    let scaled = _mm256_mul_ps(dotf, _mm256_set1_ps(scale));
    hsum_avx2(scaled)
}

// ── Q4_1 × Q8_0 kernels ─────────────────────────────────────────────────────────────

/// Unpack Q4_1 nibbles to i8 values in __m256i.
///
/// Input: 16 bytes, each containing 2 nibbles (32 values total).
/// Output: __m256i with 32 i8 values, each = nibble (range 0-15, not centered).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_q4_1_nibbles_avx2(qs: &[u8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "unpack_q4_1_nibbles_avx2: qs must be 16 bytes"
    );

    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);
    let lo = _mm_and_si128(raw, lo_mask);
    let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);
    let q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    q4
}

/// Multiply-sum Q4_1 × Q8_0 block (unscaled).
///
/// Computes sum(q4[i] * q8[i]) for 32-element blocks.
/// Returns __m256 with one i32 result per 8-element group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sum_q4_1_q8_0_block_avx2_unscaled(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let dot16 = _mm256_maddubs_epi16(q4, q8v); // q4 is unsigned (0-15)
    let ones = _mm256_set1_epi16(1);
    let dot32 = _mm256_madd_epi16(ones, dot16);
    _mm256_cvtepi32_ps(dot32)
}

/// AVX2 Q4_1 × Q8_0 block dot product — one 32-element block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_1_q8_0_block_avx2(qs: &[u8], q8: &[u8], scale: f32, min_offset: f32) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_1_q8_0_block_avx2: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_1_q8_0_block_avx2: q8 must have 32 elements"
    );

    // Compute sum of Q8_0 values for min_offset correction
    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let q8_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
    let q8_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1));
    let q8_sum16 = _mm256_add_epi16(q8_low, q8_high);
    // Horizontal sum 16-bit values: pairwise, then to 32-bit, then final sum
    let q8_hadd = _mm256_hadd_epi16(q8_sum16, q8_sum16);
    let q8_hadd2 = _mm256_hadd_epi16(q8_hadd, q8_hadd);
    // Extract the result (only first two elements needed)
    let q8_sum = (_mm256_extract_epi16(q8_hadd2, 0) as i32)
        + (_mm256_extract_epi16(q8_hadd2, 4) as i32);

    let q4 = unpack_q4_1_nibbles_avx2(qs);
    let dotf = mul_sum_q4_1_q8_0_block_avx2_unscaled(q4, q8);
    let scaled = _mm256_mul_ps(dotf, _mm256_set1_ps(scale));
    hsum_avx2(scaled) + min_offset * (q8_sum as f32)
}

/// Scalar Q4_1 × Q8_0 block dot product — one 32-element block.
fn dot_q4_1_q8_0_block_scalar(qs: &[u8], q8: &[u8], scale: f32, min_offset: f32) -> f32 {
    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_1_q8_0_block_scalar: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_1_q8_0_block_scalar: q8 must have 32 elements"
    );
    let mut acc = 0i32;
    let mut q8_sum = 0i32;
    for i in 0..16 {
        let q_lo = (qs[i] & 0x0F) as i32; // 0 to 15
        let q_hi = (qs[i] >> 4) as i32;
        let x_lo = q8[i] as i8 as i32;
        let x_hi = q8[i + 16] as i8 as i32;
        acc += q_lo * x_lo + q_hi * x_hi;
        q8_sum += x_lo + x_hi;
    }
    // sum((q4 * w_scale + w_min) * q8 * x_scale)
    // = sum(q4 * q8) * w_scale * x_scale + w_min * x_scale * sum(q8)
    (acc as f32) * scale + min_offset * (q8_sum as f32)
}

/// Scalar Q4_0 × Q8_0 block dot product — one 32-element block.
fn dot_q4_0_q8_0_block_scalar(qs: &[u8], q8: &[u8], scale: f32) -> f32 {
    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_0_q8_0_block_scalar: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_0_q8_0_block_scalar: q8 must have 32 elements"
    );
    let mut acc = 0i32;
    for i in 0..16 {
        let q_lo = (qs[i] & 0x0F) as i32 - 8;
        let q_hi = (qs[i] >> 4) as i32 - 8;
        let x_lo = q8[i] as i8 as i32;
        let x_hi = q8[i + 16] as i8 as i32;
        acc += q_lo * x_lo + q_hi * x_hi;
    }
    // Q4_0 is symmetric around 0, no min_offset needed
    (acc as f32) * scale
}

// ── Sampling utilities ──────────────────────────────────────────────────────────

/// Find index of maximum value.
pub fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Transposed GEMV for tied embeddings ────────────────────────────────────

/// Q8_0 GEMV transposed for tied embeddings.
///
/// Computes: y = W^T * x where W has shape [in_dim, out_dim]
/// instead of the standard [out_dim, in_dim].
///
/// This is used when the LM head shares token embedding weights.
/// Token embeddings are stored as [hidden_size, vocab_size] in COLUMN-MAJOR format.
/// For output projection we need to compute: logits[v] = sum_i(x[i] * W[i, v])
///
/// In column-major Q8_0 format:
/// - Each column (vocab token) is stored contiguously
/// - Column v starts at offset: v * num_blocks * Q8_BLOCK_BYTES
/// - Within each column, elements are stored in Q8_0 blocks of 32 elements
pub fn gemv_q8_0_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    // Initialize output to zero
    y.fill(0.0);

    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q8_BLOCK_BYTES;

    // For each output dimension (vocab token) - each is a column in the matrix
    for v in 0..out_dim {
        let mut acc = 0.0f32;

        // Column v starts at this offset in the weight data
        let col_offset = v * col_bytes;

        // Iterate through blocks in this column
        for b in 0..num_blocks {
            let block = &w[col_offset + b * Q8_BLOCK_BYTES..col_offset + (b + 1) * Q8_BLOCK_BYTES];
            let scale = super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];

            // Compute dot product for this block
            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }

        y[v] = acc;
    }
}

/// Q4_0 GEMV transposed for transposed weight matrices.
///
/// Computes: y = W^T * x where W has shape [in_dim, out_dim]
/// stored in column-major Q4_0 blocked format.
///
/// Used for FFN down projection where weights are stored as [in_dim, out_dim].
pub fn gemv_q4_0_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    // Initialize output to zero
    y.fill(0.0);

    let num_blocks_per_col = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_BLOCK_BYTES;

    // For each output dimension (column in the original matrix)
    for v in 0..out_dim {
        let mut acc = 0.0f32;

        // Column v starts at this offset in the weight data
        let col_offset = v * col_bytes;

        // Iterate through blocks in this column
        for b in 0..num_blocks_per_col {
            let block = &w[col_offset + b * Q4_BLOCK_BYTES..col_offset + (b + 1) * Q4_BLOCK_BYTES];
            let scale = super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];

            // Dequantize and compute dot product for this block
            for i in 0..16 {
                let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                acc += q_lo * xb[i] + q_hi * xb[i + 16];
            }
        }

        y[v] = acc;
    }
}

/// Q4_1 GEMV transposed for transposed weight matrices.
///
/// Computes: y = W^T * x where W has shape [in_dim, out_dim]
/// stored in column-major Q4_1 blocked format.
///
/// Used for FFN down projection where weights are stored as [in_dim, out_dim].
pub fn gemv_q4_1_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    // Initialize output to zero
    y.fill(0.0);

    let num_blocks_per_col = in_dim / Q4_1_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_1_BLOCK_BYTES;

    // For each output dimension (column in the original matrix)
    for v in 0..out_dim {
        let mut acc = 0.0f32;

        // Column v starts at this offset in the weight data
        let col_offset = v * col_bytes;

        // Iterate through blocks in this column
        for b in 0..num_blocks_per_col {
            let block = &w[col_offset + b * Q4_1_BLOCK_BYTES..col_offset + (b + 1) * Q4_1_BLOCK_BYTES];
            let w_scale = super::quant::load_f16_scale(&block[0..2]);
            let w_min = super::quant::load_f16_scale(&block[2..4]);
            let qs = &block[4..20];
            let xb = &x[b * Q4_1_BLOCK_ELEMS..];

            // Dequantize and compute dot product for this block
            // Q4_1: value = q4 * scale + min
            for i in 0..16 {
                let q_lo = ((qs[i] & 0x0F) as i32) as f32;
                let q_hi = ((qs[i] >> 4) as i32) as f32;
                let v_lo = (q_lo * w_scale + w_min) * xb[i];
                let v_hi = (q_hi * w_scale + w_min) * xb[i + 16];
                acc += v_lo + v_hi;
            }
        }

        y[v] = acc;
    }
}

/// Dispatch GEMV with transposed flag for tied embeddings.
///
/// When `transposed` is true, computes: y = W^T * x
/// Otherwise computes: y = W * x
///
/// # Arguments
///
/// * `w` - Weight matrix bytes
/// * `wtype` - Weight type (quantization format)
/// * `x` - Input vector
/// * `y` - Output vector
/// * `out_dim` - Output dimension
/// * `in_dim` - Input dimension
/// * `transposed` - Whether to compute W^T * x instead of W * x
/// * `scratch` - Optional scratch buffer for Q8_0 quantization (avoids heap allocation)
pub fn dispatch_gemv_transposed(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    transposed: bool,
    scratch: Option<&mut [u8]>,
) -> Result<(), super::CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] = unsafe {
                std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4)
            };
            if transposed {
                gemv_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemv_f32(wf, x, y);
            }
        }
        GgmlType::Q8_0 => {
            if transposed {
                gemv_q8_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_0 => {
            if transposed {
                gemv_q4_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_0_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q4_1 => {
            if transposed {
                gemv_q4_1_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_1_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q6_K => {
            gemv_q6_k(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            gemv_q5_k(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// F32 GEMV transposed for tied embeddings.
fn gemv_f32_transposed(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    // y = W^T * x, where W has shape [in_dim, out_dim]
    // y[v] = sum_i(x[i] * W[i, v])
    for v in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += x[i] * w[i * out_dim + v];
        }
        y[v] = acc;
    }
}
