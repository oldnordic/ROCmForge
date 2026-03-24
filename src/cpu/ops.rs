//! CPU primitive operations for inference.
//!
//! Pure f32 operations - no SIMD by default. Operations are:
//! - RMS normalization
//! - RoPE positional embeddings (NeoX and classic styles)
//! - Attention (GQA-aware flash attention)
//! - SwiGLU activation
//! - Softmax and sampling utilities

use rayon::prelude::*;

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

    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_h)| {
            let kv_h = h / kv_group;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];

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
        });
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

use super::quant::{Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS, Q8_BLOCK_BYTES, Q8_BLOCK_ELEMS};
use crate::loader::GgmlType;

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
                let block = &row_w[b * Q4_BLOCK_BYTES..];
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
                let block = &row_w[b * Q8_BLOCK_BYTES..];
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

/// Dispatch GEMM by weight type.
pub fn dispatch_gemm(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), super::CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            gemm_f32(wf, x, y, out_dim, in_dim);
        }
        GgmlType::Q4_0 => gemm_q4_0(w, x, y, out_dim, in_dim),
        GgmlType::Q8_0 => gemm_q8_0(w, x, y, out_dim, in_dim),
        other => return Err(super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

// ── GEMV (matrix-vector multiply for decode) ────────────────────────────────

/// F32 GEMV: y[row] = dot(W[row, :], x)
///
/// W layout: [out_dim, in_dim] row-major.
pub fn gemv_f32(w: &[f32], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    for (row, out) in y.iter_mut().enumerate() {
        let row_w = &w[row * in_dim..(row + 1) * in_dim];
        *out = row_w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
}

/// Q4_0 GEMV: dequant on-the-fly.
pub fn gemv_q4_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
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

/// Q8_0 GEMV: dequant on-the-fly.
pub fn gemv_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
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

/// Dispatch GEMV based on weight type.
///
/// Computes: y = W * x (matrix-vector multiply)
pub fn dispatch_gemv(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), super::CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] = unsafe {
                std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4)
            };
            gemv_f32(wf, x, y);
        }
        GgmlType::Q4_0 => gemv_q4_0(w, x, y, out_dim, in_dim),
        GgmlType::Q8_0 => gemv_q8_0(w, x, y, out_dim, in_dim),
        other => return Err(super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
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

// ── Tests ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_produces_correct_output() {
        let x = vec![3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];

        rms_norm(&x, &w, &mut out, 1e-6);

        let expected_rms = (12.5_f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 / expected_rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / expected_rms).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_batch_processes_all_rows() {
        // 2 rows of 2 elements each
        let x = vec![3.0, 4.0, 6.0, 8.0];
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 4];

        rms_norm_batch(&x, &w, &mut out, 2, 1e-6);

        // First row
        let rms0 = ((9.0_f32 + 16.0) / 2.0 + 1e-6).sqrt();
        assert!((out[0] - 3.0 / rms0).abs() < 1e-5);

        // Second row
        let rms1 = ((36.0_f32 + 64.0) / 2.0 + 1e-6).sqrt();
        assert!((out[2] - 6.0 / rms1).abs() < 1e-5);
    }

    #[test]
    fn silu_activation_correct() {
        // silu(0) = 0
        assert!((silu(0.0) - 0.0).abs() < 1e-6);

        // silu(1) ≈ 0.731
        assert!((silu(1.0) - 0.7310585786300049).abs() < 1e-5);

        // silu(-1) ≈ -0.2689
        assert!((silu(-1.0) - (-0.2689414213699951)).abs() < 1e-5);
    }

    #[test]
    fn rope_neox_correct() {
        // Single head, head_dim=4, position 0, theta=10000
        let mut x = vec![1.0, 2.0, 3.0, 4.0];

        rope(&mut x, 1, 4, 0, 10000.0, true);

        // Position 0: angle = 0, sin=0, cos=1, so x should be unchanged
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn rope_classic_correct() {
        // Single head, head_dim=4, position 0, theta=10000
        let mut x = vec![1.0, 2.0, 3.0, 4.0];

        rope(&mut x, 1, 4, 0, 10000.0, false);

        // Position 0: angle = 0, sin=0, cos=1
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Largest should have highest probability
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn argmax_finds_max() {
        let x = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&x), 1);

        let y = vec![-1.0, -5.0, -3.0];
        assert_eq!(argmax(&y), 0);
    }

    #[test]
    fn flash_attn_decode_single_head() {
        // 1 head, head_dim=4, 2 cached positions
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k_cache = vec![
            1.0, 0.0, 0.0, 0.0,  // pos 0
            0.0, 1.0, 0.0, 0.0,  // pos 1
        ];
        let v_cache = vec![
            1.0, 1.0, 1.0, 1.0,  // pos 0
            2.0, 2.0, 2.0, 2.0,  // pos 1
        ];
        let mut out = vec![0.0; 4];

        flash_attn_decode(&q, &k_cache, &v_cache, &mut out, 2, 1, 1, 4);

        // q·k[0] = 1, q·k[1] = 0
        // softmax scores: exp(1)/(exp(1)+exp(0)) ≈ 0.731, exp(0)/(exp(1)+exp(0)) ≈ 0.269
        // output = 0.731 * v[0] + 0.269 * v[1] ≈ 0.731*1 + 0.269*2 ≈ 1.269
        assert!(out[0] > 1.0 && out[0] < 1.5, "out[0] = {}, expected ~1.27", out[0]);
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
        assert!((y[0] - 14.0_f32).abs() < 1e-5);
        assert!((y[1] - 32.0_f32).abs() < 1e-5);
    }

    #[test]
    fn gemv_q4_0_matches_f32() {
        // Create a simple Q4_0 weight matrix: 2 rows, 32 cols (1 block per row)
        // Scale = 1.0 for both blocks
        let mut w_q4 = vec![
            // Block 0 (row 0): scale=1.0 (f16 0x3C00), then 16 bytes of nibbles
            0x00, 0x3C, // f16 1.0
            // nibbles: each byte packs lo+hi nibbles, value = nibble - 8
            // Let's use simple values: lo=8 (→0), hi=8 (→0) for all
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
            // Block 1 (row 1): same
            0x00, 0x3C,
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
            0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
        ];

        // Input: 32 zeros
        let x = vec![0.0f32; 32];
        let mut y = vec![0.0f32; 2];

        gemv_q4_0(&w_q4, &x, &mut y, 2, 32);

        // All zeros input, all zeros weights → output should be 0
        assert!((y[0] - 0.0_f32).abs() < 1e-5);
        assert!((y[1] - 0.0_f32).abs() < 1e-5);

        // Now test with non-zero values
        // Weights: nibbles 9,9 → value = 9-8 = 1, scale=1.0 → dequant=1.0
        // But nibbles are packed: byte 0x99 means lo=9, hi=9
        let mut w_q4_ones = vec![
            // Block 0: scale=1.0
            0x00, 0x3C,
            // All nibbles = 9 → dequant = 1.0
            0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
            0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
            // Block 1: scale=1.0
            0x00, 0x3C,
            0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
            0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
        ];

        // Input: all 1.0
        let x_ones = vec![1.0f32; 32];
        let mut y2 = vec![0.0f32; 2];

        gemv_q4_0(&w_q4_ones, &x_ones, &mut y2, 2, 32);

        // Each row: sum of 32 values, each = 1.0 * 1.0 = 1.0
        // So output should be 32.0
        assert!((y2[0] - 32.0_f32).abs() < 1e-3, "y2[0] = {}, expected 32.0", y2[0]);
        assert!((y2[1] - 32.0_f32).abs() < 1e-3, "y2[1] = {}, expected 32.0", y2[1]);
    }

    #[test]
    fn gemv_q4_0_large_dim() {
        // Test with multiple blocks (simulating ffn_down dimensions)
        // 896 rows, 4864 cols = 896 rows, 152 blocks per row
        let out_dim = 896;
        let in_dim = 4864;
        let num_blocks = in_dim / 32;
        let row_bytes = num_blocks * 18;

        // Create weight tensor: all zeros (nibble = 8 → value = 0)
        let mut w = vec![0u8; out_dim * row_bytes];
        for row in 0..out_dim {
            for b in 0..num_blocks {
                let off = row * row_bytes + b * 18;
                w[off] = 0x00;
                w[off + 1] = 0x3C; // scale = 1.0
                for i in 0..16 {
                    w[off + 2 + i] = 0x88; // nibble 8 → value 0
                }
            }
        }

        // Input: all zeros
        let x = vec![0.0f32; in_dim];
        let mut y = vec![0.0f32; out_dim];

        gemv_q4_0(&w, &x, &mut y, out_dim, in_dim);

        // All zeros → output should be all zeros
        for (i, &yi) in y.iter().enumerate() {
            assert!((yi - 0.0_f32).abs() < 1e-5, "y[{}] = {}, expected 0", i, yi);
        }
    }
}
