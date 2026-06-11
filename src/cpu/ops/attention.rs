//! Flash attention (GQA-aware).

use super::activation::online_softmax_update;
use rayon::prelude::*;

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
    _seq_len: usize,
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
                    let k_th = &k[t * kv_stride + kv_h * head_dim
                        ..t * kv_stride + kv_h * head_dim + head_dim];
                    let v_th = &v[t * kv_stride + kv_h * head_dim
                        ..t * kv_stride + kv_h * head_dim + head_dim];

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
