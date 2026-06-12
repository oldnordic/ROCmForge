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
#[allow(clippy::too_many_arguments, reason = "function has many parameters by design")]
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

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

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
            let score = if has_avx2 {
                unsafe { dot_f32_avx2(q_h, k_t) * scale }
            } else {
                q_h.iter()
                    .zip(k_t.iter())
                    .map(|(qi, ki)| qi * ki)
                    .sum::<f32>()
                    * scale
            };

            // Online softmax update
            (m, l) = if has_avx2 {
                unsafe { online_softmax_update_avx2(score, m, l, &mut acc, v_t) }
            } else {
                online_softmax_update(score, m, l, &mut acc, v_t)
            };
        }

        // Normalize and write output
        if has_avx2 {
            unsafe { normalize_f32_avx2(out_h, &acc, l) };
        } else {
            for (oi, ai) in out_h.iter_mut().zip(acc.iter()) {
                *oi = ai / l;
            }
        }
    }
}

// ── AVX2 helpers ────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_ps_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

/// AVX2 dot product for f32 slices of any length.
#[cfg(target_arch = "x86_64")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let mut sum = hsum256_ps_avx(acc);
    for i in chunks * 8..n {
        sum += a[i] * b[i];
    }
    sum
}

/// AVX2 online softmax accumulator update.
#[cfg(target_arch = "x86_64")]
unsafe fn online_softmax_update_avx2(
    score: f32,
    m_prev: f32,
    l_prev: f32,
    acc: &mut [f32],
    v: &[f32],
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let m_new = m_prev.max(score);
    let exp_diff = (m_prev - m_new).exp();
    let exp_score = (score - m_new).exp();

    let n = acc.len();
    let chunks = n / 8;
    let diff_vec = _mm256_set1_ps(exp_diff);
    let score_vec = _mm256_set1_ps(exp_score);
    for i in 0..chunks {
        let av = _mm256_loadu_ps(acc.as_ptr().add(i * 8));
        let vv = _mm256_loadu_ps(v.as_ptr().add(i * 8));
        let result = _mm256_fmadd_ps(vv, score_vec, _mm256_mul_ps(av, diff_vec));
        _mm256_storeu_ps(acc.as_mut_ptr().add(i * 8), result);
    }
    for i in chunks * 8..n {
        acc[i] = acc[i] * exp_diff + exp_score * v[i];
    }

    let l_new = l_prev * exp_diff + exp_score;
    (m_new, l_new)
}

/// AVX2 normalize: out[i] = acc[i] / l
#[cfg(target_arch = "x86_64")]
unsafe fn normalize_f32_avx2(out: &mut [f32], acc: &[f32], l: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let chunks = n / 8;
    let inv_l = _mm256_set1_ps(1.0 / l);
    for i in 0..chunks {
        let av = _mm256_loadu_ps(acc.as_ptr().add(i * 8));
        let result = _mm256_mul_ps(av, inv_l);
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), result);
    }
    for i in chunks * 8..n {
        out[i] = acc[i] / l;
    }
}

/// Causal flash attention for prefill.
///
/// q: [seq_len, num_heads, head_dim]   (row-major)
/// k: [seq_len, num_kv_heads, head_dim]
/// v: [seq_len, num_kv_heads, head_dim]
/// out: [seq_len, num_heads, head_dim]
///
#[allow(clippy::too_many_arguments, reason = "function has many parameters by design")]
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

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

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

                    let score = if has_avx2 {
                        unsafe { dot_f32_avx2(q_sh, k_th) * scale }
                    } else {
                        q_sh.iter()
                            .zip(k_th.iter())
                            .map(|(qi, ki)| qi * ki)
                            .sum::<f32>()
                            * scale
                    };

                    (m, l) = if has_avx2 {
                        unsafe { online_softmax_update_avx2(score, m, l, &mut acc, v_th) }
                    } else {
                        online_softmax_update(score, m, l, &mut acc, v_th)
                    };
                }

                if has_avx2 {
                    unsafe { normalize_f32_avx2(out_sh, &acc, l) };
                } else {
                    for (oi, ai) in out_sh.iter_mut().zip(acc.iter()) {
                        *oi = ai / l;
                    }
                }
            }
        });
}
