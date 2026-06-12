//! RoPE positional embeddings.

// ── RoPE positional embeddings ───────────────────────────────────────────────────

/// Apply RoPE (Rotary Position Embeddings) to a vector using precomputed sin/cos.
///
/// Input shape: `[num_heads, head_dim]`
/// `sin` / `cos`: precomputed arrays of length `head_dim / 2`,
///                where `sin[i] = sin(pos * freq[i])`, `cos[i] = cos(pos * freq[i])`
///
/// # Modes
/// - `neox = true`: GPT-NeoX style (pairs within same half) — Qwen2, GPT-NeoX
/// - `neox = false`: Classic RoPE (consecutive pairs) — LLaMA, Mistral
pub fn rope(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    sin: &[f32],
    cos: &[f32],
    neox: bool,
) {
    let total_len = num_heads * head_dim;
    let half = head_dim / 2;
    debug_assert_eq!(x.len(), total_len, "rope input dimension mismatch");
    debug_assert_eq!(sin.len(), half, "rope sin length mismatch");
    debug_assert_eq!(cos.len(), half, "rope cos length mismatch");

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    for h in 0..num_heads {
        let base = h * head_dim;

        if neox && has_avx2 {
            unsafe {
                rope_neox_avx2(&mut x[base..base + head_dim], half, sin, cos);
            }
        } else {
            for i in 0..half {
                let sin_a = sin[i];
                let cos_a = cos[i];

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
}

/// AVX2 NeoX RoPE: process 8 pairs at a time.
///
/// Layout: `x[0..half]` is first half, `x[half..head_dim]` is second half.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn rope_neox_avx2(x: &mut [f32], half: usize, sin: &[f32], cos: &[f32]) {
    use std::arch::x86_64::*;
    let chunks = half / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let x0 = _mm256_loadu_ps(x.as_ptr().add(offset));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(offset + half));
        let s = _mm256_loadu_ps(sin.as_ptr().add(offset));
        let c = _mm256_loadu_ps(cos.as_ptr().add(offset));

        // x0' = x0 * cos - x1 * sin
        let x1_sin = _mm256_mul_ps(x1, s);
        let x0_out = _mm256_fmsub_ps(x0, c, x1_sin);

        // x1' = x0 * sin + x1 * cos
        let x0_sin = _mm256_mul_ps(x0, s);
        let x1_out = _mm256_fmadd_ps(x1, c, x0_sin);

        _mm256_storeu_ps(x.as_mut_ptr().add(offset), x0_out);
        _mm256_storeu_ps(x.as_mut_ptr().add(offset + half), x1_out);
    }

    // Tail: scalar fallback for remaining pairs
    for i in chunks * 8..half {
        let x0 = x[i];
        let x1 = x[i + half];
        let sin_a = sin[i];
        let cos_a = cos[i];
        x[i] = x0 * cos_a - x1 * sin_a;
        x[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

/// Convenience wrapper: compute sin/cos from `pos` and `freq`, then apply RoPE.
///
/// `freq`: precomputed frequencies, length = head_dim/2,
///         where `freq[i] = 1/theta^(2i/head_dim)`
pub fn rope_with_pos(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    pos: usize,
    freq: &[f32],
    neox: bool,
) {
    let half = head_dim / 2;
    let mut sin = vec![0.0f32; half];
    let mut cos = vec![0.0f32; half];
    for i in 0..half {
        let angle = pos as f32 * freq[i];
        let (s, c) = angle.sin_cos();
        sin[i] = s;
        cos[i] = c;
    }
    rope(x, num_heads, head_dim, &sin, &cos, neox);
}

/// Batched RoPE: apply to each row with position `start_pos + s`.
///
/// Computes sin/cos per row; for decode paths that reuse the same position,
/// prefer computing sin/cos once and calling `rope()` directly.
pub fn rope_batch(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    start_pos: usize,
    freq: &[f32],
    neox: bool,
) {
    let row_len = num_heads * head_dim;
    let seq_len = x.len() / row_len;

    for s in 0..seq_len {
        let row = &mut x[s * row_len..(s + 1) * row_len];
        rope_with_pos(row, num_heads, head_dim, start_pos + s, freq, neox);
    }
}
