//! RMS normalization.

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

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { rms_norm_avx2(x, w, out, eps) };
        return;
    }

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

// ── AVX2 implementation ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_ps_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[cfg(target_arch = "x86_64")]
unsafe fn rms_norm_avx2(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;

    // Compute sum of squares using FMA
    let mut sum_vec = _mm256_setzero_ps();
    for i in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        sum_vec = _mm256_fmadd_ps(xv, xv, sum_vec);
    }
    let mut sum_sq = hsum256_ps_avx(sum_vec);
    for item in x.iter().take(n).skip(chunks * 8) {
        sum_sq += item * item;
    }

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = _mm256_set1_ps(1.0 / rms);

    // Normalize and scale
    for i in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i * 8));
        let normed = _mm256_mul_ps(_mm256_mul_ps(xv, inv_rms), wv);
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), normed);
    }
    for i in chunks * 8..n {
        out[i] = x[i] / rms * w[i];
    }
}
