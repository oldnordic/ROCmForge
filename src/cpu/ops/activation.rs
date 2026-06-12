//! Activation functions, softmax, and sampling utilities.

// ── Activation functions ────────────────────────────────────────────────────────

/// SiLU activation: x / (1 + exp(-x))
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU fuse in-place: up[i] = silu(gate[i]) * up[i]
pub fn silu_fuse(gate: &[f32], up: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len(), "gate/up dimension mismatch");
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { silu_fuse_avx2(gate, up) };
        return;
    }
    for (g, u) in gate.iter().zip(up.iter_mut()) {
        *u *= silu(*g);
    }
}

// ── AVX2 implementation ─────────────────────────────────────────────────────────

/// AVX2 silu_fuse using vectorized loads/stores with scalar compute.
/// AVX2 lacks a native `exp` intrinsic; each lane is computed scalarly
/// but memory operations are batched via 256-bit registers.
#[cfg(target_arch = "x86_64")]
unsafe fn silu_fuse_avx2(gate: &[f32], up: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = gate.len();
    let chunks = n / 8;

    for i in 0..chunks {
        // Load gate and up into temporary arrays for scalar compute
        let mut gate_buf = [0.0f32; 8];
        let mut up_buf = [0.0f32; 8];
        _mm256_storeu_ps(gate_buf.as_mut_ptr(), _mm256_loadu_ps(gate.as_ptr().add(i * 8)));
        _mm256_storeu_ps(up_buf.as_mut_ptr(), _mm256_loadu_ps(up.as_ptr().add(i * 8)));

        for j in 0..8 {
            let g = gate_buf[j];
            up_buf[j] *= g / (1.0 + (-g).exp());
        }

        _mm256_storeu_ps(up.as_mut_ptr().add(i * 8), _mm256_loadu_ps(up_buf.as_ptr()));
    }

    // Scalar tail
    for i in chunks * 8..n {
        up[i] *= silu(gate[i]);
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

// ── Sampling utilities ──────────────────────────────────────────────────────────

/// Find index of maximum value.
pub fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .expect("invariant: partial_cmp failed (NaN in argmax)")
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}
