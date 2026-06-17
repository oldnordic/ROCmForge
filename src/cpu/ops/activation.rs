//! Activation functions, softmax, and sampling utilities.

// ── Activation functions ────────────────────────────────────────────────────────

/// SiLU activation: x / (1 + exp(-x))
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GeLU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let c = x + 0.044715 * x * x * x;
    0.5 * x * (1.0 + (c * 0.797_884_6).tanh())
}

/// Apply GeLU in-place to a slice.
pub fn gelu_inplace(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = gelu(*x);
    }
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
        // Load gate and up into local arrays for scalar compute
        let mut gate_buf = [0.0f32; 8];
        let mut up_buf = [0.0f32; 8];
        _mm256_storeu_ps(
            gate_buf.as_mut_ptr(),
            _mm256_loadu_ps(gate.as_ptr().add(i * 8)),
        );
        _mm256_storeu_ps(up_buf.as_mut_ptr(), _mm256_loadu_ps(up.as_ptr().add(i * 8)));

        for j in 0..8 {
            let g = gate_buf[j];
            up_buf[j] *= g / (1.0 + (-g).exp());
        }

        _mm256_storeu_ps(up.as_mut_ptr().add(i * 8), _mm256_loadu_ps(up_buf.as_ptr()));
    }

    // Scalar tail
    let tail_start = chunks * 8;
    for (u, g) in up[tail_start..].iter_mut().zip(&gate[tail_start..]) {
        *u *= silu(*g);
    }
}

/// Horizontal max reduction of an AVX2 `__m256` register.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hmax256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let x128 = _mm_max_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
    let x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

// ── Softmax ─────────────────────────────────────────────────────────────────────

/// Softmax in-place: x[i] = exp(x[i] - max) / sum
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { softmax_avx2(x) };
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

/// AVX2 softmax: vectorized max-find + normalize; scalar exp.
#[cfg(target_arch = "x86_64")]
unsafe fn softmax_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;

    // Find max
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        max_vec = _mm256_max_ps(max_vec, xv);
    }
    let mut max = hmax256_ps(max_vec);
    for item in x.chunks_exact(8).remainder() {
        max = max.max(*item);
    }

    // Exp and sum (scalar — AVX2 lacks native exp)
    let mut sum = 0.0f32;
    for item in x.iter_mut() {
        *item = (*item - max).exp();
        sum += *item;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = _mm256_set1_ps(1.0 / sum);
        for i in 0..chunks {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            let result = _mm256_mul_ps(xv, inv_sum);
            _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), result);
        }
        let tail_start = chunks * 8;
        for item in &mut x[tail_start..] {
            *item /= sum;
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
    if x.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        let max_val = unsafe { argmax_find_max_avx2(x) };
        return x.iter().position(|&v| v == max_val).unwrap_or(0);
    }

    x.iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .expect("invariant: partial_cmp failed (NaN in argmax)")
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// AVX2 helper: find the maximum value in a f32 slice.
#[cfg(target_arch = "x86_64")]
unsafe fn argmax_find_max_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    let mut max_val = x[0];
    if chunks > 0 {
        let mut max_vec = _mm256_loadu_ps(x.as_ptr());
        for i in 1..chunks {
            let val = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            max_vec = _mm256_max_ps(max_vec, val);
        }
        max_val = hmax256_ps(max_vec).max(max_val);
    }
    for item in x.chunks_exact(8).remainder() {
        if *item > max_val {
            max_val = *item;
        }
    }
    max_val
}
