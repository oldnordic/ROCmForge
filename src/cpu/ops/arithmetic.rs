//! Bias add and residual operations.

// ── Residual operations ─────────────────────────────────────────────────────────

/// In-place bias add: x[i] += bias[i]
pub fn add_bias(x: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(x.len(), bias.len(), "bias dimension mismatch");
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { add_bias_avx2(x, bias) };
        return;
    }
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
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { residual_add_avx2(a, b) };
        return;
    }
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

// ── AVX2 implementations ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
unsafe fn add_bias_avx2(x: &mut [f32], bias: &[f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let bv = _mm256_loadu_ps(bias.as_ptr().add(i * 8));
        let sum = _mm256_add_ps(xv, bv);
        _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), sum);
    }
    for i in chunks * 8..n {
        x[i] += bias[i];
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn residual_add_avx2(a: &mut [f32], b: &[f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let sum = _mm256_add_ps(av, bv);
        _mm256_storeu_ps(a.as_mut_ptr().add(i * 8), sum);
    }
    for i in chunks * 8..n {
        a[i] += b[i];
    }
}
