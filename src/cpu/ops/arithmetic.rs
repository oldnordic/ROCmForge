//! Bias add and residual operations.

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
