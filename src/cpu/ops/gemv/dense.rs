use super::super::avx2::dot_f32_avx2;
use crate::cpu::quant::load_f16_scale;
use rayon::prelude::*;

/// F32 GEMV: y[row] = dot(W[row, :], x)
///
/// W layout: [out_dim, in_dim] row-major.
pub fn gemv_f32(w: &[f32], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();

    // AVX2 feature detection (cached)
    let features = crate::cpu::features::CpuFeatures::get();
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = features.has_avx2 && features.has_fma && in_dim.is_multiple_of(8);
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

/// F32 GEMV fallback for unaligned byte slices.
pub(crate) fn gemv_f32_bytes(w: &[u8], x: &[f32], y: &mut [f32], _out_dim: usize, in_dim: usize) {
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_start = row * in_dim * 4;
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            let b = &w[row_start + i * 4..row_start + i * 4 + 4];
            let wi = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            acc += wi * x[i];
        }
        *out = acc;
    });
}

/// F32 GEMV transposed for tied embeddings.
pub(crate) fn gemv_f32_transposed(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += x[i] * w[i * out_dim + v];
        }
        *out = acc;
    });
}

/// F32 GEMV transposed for unaligned byte slices.
pub(crate) fn gemv_f32_transposed_bytes(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            let b = &w[(i * out_dim + v) * 4..(i * out_dim + v) * 4 + 4];
            let wi = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            acc += x[i] * wi;
        }
        *out = acc;
    });
}

/// F16 GEMV: y[row] = dot(W[row, :], x)
pub(crate) fn gemv_f16(w: &[u8], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_offset = row * in_dim * 2;
        let mut acc = 0.0f32;
        for (i, x_val) in x.iter().enumerate() {
            let offset = row_offset + i * 2;
            let val = load_f16_scale(&w[offset..offset + 2]);
            acc += val * x_val;
        }
        *out = acc;
    });
}

/// F16 GEMV transposed for tied embeddings.
pub(crate) fn gemv_f16_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    _in_dim: usize,
) {
    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        for (i, x_val) in x.iter().enumerate() {
            let offset = (i * out_dim + v) * 2;
            let val = load_f16_scale(&w[offset..offset + 2]);
            acc += x_val * val;
        }
        *out = acc;
    });
}

/// BF16 GEMV: y[row] = dot(W[row, :], x)
pub(crate) fn gemv_bf16(w: &[u8], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_offset = row * in_dim * 2;
        let mut acc = 0.0f32;
        for (i, x_val) in x.iter().enumerate() {
            let offset = row_offset + i * 2;
            let bits = u16::from_le_bytes([w[offset], w[offset + 1]]);
            let val = half::bf16::from_bits(bits).to_f32();
            acc += val * x_val;
        }
        *out = acc;
    });
}

/// BF16 GEMV transposed for tied embeddings.
pub(crate) fn gemv_bf16_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    _in_dim: usize,
) {
    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        for (i, x_val) in x.iter().enumerate() {
            let offset = (i * out_dim + v) * 2;
            let bits = u16::from_le_bytes([w[offset], w[offset + 1]]);
            let val = half::bf16::from_bits(bits).to_f32();
            acc += x_val * val;
        }
        *out = acc;
    });
}
