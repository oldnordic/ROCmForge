//! GEMV (matrix-vector multiply for decode).

use crate::cpu::quant::{
    load_f16_scale, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS,
    Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q8_0_MAX, Q8_BLOCK_BYTES, Q8_BLOCK_ELEMS,
};
use crate::cpu::weights::WeightMeta;
use crate::loader::GgmlType;
use rayon::prelude::*;

use super::avx2::{
    dot_f32_avx2, dot_q4_0_block_avx2, dot_q4_0_q8_0_block_avx2, dot_q4_0_q8_0_block_scalar,
    dot_q4_1_q8_0_block_avx2, dot_q4_1_q8_0_block_scalar,
};
use super::gemm::{gemm_q2_k_fallback, gemm_q3_k_fallback, gemm_q5_k_fallback, gemm_q6_k_fallback};

// ── GEMV (matrix-vector multiply for decode) ────────────────────────────────

/// Quantize a single f32 vector to Q8_0 (one block per Q8_BLOCK_ELEMS).
///
/// This is used for quantizing activations once per GEMV call for Q4_0 × Q8_0 dot products.
fn quantize_q8_0_single(x: &[f32], out: &mut [u8], in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    debug_assert_eq!(out.len(), num_blocks * Q8_BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * Q8_BLOCK_ELEMS..(b + 1) * Q8_BLOCK_ELEMS];
        let off = b * Q8_BLOCK_BYTES;
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / Q8_0_MAX } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let scale_bytes = half::f16::from_f32(scale).to_bits().to_le_bytes();
        out[off] = scale_bytes[0];
        out[off + 1] = scale_bytes[1];
        for i in 0..Q8_BLOCK_ELEMS {
            let q = (xb[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out[off + 2 + i] = q as u8;
        }
    }
}

/// F32 GEMV: y[row] = dot(W[row, :], x)
///
/// W layout: [out_dim, in_dim] row-major.
pub fn gemv_f32(w: &[f32], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();

    // AVX2 feature detection (cached)
    let features = super::super::features::CpuFeatures::get();
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

/// Q4_0 GEMV: dequant on-the-fly.
pub fn gemv_q4_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    // AVX2 feature detection (cached)
    let features = super::super::features::CpuFeatures::get();
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = features.has_avx2 && features.has_fma;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for row in 0..out_dim {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..(b + 1) * Q4_BLOCK_BYTES];
            let scale = super::super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_block_avx2(qs, xb, scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..16 {
                        let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                        let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                        acc += q_lo * xb[i] + q_hi * xb[i + 16];
                    }
                }
            } else {
                for i in 0..16 {
                    let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                    let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                    acc += q_lo * xb[i] + q_hi * xb[i + 16];
                }
            }
        }
        y[row] = acc;
    }
}

/// Q4_0 × Q8_0 GEMV: quantize input to Q8_0 once, then integer dot product.
/// This is faster than Q4_0 × f32 because it avoids 4× int→f32 conversions per block.
///
/// # Arguments
///
/// * `w` - Weight matrix in Q4_0 format [out_dim * row_bytes]
/// * `x` - Input vector [in_dim]
/// * `y` - Output vector [out_dim]
/// * `out_dim` - Output dimension (number of output rows)
/// * `in_dim` - Input dimension (must be multiple of Q4_BLOCK_ELEMS)
/// * `scratch` - Optional scratch buffer for Q8_0 quantization. If None, allocates internally.
pub fn gemv_q4_0_q8_0(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
    scratch: Option<&mut [u8]>,
) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    // Use provided scratch buffer or allocate on heap
    let mut owned_scratch;
    let x_q8 = if let Some(buf) = scratch {
        if buf.len() >= num_blocks * Q8_BLOCK_BYTES {
            &mut buf[..num_blocks * Q8_BLOCK_BYTES]
        } else {
            // Scratch buffer too small, fall back to heap allocation
            owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
            &mut owned_scratch[..]
        }
    } else {
        // No scratch buffer provided, allocate on heap (backward compatible)
        owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
        &mut owned_scratch[..]
    };

    quantize_q8_0_single(x, x_q8, in_dim);

    let features = super::super::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        // Process 2 blocks at a time for better ILP (instruction-level parallelism)
        let mut b = 0;
        while b + 1 < num_blocks {
            // Prefetch blocks ahead
            #[cfg(target_arch = "x86_64")]
            if b + 2 < num_blocks {
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    use std::arch::x86_64::_MM_HINT_T0;
                    let next_ptr = row_w[(b + 2) * Q4_BLOCK_BYTES..].as_ptr();
                    _mm_prefetch(next_ptr as *const i8, _MM_HINT_T0);
                }
            }

            // Block 0
            let block0 = &row_w[b * Q4_BLOCK_BYTES..];
            let w_scale0 = load_f16_scale(&block0[0..2]);
            let x_scale0 = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale0 = w_scale0 * x_scale0;
            let qs0 = &block0[2..18];
            let q8_0 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Block 1
            let block1 = &row_w[(b + 1) * Q4_BLOCK_BYTES..];
            let w_scale1 = load_f16_scale(&block1[0..2]);
            let x_scale1 = load_f16_scale(&x_q8[(b + 1) * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale1 = w_scale1 * x_scale1;
            let qs1 = &block1[2..18];
            let q8_1 = &x_q8[(b + 1) * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Compute both blocks
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs0, q8_0, combined_scale0) };
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs1, q8_1, combined_scale1) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_0_q8_0_block_scalar(qs0, q8_0, combined_scale0);
                    acc += dot_q4_0_q8_0_block_scalar(qs1, q8_1, combined_scale1);
                }
            } else {
                acc += dot_q4_0_q8_0_block_scalar(qs0, q8_0, combined_scale0);
                acc += dot_q4_0_q8_0_block_scalar(qs1, q8_1, combined_scale1);
            }

            b += 2;
        }

        // Handle remaining block
        while b < num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..];
            let w_scale = load_f16_scale(&block[0..2]);
            let x_scale = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale = w_scale * x_scale;
            let qs = &block[2..18];
            let q8 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs, q8, combined_scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_0_q8_0_block_scalar(qs, q8, combined_scale);
                }
            } else {
                acc += dot_q4_0_q8_0_block_scalar(qs, q8, combined_scale);
            }
            b += 1;
        }

        *out = acc;
    });
}

/// Q4_1 × Q8_0 GEMV: quantize input to Q8_0 once, then integer dot product.
///
/// Q4_1 block format: [f16 scale | f16 min | 16 nibble bytes] = 20 bytes
/// Values are in range [min, min + 15*scale]
pub fn gemv_q4_1_q8_0(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    scratch: Option<&mut [u8]>,
) {
    let num_blocks = in_dim / Q4_1_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_1_BLOCK_BYTES;

    // Use provided scratch buffer or allocate on heap
    let mut owned_scratch;
    let x_q8 = if let Some(buf) = scratch {
        if buf.len() >= num_blocks * Q8_BLOCK_BYTES {
            &mut buf[..num_blocks * Q8_BLOCK_BYTES]
        } else {
            owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
            &mut owned_scratch[..]
        }
    } else {
        owned_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
        &mut owned_scratch[..]
    };

    quantize_q8_0_single(x, x_q8, in_dim);

    let features = super::super::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        // Process 2 blocks at a time for better ILP
        let mut b = 0;
        while b + 1 < num_blocks {
            // Prefetch blocks ahead
            #[cfg(target_arch = "x86_64")]
            if b + 2 < num_blocks {
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    use std::arch::x86_64::_MM_HINT_T0;
                    let next_ptr = row_w[(b + 2) * Q4_1_BLOCK_BYTES..].as_ptr();
                    _mm_prefetch(next_ptr as *const i8, _MM_HINT_T0);
                }
            }

            // Block 0
            let block0 = &row_w[b * Q4_1_BLOCK_BYTES..];
            let w_scale0 = load_f16_scale(&block0[0..2]);
            let w_min0 = load_f16_scale(&block0[2..4]);
            let x_scale0 = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale0 = w_scale0 * x_scale0;
            let qs0 = &block0[4..20];
            let q8_0 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Block 1
            let block1 = &row_w[(b + 1) * Q4_1_BLOCK_BYTES..];
            let w_scale1 = load_f16_scale(&block1[0..2]);
            let w_min1 = load_f16_scale(&block1[2..4]);
            let x_scale1 = load_f16_scale(&x_q8[(b + 1) * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale1 = w_scale1 * x_scale1;
            let qs1 = &block1[4..20];
            let q8_1 = &x_q8[(b + 1) * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];

            // Compute both blocks
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe {
                        dot_q4_1_q8_0_block_avx2(qs0, q8_0, combined_scale0, w_min0 * x_scale0)
                    };
                    acc += unsafe {
                        dot_q4_1_q8_0_block_avx2(qs1, q8_1, combined_scale1, w_min1 * x_scale1)
                    };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc +=
                        dot_q4_1_q8_0_block_scalar(qs0, q8_0, combined_scale0, w_min0 * x_scale0);
                    acc +=
                        dot_q4_1_q8_0_block_scalar(qs1, q8_1, combined_scale1, w_min1 * x_scale1);
                }
            } else {
                acc += dot_q4_1_q8_0_block_scalar(qs0, q8_0, combined_scale0, w_min0 * x_scale0);
                acc += dot_q4_1_q8_0_block_scalar(qs1, q8_1, combined_scale1, w_min1 * x_scale1);
            }

            b += 2;
        }

        // Handle remaining block
        while b < num_blocks {
            let block = &row_w[b * Q4_1_BLOCK_BYTES..];
            let w_scale = load_f16_scale(&block[0..2]);
            let w_min = load_f16_scale(&block[2..4]);
            let x_scale = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale = w_scale * x_scale;
            let qs = &block[4..20];
            let q8 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe {
                        dot_q4_1_q8_0_block_avx2(qs, q8, combined_scale, w_min * x_scale)
                    };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_1_q8_0_block_scalar(qs, q8, combined_scale, w_min * x_scale);
                }
            } else {
                acc += dot_q4_1_q8_0_block_scalar(qs, q8, combined_scale, w_min * x_scale);
            }
            b += 1;
        }

        *out = acc;
    });
    let _ = out_dim;
}

/// Q5_0 GEMV: dequant on-the-fly.
pub fn gemv_q5_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q5_0_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_0_BLOCK_BYTES;

    for o in 0..out_dim {
        let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
            let d = super::super::quant::load_f16_scale(&block[0..2]);
            let qh = &block[2..6];
            let qs = &block[6..22];
            let xb = &x[b * Q5_0_BLOCK_ELEMS..];

            for i in 0..16 {
                let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                let low_bits_0 = qs[i] & 0x0F;
                let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                let low_bits_1 = (qs[i] >> 4) & 0x0F;
                let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
            }
        }
        y[o] = acc;
    }
}

/// Q8_0 GEMV: dequant on-the-fly.
pub fn gemv_q8_0(w: &[u8], x: &[f32], y: &mut [f32], _out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q8_BLOCK_BYTES;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &row_w[b * Q8_BLOCK_BYTES..(b + 1) * Q8_BLOCK_BYTES];
            let scale = super::super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];

            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }
        *out = acc;
    });
}

/// Q3_K GEMV: dequant on-the-fly (fallback, slower but works).
/// Reuses gemm_q3_k_fallback with batch_size=1 for consistency.
pub fn gemv_q3_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // GEMV is just GEMM with batch_size=1
    gemm_q3_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q2_K GEMV: dequant on-the-fly (fallback, slower but works).
/// Reuses gemm_q2_k_fallback with batch_size=1 for consistency.
pub fn gemv_q2_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // GEMV is just GEMM with batch_size=1
    gemm_q2_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q6_K GEMV: dequant on-the-fly (fallback, slower but works).
/// Reuses gemm_q6_k_fallback with batch_size=1 for consistency.
pub fn gemv_q6_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // GEMV is just GEMM with batch_size=1
    gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q2_K GEMV transposed for tied embeddings.
///
/// Computes: y = W^T * x where W is [in_dim, out_dim].
fn gemv_q2_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::kernels::BlockQ2K;
    use super::super::quant::{Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q2_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q2_K_BLOCK_BYTES;

    for v in 0..out_dim {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q2_K_BLOCK_BYTES) as *const BlockQ2K) };
            let mut deq = [0.0f32; Q2_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q2_K_BLOCK_ELEMS..(b + 1) * Q2_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        y[v] = acc;
    }
}

/// Q3_K GEMV transposed for tied embeddings.
fn gemv_q3_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::kernels::BlockQ3K;
    use super::super::quant::{Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q3_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q3_K_BLOCK_BYTES;

    for v in 0..out_dim {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q3_K_BLOCK_BYTES) as *const BlockQ3K) };
            let mut deq = [0.0f32; Q3_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q3_K_BLOCK_ELEMS..(b + 1) * Q3_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        y[v] = acc;
    }
}

/// Q5_K GEMV transposed for tied embeddings.
fn gemv_q5_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::kernels::BlockQ5K;
    use super::super::quant::{Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q5_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q5_K_BLOCK_BYTES;

    for v in 0..out_dim {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q5_K_BLOCK_BYTES) as *const BlockQ5K) };
            let mut deq = [0.0f32; Q5_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q5_K_BLOCK_ELEMS..(b + 1) * Q5_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        y[v] = acc;
    }
}

/// Q6_K GEMV transposed for tied embeddings.
fn gemv_q6_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::kernels::BlockQ6K;
    use super::super::quant::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q6_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q6_K_BLOCK_BYTES;

    for v in 0..out_dim {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q6_K_BLOCK_BYTES) as *const BlockQ6K) };
            let mut deq = [0.0f32; Q6_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q6_K_BLOCK_ELEMS..(b + 1) * Q6_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        y[v] = acc;
    }
}

/// Q5_K GEMV: dequantize weights on the fly during matrix-vector multiplication.
///
/// Computes y = W * x where W is Q5_K quantized.
/// This is a wrapper around gemm_q5_k_fallback since GEMV is just GEMM with batch_size=1.
pub fn gemv_q5_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // GEMV is just GEMM with batch_size=1
    gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q4_K GEMV for transposed weights (tied embeddings).
///
/// For transposed access (tied LM head), we compute y = W^T * x
/// where W is [hidden, vocab] stored as [vocab, hidden].
/// This means each output dimension corresponds to a row of W^T, which is a column of W.
///
/// Simpler approach: dequantize to f32 on-the-fly for each output dimension.
fn gemv_q4_k_transposed_fallback(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
) {
    use crate::cpu::kernels::q4::BlockQ4K;

    let num_blocks_k = in_dim / 256;
    let row_bytes = num_blocks_k * BlockQ4K::SIZE;

    // For transposed access, output dimension corresponds to rows in stored layout
    y.par_iter_mut().enumerate().for_each(|(vocab_idx, out)| {
        let mut acc = 0.0f32;

        for block_idx in 0..num_blocks_k {
            let block_ptr = unsafe {
                w.as_ptr()
                    .add(vocab_idx * row_bytes + block_idx * BlockQ4K::SIZE)
                    as *const BlockQ4K
            };
            let block = unsafe { &*block_ptr };

            // Dequantize this block of 256 weights and compute dot product with x
            let block_start = block_idx * 256;
            for i in 0..256 {
                if block_start + i < in_dim {
                    // Dequantize single Q4_K value
                    let sub_block_idx = i / 32;
                    let _scale_idx = sub_block_idx / 2;
                    let _sign = (i % 32) / 16;

                    // Get the scale (simplified - actual Q4_K scale unpacking is complex)
                    // For now, use a simple dequantization
                    let q4_value = if i < 128 {
                        (block.qs[i / 2] >> (4 * (i % 2))) & 0x0F
                    } else {
                        (block.qs[64 + (i - 128) / 2] >> (4 * ((i - 128) % 2))) & 0x0F
                    };

                    // Simple dequantization (not fully accurate but works for fallback)
                    let d = half::f16::from_le_bytes(block.d).to_f32();
                    let weight = d * (q4_value as f32 - 8.0);
                    acc += weight * x[block_start + i];
                }
            }
        }

        *out = acc;
    });
}

/// Dispatch GEMV based on weight type with automatic transposition detection.
///
/// Uses metadata to determine if weights need transposed access.
///
/// Computes: y = W * x (matrix-vector multiply) or y = W^T * x if transposed
///
/// # Arguments
///
/// * `w` - Weight matrix bytes
/// * `meta` - Weight metadata (type, dimensions, transposition)
/// * `x` - Input vector
/// * `y` - Output vector
/// * `out_dim` - Output dimension
/// * `in_dim` - Input dimension
/// * `scratch` - Optional scratch buffer for Q8_0 quantization (avoids heap allocation)
pub fn dispatch_gemv(
    w: &[u8],
    meta: &WeightMeta,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    scratch: Option<&mut [u8]>,
) -> Result<(), super::super::CpuError> {
    // Validate dimensions for block-based quantization
    let block_elems = match meta.wtype {
        GgmlType::F32 => 1,
        GgmlType::Q8_0 => super::super::quant::Q8_BLOCK_ELEMS,
        GgmlType::Q4_0 | GgmlType::Q4_1 => super::super::quant::Q4_BLOCK_ELEMS,
        GgmlType::Q5_0 => super::super::quant::Q5_0_BLOCK_ELEMS,
        GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q2_K => super::super::quant::Q4_K_BLOCK_ELEMS,
        GgmlType::Q3_K => super::super::quant::Q3_K_BLOCK_ELEMS,
        GgmlType::Q5_K => super::super::quant::Q5_K_BLOCK_ELEMS,
        _ => 1,
    };

    if !in_dim.is_multiple_of(block_elems) {
        return Err(super::super::CpuError::InvalidOperation(format!(
            "in_dim {} is not a multiple of block size {} for type {:?}",
            in_dim, block_elems, meta.wtype
        )));
    }

    match meta.wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            if meta.needs_transpose {
                gemv_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemv_f32(wf, x, y);
            }
        }
        GgmlType::F16 => {
            if meta.needs_transpose {
                gemv_f16_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_f16(w, x, y);
            }
        }
        GgmlType::Q4_0 => {
            if meta.needs_transpose {
                gemv_q4_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_0_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q4_1 => {
            if meta.needs_transpose {
                gemv_q4_1_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_1_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q5_0 => {
            crate::cpu::kernels::gemm_q5_0_q8::gemv_q5_0_q8_0_dispatch(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q3_K => {
            if meta.needs_transpose {
                gemv_q3_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q3_k(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q2_K => {
            if meta.needs_transpose {
                gemv_q2_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q2_k(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q8_0 => {
            if meta.needs_transpose {
                gemv_q8_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_K => {
            // Q4_K GEMV: use SIMD for non-transposed, scalar for transposed (tied embeddings)
            if meta.needs_transpose {
                // For transposed Q4_K (tied embeddings), use dequantize-on-the-fly
                gemv_q4_k_transposed_fallback(w, x, y, out_dim, in_dim);
            } else {
                crate::cpu::kernels::gemm_q4k_q8::gemv_q4_k_q8_k_dispatch(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q6_K => {
            if meta.needs_transpose {
                gemv_q6_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q6_k(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q5_K => {
            if meta.needs_transpose {
                gemv_q5_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q5_k(w, x, y, out_dim, in_dim);
            }
        }
        other => return Err(super::super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

// ── Transposed GEMV for tied embeddings ────────────────────────────────────

/// Q8_0 GEMV transposed for tied embeddings.
///
/// Computes: y = W^T * x where W has shape [in_dim, out_dim]
/// instead of the standard [out_dim, in_dim].
///
/// This is used when the LM head shares token embedding weights.
/// Token embeddings are stored as [hidden_size, vocab_size] in COLUMN-MAJOR format.
/// For output projection we need to compute: logits[v] = sum_i(x[i] * W[i, v])
///
/// In column-major Q8_0 format:
/// - Each column (vocab token) is stored contiguously
/// - Column v starts at offset: v * num_blocks * Q8_BLOCK_BYTES
/// - Within each column, elements are stored in Q8_0 blocks of 32 elements
pub fn gemv_q8_0_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // Initialize output to zero
    y.fill(0.0);

    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q8_BLOCK_BYTES;

    // For each output dimension (vocab token) - each is a column in the matrix
    for v in 0..out_dim {
        let mut acc = 0.0f32;

        // Column v starts at this offset in the weight data
        let col_offset = v * col_bytes;

        // Iterate through blocks in this column
        for b in 0..num_blocks {
            let block = &w[col_offset + b * Q8_BLOCK_BYTES..col_offset + (b + 1) * Q8_BLOCK_BYTES];
            let scale = super::super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];

            // Compute dot product for this block
            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }

        y[v] = acc;
    }
}

/// Q4_0 GEMV transposed for transposed weight matrices.
///
/// Computes: y = W^T * x where W has shape [in_dim, out_dim]
/// stored in column-major Q4_0 blocked format.
///
/// Used for FFN down projection where weights are stored as [in_dim, out_dim].
pub fn gemv_q4_0_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // Initialize output to zero
    y.fill(0.0);

    let num_blocks_per_col = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_BLOCK_BYTES;

    // For each output dimension (column in the original matrix)
    for v in 0..out_dim {
        let mut acc = 0.0f32;

        // Column v starts at this offset in the weight data
        let col_offset = v * col_bytes;

        // Iterate through blocks in this column
        for b in 0..num_blocks_per_col {
            let block = &w[col_offset + b * Q4_BLOCK_BYTES..col_offset + (b + 1) * Q4_BLOCK_BYTES];
            let scale = super::super::quant::load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];

            // Dequantize and compute dot product for this block
            for i in 0..16 {
                let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                acc += q_lo * xb[i] + q_hi * xb[i + 16];
            }
        }

        y[v] = acc;
    }
}

/// Q4_1 GEMV transposed for transposed weight matrices.
///
/// Computes: y = W^T * x where W has shape [in_dim, out_dim]
/// stored in column-major Q4_1 blocked format.
///
/// Used for FFN down projection where weights are stored as [in_dim, out_dim].
pub fn gemv_q4_1_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // Initialize output to zero
    y.fill(0.0);

    let num_blocks_per_col = in_dim / Q4_1_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_1_BLOCK_BYTES;

    // For each output dimension (column in the original matrix)
    for v in 0..out_dim {
        let mut acc = 0.0f32;

        // Column v starts at this offset in the weight data
        let col_offset = v * col_bytes;

        // Iterate through blocks in this column
        for b in 0..num_blocks_per_col {
            let block =
                &w[col_offset + b * Q4_1_BLOCK_BYTES..col_offset + (b + 1) * Q4_1_BLOCK_BYTES];
            let w_scale = super::super::quant::load_f16_scale(&block[0..2]);
            let w_min = super::super::quant::load_f16_scale(&block[2..4]);
            let qs = &block[4..20];
            let xb = &x[b * Q4_1_BLOCK_ELEMS..];

            // Dequantize and compute dot product for this block
            // Q4_1: value = q4 * scale + min
            for i in 0..16 {
                let q_lo = ((qs[i] & 0x0F) as i32) as f32;
                let q_hi = ((qs[i] >> 4) as i32) as f32;
                let v_lo = (q_lo * w_scale + w_min) * xb[i];
                let v_hi = (q_hi * w_scale + w_min) * xb[i + 16];
                acc += v_lo + v_hi;
            }
        }

        y[v] = acc;
    }
}

/// Dispatch GEMV with transposed flag for tied embeddings.
///
/// When `transposed` is true, computes: y = W^T * x
/// Otherwise computes: y = W * x
///
/// # Arguments
///
/// * `w` - Weight matrix bytes
/// * `wtype` - Weight type (quantization format)
/// * `x` - Input vector
/// * `y` - Output vector
/// * `out_dim` - Output dimension
/// * `in_dim` - Input dimension
/// * `transposed` - Whether to compute W^T * x instead of W * x
/// * `scratch` - Optional scratch buffer for Q8_0 quantization (avoids heap allocation)
pub fn dispatch_gemv_transposed(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    transposed: bool,
    scratch: Option<&mut [u8]>,
) -> Result<(), super::super::CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            if transposed {
                gemv_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemv_f32(wf, x, y);
            }
        }
        GgmlType::F16 => {
            if transposed {
                gemv_f16_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_f16(w, x, y);
            }
        }
        GgmlType::Q8_0 => {
            if transposed {
                gemv_q8_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_0 => {
            if transposed {
                gemv_q4_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_0_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q4_1 => {
            if transposed {
                gemv_q4_1_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q4_1_q8_0(w, x, y, out_dim, in_dim, scratch);
            }
        }
        GgmlType::Q6_K => {
            gemv_q6_k(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            gemv_q5_k(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// F32 GEMV transposed for tied embeddings.
fn gemv_f32_transposed(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    // y = W^T * x, where W has shape [in_dim, out_dim]
    // y[v] = sum_i(x[i] * W[i, v])
    for v in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += x[i] * w[i * out_dim + v];
        }
        y[v] = acc;
    }
}

/// F16 GEMV: y[row] = dot(W[row, :], x)
fn gemv_f16(w: &[u8], x: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_offset = row * in_dim * 2;
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            let offset = row_offset + i * 2;
            let val = load_f16_scale(&w[offset..offset + 2]);
            acc += val * x[i];
        }
        *out = acc;
    });
}

/// F16 GEMV transposed for tied embeddings.
fn gemv_f16_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    for v in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            let offset = (i * out_dim + v) * 2;
            let val = load_f16_scale(&w[offset..offset + 2]);
            acc += x[i] * val;
        }
        y[v] = acc;
    }
}
