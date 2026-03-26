//! AVX2-optimized Q5_0 × Q8_0 matrix multiplication.
//!
//! Port of llama.cpp's ggml_vec_dot_q5_0_q8_0 algorithm.
//! Reference: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:762

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;
use rayon::prelude::*;

use crate::cpu::quant::{Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q8_BLOCK_BYTES, Q8_BLOCK_ELEMS};

/// Q5_0 block structure (22 bytes for 32 weights).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_0 {
    /// Scale (f16)
    pub d: [u8; 2],
    /// High bits (1 bit per value, 4 bytes for 32 values)
    pub qh: [u8; 4],
    /// Quantized weights, 4 bits each, 2 per byte (16 bytes)
    pub qs: [u8; 16],
}

impl BlockQ5_0 {
    pub const SIZE: usize = 2 + 4 + 16; // 22 bytes
    pub const N_WEIGHTS: usize = 32;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_nibbles_32(nibbles: &[u8; 16]) -> __m256i {
    // Convert 16 nibble bytes to 32 int8 bytes by expanding each nibble to a byte
    // Each nibble is stored with offset 8 (to make it signed)
    let mut result = [0i8; 32];

    for i in 0..16 {
        result[i * 2] = ((nibbles[i] & 0x0F) as i8).wrapping_sub(8);
        result[i * 2 + 1] = (((nibbles[i] >> 4) & 0x0F) as i8).wrapping_sub(8);
    }

    _mm256_loadu_si256(result.as_ptr() as *const __m256i)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bytes_from_bits_32(bits: &[u8; 4]) -> __m256i {
    // Expand 4 bytes to 32 bytes: each bit becomes a byte with value 0 or 16
    let mut result = [0i8; 32];

    for byte_idx in 0..4 {
        let byte = bits[byte_idx];
        for bit_idx in 0..8 {
            let val = if (byte >> bit_idx) & 1 == 1 { 16_i8 } else { 0 };
            result[byte_idx * 8 + bit_idx] = val;
        }
    }

    _mm256_loadu_si256(result.as_ptr() as *const __m256i)
}

/// Multiply and sum pairs of int8 values, returning f32.
///
/// This is equivalent to: sum_i (x[i] * y[i]) for i in 0..32
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_sum_i8_32_float(x: __m256i, y: __m256i) -> f32 {
    // Multiply 32 pairs of int8, get 32 int16 results (as two __m256i)
    let p0 = _mm256_maddubs_epi16(x, y);
    let p1 = _mm256_unpackhi_epi16(p0, _mm256_setzero_si256());
    let p0 = _mm256_unpacklo_epi16(p0, _mm256_setzero_si256());

    // Convert int16 to f32 and sum
    let f0 = _mm256_cvtepi32_ps(p0);
    let f1 = _mm256_cvtepi32_ps(p1);

    // Horizontal sum
    let sum = _mm256_add_ps(f0, f1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    tmp.iter().sum()
}

/// AVX2 Q5_0 × Q8_0 dot product.
///
/// Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c:762
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dot_q5_0_q8_0_block_avx2(
    q5_block: &BlockQ5_0,
    q8_block: &[u8], // Q8_0 format: 2 bytes scale + 32 bytes int8
) -> f32 {
    // Compute combined scale for the block
    let d = f16::from_le_bytes([q5_block.d[0], q5_block.d[1]]).to_f32();
    let d_y = f16::from_le_bytes([q8_block[0], q8_block[1]]).to_f32();
    let d = d * d_y;

    // Expand Q5_0 quantized values
    let mut qx = [0i8; 32];
    for i in 0..16 {
        let h0 = ((q5_block.qh[i >> 3] >> (i & 7)) & 1) << 4;
        let l0 = q5_block.qs[i] & 0x0F;
        qx[i * 2] = (((h0 | l0) as i8).wrapping_sub(16));

        let h1 = ((q5_block.qh[(i >> 3) + 2] >> (i & 7)) & 1) << 4;
        let l1 = (q5_block.qs[i] >> 4) & 0x0F;
        qx[i * 2 + 1] = (((h1 | l1) as i8).wrapping_sub(16));
    }

    let qx = _mm256_loadu_si256(qx.as_ptr() as *const __m256i);
    let qy = _mm256_loadu_si256(q8_block[2..].as_ptr() as *const __m256i);

    // Multiply-accumulate
    let q = mul_sum_i8_32_float(qx, qy);

    // Apply scale and return
    d * q
}

/// AVX2 Q5_0 × Q8_0 GEMV: y = W * x
///
/// # Arguments
/// * `w` - Q5_0 weights (row-major: each row is blocks of 32)
/// * `x` - Input vector (f32, length = in_dim)
/// * `y` - Output vector (f32, length = out_dim)
/// * `out_dim` - Number of output rows
/// * `in_dim` - Inner dimension (must be multiple of 32)
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
pub fn gemv_q5_0_q8_0_avx2(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(in_dim % 32 == 0, "in_dim must be multiple of 32 for Q5_0");
    assert!(is_x86_feature_detected!("avx2"), "AVX2 required");

    let num_blocks = in_dim / 32;
    let row_bytes = num_blocks * BlockQ5_0::SIZE;

    // Quantize input to Q8_0 once
    let mut x_q8 = vec![0u8; num_blocks * Q8_BLOCK_BYTES];
    for b in 0..num_blocks {
        let x_block = &x[b * 32..(b + 1) * 32];
        let q8_block = &mut x_q8[b * Q8_BLOCK_BYTES..(b + 1) * Q8_BLOCK_BYTES];
        quantize_q8_0_block(x_block, q8_block);
    }

    unsafe {
        y.par_iter_mut().enumerate().for_each(|(row, out)| {
            let mut acc = 0.0f32;

            for b in 0..num_blocks {
                let w_ptr = w.as_ptr().add(row * row_bytes + b * BlockQ5_0::SIZE) as *const BlockQ5_0;
                let q5_block = &*w_ptr;

                let x_q8_ptr = x_q8.as_ptr().add(b * Q8_BLOCK_BYTES);
                let q8_block = std::slice::from_raw_parts(x_q8_ptr, Q8_BLOCK_BYTES);

                acc += dot_q5_0_q8_0_block_avx2(q5_block, q8_block);
            }

            *out = acc;
        });
    }
}

/// Dispatch Q5_0 × Q8_0 GEMV with SIMD if available.
pub fn gemv_q5_0_q8_0_dispatch(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            gemv_q5_0_q8_0_avx2(w, x, y, out_dim, in_dim);
            return;
        }
    }

    // Fallback to scalar
    gemv_q5_0_q8_0_scalar(w, x, y, out_dim, in_dim);
}

/// Scalar fallback for Q5_0 × Q8_0 GEMV.
fn gemv_q5_0_q8_0_scalar(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks = in_dim / 32;
    let row_bytes = num_blocks * BlockQ5_0::SIZE;

    for o in 0..out_dim {
        let mut acc = 0.0f32;

        for b in 0..num_blocks {
            let block = &w[o * row_bytes + b * BlockQ5_0::SIZE..o * row_bytes + (b + 1) * BlockQ5_0::SIZE];
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let qh = &block[2..6];
            let qs = &block[6..22];
            let xb = &x[b * 32..];

            for i in 0..16 {
                let high_bit_0 = ((qh[i >> 3] >> (i & 7)) & 1) << 4;
                let low_bits_0 = qs[i] & 0x0F;
                let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                let high_bit_1 = ((qh[i >> 3 + 2] >> (i & 7)) & 1) << 4;
                let low_bits_1 = (qs[i] >> 4) & 0x0F;
                let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
            }
        }

        y[o] = acc;
    }
}

/// Quantize a single f32 vector block to Q8_0 format.
fn quantize_q8_0_block(x: &[f32], out: &mut [u8]) {
    const Q8_0_MAX: f32 = 127.0;
    assert_eq!(x.len(), 32);
    assert_eq!(out.len(), 34); // Q8_BLOCK_BYTES

    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if amax > 0.0 { amax / Q8_0_MAX } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let scale_bytes = f16::from_f32(scale).to_le_bytes();
    out[0] = scale_bytes[0];
    out[1] = scale_bytes[1];

    for i in 0..32 {
        let q = (x[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        out[2 + i] = q as u8;
    }
}
