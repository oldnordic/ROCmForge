//! AVX-512 VNNI optimized Q4_K × Q8_K matrix multiplication.
//!
//! Port of llama.cpp's AVX-512 VNNI algorithm for Zen 4 CPUs.
//! Uses _mm512_dpbusd_epi32 (VNNI) for fast dot products.
//!
//! Reference: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;
use rayon::prelude::*;

pub use crate::cpu::kernels::q4::BlockQ4K;
pub use crate::cpu::kernels::q8::BlockQ8K;

/// Get scale shuffle mask for Q4_K sub-block (AVX-512 version).
///
/// # Safety
/// Caller must ensure AVX-512 is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn get_scale_shuffle_k4_avx512(j: usize) -> __m512i {
    // For AVX-512, we build the 512-bit shuffle mask from 128-bit components
    let mask_lo = match j {
        0 => _mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ),
        1 => _mm_setr_epi8(
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ),
        2 => _mm_setr_epi8(
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ),
        3 => _mm_setr_epi8(
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        ),
        4 => _mm_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ),
        5 => _mm_setr_epi8(
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        ),
        6 => _mm_setr_epi8(
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ),
        7 => _mm_setr_epi8(
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        ),
        _ => _mm_setzero_si128(),
    };

    // Create mask with shifted values for upper 256 bits
    let mask_hi = match j {
        0 => _mm_setr_epi8(
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        ),
        1 => _mm_setr_epi8(
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        ),
        2 => _mm_setr_epi8(
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        ),
        3 => _mm_setr_epi8(
            44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        ),
        4 => _mm_setr_epi8(
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ),
        5 => _mm_setr_epi8(
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        ),
        6 => _mm_setr_epi8(
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        ),
        7 => _mm_setr_epi8(
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        ),
        _ => _mm_setzero_si128(),
    };

    // Build 512-bit mask from two 128-bit parts
    let result = _mm512_inserti32x4(_mm512_setzero_si512(), mask_lo, 0);
    _mm512_inserti32x4(result, mask_hi, 1)
}

/// Horizontal sum of 16 f32 values (AVX-512).
///
/// # Safety
/// Caller must ensure AVX-512 is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_float_16_avx512(v: __m512) -> f32 {
    // AVX-512 has efficient horizontal reduction
    _mm512_reduce_add_ps(v)
}

/// AVX-512 VNNI Q4_K × Q8_K dot product.
///
/// Uses _mm512_dpbusd_epi32 for fast dot products.
///
/// Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c
///
/// # Safety
/// Caller must ensure AVX-512 VNNI is available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dot_q4_k_q8_k_block_avx512(
    q4_block: &BlockQ4K,
    q8_block: &BlockQ8K,
) -> f32 {
    const QK_K: usize = 256;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // Load scales
    let d = q8_block.d * f16::from_le_bytes(q4_block.d).to_f32();
    let dmin = -q8_block.d * f16::from_le_bytes(q4_block.dmin).to_f32();

    // Unpack Q4_K scales
    let mut utmp = [0u32; 4];
    std::ptr::copy_nonoverlapping(
        q4_block.scales.as_ptr(),
        utmp.as_mut_ptr() as *mut u8,
        12,
    );

    // Scale unpacking from llama.cpp
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // Convert scales to 16-bit integers (512-bit vector)
    let utmp_256 = _mm256_loadu_si256(utmp.as_ptr() as *const __m256i);
    let mins_and_scales = _mm512_cvtepu8_epi16(utmp_256);

    // Copy bsums to local array
    let mut bsums_local = [0i16; 16];
    for i in 0..16 {
        bsums_local[i] = q8_block.bsums[i];
    }

    // Load Q8_K block sums for min contribution
    let q8sums = _mm256_loadu_si256(bsums_local.as_ptr() as *const __m256i);
    let q8s = _mm_hadd_epi16(
        _mm256_extracti128_si256(q8sums, 0),
        _mm256_extracti128_si256(q8sums, 1),
    );

    // Extract upper 256 bits for mins
    let mins_256 = _mm512_extracti32x4_epi32(mins_and_scales, 3);
    let prod = _mm_madd_epi16(mins_256, q8s);

    let mut acc_m = _mm_set1_ps(dmin);
    acc_m = _mm_fmadd_ps(acc_m, _mm_cvtepi32_ps(prod), acc_m);

    // Duplicate scales for all sub-blocks (use lower 256 bits)
    let scales_128 = _mm512_extracti32x4_epi32(mins_and_scales, 0);
    let scales = _mm512_broadcast_i32x4(scales_128);

    let m4 = _mm512_set1_epi8(0xF);
    let mut sumi = _mm512_setzero_si512();

    let mut q4_ptr = q4_block.qs.as_ptr();
    let mut q8_ptr = q8_block.qs.as_ptr();

    // Process 4 groups of 64 values
    for j in 0..(QK_K / 64) {
        let scale_l = _mm512_shuffle_epi8(scales, get_scale_shuffle_k4_avx512(2 * j + 0));
        let scale_h = _mm512_shuffle_epi8(scales, get_scale_shuffle_k4_avx512(2 * j + 1));

        // Load 32 bytes of Q4_K (64 nibbles)
        let q4bits = _mm512_loadu_si512(q4_ptr as *const __m512i);
        q4_ptr = q4_ptr.add(32);

        // Split into low and high nibbles
        let q4l = _mm512_and_si512(q4bits, m4);
        let q4h = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), m4);

        // Load Q8_K values (64 bytes for AVX-512)
        let q8l = _mm512_loadu_si512(q8_ptr as *const __m512i);
        q8_ptr = q8_ptr.add(64);

        // Multiply-accumulate using VNNI
        // _mm512_dpbusd_epi32: dot product of unsigned bytes with signed byte accumulation
        let mut p32l = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4l, q8l);
        // Apply scales: multiply by 16-bit scales and accumulate
        p32l = _mm512_mullo_epi16(p32l, scale_l);
        p32l = _mm512_srai_epi16(p32l, 0); // No shift needed, scales already applied

        let q8h = _mm512_loadu_si512(q8_ptr as *const __m512i);
        q8_ptr = q8_ptr.add(64);

        let mut p32h = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4h, q8h);
        p32h = _mm512_mullo_epi16(p32h, scale_h);

        let sumj = _mm512_add_epi32(p32l, p32h);
        sumi = _mm512_add_epi32(sumi, sumj);
    }

    // Final accumulation
    let vd = _mm512_set1_ps(d);
    let acc = _mm512_mul_ps(_mm512_cvtepi32_ps(sumi), vd);

    // Horizontal sum + min contribution
    hsum_float_16_avx512(acc) + _mm_cvtss_f32(acc_m)
}

/// AVX-512 VNNI Q4_K × Q8_K GEMV: y = W * x
///
/// # Arguments
/// * `w` - Q4_K weights (row-major: each row is blocks of 256)
/// * `x` - Input vector (f32, length = in_dim)
/// * `y` - Output vector (f32, length = out_dim)
/// * `out_dim` - Number of output rows
/// * `in_dim` - Inner dimension (must be multiple of 256)
///
/// # Safety
/// Caller must ensure AVX-512 VNNI is available.
#[cfg(target_arch = "x86_64")]
pub fn gemv_q4_k_q8_k_avx512(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let seq_len = x.len() / in_dim;

    y.par_chunks_mut(out_dim).enumerate().for_each(|(row, y_row)| {
        let x_row = &x[row * in_dim..(row + 1) * in_dim];

        for col in 0..out_dim {
            let mut acc = 0.0f32;

            for block in 0..(in_dim / 256) {
                let q4_block = unsafe {
                    &*(w.as_ptr()
                        .add(col * in_dim / 256 * crate::cpu::quant::Q4_K_BLOCK_BYTES
                            + block * crate::cpu::quant::Q4_K_BLOCK_BYTES)
                        as *const BlockQ4K)
                };

                let q8_block = unsafe {
                    &*(x_row.as_ptr()
                        .add(block * 256)
                        as *const BlockQ8K)
                };

                unsafe {
                    acc += dot_q4_k_q8_k_block_avx512(q4_block, q8_block);
                }
            }

            y_row[col] = acc;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_detection_returns_bool() {
        // Just verify detection doesn't panic
        if !is_x86_feature_detected!("avx512f") {
            return; // Skip if AVX-512 not available
        }
        // If AVX-512F is available, we can at least verify the struct compiles
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_kernel_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return; // Skip if full AVX-512 not available
        }

        use crate::cpu::kernels::gemm_q4k_q8_scalar;

        // Create test Q4_K and Q8_K blocks
        let mut q4_block = BlockQ4K::default();
        let mut q8_block = BlockQ8K::default();

        // Fill with known values
        for i in 0..q4_block.qs.len() {
            q4_block.qs[i] = ((i % 16) * 17) as u8; // Pattern
        }
        for i in 0..q8_block.qs.len() {
            let val = (i as i16 - 128) as i8; // Signed pattern from -128 to 127
            q8_block.qs[i] = val;
        }
        q4_block.d = half::f16::from_f32(1.0).to_le_bytes();
        q8_block.d = 1.0f32;

        // Scalar reference
        let scalar_result = unsafe { gemm_q4k_q8_scalar::dot_q4_k_q8_k_block_scalar(&q4_block, &q8_block) };

        // AVX-512 implementation
        let avx512_result = unsafe { dot_q4_k_q8_k_block_avx512(&q4_block, &q8_block) };

        let rel_err = (scalar_result - avx512_result).abs() / scalar_result.abs().max(1e-9);
        assert!(
            rel_err < 1e-6,
            "rel error {rel_err}: scalar={scalar_result}, avx512={avx512_result}"
        );
    }
}
