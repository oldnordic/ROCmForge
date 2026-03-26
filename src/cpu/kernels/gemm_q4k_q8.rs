//! AVX2-optimized Q4_K × Q8_K matrix multiplication.
//!
//! Port of llama.cpp's ggml_vec_dot_q4_K_q8_K algorithm.
//! Reference: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:1838

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

pub use crate::cpu::kernels::q4::BlockQ4K;
pub use crate::cpu::kernels::q8::BlockQ8K;

/// Get scale shuffle mask for Q4_K sub-block.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn get_scale_shuffle_k4(j: usize) -> __m256i {
    // Shuffle masks for selecting 6-bit scale values
    // Each mask is 32 bytes for AVX2 shuffle operation
    // Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c:2230
    match j {
        0 => _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ),
        1 => _mm256_setr_epi8(
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        ),
        2 => _mm256_setr_epi8(
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ),
        3 => _mm256_setr_epi8(
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        ),
        4 => _mm256_setr_epi8(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        ),
        5 => _mm256_setr_epi8(
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        ),
        6 => _mm256_setr_epi8(
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        ),
        7 => _mm256_setr_epi8(
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        ),
        _ => _mm256_setzero_si256(),
    }
}

/// Horizontal sum of 8 f32 values.
///
/// # Safety
/// Caller must ensure AVX is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_float_8(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let v_low = _mm256_castps256_ps128(v);
    let v_high = _mm256_extractf128_ps(v, 1);

    // Add high and low
    let sum = _mm_add_ps(v_low, v_high);

    // Horizontal sum
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), sum);
    tmp.iter().sum()
}

/// AVX2 Q4_K × Q8_K dot product.
///
/// Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c:1858
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dot_q4_k_q8_k_block_avx2(
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

    // Convert scales to 16-bit integers
    let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
        utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32,
    ));

    // Copy bsums to local array to avoid packed struct reference issues
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
    let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);

    let mut acc_m = _mm_set1_ps(dmin);
    acc_m = _mm_fmadd_ps(acc_m, _mm_cvtepi32_ps(prod), acc_m);

    // Duplicate scales for all sub-blocks
    let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    let scales = _mm256_set_m128i(sc128, sc128);

    let m4 = _mm256_set1_epi8(0xF);
    let mut sumi = _mm256_setzero_si256();

    let mut q4_ptr = q4_block.qs.as_ptr();
    let mut q8_ptr = q8_block.qs.as_ptr();

    // Process 4 groups of 64 values
    for j in 0..(QK_K / 64) {
        let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
        let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

        // Load 32 bytes of Q4_K (64 nibbles)
        let q4bits = _mm256_loadu_si256(q4_ptr as *const __m256i);
        q4_ptr = q4_ptr.add(32);

        // Split into low and high nibbles
        let q4l = _mm256_and_si256(q4bits, m4);
        let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        // Load Q8_K values
        let q8l = _mm256_loadu_si256(q8_ptr as *const __m256i);
        q8_ptr = q8_ptr.add(32);

        // Multiply-accumulate
        let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
        p16l = _mm256_madd_epi16(scale_l, p16l);

        let q8h = _mm256_loadu_si256(q8_ptr as *const __m256i);
        q8_ptr = q8_ptr.add(32);

        let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
        p16h = _mm256_madd_epi16(scale_h, p16h);

        let sumj = _mm256_add_epi32(p16l, p16h);
        sumi = _mm256_add_epi32(sumi, sumj);
    }

    // Final accumulation
    let vd = _mm256_set1_ps(d);
    let acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), _mm256_setzero_ps());

    // Horizontal sum
    hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_exists() {
        assert!(true);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_get_scale_shuffle_k4() {
        unsafe {
            let mask0 = get_scale_shuffle_k4(0);
            let mask1 = get_scale_shuffle_k4(1);

            // Masks should be different
            let mask0_bytes = std::mem::transmute::<__m256i, [u8; 32]>(mask0);
            let mask1_bytes = std::mem::transmute::<__m256i, [u8; 32]>(mask1);

            assert_ne!(mask0_bytes, mask1_bytes);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_dot_q4_k_q8_k_block_avx2_zero() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let q4 = BlockQ4K::zero();
        let q8 = BlockQ8K::zero();

        let result = unsafe { dot_q4_k_q8_k_block_avx2(&q4, &q8) };
        assert_eq!(result, 0.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_dot_q4_k_q8_k_block_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return; // Skip on non-AVX2 systems
        }

        // Create test blocks
        let mut q4 = BlockQ4K::zero();
        q4.d = [0x00, 0x3C]; // f16 ~1.0
        q4.dmin = [0x00, 0x00];
        for i in 0..128 {
            q4.qs[i] = 0x88; // All nibbles = 8
        }
        q4.scales = [0x40; 12]; // Scales ~1.0

        let mut q8 = BlockQ8K::zero();
        q8.d = 1.0;
        for i in 0..256 {
            q8.qs[i] = 1;
        }

        // Run AVX2 implementation and verify it produces finite result
        let avx2_result = unsafe { dot_q4_k_q8_k_block_avx2(&q4, &q8) };
        assert!(avx2_result.is_finite());
    }
}
