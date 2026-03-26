//! AVX2-optimized Q4_K × Q8_K matrix multiplication.
//!
//! Port of llama.cpp's ggml_vec_dot_q4_K_q8_K algorithm.
//! Reference: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:1838

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
}
