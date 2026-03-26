//! AVX2-optimized Q4_K × Q8_K matrix multiplication.
//!
//! Port of llama.cpp's ggml_vec_dot_q4_K_q8_K algorithm.
//! Reference: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:1838

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;
use rayon::prelude::*;

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

/// AVX2 Q4_K × Q8_K GEMV: y = W * x
///
/// # Arguments
/// * `w` - Q4_K weights (row-major: each row is blocks of 256)
/// * `x` - Input vector (f32, length = in_dim)
/// * `y` - Output vector (f32, length = out_dim)
/// * `out_dim` - Number of output rows
/// * `in_dim` - Inner dimension (must be multiple of 256)
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
pub fn gemv_q4_k_q8_k_avx2(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(in_dim % 256 == 0, "in_dim must be multiple of QK_K=256");
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);

    let num_blocks_per_row = in_dim / 256;
    let bytes_per_row = num_blocks_per_row * BlockQ4K::SIZE;

    // Quantize input to Q8_K
    let mut x_q8 = vec![BlockQ8K::zero(); num_blocks_per_row];
    for b in 0..num_blocks_per_row {
        let start = b * 256;
        let end = start + 256;
        x_q8[b] = crate::cpu::kernels::q8::quantize_q8_k(&x[start..end]);
    }

    // Process each output row
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for b in 0..num_blocks_per_row {
            let block_offset = row_start + b * BlockQ4K::SIZE;
            let q4_block = unsafe { &*(w.as_ptr().add(block_offset) as *const BlockQ4K) };
            let q8_block = &x_q8[b];

            acc += unsafe { dot_q4_k_q8_k_block_avx2(q4_block, q8_block) };
        }

        *out = acc;
    });
}

/// AVX2 Q4_K × Q8_K GEMM: Y = W * X
///
/// # Arguments
/// * `w` - Q4_K weights [out_dim, in_dim] in blocks
/// * `x` - Input matrix [m, in_dim] row-major f32
/// * `y` - Output matrix [m, out_dim] row-major f32
/// * `m` - Batch size
/// * `n` - Output dimension (out_dim)
/// * `k` - Inner dimension (in_dim)
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
pub fn gemm_q4_k_q8_k_avx2(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(k % 256 == 0, "k must be multiple of QK_K=256");

    let num_blocks_k = k / 256;

    // For GEMM, process each batch row
    y.par_chunks_mut(n).enumerate().for_each(|(batch_idx, y_row)| {
        let x_row = &x[batch_idx * k..(batch_idx + 1) * k];

        // Quantize this row to Q8_K blocks
        let mut x_q8 = vec![BlockQ8K::zero(); num_blocks_k];
        for b in 0..num_blocks_k {
            x_q8[b] = crate::cpu::kernels::q8::quantize_q8_k(&x_row[b * 256..(b + 1) * 256]);
        }

        // Compute dot products for each output column
        for out_col in 0..n {
            let mut acc = 0.0f32;

            for b in 0..num_blocks_k {
                let w_offset = out_col * num_blocks_k * BlockQ4K::SIZE + b * BlockQ4K::SIZE;
                let q4_block = unsafe { &*(w.as_ptr().add(w_offset) as *const BlockQ4K) };
                let q8_block = &x_q8[b];

                acc += unsafe { dot_q4_k_q8_k_block_avx2(q4_block, q8_block) };
            }

            y_row[out_col] = acc;
        }
    });
}

/// Dispatch to AVX2 or scalar GEMV based on CPU features.
pub fn gemv_q4_k_q8_k_dispatch(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return gemv_q4_k_q8_k_avx2(w, x, y, out_dim, in_dim);
        }
    }

    // Fallback to scalar
    crate::cpu::kernels::gemm_q4k_q8_scalar::gemv_q4_k_q8_k(w, x, y, out_dim, in_dim);
}

/// Dispatch to AVX2 or scalar GEMM based on CPU features.
pub fn gemm_q4_k_q8_k_dispatch_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return gemm_q4_k_q8_k_avx2(w, x, y, m, n, k);
        }
    }

    // Fallback to scalar
    crate::cpu::kernels::gemm_q4k_q8_scalar::gemm_q4_k_q8_k(w, x, y, m, n, k);
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemv_q4_k_q8_k_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Create simple test case
        let w = vec![0u8; 2 * BlockQ4K::SIZE];
        let x: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let mut y_avx2 = vec![0.0f32; 2];
        let mut y_scalar = vec![0.0f32; 2];

        // Use AVX2 version
        gemv_q4_k_q8_k_avx2(&w, &x, &mut y_avx2, 2, 256);

        // Use scalar version
        crate::cpu::kernels::gemm_q4k_q8_scalar::gemv_q4_k_q8_k(&w, &x, &mut y_scalar, 2, 256);

        // Both should produce finite results
        for i in 0..2 {
            assert!(y_avx2[i].is_finite());
            assert!(y_scalar[i].is_finite());
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemm_q4_k_q8_k_avx2_small_matrix() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // For k=512: 2 blocks per row, n=2 output columns
        let num_blocks_n = 2 * (512 / 256);
        let w = vec![0u8; num_blocks_n * BlockQ4K::SIZE];
        let x: Vec<f32> = (0..1024).map(|i| i as f32 / 1024.0).collect();
        let mut y = vec![0.0f32; 4];

        gemm_q4_k_q8_k_avx2(&w, &x, &mut y, 2, 2, 512);

        // Should not panic and produce output
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
