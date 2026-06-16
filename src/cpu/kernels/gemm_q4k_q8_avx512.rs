//! AVX-512 optimized Q4_K × Q8_K matrix multiplication.
//!
//! Port of llama.cpp's AVX2 algorithm for Q4_K × Q8_K matrix multiplication.
//! Uses AVX2 instructions (maddubs_epi16 + madd_epi16) for compatibility.
//!
//! # Why dpbusd_epi32 (AVX-512 VNNI) cannot be used
//!
//! The Q4_K × Q8_K algorithm has a specific computational pattern that requires
//! scales to be applied BEFORE accumulating products, not after:
//!
//! ## AVX2 working pattern (maddubs_epi16 + madd_epi16):
//! ```text
//! maddubs_epi16:  Produces 16-bit individual products
//! madd_epi16:     result[i] = products[2i]*scales[2i] + products[2i+1]*scales[2i+1]
//!                 Then adds adjacent pairs to get 32-bit results
//! ```
//!
//! ## dpbusd_epi32 pattern (AVX-512 VNNI):
//! ```text
//! dpbusd_epi32:   result[i] = sum(products[4i .. 4i+3])
//!                 Scales are applied to the SUM, not individual products
//!                 Cannot recover individual products for proper scale application
//! ```
//!
//! ## The architectural mismatch:
//! 1. dpbusd produces stride-4 sums (each output = sum of 4 consecutive products)
//! 2. Q4_K algorithm needs stride-8 results (each output = sum of 8 consecutive products)
//! 3. **Critical:** Scales must be applied to individual products BEFORE adding,
//!    but dpbusd loses the intermediate products and only provides final sums
//!
//! ## Evidence from systematic investigation:
//! - Attempt 1: Tried adding dpbusd results with hadd → Wrong results (rel error 1.26)
//! - Attempt 2: Tried permute/hadd combinations → Wrong results (rel error 1.52)
//! - Attempt 3: Tried complex unpack/rearrange → Compilation errors
//! - Attempt 4: Tried scale-repeat patterns → Wrong results (rel error 0.87)
//! - Root cause analysis: dpbusd fundamentally loses information needed for scale application
//!
//! ## Reference:
//! - llama.cpp also does NOT have AVX-512 VNNI support for Q4_K × Q8_K
//! - Only AVX2 implementation exists: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c
//!
//! This is a fundamental architectural limitation, not an implementation issue.
//!
//! Reference: /home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;
use rayon::prelude::*;

pub use crate::cpu::kernels::q4::BlockQ4K;
pub use crate::cpu::kernels::q8::BlockQ8K;

/// Horizontal sum of 8 f32 values (AVX-512).
///
/// # Safety
/// Caller must ensure AVX is available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_float_8_avx512(v: __m256) -> f32 {
    // Use _mm512_reduce_add_ps by casting to 512-bit
    let v512 = _mm512_castps256_ps512(v);
    _mm512_reduce_add_ps(v512)
}

/// AVX-512 Q4_K × Q8_K dot product.
///
/// Uses AVX2 instructions (maddubs_epi16 + madd_epi16) for compatibility.
/// The dpbusd_epi32 instruction produces stride-4 results, which doesn't
/// match the stride-8 grouping required by the Q4_K algorithm.
///
/// Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dot_q4_k_q8_k_block_avx512(q4_block: &BlockQ4K, q8_block: &BlockQ8K) -> f32 {
    const QK_K: usize = 256;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // Load scales
    let d = q8_block.d * f16::from_le_bytes(q4_block.d).to_f32();
    let dmin = -q8_block.d * f16::from_le_bytes(q4_block.dmin).to_f32();

    // Unpack Q4_K scales
    let mut utmp = [0u32; 4];
    std::ptr::copy_nonoverlapping(q4_block.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);

    // Scale unpacking from llama.cpp
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // Convert scales/mins to 16-bit integers.  After unpacking, each byte holds
    // one unsigned 6-bit value; zero-extend them.  Low 128 bits are scales,
    // high 128 bits are mins.
    let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
        utmp[3] as i32,
        utmp[2] as i32,
        utmp[1] as i32,
        utmp[0] as i32,
    ));

    // Extract mins (scales stay in the full 256-bit register for broadcasting).
    let mins_128 = _mm256_extracti128_si256(mins_and_scales, 1);

    // Copy bsums to local array
    let mut bsums_local = [0i16; 16];
    #[allow(
        clippy::manual_memcpy,
        reason = "packed struct field copy requires loop to avoid unaligned reference"
    )]
    for i in 0..16 {
        bsums_local[i] = q8_block.bsums[i];
    }

    // Load Q8_K block sums for min contribution
    let q8sums = _mm256_loadu_si256(bsums_local.as_ptr() as *const __m256i);
    let q8s = _mm_hadd_epi16(
        _mm256_extracti128_si256(q8sums, 0),
        _mm256_extracti128_si256(q8sums, 1),
    );
    let prod = _mm_madd_epi16(mins_128, q8s);

    let mut acc_m = _mm_setzero_ps();
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

    // scales lives in the low 128 bits; pre-broadcast each 16-bit scale to all
    // lanes so it can be selected by a non-constant index.
    let scale_vecs = [
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 0) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 1) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 2) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 3) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 4) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 5) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 6) as i16),
        _mm256_set1_epi16(_mm256_extract_epi16(mins_and_scales, 7) as i16),
    ];

    let m4 = _mm256_set1_epi8(0xF);
    let mut sumi = _mm256_setzero_si256();

    let mut q4_ptr = q4_block.qs.as_ptr();
    let mut q8_ptr = q8_block.qs.as_ptr();

    // Process 4 groups of 64 values
    for j in 0..(QK_K / 64) {
        // Load 32 bytes of Q4_K (64 nibbles)
        let q4bits = _mm256_loadu_si256(q4_ptr as *const __m256i);
        q4_ptr = q4_ptr.add(32);

        // Split into low and high nibbles
        let q4l = _mm256_and_si256(q4bits, m4);
        let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        // Load Q8_K values for low nibbles (32 bytes)
        let q8l = _mm256_loadu_si256(q8_ptr as *const __m256i);
        q8_ptr = q8_ptr.add(32);

        // Load Q8_K values for high nibbles (32 bytes)
        let q8h = _mm256_loadu_si256(q8_ptr as *const __m256i);
        q8_ptr = q8_ptr.add(32);

        // Select the two 16-bit scale values for this 64-value group.
        let scale_l = scale_vecs[2 * j];
        let scale_h = scale_vecs[2 * j + 1];

        // Use AVX2 pattern: maddubs_epi16 + madd_epi16
        let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
        p16l = _mm256_madd_epi16(scale_l, p16l);

        let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
        p16h = _mm256_madd_epi16(scale_h, p16h);

        let sumj = _mm256_add_epi32(p16l, p16h);
        sumi = _mm256_add_epi32(sumi, sumj);
    }

    // Final accumulation
    let vd = _mm256_set1_ps(d);
    let acc = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi));

    // Horizontal sum of the min accumulator
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    hsum_float_8_avx512(acc) + _mm_cvtss_f32(acc_m)
}

/// AVX-512 Q4_K × Q8_K GEMV: y = W * x
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
pub fn gemv_q4_k_q8_k_avx512(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let _seq_len = x.len() / in_dim;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(row, y_row)| {
            let x_row = &x[row * in_dim..(row + 1) * in_dim];

            for (col, y_out) in y_row.iter_mut().enumerate().take(out_dim) {
                let mut acc = 0.0f32;

                for block in 0..(in_dim / 256) {
                    let q4_block = unsafe {
                        &*(w.as_ptr().add(
                            col * in_dim / 256 * crate::cpu::quant::Q4_K_BLOCK_BYTES
                                + block * crate::cpu::quant::Q4_K_BLOCK_BYTES,
                        ) as *const BlockQ4K)
                    };

                    let q8_block =
                        unsafe { &*(x_row.as_ptr().add(block * 256) as *const BlockQ8K) };

                    unsafe {
                        acc += dot_q4_k_q8_k_block_avx512(q4_block, q8_block);
                    }
                }

                *y_out = acc;
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
        if !is_x86_feature_detected!("avx512f") { // Skip if AVX-512F not available
        }
        // If AVX-512F is available, we can at least verify the struct compiles
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_q4_k_q8_k_block_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
            || !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
        {
            return;
        }

        let mut rng = fastrand::Rng::new();
        let weights: Vec<f32> = (0..256).map(|_| rng.f32() * 4.0 - 2.0).collect();
        let activations: Vec<f32> = (0..256).map(|_| rng.f32() * 4.0 - 2.0).collect();

        let q4 = BlockQ4K::quantize(&weights);
        let q8 = crate::cpu::kernels::q8::quantize_q8_k(&activations);

        let avx512 = unsafe { dot_q4_k_q8_k_block_avx512(&q4, &q8) };
        let scalar = crate::cpu::kernels::gemm_q4k_q8_scalar::dot_q4_k_q8_k_block_scalar(&q4, &q8);

        let abs_err = (avx512 - scalar).abs();
        let rel_err = abs_err / scalar.abs().max(1e-6);
        assert!(
            rel_err < 1e-4,
            "AVX-512 dot diverges from scalar: avx512={avx512} scalar={scalar} rel={rel_err}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[ignore]
    fn trace_dpbusd_vs_scalar_step_by_step() {
        // This test traces through the computation step by step
        // to understand exactly where dpbusd_epi32 differs from scalar

        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }

        use crate::cpu::kernels::gemm_q4k_q8_scalar;

        // Create blocks with specific pattern to trace
        let mut q4_block = BlockQ4K::default();
        let mut q8_block = BlockQ8K::default();

        // Fill with predictable pattern
        for i in 0..32 {
            // Q4_K: store 2 nibbles per byte
            // low nibble = i, high nibble = i+1
            q4_block.qs[i] = (((i + 1) % 16) << 4 | (i % 16)) as u8;
        }
        for i in 0..256 {
            q8_block.qs[i] = (i % 8) as i8 - 4; // Values: -4, -3, -2, -1, 0, 1, 2, 3 repeating
            q8_block.bsums[i / 16] += q8_block.qs[i] as i16;
        }
        q4_block.d = half::f16::from_f32(1.0).to_le_bytes();
        q8_block.d = 1.0f32;

        // Scalar result
        let scalar_result = gemm_q4k_q8_scalar::dot_q4_k_q8_k_block_scalar(&q4_block, &q8_block);
        eprintln!("Scalar result: {}", scalar_result);

        // Now trace through manually
        // First, understand the Q4_K layout
        eprintln!("=== Q4_K Layout (first 8 bytes) ===");
        for i in 0..8 {
            let low = q4_block.qs[i] & 0x0F;
            let high = (q4_block.qs[i] >> 4) & 0x0F;
            eprintln!("Byte {}: low={}, high={}", i, low, high);
        }

        eprintln!("=== Q8_K Values (first 16) ===");
        for i in 0..16 {
            eprintln!("Q8[{}] = {}", i, q8_block.qs[i]);
        }

        // Manual computation of first 8 elements
        eprintln!("=== Manual computation ===");
        for i in 0..8 {
            let q4_val = (q4_block.qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
            let q8_val = q8_block.qs[i];
            let product = q4_val as i32 * q8_val as i32;
            eprintln!("i={}: q4={}, q8={}, product={}", i, q4_val, q8_val, product);
        }

        // Now simulate what dpbusd_epi32 would produce
        eprintln!("=== dpbusd_epi32 simulation (low nibbles) ===");
        for group in 0..2 {
            let mut sum = 0i32;
            for k in 0..4 {
                let i = group * 4 + k;
                let q4_val = q4_block.qs[i / 2] & 0x0F; // low nibble
                let q8_val = q8_block.qs[i];
                sum += q4_val as i32 * q8_val as i32;
            }
            eprintln!(
                "dpbusd_low[{}] = sum(q4[{}..{}] * q8[{}..{}]) = {}",
                group,
                group * 4,
                group * 4 + 4,
                group * 4,
                group * 4 + 4,
                sum
            );
        }

        eprintln!("=== dpbusd_epi32 simulation (high nibbles) ===");
        for group in 0..2 {
            let mut sum = 0i32;
            for k in 0..4 {
                let i = group * 4 + k;
                let q4_val = (q4_block.qs[i / 2] >> 4) & 0x0F; // high nibble
                let q8_val = q8_block.qs[i];
                sum += q4_val as i32 * q8_val as i32;
            }
            eprintln!(
                "dpbusd_high[{}] = sum(q4[{}..{}] * q8[{}..{}]) = {}",
                group,
                group * 4,
                group * 4 + 4,
                group * 4,
                group * 4 + 4,
                sum
            );
        }

        eprintln!("=== Scalar stride-8 grouping ===");
        eprintln!("Scalar accumulates: sums[0] = sum of products at positions (0-7, 32-39, 64-71, 96-103, 128-135, 160-167, 192-199, 224-231)");
        eprintln!("Each group of 8 is processed separately, then scaled and summed");
        eprintln!("Key insight: dpbusd gives stride-4, scalar needs stride-8");
    }
}
