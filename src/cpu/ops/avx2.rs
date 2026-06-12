//! AVX2 intrinsics and scalar dot-product helpers.

use crate::cpu::quant::Q8_BLOCK_ELEMS;

// ── AVX2 kernels ─────────────────────────────────────────────────────────────

/// Horizontal sum of __m256 register.
///
/// Folds 8-lane f32 vector to scalar sum using SSE instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Fold top 4 lanes into bottom 4
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    // Fold top 2 into bottom 2
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    // Fold top 1 into bottom 1
    let sum1 = _mm_add_ss(sum2, _mm_movehl_ps(sum2, sum2));
    _mm_cvtss_f32(sum1)
}

/// Unpack Q4_0 nibbles to i8 values in __m256i.
///
/// Input: 16 bytes, each containing 2 nibbles (32 values total).
/// Output: __m256i with 32 i8 values, each = nibble - 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_q4_0_nibbles_avx2(qs: &[u8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "unpack_q4_0_nibbles_avx2: qs must be 16 bytes"
    );

    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);
    let lo = _mm_and_si128(raw, lo_mask);
    let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);
    let q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    _mm256_sub_epi8(q4, _mm256_set1_epi8(8))
}

/// Multiply-sum Q4_0 × Q8_0 block (unscaled).
///
/// Computes sum(q4[i] * q8[i]) for 32-element blocks.
/// Returns __m256 with one i32 result per 8-element group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[target_feature(enable = "avxvnni")]
unsafe fn mul_sum_q4_0_q8_0_block_avx2_vnni(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "mul_sum_q4_0_q8_0_block_avx2_vnni: q8 must have 32 elements"
    );

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    // AVX2VNNI: compute dot product of signed i8 vectors
    // This does both multiply and horizontal sum in one instruction
    let zero = _mm256_setzero_si256();
    let dot32 = _mm256_dpwssd_avx_epi32(zero, q4, q8v);
    _mm256_cvtepi32_ps(dot32)
}

/// Multiply-sum Q4_0 × Q8_0 block (unscaled) without VNNI.
///
/// Computes sum(q4[i] * q8[i]) for 32-element blocks.
/// Returns __m256 with one i32 result per 8-element group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sum_q4_0_q8_0_block_avx2_unscaled(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "mul_sum_q4_0_q8_0_block_avx2_unscaled: q8 must have 32 elements"
    );

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let q4_abs = _mm256_sign_epi8(q4, q4);
    let q8_signed = _mm256_sign_epi8(q8v, q4);
    let dot16 = _mm256_maddubs_epi16(q4_abs, q8_signed);
    let ones = _mm256_set1_epi16(1);
    let dot32 = _mm256_madd_epi16(ones, dot16);
    _mm256_cvtepi32_ps(dot32)
}

/// AVX2+FMA dot product: sum(a[i] * b[i]) for f32 slices.
///
/// # Safety
/// `a` and `b` must have the same length, which must be a multiple of 8.
/// Caller must ensure AVX2+FMA are available (checked via is_x86_feature_detected!).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot_f32_avx2: a and b must have the same length"
    );
    debug_assert_eq!(n % 8, 0, "dot_f32_avx2: length must be multiple of 8");

    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    hsum_avx2(acc)
}

/// AVX2 Q4_0 block dot product — processes one 32-element block in 4 FMA ops.
///
/// # Safety
/// `qs` must be exactly 16 bytes. `xb` must be at least 32 floats.
/// Layout: qs[i] contains lo nibble (→ x[i]) and hi nibble (→ x[i+16])
/// Dequant: (nibble - 8) * scale
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_block_avx2(qs: &[u8], xb: &[f32], scale: f32) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(qs.len(), 16, "dot_q4_0_block_avx2: qs must be 16 bytes");
    debug_assert!(
        xb.len() >= 32,
        "dot_q4_0_block_avx2: xb must have at least 32 elements"
    );

    // Load 16 nibble bytes into a 128-bit register
    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);

    // Extract lo nibbles (bits 0..3): AND with 0x0F
    let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);
    let lo_bytes = _mm_and_si128(raw, lo_mask);

    // Extract hi nibbles (bits 4..7): shift right 4 then mask
    let hi_bytes = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);

    // Subtract 8 from each nibble (as i8) to get signed values -8..7
    let eight = _mm_set1_epi8(8i8);
    let lo_signed = _mm_sub_epi8(lo_bytes, eight);
    let hi_signed = _mm_sub_epi8(hi_bytes, eight);

    let scale_v = _mm256_set1_ps(scale);
    let mut acc = _mm256_setzero_ps();

    // lo nibbles 0..7 dot x[0..7] and lo nibbles 8..15 dot x[8..15]
    let lo_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_signed));
    let lo_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(lo_signed, 8)));
    let x0 = _mm256_loadu_ps(xb.as_ptr());
    let x1 = _mm256_loadu_ps(xb.as_ptr().add(8));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(lo_f0, scale_v), x0, acc);
    acc = _mm256_fmadd_ps(_mm256_mul_ps(lo_f1, scale_v), x1, acc);

    // hi nibbles 0..7 dot x[16..23] and hi nibbles 8..15 dot x[24..31]
    let hi_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_signed));
    let hi_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(hi_signed, 8)));
    let x2 = _mm256_loadu_ps(xb.as_ptr().add(16));
    let x3 = _mm256_loadu_ps(xb.as_ptr().add(24));
    acc = _mm256_fmadd_ps(_mm256_mul_ps(hi_f0, scale_v), x2, acc);
    acc = _mm256_fmadd_ps(_mm256_mul_ps(hi_f1, scale_v), x3, acc);

    hsum_avx2(acc)
}

/// AVX2 Q4_0 × Q8_0 block dot product — one 32-element block.
///
/// # Safety
/// Uses FMA accumulation for better performance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_q8_0_block_avx2(qs: &[u8], q8: &[u8], scale: f32) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_0_q8_0_block_avx2: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_0_q8_0_block_avx2: q8 must have 32 elements"
    );

    let q4 = unpack_q4_0_nibbles_avx2(qs);

    // Use cached CPU features to select implementation
    #[cfg(target_arch = "x86_64")]
    let dotf = if super::super::features::CpuFeatures::get().has_avxvnni {
        mul_sum_q4_0_q8_0_block_avx2_vnni(q4, q8)
    } else {
        mul_sum_q4_0_q8_0_block_avx2_unscaled(q4, q8)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let dotf = mul_sum_q4_0_q8_0_block_avx2_unscaled(q4, q8);
    let scaled = _mm256_mul_ps(dotf, _mm256_set1_ps(scale));
    hsum_avx2(scaled)
}

// ── Q4_1 × Q8_0 kernels ─────────────────────────────────────────────────────────────

/// Unpack Q4_1 nibbles to i8 values in __m256i.
///
/// Input: 16 bytes, each containing 2 nibbles (32 values total).
/// Output: __m256i with 32 i8 values, each = nibble (range 0-15, not centered).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_q4_1_nibbles_avx2(qs: &[u8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "unpack_q4_1_nibbles_avx2: qs must be 16 bytes"
    );

    let raw = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0F_u8 as i8);
    let lo = _mm_and_si128(raw, lo_mask);
    let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask);

    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
}

/// Multiply-sum Q4_1 × Q8_0 block (unscaled).
///
/// Computes sum(q4[i] * q8[i]) for 32-element blocks.
/// Returns __m256 with one i32 result per 8-element group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sum_q4_1_q8_0_block_avx2_unscaled(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let dot16 = _mm256_maddubs_epi16(q4, q8v); // q4 is unsigned (0-15)
    let ones = _mm256_set1_epi16(1);
    let dot32 = _mm256_madd_epi16(ones, dot16);
    _mm256_cvtepi32_ps(dot32)
}

/// AVX2 Q4_1 × Q8_0 block dot product — one 32-element block.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_1_q8_0_block_avx2(qs: &[u8], q8: &[u8], scale: f32, min_offset: f32) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_1_q8_0_block_avx2: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_1_q8_0_block_avx2: q8 must have 32 elements"
    );

    // Compute sum of Q8_0 values for min_offset correction
    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let q8_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
    let q8_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1));
    let q8_sum16 = _mm256_add_epi16(q8_low, q8_high);
    // Horizontal sum 16-bit values: pairwise, then to 32-bit, then final sum
    let q8_hadd = _mm256_hadd_epi16(q8_sum16, q8_sum16);
    let q8_hadd2 = _mm256_hadd_epi16(q8_hadd, q8_hadd);
    // Extract the result (only first two elements needed)
    let q8_sum =
        (_mm256_extract_epi16(q8_hadd2, 0) as i32) + (_mm256_extract_epi16(q8_hadd2, 4) as i32);

    let q4 = unpack_q4_1_nibbles_avx2(qs);
    let dotf = mul_sum_q4_1_q8_0_block_avx2_unscaled(q4, q8);
    let scaled = _mm256_mul_ps(dotf, _mm256_set1_ps(scale));
    hsum_avx2(scaled) + min_offset * (q8_sum as f32)
}

/// Scalar Q4_1 × Q8_0 block dot product — one 32-element block.
pub fn dot_q4_1_q8_0_block_scalar(qs: &[u8], q8: &[u8], scale: f32, min_offset: f32) -> f32 {
    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_1_q8_0_block_scalar: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_1_q8_0_block_scalar: q8 must have 32 elements"
    );
    let mut acc = 0i32;
    let mut q8_sum = 0i32;
    for i in 0..16 {
        let q_lo = (qs[i] & 0x0F) as i32; // 0 to 15
        let q_hi = (qs[i] >> 4) as i32;
        let x_lo = q8[i] as i8 as i32;
        let x_hi = q8[i + 16] as i8 as i32;
        acc += q_lo * x_lo + q_hi * x_hi;
        q8_sum += x_lo + x_hi;
    }
    // sum((q4 * w_scale + w_min) * q8 * x_scale)
    // = sum(q4 * q8) * w_scale * x_scale + w_min * x_scale * sum(q8)
    (acc as f32) * scale + min_offset * (q8_sum as f32)
}

/// Scalar Q4_0 × Q8_0 block dot product — one 32-element block.
pub fn dot_q4_0_q8_0_block_scalar(qs: &[u8], q8: &[u8], scale: f32) -> f32 {
    debug_assert_eq!(
        qs.len(),
        16,
        "dot_q4_0_q8_0_block_scalar: qs must be 16 bytes"
    );
    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_0_q8_0_block_scalar: q8 must have 32 elements"
    );
    let mut acc = 0i32;
    for i in 0..16 {
        let q_lo = (qs[i] & 0x0F) as i32 - 8;
        let q_hi = (qs[i] >> 4) as i32 - 8;
        let x_lo = q8[i] as i8 as i32;
        let x_hi = q8[i + 16] as i8 as i32;
        acc += q_lo * x_lo + q_hi * x_hi;
    }
    // Q4_0 is symmetric around 0, no min_offset needed
    (acc as f32) * scale
}
