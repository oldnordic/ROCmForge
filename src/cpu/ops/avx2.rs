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

/// Multiply-sum Q4_0 × Q8_0 block using AVX-VNNI.
///
/// Computes `scale * sum((q4[i] - 8) * q8[i])` for a 32-element block.
/// `q4` contains signed nibbles (`nibble - 8`).  The VNNI `dpbusd`
/// instruction performs unsigned×signed products, so we add the `8`
/// offset back to `q4` and subtract `8 * scale * sum(q8)` from the
/// result.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[target_feature(enable = "avxvnni")]
unsafe fn dot_q4_0_q8_0_block_avx2_vnni(
    q4: std::arch::x86_64::__m256i,
    q8: &[u8],
    scale: f32,
) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(
        q8.len(),
        Q8_BLOCK_ELEMS,
        "dot_q4_0_q8_0_block_avx2_vnni: q8 must have 32 elements"
    );

    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);

    // Re-bias q4 to unsigned (0..15) for dpbusd.
    let q4_u = _mm256_add_epi8(q4, _mm256_set1_epi8(8));
    let zero = _mm256_setzero_si256();
    let dot32 = _mm256_dpbusd_avx_epi32(zero, q4_u, q8v);
    let dotf = _mm256_cvtepi32_ps(dot32);
    let scale_v = _mm256_set1_ps(scale);
    let unsigned_sum = hsum_avx2(_mm256_mul_ps(dotf, scale_v));

    // Compute signed sum of the Q8_0 bytes for the offset correction.
    let q8_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
    let q8_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1));
    let q8_sum16 = _mm256_add_epi16(q8_lo, q8_hi);
    let ones16 = _mm256_set1_epi16(1);
    let pair_sums = _mm256_madd_epi16(ones16, q8_sum16);
    let q8_sum = hsum_avx2(_mm256_cvtepi32_ps(pair_sums));

    unsigned_sum - 8.0 * scale * q8_sum
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
    if super::super::features::CpuFeatures::get().has_avxvnni {
        return dot_q4_0_q8_0_block_avx2_vnni(q4, q8, scale);
    }
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

    // Compute sum of Q8_0 values for min_offset correction.
    // q8_sum16[i] = q8[i] + q8[i+16]; sum all 16 lanes with madd/hsum.
    let q8v = _mm256_loadu_si256(q8.as_ptr() as *const __m256i);
    let q8_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q8v));
    let q8_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q8v, 1));
    let q8_sum16 = _mm256_add_epi16(q8_low, q8_high);
    let ones16 = _mm256_set1_epi16(1);
    let q8_sum32 = _mm256_madd_epi16(ones16, q8_sum16);
    let q8_sum = hsum_avx2(_mm256_cvtepi32_ps(q8_sum32)) as i32;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fake Q4_1 block and compare the AVX2 dot product against the
    /// scalar implementation and a direct f32 reference.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_1_q8_0_block_matches_scalar_and_reference() {
        use crate::cpu::quant::Q8_BLOCK_ELEMS;
        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        let w_scale = 0.1_f32;
        let w_min = -0.5_f32;

        // 16 weight bytes -> 32 nibbles
        let mut qs = [0u8; 16];
        for i in 0..16 {
            let lo: u8 = rng.u8(0..16);
            let hi: u8 = rng.u8(0..16);
            qs[i] = (hi << 4) | lo;
        }

        // Build an activation vector and quantize it to Q8_0 (one block).
        let mut x = [0.0_f32; Q8_BLOCK_ELEMS];
        for v in x.iter_mut() {
            *v = rng.f32() * 10.0 - 5.0;
        }
        let x_scale = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
        let mut q8 = [0u8; Q8_BLOCK_ELEMS];
        for (i, &v) in x.iter().enumerate() {
            let q = (v / x_scale).round().clamp(-127.0, 127.0) as i8;
            q8[i] = q as u8;
        }

        // Direct f32 reference.
        let mut expected = 0.0_f32;
        for i in 0..16 {
            let q_lo = (qs[i] & 0x0F) as f32;
            let q_hi = (qs[i] >> 4) as f32;
            let w_lo = w_scale * q_lo + w_min;
            let w_hi = w_scale * q_hi + w_min;
            expected += w_lo * x[i] + w_hi * x[i + 16];
        }

        let combined_scale = w_scale * x_scale;
        let min_offset = w_min * x_scale;
        let scalar = dot_q4_1_q8_0_block_scalar(&qs, &q8, combined_scale, min_offset);
        let avx2 = unsafe { dot_q4_1_q8_0_block_avx2(&qs, &q8, combined_scale, min_offset) };

        // Tolerate the ~0.5% error introduced by the Q8_0 activation
        // quantization; the important comparison is AVX2 vs scalar.
        let tol = (expected.abs() * 0.01).max(1e-3);
        assert!(
            (scalar - expected).abs() < tol,
            "scalar {} vs reference {} (diff {})",
            scalar,
            expected,
            (scalar - expected).abs()
        );
        assert!(
            (avx2 - expected).abs() < tol,
            "avx2 {} vs reference {} (diff {})",
            avx2,
            expected,
            (avx2 - expected).abs()
        );
        assert!(
            (avx2 - scalar).abs() < 1e-3,
            "avx2 {} vs scalar {} (diff {})",
            avx2,
            scalar,
            (avx2 - scalar).abs()
        );
    }

    /// Verify the AVX2 Q4_0 block dot against a scalar reference.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_0_f32_block_matches_reference() {
        use crate::cpu::quant::Q4_BLOCK_ELEMS;
        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        let scale = 0.05_f32;

        let mut qs = [0u8; 16];
        for i in 0..16 {
            let lo: u8 = rng.u8(0..16);
            let hi: u8 = rng.u8(0..16);
            qs[i] = (hi << 4) | lo;
        }

        let mut x = [0.0_f32; Q4_BLOCK_ELEMS];
        for v in x.iter_mut() {
            *v = rng.f32() * 4.0 - 2.0;
        }

        let mut expected = 0.0_f32;
        for i in 0..16 {
            let q_lo = (qs[i] & 0x0F) as i32 - 8;
            let q_hi = (qs[i] >> 4) as i32 - 8;
            expected += scale * (q_lo as f32) * x[i] + scale * (q_hi as f32) * x[i + 16];
        }

        let avx2 = unsafe { dot_q4_0_block_avx2(&qs, &x, scale) };
        assert!(
            (avx2 - expected).abs() < 1e-3,
            "avx2 {} vs reference {} (diff {})",
            avx2,
            expected,
            (avx2 - expected).abs()
        );
    }

    /// Verify the AVX2 Q4_0 x Q8_0 block dot against the scalar implementation.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn q4_0_q8_0_block_matches_scalar() {
        use crate::cpu::quant::Q8_BLOCK_ELEMS;
        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        let w_scale = 0.05_f32;

        let mut qs = [0u8; 16];
        for i in 0..16 {
            let lo: u8 = rng.u8(0..16);
            let hi: u8 = rng.u8(0..16);
            qs[i] = (hi << 4) | lo;
        }

        let mut x = [0.0_f32; Q8_BLOCK_ELEMS];
        for v in x.iter_mut() {
            *v = rng.f32() * 4.0 - 2.0;
        }
        let x_scale = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
        let mut q8 = [0u8; Q8_BLOCK_ELEMS];
        for (i, &v) in x.iter().enumerate() {
            let q = (v / x_scale).round().clamp(-127.0, 127.0) as i8;
            q8[i] = q as u8;
        }

        let combined = w_scale * x_scale;
        let scalar = dot_q4_0_q8_0_block_scalar(&qs, &q8, combined);
        let avx2 = unsafe { dot_q4_0_q8_0_block_avx2(&qs, &q8, combined) };
        assert!(
            (avx2 - scalar).abs() < 1e-3,
            "avx2 {} vs scalar {} (diff {})",
            avx2,
            scalar,
            (avx2 - scalar).abs()
        );
    }
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
