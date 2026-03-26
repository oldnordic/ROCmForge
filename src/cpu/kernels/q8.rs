//! Q8_K quantization kernel implementation.
//!
//! Implements the Q8_K quantization format used by GGUF:
//! - 256 values per block (QK_K = 256)
//! - 4 bytes for delta scale (f32)
//! - 256 bytes for 8-bit quantized values (signed)
//! - 32 bytes for block sums in groups of 16 (16 × i16)
//! - Total: 292 bytes per block

/// Quantize 256 f32 values to Q8_K format.
///
/// # Algorithm
/// 1. Find max absolute value
/// 2. Compute scale: d = max_abs / 127.0
/// 3. Quantize: qs[i] = round(f32[i] / d)
/// 4. Compute bsums: bsums[j] = sum(qs[j*16 .. (j+1)*16])
///
/// Reference: llama.cpp ggml/src/ggml-cpu/ops.c:quantize_row_q8_K
pub fn quantize_q8_k(values: &[f32]) -> BlockQ8K {
    assert_eq!(values.len(), 256, "Q8_K requires exactly 256 values");

    // Find max absolute value
    let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    // Compute scale (avoid division by zero)
    let d = if max_abs > 1e-7 { max_abs / 127.0 } else { 1.0 };

    let mut qs = [0i8; 256];
    let mut bsums = [0i16; 16];

    // Quantize and compute block sums
    for i in 0..256 {
        let q = (values[i] / d).round() as i16;
        let q_clamped = q.clamp(-127, 127) as i8;
        qs[i] = q_clamped;
        bsums[i / 16] += q_clamped as i16;
    }

    BlockQ8K { d, qs, bsums }
}

/// Q8_K block for intermediate dot product computation.
///
/// This is NOT for storage - it's for efficient accumulation during
/// Q4_K × Q8_K matrix multiplication. The bsums field enables fast
/// min calculation in the dot product.
///
/// Size: 4 + 256 + 32 = 292 bytes (packed to match llama.cpp layout)
/// Reference: llama.cpp ggml/src/ggml-common.h:329
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockQ8K {
    /// Delta scale (f32)
    pub d: f32,
    /// 8-bit quantized values (signed, QK_K = 256 elements)
    pub qs: [i8; 256],
    /// Sum of quants in groups of 16 (16 int16_t values)
    pub bsums: [i16; 16],
}

impl BlockQ8K {
    /// Size of the block in bytes
    pub const SIZE: usize = 4 + 256 + 32; // 292 bytes

    /// Number of elements per block (QK_K)
    pub const N_ELEMS: usize = 256;

    /// Create a zero block (all values dequantize to 0).
    pub fn zero() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }
    }

    /// Dequantize Q8_K block back to f32.
    ///
    /// Used for testing and verification.
    pub fn dequantize(&self, output: &mut [f32]) {
        assert_eq!(output.len(), 256);

        for i in 0..256 {
            output[i] = self.qs[i] as f32 * self.d;
        }
    }
}

impl Default for BlockQ8K {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_q8k_size_is_correct() {
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), BlockQ8K::SIZE);
    }

    #[test]
    fn block_q8k_alignment() {
        // Packed struct has alignment of 1 (minimum), but we'll ensure
        // allocations are 16-byte aligned for AVX2 loads
        assert_eq!(std::mem::align_of::<BlockQ8K>(), 1);
    }

    #[test]
    fn quantize_q8_k_zero_input() {
        let values = [0.0f32; 256];
        let block = quantize_q8_k(&values);

        // Zero input should give zero quants
        assert_eq!(block.qs, [0i8; 256]);
        // Compare bsums element by element (packed struct - copy value first)
        for i in 0..16 {
            let bsum_value = block.bsums[i];
            assert_eq!(bsum_value, 0);
        }
    }

    #[test]
    fn quantize_q8_k_range() {
        let values: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let block = quantize_q8_k(&values);

        // Scale should be approximately 255/127
        assert!(block.d > 1.5 && block.d < 2.5);

        // All quants should be in range
        for &q in &block.qs {
            assert!(q >= -127 && q <= 127);
        }
    }

    #[test]
    fn quantize_q8_k_bsums_correctness() {
        let values: Vec<f32> = (0..256).map(|i| i as f32 - 128.0).collect();
        let block = quantize_q8_k(&values);

        // Verify bsums match actual sums
        for i in 0..16 {
            let expected: i16 = block.qs[i * 16..(i + 1) * 16].iter().map(|&x| x as i16).sum();
            let bsum_value = block.bsums[i];
            assert_eq!(bsum_value, expected);
        }
    }

    #[test]
    fn quantize_q8_k_roundtrip() {
        let mut rng = fastrand::Rng::new();
        let values: Vec<f32> = (0..256).map(|_| rng.f32() * 200.0 - 100.0).collect();
        let original = values.clone();

        let block = quantize_q8_k(&values);
        let mut reconstructed = vec![0.0f32; 256];
        block.dequantize(&mut reconstructed);

        // Check reconstruction error
        let max_error = original.iter().zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Should be reasonably close
        assert!(max_error < block.d * 2.0);
    }
}
