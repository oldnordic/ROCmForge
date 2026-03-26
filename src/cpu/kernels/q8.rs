//! Q8_K quantization kernel implementation.
//!
//! Implements the Q8_K quantization format used by GGUF:
//! - 256 values per block (QK_K = 256)
//! - 4 bytes for delta scale (f32)
//! - 256 bytes for 8-bit quantized values (signed)
//! - 32 bytes for block sums in groups of 16 (16 × i16)
//! - Total: 292 bytes per block

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
}
