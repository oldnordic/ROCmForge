//! Q2_K quantization kernel implementation.
//!
//! Implements the Q2_K quantization format used by GGUF:
//! - 256 values per super-block (QK_K = 256)
//! - 16 bytes for packed scales + mins (4 bits each)
//! - 64 bytes for 2-bit quantized values (qs[QK_K/4])
//! - 2 bytes for super-block scale (fp16)
//! - 2 bytes for super-block min  (fp16)
//! - Total: 84 bytes per block = 2.625 bits per weight
//!
//! Format details (from llama.cpp ggml-common.h):
//!   block_q2_K {
//!       uint8_t scales[QK_K/16];          // 16 bytes: scales/mins, 4 bits each
//!       uint8_t qs[QK_K/4];                 // 64 bytes: 2-bit values
//!       ggml_half d;                        // super-block scale
//!       ggml_half dmin;                     // super-block min
//!   }
//!
//! Each weight is represented as x = dl * q - ml where:
//!   dl = d * (sc & 0xF)    // lower 4 bits of scales byte = quantized scale
//!   ml = dmin * (sc >> 4)  // upper 4 bits of scales byte = quantized min
//!   q  = (qs[l] >> shift) & 3  // 2-bit value (0..3)
//!
//! 16 sub-blocks of 16 elements each. 2 scale bytes per sub-block pair,
//! giving 16 scale/min pairs total.

use crate::cpu::quant::load_f16_scale;

/// Q2_K block structure (84 bytes for 256 weights).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ2K {
    /// Packed scales and mins, quantized with 4 bits (16 bytes).
    pub scales: [u8; 16],
    /// 2-bit quantized values, 4 values per byte (64 bytes).
    pub qs: [u8; 64],
    /// Super-block scale (fp16, 2 bytes).
    pub d: [u8; 2],
    /// Super-block min scale (fp16, 2 bytes).
    pub dmin: [u8; 2],
}

impl BlockQ2K {
    /// Size of the block in bytes.
    pub const SIZE: usize = 16 + 64 + 2 + 2; // 84 bytes

    /// Number of weights per block (QK_K).
    pub const N_WEIGHTS: usize = 256;

    /// Create a zero block (all weights dequantize to 0).
    pub fn zero() -> Self {
        Self {
            scales: [0x00; 16],
            qs: [0x00; 64],
            d: [0x00; 2],
            dmin: [0x00; 2],
        }
    }

    /// Dequantize the 256 weights in this block to f32.
    ///
    /// # Algorithm
    ///
    /// 1. Parse d and dmin from fp16.
    /// 2. Iterate over 2 chunks of 128 elements.
    /// 3. For each chunk, iterate shift 0,2,4,6 to extract 2-bit values.
    /// 4. Read 2 scale bytes per 32-element batch.
    /// 5. Dequantize: weight = d_all * scale_quant * q_val - dmin_all * min_quant
    ///
    /// # Arguments
    ///
    /// * `output` - Output array of 256 f32 values.
    pub fn dequantize(&self, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            Self::N_WEIGHTS,
            "output must have 256 elements"
        );

        let d_all = load_f16_scale(&self.d);
        let min_all = load_f16_scale(&self.dmin);

        let mut scale_idx = 0usize;
        let mut out_idx = 0usize;

        // qs is processed in two halves of 32 bytes each
        for q_chunk in 0..2 {
            let q_base = q_chunk * 32;

            // 4 shifts per chunk: 0, 2, 4, 6
            for shift in (0..8).step_by(2) {
                // Two scale bytes per shift iteration
                let sc0 = self.scales[scale_idx];
                let dl0 = d_all * (sc0 & 0x0F) as f32;
                let ml0 = min_all * (sc0 >> 4) as f32;
                scale_idx += 1;

                let sc1 = self.scales[scale_idx];
                let dl1 = d_all * (sc1 & 0x0F) as f32;
                let ml1 = min_all * (sc1 >> 4) as f32;
                scale_idx += 1;

                // First 16 elements (sc0)
                for l in 0..16 {
                    let q_val = ((self.qs[q_base + l] >> shift) & 0x03) as f32;
                    output[out_idx] = dl0 * q_val - ml0;
                    out_idx += 1;
                }

                // Next 16 elements (sc1)
                for l in 0..16 {
                    let q_val = ((self.qs[q_base + 16 + l] >> shift) & 0x03) as f32;
                    output[out_idx] = dl1 * q_val - ml1;
                    out_idx += 1;
                }
            }
        }

        debug_assert_eq!(out_idx, Self::N_WEIGHTS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size() {
        assert_eq!(BlockQ2K::SIZE, 84);
        assert_eq!(BlockQ2K::N_WEIGHTS, 256);
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
    }

    #[test]
    fn zero_block() {
        let block = BlockQ2K::zero();
        assert_eq!(block.scales, [0x00; 16]);
        assert_eq!(block.qs, [0x00; 64]);
        assert_eq!(block.d, [0x00; 2]);
        assert_eq!(block.dmin, [0x00; 2]);
    }

    #[test]
    fn dequantize_zero_block() {
        let block = BlockQ2K::zero();
        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // d=0, dmin=0 → all weights = 0
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }
}
