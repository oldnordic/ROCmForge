//! Q6_K quantization kernel implementation.
//!
//! Implements the Q6_K quantization format used by GGUF:
//! - 256 values per super-block (QK_K = 256)
//! - 128 bytes for quantized low 4-bit values (ql[QK_K/2] = ql[128])
//! - 64 bytes for high 2-bit values (qh[QK_K/4] = qh[64])
//! - 16 bytes for scales (int8)
//! - 2 bytes for super-block scale (fp16)
//! - Total: 210 bytes per block = 6 bits per weight
//!
//! Format details (from llama.cpp ggml-common.h):
//!   struct block_q6_K {
//!       uint8_t ql[QK_K/2];      // 128 bytes
//!       uint8_t qh[QK_K/4];      // 64 bytes
//!       int8_t  scales[QK_K/16]; // 16 bytes
//!       ggml_half d;             // 2 bytes
//!   };
//!
//! Each weight is 6 bits: low 4 bits from ql, high 2 bits from qh.
//! Value = d * scales[is] * (ql_low | (qh_bits << 4)) - 32

use crate::cpu::quant::load_f16_scale;

/// Q6_K block structure (210 bytes for 256 weights).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    /// Low 4-bit quantized values (128 bytes).
    pub ql: [u8; 128],
    /// High 2-bit quantized values (64 bytes).
    pub qh: [u8; 64],
    /// Per-sub-block scales, 16 int8 values (16 bytes).
    pub scales: [i8; 16],
    /// Super-block scale (fp16, 2 bytes).
    pub d: [u8; 2],
}

impl BlockQ6K {
    /// Size of the block in bytes.
    pub const SIZE: usize = 128 + 64 + 16 + 2; // 210 bytes

    /// Number of weights per block (QK_K).
    pub const N_WEIGHTS: usize = 256;

    /// Create a zero block.
    pub fn zero() -> Self {
        Self {
            ql: [0x00; 128],
            qh: [0x00; 64],
            scales: [0i8; 16],
            d: [0x00; 2],
        }
    }

    /// Dequantize the 256 weights in this block to f32.
    ///
    /// Follows llama.cpp `dequantize_row_q6_K` exactly.
    pub fn dequantize(&self, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            Self::N_WEIGHTS,
            "output must have 256 elements"
        );

        let d = load_f16_scale(&self.d);

        let mut out_idx = 0usize;
        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let sc_base = half * 8;

            for l in 0..32 {
                let is = l / 16;

                let sc1 = d * (self.scales[sc_base + is] as f32);
                let sc2 = d * (self.scales[sc_base + is + 2] as f32);
                let sc3 = d * (self.scales[sc_base + is + 4] as f32);
                let sc4 = d * (self.scales[sc_base + is + 6] as f32);

                let q1 = ((self.ql[ql_base + l] & 0xF) | ((self.qh[qh_base + l] & 3) << 4))
                    as i8 as f32
                    - 32.0;
                let q2 = ((self.ql[ql_base + l + 32] & 0xF)
                    | (((self.qh[qh_base + l] >> 2) & 3) << 4)) as i8
                    as f32
                    - 32.0;
                let q3 = (((self.ql[ql_base + l] >> 4) & 0xF)
                    | (((self.qh[qh_base + l] >> 4) & 3) << 4)) as i8
                    as f32
                    - 32.0;
                let q4 = (((self.ql[ql_base + l + 32] >> 4) & 0xF)
                    | (((self.qh[qh_base + l] >> 6) & 3) << 4)) as i8
                    as f32
                    - 32.0;

                output[out_idx + l] = sc1 * q1;
                output[out_idx + l + 32] = sc2 * q2;
                output[out_idx + l + 64] = sc3 * q3;
                output[out_idx + l + 96] = sc4 * q4;
            }
            out_idx += 128;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size() {
        assert_eq!(BlockQ6K::SIZE, 210);
        assert_eq!(BlockQ6K::N_WEIGHTS, 256);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
    }

    #[test]
    fn zero_block() {
        let block = BlockQ6K::zero();
        assert_eq!(block.ql, [0x00; 128]);
        assert_eq!(block.qh, [0x00; 64]);
        assert_eq!(block.scales, [0i8; 16]);
        assert_eq!(block.d, [0x00; 2]);
    }

    #[test]
    fn dequantize_zero_block() {
        let block = BlockQ6K::zero();
        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Zero block with d=0 → all weights = 0
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }
}
