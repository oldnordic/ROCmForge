//! Q5_K quantization kernel implementation.
//!
//! Implements the Q5_K quantization format used by GGUF:
//! - 256 values per super-block (QK_K = 256)
//! - 128 bytes for low 4-bit quantized values (ql[QK_K/2] = ql[128])
//! - 32 bytes for high bit (qh[QK_K/8] = qh[32])
//! - 12 bytes for scales, quantized with 6 bits
//! - 2 bytes for super-block scale (fp16)
//! - Total: 174 bytes per block = ~5.4 bits per weight
//!
//! Format details:
//! - d: f16 super-block scale (2 bytes)
//! - scales[12]: 12 scales packed as 6-bit values (need unpacking)
//! - ql[128]: low 4-bit quantized weights, 2 values per byte
//! - qh[32]: high bit of 5-bit quantization (1 bit per element)
//!
//! Each weight is represented as 5 bits: low 4 bits from ql, high bit from qh.
//! The 5-bit value is in range [-16, 15].

/// f16 representation for Q5_K scales.
#[derive(Debug, Clone, Copy)]
struct Half16(u16);

impl Half16 {
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    pub fn to_f32(self) -> f32 {
        // Simple f16 to f32 conversion
        let bits = self.0;
        let sign = ((bits >> 15) & 1) as i32;
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let mantissa = (bits & 0x3FF) as u32;

        if exponent == 0 {
            if mantissa == 0 {
                if sign != 0 { -0.0 } else { 0.0 }
            } else {
                let m = mantissa as f32 / (1u32 << 10) as f32;
                let e = -(1 << (10 - 1));
                if sign != 0 { -m * 2f32.powi(e - 14) } else { m * 2f32.powi(e - 14) }
            }
        } else if exponent == 31 {
            if mantissa == 0 {
                if sign != 0 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            let m = 1.0 + (mantissa as f32) / (1u32 << 10) as f32;
            let e = exponent - 15;
            if sign != 0 { -m * 2f32.powi(e) } else { m * 2f32.powi(e) }
        }
    }
}

/// Q5_K block structure (176 bytes for 256 weights).
///
/// This is the on-disk and in-memory representation of Q5_K quantized weights.
/// Matches llama.cpp block_q5_K structure.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    /// Super-block scale (fp16)
    pub d: [u8; 2],
    /// Super-block minimum scale (fp16)
    pub dmin: [u8; 2],
    /// Scales and mins, quantized with 6 bits (12 bytes, packed format)
    pub scales: [u8; 12],
    /// High bit mask for 5-bit quantization (1 bit per element)
    pub qh: [u8; 32],
    /// Low 4 bits of quantization (2 elements per byte)
    pub ql: [u8; 128],
}

impl BlockQ5K {
    /// Size of the block in bytes.
    pub const SIZE: usize = 2 + 2 + 12 + 32 + 128; // 176 bytes

    /// Number of weights per block (QK_K).
    pub const N_WEIGHTS: usize = 256;

    /// Create a zero block (all weights dequantize to 0).
    pub fn zero() -> Self {
        Self {
            d: [0x00, 0x00],
            dmin: [0x00, 0x00],
            scales: [0x00; 12],
            qh: [0x00; 32],
            ql: [0x00; 128],
        }
    }

    /// Dequantize the 256 weights in this block to f32.
    ///
    /// # Algorithm
    ///
    /// 1. Parse super-block scale d and dmin from fp16
    /// 2. Unpack the 8 scales and 8 mins from packed 6-bit format
    /// 3. For each group of 32 elements:
    ///    - Extract 5-bit quantized values (4 bits from ql + 1 bit from qh)
    ///    - Apply affine transformation: d * scale * q - dmin * min
    ///
    /// # Arguments
    ///
    /// * `output` - Output array of 256 f32 values
    pub fn dequantize(&self, output: &mut [f32]) {
        assert_eq!(output.len(), Self::N_WEIGHTS, "output must have 256 elements");

        // Parse super-block scales from f16
        let d = Half16::from_le_bytes(self.d).to_f32();
        let dmin = Half16::from_le_bytes(self.dmin).to_f32();

        // Unpack scales and mins from packed 6-bit format (get_scale_min_k4 pattern)
        let mut scales = [0i8; 8];
        let mut mins = [0i8; 8];
        for j in 0..8 {
            if j < 4 {
                scales[j] = (self.scales[j] & 63) as i8;
                mins[j] = (self.scales[j + 4] & 63) as i8;
            } else {
                scales[j] = ((self.scales[j + 4] & 0xF) | ((self.scales[j - 4] >> 6) << 4)) as i8;
                mins[j] = ((self.scales[j + 4] >> 4) | ((self.scales[j] >> 6) << 4)) as i8;
            }
        }

        let mut y_idx = 0;
        let mut scale_idx = 0;

        // Process 256 elements as 4 chunks of 64 elements each
        for chunk in 0..4 {
            let ql = &self.ql[chunk * 32..];
            let qh = &self.qh[chunk * 8..];
            let u1 = 1u8.wrapping_shl(2 * chunk as u32);
            let u2 = 1u8.wrapping_shl(2 * chunk as u32 + 1);

            // First 32 elements: d1 * q - m1
            let d1 = d * scales[scale_idx] as f32;
            let m1 = dmin * mins[scale_idx] as f32;
            scale_idx += 1;
            for l in 0..32 {
                let ql_bits = ql[l] & 0x0F;
                let hbit = if qh[l >> 3] & u1 != 0 { 16 } else { 0 };
                let q = (ql_bits + hbit) as f32;
                output[y_idx + l] = d1 * q - m1;
            }

            // Next 32 elements: d2 * q - m2
            let d2 = d * scales[scale_idx] as f32;
            let m2 = dmin * mins[scale_idx] as f32;
            scale_idx += 1;
            for l in 0..32 {
                let ql_bits = ql[l] >> 4;
                let hbit = if qh[l >> 3] & u2 != 0 { 16 } else { 0 };
                let q = (ql_bits + hbit) as f32;
                output[y_idx + 32 + l] = d2 * q - m2;
            }

            y_idx += 64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size() {
        assert_eq!(BlockQ5K::SIZE, 176);
        assert_eq!(BlockQ5K::N_WEIGHTS, 256);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
    }

    #[test]
    fn zero_block() {
        let block = BlockQ5K::zero();
        assert_eq!(block.ql, [0x00; 128]);
        assert_eq!(block.qh, [0x00; 32]);
        assert_eq!(block.scales, [0x00; 12]);
        assert_eq!(block.d, [0x00, 0x00]);
        assert_eq!(block.dmin, [0x00, 0x00]);
    }

    #[test]
    fn dequantize_zero_block() {
        let block = BlockQ5K::zero();
        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Zero block with d=0, dmin=0 should produce all zeros
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }
}
