//! Q3_K quantization kernel implementation.
//!
//! Implements the Q3_K quantization format used by GGUF:
//! - 256 values per super-block (QK_K = 256)
//! - 32 bytes for high bit mask (hmask[QK_K/8] = hmask[32])
//! - 64 bytes for low 2-bit quantized values (qs[QK_K/4] = qs[64])
//! - 12 bytes for scales, quantized with 6 bits
//! - 2 bytes for super-block scale (fp16)
//! - Total: 110 bytes per block = ~3.4 bits per weight
//!
//! Format details:
//! - d: f16 super-block scale (2 bytes)
//! - scales[12]: 12 scales packed as 6-bit values (need unpacking)
//! - qs[64]: 2-bit quantized weights, 4 per byte (64 bytes)
//! - hmask[32]: high bit of 3-bit quantization (1 bit per element)
//!
//! Each weight is represented as 3 bits: low 2 bits from qs (with shift),
//! high bit from hmask. The 3-bit value is in range [-4, 3].

/// f16 representation for Q3_K scales.
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
                // Zero
                if sign != 0 {
                    -0.0
                } else {
                    0.0
                }
            } else {
                // Subnormal number
                let m = mantissa as f32 / (1u32 << 10) as f32;
                let e = -(1 << (10 - 1));
                if sign != 0 {
                    -m * 2f32.powi(e - 14)
                } else {
                    m * 2f32.powi(e - 14)
                }
            }
        } else if exponent == 31 {
            // Infinity or NaN
            if mantissa == 0 {
                if sign != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            // Normalized number
            let m = 1.0 + (mantissa as f32) / (1u32 << 10) as f32;
            let e = exponent - 15;
            if sign != 0 {
                -m * 2f32.powi(e)
            } else {
                m * 2f32.powi(e)
            }
        }
    }
}

/// Q3_K block structure (110 bytes for 256 weights).
///
/// This is the on-disk and in-memory representation of Q3_K quantized weights.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ3K {
    /// High bit mask for 3-bit quantization (1 bit per element)
    pub hmask: [u8; 32],
    /// Low 2 bits of quantization (4 elements per byte)
    pub qs: [u8; 64],
    /// Scales, quantized with 6 bits (12 bytes, packed format)
    pub scales: [u8; 12],
    /// Super-block scale (fp16)
    pub d: [u8; 2],
}

impl BlockQ3K {
    /// Size of the block in bytes.
    pub const SIZE: usize = 32 + 64 + 12 + 2; // 110 bytes

    /// Number of weights per block (QK_K).
    pub const N_WEIGHTS: usize = 256;

    /// Create a zero block (all weights dequantize to 0).
    pub fn zero() -> Self {
        Self {
            hmask: [0x00; 32],
            qs: [0x00; 64],
            scales: [0x00; 12],
            d: [0x00, 0x00],
        }
    }

    /// Dequantize the 256 weights in this block to f32.
    ///
    /// # Algorithm
    ///
    /// 1. Parse super-block scale d from fp16
    /// 2. Unpack the 12 scales from 6-bit packed format
    /// 3. For each group of 128 elements:
    ///    - Extract 3-bit quantized values (2 bits from qs + 1 bit from hmask)
    ///    - Apply corresponding scale and super-block scale
    ///
    /// # Arguments
    ///
    /// * `output` - Output array of 256 f32 values
    pub fn dequantize(&self, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            Self::N_WEIGHTS,
            "output must have 256 elements"
        );

        // Parse super-block scale from f16
        let d_all = Half16::from_le_bytes(self.d).to_f32();

        // Unpack the 12 scales (packed 6-bit format) into 16 scale values
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([
            self.scales[0],
            self.scales[1],
            self.scales[2],
            self.scales[3],
        ]);
        aux[1] = u32::from_le_bytes([
            self.scales[4],
            self.scales[5],
            self.scales[6],
            self.scales[7],
        ]);
        aux[2] = u32::from_le_bytes([
            self.scales[8],
            self.scales[9],
            self.scales[10],
            self.scales[11],
        ]);

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let mut scales = [0i8; 16];
        for i in 0..4 {
            let bytes = aux[i].to_le_bytes();
            for j in 0..4 {
                scales[i * 4 + j] = bytes[j] as i8;
            }
        }

        let mut y_idx = 0;
        let mut scale_idx = 0;
        let mut m = 1u8;

        // Process 256 elements as two 128-element chunks
        for chunk in 0..2 {
            let qs = &self.qs[chunk * 32..(chunk + 1) * 32];

            let mut shift = 0i32;

            // 4 groups of 32 elements each (4 * 32 = 128 elements)
            for _ in 0..4 {
                // First 16 elements of this 32-element group
                let dl = d_all * (scales[scale_idx] as f32 - 32.0);
                scale_idx += 1;

                for l in 0..16 {
                    let ql = (qs[l] >> shift) & 0x03;
                    let hbit = if self.hmask[l] & m != 0 { 0 } else { 4 };
                    output[y_idx + l] = dl * (ql as i8 - hbit) as f32;
                }

                // Next 16 elements of this 32-element group
                let dl = d_all * (scales[scale_idx] as f32 - 32.0);
                scale_idx += 1;

                for l in 0..16 {
                    let ql = (qs[l + 16] >> shift) & 0x03;
                    let hbit = if self.hmask[l + 16] & m != 0 { 0 } else { 4 };
                    output[y_idx + 16 + l] = dl * (ql as i8 - hbit) as f32;
                }

                shift += 2;
                m <<= 1;
                y_idx += 32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size() {
        assert_eq!(BlockQ3K::SIZE, 110);
        assert_eq!(BlockQ3K::N_WEIGHTS, 256);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
    }

    #[test]
    fn zero_block() {
        let block = BlockQ3K::zero();
        assert_eq!(block.hmask, [0x00; 32]);
        assert_eq!(block.qs, [0x00; 64]);
        assert_eq!(block.scales, [0x00; 12]);
        assert_eq!(block.d, [0x00, 0x00]);
    }

    #[test]
    fn dequantize_zero_block() {
        let block = BlockQ3K::zero();
        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Zero block with d=0 should produce all zeros
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn dequantize_symmetric_values() {
        let mut block = BlockQ3K::zero();

        // Set d = 1.0 (fp16)
        block.d = 0x3C00u16.to_le_bytes(); // fp16 1.0

        // Set scales such that they unpack to 33, which gives scale = 1 after offset (33 - 32 = 1)
        // scales[0..4] = 17 (0x11)
        // scales[4..8] = 17 (0x11)
        // scales[8..12] = 170 (0xAA)
        for i in 0..4 {
            block.scales[i] = 17;
            block.scales[i + 4] = 17;
            block.scales[i + 8] = 170;
        }

        // Set all qs to 0 (low 2 bits = 0)
        // Set all hmask to 0 (high bit = 1)
        // This gives q = 0 - 4 = -4
        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // All values should be -4.0
        for &val in &output {
            assert!((val - (-4.0)).abs() < 0.01);
        }
    }
}
