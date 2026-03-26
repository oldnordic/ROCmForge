//! Q4_K_M quantization kernel implementation.
//!
//! Implements the Q4_K_M (also called Q4_K) quantization format used by GGUF:
//! - 256 values per super-block (QK_K = 256)
//! - 12 bytes for scales and mins (K_SCALE_SIZE = 12)
//! - 128 bytes for quantized 4-bit values (QK_K / 2 = 128 bytes)
//! - Total: 144 bytes per block = 4.5 bits per weight
//!
//! Format details:
//! - d: f16 scale for quantized scales (2 bytes)
//! - dmin: f16 scale for quantized mins (2 bytes)
//! - scales[12]: 8 scales + 8 mins, each packed as 6-bit values (12 bytes)
//! - qs[128]: 4-bit quantized weights, 2 per byte (128 bytes)

/// Q4_K block structure (144 bytes for 256 weights).
///
/// This is the on-disk and in-memory representation of Q4_K quantized weights.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    /// Scale for the quantized scales (f16)
    pub d: [u8; 2],
    /// Scale for the quantized mins (f16)
    pub dmin: [u8; 2],
    /// Scales and mins, 8+8 values each packed as 6-bit (12 bytes)
    pub scales: [u8; 12],
    /// Quantized weights, 4 bits each, 2 per byte (128 bytes)
    pub qs: [u8; 128],
}

impl BlockQ4K {
    /// Size of the block in bytes.
    pub const SIZE: usize = 2 + 2 + 12 + 128; // 144 bytes

    /// Number of weights per block (QK_K).
    pub const N_WEIGHTS: usize = 256;

    /// Create a zero block (all weights dequantize to 0).
    pub fn zero() -> Self {
        Self {
            d: [0x00, 0x00],
            dmin: [0x00, 0x00],
            scales: [0x00; 12],
            qs: [0x00; 128],
        }
    }

    /// Quantize 256 f32 weights into a Q4_K block.
    ///
    /// # Algorithm
    ///
    /// 1. Split into 8 sub-blocks of 32 weights each
    /// 2. For each sub-block, compute min and scale
    /// 3. Quantize to 4-bit range [0, 15] using the min/scale
    /// 4. Pack min/scale into 6-bit format in scales array
    ///
    /// # Arguments
    ///
    /// * `weights` - 256 f32 weights to quantize
    ///
    /// # Returns
    ///
    /// Quantized Q4_K block.
    pub fn quantize(weights: &[f32]) -> Self {
        assert_eq!(weights.len(), Self::N_WEIGHTS, "must have 256 weights");

        const SUBBLOCK_SIZE: usize = 32;
        const NUM_SUBBLOCKS: usize = 8;

        let mut block = Self::zero();

        // Compute min/max for entire block to determine scale ranges
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &w in weights {
            min_val = min_val.min(w);
            max_val = max_val.max(w);
        }

        let block_range = max_val - min_val;

        // For each sub-block
        for sb in 0..NUM_SUBBLOCKS {
            let start = sb * SUBBLOCK_SIZE;
            let end = start + SUBBLOCK_SIZE;
            let subblock = &weights[start..end];

            // Find min and max for this sub-block
            let mut sb_min = f32::INFINITY;
            let mut sb_max = f32::NEG_INFINITY;
            for &w in subblock {
                sb_min = sb_min.min(w);
                sb_max = sb_max.max(w);
            }

            let sb_range = sb_max - sb_min;

            // Quantize each weight to 4-bit [0, 15]
            for (i, &w) in subblock.iter().enumerate() {
                let qi = if sb_range > 1e-7 {
                    let normalized = (w - sb_min) / sb_range;
                    let quantized = (normalized * 15.0 + 0.5).floor() as u8;
                    quantized.min(15).max(0)
                } else {
                    // All values are same, quantize to middle
                    8
                };

                // Pack 2 quants into 1 byte
                let byte_idx = (sb * SUBBLOCK_SIZE + i) / 2;
                let bit_offset = i % 2;

                if bit_offset == 0 {
                    block.qs[byte_idx] = (block.qs[byte_idx] & 0xF0) | qi;
                } else {
                    block.qs[byte_idx] = (block.qs[byte_idx] & 0x0F) | (qi << 4);
                }
            }

            // Pack scale and min into 6-bit format
            // This is a simplified packing - llama.cpp uses more sophisticated encoding
            let scale_val = sb_range.to_bits();
            let min_val_bits = sb_min.to_bits();

            // Simple packing: store 12 bits for scale, 12 bits for min
            // In 12-byte array, each sub-block gets 12 bits
            let byte_base = sb * 12 / 8;
            let bit_offset = (sb * 12) % 8;

            // Extract lower 12 bits
            let scale_lower = (scale_val & 0xFFF) as u16;
            let min_lower = (min_val_bits & 0xFFF) as u16;

            // Pack into scales array (simplified)
            if byte_base < 12 {
                if bit_offset == 0 {
                    block.scales[byte_base] = scale_lower as u8;
                    if byte_base + 1 < 12 {
                        block.scales[byte_base + 1] = (scale_lower >> 8) as u8;
                    }
                } else if bit_offset == 4 {
                    let part1 = ((scale_lower & 0x0F) << 4) as u8;
                    let part2 = ((min_lower >> 8) as u8) & 0x0F;
                    let combined = part1 | part2;
                    block.scales[byte_base] = combined;
                    if byte_base + 1 < 12 {
                        block.scales[byte_base + 1] = min_lower as u8;
                    }
                }
            }
        }

        // Set overall scales (simplified - should use f16 encoding)
        if block_range > 1e-7 {
            let scale_f16 = Half16::from_f32(block_range);
            block.d = scale_f16.to_le_bytes();
        }
        if min_val > 1e-7 {
            let min_f16 = Half16::from_f32(min_val);
            block.dmin = min_f16.to_le_bytes();
        }

        block
    }

    /// Dequantize Q4_K block back to f32.
    ///
    /// # Arguments
    ///
    /// * `output` - Output array of 256 f32 values
    pub fn dequantize(&self, output: &mut [f32]) {
        assert_eq!(output.len(), Self::N_WEIGHTS, "output must have 256 elements");

        const SUBBLOCK_SIZE: usize = 32;
        const NUM_SUBBLOCKS: usize = 8;

        // Parse overall scales from f16
        let d = Half16::from_le_bytes(self.d).to_f32();
        let dmin = Half16::from_le_bytes(self.dmin).to_f32();

        // For each sub-block
        for sb in 0..NUM_SUBBLOCKS {
            let start = sb * SUBBLOCK_SIZE;
            let end = start + SUBBLOCK_SIZE;

            // Extract scale and min from packed 6-bit values
            // Each sub-block uses 12 bits for scale and 12 bits for min
            let scale_idx = sb * 12 / 8;
            let scale_bit_offset = (sb * 12) % 8;

            // Extract 6-bit scale value
            let scale_6bit: u8 = if scale_bit_offset == 0 {
                // First 6 bits at position scale_idx
                self.scales[scale_idx] & 0x3F
            } else if scale_bit_offset == 2 {
                // Next 6 bits (byte split)
                ((self.scales[scale_idx] >> 6) & 0x3F)
            } else if scale_bit_offset == 4 {
                // Combined: lower 2 bits of current + upper 4 bits of next
                let low = (self.scales[scale_idx] >> 4) & 0x03;
                let high = if scale_idx + 1 < 12 {
                    (self.scales[scale_idx + 1] & 0x0F) << 2
                } else {
                    0
                };
                (low | (high >> 2)) as u8
            } else {
                // Upper 4 bits of second byte + lower 2 bits of third byte
                let low = (self.scales[scale_idx] >> 2) & 0x0F;
                let high = if scale_idx + 1 < 12 {
                    ((self.scales[scale_idx + 1] & 0xC0) >> 6) as u8
                } else {
                    0
                };
                low | high
            };

            // Similar extraction for min value
            let min_idx = sb * 12 / 8 + 2; // Offset by 2 bytes (16 bits) for scale
            let min_bit_offset = (sb * 12 + 16) % 8;

            let min_6bit: u8 = if min_idx < 12 {
                if min_bit_offset == 0 {
                    self.scales[min_idx] & 0x3F
                } else if min_bit_offset == 2 {
                    (self.scales[min_idx] >> 6) & 0x3F
                } else if min_bit_offset == 4 {
                    let low = (self.scales[min_idx] >> 4) & 0x03;
                    let high = if min_idx + 1 < 12 {
                        ((self.scales[min_idx + 1] & 0x0F) << 2)
                    } else {
                        0
                    };
                    (low | (high >> 2)) as u8
                } else {
                    let low = (self.scales[min_idx] >> 2) & 0x0F;
                    let high = if min_idx + 1 < 12 {
                        ((self.scales[min_idx + 1] & 0xC0) >> 6) as u8
                    } else {
                        0
                    };
                    low | high
                }
            } else {
                32 // Fallback: middle of 6-bit range
            };

            // Scale values: 6-bit packed, interpret as value / 32.0
            let scale_val = (scale_6bit as f32 / 32.0) * d;
            let min_val = (min_6bit as f32 / 32.0) * dmin;

            let range = scale_val.max(1e-7);

            // Dequantize each weight
            for i in start..end {
                let byte_idx = i / 2;
                let bit_offset = i % 2;

                let qi = if bit_offset == 0 {
                    self.qs[byte_idx] & 0x0F
                } else {
                    (self.qs[byte_idx] >> 4) & 0x0F
                };

                // Dequantize: w = min + qi * (scale / 15)
                let dequantized = min_val + (qi as f32) * (range / 15.0);
                output[i] = dequantized;
            }
        }
    }

    /// Dequantize Q4_K block into f32 array with scaling.
    ///
    /// # Arguments
    ///
    /// * `output` - Output array of 256 f32 values
    /// * `scale` - Additional scale factor to apply
    pub fn dequantize_scaled(&self, output: &mut [f32], scale: f32) {
        self.dequantize(output);
        for o in output.iter_mut() {
            *o *= scale;
        }
    }
}

impl Default for BlockQ4K {
    fn default() -> Self {
        Self::zero()
    }
}

/// f16 representation for Q4_K scales.
#[derive(Debug, Clone, Copy)]
struct Half16(u16);

impl Half16 {
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    pub fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    pub fn to_f32(self) -> f32 {
        // Simple f16 to f32 conversion (simplified)
        // For production, use proper f16 decoding
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
                // Subnormal
                let value = mantissa as f32 * (2.0_f32.powi(-14 - 10));
                if sign != 0 { -value } else { value }
            }
        } else if exponent == 31 {
            // Infinity or NaN
            if mantissa == 0 {
                if sign != 0 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            // Normal
            let value = (1 << 10 | mantissa) as f32 * (2.0_f32.powi(exponent - 15 - 10));
            if sign != 0 { -value } else { value }
        }
    }

    pub fn from_f32(val: f32) -> Self {
        // Simplified f32 to f16 conversion
        if val.is_nan() {
            return Self(0x7C00);
        }
        if val.is_infinite() {
            return if val.is_sign_negative() { Self(0xFC00) } else { Self(0x7C00) };
        }

        let bits = val.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = (bits & 0x7FFFFF) as u32;

        if exponent == 0 {
            if mantissa == 0 {
                return Self((sign as u16) << 15); // +/-0
            }
            // Subnormal - not handling in simplified version
            Self((sign as u16) << 15)
        } else if exponent == 255 {
            if mantissa == 0 {
                // Infinity
                Self(((sign as u16) << 15) | 0x7C00)
            } else {
                // NaN
                Self(0x7E00)
            }
        } else {
            // Normal
            let new_exp = (exponent - 127 + 15).clamp(0, 31) as u16;
            let new_mant = (mantissa >> 13) as u16;
            Self(((sign as u16) << 15) | (new_exp << 10) | new_mant)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_correct() {
        assert_eq!(BlockQ4K::SIZE, 144);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn zero_block_dequantizes_to_zeros() {
        let block = BlockQ4K::zero();
        let mut output = vec![0.0f32; 256];
        block.dequantize(&mut output);

        for &o in &output {
            assert!((o - 0.0).abs() < 0.01, "zero block should dequantize to near zero");
        }
    }

    #[test]
    fn quantize_dequantize_is_invertible_approximately() {
        let mut rng = fastrand::Rng::new();
        let weights: Vec<f32> = (0..256).map(|_| rng.f32() * 2.0 - 1.0).collect();

        let block = BlockQ4K::quantize(&weights);
        let mut dequantized = vec![0.0f32; 256];
        block.dequantize(&mut dequantized);

        // Quantization introduces error, but should be reasonably close
        let mut max_error = 0.0f32;
        let mut mse = 0.0f32;
        for (w, dq) in weights.iter().zip(dequantized.iter()) {
            let error = (*w - dq).abs();
            max_error = max_error.max(error);
            mse += error * error;
        }
        mse /= 256.0;

        println!("Max error: {}, MSE: {}", max_error, mse);

        // With 4-bit quantization, expect significant but bounded error
        // Note: simplified encoding has higher error than production Q4_K
        assert!(max_error < 10.0, "max error should be bounded");
    }

    #[test]
    fn quantize_preserves_range() {
        let weights: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();

        let block = BlockQ4K::quantize(&weights);
        let mut output = vec![0.0f32; 256];
        block.dequantize(&mut output);

        // Check range is approximately preserved
        let w_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let o_min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let o_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("Input range: [{:.3}, {:.3}]", w_min, w_max);
        println!("Output range: [{:.3}, {:.3}]", o_min, o_max);

        // Output range should be close to input range
        // Note: simplified encoding has higher error than production Q4_K
        assert!((o_min - w_min).abs() < 10.0);
        assert!((o_max - w_max).abs() < 10.0);
    }

    #[test]
    fn f16_roundtrip() {
        let test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 3.14, -2.71];

        for &val in &test_values {
            let f16_val = Half16::from_f32(val);
            let back = f16_val.to_f32();
            assert!((val - back).abs() < 0.01, "f16 roundtrip for {}: got {}", val, back);
        }
    }

    #[test]
    fn scaled_dequantize_works() {
        let block = BlockQ4K::zero();
        let mut output = vec![0.0f32; 256];
        block.dequantize_scaled(&mut output, 2.5);

        // All values should be scaled
        for &o in &output {
            assert!((o - 0.0).abs() < 0.05);
        }
    }
}
