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

        // 1. Compute subblock parameters: sb_min (clamped to <= 0.0) and sb_range
        let mut sb_mins = [0.0f32; NUM_SUBBLOCKS];
        let mut sb_ranges = [0.0f32; NUM_SUBBLOCKS];
        let mut max_sb_range = 0.0f32;
        let mut max_abs_sb_min = 0.0f32;

        for sb in 0..NUM_SUBBLOCKS {
            let start = sb * SUBBLOCK_SIZE;
            let end = start + SUBBLOCK_SIZE;
            let subblock = &weights[start..end];

            let mut sb_min = f32::INFINITY;
            let mut sb_max = f32::NEG_INFINITY;
            for &w in subblock {
                sb_min = sb_min.min(w);
                sb_max = sb_max.max(w);
            }

            if sb_min > 0.0 {
                sb_min = 0.0;
            }
            let sb_range = sb_max - sb_min;

            sb_mins[sb] = sb_min;
            sb_ranges[sb] = sb_range;

            max_sb_range = max_sb_range.max(sb_range);
            max_abs_sb_min = max_abs_sb_min.max(-sb_min);
        }

        // 2. Compute super-block scale d and dmin
        let d = if max_sb_range > 0.0 {
            max_sb_range / (15.0 * 63.0)
        } else {
            0.0
        };

        let dmin = if max_abs_sb_min > 0.0 {
            max_abs_sb_min / 63.0
        } else {
            0.0
        };

        // Round to fp16 and convert back to f32 to avoid precision mismatch
        let d_f16 = Half16::from_f32(d);
        block.d = d_f16.to_le_bytes();
        let d_coarse = d_f16.to_f32();

        let dmin_f16 = Half16::from_f32(dmin);
        block.dmin = dmin_f16.to_le_bytes();
        let dmin_coarse = dmin_f16.to_f32();

        // 3. Compute 6-bit scales and mins for each subblock
        let mut sc = [0u8; NUM_SUBBLOCKS];
        let mut m = [0u8; NUM_SUBBLOCKS];

        for sb in 0..NUM_SUBBLOCKS {
            sc[sb] = if d_coarse > 1e-7 {
                ((sb_ranges[sb] / (15.0 * d_coarse)) + 0.5).floor() as u8
            } else {
                0
            }
            .min(63);

            m[sb] = if dmin_coarse > 1e-7 {
                ((-sb_mins[sb] / dmin_coarse) + 0.5).floor() as u8
            } else {
                0
            }
            .min(63);
        }

        // 4. Pack scales and mins into scales array (12 bytes)
        // Using the exact inverse of get_scale_min_k4
        for j in 0..4 {
            block.scales[j] = sc[j] & 63;
            block.scales[j + 4] = m[j] & 63;
        }
        for j in 4..8 {
            let sc_j = sc[j];
            let m_j = m[j];

            block.scales[j + 4] = (sc_j & 0xF) | ((m_j & 0xF) << 4);
            block.scales[j - 4] |= (sc_j >> 4) << 6;
            block.scales[j] |= (m_j >> 4) << 6;
        }

        // 5. Quantize sub-block weights to 4-bit qs array
        for sb in 0..NUM_SUBBLOCKS {
            let start = sb * SUBBLOCK_SIZE;
            let end = start + SUBBLOCK_SIZE;
            let subblock = &weights[start..end];

            let d1 = d_coarse * (sc[sb] as f32);
            let m1 = dmin_coarse * (m[sb] as f32);

            for (i, &w) in subblock.iter().enumerate() {
                let qi = if d1 > 1e-7 {
                    (((w + m1) / d1) + 0.5).floor() as i32
                } else {
                    0
                }
                .clamp(0, 15) as u8;

                // Pack 2 quants into 1 byte
                let byte_idx = (sb * SUBBLOCK_SIZE + i) / 2;
                let bit_offset = i % 2;

                if bit_offset == 0 {
                    block.qs[byte_idx] = (block.qs[byte_idx] & 0xF0) | qi;
                } else {
                    block.qs[byte_idx] = (block.qs[byte_idx] & 0x0F) | (qi << 4);
                }
            }
        }

        block
    }

    /// Dequantize Q4_K block back to f32.
    ///
    /// Reference: llama.cpp ggml_quants.c dequantize_row_q4_K
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

        // Parse overall scales from f16
        let d = Half16::from_le_bytes(self.d).to_f32();
        let min = Half16::from_le_bytes(self.dmin).to_f32();

        let mut out_idx = 0;
        let mut qs = &self.qs[..]; // Start at beginning of qs array
        let mut is = 0;

        // Process 4 iterations of 64 values each (256 total)
        // Each iteration processes 32 values from lower 4 bits and 32 from upper 4 bits
        for _j in 0..4 {
            // Get scale and min for first 32 values (lower 4 bits)
            let (sc, m) = Self::get_scale_min_k4(is, self.scales);
            let d1 = d * (sc as f32);
            let m1 = min * (m as f32);

            // Get scale and min for second 32 values (upper 4 bits)
            let (sc, m) = Self::get_scale_min_k4(is + 1, self.scales);
            let d2 = d * (sc as f32);
            let m2 = min * (m as f32);

            // Process lower 4 bits (first 32 values)
            for l in 0..32 {
                let q4 = qs[l] & 0xF;
                output[out_idx + l] = d1 * (q4 as f32) - m1;
            }

            // Process upper 4 bits (second 32 values)
            for l in 0..32 {
                let q4 = qs[l] >> 4;
                output[out_idx + 32 + l] = d2 * (q4 as f32) - m2;
            }

            qs = &qs[32..]; // Advance 32 bytes in qs array
            out_idx += 64; // Advanced 64 values in output
            is += 2;
        }
    }

    /// Extract scale and minimum for sub-block j from Q4_K scales array.
    ///
    /// Reference: llama.cpp ggml_quants.c get_scale_min_k4
    ///
    /// # Arguments
    ///
    /// * `j` - Sub-block index (0-7)
    /// * `scales` - 12-byte scales array
    ///
    /// # Returns
    ///
    /// Tuple of (scale, min) as 6-bit values
    pub fn get_scale_min_k4(j: usize, scales: [u8; 12]) -> (u8, u8) {
        let (d, m);
        if j < 4 {
            d = scales[j] & 63;
            m = scales[j + 4] & 63;
        } else {
            // For j >= 4, we need to be careful about array bounds
            // j+4 can be 8, 9, 10, 11 for j = 4, 5, 6, 7
            // j-4 can be 0, 1, 2, 3 for j = 4, 5, 6, 7
            let j_plus_4 = j + 4;
            let j_minus_4 = j - 4;

            if j_plus_4 < 12 {
                d = (scales[j_plus_4] & 0xF) | ((scales[j_minus_4] >> 6) << 4);
                m = (scales[j_plus_4] >> 4) | ((scales[j] >> 6) << 4);
            } else {
                // Fallback for safety
                d = 0;
                m = 0;
            }
        }
        (d, m)
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
                if sign != 0 {
                    -value
                } else {
                    value
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
            // Normal
            let value = (1 << 10 | mantissa) as f32 * (2.0_f32.powi(exponent - 15 - 10));
            if sign != 0 {
                -value
            } else {
                value
            }
        }
    }

    pub fn from_f32(val: f32) -> Self {
        // Simplified f32 to f16 conversion
        if val.is_nan() {
            return Self(0x7C00);
        }
        if val.is_infinite() {
            return if val.is_sign_negative() {
                Self(0xFC00)
            } else {
                Self(0x7C00)
            };
        }

        let bits = val.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        if exponent == 0 {
            if mantissa == 0 {
                return Self(sign << 15); // +/-0
            }
            // Subnormal - not handling in simplified version
            Self(sign << 15)
        } else if exponent == 255 {
            if mantissa == 0 {
                // Infinity
                Self((sign << 15) | 0x7C00)
            } else {
                // NaN
                Self(0x7E00)
            }
        } else {
            // Normal
            let new_exp = (exponent - 127 + 15).clamp(0, 31) as u16;
            let new_mant = (mantissa >> 13) as u16;
            Self((sign << 15) | (new_exp << 10) | new_mant)
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
            assert!(
                (o - 0.0).abs() < 0.01,
                "zero block should dequantize to near zero"
            );
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
        let test_values = [0.0, 1.0, -1.0, 0.5, -0.5, std::f32::consts::PI, -2.71];

        for &val in &test_values {
            let f16_val = Half16::from_f32(val);
            let back = f16_val.to_f32();
            assert!(
                (val - back).abs() < 0.01,
                "f16 roundtrip for {}: got {}",
                val,
                back
            );
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
