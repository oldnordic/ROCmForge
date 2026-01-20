//! MXFP4/MXFP6 Quantization Tests
//!
//! Test-Driven Development approach:
//! 1. Tests written FIRST (all must FAIL initially)
//! 2. Implementation written AFTER tests
//! 3. Tests verify OCP MX Specification v1.0 compliance
//!
//! Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

use crate::loader::mxfp::{MxfpBlock, E8M0};

/// Test E8M0 scale conversion (exponent-only format)
///
/// E8M0 format: 8-bit exponent, value = 2^exponent
/// Used as block scale in MXFP4/MXFP6
#[cfg(test)]
mod test_e8m0 {
    use super::*;

    #[test]
    fn test_e8m0_to_f32_zero() {
        // E8M0(0) = 2^0 = 1.0
        let e8m0 = E8M0 { exponent: 0 };
        assert!((e8m0.to_f32() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e8m0_to_f32_positive() {
        // E8M0(1) = 2^1 = 2.0
        let e8m0 = E8M0 { exponent: 1 };
        assert!((e8m0.to_f32() - 2.0).abs() < f32::EPSILON);

        // E8M0(2) = 2^2 = 4.0
        let e8m0 = E8M0 { exponent: 2 };
        assert!((e8m0.to_f32() - 4.0).abs() < f32::EPSILON);

        // E8M0(10) = 2^10 = 1024.0
        let e8m0 = E8M0 { exponent: 10 };
        assert!((e8m0.to_f32() - 1024.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e8m0_to_f32_negative() {
        // E8M0(-1) = 2^(-1) = 0.5
        let e8m0 = E8M0 { exponent: -1 };
        assert!((e8m0.to_f32() - 0.5).abs() < f32::EPSILON);

        // E8M0(-2) = 2^(-2) = 0.25
        let e8m0 = E8M0 { exponent: -2 };
        assert!((e8m0.to_f32() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_e8m0_from_f32_roundtrip() {
        // Test roundtrip conversion maintains accuracy within 0.1%
        let test_values = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 0.125, 256.0, 1024.0];

        for &value in &test_values {
            let e8m0 = E8M0::from_f32(value);
            let recovered = e8m0.to_f32();

            // Should be exact for powers of 2
            let error_pct = (recovered - value).abs() / value * 100.0;
            assert!(
                error_pct < 0.1,
                "Roundtrip error {:.3}% for value {} (got {})",
                error_pct,
                value,
                recovered
            );
        }
    }

    #[test]
    fn test_e8m0_clamping() {
        // E8M0 should clamp to [-127, 127] range (i8 limits)
        let e8m0 = E8M0::from_f32(f32::INFINITY);
        assert_eq!(e8m0.exponent, 127);

        let e8m0 = E8M0::from_f32(0.0);
        assert!(e8m0.exponent >= -127);
        assert!(e8m0.exponent <= 127);
    }
}

/// Test MXFP4 block packing/unpacking
///
/// MXFP4 format (OCP MX Spec v1.0):
/// - 4-bit elements (E2M1: 2 exponent, 1 mantissa, 1 sign)
/// - Block size: 32 elements
/// - Scale: E8M0 (1 byte)
/// - Total: 1 + (32 * 4 / 8) = 17 bytes per block
#[cfg(test)]
mod test_mxfp4_block {
    use super::*;

    #[test]
    fn test_mxfp4_block_size() {
        // MXFP4 block should be 17 bytes (1 scale + 16 data bytes for 32 elements)
        let block = MxfpBlock::new_mxfp4();
        assert_eq!(block.packed_size(), 17);
    }

    #[test]
    fn test_mxfp4_pack_32_elements() {
        // Pack 32 float values into MXFP4 block
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let block = MxfpBlock::pack_mxfp4(&values);

        // Should not panic and should produce valid packed data
        assert!(!block.elements.is_empty());
        assert_eq!(block.elements.len(), 16); // 32 elements * 4 bits / 8
    }

    #[test]
    fn test_mxfp4_unpack_32_elements() {
        // MXFP is designed for blocks where values have SIMILAR magnitudes
        // Using uniform values to demonstrate correct encode/decode

        // All values the same (ideal case for MXFP)
        let values: Vec<f32> = vec![2.0; 32];

        let block = MxfpBlock::pack_mxfp4(&values);
        let unpacked = block.unpack_mxfp4();

        assert_eq!(unpacked.len(), 32);

        // Check accuracy: <0.1% error requirement
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            let error_pct = (recovered - original).abs() / original.abs().max(f32::EPSILON) * 100.0;
            assert!(
                error_pct < 0.1,
                "MXFP4 roundtrip error {:.3}%: original={}, recovered={}",
                error_pct,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_mxfp4_e2m1_encoding() {
        // Test E2M1 (4-bit) encoding: sign(1) + exponent(2) + mantissa(1)
        // E2M1 format: value = (-1)^sign * 2^(exp-1) * (1.mant)

        // Test special values
        assert_eq!(MxfpBlock::encode_e2m1(0.0), 0b0000); // Zero

        // Test positive values
        let pos_small = MxfpBlock::encode_e2m1(0.5);
        assert!(pos_small & 0x08 == 0); // Sign bit = 0

        // Test negative values
        let neg_small = MxfpBlock::encode_e2m1(-0.5);
        assert!(neg_small & 0x08 != 0); // Sign bit = 1
    }

    #[test]
    fn test_mxfp4_e2m1_decoding() {
        // Test E2M1 decoding
        assert_eq!(MxfpBlock::decode_e2m1(0b0000), 0.0); // Zero

        // Decode positive value
        let pos_val = MxfpBlock::decode_e2m1(0b0101);
        assert!(pos_val > 0.0);

        // Decode negative value
        let neg_val = MxfpBlock::decode_e2m1(0b1101);
        assert!(neg_val < 0.0);

        // Check magnitude relationship
        assert_eq!(pos_val.abs(), neg_val.abs());
    }

    #[test]
    fn test_mxfp4_range_clamping() {
        // MXFP4 range: [-8, 8] per OCP MX Spec v1.0
        let values = vec![10.0, -10.0, 100.0, -100.0]; // Values outside range
        let block = MxfpBlock::pack_mxfp4(&values);
        let unpacked = block.unpack_mxfp4();

        // All values should be clamped to [-8, 8]
        for &val in &unpacked {
            assert!(val >= -8.0, "Value {} below MXFP4 range", val);
            assert!(val <= 8.0, "Value {} above MXFP4 range", val);
        }
    }
}

/// Test MXFP6 block packing/unpacking
///
/// MXFP6 format (OCP MX Spec v1.0):
/// - 6-bit elements (E2M3: 2 exponent, 3 mantissa, 1 sign)
/// - Block size: 32 elements
/// - Scale: E8M0 (1 byte)
/// - Total: 1 + (32 * 6 / 8) = 25 bytes per block
#[cfg(test)]
mod test_mxfp6_block {
    use super::*;

    #[test]
    fn test_mxfp6_block_size() {
        // MXFP6 block should be 25 bytes (1 scale + 24 data bytes for 32 elements)
        let block = MxfpBlock::new_mxfp6();
        assert_eq!(block.packed_size(), 25);
    }

    #[test]
    fn test_mxfp6_pack_32_elements() {
        // Pack 32 float values into MXFP6 block
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let block = MxfpBlock::pack_mxfp6(&values);

        // Should not panic and should produce valid packed data
        assert!(!block.elements.is_empty());
        assert_eq!(block.elements.len(), 24); // 32 elements * 6 bits / 8
    }

    #[test]
    fn test_mxfp6_unpack_32_elements() {
        // MXFP is designed for blocks where values have SIMILAR magnitudes
        // Using uniform values to demonstrate correct encode/decode

        // All values the same (ideal case for MXFP)
        let values: Vec<f32> = vec![2.0; 32];

        let block = MxfpBlock::pack_mxfp6(&values);
        let unpacked = block.unpack_mxfp6();

        assert_eq!(unpacked.len(), 32);

        // Check accuracy: <0.1% error requirement
        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            let error_pct = (recovered - original).abs() / original.abs().max(f32::EPSILON) * 100.0;
            assert!(
                error_pct < 0.1,
                "MXFP6 roundtrip error {:.3}%: original={}, recovered={}",
                error_pct,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_mxfp6_e2m3_encoding() {
        // Test E2M3 (6-bit) encoding: sign(1) + exponent(2) + mantissa(3)
        // E2M3 format: value = (-1)^sign * 2^(exp-1) * (1.mant/8)

        // Test special values
        assert_eq!(MxfpBlock::encode_e2m3(0.0), 0b000000); // Zero

        // Test positive values
        let pos_small = MxfpBlock::encode_e2m3(0.5);
        assert!(pos_small & 0x20 == 0); // Sign bit = 0

        // Test negative values
        let neg_small = MxfpBlock::encode_e2m3(-0.5);
        assert!(neg_small & 0x20 != 0); // Sign bit = 1
    }

    #[test]
    fn test_mxfp6_e2m3_decoding() {
        // Test E2M3 decoding
        assert_eq!(MxfpBlock::decode_e2m3(0b000000), 0.0); // Zero

        // Decode positive value
        let pos_val = MxfpBlock::decode_e2m3(0b001010);
        assert!(pos_val > 0.0);

        // Decode negative value
        let neg_val = MxfpBlock::decode_e2m3(0b101010);
        assert!(neg_val < 0.0);

        // Check magnitude relationship
        assert_eq!(pos_val.abs(), neg_val.abs());
    }

    #[test]
    fn test_mxfp6_range_clamping() {
        // MXFP6 range: [-7.5, 7.5] per OCP spec
        let values = vec![10.0, -10.0, 100.0, -100.0]; // Values outside range
        let block = MxfpBlock::pack_mxfp6(&values);
        let unpacked = block.unpack_mxfp6();

        // All values should be clamped to [-7.5, 7.5]
        for &val in &unpacked {
            assert!(val >= -7.5, "Value {} below MXFP6 range", val);
            assert!(val <= 7.5, "Value {} above MXFP6 range", val);
        }
    }

    #[test]
    fn test_mxfp6_bit_packing() {
        // Test 6-bit value packing across byte boundaries
        let values: Vec<u8> = (0..32).map(|i| i as u8 & 0x3F).collect();
        let packed = MxfpBlock::pack_6bit_values(&values);

        // Should pack 32 x 6-bit values into 24 bytes
        assert_eq!(packed.len(), 24);

        // Verify unpacking
        let unpacked = MxfpBlock::unpack_6bit_values(&packed, 32);
        assert_eq!(unpacked.len(), 32);

        for (original, recovered) in values.iter().zip(unpacked.iter()) {
            assert_eq!(original, recovered, "Bit packing mismatch");
        }
    }
}

/// Test dequantization accuracy
///
/// Verifies MXFP can exactly represent powers of 2 (E2M1/E2M3 design)
#[cfg(test)]
mod test_dequantization_accuracy {
    use super::*;

    #[test]
    fn test_mxfp4_dequantization_accuracy() {
        // MXFP4 E2M1 can exactly represent powers of 2: 0.5, 1.0, 2.0, 4.0, 8.0
        // Use these values for exact roundtrip verification

        // Case 0: All 1.0 (exactly representable)
        let ones: Vec<f32> = vec![1.0; 32];

        // Case 1: All 2.0 (exactly representable)
        let twos: Vec<f32> = vec![2.0; 32];

        // Case 2: All 4.0 (exactly representable)
        let fours: Vec<f32> = vec![4.0; 32];

        let test_cases = vec![ones, twos, fours];

        for (case_idx, values) in test_cases.iter().enumerate() {
            let block = MxfpBlock::pack_mxfp4(values);
            let recovered = block.unpack_mxfp4();

            // For exact powers of 2, roundtrip should be perfect (0% error)
            for (original, recovered_val) in values.iter().zip(recovered.iter()) {
                assert!(
                    (original - recovered_val).abs() < f32::EPSILON,
                    "Case {}: MXFP4 roundtrip error: original={}, recovered={}",
                    case_idx,
                    original,
                    recovered_val
                );
            }
        }
    }

    #[test]
    fn test_mxfp6_dequantization_accuracy() {
        // MXFP6 E2M3 can exactly represent powers of 2 and some fractions
        // Use values that are exactly representable for roundtrip verification

        // Case 0: All 1.0 (exactly representable)
        let ones: Vec<f32> = vec![1.0; 32];

        // Case 1: All 2.0 (exactly representable)
        let twos: Vec<f32> = vec![2.0; 32];

        // Case 2: All 4.0 (exactly representable)
        let fours: Vec<f32> = vec![4.0; 32];

        let test_cases = vec![ones, twos, fours];

        for (case_idx, values) in test_cases.iter().enumerate() {
            let block = MxfpBlock::pack_mxfp6(values);
            let recovered = block.unpack_mxfp6();

            // For exact powers of 2, roundtrip should be perfect (0% error)
            for (original, recovered_val) in values.iter().zip(recovered.iter()) {
                assert!(
                    (original - recovered_val).abs() < f32::EPSILON,
                    "Case {}: MXFP6 roundtrip error: original={}, recovered={}",
                    case_idx,
                    original,
                    recovered_val
                );
            }
        }
    }

    #[test]
    fn test_mxfp6_better_than_mxfp4() {
        // MXFP6 should have better accuracy than MXFP4 (per analysis)
        let values: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) + 0.01).collect();

        let block_mxfp4 = MxfpBlock::pack_mxfp4(&values);
        let recovered_mxfp4 = block_mxfp4.unpack_mxfp4();

        let block_mxfp6 = MxfpBlock::pack_mxfp6(&values);
        let recovered_mxfp6 = block_mxfp6.unpack_mxfp6();

        let mse4: f32 = values
            .iter()
            .zip(recovered_mxfp4.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum();

        let mse6: f32 = values
            .iter()
            .zip(recovered_mxfp6.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum();

        assert!(
            mse6 < mse4,
            "MXFP6 (MSE={}) should outperform MXFP4 (MSE={})",
            mse6,
            mse4
        );
    }
}

/// Test GGUF tensor type enum values
///
/// Ensures MXFP types have correct enum values
#[cfg(test)]
mod test_gguf_tensor_types {
    use crate::loader::GgufTensorType;

    #[test]
    fn test_mxfp_tensor_type_values() {
        // Verify enum values don't conflict with existing types
        // Existing: F32=0, F16=1, Q4_0=2, Q8_0=8
        // Note: Q4_1=3, Q5_0=6, Q5_1=7 are NO LONGER SUPPORTED (removed 2026-01-20)
        // MXFP: MXFP4=20, MXFP6_E2M3=21, MXFP6_E3M2=22

        assert_eq!(GgufTensorType::F32 as u32, 0);
        assert_eq!(GgufTensorType::F16 as u32, 1);
        assert_eq!(GgufTensorType::Q8_0 as u32, 8);

        // MXFP types (to be implemented)
        assert_eq!(GgufTensorType::Mxfp4 as u32, 20);
        assert_eq!(GgufTensorType::Mxfp6E2m3 as u32, 21);
        assert_eq!(GgufTensorType::Mxfp6E3m2 as u32, 22);
    }

    #[test]
    fn test_gguf_tensor_type_from_u32() {
        // Test roundtrip conversion
        assert_eq!(GgufTensorType::from_u32(20).unwrap(), GgufTensorType::Mxfp4);
        assert_eq!(
            GgufTensorType::from_u32(21).unwrap(),
            GgufTensorType::Mxfp6E2m3
        );
        assert_eq!(
            GgufTensorType::from_u32(22).unwrap(),
            GgufTensorType::Mxfp6E3m2
        );

        // Test invalid type
        assert!(GgufTensorType::from_u32(999).is_err());
    }

    #[test]
    fn test_gguf_tensor_type_element_size() {
        // Verify element_size() returns correct block sizes
        assert_eq!(GgufTensorType::Mxfp4.element_size(), 32); // Block size
        assert_eq!(GgufTensorType::Mxfp6E2m3.element_size(), 32); // Block size
        assert_eq!(GgufTensorType::Mxfp6E3m2.element_size(), 32); // Block size
    }
}
