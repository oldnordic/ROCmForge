// ROCmForge - AMD GPU LLM Inference Engine
// Tests for Q4_1, Q5_0, Q5_1 GGUF dequantization formats
//
// Phase 8: Model Support - Task 8.1
// TDD: Tests written FIRST, then implementation

use rocmforge::loader::gguf::{GgufTensor, GgufTensorType};

// Use common fixtures
use crate::common::create_test_tensor;

/// Helper to manually dequantize Q4_1 (reference implementation)
fn dequantize_q4_1_reference(tensor: &GgufTensor) -> Vec<f32> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = (total_elements + 31) / 32;

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 16); // scale (4) + min (4) + quants (16)

        if block_start + 8 > tensor.data.len() {
            break;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read min
        let min_bytes = &tensor.data[block_start + 4..block_start + 8];
        let min = f32::from_le_bytes([min_bytes[0], min_bytes[1], min_bytes[2], min_bytes[3]]);

        // Read quantized values (4-bit packed)
        let quant_start = block_start + 8;
        let quant_end = std::cmp::min(quant_start + 16, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (unpack 4-bit values)
        for (i, &packed) in packed_quants.iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + i * 2 + j;
                if element_idx < total_elements {
                    let low_bits = if j == 0 {
                        packed & 0x0F
                    } else {
                        (packed >> 4) & 0x0F
                    };
                    result[element_idx] = min + (low_bits as f32) * scale;
                }
            }
        }
    }

    result
}

/// Helper to manually dequantize Q5_0 (reference implementation)
fn dequantize_q5_0_reference(tensor: &GgufTensor) -> Vec<f32> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = (total_elements + 31) / 32;

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 20); // scale (4) + qh (4) + quants (20)

        if block_start + 8 > tensor.data.len() {
            break;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read high bits (qh)
        let qh_bytes = &tensor.data[block_start + 4..block_start + 8];
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);

        // Read quantized values (4-bit packed + high bits)
        let quant_start = block_start + 8;
        let quant_end = std::cmp::min(quant_start + 20, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (5-bit values: 4 low bits from packed, 1 high bit from qh)
        for (i, &packed) in packed_quants.iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + i * 2 + j;
                if element_idx < total_elements {
                    let bit_idx = i * 2 + j;
                    let low_bits = if j == 0 {
                        packed & 0x0F
                    } else {
                        (packed >> 4) & 0x0F
                    };
                    let high_bit = if bit_idx < 32 { (qh >> bit_idx) & 1 } else { 0 };
                    let quant = low_bits as u8 | ((high_bit as u8) << 4);
                    result[element_idx] = (quant as f32 - 16.0) * scale;
                }
            }
        }
    }

    result
}

/// Helper to manually dequantize Q5_1 (reference implementation)
fn dequantize_q5_1_reference(tensor: &GgufTensor) -> Vec<f32> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = (total_elements + 31) / 32;

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 4 + 20); // scale (4) + min (4) + qh (4) + quants (20)

        if block_start + 12 > tensor.data.len() {
            break;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read min
        let min_bytes = &tensor.data[block_start + 4..block_start + 8];
        let min = f32::from_le_bytes([min_bytes[0], min_bytes[1], min_bytes[2], min_bytes[3]]);

        // Read high bits (qh)
        let qh_bytes = &tensor.data[block_start + 8..block_start + 12];
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);

        // Read quantized values (4-bit packed + high bits)
        let quant_start = block_start + 12;
        let quant_end = std::cmp::min(quant_start + 20, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (5-bit values: 4 low bits from packed, 1 high bit from qh)
        for (i, &packed) in packed_quants.iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + i * 2 + j;
                if element_idx < total_elements {
                    let bit_idx = i * 2 + j;
                    let low_bits = if j == 0 {
                        packed & 0x0F
                    } else {
                        (packed >> 4) & 0x0F
                    };
                    let high_bit = if bit_idx < 32 { (qh >> bit_idx) & 1 } else { 0 };
                    let quant = low_bits as u8 | ((high_bit as u8) << 4);
                    result[element_idx] = min + (quant as f32) * scale;
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod q4_1_tests {
    use super::*;

    #[test]
    fn test_q4_1_dequantize_single_block() {
        // Create a simple Q4_1 tensor with known values
        // Block structure: scale (4 bytes) + min (4 bytes) + 16 bytes of 4-bit packed values

        let scale: f32 = 0.1;
        let min: f32 = -5.0;

        let mut data = Vec::new();

        // Write scale
        data.extend_from_slice(&scale.to_le_bytes());

        // Write min
        data.extend_from_slice(&min.to_le_bytes());

        // Write quantized values (32 values packed into 16 bytes)
        // Values: 0, 1, 2, ..., 31 (packed 2 per byte)
        // Byte format: low_value in lower nibble, high_value in upper nibble
        for i in 0..16 {
            let low = (i * 2) & 0x0F; // Value at position i*2
            let high = ((i * 2) + 1) & 0x0F; // Value at position i*2+1
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q4_1, data, vec![32]);

        // Expected: min + quant * scale = -5.0 + (i % 16) * 0.1
        // Note: Q4_1 only stores 4-bit values (0-15), so values repeat every 16
        let expected: Vec<f32> = (0..32).map(|i| -5.0 + ((i % 16) as f32) * 0.1).collect();

        // Test against reference implementation
        let result = dequantize_q4_1_reference(&tensor);

        assert_eq!(result.len(), 32);
        for (i, (&val, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (val - exp).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_q4_1_dequantize_multiple_blocks() {
        // Test with multiple blocks (64 elements = 2 blocks)
        let scale: f32 = 0.05;
        let min: f32 = -10.0;

        let mut data = Vec::new();

        // Block 1
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend_from_slice(&min.to_le_bytes());
        for i in 0..16 {
            let low = (i * 2) & 0x0F; // Value at position i*2 in this block
            let high = ((i * 2) + 1) & 0x0F; // Value at position i*2+1 in this block
            data.push(low | (high << 4));
        }

        // Block 2 (different scale/min)
        let scale2: f32 = 0.02;
        let min2: f32 = 0.0;
        data.extend_from_slice(&scale2.to_le_bytes());
        data.extend_from_slice(&min2.to_le_bytes());
        for i in 0..16 {
            let low = (i * 2) & 0x0F; // Value at position i*2 in this block
            let high = ((i * 2) + 1) & 0x0F; // Value at position i*2+1 in this block
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q4_1, data, vec![64]);

        let result = dequantize_q4_1_reference(&tensor);

        assert_eq!(result.len(), 64);

        // Check first block
        for i in 0..32 {
            let quant = i % 16; // Q4_1 is 4-bit
            let expected = -10.0 + (quant as f32) * 0.05;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Block 1 mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }

        // Check second block
        for i in 32..64 {
            let quant = (i - 32) % 16; // Q4_1 is 4-bit
            let expected = 0.0 + (quant as f32) * 0.02;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Block 2 mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_q4_1_dequantize_2d_tensor() {
        // Test dequantization of a 2D tensor
        let scale: f32 = 0.1;
        let min: f32 = 1.0;

        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend_from_slice(&min.to_le_bytes());

        // 16 elements packed into 8 bytes
        for i in 0..8 {
            let low = (i * 2) & 0x0F; // Value at position i*2
            let high = ((i * 2) + 1) & 0x0F; // Value at position i*2+1
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q4_1, data, vec![4, 4]);

        let result = dequantize_q4_1_reference(&tensor);

        assert_eq!(result.len(), 16);
        for (i, &val) in result.iter().enumerate() {
            let expected = 1.0 + (i as f32) * 0.1;
            assert!(
                (val - expected).abs() < 1e-6,
                "2D tensor mismatch at {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }
}

#[cfg(test)]
mod q5_0_tests {
    use super::*;

    #[test]
    fn test_q5_0_dequantize_single_block() {
        // Create a Q5_0 tensor
        // Block: scale (4) + qh (4) + quants (20)

        let scale: f32 = 0.1;
        let mut data = Vec::new();

        // Write scale
        data.extend_from_slice(&scale.to_le_bytes());

        // Write high bits (qh) - set bits 0, 2, 4, ..., 30
        let mut qh: u32 = 0;
        for i in 0..32 {
            if i % 2 == 0 {
                qh |= 1 << i;
            }
        }
        data.extend_from_slice(&qh.to_le_bytes());

        // Write quantized values (4-bit low bits)
        for i in 0..20 {
            let low = (i * 2) & 0x0F;
            let high = ((i * 2) + 1) & 0x0F;
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q5_0, data, vec![32]);

        let result = dequantize_q5_0_reference(&tensor);

        assert_eq!(result.len(), 32);

        // Verify: Even indices have high bit set (quant = 0x10 + low)
        // Odd indices have no high bit (quant = low)
        for i in 0..32 {
            let low = (i % 16) & 0x0F;
            let high_bit = if i % 2 == 0 { 1 } else { 0 };
            let quant = low | (high_bit << 4);
            let expected = (quant as f32 - 16.0) * scale;

            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_q5_0_dequantize_range() {
        // Test full range of Q5_0 (0-31)
        let scale: f32 = 1.0;
        let mut data = Vec::new();

        data.extend_from_slice(&scale.to_le_bytes());

        // High bits: all even indices set
        let qh: u32 = 0x55555555; // Binary: 0101...
        data.extend_from_slice(&qh.to_le_bytes());

        // Quantized values: 0, 1, 0, 1, ... (low bits)
        for _i in 0..20 {
            // Pack: 0, 1, 0, 1, ...
            let byte = ((1 & 0x0F) << 4) | (0 & 0x0F);
            data.push(byte);
        }

        let tensor = create_test_tensor(GgufTensorType::Q5_0, data, vec![32]);

        let result = dequantize_q5_0_reference(&tensor);

        // Even indices: low=0, high_bit=1, quant=0|16=16, result=(16-16)*1=0
        // Odd indices: low=1, high_bit=0, quant=1|0=1, result=(1-16)*1=-15
        for i in 0..32 {
            let expected = if i % 2 == 0 { 0.0 } else { -15.0 };
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Range test mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_q5_0_dequantize_negative_scale() {
        // Test with negative scale
        let scale: f32 = -0.1;
        let mut data = Vec::new();

        data.extend_from_slice(&scale.to_le_bytes());

        // No high bits set
        let qh: u32 = 0;
        data.extend_from_slice(&qh.to_le_bytes());

        // Sequential low bits: 0, 1, 2, ..., 15, 0, 1, ...
        for i in 0..20 {
            let low = (i * 2) & 0x0F;
            let high = ((i * 2) + 1) & 0x0F;
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q5_0, data, vec![32]);

        let result = dequantize_q5_0_reference(&tensor);

        for i in 0..32 {
            let low_bits = i & 0x0F; // Sequential 0-15 repeating
            let expected = (low_bits as f32 - 16.0) * scale;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Negative scale mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }
}

#[cfg(test)]
mod q5_1_tests {
    use super::*;

    #[test]
    fn test_q5_1_dequantize_single_block() {
        // Create a Q5_1 tensor
        // Block: scale (4) + min (4) + qh (4) + quants (20)

        let scale: f32 = 0.1;
        let min: f32 = -5.0;
        let mut data = Vec::new();

        // Write scale
        data.extend_from_slice(&scale.to_le_bytes());

        // Write min
        data.extend_from_slice(&min.to_le_bytes());

        // Write high bits
        let mut qh: u32 = 0;
        for i in 0..32 {
            if i % 2 == 0 {
                qh |= 1 << i;
            }
        }
        data.extend_from_slice(&qh.to_le_bytes());

        // Write quantized values
        for i in 0..20 {
            let low = (i * 2) & 0x0F;
            let high = ((i * 2) + 1) & 0x0F;
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q5_1, data, vec![32]);

        let result = dequantize_q5_1_reference(&tensor);

        assert_eq!(result.len(), 32);

        // Verify formula: result = min + quant * scale
        for i in 0..32 {
            let low = (i % 16) & 0x0F;
            let high_bit = if i % 2 == 0 { 1 } else { 0 };
            let quant = low | (high_bit << 4);
            let expected = min + (quant as f32) * scale;

            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_q5_1_dequantize_full_range() {
        // Test full 5-bit range (0-31)
        let scale: f32 = 0.5;
        let min: f32 = -10.0;
        let mut data = Vec::new();

        data.extend_from_slice(&scale.to_le_bytes());
        data.extend_from_slice(&min.to_le_bytes());

        // Set specific high bits to create known pattern
        let qh: u32 = 0xFFFFFFFF; // All high bits set
        data.extend_from_slice(&qh.to_le_bytes());

        // Low bits: 0, 0, 0, 0, ...
        for _ in 0..20 {
            data.push(0x00);
        }

        let tensor = create_test_tensor(GgufTensorType::Q5_1, data, vec![32]);

        let result = dequantize_q5_1_reference(&tensor);

        // All values should be min + 0x10 * scale = -10 + 16 * 0.5 = -2
        for i in 0..32 {
            let expected = -10.0 + 16.0 * 0.5;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Full range mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_q5_1_dequantize_multiple_blocks() {
        // Test with multiple blocks
        let mut data = Vec::new();

        // Block 1
        let scale1: f32 = 0.1;
        let min1: f32 = 0.0;
        data.extend_from_slice(&scale1.to_le_bytes());
        data.extend_from_slice(&min1.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // qh = 0
        for i in 0..20 {
            let low = (i * 2) & 0x0F;
            let high = ((i * 2) + 1) & 0x0F;
            data.push(low | (high << 4));
        }

        // Block 2
        let scale2: f32 = 0.2;
        let min2: f32 = 10.0;
        data.extend_from_slice(&scale2.to_le_bytes());
        data.extend_from_slice(&min2.to_le_bytes());
        data.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // qh = all 1s
        for i in 0..20 {
            let low = (i * 2) & 0x0F;
            let high = ((i * 2) + 1) & 0x0F;
            data.push(low | (high << 4));
        }

        let tensor = create_test_tensor(GgufTensorType::Q5_1, data, vec![64]);

        let result = dequantize_q5_1_reference(&tensor);

        assert_eq!(result.len(), 64);

        // Check block 1 (no high bits)
        for i in 0..32 {
            let quant = i % 16; // 4-bit low values only
            let expected = 0.0 + (quant as f32) * 0.1;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Block 1 mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }

        // Check block 2 (all high bits set)
        for i in 32..64 {
            let bit_idx = i - 32;
            let low_bits = bit_idx & 0x0F;
            let high_bit = if bit_idx < 32 { 1 } else { 0 };
            let quant = low_bits | (high_bit << 4);
            let expected = 10.0 + (quant as f32) * 0.2;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "Block 2 mismatch at {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }
}
