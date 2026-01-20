//! Quantization and dequantization functions

use super::gguf_tensor::GgufTensor;
use super::mxfp::MxfpBlock;
use super::tensor_type::GgufTensorType;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Dequantize Q8_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Each block is processed independently.
pub fn dequant_q8_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let blocks = total_elements.div_ceil(32);

    // Pre-allocate result vector
    let result = vec![0.0f32; total_elements];
    let result_lock = Arc::new(RwLock::new(result));

    // Process blocks in parallel using Rayon
    // Each block is independent - perfect for data parallelism
    (0..blocks).into_par_iter().for_each(|block_idx| {
        let block_start = block_idx * (4 + 32); // scale (4) + quants (32)

        if block_start + 4 > tensor.data.len() {
            return;
        }

        // Read scale (this is safe because we only read)
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values
        let quant_start = block_start + 4;
        let quant_end = std::cmp::min(quant_start + 32, tensor.data.len());
        let quants = &tensor.data[quant_start..quant_end];

        // Dequantize and write to shared result
        if let Ok(mut result) = result_lock.write() {
            for (i, &q) in quants.iter().enumerate() {
                let element_idx = block_idx * 32 + i;
                if element_idx < total_elements {
                    result[element_idx] = (q as f32 - 128.0) * scale;
                }
            }
        }
    });

    // Extract result from Arc<RwLock>
    let result = Arc::try_unwrap(result_lock)
        .map_err(|_e| anyhow::anyhow!("Failed to extract result: Arc still has owners"))?
        .into_inner()
        .map_err(|_e| anyhow::anyhow!("Failed to get inner value: RwLock poisoned"))?;

    Ok(result)
}

/// Dequantize Q4_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Q4_0 is the most common quantization format.
pub fn dequant_q4_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let blocks = total_elements.div_ceil(32);

    // Pre-allocate result vector
    let result = vec![0.0f32; total_elements];
    let result_lock = Arc::new(RwLock::new(result));

    // Process blocks in parallel using Rayon
    (0..blocks).into_par_iter().for_each(|block_idx| {
        let block_start = block_idx * (4 + 16); // scale (4) + quants (16 bytes for 32 values)

        if block_start + 4 > tensor.data.len() {
            return;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values (4-bit packed)
        let quant_start = block_start + 4;
        let quant_end = std::cmp::min(quant_start + 16, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (unpack 4-bit values) and write to shared result
        if let Ok(mut result) = result_lock.write() {
            for (i, &packed) in packed_quants.iter().enumerate() {
                for j in 0..2 {
                    let element_idx = block_idx * 32 + i * 2 + j;
                    if element_idx < total_elements {
                        let quant = if j == 0 {
                            packed & 0x0F
                        } else {
                            (packed >> 4) & 0x0F
                        };
                        result[element_idx] = (quant as f32 - 8.0) * scale;
                    }
                }
            }
        }
    });

    // Extract result from Arc<RwLock>
    let result = Arc::try_unwrap(result_lock)
        .map_err(|_e| anyhow::anyhow!("Failed to extract result: Arc still has owners"))?
        .into_inner()
        .map_err(|_e| anyhow::anyhow!("Failed to get inner value: RwLock poisoned"))?;

    Ok(result)
}

/// Dequantize MXFP4 tensor to FP32
pub fn dequant_mxfp4(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * 17; // 1 scale byte + 16 data bytes

        if block_start + 1 > tensor.data.len() {
            break;
        }

        // Read scale (E8M0)
        let scale_exp = tensor.data[block_start] as i8;
        let scale = 2.0_f32.powi(scale_exp as i32);

        // Read MXFP4 elements (4-bit packed)
        let data_start = block_start + 1;
        let data_end = std::cmp::min(data_start + 16, tensor.data.len());

        for (byte_offset, &packed) in tensor.data[data_start..data_end].iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + byte_offset * 2 + j;
                if element_idx < total_elements {
                    let e2m1_bits = if j == 0 {
                        (packed >> 4) & 0x0F
                    } else {
                        packed & 0x0F
                    };

                    // Decode E2M1
                    let decoded = MxfpBlock::decode_e2m1(e2m1_bits);
                    let mut val = scale * decoded;
                    val = val.clamp(-8.0, 8.0); // MXFP4 range per OCP MX Spec v1.0
                    result[element_idx] = val;
                }
            }
        }
    }

    Ok(result)
}

/// Dequantize MXFP6 tensor to FP32
pub fn dequant_mxfp6(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * 25; // 1 scale byte + 24 data bytes

        if block_start + 1 > tensor.data.len() {
            break;
        }

        // Read scale (E8M0)
        let scale_exp = tensor.data[block_start] as i8;
        let scale = 2.0_f32.powi(scale_exp as i32);

        // Read MXFP6 elements (6-bit packed)
        let data_start = block_start + 1;
        let data_end = std::cmp::min(data_start + 24, tensor.data.len());
        let packed_data = &tensor.data[data_start..data_end];

        // Unpack 6-bit values
        for i in 0..32 {
            let element_idx = block_idx * 32 + i;
            if element_idx >= total_elements {
                break;
            }

            // Extract 6-bit value
            let bit_offset = (i * 6) % 8;
            let byte_idx = (i * 6) / 8;

            if byte_idx + 1 < packed_data.len() {
                let combined =
                    ((packed_data[byte_idx + 1] as u16) << 8) | (packed_data[byte_idx] as u16);
                let e2m3_bits = ((combined >> (10 - bit_offset)) & 0x3F) as u8;

                // Decode E2M3
                let decoded = MxfpBlock::decode_e2m3(e2m3_bits);
                let mut val = scale * decoded;
                val = val.clamp(-7.5, 7.5); // MXFP6 range
                result[element_idx] = val;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q4_K tensor to FP32
/// Q4_K uses super-block structure with 256-byte blocks containing 8 sub-blocks
/// Each sub-block has its own scale and 4-bit quantized values
pub fn dequant_q4_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Q4_K super-block structure:
        // - 16 bytes: 8 half-precision scales (2 bytes each) for 8 sub-blocks
        // - 16 bytes: 8 int8 mins (1 byte each) for 8 sub-blocks
        // - 160 bytes: 8 sub-blocks of 4-bit quantized values (20 bytes each, packed)
        // - 64 bytes: additional data (likely for QK format)

        let scales_start = block_start;
        let mins_start = block_start + 16;
        let quants_start = block_start + 32;

        // Process each of the 8 sub-blocks (32 elements each)
        for sub_block_idx in 0..8 {
            let sub_block_start = block_idx * 256 + sub_block_idx * 32;
            let scale_idx = sub_block_idx;
            let min_idx = sub_block_idx;

            // Get scale for this sub-block
            let scale_offset = scales_start + scale_idx * 2;
            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0
            };

            // Get min for this sub-block
            let min_offset = mins_start + min_idx;
            let min = if min_offset < tensor.data.len() {
                tensor.data[min_offset] as i8 as f32
            } else {
                0.0
            };

            // Extract 4-bit quantized values for this sub-block (32 values)
            for i in 0..32 {
                let element_idx = sub_block_start + i;
                if element_idx >= total_elements {
                    break;
                }

                let bit_pos = i * 4;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quant_offset = quants_start + sub_block_idx * 20 + byte_idx;

                let quant = if quant_offset + 1 < tensor.data.len() {
                    let combined = ((tensor.data[quant_offset + 1] as u16) << 8)
                                   | (tensor.data[quant_offset] as u16);

                    ((combined >> bit_offset) & 0xF) as u8
                } else {
                    0
                };

                result[element_idx] = min + (quant as f32) * scale;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q5_K tensor to FP32
/// Q5_K uses super-block structure with 256-byte blocks
/// Format: 16 half-precision scales + 16 int8 mins + 160 bytes 5-bit quants + 48 bytes additional
pub fn dequant_q5_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Q5_K super-block structure:
        // - 32 bytes: 16 half-precision scales (2 bytes each) for 16 sub-blocks
        // - 16 bytes: 16 int8 mins (1 byte each) for 16 sub-blocks
        // - 160 bytes: 16 sub-blocks of 5-bit quantized values (10 bytes each)
        // - 48 bytes: additional data

        let scales_start = block_start;
        let mins_start = block_start + 32;
        let quants_start = block_start + 48;

        // Process each of the 16 sub-blocks (16 elements each)
        for sub_block_idx in 0..16 {
            let sub_block_start = block_idx * 256 + sub_block_idx * 16;

            // Get scale for this sub-block
            let scale_offset = scales_start + sub_block_idx * 2;
            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0
            };

            // Get min for this sub-block
            let min_offset = mins_start + sub_block_idx;
            let min = if min_offset < tensor.data.len() {
                tensor.data[min_offset] as i8 as f32
            } else {
                0.0
            };

            // Extract 5-bit quantized values for this sub-block (16 values)
            for i in 0..16 {
                let element_idx = sub_block_start + i;
                if element_idx >= total_elements {
                    break;
                }

                // 5-bit values packed: 16 values * 5 bits = 80 bits = 10 bytes
                let bit_pos = i * 5;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quant_offset = quants_start + sub_block_idx * 10 + byte_idx;

                let quant = if quant_offset + 2 <= tensor.data.len() {
                    // Read 16 bits to ensure we can extract 5 bits that may span 2 bytes
                    let combined = ((tensor.data[quant_offset + 1] as u16) << 8)
                                   | (tensor.data[quant_offset] as u16);
                    ((combined >> bit_offset) & 0x1F) as u8
                } else {
                    0
                };

                result[element_idx] = min + (quant as f32) * scale;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q6_K tensor to FP32
/// Q6_K uses 256-byte blocks encoding 256 elements
/// Format: scales (16 bytes) + quantized values (240 bytes for 256*6/8 = 192 bytes + padding)
pub fn dequant_q6_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Read scales (16 half-precision floats = 32 bytes)
        // Q6_K uses half-precision scales for each group of 16 elements
        let scales_start = block_start;

        // Read quantized values (6-bit packed, 256*6/8 = 192 bytes)
        let quants_start = block_start + 32;
        let quants_end = block_start + 224;

        // Dequantize block
        for i in 0..256 {
            let element_idx = block_idx * 256 + i;
            if element_idx >= total_elements {
                break;
            }

            // Get scale for this group (every 16 elements share a scale)
            let scale_idx = i / 16;
            let scale_offset = scales_start + scale_idx * 2;

            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0 // fallback scale
            };

            // Extract 6-bit quantized value
            let bit_offset = (i * 6) % 8;
            let byte_idx = (i * 6) / 8;

            if quants_start + byte_idx + 1 < quants_end {
                let combined = ((tensor.data[quants_start + byte_idx + 1] as u16) << 8)
                               | (tensor.data[quants_start + byte_idx] as u16);

                let quant_val = ((combined >> bit_offset) & 0x3F) as u8;

                // Convert to signed range and scale
                let signed_val = if quant_val >= 32 {
                    (quant_val as i8 - 64) as f32
                } else {
                    quant_val as f32
                };

                result[element_idx] = signed_val * scale;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q3_K tensor to FP32
/// Q3_K uses super-block structure with 256-byte blocks
/// Format: scales + quants with 3-bit packed values
pub fn dequant_q3_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Q3_K super-block structure:
        // - 32 bytes: 16 half-precision scales (2 bytes each) for 16 sub-blocks
        // - 4 bytes: qh (high bits for 3-bit quants)
        // - 160 bytes: 3-bit quantized values (256 * 3 / 8 = 96 bytes, but padded)
        // - 60 bytes: additional data

        let scales_start = block_start;
        let qh_start = block_start + 32;
        let quants_start = block_start + 36;

        // Process each of the 16 sub-blocks (16 elements each)
        for sub_block_idx in 0..16 {
            let sub_block_start = block_idx * 256 + sub_block_idx * 16;

            // Get scale for this sub-block
            let scale_offset = scales_start + sub_block_idx * 2;
            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0
            };

            // Read high bits (qh) - 2 bits per element
            let qh_offset = qh_start + sub_block_idx / 4;
            let qh_shift = (sub_block_idx % 4) * 2;
            let qh = if qh_offset < tensor.data.len() {
                (tensor.data[qh_offset] >> qh_shift) & 0x03
            } else {
                0
            };

            // Extract 3-bit quantized values for this sub-block (16 values)
            for i in 0..16 {
                let element_idx = sub_block_start + i;
                if element_idx >= total_elements {
                    break;
                }

                // 3-bit values packed: 16 values * 3 bits = 48 bits = 6 bytes per sub-block
                let bit_pos = i * 3;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quant_offset = quants_start + sub_block_idx * 6 + byte_idx;

                let quant = if quant_offset + 1 < tensor.data.len() {
                    let combined = ((tensor.data[quant_offset + 1] as u16) << 8)
                                   | (tensor.data[quant_offset] as u16);
                    let low_bits = ((combined >> bit_offset) & 0x07) as u8;

                    // Combine with high bits from qh
                    let high_bit = if i < 8 { (qh >> i) & 1 } else { 0 };
                    (low_bits | (high_bit << 3)) as i8 as f32 - 4.0
                } else {
                    0.0
                };

                result[element_idx] = quant * scale;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q2_K tensor to FP32
/// Q2_K uses super-block structure with 256-byte blocks (most complex K-quant format)
pub fn dequant_q2_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Q2_K super-block structure:
        // - 32 bytes: 16 half-precision scales (2 bytes each)
        // - 32 bytes: 16 half-precision mins (2 bytes each)
        // - 4 bytes: qh (high bits for 2-bit quants)
        // - 136 bytes: 2-bit quantized values (256 * 2 / 8 = 64 bytes + padding)
        // - 52 bytes: additional data

        let scales_start = block_start;
        let mins_start = block_start + 32;
        let qh_start = block_start + 64;
        let quants_start = block_start + 68;

        // Process each of the 16 sub-blocks (16 elements each)
        for sub_block_idx in 0..16 {
            let sub_block_start = block_idx * 256 + sub_block_idx * 16;

            // Get scale for this sub-block
            let scale_offset = scales_start + sub_block_idx * 2;
            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0
            };

            // Get min for this sub-block
            let min_offset = mins_start + sub_block_idx * 2;
            let min = if min_offset + 2 <= tensor.data.len() {
                let min_bits = u16::from_le_bytes([
                    tensor.data[min_offset],
                    tensor.data[min_offset + 1],
                ]);
                half::f16::from_bits(min_bits).to_f32()
            } else {
                0.0
            };

            // Read high bits (qh) - Q2_K has 1 high bit per pair of elements
            let qh_offset = qh_start + sub_block_idx / 8;
            let qh_shift = (sub_block_idx % 8) * 1;
            let qh = if qh_offset < tensor.data.len() {
                (tensor.data[qh_offset] >> qh_shift) & 0x01
            } else {
                0
            };

            // Extract 2-bit quantized values for this sub-block (16 values)
            for i in 0..16 {
                let element_idx = sub_block_start + i;
                if element_idx >= total_elements {
                    break;
                }

                // 2-bit values packed: 16 values * 2 bits = 32 bits = 4 bytes per sub-block
                let bit_pos = i * 2;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quant_offset = quants_start + sub_block_idx * 4 + byte_idx;

                let quant = if quant_offset + 1 < tensor.data.len() {
                    let combined = ((tensor.data[quant_offset + 1] as u16) << 8)
                                   | (tensor.data[quant_offset] as u16);
                    let low_bits = ((combined >> bit_offset) & 0x03) as u8;

                    // Combine with high bit from qh
                    let high_bit = if i < 8 { (qh >> i) & 1 } else { 0 };
                    (low_bits | (high_bit << 2)) as i8 as f32
                } else {
                    0.0
                };

                result[element_idx] = min + quant * scale;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::gguf_tensor::GgufTensor;
    use crate::loader::tensor_type::GgufTensorType;
    use crate::loader::TensorShape;

    fn create_test_tensor(tensor_type: GgufTensorType, data: Vec<u8>, shape: Vec<usize>) -> GgufTensor {
        GgufTensor {
            name: "test".to_string(),
            tensor_type,
            shape: TensorShape::from_dims(&shape),
            quant_type: tensor_type.to_string().to_string(),
            data,
            offset: 0,
        }
    }

    #[test]
    fn test_dequant_q5_k_zeros() {
        // Test Q5_K with all zeros
        let mut data = vec![0u8; 256];

        // Set a single scale (half precision 1.0)
        data[0] = 0x00;
        data[1] = 0x3C; // 1.0 in half precision

        let tensor = create_test_tensor(GgufTensorType::Q5_K, data, vec![256]);
        let result = dequant_q5_k(&tensor).unwrap();

        assert_eq!(result.len(), 256);
        // All values should be 0 (min + quant * scale where quant = 0)
        for (i, val) in result.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "Zero test mismatch at {}: expected ~0, got {}",
                i, val
            );
        }
    }

    #[test]
    fn test_dequant_q5_k_positive() {
        // Test Q5_K with known positive values
        let mut data = vec![0u8; 256];

        // Set scale = 1.0 in half precision (0x3C00)
        data[0] = 0x00;
        data[1] = 0x3C;

        // Set min = 0 (already 0 from initialization)

        // Set quantized values to known pattern
        // Q5_K: 16 sub-blocks, 16 elements each, 5 bits per element
        // First sub-block (elements 0-15): set quants to 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        // Packed: 1<<0 | 2<<5 | 3<<10 | 4<<15 | 5<<20 | 6<<25 | 7<<30 | ...
        // This is complex, so let's use a simpler pattern

        // Set first quant to 1 (bit 0-4), second quant to 0
        data[48] = 0x01; // First 5 bits = 1

        let tensor = create_test_tensor(GgufTensorType::Q5_K, data, vec![256]);
        let result = dequant_q5_k(&tensor).unwrap();

        assert_eq!(result.len(), 256);
        // First element should be ~1 (min=0, quant=1, scale=1.0)
        assert!((result[0] - 1.0).abs() < 0.1, "First element: {}", result[0]);
        // Rest should be ~0
        for i in 1..16 {
            assert!(result[i].abs() < 0.1, "Element at {}: {}", i, result[i]);
        }
    }

    #[test]
    fn test_dequant_q5_k_partial_block() {
        // Test Q5_K with partial block
        let mut data = vec![0u8; 256];

        // Set scale = 1.0
        data[0] = 0x00;
        data[1] = 0x3C;

        let tensor = create_test_tensor(GgufTensorType::Q5_K, data, vec![100]);
        let result = dequant_q5_k(&tensor).unwrap();

        assert_eq!(result.len(), 100);
        // All should be ~0 with zeros data
        for val in &result[..100] {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_dequant_q5_k_multiple_blocks() {
        // Test Q5_K with multiple blocks
        let mut data = vec![0u8; 512]; // 2 blocks

        // Block 1: scale = 1.0, all zeros
        data[0] = 0x00;
        data[1] = 0x3C;

        // Block 2: scale = 2.0 (half precision)
        data[256] = 0x00;
        data[257] = 0x40;

        // Set some non-zero quants in second block (first quant = 31)
        data[256 + 48] = 0xFF; // Lower 5 bits set = 31
        data[256 + 48 + 1] = 0x80; // Next 5 bits start here

        let tensor = create_test_tensor(GgufTensorType::Q5_K, data, vec![512]);
        let result = dequant_q5_k(&tensor).unwrap();

        assert_eq!(result.len(), 512);
        // First block should be ~0
        for i in 0..256 {
            assert!(result[i].abs() < 1e-5, "Block 1 at {}: {}", i, result[i]);
        }
        // Second block first element should be ~31 * 2.0
        assert!((result[256] - 62.0).abs() < 1.0, "Block 2 first: {}", result[256]);
    }

    #[test]
    fn test_dequant_q3_k_zeros() {
        // Test Q3_K with all zeros
        let mut data = vec![0u8; 256];

        // Set scale = 1.0 in half precision
        data[0] = 0x00;
        data[1] = 0x3C;

        let tensor = create_test_tensor(GgufTensorType::Q3_K, data, vec![256]);
        let result = dequant_q3_k(&tensor).unwrap();

        assert_eq!(result.len(), 256);
        // With zeros, quant=0, so result = (0 - 4) * 1.0 = -4.0
        for val in &result[..16] {
            assert!((val - (-4.0)).abs() < 0.1, "Expected -4.0, got {}", val);
        }
    }

    #[test]
    fn test_dequant_q3_k_positive() {
        // Test Q3_K with known values
        let mut data = vec![0u8; 256];

        // Set scale = 1.0
        data[0] = 0x00;
        data[1] = 0x3C;

        // Set first quant to 1 (bit 0-2)
        data[36] = 0x01; // First 3 bits = 1

        let tensor = create_test_tensor(GgufTensorType::Q3_K, data, vec![256]);
        let result = dequant_q3_k(&tensor).unwrap();

        assert_eq!(result.len(), 256);
        // First element: quant=1, result = (1 - 4) = -3
        assert!((result[0] - (-3.0)).abs() < 0.1, "First element: {}", result[0]);
        // Rest of first sub-block should be -4 (quant=0)
        for i in 1..16 {
            assert!((result[i] - (-4.0)).abs() < 0.1, "Element at {}: {}", i, result[i]);
        }
    }

    #[test]
    fn test_dequant_q3_k_partial_block() {
        // Test Q3_K with partial block (less than 256 elements)
        let mut data = vec![0u8; 256];

        data[0] = 0x00;
        data[1] = 0x3C;

        let tensor = create_test_tensor(GgufTensorType::Q3_K, data, vec![100]);
        let result = dequant_q3_k(&tensor).unwrap();

        assert_eq!(result.len(), 100);
        // First 100 elements should be -4 (quant=0, offset=-4)
        // Only check first sub-block (16 elements) since that's where our test data is
        for i in 0..16usize {
            if i < 100 {
                assert!((result[i] - (-4.0)).abs() < 0.1, "Element at {}: {}", i, result[i]);
            }
        }
    }

    #[test]
    fn test_dequant_q2_k_zeros() {
        // Test Q2_K with all zeros
        let mut data = vec![0u8; 256];

        // Set scale = 1.0 in half precision
        data[0] = 0x00;
        data[1] = 0x3C;

        // Min is already 0 from initialization

        let tensor = create_test_tensor(GgufTensorType::Q2_K, data, vec![256]);
        let result = dequant_q2_k(&tensor).unwrap();

        assert_eq!(result.len(), 256);
        // All should be 0 (min=0, quant=0, scale=1.0)
        for val in &result[..16] {
            assert!(val.abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequant_q2_k_positive() {
        // Test Q2_K with known values
        let mut data = vec![0u8; 256];

        // Set scale = 1.0
        data[0] = 0x00;
        data[1] = 0x3C;

        // Set first quant to 1 (bit 0-1)
        data[68] = 0x01; // First 2 bits = 1

        let tensor = create_test_tensor(GgufTensorType::Q2_K, data, vec![256]);
        let result = dequant_q2_k(&tensor).unwrap();

        assert_eq!(result.len(), 256);
        // First element should be ~1 (min=0, quant=1, scale=1)
        assert!((result[0] - 1.0).abs() < 0.1, "First element: {}", result[0]);
    }

    #[test]
    fn test_dequant_q2_k_partial_block() {
        // Test Q2_K with partial block
        let mut data = vec![0u8; 256];

        data[0] = 0x00;
        data[1] = 0x3C;

        let tensor = create_test_tensor(GgufTensorType::Q2_K, data, vec![100]);
        let result = dequant_q2_k(&tensor).unwrap();

        assert_eq!(result.len(), 100);
        // Should all be ~0
        for i in 0..16usize {
            if i < 100 {
                assert!(result[i].abs() < 1e-5, "Element at {}: {}", i, result[i]);
            }
        }
    }
}

/// Generic dequantization dispatcher
///
/// Routes to the appropriate dequantization function based on tensor type
pub fn dequantize(tensor: &GgufTensor) -> Result<Vec<f32>> {
    match tensor.tensor_type {
        GgufTensorType::F32 => Ok(tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        GgufTensorType::F16 => Ok(tensor
            .data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        GgufTensorType::Q8_0 => dequant_q8_0(tensor),
        GgufTensorType::Q4_0 => dequant_q4_0(tensor),
        GgufTensorType::Mxfp4 => dequant_mxfp4(tensor),
        GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => dequant_mxfp6(tensor),
        GgufTensorType::Q2_K => dequant_q2_k(tensor),
        GgufTensorType::Q3_K => dequant_q3_k(tensor),
        GgufTensorType::Q4_K => dequant_q4_k(tensor),
        GgufTensorType::Q5_K => dequant_q5_k(tensor),
        GgufTensorType::Q6_K => dequant_q6_k(tensor),
    }
}
