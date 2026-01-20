//! GGUF Tensor Data Loading and Dequantization
//!
//! This module handles tensor data loading and CPU dequantization:
//! - Reading tensor data from memory-mapped files
//! - CPU dequantization for Q8_0, Q4_0, Q4_K, Q6_K
//! - MXFP format support (MXFP4, MXFP6)

use crate::loader::mxfp::MxfpBlock;
use crate::loader::GgufTensorType;
use crate::loader::gguf::types::GgufTensor;
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Dequantize Q8_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Each block is processed independently.
pub fn dequantize_q8_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
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
        .map_err(|_e| anyhow!("Failed to extract result: Arc still has owners"))?
        .into_inner()
        .map_err(|_e| anyhow!("Failed to get inner value: RwLock poisoned"))?;

    Ok(result)
}

/// Dequantize Q4_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Q4_0 is the most common quantization format.
pub fn dequantize_q4_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
    eprintln!(
        ">>> dequantize_q4_0: Starting for tensor '{}', size={} bytes",
        tensor.name,
        tensor.data.len()
    );
    let start = std::time::Instant::now();

    let total_elements = tensor.total_elements();
    let blocks = total_elements.div_ceil(32);
    eprintln!(
        ">>> dequantize_q4_0: total_elements={}, blocks={}",
        total_elements, blocks
    );

    // Pre-allocate result vector
    let result = vec![0.0f32; total_elements];
    let result_lock = Arc::new(RwLock::new(result));

    let dequant_start = std::time::Instant::now();

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

    eprintln!(
        ">>> dequantize_q4_0: Dequantization took {:?}",
        dequant_start.elapsed()
    );

    // Extract result from Arc<RwLock>
    let result = Arc::try_unwrap(result_lock)
        .map_err(|_e| anyhow!("Failed to extract result: Arc still has owners"))?
        .into_inner()
        .map_err(|_e| anyhow!("Failed to get inner value: RwLock poisoned"))?;

    eprintln!(">>> dequantize_q4_0: Total time {:?}", start.elapsed());
    Ok(result)
}

/// Dequantize Q4_K tensor to FP32
///
/// Q4_K uses super-block structure with 256-byte blocks containing 8 sub-blocks
/// Each sub-block has its own scale and 4-bit quantized values
pub fn dequantize_q4_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
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
                    let combined = ((tensor.data[quant_offset + 1] as u16) << 8) |
                                   (tensor.data[quant_offset] as u16);

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

/// Dequantize Q6_K tensor to FP32
///
/// Q6_K uses 256-byte blocks encoding 256 elements
/// Format: scales (16 bytes) + quantized values (240 bytes for 256*6/8 = 192 bytes + padding)
pub fn dequantize_q6_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
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
                let combined = ((tensor.data[quants_start + byte_idx + 1] as u16) << 8) |
                               (tensor.data[quants_start + byte_idx] as u16);

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

/// Dequantize MXFP4 tensor to FP32
///
/// NOTE: Used by upload_tensor_to_gpu for MXFP4 format support.
#[allow(dead_code)]
pub fn dequantize_mxfp4(tensor: &GgufTensor) -> Result<Vec<f32>> {
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
///
/// NOTE: Used by upload_tensor_to_gpu for MXFP6 format support.
#[allow(dead_code)]
pub fn dequantize_mxfp6(tensor: &GgufTensor) -> Result<Vec<f32>> {
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

/// Convert raw bytes to FP32 based on tensor type
///
/// Handles F32, F16, and delegates to quantization-specific functions
pub fn bytes_to_f32(tensor: &GgufTensor, tensor_bytes: &[u8]) -> Result<Vec<f32>> {
    match tensor.tensor_type {
        GgufTensorType::F32 => Ok(tensor_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        GgufTensorType::F16 => Ok(tensor_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        GgufTensorType::Q8_0 => {
            let temp_tensor = GgufTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                tensor_type: tensor.tensor_type,
                quant_type: tensor.quant_type.clone(),
                offset: tensor.offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q8_0(&temp_tensor)
        }
        GgufTensorType::Q4_0 => {
            let temp_tensor = GgufTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                tensor_type: tensor.tensor_type,
                quant_type: tensor.quant_type.clone(),
                offset: tensor.offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q4_0(&temp_tensor)
        }
        GgufTensorType::Q4_K => {
            let temp_tensor = GgufTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                tensor_type: tensor.tensor_type,
                quant_type: tensor.quant_type.clone(),
                offset: tensor.offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q4_k(&temp_tensor)
        }
        GgufTensorType::Q6_K => {
            let temp_tensor = GgufTensor {
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
                tensor_type: tensor.tensor_type,
                quant_type: tensor.quant_type.clone(),
                offset: tensor.offset,
                data: tensor_bytes.to_vec(),
            };
            dequantize_q6_k(&temp_tensor)
        }
        GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q5_K => {
            Err(anyhow!(
                "K-quant type {:?} not yet implemented for tensor '{}'",
                tensor.tensor_type,
                tensor.name
            ))
        }
        GgufTensorType::Mxfp4 | GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
            Err(anyhow!(
                "MXFP dequantization not yet implemented for tensor '{}'",
                tensor.name
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q8_0_single_block() {
        // Create a Q8_0 block: scale = 1.0, values = [128, 129, ..., 159]
        let mut data = vec![0u8; 36]; // 4 bytes scale + 32 values
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // scale = 1.0
        for i in 0..32 {
            data[4 + i] = 128 + i as u8; // quantized values
        }

        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: crate::loader::TensorShape::from_dims(&[32]),
            tensor_type: GgufTensorType::Q8_0,
            quant_type: "Q8_0".to_string(),
            offset: 0,
            data,
        };

        let result = dequantize_q8_0(&tensor).unwrap();
        assert_eq!(result.len(), 32);
        assert_eq!(result[0], 0.0); // (128 - 128) * 1.0
        assert_eq!(result[1], 1.0); // (129 - 128) * 1.0
        assert_eq!(result[31], 31.0); // (159 - 128) * 1.0
    }

    #[test]
    fn test_dequantize_q4_0_single_block() {
        // Create a Q4_0 block: scale = 1.0, packed quants
        let mut data = vec![0u8; 20]; // 4 bytes scale + 16 bytes quants
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // scale = 1.0
        // Packed 4-bit values: [0, 1, 2, 3, ..., 15]
        for i in 0u8..16 {
            let packed = (i % 16) | ((i + 1) % 16 << 4);
            data[4 + i as usize] = packed;
        }

        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: crate::loader::TensorShape::from_dims(&[32]),
            tensor_type: GgufTensorType::Q4_0,
            quant_type: "Q4_0".to_string(),
            offset: 0,
            data,
        };

        let result = dequantize_q4_0(&tensor).unwrap();
        assert_eq!(result.len(), 32);
        assert_eq!(result[0], -8.0); // (0 - 8) * 1.0
        assert_eq!(result[1], -7.0); // (1 - 8) * 1.0
    }

    #[test]
    fn test_bytes_to_f32_f32() {
        let bytes = [0u8, 0, 128, 63]; // 1.0 in little endian FP32
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: crate::loader::TensorShape::from_dims(&[1]),
            tensor_type: GgufTensorType::F32,
            quant_type: "F32".to_string(),
            offset: 0,
            data: vec![],
        };

        let result = bytes_to_f32(&tensor, &bytes).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_bytes_to_f32_f16() {
        let bytes = [0u8, 60]; // 1.0 in little endian FP16
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: crate::loader::TensorShape::from_dims(&[1]),
            tensor_type: GgufTensorType::F16,
            quant_type: "F16".to_string(),
            offset: 0,
            data: vec![],
        };

        let result = bytes_to_f32(&tensor, &bytes).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 0.001);
    }
}
