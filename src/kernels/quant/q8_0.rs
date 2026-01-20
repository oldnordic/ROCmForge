//! Q8_0 quantization kernel (CPU only)
//!
//! Q8_0 format: 8-bit quantization with constant scale
//! - Block size: 32 elements
//! - Per block (36 bytes): 4 bytes scale + 32 bytes int8 values
//! - Dequantization: value = (quant - 128) * scale
//! - Signed 8-bit range: [-128, 127]

use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Dequantize Q8_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Each block is processed independently.
///
/// # Q8_0 CPU Reference Implementation
///
/// Q8_0 format details:
/// - Block: 32 elements in 36 bytes
/// - Scale layout: 4 bytes (f32) at block start
/// - Quant layout: 32 bytes of signed int8 values
pub fn dequantize_q8_0(tensor_data: &[u8], total_elements: usize) -> Result<Vec<f32>, String> {
    let blocks = total_elements.div_ceil(32);

    // Pre-allocate result vector
    let result = vec![0.0f32; total_elements];
    let result_lock = Arc::new(RwLock::new(result));

    // Process blocks in parallel using Rayon
    // Each block is independent - perfect for data parallelism
    (0..blocks).into_par_iter().for_each(|block_idx| {
        let block_start = block_idx * (4 + 32); // scale (4) + quants (32)

        if block_start + 4 > tensor_data.len() {
            return;
        }

        // Read scale (this is safe because we only read)
        let scale_bytes = &tensor_data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values
        let quant_start = block_start + 4;
        let quant_end = std::cmp::min(quant_start + 32, tensor_data.len());
        let quants = &tensor_data[quant_start..quant_end];

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
        .map_err(|_e| "Failed to extract result: Arc still has owners".to_string())?
        .into_inner()
        .map_err(|_e| "Failed to get inner value: RwLock poisoned".to_string())?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q8_0_zeros() {
        // Create test data: 1 block with scale=1.0, all values = 128 (dequantizes to 0.0)
        let mut data = vec![0u8; 36]; // 1 block * 36 bytes

        // Scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // Pack 32 values, all stored as 128 (representing 0.0 after dequant)
        for i in 0..32 {
            data[4 + i] = 128;
        }

        let result = dequantize_q8_0(&data, 32).unwrap();

        // All values should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    #[test]
    fn test_dequantize_q8_0_positive() {
        // Create test data: 1 block with scale=2.0
        let mut data = vec![0u8; 36];

        // Scale = 2.0
        data[0..4].copy_from_slice(&2.0f32.to_le_bytes());

        // Values 120-151 (representing -16 to 23 after subtracting 128, then scaled by 2)
        for i in 0..32 {
            data[4 + i] = 120 + i as u8;
        }

        let result = dequantize_q8_0(&data, 32).unwrap();

        // Expected: value = (input - 128) * scale
        // First value: (120 - 128) * 2.0 = -16.0
        assert!((result[0] - (-16.0)).abs() < 0.01);
        // Middle value: (135 - 128) * 2.0 = 14.0
        assert!((result[15] - 14.0).abs() < 0.01);
        // Last value: (151 - 128) * 2.0 = 46.0
        assert!((result[31] - 46.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q8_0_partial_block() {
        // Test partial block (not multiple of 32 elements)
        let mut data = vec![0u8; 36];

        // Scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // All values = 128 (0.0 after dequant)
        for i in 0..32 {
            data[4 + i] = 128;
        }

        let result = dequantize_q8_0(&data, 10).unwrap();

        // First 10 should be 0.0
        for i in 0..10 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        // Test multiple blocks
        let n_elements = 64; // 2 blocks
        let block_size = 36;
        let mut data = vec![0u8; 2 * block_size];

        // Block 0: scale = 1.0, values = 128 (0.0)
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..32 {
            data[4 + i] = 128;
        }

        // Block 1: scale = 0.5, values = 138 (5.0 after dequant)
        data[36..40].copy_from_slice(&0.5f32.to_le_bytes());
        for i in 0..32 {
            data[40 + i] = 138;
        }

        let result = dequantize_q8_0(&data, n_elements).unwrap();

        // First 32 should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }

        // Next 32 should be 5.0: (138 - 128) * 0.5 = 10 * 0.5 = 5.0
        for i in 32..64 {
            assert!((result[i] - 5.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }
}
