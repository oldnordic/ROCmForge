//! Q4_0 dequantization operations for GPU
//!
//! PHASE 5: GPU-side Q4_0 dequantization
//!
//! This module provides GPU dequantization for Q4_0 quantized weights.
//! Currently uses CPU-side dequantization followed by GPU upload.
//! Future work will implement native HIP kernel for on-device dequantization.

use crate::backend::HipBackend;

/// Result type for Q4_0 dequantization operations
pub type Q4_0DequantResult<T> = Result<T, String>;

/// Q4_0 block header (scale only)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Block {
    /// Scale factor for this block
    pub scale: f32,
    // 16 bytes of packed 4-bit values follow in the actual data
}

/// Dequantize Q4_0 weights to f32 on GPU
///
/// # Format
/// - Each block has 32 values packed into 16 bytes
/// - Each value is 4 bits, interpreted as signed: unpacked - 8
/// - Scale applies to all 32 values in the block
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q4_0 quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # Note
/// Currently dequantizes on CPU then uploads to GPU.
/// TODO: Implement native HIP kernel for on-device dequantization (05-02).
pub fn dequantize_q4_0_gpu(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Q4_0DequantResult<()> {
    // Calculate blocks (32 elements per block)
    let num_blocks = (num_elements + 31) / 32;
    let block_size = 20; // 4 bytes scale + 16 bytes packed data

    // Pre-allocate result vector
    let mut dequantized = vec![0.0f32; num_elements];

    // Dequantize each block
    for block_idx in 0..num_blocks {
        let block_offset = block_idx * block_size;
        if block_offset + 4 > quantized_data.len() {
            break;
        }

        // Read scale
        let scale = f32::from_le_bytes([
            quantized_data[block_offset],
            quantized_data[block_offset + 1],
            quantized_data[block_offset + 2],
            quantized_data[block_offset + 3],
        ]);

        // Unpack 4-bit values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..16 {
            if data_start + i >= quantized_data.len() {
                break;
            }
            let packed = quantized_data[data_start + i];

            // Low nibble
            let low = (packed & 0x0F) as i32 - 8;
            if base_idx + i * 2 < num_elements {
                dequantized[base_idx + i * 2] = scale * low as f32;
            }

            // High nibble
            let high = ((packed >> 4) & 0x0F) as i32 - 8;
            if base_idx + i * 2 + 1 < num_elements {
                dequantized[base_idx + i * 2 + 1] = scale * high as f32;
            }
        }
    }

    // Upload dequantized data to GPU
    output
        .copy_from_host(&dequantized)
        .map_err(|e| format!("Failed to upload dequantized data: {}", e))
}

/// Dequantize Q4_0 weights to f32 (CPU version, for testing)
///
/// This is the reference CPU implementation used for testing
/// and comparison with the GPU version.
pub fn dequantize_q4_0_cpu(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];
    let block_size = 20; // 4 bytes scale + 16 bytes packed data

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * block_size;
        if block_offset + 4 > data.len() {
            break;
        }

        // Read scale
        let scale = f32::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
            data[block_offset + 2],
            data[block_offset + 3],
        ]);

        // Unpack 4-bit values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..16 {
            if data_start + i >= data.len() {
                break;
            }
            let packed = data[data_start + i];

            // Low nibble
            let low = (packed & 0x0F) as i32 - 8;
            if base_idx + i * 2 < n_elements {
                result[base_idx + i * 2] = scale * low as f32;
            }

            // High nibble
            let high = ((packed >> 4) & 0x0F) as i32 - 8;
            if base_idx + i * 2 + 1 < n_elements {
                result[base_idx + i * 2 + 1] = scale * high as f32;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q4_0_cpu_zeros() {
        // Create test data: 1 block with scale=1.0, all values = 8 (dequantizes to 0.0)
        let mut data = vec![0u8; 20]; // 1 block * 20 bytes

        // Scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // Pack 32 values, all stored as 8 (representing 0.0 after dequant)
        for i in 0..16 {
            data[4 + i] = 0x88; // Both nibbles = 8
        }

        let result = dequantize_q4_0_cpu(&data, 32);

        // All values should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    #[test]
    fn test_dequantize_q4_0_cpu_positive() {
        // Create test data: 1 block with scale=2.0, values from 0-15
        let mut data = vec![0u8; 20];

        // Scale = 2.0
        data[0..4].copy_from_slice(&2.0f32.to_le_bytes());

        // Pack values 0, 1, 2, ..., 15 (both nibbles vary)
        // Byte i: high nibble = (i+1), low nibble = i
        for i in 0u8..16 {
            data[4 + i as usize] = ((i + 1) << 4) | i;
        }

        let result = dequantize_q4_0_cpu(&data, 32);

        // Expected: value = scale * ((value - 8))
        // First value (byte 0, low nibble): 0 -> 0 - 8 = -8, * 2.0 = -16.0
        assert!((result[0] - (-16.0)).abs() < 0.01);
        // Second value (byte 0, high nibble): 1 -> 1 - 8 = -7, * 2.0 = -14.0
        assert!((result[1] - (-14.0)).abs() < 0.01);
        // Last value (byte 15, high nibble): 0 -> 0 - 8 = -8, * 2.0 = -16.0
        assert!((result[31] - (-16.0)).abs() < 0.01);
        // Second to last value (byte 15, low nibble): 15 -> 15 - 8 = 7, * 2.0 = 14.0
        assert!((result[30] - 14.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q4_0_cpu_negative_scale() {
        // Test with negative scale
        let mut data = vec![0u8; 20];

        // Scale = -1.5
        data[0..4].copy_from_slice(&(-1.5f32).to_le_bytes());

        // All values = 8 (representing 0.0)
        for i in 0..16 {
            data[4 + i] = 0x88;
        }

        let result = dequantize_q4_0_cpu(&data, 32);

        // All values should be 0.0 (8 - 8 = 0, * -1.5 = 0)
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    #[test]
    fn test_dequantize_q4_0_cpu_partial_block() {
        // Test partial block (not multiple of 32 elements)
        let mut data = vec![0u8; 20];

        // Scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // All values = 8 (0.0 after dequant)
        for i in 0..16 {
            data[4 + i] = 0x88;
        }

        let result = dequantize_q4_0_cpu(&data, 10); // Only 10 elements

        // First 10 should be 0.0
        for i in 0..10 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    #[test]
    fn test_dequantize_q4_0_cpu_multiple_blocks() {
        // Test multiple blocks
        let n_elements = 64; // 2 blocks
        let n_blocks = (n_elements + 31) / 32;
        let block_size = 20;
        let mut data = vec![0u8; n_blocks * block_size];

        // Block 0: scale = 1.0, values = 8 (0.0)
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..16 {
            data[4 + i] = 0x88;
        }

        // Block 1: scale = 2.0, values = 12 (4.0)
        data[20..24].copy_from_slice(&2.0f32.to_le_bytes());
        for i in 0..16 {
            data[24 + i] = 0xCC; // Both nibbles = 12
        }

        let result = dequantize_q4_0_cpu(&data, n_elements);

        // First 32 should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }

        // Next 32 should be 8.0 (12 - 8 = 4, * 2.0 = 8.0)
        for i in 32..64 {
            assert!((result[i] - 8.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    /// Test GPU dequantization (requires GPU, marked as ignore)
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore]
    fn test_dequantize_q4_0_gpu_basic() {
        use crate::backend::HipBackend;

        // Create backend
        let backend = HipBackend::new().expect("Failed to create HIP backend");

        // Create test data: 1 block with scale=1.0, all values = 8
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        for i in 0..16 {
            data[4 + i] = 0x88;
        }

        // Allocate output buffer
        let output = backend
            .allocate_buffer(32 * 4)
            .expect("Failed to allocate output buffer");

        // Dequantize on GPU
        dequantize_q4_0_gpu(&backend, &data, &output, 32)
            .expect("Dequantization failed");

        // Synchronize and read back
        backend.synchronize().expect("Sync failed");
        let mut result = vec![0.0f32; 32];
        output.copy_to_host(&mut result).expect("Copy to host failed");

        // All values should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }
}
