//! Q8_0 quantized matmul operations
//!
//! # Format Specification
//! - Block size: 32 elements
//! - Per block: scale (f32, 4 bytes) + 32 bytes int8 values = 36 bytes
//! - Dequantization: value = scale * int8_value

use crate::backend::hip_backend::HipBackend;
use crate::backend::HipBuffer;

use super::common::{Q8_0_BLOCK_SIZE, Q8_0_ELEMENTS_PER_BLOCK, QuantizedResult};

/// Dequantize Q8_0 weights to f32 (CPU reference implementation)
///
/// # Format
/// - Each block has 32 int8 values
/// - Scale applies to all 32 values in the block
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * Q8_0_BLOCK_SIZE;
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

        // Read int8 values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..32 {
            let idx = data_start + i;
            if idx >= data.len() {
                break;
            }
            let elem_idx = base_idx + i;
            if elem_idx < n_elements {
                let int8_val = data[idx] as i8;
                result[elem_idx] = scale * int8_val as f32;
            }
        }
    }

    result
}

/// MatMul with Q8_0 quantized weights
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q8_0 quantized weight data
/// - `input`: Input tensor (f32)
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
///
/// # Note
/// Currently dequantizes on CPU then performs matmul on GPU.
/// TODO: Implement native HIP kernel for on-device dequantization.
pub fn matmul_q8_0(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &HipBuffer,
) -> QuantizedResult<()> {
    // Dequantize weights
    let n_elements = n_rows * n_cols;
    let dequant_weights = dequantize_q8_0(quantized_weights, n_elements);

    // Upload dequantized weights to GPU
    let weight_bytes = n_elements * 4;
    let weight_buffer = backend
        .allocate_buffer(weight_bytes)
        .map_err(|e| format!("Failed to allocate weight buffer: {}", e))?;

    weight_buffer
        .copy_from_host(&dequant_weights)
        .map_err(|e| format!("Failed to upload weights: {}", e))?;

    // Perform matmul using standard matmul op via ggml ops for backward compatibility
    let m = 1i32; // Input is typically [1, n_cols]
    let k = n_cols as i32;
    let n = n_rows as i32;

    crate::ggml::hip_backend::ops::matmul::matmul(
        backend,
        input,
        &weight_buffer,
        m,
        n,
        k,
        output,
    )
    .map_err(|e| format!("MatMul failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q8_0_simple() {
        // Create simple Q8_0 data: 1 block
        let mut data = vec![0u8; 36]; // 1 block * 36 bytes

        // Scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        // Int8 values: [-16, -15, ..., 15]
        for i in 0..32 {
            data[4 + i] = (i as i8 - 16) as u8;
        }

        let result = dequantize_q8_0(&data, 32);

        for i in 0..32 {
            let expected = (i as i8 - 16) as f32;
            assert!((result[i] - expected).abs() < 0.01, "result[{}]={}, expected={}", i, result[i], expected);
        }
    }
}
