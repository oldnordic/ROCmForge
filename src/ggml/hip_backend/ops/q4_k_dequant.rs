//! Q4_K dequantization operations for GPU
//!
//! PHASE 17: GPU-side Q4_K dequantization
//!
//! This module provides GPU dequantization for Q4_K quantized weights.
//! Q4_K is a "K-quant" format optimized for quality/size trade-off.
//!
//! # Q4_K Format Specification
//!
//! - Super-block size: 256 elements (8 sub-blocks of 32 elements each)
//! - Per super-block (256 bytes):
//!   - 16 bytes: 8 half-precision scales (2 bytes each) for 8 sub-blocks
//!   - 16 bytes: 8 int8 mins (1 byte each) for 8 sub-blocks
//!   - 224 bytes: 8 sub-blocks of 4-bit quantized values (28 bytes each, packed)
//! - Each sub-block (32 elements): scale (f16) + min (int8) + 28 bytes packed 4-bit values
//! - Dequantization: value = min + (quant * scale)

use std::env;
use std::sync::Mutex;

use crate::backend::{HipBackend, HipKernel, HipModule};

/// Result type for Q4_K dequantization operations
pub type Q4_KDequantResult<T> = Result<T, String>;

/// Q4_K dequantization cache containing loaded module and kernel
pub struct Q4_KDequantCache {
    #[allow(dead_code)]
    module: HipModule,
    kernel: HipKernel,
}

/// Global cache for Q4_K dequantization kernel
static Q4_K_DEQUANT_CACHE: Mutex<Option<Q4_KDequantCache>> = Mutex::new(None);

/// Initialize or retrieve the cached Q4_K dequantization kernel
///
/// Loads the HSACO file specified by Q4_K_DEQUANT_HSACO environment variable
/// and extracts the q4_k_to_fp32_kernel function.
pub fn get_or_init_q4_k_dequant_cache(backend: &HipBackend) -> Q4_KDequantResult<&'static Q4_KDequantCache> {
    // Fast path: cache already initialized
    {
        let cache = Q4_K_DEQUANT_CACHE.lock().unwrap();
        if cache.is_some() {
            // SAFETY: We're extending the lifetime of a reference to static data.
            // The cache lives for the entire program duration (static Mutex).
            return Ok(unsafe {
                &*(cache.as_ref().unwrap() as *const Q4_KDequantCache)
            });
        }
    }

    // Slow path: initialize cache
    let hsaco_path = env::var("Q4_K_DEQUANT_HSACO")
        .map_err(|_| "Q4_K_DEQUANT_HSACO environment variable not set".to_string())?;

    // Load the module from HSACO file
    let module = backend
        .load_module(&hsaco_path)
        .map_err(|e| format!("Failed to load Q4_K dequant module: {}", e))?;

    // Get the kernel function
    let kernel = backend
        .get_kernel_function(&module, "q4_k_to_fp32_kernel")
        .map_err(|e| format!("Failed to get Q4_K dequant kernel: {}", e))?;

    let cache = Q4_KDequantCache { module, kernel };

    // Store in global cache
    {
        let mut global_cache = Q4_K_DEQUANT_CACHE.lock().unwrap();
        *global_cache = Some(cache);
    }

    // Return reference to the newly cached value
    let cache = Q4_K_DEQUANT_CACHE.lock().unwrap();
    Ok(unsafe {
        &*(cache.as_ref().unwrap() as *const Q4_KDequantCache)
    })
}

/// Dequantize Q4_K weights on GPU
///
/// # Format
/// - Each super-block has 256 elements packed into 256 bytes
/// - 8 sub-blocks of 32 elements each
/// - Each sub-block: scale (f16) + min (int8) + 28 bytes of 4-bit packed values
/// - Dequantization: value = min + (quant * scale)
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q4_K quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # Returns
/// - Ok(()) on success
/// - Err(String) if kernel launch fails
pub fn dequantize_q4_k_gpu_kernel(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Q4_KDequantResult<()> {
    // Get cached kernel
    let cache = get_or_init_q4_k_dequant_cache(backend)?;

    // Upload quantized data to GPU
    let quantized_buffer = backend
        .allocate_buffer(quantized_data.len())
        .map_err(|e| format!("Failed to allocate quantized buffer: {}", e))?;
    quantized_buffer
        .copy_from_host(quantized_data)
        .map_err(|e| format!("Failed to upload quantized data: {}", e))?;

    // Calculate grid dimensions based on super-blocks (256 elements per block)
    const BLOCK_SIZE: usize = 256;
    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Kernel arguments:
    // 1. input: const uint8_t* __restrict__ (quantized data)
    // 2. output: float* __restrict__ (dequantized FP32 data)
    // 3. num_super_blocks: int (number of super-blocks to process)
    let num_super_blocks = num_blocks;

    let args = [
        quantized_buffer.as_ptr() as *mut std::ffi::c_void,
        output.as_ptr() as *mut std::ffi::c_void,
        &(num_super_blocks as i32) as *const i32 as *mut std::ffi::c_void,
    ];

    // Launch kernel
    backend
        .launch_kernel_with_module(
            &cache.kernel,
            (num_blocks as u32, 1, 1),  // grid_dim
            (BLOCK_SIZE as u32, 1, 1),   // block_dim (256 threads per block)
            &args,
        )
        .map_err(|e| format!("Kernel launch failed: {}", e))?;

    // Synchronize to ensure completion
    backend
        .synchronize()
        .map_err(|e| format!("Synchronization failed: {}", e))?;

    Ok(())
}

/// Dequantize Q4_K weights with automatic CPU fallback
///
/// Tries GPU dequantization first. If GPU kernel fails (missing HSACO,
/// unsupported device, etc.), falls back to CPU implementation.
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q4_K quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # Returns
/// - Ok(()) on success
/// - Err(String) if both GPU and CPU paths fail
pub fn dequantize_q4_k_with_fallback(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Q4_KDequantResult<()> {
    // Try GPU first
    match dequantize_q4_k_gpu_kernel(backend, quantized_data, output, num_elements) {
        Ok(()) => Ok(()),
        Err(e) => {
            tracing::warn!("Q4_K GPU dequantization failed, falling back to CPU: {}", e);
            // CPU fallback: dequantize then upload
            let cpu_result = dequantize_q4_k_cpu(quantized_data, num_elements);
            output
                .copy_from_host(&cpu_result)
                .map_err(|e| format!("CPU fallback upload failed: {}", e))?;
            Ok(())
        }
    }
}

/// Dequantize Q4_K weights to f32 (CPU version, for testing and fallback)
///
/// This is the reference CPU implementation used for:
/// - Testing and validation
/// - Comparison with GPU version
/// - Fallback when GPU kernel is unavailable
///
/// # Q4_K CPU Reference Implementation
///
/// Q4_K format details:
/// - Super-block: 256 elements in 256 bytes
/// - 8 sub-blocks per super-block (32 elements each)
/// - Sub-block layout:
///   - Scale: 2 bytes (half-precision float)
///   - Min: 1 byte (int8)
///   - Quantized values: 28 bytes (32 * 4 bits)
pub fn dequantize_q4_k_cpu(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n_elements];
    let n_super_blocks = (n_elements + 255) / 256;

    for super_block_idx in 0..n_super_blocks {
        let super_block_offset = super_block_idx * 256;

        if super_block_offset + 256 > data.len() {
            break;
        }

        // Q4_K super-block structure:
        // - 16 bytes: 8 half-precision scales (2 bytes each)
        // - 16 bytes: 8 int8 mins (1 byte each)
        // - 224 bytes: 8 sub-blocks of 4-bit quantized values (28 bytes each)

        let scales_start = super_block_offset;
        let mins_start = super_block_offset + 16;
        let quants_start = super_block_offset + 32;

        // Process each of the 8 sub-blocks
        for sub_block_idx in 0..8 {
            let sub_block_start = super_block_idx * 256 + sub_block_idx * 32;

            if sub_block_start >= n_elements {
                break;
            }

            // Get scale for this sub-block (half-precision)
            let scale_offset = scales_start + sub_block_idx * 2;
            let scale = if scale_offset + 2 <= data.len() {
                let scale_bits = u16::from_le_bytes([
                    data[scale_offset],
                    data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0
            };

            // Get min for this sub-block (int8)
            let min_offset = mins_start + sub_block_idx;
            let min = if min_offset < data.len() {
                data[min_offset] as i8 as f32
            } else {
                0.0
            };

            // Extract 4-bit quantized values for this sub-block (32 values)
            for i in 0..32 {
                let element_idx = sub_block_start + i;
                if element_idx >= n_elements {
                    break;
                }

                let bit_pos = i * 4;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quant_offset = quants_start + sub_block_idx * 28 + byte_idx;
                if quant_offset + 1 < data.len() {
                    let combined = (data[quant_offset + 1] as u16) << 8 | (data[quant_offset] as u16);
                    let quant = ((combined >> bit_offset) & 0x0F) as u8;

                    // Dequantize: value = min + (quant * scale)
                    result[element_idx] = min + (quant as f32) * scale;
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q4_k_cpu_single_block() {
        // Create test data: 1 super-block with known values
        let mut data = vec![0u8; 256];

        // Set scales (8 half-precision values at offset 0)
        for i in 0..8 {
            let scale_bits = 1.0f32.to_f16().to_bits();
            let scale_bytes = scale_bits.to_le_bytes();
            data[i * 2] = scale_bytes[0];
            data[i * 2 + 1] = scale_bytes[1];
        }

        // Set mins (8 int8 values at offset 16)
        for i in 0..8 {
            data[16 + i] = 0;
        }

        // Set quantized values (8 * 28 = 224 bytes at offset 32)
        for i in 0..224 {
            data[32 + i] = ((i % 16) << 4) | (i % 16);
        }

        let result = dequantize_q4_k_cpu(&data, 256);

        // All values should be 0 + quant * 1.0 = quant
        for i in 0..256 {
            let sub_block = i / 32;
            let elem_in_sub = i % 32;
            let byte_idx = elem_in_sub / 8;
            let bit_offset = (elem_in_sub * 4) % 8;
            let quant_offset = sub_block * 28 + byte_idx;
            let combined = ((data[32 + quant_offset + 1] as u16) << 8) | (data[32 + quant_offset] as u16);
            let expected_quant = ((combined >> bit_offset) & 0x0F) as f32;
            let expected = expected_quant;
            assert!(
                (result[i] - expected).abs() < 0.01,
                "result[{}]={}, expected={}",
                i,
                result[i],
                expected
            );
        }
    }

    #[test]
    fn test_dequantize_q4_k_cpu_partial_block() {
        // Test partial super-block (not multiple of 256 elements)
        let mut data = vec![0u8; 256];

        // Set scales
        for i in 0..8 {
            let scale_bits = 1.0f32.to_f16().to_bits();
            let scale_bytes = scale_bits.to_le_bytes();
            data[i * 2] = scale_bytes[0];
            data[i * 2 + 1] = scale_bytes[1];
        }

        // Set mins
        for i in 0..8 {
            data[16 + i] = 0;
        }

        // Set quantized values
        for i in 0..224 {
            data[32 + i] = 0x88; // Both nibbles = 8
        }

        let result = dequantize_q4_k_cpu(&data, 100); // Only 100 elements

        // First 100 should be 8.0
        for i in 0..100 {
            let sub_block = i / 32;
            let elem_in_sub = i % 32;
            let byte_idx = elem_in_sub / 8;
            let bit_offset = (elem_in_sub * 4) % 8;
            let quant_offset = sub_block * 28 + byte_idx;
            let combined = ((data[32 + quant_offset + 1] as u16) << 8) | (data[32 + quant_offset] as u16);
            let expected_quant = ((combined >> bit_offset) & 0x0F) as f32;
            let expected = expected_quant;
            assert!(
                (result[i] - expected).abs() < 0.01,
                "result[{}]={}, expected={}",
                i,
                result[i],
                expected
            );
        }
    }
}
