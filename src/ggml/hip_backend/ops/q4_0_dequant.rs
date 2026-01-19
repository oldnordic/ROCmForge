//! Q4_0 dequantization operations for GPU
//!
//! PHASE 17: GPU-side Q4_0 dequantization
//!
//! This module provides GPU dequantization for Q4_0 quantized weights.
//! Uses precompiled HIP kernel for on-device dequantization with CPU fallback.

use std::env;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};

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

// Constants for Q4_0 format
#[allow(dead_code)] // Reserved for future Q4_0 format validation
const Q4_0_BLOCK_SIZE: usize = 20;  // 4 bytes scale + 16 bytes packed data
#[allow(dead_code)] // Reserved for future Q4_0 format validation
const Q4_0_ELEMENTS_PER_BLOCK: usize = 32;

/// Cached kernel modules and functions for Q4_0 dequantization
#[derive(Debug)]
pub struct Q4_0DequantCache {
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    module: Option<HipModule>,
    kernel: Option<HipKernel>,
}

// Global kernel cache for Q4_0 dequantization
static Q4_0_DEQUANT_CACHE: Mutex<Option<Q4_0DequantCache>> = Mutex::new(None);

/// Get or initialize the global Q4_0 dequantization kernel cache
///
/// Loads the HSACO kernel from the path specified by Q4_0_DEQUANT_HSACO env var.
/// Returns error if env var not set or HSACO file not found (graceful degradation).
pub fn get_or_init_q4_0_dequant_cache() -> Result<&'static Mutex<Option<Q4_0DequantCache>>, HipError> {
    // First check if already initialized
    {
        let cache = Q4_0_DEQUANT_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("Q4_0_DEQUANT_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&Q4_0_DEQUANT_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = Q4_0_DEQUANT_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q4_0_DEQUANT_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&Q4_0_DEQUANT_CACHE);
    }

    // Load Q4_0 dequantization kernel module and function
    let load_backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e))
    })?;

    let kernel_path = env::var("Q4_0_DEQUANT_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("Q4_0_DEQUANT_HSACO env var not set".to_string()))?;

    if !Path::new(&kernel_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "Q4_0 dequant HSACO not found: {}",
            kernel_path
        )));
    }

    let module = load_backend.load_module(&kernel_path)?;
    let kernel = load_backend.get_kernel_function(&module, "q4_0_to_fp32_batch_kernel")?;

    *cache = Some(Q4_0DequantCache {
        module: Some(module),
        kernel: Some(kernel),
    });

    Ok(&Q4_0_DEQUANT_CACHE)
}

/// Dequantize Q4_0 weights using cached GPU kernel
///
/// Uploads quantized data to GPU, launches dequantization kernel, and
/// produces FP32 output on device.
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q4_0 quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # GPU Kernel
/// Uses q4_0_to_fp32_batch_kernel with:
/// - Block size: 256 threads
/// - Grid size: (num_elements + 255) / 256
#[cfg(feature = "rocm")]
pub fn dequantize_q4_0_kernel_cached(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Result<(), HipError> {
    let cache_ref = get_or_init_q4_0_dequant_cache()?;
    let cache = cache_ref
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q4_0 cache lock poisoned: {}", e)))?;
    let cache_ref = cache
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("Q4_0 cache not initialized".to_string()))?;

    let kernel = cache_ref
        .kernel
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("q4_0_to_fp32_batch_kernel not loaded".to_string()))?;

    // Allocate GPU buffer for input (quantized data)
    let input_buffer = backend.allocate_buffer(quantized_data.len())?;
    input_buffer.copy_from_host(quantized_data)?;

    // Calculate grid dimensions
    let block_size = 256u32;
    let grid_size = ((num_elements as u32) + block_size - 1) / block_size;

    // Prepare kernel arguments
    let mut input_arg = input_buffer.as_ptr() as *const u8;
    let mut output_arg = output.as_mut_ptr() as *mut f32;
    let mut count_arg = num_elements as i32;

    let args: &[*mut c_void] = &[
        &mut input_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut count_arg as *mut _ as *mut c_void,
    ];

    // Launch kernel
    backend
        .launch_kernel_with_module_shared(kernel, (grid_size, 1, 1), (block_size, 1, 1), args, 0)?;

    // Synchronize to ensure completion
    backend.synchronize()?;

    Ok(())
}

/// Dequantize Q4_0 weights using CPU-side dequantization then GPU upload
///
/// This is the fallback path when GPU kernel is unavailable.
/// Dequantizes on CPU, then uploads FP32 data to GPU.
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q4_0 quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # Note
/// This uses ~17x more memory bandwidth than GPU kernel path.
/// Prefer using `dequantize_q4_0_kernel_cached` when available.
pub fn dequantize_q4_0_cpu_upload(
    _backend: &HipBackend,
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

/// Dequantize Q4_0 weights with automatic GPU/CPU fallback
///
/// Tries GPU kernel first, falls back to CPU upload if GPU unavailable.
/// Provides graceful degradation when HSACO file is missing or GPU unavailable.
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q4_0 quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
pub fn dequantize_q4_0_with_fallback(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Q4_0DequantResult<()> {
    #[cfg(feature = "rocm")]
    {
        // Try GPU kernel first
        match dequantize_q4_0_kernel_cached(backend, quantized_data, output, num_elements) {
            Ok(()) => Ok(()),
            Err(HipError::KernelLoadFailed(_)) | Err(HipError::InitializationFailed(_)) => {
                // Fall back to CPU path if kernel not available
                dequantize_q4_0_cpu_upload(backend, quantized_data, output, num_elements)
            }
            Err(e) => Err(format!("GPU dequantization failed: {}", e)),
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        // No ROCm support, use CPU path
        dequantize_q4_0_cpu_upload(backend, quantized_data, output, num_elements)
    }
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
}
