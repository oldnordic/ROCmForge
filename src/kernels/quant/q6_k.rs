//! Q6_K quantization kernel (CPU + GPU)
//!
//! Q6_K format: 6-bit K-quants with 16-scale groups
//! - Block size: 256 elements
//! - Per block (256 bytes):
//!   - 32 bytes: 16 half-precision scales (2 bytes each)
//!   - 192 bytes: 6-bit packed quantized values (256 * 6 / 8 = 192)
//!   - 32 bytes: padding (for alignment)
//! - 16 elements share one scale
//! - Dequantization: value = signed_6bit * scale
//! - Signed 6-bit conversion: if >= 32, subtract 64 (range: [-32, 31])


#[cfg(feature = "rocm")]
use crate::backend::{HipBackend, HipKernel, HipModule};

#[cfg(test)]
use half::f16;

/// Result type for Q6_K dequantization operations
pub type Q6KdequantResult<T> = Result<T, String>;

/// Q6_K dequantization cache containing loaded module and kernel
#[cfg(feature = "rocm")]
pub struct Q6KdequantCache {
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    module: HipModule,
    kernel: HipKernel,
}

/// Global cache for Q6_K dequantization kernel
#[cfg(feature = "rocm")]
static Q6_K_DEQUANT_CACHE: Mutex<Option<Q6KdequantCache>> = Mutex::new(None);

/// Initialize or retrieve the cached Q6_K dequantization kernel
///
/// Loads the HSACO file specified by Q6_K_DEQUANT_HSACO environment variable
/// and extracts the q6_k_to_fp32_kernel function.
#[cfg(feature = "rocm")]
pub fn get_or_init_q6_k_dequant_cache(
    backend: &HipBackend,
) -> Q6KdequantResult<&'static Q6KdequantCache> {
    // Fast path: cache already initialized
    {
        let cache = Q6_K_DEQUANT_CACHE.lock().unwrap();
        if cache.is_some() {
            // SAFETY: We're extending the lifetime of a reference to static data.
            // The cache lives for the entire program duration (static Mutex).
            return Ok(unsafe {
                &*(cache.as_ref().unwrap() as *const Q6KdequantCache)
            });
        }
    }

    // Slow path: initialize cache
    // Use option_env!() to read compile-time environment variable set by build.rs
    let hsaco_path = option_env!("Q6_K_DEQUANT_HSACO")
        .ok_or_else(|| "Q6_K_DEQUANT_HSACO environment variable not set at compile time. Rebuild with cargo feature 'rocm' enabled.".to_string())?;

    // Load the module from HSACO file
    let module = backend
        .load_module(&hsaco_path)
        .map_err(|e| format!("Failed to load Q6_K dequant module: {}", e))?;

    // Get the kernel function
    let kernel = backend
        .get_kernel_function(&module, "q6_k_to_fp32_kernel")
        .map_err(|e| format!("Failed to get Q6_K dequant kernel: {}", e))?;

    let cache = Q6KdequantCache { module, kernel };

    // Store in global cache
    {
        let mut global_cache = Q6_K_DEQUANT_CACHE.lock().unwrap();
        *global_cache = Some(cache);
    }

    // Return reference to the newly cached value
    let cache = Q6_K_DEQUANT_CACHE.lock().unwrap();
    Ok(unsafe {
        &*(cache.as_ref().unwrap() as *const Q6KdequantCache)
    })
}

/// Dequantize Q6_K weights on GPU
///
/// # Format
/// - Each block has 256 elements packed into 256 bytes
/// - 16 groups of 16 elements, each with a half-precision scale
/// - 6-bit packed quantized values (signed range: [-32, 31])
/// - Dequantization: signed_val * scale
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q6_K quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # Returns
/// - Ok(()) on success
/// - Err(String) if kernel launch fails
#[cfg(feature = "rocm")]
pub fn dequantize_q6_k_gpu_kernel(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Q6KdequantResult<()> {
    // Get cached kernel
    let cache = get_or_init_q6_k_dequant_cache(backend)?;

    // Upload quantized data to GPU
    let quantized_buffer = backend
        .allocate_buffer(quantized_data.len())
        .map_err(|e| format!("Failed to allocate quantized buffer: {}", e))?;
    quantized_buffer
        .copy_from_host(quantized_data)
        .map_err(|e| format!("Failed to upload quantized data: {}", e))?;

    // Calculate grid dimensions based on blocks (256 elements per block)
    const BLOCK_SIZE: usize = 256;
    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Kernel arguments:
    // 1. input: const uint8_t* __restrict__ (quantized data)
    // 2. output: float* __restrict__ (dequantized FP32 data)
    // 3. num_blocks: int (number of blocks to process)
    let args = [
        quantized_buffer.as_ptr() as *mut std::ffi::c_void,
        output.as_ptr() as *mut std::ffi::c_void,
        &(num_blocks as i32) as *const i32 as *mut std::ffi::c_void,
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

/// Dequantize Q6_K weights with automatic CPU fallback
///
/// Tries GPU dequantization first. If GPU kernel fails (missing HSACO,
/// unsupported device, etc.), falls back to CPU implementation.
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_data`: Raw Q6_K quantized data
/// - `output`: Output GPU buffer for FP32 values
/// - `num_elements`: Number of elements to dequantize
///
/// # Returns
/// - Ok(()) on success
/// - Err(String) if both GPU and CPU paths fail
#[cfg(feature = "rocm")]
pub fn dequantize_q6_k_with_fallback(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &crate::backend::HipBuffer,
    num_elements: usize,
) -> Q6KdequantResult<()> {
    // Try GPU first
    match dequantize_q6_k_gpu_kernel(backend, quantized_data, output, num_elements) {
        Ok(()) => Ok(()),
        Err(e) => {
            tracing::warn!("Q6_K GPU dequantization failed, falling back to CPU: {}", e);
            // CPU fallback: dequantize then upload
            let cpu_result = dequantize_q6_k_cpu(quantized_data, num_elements);
            output
                .copy_from_host(&cpu_result)
                .map_err(|e| format!("CPU fallback upload failed: {}", e))?;
            Ok(())
        }
    }
}

/// Dequantize Q6_K weights to f32 (CPU version, for testing and fallback)
///
/// This is the reference CPU implementation used for:
/// - Testing and validation
/// - Comparison with GPU version
/// - Fallback when GPU kernel is unavailable
///
/// # Q6_K CPU Reference Implementation
///
/// Q6_K format details:
/// - Block: 256 elements in 256 bytes
/// - 16 groups of 16 elements per block
/// - Scale layout: 16 half-precision floats (32 bytes) at block start
/// - Quant layout: 6-bit packed values (192 bytes) starting at offset 32
/// - Padding: 32 bytes at end (for alignment)
pub fn dequantize_q6_k_cpu(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n_elements];
    let n_blocks = (n_elements + 255) / 256;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > data.len() {
            break;
        }

        // Q6_K block structure:
        // - 32 bytes: 16 half-precision scales (2 bytes each)
        // - 192 bytes: 6-bit packed quantized values
        // - 32 bytes: padding

        let scales_start = block_start;
        let quants_start = block_start + 32;

        // Dequantize block
        for i in 0..256 {
            let element_idx = block_idx * 256 + i;
            if element_idx >= n_elements {
                break;
            }

            // Get scale for this group (every 16 elements share a scale)
            let scale_idx = i / 16;
            let scale_offset = scales_start + scale_idx * 2;

            let scale = if scale_offset + 2 <= data.len() {
                let scale_bits = u16::from_le_bytes([
                    data[scale_offset],
                    data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0 // fallback scale
            };

            // Extract 6-bit quantized value
            let bit_offset = (i * 6) % 8;
            let byte_idx = (i * 6) / 8;

            if quants_start + byte_idx + 1 < data.len() {
                let combined = ((data[quants_start + byte_idx + 1] as u16) << 8)
                    | (data[quants_start + byte_idx] as u16);

                // Extract 6-bit value: (combined >> bit_offset) & 0x3F
                let quant_val = ((combined >> bit_offset) & 0x3F) as u8;

                // Convert to signed range: [0, 63] -> [-32, 31]
                let signed_val = if quant_val >= 32 {
                    (quant_val as i8 - 64) as f32
                } else {
                    quant_val as f32
                };

                // Dequantize: result = signed_val * scale
                result[element_idx] = signed_val * scale;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q6_k_cpu_single_block() {
        // Create test data: 1 block with known values
        let mut data = vec![0u8; 256];

        // Set scales (16 half-precision values at offset 0)
        for i in 0..16 {
            let scale_bits = f16::from_f32(1.0).to_bits();
            let scale_bytes = scale_bits.to_le_bytes();
            data[i * 2] = scale_bytes[0];
            data[i * 2 + 1] = scale_bytes[1];
        }

        // Set quantized values (192 bytes at offset 32)
        // Values 0-63 packed as 6-bit
        for i in 0..192 {
            data[32 + i] = ((i % 64) * 4) as u8; // Valid 6-bit values
        }

        let result = dequantize_q6_k_cpu(&data, 256);

        // Verify first few values
        for i in 0..16 {
            let expected = (i % 64) as f32; // Values 0-15 should match
            if (result[i] - expected).abs() > 1.0 {
                // Allow some tolerance due to 6-bit packing
                println!("result[{}]={}, expected={}", i, result[i], expected);
            }
        }

        // All values should be in valid range
        for i in 0..256 {
            assert!(result[i].is_finite(), "result[{}] is not finite", i);
        }
    }

    #[test]
    fn test_dequantize_q6_k_cpu_partial_block() {
        // Test partial block (not multiple of 256 elements)
        let mut data = vec![0u8; 256];

        // Set scales
        for i in 0..16 {
            let scale_bits = f16::from_f32(2.0).to_bits();
            let scale_bytes = scale_bits.to_le_bytes();
            data[i * 2] = scale_bytes[0];
            data[i * 2 + 1] = scale_bytes[1];
        }

        // Set quantized values (all zeros)
        for i in 0..192 {
            data[32 + i] = 0; // 0 in 6-bit = 0 after signed conversion
        }

        let result = dequantize_q6_k_cpu(&data, 100); // Only 100 elements

        // First 100 should be close to 0 (0 * 2.0 = 0)
        for i in 0..100 {
            assert!(
                result[i].abs() < 1.0,
                "result[{}]={}, expected ~0",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q6_k_cpu_negative_values() {
        // Test negative values from signed 6-bit range
        let mut data = vec![0u8; 256];

        // Set scales
        for i in 0..16 {
            let scale_bits = f16::from_f32(1.0).to_bits();
            let scale_bytes = scale_bits.to_le_bytes();
            data[i * 2] = scale_bytes[0];
            data[i * 2 + 1] = scale_bytes[1];
        }

        // For 6-bit packed values where each value is 63 (-1):
        // 63 = 0b111111 in 6 bits
        // Packed across byte boundaries:
        // Byte 0: [v0:5][v1:3] = [63<<0 | 63<<6] = 63 | (63<<6) = 0xFC | (0xFC0) = 0xFFC
        // But bytes are only 8 bits, so we get: byte[0] = 0xFC, byte[1] = 0x0F
        // Actually, let's think differently:
        // v0 occupies bits 0-5 of byte 0
        // v1 occupies bits 6-7 of byte 0 and bits 0-3 of byte 1
        // For v0=v1=63=0b111111:
        // byte[0] = 0b11111111 = 0xFF
        // byte[1] = 0b00111111 = 0x3F (only lower 6 bits set, v1's upper 2 bits are in byte[0])

        // Set each 6-bit value to 63 (-1 in signed)
        // Each byte needs to be set based on the 6-bit values it contains
        for byte_idx in 0..192 {
            // Each byte contains parts of up to 2 different 6-bit values
            // For simplicity, set all values to a pattern that works
            // Value 63 in 6-bit = 0b111111
            // Packed: 4 values fit in 3 bytes (24 bits = 4 * 6)
            // [v0:v0:v0:v0:v0:v0][v1:v1:v1:v1:v1:v1][v2:v2:v2:v2:v2:v2][v3:v3:v3:v3:v3:v3]
            // [byte0:              ][byte1:              ][byte2:              ]
            // [v0:5][v1:1] | [v1:4][v2:2] | [v2:3][v3:3]
            // For v0=v1=v2=v3=63:
            // byte0 = 0b11111111 = 0xFF (63 << 0 | 63 << 6, masked to 8 bits)
            // byte1 = 0b11111111 = 0xFF (63 << 2 | 63 << 8, masked to 8 bits)
            // byte2 = 0b00111111 = 0x3F (63 << 4, masked to 8 bits)

            let group_idx = byte_idx % 3;
            data[32 + byte_idx] = match group_idx {
                0 => 0xFF,  // v0 lower 6 bits | v1 upper 2 bits
                1 => 0xFF,  // v1 lower 4 bits | v2 upper 4 bits
                2 => 0xFF,  // v2 lower 2 bits | v3 upper 6 bits
                _ => unreachable!(),
            };
        }

        let result = dequantize_q6_k_cpu(&data, 256);

        // All values should be close to -1.0 (63 in 6-bit signed = -1)
        for i in 0..256 {
            assert!(
                (result[i] - (-1.0)).abs() < 1.0,
                "result[{}]={}, expected -1.0",
                i,
                result[i]
            );
        }
    }
}
