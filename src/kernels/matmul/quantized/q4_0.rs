//! Q4_0 quantized matmul operations
//!
//! # Format Specification
//! - Block size: 32 elements
//! - Per block: scale (f32, 4 bytes) + 16 bytes 4-bit packed values = 20 bytes
//! - Dequantization: value = scale * ((packed & 0x0F) - 8)

use std::env;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};
use crate::backend::HipBuffer;

use super::common::{Q4_0_BLOCK_SIZE, Q4_0_ELEMENTS_PER_BLOCK, QuantizedResult};

/// Cached kernel modules and functions for Q4_0 matmul
#[derive(Debug)]
struct Q4_0KernelCache {
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    module: Option<HipModule>,
    kernel: Option<HipKernel>,
}

// Global kernel cache
static Q4_0_CACHE: Mutex<Option<Q4_0KernelCache>> = Mutex::new(None);

/// Get or initialize the global Q4_0 kernel cache
fn get_or_init_q4_0_cache() -> Result<&'static Mutex<Option<Q4_0KernelCache>>, HipError> {
    // First check if already initialized
    {
        let cache = Q4_0_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("Q4_0_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&Q4_0_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = Q4_0_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q4_0_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&Q4_0_CACHE);
    }

    // Load Q4_0 matmul kernel module and function
    let load_backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e))
    })?;

    let kernel_path = env::var("Q4_0_MATMUL_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("Q4_0_MATMUL_HSACO env var not set".to_string()))?;

    if !Path::new(&kernel_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "Q4_0 matmul HSACO not found: {}",
            kernel_path
        )));
    }

    let module = load_backend.load_module(&kernel_path)?;
    let kernel = load_backend.get_kernel_function(&module, "q4_0_matmul_kernel")?;

    *cache = Some(Q4_0KernelCache {
        module: Some(module),
        kernel: Some(kernel),
    });

    Ok(&Q4_0_CACHE)
}

/// Dequantize Q4_0 weights to f32 (CPU reference implementation)
///
/// # Format
/// - Each block has 32 values packed into 16 bytes
/// - Each value is 4 bits, interpreted as signed: unpacked - 8
/// - Scale applies to all 32 values in the block
pub fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * Q4_0_BLOCK_SIZE;
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

/// MatMul with Q4_0 quantized weights using fused dequant+matmul kernel
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q4_0 quantized weight data [n_rows x n_cols]
/// - `input`: Input tensor (f32), typically [1 x n_cols]
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
///
/// # Memory Layout
/// - Input: [M x K] row-major (FP32), typically M=1, K=n_cols
/// - Weights: [n_rows x n_cols] row-major (Q4_0 format)
/// - Output: [M x n_rows] row-major (FP32)
///
/// # GPU Implementation
/// Uses fused kernel that dequantizes weights on-the-fly during matmul,
/// eliminating the intermediate FP32 weight buffer.
#[cfg(feature = "rocm")]
pub fn matmul_q4_0(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &HipBuffer,
) -> QuantizedResult<()> {
    // For existing API: weights are [n_rows x n_cols], input is [1 x n_cols]
    // We compute: output[1 x n_rows] = input[1 x n_cols] @ weights.T[n_cols x n_rows]
    //
    // For the fused kernel, we need to transpose:
    // - m = 1 (single token)
    // - n = n_rows (output dimension)
    // - k = n_cols (inner dimension)

    let m = 1;
    let n = n_rows;
    let k = n_cols;

    // Upload quantized weights to GPU (no dequantization needed)
    let weight_bytes = quantized_weights.len();
    let weight_buffer = backend
        .allocate_buffer(weight_bytes)
        .map_err(|e| format!("Failed to allocate weight buffer: {}", e))?;

    weight_buffer
        .copy_from_host(quantized_weights)
        .map_err(|e| format!("Failed to upload weights: {}", e))?;

    // Get input device pointer
    let input_ptr = input.as_ptr() as *const f32;

    // Get output device pointer
    let output_ptr = output.as_mut_ptr() as *mut f32;

    // Get weight device pointer
    let weight_ptr = weight_buffer.as_ptr() as *const u8;

    // Launch fused kernel
    unsafe {
        matmul_q4_0_gpu(backend, input_ptr, weight_ptr, output_ptr, m, n, k)
            .map_err(|e| format!("Fused matmul failed: {}", e))?;
    }

    backend
        .synchronize()
        .map_err(|e| format!("Failed to synchronize: {}", e))?;

    Ok(())
}

/// MatMul with Q4_0 quantized weights (non-rocm fallback)
///
/// This version is used when the rocm feature is not enabled.
/// It uses CPU dequantization followed by the standard matmul operation.
#[cfg(not(feature = "rocm"))]
pub fn matmul_q4_0(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &HipBuffer,
) -> QuantizedResult<()> {
    // Use CPU fallback
    matmul_q4_0_cpu_fallback(backend, quantized_weights, input, n_rows, n_cols, output)
}

/// Launch Q4_0 fused dequant+matmul kernel
///
/// # Safety
/// Caller must ensure all pointers are valid and synchronized.
#[cfg(feature = "rocm")]
pub(crate) unsafe fn matmul_q4_0_gpu(
    backend: &HipBackend,
    activations: *const f32,
    weights_q4_0: *const u8,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), HipError> {
    let cache_ref = get_or_init_q4_0_cache()?;
    let cache = cache_ref
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q4_0 cache lock poisoned: {}", e)))?;
    let cache_ref = cache
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("Q4_0 cache not initialized".to_string()))?;

    let kernel = cache_ref
        .kernel
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("q4_0_matmul_kernel not loaded".to_string()))?;

    // Calculate grid/block dimensions
    // Grid: M blocks (one per output row)
    // Block: 256 threads
    let block_dim = (256, 1, 1);
    let grid_dim = (m as u32, 1, 1);
    let shared_mem_bytes = 0;

    // Prepare kernel arguments - ALL args must be copied to mut locals first
    let mut activations_arg = activations;
    let mut weights_arg = weights_q4_0;
    let mut output_arg = output;
    let mut m_arg = m as i32;
    let mut n_arg = n as i32;
    let mut k_arg = k as i32;

    let args: &[*mut c_void] = &[
        &mut activations_arg as *mut _ as *mut c_void,
        &mut weights_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut m_arg as *mut _ as *mut c_void,
        &mut n_arg as *mut _ as *mut c_void,
        &mut k_arg as *mut _ as *mut c_void,
    ];

    backend
        .launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, shared_mem_bytes)
        .map_err(|e| HipError::KernelLaunchFailed(format!("Failed to launch q4_0_matmul kernel: {:?}", e)))?;

    Ok(())
}

/// MatMul with Q4_0 quantized weights (CPU fallback)
///
/// This is the CPU fallback for when GPU is unavailable or for small matrices
/// where the GPU overhead isn't worth it.
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q4_0 quantized weight data
/// - `input`: Input tensor (f32)
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
pub fn matmul_q4_0_cpu_fallback(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &HipBuffer,
) -> QuantizedResult<()> {
    // Dequantize weights on CPU
    let n_elements = n_rows * n_cols;
    let dequant_weights = dequantize_q4_0(quantized_weights, n_elements);

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
    fn test_dequantize_q4_0_simple() {
        // Create simple Q4_0 data: 1 block with 32 identical values
        // Q4_0: 4-bit values (0-15), dequantized as (value - 8) * scale
        let mut data = vec![0u8; 20]; // 1 block * 20 bytes

        // Block 0: scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // Pack 32 values, all stored as 8 (representing 0 after dequantization)
        // Each byte holds two 4-bit values: low nibble + high nibble
        for i in 0..16 {
            data[4 + i] = 0x88; // Both nibbles = 8, representing 0.0
        }

        let result = dequantize_q4_0(&data, 32);

        // All 32 values should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    /// Test fused kernel against CPU reference
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore] // Requires GPU hardware
    fn test_q4_0_matmul_fused_vs_reference() {
        use crate::backend::HipBackend;

        // Create backend
        let backend = HipBackend::new().expect("Failed to create HIP backend");

        // Test dimensions: small for quick testing
        let m = 2;  // 2 rows
        let n = 4;  // 4 columns
        let k = 8;  // 8 inner dimension

        // Create test activations [M x K]
        let activations: Vec<f32> = (0..(m * k))
            .map(|i| (i as f32) * 0.1)
            .collect();

        // Create test Q4_0 weights [K x N]
        // For simplicity, use constant scale and vary quantized values
        let n_weight_elements = k * n;
        let n_weight_blocks = (n_weight_elements + 31) / 32;
        let mut quantized_weights = vec![0u8; n_weight_blocks * Q4_0_BLOCK_SIZE];

        // Fill each block with scale=1.0 and quantized values 0-15
        for block_idx in 0..n_weight_blocks {
            let block_offset = block_idx * Q4_0_BLOCK_SIZE;
            quantized_weights[block_offset..block_offset + 4]
                .copy_from_slice(&1.0f32.to_le_bytes());

            for i in 0..16 {
                let val = ((block_idx * 16 + i) % 16) as u8;
                quantized_weights[block_offset + 4 + i] = (val << 4) | val;
            }
        }

        // Allocate GPU buffers
        let input_buffer = backend
            .allocate_buffer(activations.len() * 4)
            .expect("Failed to allocate input buffer");
        let output_buffer = backend
            .allocate_buffer(m * n * 4)
            .expect("Failed to allocate output buffer");
        let output_ref_buffer = backend
            .allocate_buffer(m * n * 4)
            .expect("Failed to allocate reference output buffer");

        // Upload activations
        input_buffer
            .copy_from_host(&activations)
            .expect("Failed to upload activations");

        // Compute using fused kernel
        matmul_q4_0(
            &backend,
            &quantized_weights,
            &input_buffer,
            n,  // n_rows
            k,  // n_cols
            &output_buffer,
        )
        .expect("Fused matmul failed");

        // Compute using CPU reference
        matmul_q4_0_cpu_fallback(
            &backend,
            &quantized_weights,
            &input_buffer,
            n,
            k,
            &output_ref_buffer,
        )
        .expect("CPU reference matmul failed");

        // Read back results
        let mut result = vec![0.0f32; m * n];
        let mut result_ref = vec![0.0f32; m * n];
        backend
            .copy_from_device_safe(&output_buffer, &mut result)
            .expect("Failed to copy result");
        backend
            .copy_from_device_safe(&output_ref_buffer, &mut result_ref)
            .expect("Failed to copy reference result");

        // Compare with tolerance for floating-point differences
        for i in 0..(m * n) {
            let diff = (result[i] - result_ref[i]).abs();
            assert!(
                diff < 0.1,
                "Mismatch at index {}: fused={}, ref={}, diff={}",
                i, result[i], result_ref[i], diff
            );
        }
    }
}
