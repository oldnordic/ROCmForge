//! Q6_K quantized matmul operations
//!
//! # Format Specification
//! - Block size: 256 elements
//! - Per block (256 bytes): 32 bytes scales + 192 bytes 6-bit quants + 32 bytes padding
//! - Dequantization: value = signed_6bit * scale

use std::env;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};
use crate::backend::HipBuffer;

use super::common::{f16_to_f32_cpu, Q6_K_BLOCK_SIZE, Q6_K_ELEMENTS_PER_BLOCK, QuantizedResult};

/// Cached kernel modules and functions for Q6_K matmul
///
/// This cache infrastructure was part of Phase 24 kernel migration.
/// It is reserved for future HSACO lazy-loading optimization but currently
/// unused because kernels are loaded directly via HipModule::from_file().
#[derive(Debug)]
#[allow(non_camel_case_types)] // Matches GGUF quantization format naming
#[allow(dead_code)] // Reserved for future HSACO lazy-loading optimization
struct Q6_KKernelCache {
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    module: Option<HipModule>,
    kernel: Option<HipKernel>,
}

/// Global kernel cache for lazy-loaded HSACO modules
#[allow(dead_code)] // Reserved for future HSACO lazy-loading optimization
static Q6_K_CACHE: Mutex<Option<Q6_KKernelCache>> = Mutex::new(None);

/// Get or initialize the global Q6_K kernel cache
#[allow(dead_code)] // Reserved for future HSACO lazy-loading optimization
fn get_or_init_q6_k_cache() -> Result<&'static Mutex<Option<Q6_KKernelCache>>, HipError> {
    // First check if already initialized
    {
        let cache = Q6_K_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("Q6_K_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&Q6_K_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = Q6_K_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q6_K_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&Q6_K_CACHE);
    }

    // Load Q6_K matmul kernel module and function
    let load_backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e))
    })?;

    let kernel_path = env::var("Q6_K_MATMUL_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("Q6_K_MATMUL_HSACO env var not set".to_string()))?;

    if !Path::new(&kernel_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "Q6_K matmul HSACO not found: {}",
            kernel_path
        )));
    }

    let module = load_backend.load_module(&kernel_path)?;
    let kernel = load_backend.get_kernel_function(&module, "q6_k_matmul_kernel")?;

    *cache = Some(Q6_KKernelCache {
        module: Some(module),
        kernel: Some(kernel),
    });

    Ok(&Q6_K_CACHE)
}

/// Dequantize Q6_K weights to f32 (CPU reference implementation)
///
/// # Format
/// - Block size: 256 elements
/// - Per block (256 bytes): 32 bytes scales + 192 bytes 6-bit quants + 32 bytes padding
/// - Dequantization: value = signed_6bit * scale
pub fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + Q6_K_ELEMENTS_PER_BLOCK - 1) / Q6_K_ELEMENTS_PER_BLOCK;
    let mut result = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * Q6_K_BLOCK_SIZE;
        if block_offset + 32 > data.len() {
            break;
        }

        for i in 0..Q6_K_ELEMENTS_PER_BLOCK {
            let elem_idx = block_idx * Q6_K_ELEMENTS_PER_BLOCK + i;
            if elem_idx >= n_elements {
                break;
            }

            // Calculate which scale to use (16 elements per scale)
            let scale_idx = i / 16;
            let scale_offset = block_offset + scale_idx * 2;

            if scale_offset + 2 > data.len() {
                break;
            }

            let scale_bits = u16::from_le_bytes([
                data[scale_offset],
                data[scale_offset + 1],
            ]);
            let scale = f16_to_f32_cpu(scale_bits);

            // Extract 6-bit value
            let quants_offset = block_offset + 32;
            let bit_pos = i * 6;
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            let q_idx = quants_offset + byte_idx;
            if q_idx + 1 >= data.len() {
                break;
            }

            let combined = ((data[q_idx + 1] as u16) << 8) | (data[q_idx] as u16);
            let quant = ((combined >> bit_offset) & 0x3F) as i32;

            // Convert to signed: 0-31 -> 0 to 31, 32-63 -> -32 to -1
            let signed_val = if quant >= 32 { quant - 64 } else { quant };

            result[elem_idx] = signed_val as f32 * scale;
        }
    }

    result
}

/// MatMul with Q6_K quantized weights using fused dequant+matmul kernel
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q6_K quantized weight data [n_rows x n_cols]
/// - `input`: Input tensor (f32), typically [1 x n_cols]
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
///
/// # Q6_K Format
/// - Block size: 256 elements
/// - Per block (256 bytes): 32 bytes scales + 192 bytes 6-bit quants + 32 bytes padding
/// - Dequantization: value = signed_6bit * scale
///
/// # Memory Layout
/// - Input: [M x K] row-major (FP32), typically M=1, K=n_cols
/// - Weights: [n_rows x n_cols] row-major (Q6_K format)
/// - Output: [M x n_rows] row-major (FP32)
pub fn matmul_q6_k(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &HipBuffer,
) -> QuantizedResult<()> {
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
        matmul_q6_k_gpu(backend, input_ptr, weight_ptr, output_ptr, m, n, k)
            .map_err(|e| format!("Fused Q6_K matmul failed: {}", e))?;
    }

    backend
        .synchronize()
        .map_err(|e| format!("Failed to synchronize: {}", e))?;

    Ok(())
}

/// Launch Q6_K fused dequant+matmul kernel
///
/// # Safety
/// Caller must ensure all pointers are valid and synchronized.
unsafe fn matmul_q6_k_gpu(
    backend: &HipBackend,
    activations: *const f32,
    weights_q6_k: *const u8,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), HipError> {
    let cache_ref = get_or_init_q6_k_cache()?;
    let cache = cache_ref
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q6_K cache lock poisoned: {}", e)))?;
    let cache_ref = cache
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("Q6_K cache not initialized".to_string()))?;

    let kernel = cache_ref
        .kernel
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("q6_k_matmul_kernel not loaded".to_string()))?;

    let block_dim = (256, 1, 1);
    let grid_dim = (m as u32, 1, 1);
    let shared_mem_bytes = 0;

    let mut activations_arg = activations;
    let mut weights_arg = weights_q6_k;
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
        .map_err(|e| HipError::KernelLaunchFailed(format!("Failed to launch q6_k_matmul kernel: {:?}", e)))?;

    Ok(())
}
