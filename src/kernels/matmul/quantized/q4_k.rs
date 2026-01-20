//! Q4_K quantized matmul operations
//!
//! # Format Specification
//! - Super-block size: 256 elements (8 sub-blocks of 32 elements each)
//! - Per super-block (256 bytes): 16 bytes scales + 16 bytes mins + 224 bytes quants
//! - Dequantization: value = min + (quant * scale)

use std::env;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};
use crate::backend::HipBuffer;

use super::common::{f16_to_f32_cpu, Q4_K_ELEMENTS_PER_BLOCK, Q4_K_SUPER_BLOCK_SIZE, QuantizedResult};

/// Cached kernel modules and functions for Q4_K matmul
///
/// This cache infrastructure was part of Phase 24 kernel migration.
/// It is reserved for future HSACO lazy-loading optimization but currently
/// unused because kernels are loaded directly via HipModule::from_file().
#[derive(Debug)]
#[allow(non_camel_case_types)] // Matches GGUF quantization format naming
#[allow(dead_code)] // Reserved for future HSACO lazy-loading optimization
struct Q4_KKernelCache {
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    module: Option<HipModule>,
    kernel: Option<HipKernel>,
}

/// Global kernel cache for lazy-loaded HSACO modules
#[allow(dead_code)] // Reserved for future HSACO lazy-loading optimization
static Q4_K_CACHE: Mutex<Option<Q4_KKernelCache>> = Mutex::new(None);

/// Get or initialize the global Q4_K kernel cache
#[allow(dead_code)] // Reserved for future HSACO lazy-loading optimization
fn get_or_init_q4_k_cache() -> Result<&'static Mutex<Option<Q4_KKernelCache>>, HipError> {
    // First check if already initialized
    {
        let cache = Q4_K_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("Q4_K_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&Q4_K_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = Q4_K_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q4_K_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&Q4_K_CACHE);
    }

    // Load Q4_K matmul kernel module and function
    let load_backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e))
    })?;

    let kernel_path = env::var("Q4_K_MATMUL_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("Q4_K_MATMUL_HSACO env var not set".to_string()))?;

    if !Path::new(&kernel_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "Q4_K matmul HSACO not found: {}",
            kernel_path
        )));
    }

    let module = load_backend.load_module(&kernel_path)?;
    let kernel = load_backend.get_kernel_function(&module, "q4_k_matmul_kernel")?;

    *cache = Some(Q4_KKernelCache {
        module: Some(module),
        kernel: Some(kernel),
    });

    Ok(&Q4_K_CACHE)
}

/// Dequantize Q4_K weights to f32 (CPU reference implementation)
///
/// # Format
/// - Super-block size: 256 elements (8 sub-blocks of 32 elements each)
/// - Per super-block (256 bytes): 16 bytes scales + 16 bytes mins + 224 bytes quants
/// - Dequantization: value = min + (quant * scale)
pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_super_blocks = (n_elements + Q4_K_ELEMENTS_PER_BLOCK - 1) / Q4_K_ELEMENTS_PER_BLOCK;
    let mut result = vec![0.0f32; n_elements];

    for super_block_idx in 0..n_super_blocks {
        let super_block_offset = super_block_idx * Q4_K_SUPER_BLOCK_SIZE;
        if super_block_offset + 32 > data.len() {
            break;
        }

        // Process each sub-block (8 sub-blocks per super-block)
        for sub_block_idx in 0..8 {
            let base_idx = super_block_idx * Q4_K_ELEMENTS_PER_BLOCK + sub_block_idx * 32;

            // Read scale (half-precision)
            let scale_offset = super_block_offset + sub_block_idx * 2;
            if scale_offset + 2 > data.len() {
                break;
            }
            let scale_bits = u16::from_le_bytes([
                data[scale_offset],
                data[scale_offset + 1],
            ]);
            let scale = f16_to_f32_cpu(scale_bits);

            // Read min (int8)
            let min_offset = super_block_offset + 16 + sub_block_idx;
            if min_offset >= data.len() {
                break;
            }
            let min = data[min_offset] as i8 as f32;

            // Unpack 4-bit values for this sub-block
            let quants_base = super_block_offset + 32 + sub_block_idx * 20;
            for i in 0..32 {
                let elem_idx = base_idx + i;
                if elem_idx >= n_elements {
                    break;
                }

                let bit_pos = i * 4;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quants_idx = quants_base + byte_idx;
                if quants_idx + 1 >= data.len() {
                    break;
                }

                let combined = ((data[quants_idx + 1] as u16) << 8) | (data[quants_idx] as u16);
                let quant = ((combined >> bit_offset) & 0x0F) as f32;

                result[elem_idx] = min + (quant * scale);
            }
        }
    }

    result
}

/// MatMul with Q4_K quantized weights using fused dequant+matmul kernel
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q4_K quantized weight data [n_rows x n_cols]
/// - `input`: Input tensor (f32), typically [1 x n_cols]
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
///
/// # Q4_K Format
/// - Super-block size: 256 elements (8 sub-blocks of 32 elements each)
/// - Per super-block (256 bytes): 16 bytes scales + 16 bytes mins + 224 bytes quants
/// - Dequantization: value = min + (quant * scale)
///
/// # Memory Layout
/// - Input: [M x K] row-major (FP32), typically M=1, K=n_cols
/// - Weights: [n_rows x n_cols] row-major (Q4_K format)
/// - Output: [M x n_rows] row-major (FP32)
///
/// # GPU Implementation
/// Uses fused kernel that dequantizes weights on-the-fly during matmul,
/// eliminating the intermediate FP32 weight buffer.
#[cfg(feature = "rocm")]
pub fn matmul_q4_k(
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
        matmul_q4_k_gpu(backend, input_ptr, weight_ptr, output_ptr, m, n, k)
            .map_err(|e| format!("Fused Q4_K matmul failed: {}", e))?;
    }

    backend
        .synchronize()
        .map_err(|e| format!("Failed to synchronize: {}", e))?;

    Ok(())
}

/// MatMul with Q4_K quantized weights (non-rocm fallback)
#[cfg(not(feature = "rocm"))]
pub fn matmul_q4_k(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &HipBuffer,
) -> QuantizedResult<()> {
    // Use CPU fallback - dequantize then standard matmul
    let n_elements = n_rows * n_cols;
    let dequant_weights = dequantize_q4_k(quantized_weights, n_elements);

    let weight_bytes = n_elements * 4;
    let weight_buffer = backend
        .allocate_buffer(weight_bytes)
        .map_err(|e| format!("Failed to allocate weight buffer: {}", e))?;

    weight_buffer
        .copy_from_host(&dequant_weights)
        .map_err(|e| format!("Failed to upload weights: {}", e))?;

    let m = 1i32;
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

/// Launch Q4_K fused dequant+matmul kernel
///
/// # Safety
/// Caller must ensure all pointers are valid and synchronized.
#[cfg(feature = "rocm")]
unsafe fn matmul_q4_k_gpu(
    backend: &HipBackend,
    activations: *const f32,
    weights_q4_k: *const u8,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), HipError> {
    let cache_ref = get_or_init_q4_k_cache()?;
    let cache = cache_ref
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("Q4_K cache lock poisoned: {}", e)))?;
    let cache_ref = cache
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("Q4_K cache not initialized".to_string()))?;

    let kernel = cache_ref
        .kernel
        .as_ref()
        .ok_or_else(|| HipError::KernelLoadFailed("q4_k_matmul_kernel not loaded".to_string()))?;

    let block_dim = (256, 1, 1);
    let grid_dim = (m as u32, 1, 1);
    let shared_mem_bytes = 0;

    let mut activations_arg = activations;
    let mut weights_arg = weights_q4_k;
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
        .map_err(|e| HipError::KernelLaunchFailed(format!("Failed to launch q4_k_matmul kernel: {:?}", e)))?;

    Ok(())
}
