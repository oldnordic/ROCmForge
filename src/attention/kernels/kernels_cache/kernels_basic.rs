//! Basic attention GPU kernels: scale, mask, softmax, RoPE.

#![allow(non_snake_case)] // Kernel parameter names follow HIP conventions

use std::ffi::c_void;

use crate::backend::hip_backend::{HipError, HipKernel};

// kernels_basic is now a submodule of kernels_cache, so use super
use super::{get_attention_kernels, get_or_init_cache};

// RDNA3 (wave32) tuning constants for AMD Radeon RX 7900 XT
const BLOCK_SIZE: u32 = 256; // 8 waves of 32 threads
#[allow(dead_code)]
const WARP_SIZE: u32 = 32; // RDNA3 wavefront size

/// GPU kernel for applying scaling factor to attention scores
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - scores points to valid GPU memory
/// - The dimensions are correct
/// - No other threads are accessing the same memory concurrently
pub unsafe fn scale_gpu_kernel(
    mut scores: *mut f32,
    scale: f32,
    batch_size: u32,
    seq_len: u32,
) -> i32 {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .expect("GLOBAL_CACHE lock poisoned in scale_gpu_kernel");
            let cache_ref = cache
                .as_ref()
                .expect("KernelCache not initialized in scale_gpu_kernel");

            let kernel = cache_ref
                .scale_kernel
                .as_ref()
                .expect("scale_kernel not initialized");
            let backend = &cache_ref.backend;

            // Calculate grid dimensions
            let total_elements = batch_size * seq_len * seq_len;
            let grid_dim = (total_elements.div_ceil(BLOCK_SIZE), 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Prepare kernel arguments
            let mut scale_arg = scale;
            let mut batch_size_arg = batch_size;
            let mut seq_len_arg = seq_len;

            let args: &[*mut c_void] = &[
                &mut scores as *mut _ as *mut c_void,
                &mut scale_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
            ];

            // Validate launch config against cached device limits before launching
            match backend.launch_kernel_with_module_shared_validated(kernel, grid_dim, block_dim, args, 0) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// GPU kernel for applying causal mask to attention scores
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - scores and mask point to valid GPU memory
/// - The dimensions are correct
/// - No other threads are accessing the same memory concurrently
pub unsafe fn mask_gpu_kernel(
    mut scores: *mut f32,
    mask: *const f32,
    batch_size: u32,
    seq_len: u32,
) -> i32 {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .expect("GLOBAL_CACHE lock poisoned in mask_gpu_kernel");
            let cache_ref = cache
                .as_ref()
                .expect("KernelCache not initialized in mask_gpu_kernel");

            let kernel = cache_ref
                .mask_kernel
                .as_ref()
                .expect("mask_kernel not initialized");
            let backend = &cache_ref.backend;

            // Calculate grid dimensions
            let total_elements = batch_size * seq_len * seq_len;
            let grid_dim = (total_elements.div_ceil(BLOCK_SIZE), 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Prepare kernel arguments
            let mut batch_size_arg = batch_size;
            let mut seq_len_arg = seq_len;

            let args: &[*mut c_void] = &[
                &mut scores as *mut _ as *mut c_void,
                &mut (mask as *mut f32) as *const _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
            ];

            // Validate launch config against cached device limits before launching
            match backend.launch_kernel_with_module_shared_validated(kernel, grid_dim, block_dim, args, 0) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// GPU kernel for row-wise softmax with numerical stability
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - scores points to valid GPU memory
/// - The dimensions are correct
/// - No other threads are accessing the same memory concurrently
pub unsafe fn softmax_gpu_kernel(mut scores: *mut f32, batch_size: u32, seq_len: u32) -> i32 {
    match get_attention_kernels() {
        Ok((backend, (_qkt_ptr, softmax_ptr, _weighted_ptr))) => {
            // Reconstruct HipKernel from pointer
            let kernel = HipKernel::from_ptr(softmax_ptr);

            // One block per row
            let total_rows = batch_size * seq_len;
            let grid_dim = (total_rows, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Shared memory: 2 * BLOCK_SIZE floats (for max and sum reduction)
            let shared_mem_bytes = 2 * BLOCK_SIZE * std::mem::size_of::<f32>() as u32;

            // Prepare kernel arguments
            let mut batch_size_arg = batch_size;
            let mut seq_len_arg = seq_len;

            let args: &[*mut c_void] = &[
                &mut scores as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
            ];

            // Validate launch config against cached device limits before launching
            match backend.launch_kernel_with_module_shared_validated(
                &kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// GPU kernel for Rotary Positional Embedding (RoPE)
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - input, cos, and sin point to valid GPU memory
/// - The dimensions are correct
/// - head_dim must be even
/// - No other threads are accessing the same memory concurrently
pub unsafe fn rope_gpu_kernel(
    mut input: *mut f32,
    cos: *const f32,
    sin: *const f32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> i32 {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .expect("GLOBAL_CACHE lock poisoned in rope_gpu_kernel");
            let cache_ref = cache
                .as_ref()
                .expect("KernelCache not initialized in rope_gpu_kernel");

            let kernel = cache_ref
                .rope_kernel
                .as_ref()
                .expect("rope_kernel not initialized");
            let backend = &cache_ref.backend;

            // Grid: (seq_len, num_heads, 1) - one block per token per head
            // Block: (BLOCK_SIZE, 1, 1) - handles head_dim/2 pairs per block
            let grid_dim = (seq_len, num_heads, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Prepare kernel arguments
            let mut seq_len_arg = seq_len;
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut input as *mut _ as *mut c_void,
                &mut (cos as *mut f32) as *const _ as *mut c_void,
                &mut (sin as *mut f32) as *const _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            // Validate launch config against cached device limits before launching
            match backend.launch_kernel_with_module_shared_validated(kernel, grid_dim, block_dim, args, 0) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// GPU kernel for applying position embeddings (RoPE) to Q and K tensors
///
/// This kernel applies rotary position embeddings to both query and key tensors
/// in a single kernel launch, avoiding multiple kernel invocations.
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - q, k, cos, and sin point to valid GPU memory
/// - The dimensions are correct
/// - head_dim must be even
/// - No other threads are accessing the same memory concurrently
pub unsafe fn position_embeddings_gpu_kernel(
    q: *mut f32,
    k: *mut f32,
    cos: *const f32,
    sin: *const f32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> i32 {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = match cache_ref.lock() {
                Ok(guard) => guard,
                Err(_) => return -1,
            };
            let cache_ref = match cache.as_ref() {
                Some(c) => c,
                None => return -1,
            };

            let kernel = match cache_ref.position_embeddings_kernel.as_ref() {
                Some(k) => k,
                None => return -1,
            };
            let backend = &cache_ref.backend;

            // Grid: (seq_len, num_heads, 1) - one block per token per head
            // Block: (BLOCK_SIZE, 1, 1) - handles head_dim/2 pairs per block
            let grid_dim = (seq_len, num_heads, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Prepare kernel arguments
            let mut q_arg = q;
            let mut k_arg = k;
            let mut cos_arg = cos as *mut f32;
            let mut sin_arg = sin as *mut f32;
            let mut seq_len_arg = seq_len;
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut q_arg as *mut _ as *mut c_void,
                &mut k_arg as *mut _ as *mut c_void,
                &mut cos_arg as *mut _ as *mut c_void,
                &mut sin_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            // Validate launch config against cached device limits before launching
            match backend.launch_kernel_with_module_shared_validated(kernel, grid_dim, block_dim, args, 0) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}
