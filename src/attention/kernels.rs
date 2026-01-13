//! GPU kernel functions for attention operations
//!
//! This module provides Rust wrappers for HIP kernels that implement
//! core attention operations on GPU.

#![allow(non_snake_case)] // Kernel parameter names follow HIP conventions

use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};

// RDNA3 (wave32) tuning constants for AMD Radeon RX 7900 XT
const BLOCK_SIZE: u32 = 256; // 8 waves of 32 threads
const WARP_SIZE: u32 = 32; // RDNA3 wavefront size

/// Cached kernel modules and functions
#[derive(Debug)]
struct KernelCache {
    backend: Arc<HipBackend>,
    scale_module: Option<HipModule>,
    scale_kernel: Option<HipKernel>,
    mask_module: Option<HipModule>,
    mask_kernel: Option<HipKernel>,
    softmax_module: Option<HipModule>,
    softmax_kernel: Option<HipKernel>,
    rope_module: Option<HipModule>,
    rope_kernel: Option<HipKernel>,
    position_embeddings_module: Option<HipModule>,
    position_embeddings_kernel: Option<HipKernel>,
    qkt_matmul_module: Option<HipModule>,
    qkt_matmul_kernel: Option<HipKernel>,
    weighted_matmul_module: Option<HipModule>,
    weighted_matmul_kernel: Option<HipKernel>,
    flash_attention_nocausal_module: Option<HipModule>,
    flash_attention_nocausal_kernel: Option<HipKernel>,
    causal_mask_module: Option<HipModule>,
    causal_mask_kernel: Option<HipKernel>,
    flash_attention_causal_module: Option<HipModule>,
    flash_attention_causal_kernel: Option<HipKernel>,
    flash_attention_module: Option<HipModule>,
    flash_attention_kernel: Option<HipKernel>,
    mqa_kv_replicate_module: Option<HipModule>,
    mqa_kv_replicate_kernel: Option<HipKernel>,
}

// Global kernel cache (lazy initialization)
static GLOBAL_CACHE: Mutex<Option<KernelCache>> = Mutex::new(None);

/// Get or initialize the global kernel cache
fn get_or_init_cache() -> Result<&'static Mutex<Option<KernelCache>>, HipError> {
    // First check if already initialized
    {
        let cache = GLOBAL_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&GLOBAL_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = GLOBAL_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&GLOBAL_CACHE);
    }

    // Create backend (HipBackend::new() returns Arc<HipBackend>)
    let backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend: {}", e))
    })?;

    // Load HSACO paths from build.rs environment variables
    let scale_path = std::env::var("SCALE_HSACO")
        .ok()
        .ok_or_else(|| HipError::KernelLoadFailed("SCALE_HSACO env var not set".to_string()))?;

    if !Path::new(&scale_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            scale_path
        )));
    }

    let scale_module = backend.load_module(&scale_path)?;
    let scale_kernel = backend.get_kernel_function(&scale_module, "scale_kernel")?;

    let mask_path = std::env::var("MASK_HSACO")
        .ok()
        .ok_or_else(|| HipError::KernelLoadFailed("MASK_HSACO env var not set".to_string()))?;

    if !Path::new(&mask_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            mask_path
        )));
    }

    let mask_module = backend.load_module(&mask_path)?;
    let mask_kernel = backend.get_kernel_function(&mask_module, "mask_kernel")?;

    let softmax_path = std::env::var("SOFTMAX_HSACO")
        .ok()
        .ok_or_else(|| HipError::KernelLoadFailed("SOFTMAX_HSACO env var not set".to_string()))?;

    if !Path::new(&softmax_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            softmax_path
        )));
    }

    let softmax_module = backend.load_module(&softmax_path)?;
    let softmax_kernel = backend.get_kernel_function(&softmax_module, "softmax_kernel")?;

    // Load RoPE kernel
    let rope_path = std::env::var("ROPE_HSACO")
        .ok()
        .ok_or_else(|| HipError::KernelLoadFailed("ROPE_HSACO env var not set".to_string()))?;

    if !Path::new(&rope_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            rope_path
        )));
    }

    let rope_module = backend.load_module(&rope_path)?;
    let rope_kernel = backend.get_kernel_function(&rope_module, "rope_kernel")?;

    // Load position embeddings kernel
    let position_embeddings_path =
        std::env::var("POSITION_EMBEDDINGS_HSACO")
            .ok()
            .ok_or_else(|| {
                HipError::KernelLoadFailed("POSITION_EMBEDDINGS_HSACO env var not set".to_string())
            })?;

    if !Path::new(&position_embeddings_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            position_embeddings_path
        )));
    }

    let position_embeddings_module = backend.load_module(&position_embeddings_path)?;
    let position_embeddings_kernel =
        backend.get_kernel_function(&position_embeddings_module, "position_embeddings_kernel")?;

    // Load QK^T matmul kernel
    let qkt_matmul_path = std::env::var("QKT_MATMUL_HSACO").ok().ok_or_else(|| {
        HipError::KernelLoadFailed("QKT_MATMUL_HSACO env var not set".to_string())
    })?;

    if !Path::new(&qkt_matmul_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            qkt_matmul_path
        )));
    }

    let qkt_matmul_module = backend.load_module(&qkt_matmul_path)?;
    let qkt_matmul_kernel = backend.get_kernel_function(&qkt_matmul_module, "qkt_matmul_kernel")?;

    // Load weighted matmul kernel
    let weighted_matmul_path = std::env::var("WEIGHTED_MATMUL_HSACO").ok().ok_or_else(|| {
        HipError::KernelLoadFailed("WEIGHTED_MATMUL_HSACO env var not set".to_string())
    })?;

    if !Path::new(&weighted_matmul_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            weighted_matmul_path
        )));
    }

    let weighted_matmul_module = backend.load_module(&weighted_matmul_path)?;
    let weighted_matmul_kernel =
        backend.get_kernel_function(&weighted_matmul_module, "weighted_matmul_kernel")?;

    // Load FlashAttention non-causal kernel
    let flash_attention_nocausal_path = std::env::var("FLASH_ATTENTION_NCAUSAL_HSACO")
        .ok()
        .ok_or_else(|| {
            HipError::KernelLoadFailed("FLASH_ATTENTION_NCAUSAL_HSACO env var not set".to_string())
        })?;

    if !Path::new(&flash_attention_nocausal_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            flash_attention_nocausal_path
        )));
    }

    let flash_attention_nocausal_module = backend.load_module(&flash_attention_nocausal_path)?;
    let flash_attention_nocausal_kernel = backend.get_kernel_function(
        &flash_attention_nocausal_module,
        "flash_attention_nocausal_kernel",
    )?;

    // Load causal mask kernel
    let causal_mask_path = std::env::var("CAUSAL_MASK_HSACO").ok().ok_or_else(|| {
        HipError::KernelLoadFailed("CAUSAL_MASK_HSACO env var not set".to_string())
    })?;

    if !Path::new(&causal_mask_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            causal_mask_path
        )));
    }

    let causal_mask_module = backend.load_module(&causal_mask_path)?;
    let causal_mask_kernel =
        backend.get_kernel_function(&causal_mask_module, "causal_mask_kernel")?;

    // Load FlashAttention causal kernel
    let flash_attention_causal_path = std::env::var("FLASH_ATTENTION_CAUSAL_HSACO")
        .ok()
        .ok_or_else(|| {
            HipError::KernelLoadFailed("FLASH_ATTENTION_CAUSAL_HSACO env var not set".to_string())
        })?;

    if !Path::new(&flash_attention_causal_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            flash_attention_causal_path
        )));
    }

    let flash_attention_causal_module = backend.load_module(&flash_attention_causal_path)?;
    let flash_attention_causal_kernel = backend.get_kernel_function(
        &flash_attention_causal_module,
        "flash_attention_causal_kernel",
    )?;

    // Load FlashAttention kernel
    let flash_attention_path = std::env::var("FLASH_ATTENTION_HSACO").ok().ok_or_else(|| {
        HipError::KernelLoadFailed("FLASH_ATTENTION_HSACO env var not set".to_string())
    })?;

    if !Path::new(&flash_attention_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            flash_attention_path
        )));
    }

    let flash_attention_module = backend.load_module(&flash_attention_path)?;
    let flash_attention_kernel =
        backend.get_kernel_function(&flash_attention_module, "flash_attention_kernel")?;

    // Load MQA KV replication kernel
    let mqa_kv_replicate_path = std::env::var("MQA_KV_REPLICATE_HSACO")
        .ok()
        .ok_or_else(|| {
            HipError::KernelLoadFailed("MQA_KV_REPLICATE_HSACO env var not set".to_string())
        })?;

    if !Path::new(&mqa_kv_replicate_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            mqa_kv_replicate_path
        )));
    }

    let mqa_kv_replicate_module = backend.load_module(&mqa_kv_replicate_path)?;
    let mqa_kv_replicate_kernel =
        backend.get_kernel_function(&mqa_kv_replicate_module, "mqa_kv_replicate_fused_kernel")?;

    *cache = Some(KernelCache {
        backend,
        scale_module: Some(scale_module),
        scale_kernel: Some(scale_kernel),
        mask_module: Some(mask_module),
        mask_kernel: Some(mask_kernel),
        softmax_module: Some(softmax_module),
        softmax_kernel: Some(softmax_kernel),
        rope_module: Some(rope_module),
        rope_kernel: Some(rope_kernel),
        position_embeddings_module: Some(position_embeddings_module),
        position_embeddings_kernel: Some(position_embeddings_kernel),
        qkt_matmul_module: Some(qkt_matmul_module),
        qkt_matmul_kernel: Some(qkt_matmul_kernel),
        weighted_matmul_module: Some(weighted_matmul_module),
        weighted_matmul_kernel: Some(weighted_matmul_kernel),
        flash_attention_nocausal_module: Some(flash_attention_nocausal_module),
        flash_attention_nocausal_kernel: Some(flash_attention_nocausal_kernel),
        causal_mask_module: Some(causal_mask_module),
        causal_mask_kernel: Some(causal_mask_kernel),
        flash_attention_causal_module: Some(flash_attention_causal_module),
        flash_attention_causal_kernel: Some(flash_attention_causal_kernel),
        flash_attention_module: Some(flash_attention_module),
        flash_attention_kernel: Some(flash_attention_kernel),
        mqa_kv_replicate_module: Some(mqa_kv_replicate_module),
        mqa_kv_replicate_kernel: Some(mqa_kv_replicate_kernel),
    });

    Ok(&GLOBAL_CACHE)
}

/// GPU kernel for applying scaling factor to attention scores
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - scores points to valid GPU memory
/// - The dimensions are correct
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
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

            match backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0) {
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
#[cfg(feature = "rocm")]
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

            match backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0) {
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
#[cfg(feature = "rocm")]
pub unsafe fn softmax_gpu_kernel(mut scores: *mut f32, batch_size: u32, seq_len: u32) -> i32 {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .expect("GLOBAL_CACHE lock poisoned in softmax_gpu_kernel");
            let cache_ref = cache
                .as_ref()
                .expect("KernelCache not initialized in softmax_gpu_kernel");

            let kernel = cache_ref
                .softmax_kernel
                .as_ref()
                .expect("softmax_kernel not initialized");
            let backend = &cache_ref.backend;

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

            match backend.launch_kernel_with_module_shared(
                kernel,
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
#[cfg(feature = "rocm")]
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

            match backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// GPU kernel for QK^T matrix multiplication
///
/// Computes Query-Key transpose matrix multiply per attention head:
///   Q [batch, heads, seq_q, dim]
///   K [batch, heads, seq_k, dim]
///   Output [batch, heads, seq_q, seq_k]
///
/// For each (batch, head), computes:
///   output[seq_q, seq_k] = Q[seq_q, dim] @ K[seq_k, dim]^T
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - Q, K, and output point to valid GPU memory
/// - The dimensions are correct and consistent
/// - head_dim <= 128 (register limit in kernel)
/// - No other threads are accessing the same memory concurrently
/// GPU kernel for QK^T matrix multiplication
///
/// Computes Query-Key transpose matrix multiply per attention head:
///   Q [batch, heads, seq_q, dim]
///   K [batch, heads, seq_k, dim]
///   Output [batch, heads, seq_q, seq_k]
///
/// For each (batch, head), computes:
///   output[seq_q, seq_k] = Q[seq_q, dim] @ K[seq_k, dim]^T * scale
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - Q, K, and output point to valid GPU memory
/// - The dimensions are correct and consistent
/// - head_dim <= 128 (register limit in kernel)
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn qkt_matmul_gpu_kernel(
    q: *const f32,
    k: *const f32,
    output: *mut f32,
    batch_size: u32,
    seq_q: u32,
    seq_k: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    qkt_matmul_gpu_kernel_scaled(
        q, k, output, batch_size, seq_q, seq_k, num_heads, head_dim, 1.0,
    )
}

/// GPU kernel for QK^T matrix multiplication with explicit scale
///
/// Same as qkt_matmul_gpu_kernel but allows specifying the scale factor.
/// Use scale=1.0/sqrt(head_dim) for scaled dot-product attention.
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
#[cfg(feature = "rocm")]
pub unsafe fn qkt_matmul_gpu_kernel_scaled(
    q: *const f32,
    k: *const f32,
    output: *mut f32,
    batch_size: u32,
    seq_q: u32,
    seq_k: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("GLOBAL_CACHE lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "KernelCache not initialized".to_string())?;

            let kernel = cache_ref
                .qkt_matmul_kernel
                .as_ref()
                .ok_or_else(|| "qkt_matmul_kernel not loaded".to_string())?;
            let backend = &cache_ref.backend;

            // Grid: (seq_q, num_heads, batch_size)
            // Each block processes one query position for one head in one batch
            let grid_dim = (seq_q, num_heads, batch_size);
            let block_dim = (WARP_SIZE, 1, 1);

            // Shared memory: WARP_SIZE floats for wave32 reduction
            let shared_mem_bytes = WARP_SIZE * std::mem::size_of::<f32>() as u32;

            // Prepare kernel arguments
            let mut q_arg = q as *mut f32;
            let mut k_arg = k as *mut f32;
            let mut output_arg = output;
            let mut scale_arg = scale;
            let mut batch_size_arg = batch_size;
            let mut seq_q_arg = seq_q;
            let mut seq_k_arg = seq_k;
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut q_arg as *mut _ as *mut c_void,
                &mut k_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut scale_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_q_arg as *mut _ as *mut c_void,
                &mut seq_k_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Kernel launch failed: {:?}", e))
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// GPU kernel for weighted × V matrix multiplication (softmax weights × V)
///
/// Computes output[query_pos, dim] = sum over seq_k of (weights[query_pos, key_pos] * V[key_pos, dim])
///
/// Layout: [batch, heads, seq, dim] explicit for all tensors
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - weights, V, and output point to valid GPU memory
/// - The dimensions are correct and consistent
/// - head_dim <= 128 (register limit in kernel)
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn weighted_matmul_gpu_kernel(
    weights: *const f32,
    v: *const f32,
    output: *mut f32,
    batch_size: u32,
    seq_q: u32,
    seq_k: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("GLOBAL_CACHE lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "KernelCache not initialized".to_string())?;

            let kernel = cache_ref
                .weighted_matmul_kernel
                .as_ref()
                .ok_or_else(|| "weighted_matmul_kernel not loaded".to_string())?;
            let backend = &cache_ref.backend;

            // Grid: (seq_q, num_heads, batch_size)
            // Each block processes one query position for one head in one batch
            let grid_dim = (seq_q, num_heads, batch_size);
            let block_dim = (WARP_SIZE, 1, 1);

            // Shared memory: WARP_SIZE floats for wave32 reduction
            let shared_mem_bytes = WARP_SIZE * std::mem::size_of::<f32>() as u32;

            // Prepare kernel arguments
            let mut weights_arg = weights as *mut f32;
            let mut v_arg = v as *mut f32;
            let mut output_arg = output;
            let mut batch_size_arg = batch_size;
            let mut seq_q_arg = seq_q;
            let mut seq_k_arg = seq_k;
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut weights_arg as *mut _ as *mut c_void,
                &mut v_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_q_arg as *mut _ as *mut c_void,
                &mut seq_k_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Kernel launch failed: {:?}", e))
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// GPU kernel for FlashAttention (non-causal) - fused attention computation
///
/// Computes full attention in a single kernel: QK^T → scale → softmax → softmax × V
/// NO causal masking - this is for non-causal attention only.
///
/// Layout: [batch, heads, seq, dim] explicit for all tensors
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - Q, K, V, and output point to valid GPU memory
/// - The dimensions are correct and consistent
/// - head_dim <= 128 (register limit in kernel)
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn flash_attention_nocausal_gpu_kernel(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    output: *mut f32,
    scale: f32,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("GLOBAL_CACHE lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "KernelCache not initialized".to_string())?;

            let kernel = cache_ref
                .flash_attention_nocausal_kernel
                .as_ref()
                .ok_or_else(|| "flash_attention_nocausal_kernel not loaded".to_string())?;
            let backend = &cache_ref.backend;

            // Grid: (seq_len, num_heads, batch_size)
            // Each block processes one query position for one head in one batch
            let grid_dim = (seq_len, num_heads, batch_size);
            let block_dim = (WARP_SIZE, 1, 1);

            // Shared memory: 2 × WARP_SIZE floats for reduction + scores buffer
            let shared_mem_bytes = 2 * WARP_SIZE * std::mem::size_of::<f32>() as u32;

            // Prepare kernel arguments
            let mut q_arg = q as *mut f32;
            let mut k_arg = k as *mut f32;
            let mut v_arg = v as *mut f32;
            let mut output_arg = output;
            let mut scale_arg = scale;
            let mut batch_size_arg = batch_size;
            let mut seq_q_arg = seq_len;
            let mut seq_k_arg = seq_len; // Square for self-attention
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut q_arg as *mut _ as *mut c_void,
                &mut k_arg as *mut _ as *mut c_void,
                &mut v_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut scale_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_q_arg as *mut _ as *mut c_void,
                &mut seq_k_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Kernel launch failed: {:?}", e))
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// GPU kernel for causal mask generation
///
/// Creates a causal mask where mask[query_pos, key_pos] = -inf if key_pos > query_pos
/// Layout: [batch, heads, seq_q, seq_k] explicit
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - mask points to valid GPU memory with sufficient size
/// - The dimensions are correct
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn causal_mask_gpu_kernel(
    mask: *mut f32,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("GLOBAL_CACHE lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "KernelCache not initialized".to_string())?;

            let kernel = cache_ref
                .causal_mask_kernel
                .as_ref()
                .ok_or_else(|| "causal_mask_kernel not loaded".to_string())?;
            let backend = &cache_ref.backend;

            // Grid: (seq_len, num_heads, batch_size)
            // Each block processes one query position for one head in one batch
            let grid_dim = (seq_len, num_heads, batch_size);
            let block_dim = (WARP_SIZE, 1, 1);

            // Shared memory: not needed for causal mask kernel
            let shared_mem_bytes = 0u32;

            // Prepare kernel arguments
            let mut mask_arg = mask;
            let mut batch_size_arg = batch_size;
            let mut seq_len_arg = seq_len;
            let mut num_heads_arg = num_heads;

            let args: &[*mut c_void] = &[
                &mut mask_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Kernel launch failed: {:?}", e))
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// GPU kernel for fused causal FlashAttention
///
/// Computes the entire causal attention operation in a single kernel:
/// QK^T -> scale -> causal mask -> softmax -> softmax × V
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - Q, K, V, output point to valid GPU memory
/// - The dimensions are correct and consistent
/// - seq_k <= 32 (shared memory limitation)
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn flash_attention_causal_gpu_kernel(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    output: *mut f32,
    scale: f32,
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("GLOBAL_CACHE lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "KernelCache not initialized".to_string())?;

            let kernel = cache_ref
                .flash_attention_causal_kernel
                .as_ref()
                .ok_or_else(|| "flash_attention_causal_kernel not loaded".to_string())?;
            let backend = &cache_ref.backend;

            // Grid: (seq_len, num_heads, batch_size)
            // Each block processes one query position for one head in one batch
            let grid_dim = (seq_len, num_heads, batch_size);
            let block_dim = (WARP_SIZE, 1, 1);

            // Shared memory: 2 × WARP_SIZE floats for scores + reduction buffer
            let shared_mem_bytes = 2 * WARP_SIZE * std::mem::size_of::<f32>() as u32;

            // Prepare kernel arguments
            let mut q_arg = q as *mut f32;
            let mut k_arg = k as *mut f32;
            let mut v_arg = v as *mut f32;
            let mut output_arg = output;
            let mut scale_arg = scale;
            let mut batch_size_arg = batch_size;
            let mut seq_q_arg = seq_len;
            let mut seq_k_arg = seq_len; // Square for self-attention
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut q_arg as *mut _ as *mut c_void,
                &mut k_arg as *mut _ as *mut c_void,
                &mut v_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut scale_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_q_arg as *mut _ as *mut c_void,
                &mut seq_k_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Kernel launch failed: {:?}", e))
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// GPU kernel for FlashAttention - fused attention computation
///
/// FlashAttention computes the entire attention operation in a single kernel:
/// QK^T -> scale -> mask -> softmax -> softmax × V
///
/// This eliminates CPU round-trips and reduces memory bandwidth usage.
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - Q, K, V, output, and mask point to valid GPU memory
/// - The dimensions are correct and consistent
/// - head_dim <= 128 (register limit in kernel)
/// - mask can be nullptr for no masking
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn flash_attention_gpu_kernel(
    Q: *const f32,
    K: *const f32,
    V: *const f32,
    output: *mut f32,
    mask: *const f32,
    scale: f32,
    batch_size: u32,
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

            let kernel = match cache_ref.flash_attention_kernel.as_ref() {
                Some(k) => k,
                None => return -1,
            };
            let backend = &cache_ref.backend;

            // Grid: (seq_len, num_heads, batch_size)
            // Each block processes one query position for one head in one batch
            let grid_dim = (seq_len, num_heads, batch_size);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Prepare kernel arguments
            let mut Q_arg = Q as *mut f32;
            let mut K_arg = K as *mut f32;
            let mut V_arg = V as *mut f32;
            let mut output_arg = output;
            let mut mask_arg = mask;
            let mut scale_arg = scale;
            let mut batch_size_arg = batch_size;
            let mut seq_len_arg = seq_len;
            let mut num_heads_arg = num_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut Q_arg as *mut _ as *mut c_void,
                &mut K_arg as *mut _ as *mut c_void,
                &mut V_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut mask_arg as *mut _ as *mut c_void,
                &mut scale_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut num_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            match backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0) {
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
#[cfg(feature = "rocm")]
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

            match backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0) {
                Ok(()) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// GPU kernel for KV head replication in MQA/GQA
///
/// Replicates K and V tensors from num_kv_heads to num_q_heads.
/// Uses fused kernel for single-launch efficiency.
///
/// # Arguments
/// * `k` - Input K tensor [batch_size, seq_len, num_kv_heads, head_dim]
/// * `v` - Input V tensor [batch_size, seq_len, num_kv_heads, head_dim]
/// * `k_expanded` - Output K tensor [batch_size, seq_len, num_q_heads, head_dim]
/// * `v_expanded` - Output V tensor [batch_size, seq_len, num_q_heads, head_dim]
/// * `batch_size` - Number of batches
/// * `seq_len` - Sequence length
/// * `num_kv_heads` - Number of KV heads
/// * `num_q_heads` - Number of query heads
/// * `head_dim` - Dimension per head
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - All pointers point to valid GPU memory
/// - Output tensors have sufficient capacity
/// - Dimensions are correct and consistent
/// - No other threads are accessing the same memory concurrently
#[cfg(feature = "rocm")]
pub unsafe fn mqa_kv_replicate_gpu_kernel(
    k: *const f32,
    v: *const f32,
    k_expanded: *mut f32,
    v_expanded: *mut f32,
    batch_size: u32,
    seq_len: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("GLOBAL_CACHE lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "KernelCache not initialized".to_string())?;

            let kernel = cache_ref
                .mqa_kv_replicate_kernel
                .as_ref()
                .ok_or_else(|| "mqa_kv_replicate_kernel not loaded".to_string())?;
            let backend = &cache_ref.backend;

            // Calculate grid dimensions
            let total_elements = batch_size * seq_len * num_q_heads * head_dim;
            let grid_dim = (total_elements.div_ceil(BLOCK_SIZE), 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);

            // Prepare kernel arguments
            let mut k_arg = k as *mut f32;
            let mut v_arg = v as *mut f32;
            let mut k_expanded_arg = k_expanded;
            let mut v_expanded_arg = v_expanded;
            let mut batch_size_arg = batch_size;
            let mut seq_len_arg = seq_len;
            let mut num_kv_heads_arg = num_kv_heads;
            let mut num_q_heads_arg = num_q_heads;
            let mut head_dim_arg = head_dim;

            let args: &[*mut c_void] = &[
                &mut k_arg as *mut _ as *mut c_void,
                &mut v_arg as *mut _ as *mut c_void,
                &mut k_expanded_arg as *mut _ as *mut c_void,
                &mut v_expanded_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut num_kv_heads_arg as *mut _ as *mut c_void,
                &mut num_q_heads_arg as *mut _ as *mut c_void,
                &mut head_dim_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, 0)
                .map_err(|e| format!("Kernel launch failed: {:?}", e))
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}
