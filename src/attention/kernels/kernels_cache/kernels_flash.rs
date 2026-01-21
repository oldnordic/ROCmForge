//! Flash attention and matmul GPU kernels.

#![allow(non_snake_case)] // Kernel parameter names follow HIP conventions

use std::ffi::c_void;

use crate::backend::hip_backend::{HipError, HipKernel};

// kernels_flash is now a submodule of kernels_cache, so use super
use super::{get_attention_kernels, get_mqa_kernel_and_backend, get_or_init_cache};

// RDNA3 (wave32) tuning constants for AMD Radeon RX 7900 XT
const BLOCK_SIZE: u32 = 256; // 8 waves of 32 threads
const WARP_SIZE: u32 = 32; // RDNA3 wavefront size

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
    let (backend, (qkt_ptr, _softmax_ptr, _weighted_ptr)) = get_attention_kernels()
        .map_err(|e| format!("Failed to get attention kernels: {:?}", e))?;

    // Reconstruct HipKernel from pointer
    let kernel = HipKernel::from_ptr(qkt_ptr);

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
            &kernel,
            grid_dim,
            block_dim,
            args,
            shared_mem_bytes,
        )
        .map_err(|e| format!("Kernel launch failed: {:?}", e))
}

/// GPU kernel for weighted x V matrix multiplication (softmax weights x V)
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
    let (backend, (_qkt_ptr, _softmax_ptr, weighted_ptr)) = get_attention_kernels()
        .map_err(|e| format!("Failed to get attention kernels: {:?}", e))?;

    // Reconstruct HipKernel from pointer
    let kernel = HipKernel::from_ptr(weighted_ptr);

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
            &kernel,
            grid_dim,
            block_dim,
            args,
            shared_mem_bytes,
        )
        .map_err(|e| format!("Kernel launch failed: {:?}", e))
}

/// GPU kernel for FlashAttention (non-causal) - fused attention computation
///
/// Computes full attention in a single kernel: QK^T -> scale -> softmax -> softmax x V
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
    // Phase 33.1: Dynamic shared memory now supports seq_k up to device limits
    // Previous hardcoded limit of 32 has been removed
    // The kernel will use (seq_k + 32) * 4 bytes of shared memory

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

            // Dynamic shared memory: seq_k floats for s_scores + WARP_SIZE floats for s_partial
            // Phase 33.1 fix: Changed from hardcoded 64 floats to dynamic size based on seq_len
            let s_scores_size = seq_len * std::mem::size_of::<f32>() as u32;
            let s_partial_size = WARP_SIZE * std::mem::size_of::<f32>() as u32;
            let shared_mem_bytes = s_scores_size + s_partial_size;

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
/// QK^T -> scale -> causal mask -> softmax -> softmax x V
///
/// # Safety
/// This function is unsafe because it calls HIP kernels directly.
/// The caller must ensure that:
/// - Q, K, V, output point to valid GPU memory
/// - The dimensions are correct and consistent
/// - seq_k limited by device shared memory (Phase 33.1: now dynamic)
/// - No other threads are accessing the same memory concurrently
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
    // Phase 33.1: Dynamic shared memory now supports seq_k up to device limits
    // Previous hardcoded limit of 32 has been removed
    // The kernel will use (seq_k + 32) * 4 bytes of shared memory

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

            // Dynamic shared memory: seq_k floats for s_scores + WARP_SIZE floats for s_partial
            // Phase 33.1 fix: Changed from hardcoded 64 floats to dynamic size based on seq_len
            let s_scores_size = seq_len * std::mem::size_of::<f32>() as u32;
            let s_partial_size = WARP_SIZE * std::mem::size_of::<f32>() as u32;
            let shared_mem_bytes = s_scores_size + s_partial_size;

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
/// QK^T -> scale -> mask -> softmax -> softmax x V
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
    let (backend, kernel_ptr) = get_mqa_kernel_and_backend()
        .map_err(|e| format!("Failed to get MQA kernel: {:?}", e))?;

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

    // Reconstruct HipKernel from pointer for launch_kernel_with_module_shared
    let kernel = HipKernel::from_ptr(kernel_ptr);

    backend
        .launch_kernel_with_module_shared(&kernel, grid_dim, block_dim, args, 0)
        .map_err(|e| format!("Kernel launch failed: {:?}", e))
}
