//! GPU kernel cache and launch wrappers for sampling
//!
//! This module contains the kernel cache infrastructure and low-level
//! kernel launch functions for GPU-accelerated sampling.

#![allow(dead_code)]

#[cfg(feature = "rocm")]
use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};
use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

#[cfg(feature = "rocm")]
const BLOCK_SIZE: u32 = 256;

/// Cached kernel modules and functions for sampling operations
///
/// NOTE: We do NOT store HipBackend here because that would create a separate
/// HIP stream from the caller's. Kernels must be launched on the caller's stream
/// and synchronized on the same stream to avoid hangs.
///
/// Kernels loaded from build.rs compiled HSACO files:
/// - softmax_kernel: from SAMPLING_UTILS_HSACO
/// - temperature_scale_kernel: from TEMPERATURE_SCALE_HSACO
/// - topk_sampling_kernel: from TOPK_SAMPLING_HSACO
/// - topp_prefix_sum_kernel: from TOPP_PREFIX_SUM_HSACO
/// - topp_threshold_kernel: from TOPP_THRESHOLD_HSACO
/// - topp_sample_kernel: from TOPP_SAMPLE_HSACO
/// - topk_topp_sampling_kernel: from FUSED_SAMPLING_HSACO
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub struct SamplingKernelCache {
    // softmax_kernel from SAMPLING_UTILS_HSACO
    pub softmax_module: Option<HipModule>,
    pub softmax_kernel: Option<HipKernel>,
    // temperature_scale_kernel from TEMPERATURE_SCALE_HSACO
    pub temperature_scale_module: Option<HipModule>,
    pub temperature_scale_kernel: Option<HipKernel>,
    // topk_sampling_kernel from TOPK_SAMPLING_HSACO
    pub topk_module: Option<HipModule>,
    pub topk_kernel: Option<HipKernel>,
    // topp_prefix_sum_kernel from TOPP_PREFIX_SUM_HSACO
    pub topp_prefix_sum_module: Option<HipModule>,
    pub topp_prefix_sum_kernel: Option<HipKernel>,
    // topp_threshold_kernel from TOPP_THRESHOLD_HSACO
    pub topp_threshold_module: Option<HipModule>,
    pub topp_threshold_kernel: Option<HipKernel>,
    // topp_sample_kernel from TOPP_SAMPLE_HSACO
    pub topp_sample_module: Option<HipModule>,
    pub topp_sample_kernel: Option<HipKernel>,
    // topk_topp_sampling_kernel from FUSED_SAMPLING_HSACO
    pub fused_module: Option<HipModule>,
    pub fused_kernel: Option<HipKernel>,
}

// Global kernel cache (lazy initialization)
#[cfg(feature = "rocm")]
pub static GLOBAL_SAMPLING_CACHE: Mutex<Option<SamplingKernelCache>> = Mutex::new(None);

/// Get or initialize the global sampling kernel cache
///
/// Returns cached kernel modules and functions. The caller must provide
/// their own HipBackend for launching kernels to ensure stream consistency.
#[cfg(feature = "rocm")]
pub fn get_or_init_sampling_cache() -> Result<&'static Mutex<Option<SamplingKernelCache>>, HipError> {
    use super::super::SamplerError;

    // First check if already initialized
    {
        let cache = GLOBAL_SAMPLING_CACHE.lock()
            .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_SAMPLING_CACHE lock poisoned: {}", e)))?;
        if cache.is_some() {
            return Ok(&GLOBAL_SAMPLING_CACHE);
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = GLOBAL_SAMPLING_CACHE.lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_SAMPLING_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if cache.is_some() {
        return Ok(&GLOBAL_SAMPLING_CACHE);
    }

    // Load kernel modules using a temporary backend
    let load_backend = HipBackend::new()
        .map_err(|e| HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e)))?;

    // ========================================================================
    // Load sampling kernels from compiled HSACO files
    // ========================================================================

    // Load softmax_kernel from SAMPLING_UTILS_HSACO
    let softmax_path = std::env::var("SAMPLING_UTILS_HSACO")
        .unwrap_or_else(|_| "kernels/sampling_utils.hsaco".to_string());

    let (softmax_module, softmax_kernel) = if Path::new(&softmax_path).exists() {
        let module = load_backend.load_module(&softmax_path)?;
        let kernel = load_backend.get_kernel_function(&module, "softmax_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Softmax kernel not found at {} (SAMPLING_UTILS_HSACO), using CPU fallback", softmax_path);
        (None, None)
    };

    // Load temperature_scale_kernel from TEMPERATURE_SCALE_HSACO
    let temperature_scale_path = std::env::var("TEMPERATURE_SCALE_HSACO")
        .unwrap_or_else(|_| "kernels/temperature_scale.hsaco".to_string());

    let (temperature_scale_module, temperature_scale_kernel) = if Path::new(&temperature_scale_path).exists() {
        let module = load_backend.load_module(&temperature_scale_path)?;
        let kernel = load_backend.get_kernel_function(&module, "temperature_scale_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Temperature scale kernel not found at {} (TEMPERATURE_SCALE_HSACO), using CPU fallback", temperature_scale_path);
        (None, None)
    };

    // Load topk_sampling_kernel from TOPK_SAMPLING_HSACO
    let topk_path = std::env::var("TOPK_SAMPLING_HSACO")
        .unwrap_or_else(|_| "kernels/topk_sampling.hsaco".to_string());

    let (topk_module, topk_kernel) = if Path::new(&topk_path).exists() {
        let module = load_backend.load_module(&topk_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topk_sampling_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Top-k sampling kernel not found at {} (TOPK_SAMPLING_HSACO), using CPU fallback", topk_path);
        (None, None)
    };

    // Load topp_prefix_sum_kernel from TOPP_PREFIX_SUM_HSACO
    let topp_prefix_sum_path = std::env::var("TOPP_PREFIX_SUM_HSACO")
        .unwrap_or_else(|_| "kernels/topp_prefix_sum.hsaco".to_string());

    let (topp_prefix_sum_module, topp_prefix_sum_kernel) = if Path::new(&topp_prefix_sum_path).exists() {
        let module = load_backend.load_module(&topp_prefix_sum_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topp_prefix_sum_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Top-p prefix sum kernel not found at {} (TOPP_PREFIX_SUM_HSACO), using CPU fallback", topp_prefix_sum_path);
        (None, None)
    };

    // Load topp_threshold_kernel from TOPP_THRESHOLD_HSACO
    let topp_threshold_path = std::env::var("TOPP_THRESHOLD_HSACO")
        .unwrap_or_else(|_| "kernels/topp_threshold.hsaco".to_string());

    let (topp_threshold_module, topp_threshold_kernel) = if Path::new(&topp_threshold_path).exists() {
        let module = load_backend.load_module(&topp_threshold_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topp_threshold_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Top-p threshold kernel not found at {} (TOPP_THRESHOLD_HSACO), using CPU fallback", topp_threshold_path);
        (None, None)
    };

    // Load topp_sample_kernel from TOPP_SAMPLE_HSACO
    let topp_sample_path = std::env::var("TOPP_SAMPLE_HSACO")
        .unwrap_or_else(|_| "kernels/topp_sample.hsaco".to_string());

    let (topp_sample_module, topp_sample_kernel) = if Path::new(&topp_sample_path).exists() {
        let module = load_backend.load_module(&topp_sample_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topp_sample_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Top-p sample kernel not found at {} (TOPP_SAMPLE_HSACO), using CPU fallback", topp_sample_path);
        (None, None)
    };

    // Load topk_topp_sampling_kernel (fused) from FUSED_SAMPLING_HSACO
    let fused_path = std::env::var("FUSED_SAMPLING_HSACO")
        .unwrap_or_else(|_| "kernels/fused_sampling.hsaco".to_string());

    let (fused_module, fused_kernel) = if Path::new(&fused_path).exists() {
        let module = load_backend.load_module(&fused_path)?;
        let kernel = load_backend.get_kernel_function(&module, "topk_topp_sampling_kernel")?;
        (Some(module), Some(kernel))
    } else {
        tracing::warn!("Fused sampling kernel not found at {} (FUSED_SAMPLING_HSACO), using CPU fallback", fused_path);
        (None, None)
    };

    // Initialize cache with all loaded kernels
    *cache = Some(SamplingKernelCache {
        softmax_module,
        softmax_kernel,
        temperature_scale_module,
        temperature_scale_kernel,
        topk_module,
        topk_kernel,
        topp_prefix_sum_module,
        topp_prefix_sum_kernel,
        topp_threshold_module,
        topp_threshold_kernel,
        topp_sample_module,
        topp_sample_kernel,
        fused_module,
        fused_kernel,
    });

    Ok(&GLOBAL_SAMPLING_CACHE)
}

/// Launch temperature scale kernel on GPU
///
/// Scales logits by 1/temperature for temperature sampling.
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `logits` - GPU pointer to logits tensor [batch_size, vocab_size]
/// * `temperature` - Temperature value (higher = more random)
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn temperature_scale_kernel(
    backend: &HipBackend,
    logits: *mut f32,
    temperature: f32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.temperature_scale_kernel.as_ref()
                .ok_or_else(|| "temperature_scale_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut logits_arg = logits;
            let mut temperature_arg = temperature;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut logits_arg as *mut _ as *mut c_void,
                &mut temperature_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch temperature scale kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch top-k sampling kernel on GPU
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probabilities` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `output` - GPU pointer to output token IDs [batch_size]
/// * `top_k` - Number of top tokens to consider
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn topk_sampling_kernel(
    backend: &HipBackend,
    probabilities: *const f32,
    random_values: *const f32,
    output: *mut u32,
    top_k: u32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.topk_kernel.as_ref()
                .ok_or_else(|| "topk_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            // Prepare kernel arguments
            let mut probabilities_arg = probabilities;
            let mut random_values_arg = random_values;
            let mut output_arg = output;
            let mut top_k_arg = top_k;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut probabilities_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_k_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch top-k kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch top-p prefix sum kernel on GPU (Kernel 1 of 3 for top-p sampling)
///
/// Computes cumulative distribution function (CDF) for top-p sampling.
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probs` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `prefix_sum_out` - GPU pointer to output CDF [batch_size, vocab_size]
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn topp_prefix_sum_kernel(
    backend: &HipBackend,
    probs: *const f32,
    prefix_sum_out: *mut f32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.topp_prefix_sum_kernel.as_ref()
                .ok_or_else(|| "topp_prefix_sum_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut probs_arg = probs;
            let mut prefix_sum_out_arg = prefix_sum_out;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut probs_arg as *mut _ as *mut c_void,
                &mut prefix_sum_out_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch top-p prefix sum kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch top-p threshold kernel on GPU (Kernel 2 of 3 for top-p sampling)
///
/// Finds the threshold index for top-p sampling using binary search on CDF.
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `prefix_sum` - GPU pointer to CDF [batch_size, vocab_size]
/// * `threshold_out` - GPU pointer to output threshold indices [batch_size]
/// * `top_p` - Cumulative probability threshold
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn topp_threshold_kernel(
    backend: &HipBackend,
    prefix_sum: *const f32,
    threshold_out: *mut i32,
    top_p: f32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.topp_threshold_kernel.as_ref()
                .ok_or_else(|| "topp_threshold_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut prefix_sum_arg = prefix_sum;
            let mut threshold_out_arg = threshold_out;
            let mut top_p_arg = top_p;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut prefix_sum_arg as *mut _ as *mut c_void,
                &mut threshold_out_arg as *mut _ as *mut c_void,
                &mut top_p_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch top-p threshold kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch top-p sample kernel on GPU (Kernel 3 of 3 for top-p sampling)
///
/// Samples token IDs using binary search on CDF within threshold.
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `prefix_sum` - GPU pointer to CDF [batch_size, vocab_size]
/// * `threshold_idx` - GPU pointer to threshold indices [batch_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `sampled_tokens` - GPU pointer to output token IDs [batch_size]
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn topp_sample_kernel(
    backend: &HipBackend,
    prefix_sum: *const f32,
    threshold_idx: *const i32,
    random_values: *const f32,
    sampled_tokens: *mut i32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.topp_sample_kernel.as_ref()
                .ok_or_else(|| "topp_sample_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut prefix_sum_arg = prefix_sum;
            let mut threshold_idx_arg = threshold_idx;
            let mut random_values_arg = random_values;
            let mut sampled_tokens_arg = sampled_tokens;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut prefix_sum_arg as *mut _ as *mut c_void,
                &mut threshold_idx_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut sampled_tokens_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch top-p sample kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Legacy top-p sampling kernel wrapper (deprecated - use multi-kernel pipeline)
///
/// This function is kept for compatibility but will return an error
/// since top-p sampling now requires a 3-kernel pipeline.
/// Use topp_prefix_sum_kernel, topp_threshold_kernel, and topp_sample_kernel instead.
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probabilities` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `output` - GPU pointer to output token IDs [batch_size]
/// * `top_p` - Cumulative probability threshold
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
#[deprecated(since = "0.2.0", note = "Use topp_prefix_sum_kernel, topp_threshold_kernel, and topp_sample_kernel instead")]
pub unsafe fn topp_sampling_kernel(
    _backend: &HipBackend,
    _probabilities: *const f32,
    _random_values: *const f32,
    _output: *mut u32,
    _top_p: f32,
    _batch_size: u32,
    _vocab_size: u32,
) -> Result<(), String> {
    Err("topp_sampling_kernel is deprecated. Use the 3-kernel pipeline: topp_prefix_sum_kernel, topp_threshold_kernel, topp_sample_kernel".to_string())
}

/// Launch fused top-k + top-p sampling kernel on GPU
///
/// # Arguments
/// * `backend` - HipBackend for kernel launch
/// * `probabilities` - GPU pointer to probability tensor [batch_size, vocab_size]
/// * `random_values` - GPU pointer to random values [batch_size]
/// * `output` - GPU pointer to output token IDs [batch_size]
/// * `top_k` - Number of top tokens to consider
/// * `top_p` - Cumulative probability threshold
/// * `batch_size` - Number of batch elements
/// * `vocab_size` - Vocabulary size
///
/// # Safety
/// Caller must ensure all GPU pointers are valid and synchronized after this call.
#[cfg(feature = "rocm")]
pub unsafe fn fused_sampling_kernel(
    backend: &HipBackend,
    probabilities: *const f32,
    random_values: *const f32,
    output: *mut u32,
    top_k: u32,
    top_p: f32,
    batch_size: u32,
    vocab_size: u32,
) -> Result<(), String> {
    match get_or_init_sampling_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref.lock()
                .map_err(|e| format!("Failed to lock sampling cache: {}", e))?;
            let cache_ref = cache.as_ref()
                .ok_or_else(|| "Sampling cache not initialized".to_string())?;

            let kernel = cache_ref.fused_kernel.as_ref()
                .ok_or_else(|| "fused_kernel not loaded".to_string())?;

            let grid_dim = (batch_size, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            // Prepare kernel arguments
            let mut probabilities_arg = probabilities;
            let mut random_values_arg = random_values;
            let mut output_arg = output;
            let mut top_k_arg = top_k;
            let mut top_p_arg = top_p;
            let mut batch_size_arg = batch_size;
            let mut vocab_size_arg = vocab_size;

            let args: &[*mut c_void] = &[
                &mut probabilities_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_k_arg as *mut _ as *mut c_void,
                &mut top_p_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            backend.launch_kernel_with_module_shared(
                kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| format!("Failed to launch fused kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Generate random values for GPU sampling
///
/// For now, generates on CPU. In the future, this could use GPU RNG.
#[cfg(feature = "rocm")]
pub fn generate_random(_backend: &crate::backend::HipBackend, count: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}
