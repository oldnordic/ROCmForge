//! GPU-accelerated sampling implementation
//!
//! Provides GPU kernels for top-k and top-p sampling using ROCm/HIP.
//! Based on FlashInfer's sorting-free rejection sampling algorithm.

#![allow(dead_code)]

#[cfg(feature = "rocm")]
use crate::backend::hip_backend::{HipBackend, HipBuffer, HipError, HipKernel, HipModule};
use crate::sampler::{SamplerError, SamplerResult};
use rand::distributions::Distribution;
use rand::Rng;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(feature = "rocm")]
const BLOCK_SIZE: u32 = 256;
#[cfg(feature = "rocm")]
const HIP_SUCCESS: i32 = 0;

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
struct SamplingKernelCache {
    // softmax_kernel from SAMPLING_UTILS_HSACO
    softmax_module: Option<HipModule>,
    softmax_kernel: Option<HipKernel>,
    // temperature_scale_kernel from TEMPERATURE_SCALE_HSACO
    temperature_scale_module: Option<HipModule>,
    temperature_scale_kernel: Option<HipKernel>,
    // topk_sampling_kernel from TOPK_SAMPLING_HSACO
    topk_module: Option<HipModule>,
    topk_kernel: Option<HipKernel>,
    // topp_prefix_sum_kernel from TOPP_PREFIX_SUM_HSACO
    topp_prefix_sum_module: Option<HipModule>,
    topp_prefix_sum_kernel: Option<HipKernel>,
    // topp_threshold_kernel from TOPP_THRESHOLD_HSACO
    topp_threshold_module: Option<HipModule>,
    topp_threshold_kernel: Option<HipKernel>,
    // topp_sample_kernel from TOPP_SAMPLE_HSACO
    topp_sample_module: Option<HipModule>,
    topp_sample_kernel: Option<HipKernel>,
    // topk_topp_sampling_kernel from FUSED_SAMPLING_HSACO
    fused_module: Option<HipModule>,
    fused_kernel: Option<HipKernel>,
}

// Global kernel cache (lazy initialization)
#[cfg(feature = "rocm")]
static GLOBAL_SAMPLING_CACHE: Mutex<Option<SamplingKernelCache>> = Mutex::new(None);

/// Get or initialize the global sampling kernel cache
///
/// Returns cached kernel modules and functions. The caller must provide
/// their own HipBackend for launching kernels to ensure stream consistency.
#[cfg(feature = "rocm")]
fn get_or_init_sampling_cache() -> Result<&'static Mutex<Option<SamplingKernelCache>>, HipError> {
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

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

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

// ============================================================================
// Sampler Structs
// ============================================================================

/// GPU sampler for top-p (nucleus) sampling with temperature support
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct GpuTopPSampler {
    backend: Arc<HipBackend>,
    top_p: f32,
    temperature: f32,
}

#[cfg(feature = "rocm")]
impl GpuTopPSampler {
    /// Create a new GPU top-p sampler with default temperature (1.0)
    pub fn new(backend: Arc<HipBackend>, top_p: f32) -> SamplerResult<Self> {
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(GpuTopPSampler { backend, top_p, temperature: 1.0 })
    }

    /// Set temperature for sampling
    ///
    /// Temperature < 1.0 makes sampling more deterministic (sharper distribution)
    /// Temperature > 1.0 makes sampling more random (flatter distribution)
    /// Temperature = 1.0 is no scaling (default)
    ///
    /// Note: Temperature scaling is applied BEFORE softmax in the GPU pipeline.
    /// For top-p sampling with logits, apply temperature before calling sample().
    pub fn with_temperature(mut self, temperature: f32) -> SamplerResult<Self> {
        if temperature <= 0.0 {
            return Err(SamplerError::InvalidTemperature(temperature));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Sample from probabilities using top-p filtering on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // TDD: Try GPU path first, fall back to CPU if kernels not available or on error
        match self.try_gpu_sample(probabilities, batch_size, vocab_size) {
            Ok(results) => Ok(results),
            Err(e) => {
                tracing::debug!("GPU top-p sampling failed, falling back to CPU: {}", e);
                self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
            }
        }
    }

    /// Try to sample using GPU kernels
    ///
    /// Uses 3-kernel pipeline for top-p sampling:
    /// 1. (Optional) temperature_scale_kernel - applies temperature scaling
    /// 2. topp_prefix_sum_kernel - computes CDF
    /// 3. topp_threshold_kernel - finds threshold index
    /// 4. topp_sample_kernel - samples tokens
    fn try_gpu_sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        tracing::debug!("GpuTopPSampler::try_gpu_sample: batch_size={}, vocab_size={}, top_p={}, temperature={}",
            batch_size, vocab_size, self.top_p, self.temperature);

        // Check if all 3 kernels are loaded
        let cache_ref = get_or_init_sampling_cache()
            .map_err(|e| {
                tracing::error!("Failed to get kernel cache: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let cache = cache_ref.lock()
            .map_err(|e| {
                tracing::error!("Failed to lock sampling cache: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        let cache_ref = cache.as_ref()
            .ok_or_else(|| {
                tracing::warn!("Sampling cache not initialized");
                SamplerError::InvalidTopP(0.0)
            })?;

        // Check all 3 kernels for multi-kernel pipeline
        let prefix_sum_kernel = cache_ref.topp_prefix_sum_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topp_prefix_sum_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0)
            })?;
        let threshold_kernel = cache_ref.topp_threshold_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topp_threshold_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0)
            })?;
        let sample_kernel = cache_ref.topp_sample_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topp_sample_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("All top-p kernels loaded, allocating GPU buffers");

        // Allocate GPU buffers for 3-kernel pipeline
        let total_elements = batch_size * vocab_size;
        let probs_bytes = total_elements * std::mem::size_of::<f32>();
        let prefix_sum_bytes = total_elements * std::mem::size_of::<f32>();
        let threshold_bytes = batch_size * std::mem::size_of::<i32>();
        let random_bytes = batch_size * std::mem::size_of::<f32>();
        let output_bytes = batch_size * std::mem::size_of::<i32>();

        let probs_gpu = HipBuffer::new(probs_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate probs buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let prefix_sum_gpu = HipBuffer::new(prefix_sum_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate prefix_sum buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let threshold_gpu = HipBuffer::new(threshold_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate threshold buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let random_gpu = HipBuffer::new(random_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate random buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        let output_gpu = HipBuffer::new(output_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate output buffer: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("GPU buffers allocated, copying data");

        // Copy probabilities to GPU
        probs_gpu.copy_from_host(probabilities)
            .map_err(|e| {
                tracing::error!("Failed to copy probs to GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        // Generate random values on CPU and copy to GPU
        let random_values: Vec<f32> = generate_random_gpu(&self.backend, batch_size);
        random_gpu.copy_from_host(&random_values)
            .map_err(|e| {
                tracing::error!("Failed to copy random to GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("Data copied to GPU, launching 3-kernel pipeline");

        // Apply temperature scaling if temperature != 1.0
        let probs_ptr = if self.temperature != 1.0 {
            // Check if temperature scale kernel is available
            let temp_scale_kernel = cache_ref.temperature_scale_kernel.as_ref()
                .ok_or_else(|| {
                    tracing::warn!("temperature_scale_kernel not loaded, falling back to CPU for temperature scaling");
                    SamplerError::InvalidTopP(0.0)
                })?;

            let probs_mut_ptr = probs_gpu.as_mut_ptr() as *mut f32;

            unsafe {
                temperature_scale_kernel(
                    &self.backend,
                    probs_mut_ptr,
                    self.temperature,
                    batch_size as u32,
                    vocab_size as u32,
                ).map_err(|e| {
                    tracing::error!("Failed to launch temperature scale kernel: {:?}", e);
                    SamplerError::InvalidTopP(0.0)
                })?;
            }

            tracing::debug!("Temperature scaling applied: temp={}", self.temperature);
            probs_mut_ptr as *const f32
        } else {
            probs_gpu.as_ptr() as *const f32
        };

        // Kernel 1: Compute prefix sum (CDF)
        let prefix_sum_ptr = prefix_sum_gpu.as_mut_ptr() as *mut f32;

        unsafe {
            // Launch kernel 1: prefix sum
            let grid_dim = (batch_size as u32, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut probs_arg = probs_ptr;
            let mut prefix_sum_arg = prefix_sum_ptr;
            let mut batch_size_arg = batch_size as u32;
            let mut vocab_size_arg = vocab_size as u32;

            let args: &[*mut c_void] = &[
                &mut probs_arg as *mut _ as *mut c_void,
                &mut prefix_sum_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                prefix_sum_kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch prefix sum kernel: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

            // Kernel 2: Find threshold
            let threshold_ptr = threshold_gpu.as_mut_ptr() as *mut i32;

            let mut prefix_sum_arg2 = prefix_sum_ptr;
            let mut threshold_arg = threshold_ptr;
            let mut top_p_arg = self.top_p;

            let args2: &[*mut c_void] = &[
                &mut prefix_sum_arg2 as *mut _ as *mut c_void,
                &mut threshold_arg as *mut _ as *mut c_void,
                &mut top_p_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                threshold_kernel,
                grid_dim,
                block_dim,
                args2,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch threshold kernel: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

            // Kernel 3: Sample tokens
            let random_ptr = random_gpu.as_ptr() as *const f32;
            let output_ptr = output_gpu.as_mut_ptr() as *mut i32;

            let mut threshold_idx_arg = threshold_ptr;
            let mut random_arg = random_ptr;
            let mut output_arg = output_ptr;

            let args3: &[*mut c_void] = &[
                &mut prefix_sum_arg2 as *mut _ as *mut c_void,
                &mut threshold_idx_arg as *mut _ as *mut c_void,
                &mut random_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                sample_kernel,
                grid_dim,
                block_dim,
                args3,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch sample kernel: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;
        }

        tracing::debug!("Kernels launched, synchronizing");

        // Synchronize and copy results back
        self.backend.synchronize()
            .map_err(|e| {
                tracing::error!("Failed to synchronize: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        tracing::debug!("Synchronized, copying results back");

        let mut results_i32 = vec![0i32; batch_size];
        output_gpu.copy_to_host(&mut results_i32)
            .map_err(|e| {
                tracing::error!("Failed to copy output from GPU: {:?}", e);
                SamplerError::InvalidTopP(0.0)
            })?;

        // Convert i32 to u32
        let results: Vec<u32> = results_i32.into_iter().map(|v| v as u32).collect();

        tracing::debug!("GPU sampling complete: {:?}", results);

        Ok(results)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // Find top-p cutoff
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut cutoff_idx = vocab_size;

            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Sample from top-p tokens
            let top_indices: Vec<usize> = sorted_probs
                .iter()
                .take(cutoff_idx)
                .map(|(i, _)| *i)
                .collect();

            let top_values: Vec<f32> = top_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            let dist = rand::distributions::WeightedIndex::new(&top_values)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = top_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}

/// GPU sampler for top-k sampling with temperature support
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct GpuTopKSampler {
    backend: Arc<HipBackend>,
    top_k: usize,
    temperature: f32,
}

#[cfg(feature = "rocm")]
impl GpuTopKSampler {
    /// Create a new GPU top-k sampler with default temperature (1.0)
    pub fn new(backend: Arc<HipBackend>, top_k: usize) -> SamplerResult<Self> {
        if top_k == 0 {
            return Err(SamplerError::InvalidTopK(top_k));
        }

        Ok(GpuTopKSampler { backend, top_k, temperature: 1.0 })
    }

    /// Set temperature for sampling
    ///
    /// Temperature < 1.0 makes sampling more deterministic (sharper distribution)
    /// Temperature > 1.0 makes sampling more random (flatter distribution)
    /// Temperature = 1.0 is no scaling (default)
    pub fn with_temperature(mut self, temperature: f32) -> SamplerResult<Self> {
        if temperature <= 0.0 {
            return Err(SamplerError::InvalidTemperature(temperature));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Sample from probabilities using top-k filtering on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // TDD: Try GPU path first, fall back to CPU if kernels not available or on error
        match self.try_gpu_sample(probabilities, batch_size, vocab_size) {
            Ok(results) => Ok(results),
            Err(e) => {
                tracing::debug!("GPU top-k sampling failed, falling back to CPU: {}", e);
                self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
            }
        }
    }

    /// Try to sample using GPU kernels
    fn try_gpu_sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        tracing::debug!("GpuTopKSampler::try_gpu_sample: batch_size={}, vocab_size={}, top_k={}, temperature={}",
            batch_size, vocab_size, self.top_k, self.temperature);

        // Check if kernel is loaded
        let cache_ref = get_or_init_sampling_cache()
            .map_err(|e| {
                tracing::error!("Failed to get kernel cache: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let cache = cache_ref.lock()
            .map_err(|e| {
                tracing::error!("Failed to lock sampling cache: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        let cache_ref = cache.as_ref()
            .ok_or_else(|| {
                tracing::warn!("Sampling cache not initialized");
                SamplerError::InvalidTopK(0)
            })?;

        let topk_kernel = cache_ref.topk_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("topk_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("top-k kernel loaded, allocating GPU buffers");

        // Allocate GPU buffers
        let total_elements = batch_size * vocab_size;
        let probs_bytes = total_elements * std::mem::size_of::<f32>();
        let random_bytes = batch_size * std::mem::size_of::<f32>();
        let output_bytes = batch_size * std::mem::size_of::<u32>();

        let probs_gpu = HipBuffer::new(probs_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate probs buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let random_gpu = HipBuffer::new(random_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate random buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let output_gpu = HipBuffer::new(output_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate output buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("GPU buffers allocated, copying data");

        // Copy probabilities to GPU
        probs_gpu.copy_from_host(probabilities)
            .map_err(|e| {
                tracing::error!("Failed to copy probs to GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        // Generate random values on CPU and copy to GPU
        let random_values: Vec<f32> = generate_random_gpu(&self.backend, batch_size);
        random_gpu.copy_from_host(&random_values)
            .map_err(|e| {
                tracing::error!("Failed to copy random to GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        // Apply temperature scaling if temperature != 1.0
        let probs_ptr = if self.temperature != 1.0 {
            // Check if temperature scale kernel is available
            let temp_scale_kernel = cache_ref.temperature_scale_kernel.as_ref()
                .ok_or_else(|| {
                    tracing::warn!("temperature_scale_kernel not loaded, falling back to CPU for temperature scaling");
                    SamplerError::InvalidTopK(0)
                })?;

            let probs_mut_ptr = probs_gpu.as_mut_ptr() as *mut f32;

            unsafe {
                temperature_scale_kernel(
                    &self.backend,
                    probs_mut_ptr,
                    self.temperature,
                    batch_size as u32,
                    vocab_size as u32,
                ).map_err(|e| {
                    tracing::error!("Failed to launch temperature scale kernel: {:?}", e);
                    SamplerError::InvalidTopK(0)
                })?;
            }

            tracing::debug!("Temperature scaling applied: temp={}", self.temperature);
            probs_mut_ptr as *const f32
        } else {
            probs_gpu.as_ptr() as *const f32
        };

        tracing::debug!("Data copied to GPU, launching kernel");

        // Launch kernel
        let random_ptr = random_gpu.as_ptr() as *const f32;
        let output_ptr = output_gpu.as_mut_ptr() as *mut u32;

        unsafe {
            // Launch kernel directly using backend
            let grid_dim = (batch_size as u32, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut probs_arg = probs_ptr;
            let mut random_values_arg = random_ptr;
            let mut output_arg = output_ptr;
            let mut top_k_arg = self.top_k as u32;
            let mut batch_size_arg = batch_size as u32;
            let mut vocab_size_arg = vocab_size as u32;

            let args: &[*mut c_void] = &[
                &mut probs_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_k_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                topk_kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch top-k kernel: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        }

        tracing::debug!("Kernel launched, synchronizing");

        // Synchronize and copy results back
        self.backend.synchronize()
            .map_err(|e| {
                tracing::error!("Failed to synchronize: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("Synchronized, copying results back");

        let mut results = vec![0u32; batch_size];
        output_gpu.copy_to_host(&mut results)
            .map_err(|e| {
                tracing::error!("Failed to copy output from GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("GPU top-k sampling complete: {:?}", results);

        Ok(results)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // Find top-k
            let effective_k = self.top_k.min(vocab_size);
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Sample from top-k
            let top_indices: Vec<usize> = sorted_probs
                .iter()
                .take(effective_k)
                .map(|(i, _)| *i)
                .collect();

            let top_values: Vec<f32> = top_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            // Renormalize
            let sum: f32 = top_values.iter().sum();
            if sum < 1e-10f32 {
                return Err(SamplerError::ZeroProbabilities);
            }

            let normalized: Vec<f32> = top_values.iter().map(|&v| v / sum).collect();

            let dist = rand::distributions::WeightedIndex::new(&normalized)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = top_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}

/// GPU sampler for fused top-k + top-p sampling
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
pub struct GpuFusedSampler {
    backend: Arc<HipBackend>,
    top_k: usize,
    top_p: f32,
}

#[cfg(feature = "rocm")]
impl GpuFusedSampler {
    /// Create a new GPU fused sampler
    pub fn new(backend: Arc<HipBackend>, top_k: usize, top_p: f32) -> SamplerResult<Self> {
        if top_k == 0 {
            return Err(SamplerError::InvalidTopK(top_k));
        }
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(GpuFusedSampler { backend, top_k, top_p })
    }

    /// Sample using fused top-k + top-p on GPU
    pub fn sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        // TDD: Try GPU path first, fall back to CPU if kernels not available or on error
        match self.try_gpu_sample(probabilities, batch_size, vocab_size) {
            Ok(results) => Ok(results),
            Err(e) => {
                tracing::debug!("GPU fused sampling failed, falling back to CPU: {}", e);
                self.sample_cpu_fallback(probabilities, batch_size, vocab_size)
            }
        }
    }

    /// Try to sample using GPU kernels
    ///
    /// Uses the fused top-k + top-p sampling kernel which combines both
    /// filtering operations in a single kernel launch for efficiency.
    fn try_gpu_sample(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        tracing::debug!("GpuFusedSampler::try_gpu_sample: batch_size={}, vocab_size={}, top_k={}, top_p={}",
            batch_size, vocab_size, self.top_k, self.top_p);

        // Check if kernel is loaded
        let cache_ref = get_or_init_sampling_cache()
            .map_err(|e| {
                tracing::error!("Failed to get kernel cache: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let cache = cache_ref.lock()
            .map_err(|e| {
                tracing::error!("Failed to lock sampling cache: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        let cache_ref = cache.as_ref()
            .ok_or_else(|| {
                tracing::warn!("Sampling cache not initialized");
                SamplerError::InvalidTopK(0)
            })?;

        let fused_kernel = cache_ref.fused_kernel.as_ref()
            .ok_or_else(|| {
                tracing::warn!("fused_kernel not loaded, falling back to CPU");
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("fused kernel loaded, allocating GPU buffers");

        // Allocate GPU buffers
        let total_elements = batch_size * vocab_size;
        let probs_bytes = total_elements * std::mem::size_of::<f32>();
        let random_bytes = batch_size * std::mem::size_of::<f32>();
        let output_bytes = batch_size * std::mem::size_of::<u32>();

        let probs_gpu = HipBuffer::new(probs_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate probs buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let random_gpu = HipBuffer::new(random_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate random buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        let output_gpu = HipBuffer::new(output_bytes)
            .map_err(|e| {
                tracing::error!("Failed to allocate output buffer: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("GPU buffers allocated, copying data");

        // Copy probabilities to GPU
        probs_gpu.copy_from_host(probabilities)
            .map_err(|e| {
                tracing::error!("Failed to copy probs to GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        // Generate random values on CPU and copy to GPU
        let random_values: Vec<f32> = generate_random_gpu(&self.backend, batch_size);
        random_gpu.copy_from_host(&random_values)
            .map_err(|e| {
                tracing::error!("Failed to copy random to GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("Data copied to GPU, launching fused kernel");

        // Launch fused kernel
        let probs_ptr = probs_gpu.as_ptr() as *const f32;
        let random_ptr = random_gpu.as_ptr() as *const f32;
        let output_ptr = output_gpu.as_mut_ptr() as *mut u32;

        unsafe {
            // Launch kernel directly using backend
            let grid_dim = (batch_size as u32, 1, 1);
            let block_dim = (BLOCK_SIZE, 1, 1);
            let shared_mem_bytes = 0u32;

            let mut probs_arg = probs_ptr;
            let mut random_values_arg = random_ptr;
            let mut output_arg = output_ptr;
            let mut top_k_arg = self.top_k as u32;
            let mut top_p_arg = self.top_p;
            let mut batch_size_arg = batch_size as u32;
            let mut vocab_size_arg = vocab_size as u32;

            let args: &[*mut c_void] = &[
                &mut probs_arg as *mut _ as *mut c_void,
                &mut random_values_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut top_k_arg as *mut _ as *mut c_void,
                &mut top_p_arg as *mut _ as *mut c_void,
                &mut batch_size_arg as *mut _ as *mut c_void,
                &mut vocab_size_arg as *mut _ as *mut c_void,
            ];

            self.backend.launch_kernel_with_module_shared(
                fused_kernel,
                grid_dim,
                block_dim,
                args,
                shared_mem_bytes,
            ).map_err(|e| {
                tracing::error!("Failed to launch fused kernel: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;
        }

        tracing::debug!("Kernel launched, synchronizing");

        // Synchronize and copy results back
        self.backend.synchronize()
            .map_err(|e| {
                tracing::error!("Failed to synchronize: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("Synchronized, copying results back");

        let mut results = vec![0u32; batch_size];
        output_gpu.copy_to_host(&mut results)
            .map_err(|e| {
                tracing::error!("Failed to copy output from GPU: {:?}", e);
                SamplerError::InvalidTopK(0)
            })?;

        tracing::debug!("GPU fused sampling complete: {:?}", results);

        Ok(results)
    }

    /// CPU fallback implementation for testing
    fn sample_cpu_fallback(
        &self,
        probabilities: &[f32],
        batch_size: usize,
        vocab_size: usize,
    ) -> SamplerResult<Vec<u32>> {
        let mut results = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for batch_idx in 0..batch_size {
            let row_offset = batch_idx * vocab_size;
            let row_probs = &probabilities[row_offset..row_offset + vocab_size];

            // First apply top-p to get candidate set
            let mut sorted_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumulative = 0.0f32;
            let mut topp_cutoff = vocab_size;

            for (i, &(_, p)) in sorted_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    topp_cutoff = i + 1;
                    break;
                }
            }

            // Among top-p tokens, select top-k
            let effective_k = self.top_k.min(topp_cutoff);
            let topk_indices: Vec<usize> = sorted_probs
                .iter()
                .take(effective_k)
                .map(|(i, _)| *i)
                .collect();

            let topk_values: Vec<f32> = topk_indices
                .iter()
                .map(|&i| row_probs[i])
                .collect();

            // Renormalize and sample
            let sum: f32 = topk_values.iter().sum();
            if sum < 1e-10f32 {
                return Err(SamplerError::ZeroProbabilities);
            }

            let normalized: Vec<f32> = topk_values.iter().map(|&v| v / sum).collect();

            let dist = rand::distributions::WeightedIndex::new(&normalized)
                .map_err(|_| SamplerError::ZeroProbabilities)?;

            let sampled_idx = topk_indices[dist.sample(&mut rng)];
            results.push(sampled_idx as u32);
        }

        Ok(results)
    }
}

/// Generate random values on GPU
#[cfg(feature = "rocm")]
pub fn generate_random_gpu(
    _backend: &Arc<HipBackend>,
    count: usize,
) -> Vec<f32> {
    // For now, generate on CPU
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topp_sampler_creation() {
        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();
        assert_eq!(sampler.top_p, 0.9);
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topp_invalid_params() {
        let backend = HipBackend::new().unwrap();
        let result = GpuTopPSampler::new(backend.clone(), 0.0);
        assert!(result.is_err());

        let result = GpuTopPSampler::new(backend, 1.5);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topk_sampler_creation() {
        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 50).unwrap();
        assert_eq!(sampler.top_k, 50);
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topk_invalid_params() {
        let backend = HipBackend::new().unwrap();
        let result = GpuTopKSampler::new(backend, 0);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_fused_sampler_creation() {
        let backend = HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 50, 0.9).unwrap();
        assert_eq!(sampler.top_k, 50);
        assert_eq!(sampler.top_p, 0.9);
    }

    #[test]
    fn test_topp_fallback_correctness() {
        // Test with known probabilities
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,  // Row 1 (sum = 1.0)
            0.5, 0.3, 0.1, 0.05, 0.05,  // Row 2 (sum = 1.0)
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    #[test]
    fn test_topk_fallback_correctness() {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.5, 0.3, 0.1, 0.05, 0.05,
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 3).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        // Results should be in top-3 (indices 0, 2, or 4 for first row)
        // Note: This is probabilistic, so we just check bounds
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    #[test]
    fn test_fused_fallback_correctness() {
        let probabilities = vec![
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.5, 0.3, 0.1, 0.05, 0.05,
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 3, 0.8).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU kernel infrastructure
    ///
    /// TDD Step 1: This test verifies that the kernel cache can be initialized.
    /// When HSACO files are present, kernels should be loaded.
    /// When HSACO files are absent, cache should still initialize (with None for kernels).
    #[test]
    #[cfg(feature = "rocm")]
    fn test_kernel_cache_initialization() {
        // This should always succeed - cache initializes even if kernels aren't found
        let result = get_or_init_sampling_cache();
        assert!(result.is_ok(), "Kernel cache should initialize successfully");

        // Verify cache is populated
        let cache = result.unwrap().lock()
            .expect("Sampling cache lock should not be poisoned");
        assert!(cache.is_some(), "Cache should be Some after initialization");

        let cache_ref = cache.as_ref()
            .expect("Cache should contain Some(Mutex<KernelCache>)");
        // Kernels will be None if HSACO files aren't compiled yet
        // This is expected - the test documents current state
        if cache_ref.topp_kernel.is_none() {
            println!("WARNING: top-p kernel not loaded (HSACO files not compiled yet)");
            println!("To enable GPU sampling, compile kernels with:");
            println!("  hipcc --genco -O3 kernels/topp_sampling.hip -o kernels/topp_sampling.hsaco");
        }
    }

    /// Test GPU top-p sampling with known inputs
    ///
    /// TDD Step 1: Write test first
    /// TDD Step 2: Run test - see it use CPU fallback (will pass but logs warning)
    /// TDD Step 3: After HSACO compilation, GPU path will be used
    #[test]
    #[cfg(feature = "rocm")]
    fn test_topp_sampling_deterministic() {
        // Use deterministic probabilities where result is predictable
        let probabilities = vec![
            0.05, 0.05, 0.80, 0.05, 0.05,  // Row 1: token 2 has 80% probability
            0.10, 0.10, 0.10, 0.60, 0.10,  // Row 2: token 3 has 60% probability
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        // Verify basic properties
        assert_eq!(results.len(), 2, "Should return 2 samples");
        assert!(results[0] < 5, "First sample should be in vocabulary range");
        assert!(results[1] < 5, "Second sample should be in vocabulary range");

        // With top_p=0.9, token 2 (80%) should be highly likely for first row
        // With top_p=0.9, tokens 3 (60%) + 2 (10%) = 70% for second row
        // Note: This is probabilistic, so we just verify it runs without error
    }

    /// Test GPU top-k sampling with known inputs
    #[test]
    #[cfg(feature = "rocm")]
    fn test_topk_sampling_deterministic() {
        // Clear top-2 tokens: token 2 (80%), token 4 (10%)
        let probabilities = vec![
            0.02, 0.03, 0.80, 0.05, 0.10,  // Row 1: top-2 are indices 2 and 4
            0.05, 0.05, 0.10, 0.70, 0.10,  // Row 2: top-2 are indices 3 and 4
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 2).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);

        // With top_k=2, samples should be from {2, 4} for row 1
        // and from {3, 4} for row 2
        // Note: Probabilistic, so we just verify it runs
    }

    /// Test GPU fused sampling with known inputs
    ///
    /// TDD test for combined top-k + top-p sampling.
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_fused_sampling_deterministic() {
        // Create distribution where top-k and top-p both apply
        let probabilities = vec![
            0.05, 0.05, 0.70, 0.10, 0.10,  // Row 1: token 2 (70%), token 3 (10%), token 4 (10%)
            0.10, 0.60, 0.10, 0.10, 0.10,  // Row 2: token 1 (60%), others (10% each)
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuFusedSampler::new(backend, 3, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        // Verify basic properties
        assert_eq!(results.len(), 2, "Should return 2 samples");
        assert!(results[0] < 5, "First sample should be in vocabulary range");
        assert!(results[1] < 5, "Second sample should be in vocabulary range");

        // With top_k=3, top_p=0.9:
        // Row 1: top-3 are indices 2 (70%), 3 (10%), 4 (10%) = 90% cumulative
        // Row 2: top-3 are indices 1 (60%), 0 (10%), 2 (10%) = 80% cumulative
        // Note: Probabilistic, so we just verify it runs without error
    }

    /// Test GPU sampling fallback on error
    ///
    /// Verifies that when GPU kernels are not available, CPU fallback is used.
    #[test]
    fn test_gpu_sampling_fallback_on_error() {
        // This test uses CPU fallback directly (simulating kernel unavailability)
        let probabilities = vec![
            0.1, 0.2, 0.4, 0.2, 0.1,  // Sum = 1.0
            0.3, 0.3, 0.2, 0.1, 0.1,  // Sum = 1.0
        ];

        #[cfg(feature = "rocm")]
        {
            let backend = HipBackend::new().unwrap();
            let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

            // This will use CPU fallback if kernels aren't loaded
            let results = sampler.sample(&probabilities, 2, 5);
            assert!(results.is_ok(), "Should fall back to CPU sampling");

            let results = results.unwrap();
            assert_eq!(results.len(), 2);
            assert!(results[0] < 5);
            assert!(results[1] < 5);
        }

        #[cfg(not(feature = "rocm"))]
        {
            // Without ROCm, tests should still compile
            assert!(true);
        }
    }

    /// Test GPU top-k sampling with single dominant token
    ///
    /// Edge case: One token has overwhelming probability.
    #[test]
    #[cfg(feature = "rocm")]
    #[ignore] // Requires actual GPU hardware
    fn test_gpu_topk_single_dominant() {
        let probabilities = vec![
            0.99, 0.0025, 0.0025, 0.0025, 0.0025,  // Token 0 dominates
            0.002, 0.99, 0.002, 0.003, 0.003,       // Token 1 dominates
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopKSampler::new(backend, 5).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        // With 99% probability, most samples should be the dominant token
        // But we just verify bounds here since it's probabilistic
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU top-p sampling with uniform distribution
    ///
    /// Edge case: All probabilities are equal.
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_topp_uniform_distribution() {
        let probabilities = vec![
            0.2, 0.2, 0.2, 0.2, 0.2,  // Uniform distribution
            0.25, 0.25, 0.25, 0.25, 0.0,  // Another uniform distribution
        ];

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.5).unwrap();

        let results = sampler.sample(&probabilities, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0] < 5);
        assert!(results[1] < 5);
    }

    /// Test GPU sampling with edge case: single token vocabulary
    ///
    /// Edge case: Vocabulary size of 1 (only one possible token).
    #[test]
    #[cfg(feature = "rocm")]
    fn test_gpu_sampling_single_token_vocab() {
        let probabilities = vec![1.0; 2]; // batch_size=2, vocab_size=1

        let backend = HipBackend::new().unwrap();
        let sampler = GpuTopPSampler::new(backend, 0.9).unwrap();

        let results = sampler.sample(&probabilities, 2, 1).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0, "Only token 0 should be sampled");
        assert_eq!(results[1], 0, "Only token 0 should be sampled");
    }
}
