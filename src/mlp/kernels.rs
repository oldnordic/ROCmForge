//! MLP GPU kernel wrappers
//!
//! This module provides Rust FFI wrappers for HIP kernels implementing MLP operations.
//! SwiGLU activation and RMSNorm normalization.
//!
//! # FFI Wrapper Invariant (CRITICAL)
//!
//! ALL kernel arguments (including pointers) MUST be copied to intermediate mutable
//! variables before passing to HIP kernels. This is required by Rust's FFI ABI and
//! HIP's calling convention on AMDGPU.
//!
//! ## CORRECT Pattern
//! ```rust,ignore
//! // ALL args copied to mut locals first
//! let mut gate_arg = gate as *mut f32;
//! let mut up_arg = up as *mut f32;
//! let mut output_arg = output;
//! let mut seq_len_arg = seq_len;
//!
//! let args: &[*mut c_void] = &[
//!     &mut gate_arg as *mut _ as *mut c_void,
//!     &mut up_arg as *mut _ as *mut c_void,
//!     &mut output_arg as *mut _ as *mut c_void,
//!     &mut seq_len_arg as *mut _ as *mut c_void,
//! ];
//! ```
//!
//! ## WRONG Pattern (causes GPU memory faults)
//! ```rust,ignore
//! // Direct cast - FAILS with "Memory access fault by GPU node-1"
//! let args: &[*mut c_void] = &[
//!     gate as *mut c_void,      // WRONG
//!     up as *mut c_void,        // WRONG
//!     output as *mut c_void,    // WRONG
//! ];
//! ```
//!
//! # Reduction Invariant (CRITICAL)
//!
//! For parallel reduction using shared memory, the starting stride MUST be
//! `BLOCK_SIZE / 2` to ensure all elements participate in the reduction.
//!
//! ## CORRECT Pattern
//! ```cpp,ignore
//! for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
//!     if (tid < stride) {
//!         s_data[tid] += s_data[tid + stride];
//!     }
//!     __syncthreads();
//! }
//! ```
//!
//! ## WRONG Pattern (causes numerical errors)
//! ```cpp,ignore
//! // Only processes 31 elements for BLOCK_SIZE=256
//! for (int stride = 16; stride > 0; stride >>= 1) { ... }
//! ```

#![allow(dead_code)]

use std::ffi::c_void;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};

// RDNA3 (wave32) tuning constants for AMD Radeon RX 7900 XT
const BLOCK_SIZE: u32 = 256; // 8 waves of 32 threads
const WARP_SIZE: u32 = 32; // RDNA3 wavefront size

/// Cached kernel modules and functions for MLP operations
///
/// NOTE: We do NOT store HipBackend here because that would create a separate
/// HIP stream from the caller's. Kernels must be launched on the caller's stream
/// and synchronized on the same stream to avoid hangs.
#[derive(Debug)]
struct KernelCache {
    swiglu_module: Option<HipModule>,
    swiglu_kernel: Option<HipKernel>,
    rms_norm_module: Option<HipModule>,
    rms_norm_kernel: Option<HipKernel>,
}

// Global kernel cache (lazy initialization)
static GLOBAL_CACHE: Mutex<Option<KernelCache>> = Mutex::new(None);

/// Get or initialize the global kernel cache
///
/// Returns cached kernel modules and functions. The caller must provide
/// their own HipBackend for launching kernels to ensure stream consistency.
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

    // Load SwiGLU kernel module and function
    // Note: We use a temporary backend just for loading. The actual kernel
    // launch will use the caller's backend to ensure stream consistency.
    let load_backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend for loading: {}", e))
    })?;

    let swiglu_path = std::env::var("SWIGLU_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("SWIGLU_HSACO env var not set".to_string()))?;

    if !Path::new(&swiglu_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            swiglu_path
        )));
    }

    let swiglu_module = load_backend.load_module(&swiglu_path)?;
    let swiglu_kernel = load_backend.get_kernel_function(&swiglu_module, "swiglu_kernel")?;

    // Load RMSNorm kernel module and function
    let rms_norm_path = std::env::var("RMS_NORM_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("RMS_NORM_HSACO env var not set".to_string()))?;

    if !Path::new(&rms_norm_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            rms_norm_path
        )));
    }

    let rms_norm_module = load_backend.load_module(&rms_norm_path)?;
    let rms_norm_kernel = load_backend.get_kernel_function(&rms_norm_module, "rms_norm_kernel")?;

    // Initialize cache with modules and kernels only (no backend)
    *cache = Some(KernelCache {
        swiglu_module: Some(swiglu_module),
        swiglu_kernel: Some(swiglu_kernel),
        rms_norm_module: Some(rms_norm_module),
        rms_norm_kernel: Some(rms_norm_kernel),
    });

    Ok(&GLOBAL_CACHE)
}

/// Launch SwiGLU activation kernel
///
/// Computes: output[i] = gate[i] * swish(up[i])
/// where swish(x) = x * sigmoid(x)
///
/// # Arguments
/// * `backend` - HipBackend to use for kernel launch (ensures stream consistency)
/// * `gate` - Gate projection tensor [seq_len, intermediate_size]
/// * `up` - Up projection tensor [seq_len, intermediate_size]
/// * `output` - Output tensor [seq_len, intermediate_size]
/// * `seq_len` - Sequence length (rows)
/// * `intermediate_size` - Intermediate size (columns)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(String)` on failure
///
/// # IMPORTANT
/// The caller must synchronize on the SAME backend after calling this function
/// to ensure the kernel completes before using the output.
#[cfg(feature = "rocm")]
pub unsafe fn swiglu_gpu_kernel(
    backend: &HipBackend,
    gate: *const f32,
    up: *const f32,
    output: *mut f32,
    seq_len: u32,
    intermediate_size: u32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("SwiGLU cache lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "SwiGLU cache not initialized".to_string())?;

            let kernel = cache_ref
                .swiglu_kernel
                .as_ref()
                .ok_or_else(|| "swiglu_kernel not loaded".to_string())?;

            // Calculate grid/block dimensions
            let total_elements = (seq_len * intermediate_size) as usize;
            let block_dim = (256, 1, 1);
            let grid_dim = (
                ((total_elements as u32) + block_dim.0 - 1) / block_dim.0,
                1,
                1,
            );
            let shared_mem_bytes = 0;

            // Prepare kernel arguments - ALL args must be copied to mut locals first
            let mut gate_arg = gate as *mut f32;
            let mut up_arg = up as *mut f32;
            let mut output_arg = output;
            let mut seq_len_arg = seq_len;
            let mut intermediate_size_arg = intermediate_size;

            let args: &[*mut c_void] = &[
                &mut gate_arg as *mut _ as *mut c_void,
                &mut up_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut intermediate_size_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Failed to launch swiglu kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}

/// Launch RMSNorm kernel
///
/// Computes: output[row, j] = input[row, j] / sqrt(mean(input[row, :]^2) + eps) * weight[j]
/// where mean is computed over each row independently
///
/// # Arguments
/// * `backend` - HipBackend to use for kernel launch (ensures stream consistency)
/// * `input` - Input tensor [seq_len, hidden_size]
/// * `weight` - Weight tensor [hidden_size]
/// * `output` - Output tensor [seq_len, hidden_size]
/// * `seq_len` - Sequence length (rows)
/// * `hidden_size` - Hidden size (columns)
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(String)` on failure
///
/// # IMPORTANT
/// The caller must synchronize on the SAME backend after calling this function
/// to ensure the kernel completes before using the output.
#[cfg(feature = "rocm")]
pub unsafe fn rms_norm_gpu_kernel(
    backend: &HipBackend,
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) -> Result<(), String> {
    match get_or_init_cache() {
        Ok(cache_ref) => {
            let cache = cache_ref
                .lock()
                .map_err(|e| format!("RMSNorm cache lock poisoned: {}", e))?;
            let cache_ref = cache
                .as_ref()
                .ok_or_else(|| "RMSNorm cache not initialized".to_string())?;

            let kernel = cache_ref
                .rms_norm_kernel
                .as_ref()
                .ok_or_else(|| "rms_norm_kernel not loaded".to_string())?;

            // Calculate grid/block dimensions
            let block_dim = (256, 1, 1);
            let grid_dim = (
                (hidden_size + block_dim.0 - 1) / block_dim.0,
                seq_len,
                1,
            );
            let shared_mem_bytes = 0;

            // Prepare kernel arguments
            let mut input_arg = input as *mut f32;
            let mut weight_arg = weight as *mut f32;
            let mut output_arg = output;
            let mut seq_len_arg = seq_len;
            let mut hidden_size_arg = hidden_size;
            let mut eps_arg = eps;

            let args: &[*mut c_void] = &[
                &mut input_arg as *mut _ as *mut c_void,
                &mut weight_arg as *mut _ as *mut c_void,
                &mut output_arg as *mut _ as *mut c_void,
                &mut seq_len_arg as *mut _ as *mut c_void,
                &mut hidden_size_arg as *mut _ as *mut c_void,
                &mut eps_arg as *mut _ as *mut c_void,
            ];

            backend
                .launch_kernel_with_module_shared(
                    kernel,
                    grid_dim,
                    block_dim,
                    args,
                    shared_mem_bytes,
                )
                .map_err(|e| format!("Failed to launch rms_norm kernel: {:?}", e))?;

            Ok(())
        }
        Err(e) => Err(format!("Failed to get cache: {:?}", e)),
    }
}
