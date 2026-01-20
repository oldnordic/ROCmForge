//! Kernel cache management for attention operations.

#![allow(non_snake_case)] // Kernel parameter names follow HIP conventions
#![allow(dead_code)] // Reserved for future kernel caching optimization

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::ffi::c_void;

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};

// Public sub-modules that can access private KernelCache fields
pub mod kernels_basic;
pub mod kernels_flash;

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

// MQA-specific kernel cache (separate for lazy loading)
#[cfg(feature = "rocm")]
pub static MQA_KERNEL_CACHE: Mutex<Option<MqaKernelCache>> = Mutex::new(None);

// Minimal attention kernel cache (for basic attention without all kernels)
#[cfg(feature = "rocm")]
pub static ATTENTION_KERNEL_CACHE: Mutex<Option<AttentionKernelCache>> = Mutex::new(None);

/// MQA-specific kernel cache (lazy loaded)
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub struct MqaKernelCache {
    pub backend: Arc<HipBackend>,
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    pub mqa_kv_replicate_module: HipModule,
    pub mqa_kv_replicate_kernel: HipKernel,
}

/// Minimal attention kernel cache (only QK^T, softmax, weighted matmul)
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub struct AttentionKernelCache {
    pub backend: Arc<HipBackend>,
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    pub qkt_matmul_module: HipModule,
    pub qkt_matmul_kernel: HipKernel,
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    pub softmax_module: HipModule,
    pub softmax_kernel: HipKernel,
    #[allow(dead_code)] // Module kept alive to keep HSACO loaded in memory
    pub weighted_matmul_module: HipModule,
    pub weighted_matmul_kernel: HipKernel,
}

/// Get or initialize the global kernel cache
pub fn get_or_init_cache() -> Result<&'static Mutex<Option<KernelCache>>, HipError> {
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

/// Get or initialize the MQA-specific kernel cache
///
/// This is a minimal cache that only loads the MQA KV replication kernel,
/// avoiding dependency on other kernels that may not be compiled.
#[cfg(feature = "rocm")]
pub fn get_mqa_kernel_and_backend() -> Result<(Arc<HipBackend>, *mut c_void), HipError> {
    // First check if already initialized
    {
        let cache = MQA_KERNEL_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("MQA_KERNEL_CACHE lock poisoned: {}", e)))?;
        if let Some(ref c) = *cache {
            return Ok((c.backend.clone(), c.mqa_kv_replicate_kernel.as_ptr()));
        }
    }

    // Need to initialize - drop the read lock first
    let mut cache = MQA_KERNEL_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("MQA_KERNEL_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if let Some(ref c) = *cache {
        return Ok((c.backend.clone(), c.mqa_kv_replicate_kernel.as_ptr()));
    }

    // Create backend
    let backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend: {}", e))
    })?;

    // Load MQA KV replication kernel
    let mqa_kv_replicate_path = std::env::var("MQA_KV_REPLICATE_HSACO")
        .ok()
        .ok_or_else(|| HipError::KernelLoadFailed("MQA_KV_REPLICATE_HSACO env var not set".to_string()))?;

    if !Path::new(&mqa_kv_replicate_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            mqa_kv_replicate_path
        )));
    }

    let mqa_kv_replicate_module = backend.load_module(&mqa_kv_replicate_path)?;
    let mqa_kv_replicate_kernel =
        backend.get_kernel_function(&mqa_kv_replicate_module, "mqa_kv_replicate_fused_kernel")?;

    let kernel_ptr = mqa_kv_replicate_kernel.as_ptr();
    let backend_arc = backend.clone();
    *cache = Some(MqaKernelCache {
        backend,
        mqa_kv_replicate_module,
        mqa_kv_replicate_kernel,
    });

    Ok((backend_arc, kernel_ptr))
}

/// Get or initialize the minimal attention kernel cache
///
/// This cache only loads the kernels needed for basic attention computation:
/// - QK^T matmul
/// - Softmax
/// - Weighted matmul
#[cfg(feature = "rocm")]
pub fn get_attention_kernels() -> Result<(Arc<HipBackend>, (*mut c_void, *mut c_void, *mut c_void)), HipError> {
    // First check if already initialized
    {
        let cache = ATTENTION_KERNEL_CACHE
            .lock()
            .map_err(|e| HipError::LockPoisoned(format!("ATTENTION_KERNEL_CACHE lock poisoned: {}", e)))?;
        if let Some(ref c) = *cache {
            return Ok((
                c.backend.clone(),
                (
                    c.qkt_matmul_kernel.as_ptr(),
                    c.softmax_kernel.as_ptr(),
                    c.weighted_matmul_kernel.as_ptr(),
                ),
            ));
        }
    }

    // Need to initialize
    let mut cache = ATTENTION_KERNEL_CACHE
        .lock()
        .map_err(|e| HipError::LockPoisoned(format!("ATTENTION_KERNEL_CACHE lock poisoned: {}", e)))?;

    // Double-check in case another thread initialized while we waited
    if let Some(ref c) = *cache {
        return Ok((
            c.backend.clone(),
            (
                c.qkt_matmul_kernel.as_ptr(),
                c.softmax_kernel.as_ptr(),
                c.weighted_matmul_kernel.as_ptr(),
            ),
        ));
    }

    // Create backend
    let backend = HipBackend::new().map_err(|e| {
        HipError::InitializationFailed(format!("Failed to create HipBackend: {}", e))
    })?;

    // Load QK^T matmul kernel
    let qkt_matmul_path = std::env::var("QKT_MATMUL_HSACO")
        .ok()
        .ok_or_else(|| HipError::KernelLoadFailed("QKT_MATMUL_HSACO env var not set".to_string()))?;

    if !Path::new(&qkt_matmul_path).exists() {
        return Err(HipError::KernelLoadFailed(format!(
            "HSACO not found: {}",
            qkt_matmul_path
        )));
    }

    let qkt_matmul_module = backend.load_module(&qkt_matmul_path)?;
    let qkt_matmul_kernel =
        backend.get_kernel_function(&qkt_matmul_module, "qkt_matmul_kernel")?;

    // Load softmax kernel
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

    // Load weighted matmul kernel
    let weighted_matmul_path = std::env::var("WEIGHTED_MATMUL_HSACO")
        .ok()
        .ok_or_else(|| {
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

    let backend_arc = backend.clone();
    let qkt_ptr = qkt_matmul_kernel.as_ptr();
    let softmax_ptr = softmax_kernel.as_ptr();
    let weighted_ptr = weighted_matmul_kernel.as_ptr();

    *cache = Some(AttentionKernelCache {
        backend,
        qkt_matmul_module,
        qkt_matmul_kernel,
        softmax_module,
        softmax_kernel,
        weighted_matmul_module,
        weighted_matmul_kernel,
    });

    Ok((backend_arc, (qkt_ptr, softmax_ptr, weighted_ptr)))
}
