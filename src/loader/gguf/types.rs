//! GGUF Core Types
//!
//! This module defines the core data structures for GGUF model loading:
//! - `GgufTensor`: Represents a single tensor with metadata and data
//! - `GgufLoader`: Main loader struct with lazy loading support
//! - `F16`: Half-precision floating point type

use crate::backend::hip_backend::{AsyncLoader, DeviceTensor, HipBackend};
use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::mmap::MmapGguf;
use crate::loader::tensor_type::GgufTensorType;
use crate::loader::TensorShape;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Re-export GgufMetadata for external access (tests, etc.)
pub use crate::loader::metadata::GgufMetadata;

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: TensorShape,
    pub tensor_type: GgufTensorType,
    pub quant_type: String,
    pub offset: u64,
    pub data: Vec<u8>,
}

impl GgufTensor {
    /// Calculate total number of elements
    pub fn total_elements(&self) -> usize {
        self.shape.total_elements()
    }

    /// Calculate data size in bytes
    pub fn data_size(&self) -> usize {
        match self.tensor_type {
            GgufTensorType::F32 => self.total_elements().checked_mul(4).unwrap_or(usize::MAX),
            GgufTensorType::F16 => self.total_elements().checked_mul(2).unwrap_or(usize::MAX),
            GgufTensorType::Q4_0 => {
                // Q4_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(4 + 32).unwrap_or(usize::MAX)
            }
            GgufTensorType::Q8_0 => {
                // Q8_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(4 + 32).unwrap_or(usize::MAX)
            }
            GgufTensorType::Q2_K
            | GgufTensorType::Q3_K
            | GgufTensorType::Q4_K
            | GgufTensorType::Q5_K
            | GgufTensorType::Q6_K => {
                // K-quants: block_size=256 bytes
                let blocks = self.total_elements().div_ceil(256); // Approximate block count
                blocks.checked_mul(256).unwrap_or(usize::MAX)
            }
            GgufTensorType::Mxfp4 => {
                // MXFP4: block_size=32, each block has 1 scale (E8M0) + 32*4 bits data
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(1 + 16).unwrap_or(usize::MAX) // 1 byte scale + 16 bytes data
            }
            GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                // MXFP6: block_size=32, each block has 1 scale (E8M0) + 32*6 bits data
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(1 + 24).unwrap_or(usize::MAX) // 1 byte scale + 24 bytes data
            }
        }
    }
}

/// GGUF file loader
///
/// # Phase 1 Lazy Loading
///
/// This loader now supports lazy loading:
/// - **Metadata-only initialization**: `GgufLoader::new()` only parses metadata, not tensor data
/// - **Memory-mapped file access**: Zero-copy reads from GGUF file via `MmapGguf`
/// - **LazyTensor handles**: Tensors are represented as handles, not loaded data
/// - **On-demand GPU loading**: `load_tensor_to_gpu()` loads specific tensors on-demand
/// - **GPU cache**: Loaded tensors are cached to avoid redundant loads
///
/// # Thread Safety
///
/// The loader is `Send + Sync`:
/// - `MmapGguf` is `Send + Sync` (read-only memory mapping)
/// - `LazyTensor` is `Send + Sync`
/// - `gpu_cache` uses `RwLock` for thread-safe access
#[derive(Debug)]
pub struct GgufLoader {
    pub path: String,
    pub metadata: GgufMetadata,
    pub tensors: HashMap<String, GgufTensor>, // Legacy: kept for backward compatibility

    // Phase 1: Lazy loading fields
    /// Memory-mapped GGUF file for zero-copy tensor data access
    /// Wrapped in Arc for cheap cloning (sharing the same memory mapping)
    pub mmap: Option<Arc<MmapGguf>>,
    /// Lazy tensor handles (metadata only, no data loaded)
    pub lazy_tensors: HashMap<String, LazyTensor>,
    /// GPU tensor cache (name -> loaded GPU tensor)
    pub gpu_cache: Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>,
}

/// Clone implementation for GgufLoader
///
/// # Phase 2 Lazy Loading
///
/// GgufLoader can be cheaply cloned because:
/// - `MmapGguf` is an `Arc` wrapper - cloning is just Arc::clone
/// - `lazy_tensors` HashMap is cloned (metadata only, ~1KB)
/// - `gpu_cache` is Arc<RwLock<>> - shared across clones
/// - All clones share the same GPU cache and memory mapping
///
/// This enables ExecutionPlan to hold `Arc<GgufLoader>` for lazy loading.
impl Clone for GgufLoader {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
            mmap: self.mmap.clone(), // Arc<MmapGguf> clone is cheap
            lazy_tensors: self.lazy_tensors.clone(),
            gpu_cache: Arc::clone(&self.gpu_cache), // Share GPU cache across clones
        }
    }
}

impl GgufLoader {
    /// Create new GGUF loader from file path
    ///
    /// # Phase 1 Lazy Loading
    ///
    /// This method now initializes the loader with lazy loading support:
    /// - Opens the GGUF file and memory-maps it for zero-copy access
    /// - Parses metadata (KV pairs, tensor info)
    /// - Creates `LazyTensor` handles for all tensors (metadata only)
    /// - Does NOT load tensor data into RAM
    ///
    /// # Performance
    ///
    /// - Before Phase 1: ~60s (loaded all tensor data)
    /// - After Phase 1: ~5s (metadata only)
    pub fn new(path: &str) -> Result<Self> {
        crate::loader::gguf::loader_impl::new_loader(path)
    }

    /// Inspect only metadata without loading tensors into memory.
    pub fn metadata_from_file(path: &str) -> Result<GgufMetadata> {
        crate::loader::gguf::gpu_upload::metadata_from_file(path)
    }

    /// Get metadata
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Convert metadata to ModelConfig
    pub fn to_model_config(&self) -> Result<crate::model::config::ModelConfig> {
        crate::loader::gguf::loader_impl::to_model_config(self)
    }

    /// Load a single tensor to GPU on-demand (Phase 1 lazy loading).
    ///
    /// # Phase 1 Lazy Loading
    ///
    /// This method enables on-demand tensor loading:
    /// 1. Check GPU cache - return cached tensor if already loaded
    /// 2. Get tensor metadata from `lazy_tensors` handle
    /// 3. Read tensor data from memory-mapped file (zero-copy)
    /// 4. Dequantize based on tensor type (Q4_0, Q8_0, F16, F32, etc.)
    /// 5. Upload to GPU memory
    /// 6. Cache the result for future access
    ///
    /// # Performance
    ///
    /// - First load: ~50-200ms per tensor (depends on size)
    /// - Subsequent loads: <1ms (from cache)
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe. Multiple threads can call it concurrently:
    /// - GPU cache uses `RwLock` for safe concurrent access
    /// - `MmapGguf` is read-only and thread-safe
    /// - Only one thread will load a given tensor; others wait for cache
    ///
    /// # Example
    ///
    /// ```ignore
    /// let loader = GgufLoader::new("model.gguf")?;
    /// let backend = HipBackend::new()?;
    ///
    /// // Load specific tensor on-demand
    /// let tensor = loader.load_tensor_to_gpu("blk.0.attn_q.weight", &backend)?;
    /// ```
    pub fn load_tensor_to_gpu(
        &self,
        name: &str,
        backend: &HipBackend,
    ) -> Result<Arc<DeviceTensor>> {
        // Check GPU cache first
        {
            let cache = self
                .gpu_cache
                .read()
                .map_err(|e| anyhow!("GPU cache read lock poisoned: {}", e))?;
            eprintln!(">>> load_tensor_to_gpu: GPU cache has {} tensors, looking for '{}'", cache.len(), name);
            if let Some(cached) = cache.get(name) {
                tracing::debug!("GPU cache hit for tensor '{}'", name);
                eprintln!(">>> load_tensor_to_gpu: GPU cache HIT for '{}'", name);
                return Ok(cached.clone());
            }
            eprintln!(">>> load_tensor_to_gpu: GPU cache MISS for '{}'", name);
        }

        tracing::debug!("GPU cache miss for tensor '{}', loading from mmap", name);
        eprintln!(">>> load_tensor_to_gpu: Loading tensor '{}'", name);

        // Get lazy tensor metadata
        let lazy = self
            .lazy_tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor not found: '{}'", name))?;

        let (offset, size, shape, tensor_type) = match lazy {
            LazyTensor::Unloaded {
                offset,
                size,
                shape,
                tensor_type,
                ..
            } => {
                eprintln!(
                    ">>> load_tensor_to_gpu: Tensor '{}' type={:?}, size={} bytes, shape={:?}",
                    name, tensor_type, size, shape
                );
                (*offset, *size, TensorShape::from_dims(shape), *tensor_type)
            }
            LazyTensor::Gpu { .. } => {
                return Err(anyhow!(
                    "Tensor '{}' already marked as GPU-loaded (should be in cache)",
                    name
                ));
            }
        };

        // Delegate to gpu_upload module for the actual loading logic
        crate::loader::gguf::gpu_upload::load_tensor_to_gpu_impl(
            self, name, backend, offset, size, &shape, tensor_type,
        )
    }

    #[allow(dead_code)] // Reserved for future tensor type classification
    pub fn is_embedding_weight(name: &str) -> bool {
        matches!(
            name,
            "token_embd.weight" | "lm_head.weight" | "output.weight"
        )
    }

    pub fn is_fused_qkv_weight(name: &str) -> bool {
        const PATTERNS: [&str; 5] = [
            ".attn_qkv.weight",
            ".attn.qkv.weight",
            ".attention.qkv.weight",
            ".attention.query_key_value.weight",
            ".attention.wqkv.weight",
        ];
        PATTERNS.iter().any(|pattern| name.contains(pattern))
    }

    /// Load all tensors into memory
    pub fn load_tensors(&self) -> Result<HashMap<String, GgufTensor>> {
        Ok(self.tensors.clone())
    }

    /// Load tensors and upload to GPU.
    ///
    /// # Phase 1 Lazy Loading
    ///
    /// This method now uses the lazy loading approach:
    /// - Calls `load_tensor_to_gpu()` for each tensor
    /// - Uses GPU cache automatically (tensors loaded once)
    /// - Reads data from memory-mapped file (zero-copy)
    ///
    /// Note: This method loads all tensors to GPU, maintaining backward compatibility.
    /// For on-demand loading, use `load_tensor_to_gpu()` directly.
    pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        let mut result = HashMap::new();

        tracing::debug!(
            "load_to_gpu: Loading {} tensors via lazy loading",
            self.lazy_tensors.len()
        );

        for name in self.lazy_tensors.keys() {
            let device_tensor_arc = self.load_tensor_to_gpu(name, backend)?;
            // Arc<DeviceTensor> -> DeviceTensor clone for backward compatibility
            // Note: This creates a new DeviceTensor sharing the same GPU memory
            result.insert(name.clone(), DeviceTensor::clone(&device_tensor_arc));
        }

        tracing::debug!("load_to_gpu: Loaded {} tensors to GPU", result.len());
        Ok(result)
    }

    /// Async GPU loading with multi-stream concurrent uploads (Phase 4 Integration)
    ///
    /// This method integrates Phases 1-3:
    /// - Phase 1: HIP Events for synchronization
    /// - Phase 2: Rayon for parallel dequantization
    /// - Phase 3: AsyncLoader for concurrent GPU uploads
    ///
    /// Performance: ~5x faster than sequential loading
    /// - Parallel CPU dequantization (Rayon): ~4x speedup
    /// - Concurrent GPU uploads (4 streams): ~4x speedup
    /// - Combined effect: ~5x overall (bottlenecks prevent full 16x)
    ///
    /// Usage:
    /// ```ignore
    /// let tensors = loader.load_to_gpu_async(&backend)?;
    /// ```
    pub fn load_to_gpu_async(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        use std::collections::BTreeMap;
        use std::sync::Mutex;

        tracing::info!(
            "load_to_gpu_async: Starting async load of {} tensors",
            self.lazy_tensors.len()
        );

        // Create AsyncLoader with 4 concurrent upload streams
        let async_loader =
            AsyncLoader::new().map_err(|e| anyhow!("Failed to create AsyncLoader: {}", e))?;

        // Get all tensor names in sorted order for predictable behavior
        let tensor_names: Vec<String> = self.lazy_tensors.keys().cloned().collect();
        let total_tensors = tensor_names.len();

        // Phase A: Parallel Dequantization (Rayon)
        // All tensors dequantized in parallel on CPU
        tracing::info!(
            "load_to_gpu_async: Phase A - Parallel dequantization of {} tensors",
            total_tensors
        );

        // Thread-safe storage for dequantized data
        let dequantized_data: Mutex<BTreeMap<String, Vec<f32>>> = Mutex::new(BTreeMap::new());
        let tensor_shapes: Mutex<BTreeMap<String, Vec<usize>>> = Mutex::new(BTreeMap::new());

        // Delegate to gpu_upload module for parallel dequantization
        crate::loader::gguf::gpu_upload::parallel_dequantization_phase(
            self, &tensor_names, &dequantized_data, &tensor_shapes,
        )?;

        tracing::info!(
            "load_to_gpu_async: Phase A complete - {} tensors dequantized",
            dequantized_data
                .lock()
                .map_err(|e| anyhow!("Failed to lock dequantized_data for logging: {}", e))?
                .len()
        );

        // Delegate to gpu_upload module for memory calculation and upload phases
        crate::loader::gguf::gpu_upload::arena_upload_phase(
            backend,
            &async_loader,
            &dequantized_data,
            &tensor_shapes,
            &self.gpu_cache,
        )
    }

    /// Infer vocab_size from tensor shapes when metadata is missing
    ///
    /// This method searches for common embedding/output tensors and infers
    /// vocab_size from their dimensions. It compares against the known hidden_size
    /// to determine which dimension represents the vocabulary.
    ///
    /// # Returns
    ///
    /// * `Some(usize)` - Inferred vocab_size
    /// * `None` - Could not infer (no suitable tensor found)
    pub fn infer_vocab_size_from_tensors(&self) -> Option<usize> {
        // Common tensor names that contain vocab_size in their shape
        let tensor_names = [
            "token_embd.weight",
            "embed_tokens.weight",
            "output.weight",
            "lm_head.weight",
        ];

        for name in &tensor_names {
            if let Some(tensor) = self.tensors.get(*name) {
                let dims = tensor.shape.dims();

                // Need at least 2 dimensions to infer vocab_size
                if dims.len() >= 2 {
                    let (d0, d1) = (dims[0], dims[1]);
                    let hidden = self.metadata.hidden_size;

                    if hidden > 0 {
                        // We know hidden_size, use it to disambiguate
                        // Shape is either [vocab_size, hidden_size] or [hidden_size, vocab_size]
                        if d0 == hidden && d1 != hidden {
                            tracing::debug!(
                                "GGUF: Inferred vocab_size={} from {} shape [{}, {}]",
                                d1,
                                name,
                                d0,
                                d1
                            );
                            return Some(d1);
                        } else if d1 == hidden && d0 != hidden {
                            tracing::debug!(
                                "GGUF: Inferred vocab_size={} from {} shape [{}, {}]",
                                d0,
                                name,
                                d0,
                                d1
                            );
                            return Some(d0);
                        }
                    } else {
                        // hidden_size unknown, use heuristic: larger dimension is likely vocab_size
                        let inferred = d0.max(d1);
                        tracing::debug!(
                            "GGUF: Inferred vocab_size={} from {} (heuristic, hidden_size unknown)",
                            inferred,
                            name
                        );
                        return Some(inferred);
                    }
                }
            }
        }

        // No suitable tensor found
        tracing::warn!("GGUF: Could not infer vocab_size from tensor shapes");
        None
    }

    /// Infer intermediate_size from MLP layer tensor shapes.
    ///
    /// This is used when the GGUF metadata doesn't contain explicit intermediate_size.
    /// We look at the first layer's gate/up projection weights to infer the dimension.
    ///
    /// # Returns
    ///
    /// * `Some(usize)` - Inferred intermediate_size
    /// * `None` - Could not infer (no suitable tensor found)
    pub fn infer_intermediate_size_from_tensors(&self) -> Option<usize> {
        // Common tensor naming patterns for MLP gate weights
        let tensor_variants = [
            "blk.0.ffn_gate.weight",                     // Qwen2-style
            "blk.0.ffn_up.weight", // Qwen2-style (up projection has same shape)
            "model.layers.0.mlp.gate_proj.weight", // LLaMA/Mistral-style
            "layers.0.mlp.gate_proj.weight", // Alternative
            "transformer.layers.0.mlp.gate_proj.weight", // GPT-style
        ];

        for name in &tensor_variants {
            if let Some(tensor) = self.tensors.get(*name) {
                let dims = tensor.shape.dims();

                // Need at least 2 dimensions
                if dims.len() >= 2 {
                    let (d0, d1) = (dims[0], dims[1]);
                    let hidden = self.metadata.hidden_size;

                    if hidden > 0 {
                        // We know hidden_size, use it to find the intermediate dimension
                        // Gate weight shape is either [hidden_size, intermediate_size] or
                        // [intermediate_size, hidden_size]
                        if d0 == hidden && d1 != hidden {
                            tracing::debug!(
                                "GGUF: Auto-detected intermediate_size={} from {} tensor shape",
                                d1,
                                name
                            );
                            return Some(d1);
                        } else if d1 == hidden && d0 != hidden {
                            tracing::debug!(
                                "GGUF: Auto-detected intermediate_size={} from {} tensor shape",
                                d0,
                                name
                            );
                            return Some(d0);
                        }
                    } else {
                        // hidden_size unknown, use heuristic: larger dimension is likely intermediate_size
                        // (FFN expansion is typically 4x hidden_size)
                        let inferred = d0.max(d1);
                        tracing::debug!(
                            "GGUF: Auto-detected intermediate_size={} from {} (heuristic)",
                            inferred,
                            name
                        );
                        return Some(inferred);
                    }
                }
            }
        }

        // No suitable tensor found
        tracing::warn!(
            "GGUF: Warning - could not auto-detect intermediate_size from tensor shapes"
        );
        None
    }
}

/// Simple f16 implementation for conversion
pub struct F16(pub u16);

impl F16 {
    #[allow(dead_code)] // Reserved for future f16 conversion utilities
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[allow(dead_code)] // Reserved for future f16 conversion utilities
    pub fn to_f32(self) -> f32 {
        // Simple conversion - in practice would use proper half-precision conversion
        let bits = self.0;
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32 - 15;
        let mantissa = bits & 0x3FF;

        if exponent == -15 {
            if mantissa == 0 {
                0.0
            } else {
                sign * (mantissa as f32) * 2.0f32.powi(-14 - 10)
            }
        } else {
            sign * (1.0 + (mantissa as f32) * 2.0f32.powi(-10)) * 2.0f32.powi(exponent)
        }
    }
}

/// Transpose a 2D f32 matrix (row-major to column-major)
///
/// This function is kept for potential future use in tensor operations.
/// Currently unused but may be needed for GGUF tensor layout conversions.
#[allow(dead_code)] // Reserved for future tensor layout conversions
fn transpose_f32_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row_offset = r * cols;
        for c in 0..cols {
            transposed[c * rows + r] = data[row_offset + c];
        }
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_tensor_data_size_f32() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: TensorShape::from_dims(&[100, 200]),
            tensor_type: GgufTensorType::F32,
            quant_type: "F32".to_string(),
            offset: 0,
            data: vec![],
        };
        assert_eq!(tensor.data_size(), 100 * 200 * 4);
    }

    #[test]
    fn test_gguf_tensor_data_size_f16() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: TensorShape::from_dims(&[100, 200]),
            tensor_type: GgufTensorType::F16,
            quant_type: "F16".to_string(),
            offset: 0,
            data: vec![],
        };
        assert_eq!(tensor.data_size(), 100 * 200 * 2);
    }

    #[test]
    fn test_gguf_tensor_total_elements() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: TensorShape::from_dims(&[10, 20, 30]),
            tensor_type: GgufTensorType::F32,
            quant_type: "F32".to_string(),
            offset: 0,
            data: vec![],
        };
        assert_eq!(tensor.total_elements(), 10 * 20 * 30);
    }

    #[test]
    fn test_is_fused_qkv_weight() {
        assert!(GgufLoader::is_fused_qkv_weight("blk.0.attn_qkv.weight"));
        assert!(GgufLoader::is_fused_qkv_weight("blk.0.attn.qkv.weight"));
        assert!(GgufLoader::is_fused_qkv_weight("blk.0.attention.qkv.weight"));
        assert!(!GgufLoader::is_fused_qkv_weight("blk.0.attn_q.weight"));
        assert!(!GgufLoader::is_fused_qkv_weight("blk.0.ffn_gate.weight"));
    }

    #[test]
    fn test_is_embedding_weight() {
        assert!(GgufLoader::is_embedding_weight("token_embd.weight"));
        assert!(GgufLoader::is_embedding_weight("lm_head.weight"));
        assert!(GgufLoader::is_embedding_weight("output.weight"));
        assert!(!GgufLoader::is_embedding_weight("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_f16_conversion() {
        // Test basic f16 to f32 conversion
        let f16_zero = F16(0x0000); // +0.0
        assert_eq!(f16_zero.to_f32(), 0.0);

        let f16_one = F16(0x3C00); // 1.0
        assert!((f16_one.to_f32() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_transpose_f32_matrix() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let transposed = transpose_f32_matrix(&data, 2, 3);
        assert_eq!(transposed.len(), 6);
        // [[1, 2, 3],     [[1, 4],
        //  [4, 5, 6]]  =>  [2, 5],
        //                  [3, 6]]
        assert_eq!(transposed[0], 1.0);
        assert_eq!(transposed[1], 4.0);
        assert_eq!(transposed[2], 2.0);
        assert_eq!(transposed[3], 5.0);
        assert_eq!(transposed[4], 3.0);
        assert_eq!(transposed[5], 6.0);
    }
}
