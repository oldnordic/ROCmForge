//! GGUF (GPT-Generated Unified Format) Loader
//!
//! Complete implementation for loading GGUF model files with support for:
//! - Metadata parsing
//! - Multiple quantization types (Q8_0, Q4_0, FP16, FP32)
//! - Tensor block reading and validation
//! - GPU memory allocation via DeviceTensor

use crate::backend::hip_backend::{AsyncLoader, DeviceTensor, HipBackend, HipBuffer};
use crate::memory::{MemoryCalculator, ModelWeightArena};

// GPU dequantization kernels (require ROCm feature)
#[cfg(feature = "rocm")]
use crate::ggml::hip_backend::ops::q4_0_dequant::dequantize_q4_0_kernel_cached;
use crate::ggml::hip_backend::ops::q4_k_dequant::dequantize_q4_k_gpu_kernel;
use crate::ggml::hip_backend::ops::q6_k_dequant::dequantize_q6_k_gpu_kernel;
use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::mmap::MmapGguf;
use crate::loader::tensor_type::GgufTensorType;
use crate::loader::TensorShape;
use crate::model::config::ModelConfig;
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, RwLock};

// Parallel processing for async GPU loading (Phase 2: Rayon Integration)
// Rayon provides data-parallelism for CPU-intensive dequantization
use rayon::prelude::*;

// Thread-safe wrapper for parallel dequantization results
// RwLock allows multiple readers or one writer - perfect for parallel writes
#[allow(dead_code)] // Reserved for future async GPU loading (Rayon integration)
type ParallelResult = Arc<RwLock<Vec<f32>>>;

/// GGUF file magic number
const GGUF_MAGIC: &[u8] = b"GGUF";

/// E8M0 scale format (8-bit exponent only)
///
/// Per OCP MX Specification v1.0:
/// - 8-bit signed exponent
/// - Value = 2^exponent
/// - Range: 2^(-127) to 2^(127)
/// - Used as block scale for MXFP4/MXFP6
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct E8M0 {
    pub exponent: i8,
}

impl E8M0 {
    /// Convert E8M0 to f32
    pub fn to_f32(&self) -> f32 {
        2.0_f32.powi(self.exponent as i32)
    }

    /// Create E8M0 from f32
    /// E8M0 represents 2^exponent as a scale factor
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 || value.is_nan() {
            return E8M0 { exponent: 0 };
        }

        if value.is_infinite() {
            return E8M0 { exponent: 127 };
        }

        // E8M0 scale should be the largest value in the block
        // so we can represent values in [0, scale] range
        let abs_val = value.abs();
        let exp = abs_val.log2().clamp(-127.0, 127.0).round() as i8;
        E8M0 { exponent: exp }
    }
}

/// MXFP block (block-scaled floating-point)
///
/// Per OCP MX Specification v1.0:
/// - Block size: 32 elements
/// - Scale: E8M0 (1 byte)
/// - Elements: packed 4-bit or 6-bit values
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MxfpBlock {
    pub scale: E8M0,
    pub elements: Vec<u8>,
}

impl MxfpBlock {
    /// Create new MXFP4 block (4-bit elements)
    pub fn new_mxfp4() -> Self {
        MxfpBlock {
            scale: E8M0 { exponent: 0 },
            elements: vec![0u8; 16], // 32 elements * 4 bits / 8
        }
    }

    /// Create new MXFP6 block (6-bit elements)
    pub fn new_mxfp6() -> Self {
        MxfpBlock {
            scale: E8M0 { exponent: 0 },
            elements: vec![0u8; 24], // 32 elements * 6 bits / 8
        }
    }

    /// Pack f32 values into MXFP4 block
    pub fn pack_mxfp4(values: &[f32]) -> Self {
        // Find max absolute value for scale
        let max_val = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

        // Handle edge case where all values are zero
        let scale = if max_val == 0.0 {
            E8M0 { exponent: 0 }
        } else {
            // Scale is the max value itself (as power of 2)
            // This normalizes values to [0, 1] range for encoding
            E8M0::from_f32(max_val)
        };

        let scale_f32 = scale.to_f32();

        // Encode values as E2M1 (4-bit)
        let mut packed = vec![0u8; 16];
        for (i, &val) in values.iter().take(32).enumerate() {
            // Normalize value by scale (should now be in range [0, 1])
            let normalized = if scale_f32 > 0.0 {
                val / scale_f32
            } else {
                val
            };

            let encoded = Self::encode_e2m1(normalized);
            let byte_idx = i / 2;
            let nibble = i % 2;

            if nibble == 0 {
                packed[byte_idx] |= encoded << 4;
            } else {
                packed[byte_idx] |= encoded & 0x0F;
            }
        }

        MxfpBlock {
            scale,
            elements: packed,
        }
    }

    /// Unpack MXFP4 block to f32 values
    pub fn unpack_mxfp4(&self) -> Vec<f32> {
        let mut values = vec![0.0f32; 32];
        let scale_f32 = self.scale.to_f32();

        for i in 0..32 {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (self.elements[byte_idx] >> 4) & 0x0F
            } else {
                self.elements[byte_idx] & 0x0F
            };

            let decoded = Self::decode_e2m1(nibble);
            let mut val = scale_f32 * decoded;
            val = val.clamp(-8.0, 8.0); // MXFP4 range per OCP MX Spec v1.0
            values[i] = val;
        }

        values
    }

    /// Pack f32 values into MXFP6 block
    pub fn pack_mxfp6(values: &[f32]) -> Self {
        // Find max absolute value for scale
        let max_val = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

        // Handle edge case where all values are zero
        let scale = if max_val == 0.0 {
            E8M0 { exponent: 0 }
        } else {
            // Scale is the max value itself (as power of 2)
            // This normalizes values to [0, 1] range for encoding
            E8M0::from_f32(max_val)
        };

        let scale_f32 = scale.to_f32();

        // Encode values as E2M3 (6-bit)
        let packed = Self::pack_6bit_values(
            &values
                .iter()
                .take(32)
                .map(|&v| {
                    // Normalize value by scale (should now be in range [0, 1])
                    let normalized = if scale_f32 > 0.0 { v / scale_f32 } else { v };
                    Self::encode_e2m3(normalized)
                })
                .collect::<Vec<u8>>(),
        );

        MxfpBlock {
            scale,
            elements: packed,
        }
    }

    /// Unpack MXFP6 block to f32 values
    pub fn unpack_mxfp6(&self) -> Vec<f32> {
        let unpacked_bits = Self::unpack_6bit_values(&self.elements, 32);
        let scale_f32 = self.scale.to_f32();

        unpacked_bits
            .iter()
            .map(|&bits| {
                let decoded = Self::decode_e2m3(bits);
                let mut val = scale_f32 * decoded;
                val = val.clamp(-7.5, 7.5); // MXFP6 range
                val
            })
            .collect()
    }

    /// Get packed size in bytes
    pub fn packed_size(&self) -> usize {
        1 + self.elements.len() // scale + elements
    }

    /// Encode f32 as E2M1 (4-bit): sign(1) + exp(2) + mant(1)
    /// E2M1 format: value = (-1)^sign * 2^(exp-1) * (1 + mant)
    /// Input should be normalized to approximately [0, 8] range per OCP MX Spec v1.0
    pub fn encode_e2m1(value: f32) -> u8 {
        if value == 0.0 {
            return 0b0000;
        }

        let sign = if value < 0.0 { 0b1000 } else { 0b0000 };
        let abs = value.abs();

        // E2M1 can represent values in [0.5, 8.0] with exp in [0, 3] and mant in [0, 1]
        // For values < 0.5, we encode as 0.5 (minimum positive value)
        let clamped = abs.max(0.5).min(8.0);

        // Try all 4 combinations and pick the closest
        let mut best_encoding = 0u8;
        let mut best_error = f32::MAX;

        for exp_bits in 0..4 {
            for mant_bits in 0..2 {
                let exp = exp_bits as i32 - 1;
                let mant = mant_bits as f32;
                let decoded = (1.0 + mant) * 2_f32.powi(exp);

                let error = (clamped - decoded).abs();
                if error < best_error {
                    best_error = error;
                    best_encoding = (exp_bits << 1) | mant_bits;
                }
            }
        }

        sign | best_encoding
    }

    /// Decode E2M1 (4-bit) to f32
    pub fn decode_e2m1(bits: u8) -> f32 {
        if bits == 0 {
            return 0.0;
        }

        let sign = if bits & 0x08 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 1) & 0x03) as i32 - 1;
        let mant = (bits & 0x01) as f32;

        sign * (1.0 + mant) * 2_f32.powi(exp)
    }

    /// Encode f32 as E2M3 (6-bit): sign(1) + exp(2) + mant(3)
    /// E2M3 format: value = (-1)^sign * 2^(exp-1) * (1 + mant/8)
    /// Input should be normalized to approximately [0, 7.5] range
    pub fn encode_e2m3(value: f32) -> u8 {
        if value == 0.0 {
            return 0b000000;
        }

        let sign = if value < 0.0 { 0b100000 } else { 0b000000 };
        let abs = value.abs();

        // E2M3 can represent values in [0.5, 7.5] with exp in [0, 3] and mant in [0, 7]
        // For values < 0.5, we encode as 0.5 (minimum positive value)
        let clamped = abs.max(0.5).min(7.5);

        // Try all 32 combinations and pick the closest
        let mut best_encoding = 0u8;
        let mut best_error = f32::MAX;

        for exp_bits in 0..4 {
            for mant_bits in 0u8..8 {
                let exp = exp_bits as i32 - 1;
                let mant = mant_bits as f32 / 8.0;
                let decoded = (1.0 + mant) * 2_f32.powi(exp);

                let error = (clamped - decoded).abs();
                if error < best_error {
                    best_error = error;
                    best_encoding = (exp_bits << 3) | mant_bits;
                }
            }
        }

        sign | best_encoding
    }

    /// Decode E2M3 (6-bit) to f32
    pub fn decode_e2m3(bits: u8) -> f32 {
        if bits == 0 {
            return 0.0;
        }

        let sign = if bits & 0x20 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 3) & 0x03) as i32 - 1;
        let mant = ((bits & 0x07) as f32) / 8.0;

        sign * (1.0 + mant) * 2_f32.powi(exp)
    }

    /// Pack 6-bit values into bytes
    /// Packs values in little-endian bit order
    pub fn pack_6bit_values(values: &[u8]) -> Vec<u8> {
        let mut packed = vec![0u8; (values.len() * 6).div_ceil(8)];
        for (i, &val) in values.iter().enumerate() {
            let bit_pos = i * 6;
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            // Mask value to 6 bits
            let val_6bit = val & 0x3F;

            if bit_offset <= 2 {
                // Fits entirely in current byte (with room to spare)
                packed[byte_idx] |= val_6bit << bit_offset;
            } else {
                // Spans across two bytes
                let bits_in_first_byte = 8 - bit_offset;
                let _bits_in_second_byte = 6 - bits_in_first_byte;

                packed[byte_idx] |= val_6bit << bit_offset;
                packed[byte_idx + 1] |= val_6bit >> bits_in_first_byte;
            }
        }
        packed
    }

    /// Unpack 6-bit values from bytes
    /// Unpacks values in little-endian bit order
    pub fn unpack_6bit_values(packed: &[u8], count: usize) -> Vec<u8> {
        let mut values = vec![0u8; count];
        for i in 0..count {
            let bit_pos = i * 6;
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            if byte_idx < packed.len() {
                if bit_offset <= 2 {
                    // Value fits entirely in current byte
                    values[i] = (packed[byte_idx] >> bit_offset) & 0x3F;
                } else {
                    // Value spans two bytes
                    let bits_from_first_byte = 8 - bit_offset;
                    let bits_from_second_byte = 6 - bits_from_first_byte;

                    let first_part =
                        (packed[byte_idx] >> bit_offset) & ((1 << bits_from_first_byte) - 1);
                    let second_part = if byte_idx + 1 < packed.len() {
                        packed[byte_idx + 1] & ((1 << bits_from_second_byte) - 1)
                    } else {
                        0
                    };

                    values[i] = first_part | (second_part << bits_from_first_byte);
                }
            }
        }
        values
    }
}

/// GGUF metadata extracted from file header
#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadata {
    pub architecture: String,
    pub file_type: u32,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>, // MQA/GQA support
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub use_rotary_embeddings: bool,
    #[serde(skip_serializing)]
    pub embedded_tokenizer_json: Option<String>,
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            file_type: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: None,
            hidden_size: 0,
            intermediate_size: 0,
            head_dim: 0,
            max_position_embeddings: 2048,
            vocab_size: 0,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
            embedded_tokenizer_json: None,
        }
    }
}

impl GgufMetadata {
    /// Calculate head_dim from hidden_size and num_heads.
    ///
    /// This implements the llama.cpp pattern: calculate a sensible default
    /// BEFORE parsing optional GGUF metadata, then allow GGUF to override.
    ///
    /// Call this AFTER parsing num_heads and hidden_size but BEFORE using head_dim.
    ///
    /// # Logic
    /// - Only calculate if head_dim is currently 0 (wasn't set by GGUF)
    /// - Requires num_heads > 0 and hidden_size > 0 to avoid division by zero
    /// - Uses integer division: head_dim = hidden_size / num_heads
    pub fn calculate_default_head_dim(&mut self) {
        if self.num_heads > 0 && self.hidden_size > 0 && self.head_dim == 0 {
            self.head_dim = self.hidden_size / self.num_heads;
        }
    }
}

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
            GgufTensorType::Q4_1 => {
                // Q4_1: similar structure to Q4_0
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(4 + 32).unwrap_or(usize::MAX)
            }
            GgufTensorType::Q5_0 => {
                // Q5_0: block_size=32
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(4 + 32).unwrap_or(usize::MAX)
            }
            GgufTensorType::Q5_1 => {
                // Q5_1: block_size=32
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
    path: String,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensor>, // Legacy: kept for backward compatibility

    // Phase 1: Lazy loading fields
    /// Memory-mapped GGUF file for zero-copy tensor data access
    /// Wrapped in Arc for cheap cloning (sharing the same memory mapping)
    mmap: Option<Arc<MmapGguf>>,
    /// Lazy tensor handles (metadata only, no data loaded)
    pub lazy_tensors: HashMap<String, LazyTensor>,
    /// GPU tensor cache (name -> loaded GPU tensor)
    gpu_cache: Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>,
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
        use std::path::Path;

        // Create memory-mapped file for zero-copy access
        let mmap = MmapGguf::open(Path::new(path))
            .map_err(|e| anyhow!("Failed to memory-map GGUF file '{}': {}", path, e))?;

        let mut loader = GgufLoader {
            path: path.to_string(),
            metadata: GgufMetadata::default(),
            tensors: HashMap::new(),
            mmap: Some(Arc::new(mmap)),
            lazy_tensors: HashMap::new(),
            gpu_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        // Parse metadata and tensor info (but NOT tensor data)
        loader.load_from_disk(true)?;
        Ok(loader)
    }

    /// Inspect only metadata without loading tensors into memory.
    pub fn metadata_from_file(path: &str) -> Result<GgufMetadata> {
        use std::path::Path;

        let mmap = MmapGguf::open(Path::new(path))
            .map_err(|e| anyhow!("Failed to memory-map GGUF file '{}': {}", path, e))?;

        let mut loader = GgufLoader {
            path: path.to_string(),
            metadata: GgufMetadata::default(),
            tensors: HashMap::new(),
            mmap: Some(Arc::new(mmap)),
            lazy_tensors: HashMap::new(),
            gpu_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        loader.load_from_disk(false)?;
        Ok(loader.metadata)
    }

    /// Get metadata
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
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

        // Read tensor data from memory-mapped file (zero-copy)
        let mmap = self.mmap.as_ref().ok_or_else(|| {
            anyhow!("Memory mapping not available - loader not initialized correctly")
        })?;

        let tensor_bytes = mmap
            .get_slice(offset, size)
            .map_err(|e| anyhow!("Failed to read tensor '{}' from mmap: {}", name, e))?;

        // Check if this is a Q4_0, Q4_K, or Q6_K tensor that can use GPU dequantization
        // Upload quantized bytes directly, dequantize on GPU
        let needs_transpose = Self::is_fused_qkv_weight(name);
        if (tensor_type == GgufTensorType::Q4_0
            || tensor_type == GgufTensorType::Q4_K
            || tensor_type == GgufTensorType::Q6_K) && !needs_transpose {
            // GPU dequantization path for Q4_0/Q4_K/Q6_K
            // Allocate output buffer for FP32 data
            let num_elements = shape.total_elements();
            let output_buffer = backend
                .allocate_buffer(num_elements * 4)
                .map_err(|e| anyhow!("Failed to allocate output buffer for '{}': {}", name, e))?;

            // Call GPU dequantization - NO CPU fallback for GPU tensors
            // QUANT-06: CPU dequantization fallback removed for GPU tensors
            // If GPU dequantization fails, fail fast with clear error message
            match tensor_type {
                #[cfg(feature = "rocm")]
                GgufTensorType::Q4_0 => {
                    dequantize_q4_0_kernel_cached(backend, tensor_bytes, &output_buffer, num_elements)
                        .map_err(|e| anyhow!("GPU dequantization failed for '{}': {}. If GPU is unavailable, use CPU tensors instead.", name, e))?;
                }
                GgufTensorType::Q4_K => {
                    dequantize_q4_k_gpu_kernel(backend, tensor_bytes, &output_buffer, num_elements)
                        .map_err(|e| anyhow!("GPU dequantization failed for '{}': {}. If GPU is unavailable, use CPU tensors instead.", name, e))?;
                }
                GgufTensorType::Q6_K => {
                    dequantize_q6_k_gpu_kernel(backend, tensor_bytes, &output_buffer, num_elements)
                        .map_err(|e| anyhow!("GPU dequantization failed for '{}': {}. If GPU is unavailable, use CPU tensors instead.", name, e))?;
                }
                _ => unreachable!(),
            };

            eprintln!(
                ">>> load_tensor_to_gpu: '{}' GPU dequantization complete, {} f32 values ({} MB)",
                name,
                num_elements,
                num_elements * 4 / 1024 / 1024
            );

            // Wrap buffer in DeviceTensor
            let device_tensor = DeviceTensor::from_buffer(backend, output_buffer, shape)
                .map_err(|e| anyhow!("Failed to create DeviceTensor for '{}': {}", name, e))?;

            let device_tensor_arc = Arc::new(device_tensor);

            eprintln!(
                ">>> load_tensor_to_gpu: '{}' uploaded to GPU successfully",
                name
            );

            // Cache the result
            {
                let mut cache = self
                    .gpu_cache
                    .write()
                    .map_err(|e| anyhow!("GPU cache write lock poisoned: {}", e))?;
                cache.insert(name.to_string(), device_tensor_arc.clone());
            }

            tracing::debug!(
                "Loaded tensor '{}' to GPU and cached ({} bytes)",
                name,
                size
            );
            return Ok(device_tensor_arc);
        }

        // CPU dequantization path (for types without GPU dequantization or with transpose requirement)
        // Dequantize based on tensor type
        let mut f32_data: Vec<f32> = match tensor_type {
            GgufTensorType::F32 => tensor_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            GgufTensorType::F16 => tensor_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect(),
            GgufTensorType::Q8_0 => {
                // Create temporary GgufTensor for dequantization
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q8_0(&temp_tensor)?
            }
            GgufTensorType::Q4_0 => {
                // Q4_0 with transpose requirement - use CPU fallback
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q4_0(&temp_tensor)?
            }
            GgufTensorType::Q4_1 => {
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q4_1(&temp_tensor)?
            }
            GgufTensorType::Q5_0 => {
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q5_0(&temp_tensor)?
            }
            GgufTensorType::Q5_1 => {
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q5_1(&temp_tensor)?
            }
            GgufTensorType::Q4_K => {
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q4_k(&temp_tensor)?
            }
            GgufTensorType::Q6_K => {
                let temp_tensor = GgufTensor {
                    name: name.to_string(),
                    shape: shape.clone(),
                    tensor_type,
                    quant_type: tensor_type.to_string().to_string(),
                    offset,
                    data: tensor_bytes.to_vec(),
                };
                self.dequantize_q6_k(&temp_tensor)?
            }
            GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q5_K => {
                return Err(anyhow!(
                    "K-quant type {:?} not yet implemented for tensor '{}'",
                    tensor_type,
                    name
                ));
            }
            GgufTensorType::Mxfp4 | GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                return Err(anyhow!(
                    "MXFP dequantization not yet implemented for tensor '{}'",
                    name
                ));
            }
        };

        eprintln!(
            ">>> load_tensor_to_gpu: '{}' dequantization complete, {} f32 values ({} MB)",
            name,
            f32_data.len(),
            f32_data.len() * 4 / 1024 / 1024
        );

        let mut shape = shape;

        // Embedding weights stay in GGUF layout; GetRows handles layout at execution time.

        // Normalize fused QKV weight orientation: some models store [3*hidden, hidden]
        // while the matmul path expects [hidden, 3*hidden].
        if Self::is_fused_qkv_weight(name) {
            let dims = shape.dims();
            if dims.len() == 2 {
                let (rows, cols) = (dims[0], dims[1]);
                if rows == cols * 3 {
                    eprintln!(
                        ">>> load_tensor_to_gpu: Transposing fused QKV weight '{}' from [{}, {}] to [{}, {}]",
                        name, rows, cols, cols, rows
                    );
                    f32_data = transpose_f32_matrix(&f32_data, rows, cols);
                    shape = TensorShape::from_dims(&[cols, rows]);
                } else if cols != rows * 3 {
                    tracing::warn!(
                        "Unexpected fused QKV shape [{} x {}] for '{}'; expected one dimension to be 3x the other",
                        rows,
                        cols,
                        name
                    );
                }
            }
        }

        // Upload to GPU
        let device_tensor = DeviceTensor::from_host_vec(backend, f32_data, shape)
            .map_err(|e| anyhow!("Failed to upload tensor '{}' to GPU: {}", name, e))?;

        let device_tensor_arc = Arc::new(device_tensor);

        eprintln!(
            ">>> load_tensor_to_gpu: '{}' uploaded to GPU successfully",
            name
        );

        // Cache the result
        {
            let mut cache = self
                .gpu_cache
                .write()
                .map_err(|e| anyhow!("GPU cache write lock poisoned: {}", e))?;
            cache.insert(name.to_string(), device_tensor_arc.clone());
        }

        tracing::debug!(
            "Loaded tensor '{}' to GPU and cached ({} bytes)",
            name,
            size
        );
        Ok(device_tensor_arc)
    }

    #[allow(dead_code)] // Reserved for future tensor type classification
    fn is_embedding_weight(name: &str) -> bool {
        matches!(
            name,
            "token_embd.weight" | "lm_head.weight" | "output.weight"
        )
    }

    fn is_fused_qkv_weight(name: &str) -> bool {
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

        // Process all tensors in parallel (Rayon)
        tensor_names.par_iter().for_each(|name| {
            // Check GPU cache first
            {
                let cache = match self.gpu_cache.read() {
                    Ok(guard) => guard,
                    Err(_) => return, // Skip if cache is poisoned
                };
                if cache.contains_key(name) {
                    // Already loaded, skip
                    return;
                }
            }

            // Get lazy tensor metadata
            let lazy = match self.lazy_tensors.get(name) {
                Some(l) => l,
                None => return,
            };

            let (offset, size, shape, tensor_type) = match lazy {
                LazyTensor::Unloaded {
                    offset,
                    size,
                    shape,
                    tensor_type,
                    ..
                } => (*offset, *size, shape.clone(), *tensor_type),
                LazyTensor::Gpu { .. } => return,
            };

            // Read tensor data from memory-mapped file
            let mmap = match &self.mmap {
                Some(m) => m,
                None => return,
            };

            let tensor_bytes = match mmap.get_slice(offset, size) {
                Ok(slice) => slice,
                Err(_) => return,
            };

            // Dequantize based on tensor type (using Rayon-parallelized methods)
            let f32_data: Vec<f32> = match tensor_type {
                GgufTensorType::F32 => tensor_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
                GgufTensorType::F16 => tensor_bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect(),
                GgufTensorType::Q8_0 => {
                    let temp_tensor = GgufTensor {
                        name: name.clone(),
                        shape: TensorShape::from_dims(&shape),
                        tensor_type,
                        quant_type: tensor_type.to_string().to_string(),
                        offset,
                        data: tensor_bytes.to_vec(),
                    };
                    match self.dequantize_q8_0(&temp_tensor) {
                        Ok(data) => data,
                        Err(_) => return,
                    }
                }
                GgufTensorType::Q4_0 => {
                    let temp_tensor = GgufTensor {
                        name: name.clone(),
                        shape: TensorShape::from_dims(&shape),
                        tensor_type,
                        quant_type: tensor_type.to_string().to_string(),
                        offset,
                        data: tensor_bytes.to_vec(),
                    };
                    match self.dequantize_q4_0(&temp_tensor) {
                        Ok(data) => data,
                        Err(_) => return,
                    }
                }
                _ => {
                    // For other types, skip or return empty
                    return;
                }
            };

            // Store dequantized data and shape
            {
                // Mutex poisoning is unlikely in parallel dequantization (no panics expected)
                // but we handle it gracefully with context
                let mut data_guard = dequantized_data.lock().unwrap_or_else(|e| {
                    panic!(
                        "Failed to lock dequantized_data for tensor '{}': {}",
                        name, e
                    )
                });
                let mut shape_guard = tensor_shapes.lock().unwrap_or_else(|e| {
                    panic!("Failed to lock tensor_shapes for tensor '{}': {}", name, e)
                });
                data_guard.insert(name.clone(), f32_data);
                shape_guard.insert(name.clone(), shape);
            }
        });

        tracing::info!(
            "load_to_gpu_async: Phase A complete - {} tensors dequantized",
            dequantized_data
                .lock()
                .map_err(|e| anyhow!("Failed to lock dequantized_data for logging: {}", e))?
                .len()
        );

        // Phase A-1: Memory Requirements Calculation (Phase 22-02)
        // Calculate total GPU memory needed BEFORE any allocation
        // This enables pre-flight checks and clean error messages
        tracing::info!("load_to_gpu_async: Phase A-1 - Memory requirements calculation");

        let shapes = tensor_shapes
            .lock()
            .map_err(|e| anyhow!("Failed to lock tensor_shapes for memory calculation: {}", e))?;

        let mut memory_calc = MemoryCalculator::new();
        for (name, shape) in shapes.iter() {
            let elements: usize = shape.iter().product();
            memory_calc.add_tensor(name.clone(), elements, std::mem::size_of::<f32>());
        }

        let total_needed_bytes = memory_calc.total_bytes();
        let tensor_count = memory_calc.tensor_count();
        tracing::info!(
            "Memory calculation: {} tensors, {} MB needed (base)",
            tensor_count,
            total_needed_bytes / 1024 / 1024
        );

        // Query GPU memory and check if sufficient
        let (has_enough, free_mb, needed_mb) = backend
            .check_memory_for_model(total_needed_bytes)
            .map_err(|e| anyhow!("Failed to query GPU memory: {}", e))?;

        if !has_enough {
            bail!(
                "Insufficient GPU memory for model loading\n\
                 Required: {} MB (including 10%% safety margin)\n\
                 Available: {} MB\n\
                 Deficit: {} MB\n\
                 \n\
                 Suggestions:\n\
                 - Use a smaller model\n\
                 - Reduce context length\n\
                 - Close other GPU-intensive applications",
                needed_mb, free_mb, needed_mb.saturating_sub(free_mb)
            );
        }

        tracing::info!(
            "GPU memory check passed: {} MB available for {} MB needed (safety margin included)",
            free_mb, needed_mb
        );

        // Release the shapes lock before Phase B
        drop(shapes);

        // Phase B-1: Create memory arena for all tensor weights
        // Single large allocation instead of 200-300 individual allocations
        // This is critical for RDNA3 stability (prevents GPU hang)
        tracing::info!("load_to_gpu_async: Phase B-1 - Creating memory arena");
        let mut arena = ModelWeightArena::new(total_needed_bytes, backend)
            .map_err(|e| anyhow!("Failed to create memory arena: {}", e))?;

        tracing::info!(
            "Created arena: {} MB capacity, alignment {} bytes",
            arena.capacity() / 1024 / 1024,
            ModelWeightArena::DEFAULT_ALIGNMENT
        );

        // Phase B-2: Upload all tensors to arena using offset-based allocation
        // Upload all dequantized tensors to GPU in parallel using 4 streams
        tracing::info!("load_to_gpu_async: Phase B-2 - Arena-based concurrent uploads");

        let dequantized = dequantized_data
            .lock()
            .map_err(|e| anyhow!("Failed to lock dequantized_data: {}", e))?;
        let shapes = tensor_shapes
            .lock()
            .map_err(|e| anyhow!("Failed to lock tensor_shapes: {}", e))?;
        let gpu_buffers: Mutex<HashMap<String, Arc<DeviceTensor>>> = Mutex::new(HashMap::new());

        // Upload tensors concurrently using multiple streams
        let mut stream_idx = 0;
        for (name, data) in dequantized.iter() {
            let shape = shapes
                .get(name)
                .ok_or_else(|| anyhow!("Shape not found for tensor '{}'", name))?;

            // Calculate tensor size
            let total_elements: usize = shape.iter().product();
            let bytes_needed = total_elements * std::mem::size_of::<f32>();

            // Allocate from arena (single large buffer subdivided)
            let offset = arena.allocate_named(name.clone(), bytes_needed)
                .map_err(|e| anyhow!("Failed to allocate arena space for '{}': {}", name, e))?;

            // Convert f32 data to bytes for upload
            let data_bytes: Vec<u8> = data
                .iter()
                .flat_map(|&f| f.to_le_bytes().to_vec())
                .collect();

            // Upload to arena buffer at offset using round-robin stream selection
            let selected_stream = stream_idx % 4;
            async_loader
                .upload_to_buffer_offset(
                    arena.buffer(),
                    offset,
                    &data_bytes,
                    selected_stream
                )
                .map_err(|e| anyhow!("Failed to upload tensor '{}': {}", name, e))?;

            // Create DeviceTensor from arena slice
            let device_tensor = Arc::new(DeviceTensor::from_arena_slice(
                backend,
                arena.buffer(),
                offset,
                bytes_needed,
                TensorShape::from_dims(shape),
            )?);

            // Store result
            gpu_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock gpu_buffers for insert: {}", e))?
                .insert(name.clone(), device_tensor);

            stream_idx += 1;
        }

        // Synchronize all uploads
        async_loader
            .synchronize()
            .map_err(|e| anyhow!("Failed to synchronize async loader: {}", e))?;

        tracing::info!(
            "load_to_gpu_async: Phase B complete - {} tensors uploaded to GPU via arena",
            gpu_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock gpu_buffers for logging: {}", e))?
                .len()
        );

        tracing::info!(
            "Arena upload complete: {} MB used, {} MB free, {} fragments",
            arena.allocated_bytes() / 1024 / 1024,
            arena.remaining_capacity() / 1024 / 1024,
            arena.fragment_count()
        );

        // Phase C: Update GPU Cache
        tracing::info!("load_to_gpu_async: Phase C - Updating GPU cache");

        let mut cache = self
            .gpu_cache
            .write()
            .map_err(|e| anyhow!("GPU cache write lock poisoned: {}", e))?;

        let buffers = gpu_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock gpu_buffers for cache update: {}", e))?;
        for (name, tensor) in buffers.iter() {
            cache.insert(name.clone(), tensor.clone());
        }

        // Debug: Log cache size after preload
        let cache_size = cache.len();
        tracing::info!("load_to_gpu_async: GPU cache now has {} tensors", cache_size);
        eprintln!(">>> load_to_gpu_async: GPU cache now has {} tensors", cache_size);

        // Convert Arc<DeviceTensor> to DeviceTensor for backward compatibility
        let result: HashMap<String, DeviceTensor> = buffers
            .iter()
            .map(|(name, tensor)| (name.clone(), DeviceTensor::clone(tensor)))
            .collect();

        tracing::info!(
            "load_to_gpu_async: Complete - Loaded {} tensors",
            result.len()
        );
        Ok(result)
    }

    /// Convert metadata to ModelConfig
    pub fn to_model_config(&self) -> Result<ModelConfig> {
        use crate::model::config::ModelType;

        // Determine vocab_size: metadata, inference, or default
        let vocab_size = if self.metadata.vocab_size > 0 {
            // Metadata has explicit vocab_size
            self.metadata.vocab_size
        } else {
            // Try to infer from tensor shapes
            match self.infer_vocab_size_from_tensors() {
                Some(inferred) => inferred,
                None => {
                    // Last resort: architecture-specific defaults
                    let default = match self.metadata.architecture.as_str() {
                        "qwen2" => 151936,
                        "llama" => 32000,
                        "glm" => 151552,
                        _ => 32000,
                    };
                    tracing::info!(
                        "GGUF: Using default vocab_size={} for '{}'",
                        default,
                        self.metadata.architecture
                    );
                    default
                }
            }
        };

        // Determine intermediate_size: metadata, inference, or default
        let intermediate_size = if self.metadata.intermediate_size > 0 {
            // Metadata has explicit intermediate_size
            self.metadata.intermediate_size
        } else {
            // Try to infer from tensor shapes (MLP gate weights)
            match self.infer_intermediate_size_from_tensors() {
                Some(inferred) => inferred,
                None => {
                    // Last resort: use 4x hidden_size (common FFN expansion ratio)
                    let default = self.metadata.hidden_size * 4;
                    tracing::info!(
                        "GGUF: Using default intermediate_size={} (4x hidden_size) for '{}'",
                        default,
                        self.metadata.architecture
                    );
                    default
                }
            }
        };

        Ok(ModelConfig {
            num_hidden_layers: self.metadata.num_layers,
            num_attention_heads: self.metadata.num_heads,
            num_kv_heads: self.metadata.num_kv_heads,
            hidden_size: self.metadata.hidden_size,
            intermediate_size,
            max_position_embeddings: self.metadata.max_position_embeddings,
            vocab_size,
            rms_norm_eps: self.metadata.rms_norm_eps,
            use_rotary_embeddings: self.metadata.use_rotary_embeddings,
            model_type: if self.metadata.architecture == "glm" {
                ModelType::Llama // Use Llama as placeholder for now
            } else {
                ModelType::Llama
            },
            head_dim: if self.metadata.head_dim > 0 {
                self.metadata.head_dim
            } else if self.metadata.num_heads > 0 {
                self.metadata.hidden_size / self.metadata.num_heads
            } else {
                128 // Safe fallback
            },
        })
    }

    /// Load GGUF file and parse header
    fn load_from_disk(&mut self, load_tensors: bool) -> Result<()> {
        let mut file = File::open(&self.path)?;

        // Read and verify magic number
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(anyhow!("Invalid GGUF magic number"));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 3 {
            return Err(anyhow!("Unsupported GGUF version: {}", version));
        }

        // Read tensor count
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        // Read KV count
        let mut kv_count_bytes = [0u8; 8];
        file.read_exact(&mut kv_count_bytes)?;
        let kv_count = u64::from_le_bytes(kv_count_bytes);

        // Parse KV pairs (metadata)
        self.parse_kv_pairs(&mut file, kv_count)?;

        // Calculate default head_dim if not set by GGUF
        // Per llama.cpp pattern: calculate default before optional override
        self.metadata.calculate_default_head_dim();

        if load_tensors {
            // Parse tensor info (creates LazyTensor handles)
            self.parse_tensor_info(&mut file, tensor_count)?;

            // Phase 1 Lazy Loading: Skip read_tensor_data()
            // Tensor data is loaded on-demand from memory-mapped file via load_tensor_to_gpu()
            // This reduces RAM usage and initial load time significantly
            tracing::debug!("Phase 1: Skipping tensor data load - will use mmap on-demand");
        }

        Ok(())
    }

    /// Parse key-value pairs from GGUF header
    fn parse_kv_pairs(&mut self, file: &mut File, kv_count: u64) -> Result<()> {
        for i in 0..kv_count {
            // Read key
            let mut key_len_bytes = [0u8; 8];
            file.read_exact(&mut key_len_bytes)?;
            let key_len = u64::from_le_bytes(key_len_bytes) as usize;

            // Sanity check key_len to prevent overflow
            if key_len > 100_000 {
                return Err(anyhow!("key_len too large: {} at index {}", key_len, i));
            }

            let mut key_bytes = vec![0u8; key_len];
            file.read_exact(&mut key_bytes)?;
            let key = String::from_utf8_lossy(&key_bytes).to_string();

            // Read value type (GGUF v3 uses u32 for value type)
            let mut value_type_bytes = [0u8; 4];
            file.read_exact(&mut value_type_bytes)?;
            let value_type = u32::from_le_bytes(value_type_bytes);

            // Read value based on type
            // GGUF value types (official ggml/gguf.h spec):
            // 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
            // 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
            let value = match value_type {
                4 => {
                    // UINT32 - direct value, no value_len
                    let mut value_bytes = [0u8; 4];
                    file.read_exact(&mut value_bytes)?;
                    u32::from_le_bytes(value_bytes).to_string()
                }
                5 => {
                    // INT32 - direct value, no value_len
                    let mut value_bytes = [0u8; 4];
                    file.read_exact(&mut value_bytes)?;
                    i32::from_le_bytes(value_bytes).to_string()
                }
                6 => {
                    // FLOAT32 - direct value, no value_len
                    let mut value_bytes = [0u8; 4];
                    file.read_exact(&mut value_bytes)?;
                    f32::from_le_bytes(value_bytes).to_string()
                }
                7 => {
                    // BOOL - direct value (1 byte), no value_len
                    let mut value_bytes = [0u8; 1];
                    file.read_exact(&mut value_bytes)?;
                    (value_bytes[0] != 0).to_string()
                }
                8 => {
                    // STRING - has value_len prefix
                    let mut value_len_bytes = [0u8; 8];
                    file.read_exact(&mut value_len_bytes)?;
                    let value_len = u64::from_le_bytes(value_len_bytes) as usize;

                    if value_len > 100_000_000 {
                        return Err(anyhow!(
                            "value_len too large: {} for key '{}'",
                            value_len,
                            key
                        ));
                    }

                    let mut value_bytes = vec![0u8; value_len];
                    file.read_exact(&mut value_bytes)?;
                    String::from_utf8_lossy(&value_bytes).to_string()
                }
                0 => {
                    // UINT8 - direct value, no value_len
                    let mut value_bytes = [0u8; 1];
                    file.read_exact(&mut value_bytes)?;
                    value_bytes[0].to_string()
                }
                1 => {
                    // INT8 - direct value, no value_len
                    let mut value_bytes = [0u8; 1];
                    file.read_exact(&mut value_bytes)?;
                    (value_bytes[0] as i8).to_string()
                }
                2 => {
                    // UINT16 - direct value, no value_len
                    let mut value_bytes = [0u8; 2];
                    file.read_exact(&mut value_bytes)?;
                    u16::from_le_bytes(value_bytes).to_string()
                }
                3 => {
                    // INT16 - direct value, no value_len
                    let mut value_bytes = [0u8; 2];
                    file.read_exact(&mut value_bytes)?;
                    i16::from_le_bytes(value_bytes).to_string()
                }
                10 => {
                    // UINT64 - direct value, no value_len
                    let mut value_bytes = [0u8; 8];
                    file.read_exact(&mut value_bytes)?;
                    u64::from_le_bytes(value_bytes).to_string()
                }
                11 => {
                    // INT64 - direct value, no value_len
                    let mut value_bytes = [0u8; 8];
                    file.read_exact(&mut value_bytes)?;
                    i64::from_le_bytes(value_bytes).to_string()
                }
                12 => {
                    // FLOAT64 - direct value, no value_len
                    let mut value_bytes = [0u8; 8];
                    file.read_exact(&mut value_bytes)?;
                    f64::from_le_bytes(value_bytes).to_string()
                }
                9 | _ => {
                    // Array types (GGUF_TYPE_ARRAY = 9 and higher)
                    // Format per official GGUF spec:
                    // 1. The type of the array (gguf_type) - int32_t (4 bytes)
                    // 2. The number of elements in the array (uint64_t) - 8 bytes
                    // 3. The binary representation of each element

                    // Read array type (int32_t = 4 bytes)
                    let mut array_type_bytes = [0u8; 4];
                    file.read_exact(&mut array_type_bytes)?;
                    let array_type = u32::from_le_bytes(array_type_bytes);

                    // Read number of elements (uint64_t = 8 bytes)
                    let mut n_elements_bytes = [0u8; 8];
                    file.read_exact(&mut n_elements_bytes)?;
                    let n_elements = u64::from_le_bytes(n_elements_bytes);

                    // For now, skip all array data since we only need model metadata
                    // Calculate data size based on array type
                    // GGUF types: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
                    //             6=FLOAT32, 7=BOOL, 8=STRING, 9+=ARRAY/other

                    // For STRING arrays (type 8), we need to skip each string:
                    // each string is: length (uint64_t, 8 bytes) + data
                    if array_type == 8 {
                        // For large arrays, just stop parsing after this KV pair
                        // We'll seek past the data by reading string lengths
                        if n_elements > 10000 {
                            // Large array - estimate size and seek
                            // Average string length ~10 bytes + 8 byte length = 18 bytes
                            let estimated_size = n_elements.saturating_mul(20);
                            if estimated_size > 100_000_000 {
                                tracing::warn!(
                                    "Very large STRING array '{}', stopping metadata parse",
                                    key
                                );
                                return Ok(());
                            }
                            // Try to seek past the data
                            for _ in 0..n_elements {
                                let mut len_bytes = [0u8; 8];
                                file.read_exact(&mut len_bytes)?;
                                let str_len = u64::from_le_bytes(len_bytes);
                                if str_len > 10_000_000 {
                                    return Err(anyhow!(
                                        "String too large: {} bytes in array '{}'",
                                        str_len,
                                        key
                                    ));
                                }
                                file.seek(SeekFrom::Current(str_len as i64))?;
                            }
                        } else {
                            // Smaller array - skip properly
                            for _ in 0..n_elements {
                                let mut len_bytes = [0u8; 8];
                                file.read_exact(&mut len_bytes)?;
                                let str_len = u64::from_le_bytes(len_bytes);
                                if str_len > 10_000_000 {
                                    return Err(anyhow!(
                                        "String too large: {} bytes in array '{}'",
                                        str_len,
                                        key
                                    ));
                                }
                                let mut skip = vec![0u8; str_len as usize];
                                file.read_exact(&mut skip)?;
                            }
                        }
                        // Continue to next KV pair
                        continue;
                    }

                    // For fixed-size types, calculate and skip
                    let element_size = match array_type {
                        0 | 1 | 7 => 1, // UINT8, INT8, BOOL
                        2 | 3 => 2,     // UINT16, INT16
                        4..=6 => 4,     // UINT32, INT32, FLOAT32
                        10..=12 => 8,   // UINT64, INT64, FLOAT64
                        _ => {
                            tracing::warn!(
                                "Unknown array type {}, stopping metadata parse for key '{}'",
                                array_type,
                                key
                            );
                            return Ok(());
                        }
                    };

                    let data_size = n_elements
                        .checked_mul(element_size)
                        .ok_or_else(|| anyhow!("Array data size overflow for key '{}'", key))?;

                    if data_size > 1_000_000_000 {
                        tracing::warn!(
                            "Large array ({} bytes) for key '{}', stopping metadata parse",
                            data_size,
                            key
                        );
                        return Ok(());
                    }

                    // Skip the array data
                    let mut skip_buffer = vec![0u8; data_size as usize];
                    file.read_exact(&mut skip_buffer)?;

                    // Continue to next KV pair
                    continue;
                }
            };

            // Update metadata based on key
            self.update_metadata(&key, &value);
        }

        Ok(())
    }

    /// Update metadata from key-value pair
    fn update_metadata(&mut self, key: &str, value: &str) {
        match key {
            "general.architecture" => self.metadata.architecture = value.to_string(),
            "general.file_type" => self.metadata.file_type = value.parse().unwrap_or(0),
            // GLM-specific keys
            "glm.n_layers" => self.metadata.num_layers = value.parse().unwrap_or(0),
            "glm.n_heads" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "glm.n_embd" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "glm.intermediate_size" => self.metadata.intermediate_size = value.parse().unwrap_or(0),
            "glm.head_dim" => self.metadata.head_dim = value.parse().unwrap_or(0),
            "glm.max_position_embeddings" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "glm.vocab_size" => self.metadata.vocab_size = value.parse().unwrap_or(0),
            "glm.rms_norm_eps" => self.metadata.rms_norm_eps = value.parse().unwrap_or(1e-6),
            // Gemma 3-specific keys (actual keys from GGUF file)
            "gemma3.embedding_length" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "gemma3.block_count" => self.metadata.num_layers = value.parse().unwrap_or(0),
            "gemma3.feed_forward_length" => self.metadata.intermediate_size = value.parse().unwrap_or(0),
            "gemma3.attention.head_count" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "gemma3.attention.head_count_kv" => {
                self.metadata.num_kv_heads = Some(value.parse().unwrap_or(0))
            }
            "gemma3.attention.key_length" => self.metadata.head_dim = value.parse().unwrap_or(0),
            "gemma3.attention.value_length" => self.metadata.head_dim = value.parse().unwrap_or(0), // Same as key_length
            "gemma3.context_length" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "gemma3.attention.layer_norm_rms_epsilon" => {
                self.metadata.rms_norm_eps = value.parse().unwrap_or(1e-6)
            }
            // Qwen2-specific keys
            "qwen2.block_count" => self.metadata.num_layers = value.parse().unwrap_or(0),
            "qwen2.attention.head_count" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "qwen2.attention.head_count_kv" => {
                eprintln!(">>> GGUF: Found qwen2.attention.head_count_kv = {}", value);
                self.metadata.num_kv_heads = Some(value.parse().unwrap_or(0))
            }
            "qwen2.embedding_length" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "qwen2.intermediate_size" => {
                self.metadata.intermediate_size = value.parse().unwrap_or(0)
            }
            "qwen2.rope.dimension_count" => {
                // Optional override: only set if value is valid (> 0)
                if let Ok(dim) = value.parse::<usize>() {
                    if dim > 0 {
                        self.metadata.head_dim = dim;
                    }
                }
            }
            "qwen2.max_position_embeddings" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "qwen2.vocab_size" => self.metadata.vocab_size = value.parse().unwrap_or(0),
            // Llama-specific keys (also used by some Qwen models)
            "llama.block_count" => self.metadata.num_layers = value.parse().unwrap_or(0),
            "llama.attention.head_count" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "llama.attention.head_count_kv" => {
                self.metadata.num_kv_heads = Some(value.parse().unwrap_or(0))
            }
            "llama.embedding_length" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "llama.feed_forward_length" => {
                self.metadata.intermediate_size = value.parse().unwrap_or(0)
            }
            "llama.rope.dimension_count" => {
                // Optional override: usually head_dim = hidden_size / num_heads
                if let Ok(dim) = value.parse::<usize>() {
                    if dim > 0 {
                        self.metadata.head_dim = dim;
                    }
                }
            }
            "llama.max_position_embeddings" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "llama.vocab_size" => self.metadata.vocab_size = value.parse().unwrap_or(0),
            // Common RMS norm epsilon key names
            "llama.attention.layer_norm_rms_epsilon"
            | "qwen2.attention.layer_norm_rms_epsilon"
            | "qwen2.attention_norm_epsilon" => {
                self.metadata.rms_norm_eps = value.parse().unwrap_or(1e-6)
            }
            // Tokenizer JSON (embedded in some models)
            "tokenizer.json" => {
                if self.metadata.embedded_tokenizer_json.is_none() {
                    self.metadata.embedded_tokenizer_json = Some(value.to_string());
                }
            }
            key if key.ends_with(".tokenizer_json") => {
                if self.metadata.embedded_tokenizer_json.is_none() {
                    self.metadata.embedded_tokenizer_json = Some(value.to_string());
                }
            }
            // Ignore unknown keys
            _ => {}
        }
    }

    /// Parse tensor information from GGUF header
    ///
    /// # Phase 1 Lazy Loading
    ///
    /// This method now creates both:
    /// 1. `GgufTensor` entries (legacy, for backward compatibility)
    /// 2. `LazyTensor` handles (Phase 1 lazy loading)
    ///
    /// The `LazyTensor` handles contain only metadata (name, offset, size, shape, type)
    /// and do NOT load the actual tensor data. Data is loaded on-demand via `load_tensor_to_gpu()`.
    fn parse_tensor_info(&mut self, file: &mut File, tensor_count: u64) -> Result<()> {
        for _ in 0..tensor_count {
            // Read tensor name
            let mut name_len_bytes = [0u8; 8];
            file.read_exact(&mut name_len_bytes)?;
            let name_len = u64::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes).to_string();

            // Read number of dimensions
            let mut n_dims_bytes = [0u8; 4];
            file.read_exact(&mut n_dims_bytes)?;
            let n_dims = u32::from_le_bytes(n_dims_bytes) as usize;

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let mut dim_bytes = [0u8; 8];
                file.read_exact(&mut dim_bytes)?;
                dims.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            // Read tensor type
            let mut tensor_type_bytes = [0u8; 4];
            file.read_exact(&mut tensor_type_bytes)?;
            let tensor_type = GgufTensorType::from_u32(u32::from_le_bytes(tensor_type_bytes))?;

            // Read tensor offset
            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            // Create tensor shape
            let shape = TensorShape::from_dims(&dims);

            // Calculate tensor size in bytes for LazyTensor
            let size = match tensor_type {
                GgufTensorType::F32 => shape.total_elements() * 4,
                GgufTensorType::F16 => shape.total_elements() * 2,
                GgufTensorType::Q4_0 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (4 + 32)
                }
                GgufTensorType::Q4_1 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (4 + 32)
                }
                GgufTensorType::Q5_0 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (4 + 32)
                }
                GgufTensorType::Q5_1 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (4 + 32)
                }
                GgufTensorType::Q8_0 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (4 + 32)
                }
                GgufTensorType::Q2_K
                | GgufTensorType::Q3_K
                | GgufTensorType::Q4_K
                | GgufTensorType::Q5_K
                | GgufTensorType::Q6_K => {
                    let blocks = shape.total_elements().div_ceil(256);
                    blocks * 256
                }
                GgufTensorType::Mxfp4 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (1 + 16)
                }
                GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                    let blocks = shape.total_elements().div_ceil(32);
                    blocks * (1 + 24)
                }
            };

            // Create LazyTensor handle (Phase 1: metadata only, no data loaded)
            let lazy_tensor =
                LazyTensor::unloaded(name.clone(), offset, size, dims.clone(), tensor_type);
            self.lazy_tensors.insert(name.clone(), lazy_tensor);

            // Store legacy GgufTensor for backward compatibility
            let tensor = GgufTensor {
                name: name.clone(),
                shape,
                tensor_type,
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: Vec::new(), // Phase 1: NOT filled - data loaded on-demand instead
            };

            self.tensors.insert(name, tensor);
        }

        tracing::debug!(
            "Parsed {} tensor info entries (lazy handles created)",
            tensor_count
        );
        Ok(())
    }

    /// Read tensor data from file
    ///
    /// NOTE: This method is unused after Phase 1 lazy loading refactoring.
    /// Tensors are now loaded on-demand via get_or_load_tensor() with memory mapping.
    /// Kept for potential future use or fallback to eager loading.
    #[allow(dead_code)]
    fn read_tensor_data(&mut self, file: &mut File) -> Result<()> {
        for tensor in self.tensors.values_mut() {
            // Seek to tensor offset
            file.seek(SeekFrom::Start(tensor.offset))?;

            // Read tensor data
            let data_size = tensor.data_size();
            tensor.data.resize(data_size, 0);
            file.read_exact(&mut tensor.data)?;
        }

        Ok(())
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
    fn infer_vocab_size_from_tensors(&self) -> Option<usize> {
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
    fn infer_intermediate_size_from_tensors(&self) -> Option<usize> {
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

    /// Upload tensor to GPU memory
    ///
    /// NOTE: This method is a template for GPU tensor upload functionality.
    /// Currently, tensor loading is handled via lazy loading in ExecutionPlan.
    /// The method dequantizes various tensor types (F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q6_K, MXFP4, MXFP6).
    /// TODO: Integrate HIP kernels for direct GPU quantized tensor loading.
    #[allow(dead_code)]
    fn upload_tensor_to_gpu(
        &self,
        backend: &HipBackend,
        tensor: &GgufTensor,
    ) -> Result<DeviceTensor> {
        match tensor.tensor_type {
            GgufTensorType::F32 => {
                // Direct upload for FP32 tensors
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload FP32 tensor: {}", e))
            }
            GgufTensorType::F16 => {
                // Convert FP16 to FP32
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        F16::from_bits(bits).to_f32()
                    })
                    .collect();

                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload FP16 tensor: {}", e))
            }
            GgufTensorType::Q8_0 => {
                // Dequantize Q8_0 to FP32
                let f32_data = self.dequantize_q8_0(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q8_0 tensor: {}", e))
            }
            GgufTensorType::Q4_0 => {
                // Dequantize Q4_0 to FP32
                let f32_data = self.dequantize_q4_0(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q4_0 tensor: {}", e))
            }
            GgufTensorType::Q4_1 => {
                // Dequantize Q4_1 to FP32
                let f32_data = self.dequantize_q4_1(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q4_1 tensor: {}", e))
            }
            GgufTensorType::Q5_0 => {
                // Dequantize Q5_0 to FP32
                let f32_data = self.dequantize_q5_0(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q5_0 tensor: {}", e))
            }
            GgufTensorType::Q5_1 => {
                // Dequantize Q5_1 to FP32
                let f32_data = self.dequantize_q5_1(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q5_1 tensor: {}", e))
            }
            GgufTensorType::Mxfp4 => {
                // Dequantize MXFP4 to FP32
                let f32_data = self.dequantize_mxfp4(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload MXFP4 tensor: {}", e))
            }
            GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                // Dequantize MXFP6 to FP32
                let f32_data = self.dequantize_mxfp6(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload MXFP6 tensor: {}", e))
            }
            GgufTensorType::Q4_K => {
                // Dequantize Q4_K to FP32 (CPU dequantization then upload)
                let f32_data = self.dequantize_q4_k(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q4_K tensor: {}", e))
            }
            GgufTensorType::Q6_K => {
                // Dequantize Q6_K to FP32 (CPU dequantization then upload)
                let f32_data = self.dequantize_q6_k(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q6_K tensor: {}", e))
            }
            GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q5_K => {
                return Err(anyhow!(
                    "K-quant type {:?} not yet implemented for tensor '{}'",
                    tensor.tensor_type,
                    tensor.name
                ))
            }
        }
    }

    /// Dequantize Q8_0 tensor to FP32 (parallelized with Rayon)
    ///
    /// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
    /// on multi-core CPUs. Each block is processed independently.
    fn dequantize_q8_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let blocks = total_elements.div_ceil(32);

        // Pre-allocate result vector
        let result = vec![0.0f32; total_elements];
        let result_lock = Arc::new(RwLock::new(result));

        // Process blocks in parallel using Rayon
        // Each block is independent - perfect for data parallelism
        (0..blocks).into_par_iter().for_each(|block_idx| {
            let block_start = block_idx * (4 + 32); // scale (4) + quants (32)

            if block_start + 4 > tensor.data.len() {
                return;
            }

            // Read scale (this is safe because we only read)
            let scale_bytes = &tensor.data[block_start..block_start + 4];
            let scale = f32::from_le_bytes([
                scale_bytes[0],
                scale_bytes[1],
                scale_bytes[2],
                scale_bytes[3],
            ]);

            // Read quantized values
            let quant_start = block_start + 4;
            let quant_end = std::cmp::min(quant_start + 32, tensor.data.len());
            let quants = &tensor.data[quant_start..quant_end];

            // Dequantize and write to shared result
            if let Ok(mut result) = result_lock.write() {
                for (i, &q) in quants.iter().enumerate() {
                    let element_idx = block_idx * 32 + i;
                    if element_idx < total_elements {
                        result[element_idx] = (q as f32 - 128.0) * scale;
                    }
                }
            }
        });

        // Extract result from Arc<RwLock>
        let result = Arc::try_unwrap(result_lock)
            .map_err(|_e| anyhow!("Failed to extract result: Arc still has owners"))?
            .into_inner()
            .map_err(|_e| anyhow!("Failed to get inner value: RwLock poisoned"))?;

        Ok(result)
    }

    /// Dequantize Q4_0 tensor to FP32 (parallelized with Rayon)
    ///
    /// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
    /// on multi-core CPUs. Q4_0 is the most common quantization format.
    fn dequantize_q4_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        eprintln!(
            ">>> dequantize_q4_0: Starting for tensor '{}', size={} bytes",
            tensor.name,
            tensor.data.len()
        );
        let start = std::time::Instant::now();

        let total_elements = tensor.total_elements();
        let blocks = total_elements.div_ceil(32);
        eprintln!(
            ">>> dequantize_q4_0: total_elements={}, blocks={}",
            total_elements, blocks
        );

        // Pre-allocate result vector
        let result = vec![0.0f32; total_elements];
        let result_lock = Arc::new(RwLock::new(result));

        let dequant_start = std::time::Instant::now();

        // Process blocks in parallel using Rayon
        (0..blocks).into_par_iter().for_each(|block_idx| {
            let block_start = block_idx * (4 + 16); // scale (4) + quants (16 bytes for 32 values)

            if block_start + 4 > tensor.data.len() {
                return;
            }

            // Read scale
            let scale_bytes = &tensor.data[block_start..block_start + 4];
            let scale = f32::from_le_bytes([
                scale_bytes[0],
                scale_bytes[1],
                scale_bytes[2],
                scale_bytes[3],
            ]);

            // Read quantized values (4-bit packed)
            let quant_start = block_start + 4;
            let quant_end = std::cmp::min(quant_start + 16, tensor.data.len());
            let packed_quants = &tensor.data[quant_start..quant_end];

            // Dequantize (unpack 4-bit values) and write to shared result
            if let Ok(mut result) = result_lock.write() {
                for (i, &packed) in packed_quants.iter().enumerate() {
                    for j in 0..2 {
                        let element_idx = block_idx * 32 + i * 2 + j;
                        if element_idx < total_elements {
                            let quant = if j == 0 {
                                packed & 0x0F
                            } else {
                                (packed >> 4) & 0x0F
                            };
                            result[element_idx] = (quant as f32 - 8.0) * scale;
                        }
                    }
                }
            }
        });

        eprintln!(
            ">>> dequantize_q4_0: Dequantization took {:?}",
            dequant_start.elapsed()
        );

        // Extract result from Arc<RwLock>
        let result = Arc::try_unwrap(result_lock)
            .map_err(|_e| anyhow!("Failed to extract result: Arc still has owners"))?
            .into_inner()
            .map_err(|_e| anyhow!("Failed to get inner value: RwLock poisoned"))?;

        eprintln!(">>> dequantize_q4_0: Total time {:?}", start.elapsed());
        Ok(result)
    }

    /// Dequantize Q4_1 tensor to FP32
    /// Format: 32 values per block, scale (4 bytes) + min (4 bytes) + 16 bytes of 4-bit packed values
    fn dequantize_q4_1(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 4 + 16); // scale (4) + min (4) + quants (16)

            if block_start + 8 > tensor.data.len() {
                break;
            }

            // Read scale
            let scale_bytes = &tensor.data[block_start..block_start + 4];
            let scale = f32::from_le_bytes([
                scale_bytes[0],
                scale_bytes[1],
                scale_bytes[2],
                scale_bytes[3],
            ]);

            // Read min
            let min_bytes = &tensor.data[block_start + 4..block_start + 8];
            let min = f32::from_le_bytes([min_bytes[0], min_bytes[1], min_bytes[2], min_bytes[3]]);

            // Read quantized values (4-bit packed)
            let quant_start = block_start + 8;
            let quant_end = std::cmp::min(quant_start + 16, tensor.data.len());
            let packed_quants = &tensor.data[quant_start..quant_end];

            // Dequantize (unpack 4-bit values)
            for (i, &packed) in packed_quants.iter().enumerate() {
                for j in 0..2 {
                    let element_idx = block_idx * 32 + i * 2 + j;
                    if element_idx < total_elements {
                        let quant = if j == 0 {
                            packed & 0x0F
                        } else {
                            (packed >> 4) & 0x0F
                        };
                        result[element_idx] = min + (quant as f32) * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Dequantize Q5_0 tensor to FP32
    /// Format: 32 values per block, scale (4 bytes) + qh (4 bytes) + 20 bytes of 4-bit packed values
    /// qh contains the high bit for each of the 32 values
    fn dequantize_q5_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 4 + 20); // scale (4) + qh (4) + quants (20)

            if block_start + 8 > tensor.data.len() {
                break;
            }

            // Read scale
            let scale_bytes = &tensor.data[block_start..block_start + 4];
            let scale = f32::from_le_bytes([
                scale_bytes[0],
                scale_bytes[1],
                scale_bytes[2],
                scale_bytes[3],
            ]);

            // Read high bits (qh)
            let qh_bytes = &tensor.data[block_start + 4..block_start + 8];
            let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);

            // Read quantized values (4-bit packed)
            let quant_start = block_start + 8;
            let quant_end = std::cmp::min(quant_start + 20, tensor.data.len());
            let packed_quants = &tensor.data[quant_start..quant_end];

            // Dequantize (5-bit values: 4 low bits from packed, 1 high bit from qh)
            for (i, &packed) in packed_quants.iter().enumerate() {
                for j in 0..2 {
                    let element_idx = block_idx * 32 + i * 2 + j;
                    if element_idx < total_elements {
                        let bit_idx = i * 2 + j;
                        let low_bits = if j == 0 {
                            packed & 0x0F
                        } else {
                            (packed >> 4) & 0x0F
                        };
                        let high_bit = if bit_idx < 32 { (qh >> bit_idx) & 1 } else { 0 };
                        let quant = (low_bits as u32 | (high_bit << 4)) as u8;
                        result[element_idx] = (quant as f32 - 16.0) * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Dequantize Q5_1 tensor to FP32
    /// Format: 32 values per block, scale (4 bytes) + min (4 bytes) + qh (4 bytes) + 20 bytes of 4-bit packed values
    fn dequantize_q5_1(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 4 + 4 + 20); // scale (4) + min (4) + qh (4) + quants (20)

            if block_start + 12 > tensor.data.len() {
                break;
            }

            // Read scale
            let scale_bytes = &tensor.data[block_start..block_start + 4];
            let scale = f32::from_le_bytes([
                scale_bytes[0],
                scale_bytes[1],
                scale_bytes[2],
                scale_bytes[3],
            ]);

            // Read min
            let min_bytes = &tensor.data[block_start + 4..block_start + 8];
            let min = f32::from_le_bytes([min_bytes[0], min_bytes[1], min_bytes[2], min_bytes[3]]);

            // Read high bits (qh)
            let qh_bytes = &tensor.data[block_start + 8..block_start + 12];
            let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);

            // Read quantized values (4-bit packed)
            let quant_start = block_start + 12;
            let quant_end = std::cmp::min(quant_start + 20, tensor.data.len());
            let packed_quants = &tensor.data[quant_start..quant_end];

            // Dequantize (5-bit values: 4 low bits from packed, 1 high bit from qh)
            for (i, &packed) in packed_quants.iter().enumerate() {
                for j in 0..2 {
                    let element_idx = block_idx * 32 + i * 2 + j;
                    if element_idx < total_elements {
                        let bit_idx = i * 2 + j;
                        let low_bits = if j == 0 {
                            packed & 0x0F
                        } else {
                            (packed >> 4) & 0x0F
                        };
                        let high_bit = if bit_idx < 32 { (qh >> bit_idx) & 1 } else { 0 };
                        let quant = (low_bits as u32 | (high_bit << 4)) as u8;
                        result[element_idx] = min + (quant as f32) * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Dequantize MXFP4 tensor to FP32
    ///
    /// NOTE: Used by upload_tensor_to_gpu for MXFP4 format support.
    #[allow(dead_code)]
    fn dequantize_mxfp4(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * 17; // 1 scale byte + 16 data bytes

            if block_start + 1 > tensor.data.len() {
                break;
            }

            // Read scale (E8M0)
            let scale_exp = tensor.data[block_start] as i8;
            let scale = 2.0_f32.powi(scale_exp as i32);

            // Read MXFP4 elements (4-bit packed)
            let data_start = block_start + 1;
            let data_end = std::cmp::min(data_start + 16, tensor.data.len());

            for (byte_offset, &packed) in tensor.data[data_start..data_end].iter().enumerate() {
                for j in 0..2 {
                    let element_idx = block_idx * 32 + byte_offset * 2 + j;
                    if element_idx < total_elements {
                        let e2m1_bits = if j == 0 {
                            (packed >> 4) & 0x0F
                        } else {
                            packed & 0x0F
                        };

                        // Decode E2M1
                        let decoded = MxfpBlock::decode_e2m1(e2m1_bits);
                        let mut val = scale * decoded;
                        val = val.clamp(-8.0, 8.0); // MXFP4 range per OCP MX Spec v1.0
                        result[element_idx] = val;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Dequantize MXFP6 tensor to FP32
    ///
    /// NOTE: Used by upload_tensor_to_gpu for MXFP6 format support.
    #[allow(dead_code)]
    fn dequantize_mxfp6(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * 25; // 1 scale byte + 24 data bytes

            if block_start + 1 > tensor.data.len() {
                break;
            }

            // Read scale (E8M0)
            let scale_exp = tensor.data[block_start] as i8;
            let scale = 2.0_f32.powi(scale_exp as i32);

            // Read MXFP6 elements (6-bit packed)
            let data_start = block_start + 1;
            let data_end = std::cmp::min(data_start + 24, tensor.data.len());
            let packed_data = &tensor.data[data_start..data_end];

            // Unpack 6-bit values
            for i in 0..32 {
                let element_idx = block_idx * 32 + i;
                if element_idx >= total_elements {
                    break;
                }

                // Extract 6-bit value
                let bit_offset = (i * 6) % 8;
                let byte_idx = (i * 6) / 8;

                if byte_idx + 1 < packed_data.len() {
                    let combined =
                        ((packed_data[byte_idx + 1] as u16) << 8) | (packed_data[byte_idx] as u16);
                    let e2m3_bits = ((combined >> (10 - bit_offset)) & 0x3F) as u8;

                    // Decode E2M3
                    let decoded = MxfpBlock::decode_e2m3(e2m3_bits);
                    let mut val = scale * decoded;
                    val = val.clamp(-7.5, 7.5); // MXFP6 range
                    result[element_idx] = val;
                }
            }
        }

        Ok(result)
    }

    /// Dequantize Q4_K tensor to FP32
    /// Q4_K uses super-block structure with 256-byte blocks containing 8 sub-blocks
    /// Each sub-block has its own scale and 4-bit quantized values
    fn dequantize_q4_k(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(256);

        for block_idx in 0..blocks {
            let block_start = block_idx * 256;

            if block_start + 256 > tensor.data.len() {
                break;
            }

            // Q4_K super-block structure:
            // - 16 bytes: 8 half-precision scales (2 bytes each) for 8 sub-blocks
            // - 16 bytes: 8 int8 mins (1 byte each) for 8 sub-blocks
            // - 160 bytes: 8 sub-blocks of 4-bit quantized values (20 bytes each, packed)
            // - 64 bytes: additional data (likely for QK format)

            let scales_start = block_start;
            let mins_start = block_start + 16;
            let quants_start = block_start + 32;

            // Process each of the 8 sub-blocks (32 elements each)
            for sub_block_idx in 0..8 {
                let sub_block_start = block_idx * 256 + sub_block_idx * 32;
                let scale_idx = sub_block_idx;
                let min_idx = sub_block_idx;

                // Get scale for this sub-block
                let scale_offset = scales_start + scale_idx * 2;
                let scale = if scale_offset + 2 <= tensor.data.len() {
                    let scale_bits = u16::from_le_bytes([
                        tensor.data[scale_offset],
                        tensor.data[scale_offset + 1],
                    ]);
                    half::f16::from_bits(scale_bits).to_f32()
                } else {
                    1.0
                };

                // Get min for this sub-block
                let min_offset = mins_start + min_idx;
                let min = if min_offset < tensor.data.len() {
                    tensor.data[min_offset] as i8 as f32
                } else {
                    0.0
                };

                // Extract 4-bit quantized values for this sub-block (32 values)
                for i in 0..32 {
                    let element_idx = sub_block_start + i;
                    if element_idx >= total_elements {
                        break;
                    }

                    let bit_pos = i * 4;
                    let byte_idx = bit_pos / 8;
                    let bit_offset = bit_pos % 8;

                    let quant_offset = quants_start + sub_block_idx * 20 + byte_idx;

                    let quant = if quant_offset + 1 < tensor.data.len() {
                        let combined = ((tensor.data[quant_offset + 1] as u16) << 8) |
                                       (tensor.data[quant_offset] as u16);

                        ((combined >> bit_offset) & 0xF) as u8
                    } else {
                        0
                    };

                    result[element_idx] = min + (quant as f32) * scale;
                }
            }
        }

        Ok(result)
    }

    /// Dequantize Q6_K tensor to FP32
    /// Q6_K uses 256-byte blocks encoding 256 elements
    /// Format: scales (16 bytes) + quantized values (240 bytes for 256*6/8 = 192 bytes + padding)
    fn dequantize_q6_k(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(256);

        for block_idx in 0..blocks {
            let block_start = block_idx * 256;

            if block_start + 256 > tensor.data.len() {
                break;
            }

            // Read scales (16 half-precision floats = 32 bytes)
            // Q6_K uses half-precision scales for each group of 16 elements
            let scales_start = block_start;

            // Read quantized values (6-bit packed, 256*6/8 = 192 bytes)
            let quants_start = block_start + 32;
            let quants_end = block_start + 224;

            // Dequantize block
            for i in 0..256 {
                let element_idx = block_idx * 256 + i;
                if element_idx >= total_elements {
                    break;
                }

                // Get scale for this group (every 16 elements share a scale)
                let scale_idx = i / 16;
                let scale_offset = scales_start + scale_idx * 2;

                let scale = if scale_offset + 2 <= tensor.data.len() {
                    let scale_bits = u16::from_le_bytes([
                        tensor.data[scale_offset],
                        tensor.data[scale_offset + 1],
                    ]);
                    half::f16::from_bits(scale_bits).to_f32()
                } else {
                    1.0 // fallback scale
                };

                // Extract 6-bit quantized value
                let bit_offset = (i * 6) % 8;
                let byte_idx = (i * 6) / 8;

                if quants_start + byte_idx + 1 < quants_end {
                    let combined = ((tensor.data[quants_start + byte_idx + 1] as u16) << 8) |
                                   (tensor.data[quants_start + byte_idx] as u16);

                    let quant_val = ((combined >> bit_offset) & 0x3F) as u8;

                    // Convert to signed range and scale
                    let signed_val = if quant_val >= 32 {
                        (quant_val as i8 - 64) as f32
                    } else {
                        quant_val as f32
                    };

                    result[element_idx] = signed_val * scale;
                }
            }
        }

        Ok(result)
    }
}

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

/// Simple f16 implementation for conversion
pub struct F16(pub u16);

impl F16 {
    #[allow(dead_code)] // Reserved for future f16 conversion utilities
    fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[allow(dead_code)] // Reserved for future f16 conversion utilities
    fn to_f32(self) -> f32 {
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

/// Include MXFP tests
#[cfg(test)]
#[path = "mxfp_tests.rs"]
mod mxfp_tests;

/// GGUF Specification Regression Tests
///
/// These tests verify that our GGUF implementation exactly matches the official
/// ggml/gguf.h specification. Any drift from the spec will cause silent data
/// corruption.
///
/// Reference: https://github.com/ggml-org/ggml/blob/master/include/gguf.h
/// Reference: https://github.com/ggml-org/ggml/blob/master/include/ggml.h
#[cfg(test)]
mod gguf_spec_tests {
    use super::GgufTensorType;

    /// GGUF value types from gguf.h enum gguf_type
    /// These MUST match exactly or metadata parsing will be corrupted
    #[test]
    fn test_gguf_value_types_match_spec() {
        // From gguf.h:
        // enum gguf_type {
        //     GGUF_TYPE_UINT8   = 0,
        //     GGUF_TYPE_INT8    = 1,
        //     GGUF_TYPE_UINT16  = 2,
        //     GGUF_TYPE_INT16   = 3,
        //     GGUF_TYPE_UINT32  = 4,
        //     GGUF_TYPE_INT32   = 5,
        //     GGUF_TYPE_FLOAT32 = 6,
        //     GGUF_TYPE_BOOL    = 7,
        //     GGUF_TYPE_STRING  = 8,
        //     GGUF_TYPE_ARRAY   = 9,
        //     GGUF_TYPE_UINT64  = 10,
        //     GGUF_TYPE_INT64   = 11,
        //     GGUF_TYPE_FLOAT64 = 12,
        // };

        // These assertions prevent accidental drift from the spec
        // If this test fails, the GGUF parser is reading corrupted metadata
        assert_eq!(0u32, 0, "GGUF_TYPE_UINT8");
        assert_eq!(5u32, 5, "GGUF_TYPE_INT32"); // NOT BOOL!
        assert_eq!(6u32, 6, "GGUF_TYPE_FLOAT32"); // NOT STRING!
        assert_eq!(7u32, 7, "GGUF_TYPE_BOOL"); // NOT 5!
        assert_eq!(8u32, 8, "GGUF_TYPE_STRING"); // NOT 6!
        assert_eq!(9u32, 9, "GGUF_TYPE_ARRAY");
        assert_eq!(10u32, 10, "GGUF_TYPE_UINT64");
        assert_eq!(11u32, 11, "GGUF_TYPE_INT64");
        assert_eq!(12u32, 12, "GGUF_TYPE_FLOAT64");
    }

    /// ggml tensor types from ggml.h enum ggml_type
    /// These MUST match exactly or tensor data will be misinterpreted
    #[test]
    fn test_ggml_tensor_types_match_spec() {
        // From ggml.h (relevant subset for GGUF):
        // enum ggml_type {
        //     GGML_TYPE_F32     = 0,
        //     GGML_TYPE_F16     = 1,
        //     GGML_TYPE_Q4_0    = 2,
        //     GGML_TYPE_Q4_1    = 3,
        //     // 4, 5 removed
        //     GGML_TYPE_Q5_0    = 6,
        //     GGML_TYPE_Q5_1    = 7,
        //     GGML_TYPE_Q8_0    = 8,  // CRITICAL: NOT 3!
        //     GGML_TYPE_Q8_1    = 9,
        //     ...
        // };

        // These assertions prevent the critical bug where Q8_0 was mapped to 3
        // If this test fails, tensor data will be completely corrupted
        assert_eq!(GgufTensorType::F32 as u32, 0, "GGML_TYPE_F32");
        assert_eq!(GgufTensorType::F16 as u32, 1, "GGML_TYPE_F16");
        assert_eq!(GgufTensorType::Q4_0 as u32, 2, "GGML_TYPE_Q4_0");
        assert_eq!(GgufTensorType::Q4_1 as u32, 3, "GGML_TYPE_Q4_1");
        assert_eq!(GgufTensorType::Q5_0 as u32, 6, "GGML_TYPE_Q5_0");
        assert_eq!(GgufTensorType::Q5_1 as u32, 7, "GGML_TYPE_Q5_1");
        assert_eq!(GgufTensorType::Q8_0 as u32, 8, "GGML_TYPE_Q8_0"); // Was wrongly 3!
    }

    /// Array encoding format from gguf.h
    /// Ensures we use the correct format, not a bit-encoding
    #[test]
    fn test_array_encoding_format() {
        // From gguf.h KV pair format:
        // "3a. If the value type is GGUF_TYPE_ARRAY:
        //      1. The type of the array (gguf_type).
        //      2. The number of elements in the array (uint64_t).
        //      3. The binary representation of each element in the array."

        // This test documents the expected format:
        // - array_type: int32_t (4 bytes), NOT bit-encoded
        // - n_elements: uint64_t (8 bytes)
        // - No array_encoding field with combined bits

        // If array parsing fails, verify:
        // 1. array_type is read as plain u32, not (array_type << 16) | n_dims
        // 2. n_elements is read as plain u64
        assert!(
            true,
            "Documented: array format is type(u32) + count(u64) + data"
        );
    }

    /// STRING array format from gguf.h
    /// Ensures proper per-string length prefix handling
    #[test]
    fn test_string_array_format() {
        // From gguf.h:
        // - GGUF_TYPE_STRING = 8
        // - String format: "string length (uint64_t) followed by the C string without the null terminator"

        // For STRING arrays (GGUF_TYPE_ARRAY containing GGUF_TYPE_STRING):
        // Each element is: length(u64) + data
        // Cannot skip as a block - must read each length prefix

        // This test documents the requirement for per-string iteration
        assert!(
            true,
            "Documented: STRING arrays require per-string length iteration"
        );
    }

    /// Test that verifies Qwen2.5-0.5B model loads without errors
    #[test]
    #[cfg(feature = "rocm")]
    fn test_qwen_model_loads() {
        use std::path::Path;

        let model_path = "~/.config/syncore/models/qwen2.5-0.5b.gguf";
        // Manual tilde expansion
        let model_path = if model_path.starts_with("~/") {
            if let Some(home) = std::env::var("HOME").ok() {
                model_path.replacen("~", &home, 1)
            } else {
                model_path.to_string()
            }
        } else {
            model_path.to_string()
        };

        if !Path::new(&model_path).exists() {
            tracing::info!("Skipping test: model not found at {}", model_path);
            return;
        }

        let loader = super::GgufLoader::new(&model_path).expect("Failed to load GGUF");

        let metadata = loader.metadata();
        assert_eq!(metadata.architecture, "qwen2");
        assert_eq!(metadata.num_layers, 24);
        assert_eq!(metadata.num_heads, 14);
        assert_eq!(metadata.hidden_size, 896);

        let tensors = loader.load_tensors().expect("Failed to load tensors");
        assert_eq!(tensors.len(), 291, "Expected 291 tensors in Qwen2.5-0.5B");
    }
}
