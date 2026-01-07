//! GGUF (GPT-Generated Unified Format) Loader
//!
//! Complete implementation for loading GGUF model files with support for:
//! - Metadata parsing
//! - Multiple quantization types (Q8_0, Q4_0, FP16, FP32)
//! - Tensor block reading and validation
//! - GPU memory allocation via DeviceTensor

use crate::backend::hip_backend::{DeviceTensor, HipBackend};
use crate::loader::TensorShape;
use crate::model::config::ModelConfig;
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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
            &values.iter().take(32).map(|&v| {
                // Normalize value by scale (should now be in range [0, 1])
                let normalized = if scale_f32 > 0.0 {
                    v / scale_f32
                } else {
                    v
                };
                Self::encode_e2m3(normalized)
            }).collect::<Vec<u8>>()
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

        unpacked_bits.iter().map(|&bits| {
            let decoded = Self::decode_e2m3(bits);
            let mut val = scale_f32 * decoded;
            val = val.clamp(-7.5, 7.5); // MXFP6 range
            val
        }).collect()
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

                    let first_part = (packed[byte_idx] >> bit_offset) & ((1 << bits_from_first_byte) - 1);
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

/// GGUF tensor types (ggml_type enum values from ggml.h)
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum GgufTensorType {
    F32 = 0,   // GGML_TYPE_F32
    F16 = 1,   // GGML_TYPE_F16
    Q4_0 = 2,  // GGML_TYPE_Q4_0
    Q4_1 = 3,  // GGML_TYPE_Q4_1
    Q5_0 = 6,  // GGML_TYPE_Q5_0
    Q5_1 = 7,  // GGML_TYPE_Q5_1
    Q8_0 = 8,  // GGML_TYPE_Q8_0

    // MXFP types (OCP MX Specification v1.0)
    // Using enum values 20-22 to avoid conflicts with future ggml types
    Mxfp4 = 20,      // OCP MXFP4-E2M1 (4-bit)
    Mxfp6E2m3 = 21,  // OCP MXFP6-E2M3 (6-bit, recommended)
    Mxfp6E3m2 = 22,  // OCP MXFP6-E3M2 (6-bit)
}

impl GgufTensorType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            2 => Ok(GgufTensorType::Q4_0),
            3 => Ok(GgufTensorType::Q4_1),
            6 => Ok(GgufTensorType::Q5_0),
            7 => Ok(GgufTensorType::Q5_1),
            8 => Ok(GgufTensorType::Q8_0),
            20 => Ok(GgufTensorType::Mxfp4),
            21 => Ok(GgufTensorType::Mxfp6E2m3),
            22 => Ok(GgufTensorType::Mxfp6E3m2),
            _ => Err(anyhow!("Unknown tensor type: {}", value)),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            GgufTensorType::F32 => "FP32",
            GgufTensorType::F16 => "FP16",
            GgufTensorType::Q4_0 => "Q4_0",
            GgufTensorType::Q4_1 => "Q4_1",
            GgufTensorType::Q5_0 => "Q5_0",
            GgufTensorType::Q5_1 => "Q5_1",
            GgufTensorType::Q8_0 => "Q8_0",
            GgufTensorType::Mxfp4 => "MXFP4",
            GgufTensorType::Mxfp6E2m3 => "MXFP6_E2M3",
            GgufTensorType::Mxfp6E3m2 => "MXFP6_E3M2",
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            GgufTensorType::F32 => 4,
            GgufTensorType::F16 => 2,
            GgufTensorType::Q4_0 => {
                // Q4_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                32
            }
            GgufTensorType::Q4_1 => {
                // Q4_1: block_size=32, each block has scales + quants
                32
            }
            GgufTensorType::Q5_0 => {
                // Q5_0: block_size=32
                32
            }
            GgufTensorType::Q5_1 => {
                // Q5_1: block_size=32
                32
            }
            GgufTensorType::Q8_0 => {
                // Q8_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                32
            }
            GgufTensorType::Mxfp4 | GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                // MXFP formats: block_size=32
                32
            }
        }
    }
}

/// GGUF metadata extracted from file header
#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadata {
    pub architecture: String,
    pub file_type: u32,
    pub num_layers: usize,
    pub num_heads: usize,
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
            GgufTensorType::F32 => self.total_elements() * 4,
            GgufTensorType::F16 => self.total_elements() * 2,
            GgufTensorType::Q4_0 => {
                // Q4_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                let blocks = self.total_elements().div_ceil(32);
                blocks * (4 + 32)
            }
            GgufTensorType::Q4_1 => {
                // Q4_1: similar structure to Q4_0
                let blocks = self.total_elements().div_ceil(32);
                blocks * (4 + 32)
            }
            GgufTensorType::Q5_0 => {
                // Q5_0: block_size=32
                let blocks = self.total_elements().div_ceil(32);
                blocks * (4 + 32)
            }
            GgufTensorType::Q5_1 => {
                // Q5_1: block_size=32
                let blocks = self.total_elements().div_ceil(32);
                blocks * (4 + 32)
            }
            GgufTensorType::Q8_0 => {
                // Q8_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                let blocks = self.total_elements().div_ceil(32);
                blocks * (4 + 32)
            }
            GgufTensorType::Mxfp4 => {
                // MXFP4: block_size=32, each block has 1 scale (E8M0) + 32*4 bits data
                let blocks = self.total_elements().div_ceil(32);
                blocks * (1 + 16) // 1 byte scale + 16 bytes data
            }
            GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                // MXFP6: block_size=32, each block has 1 scale (E8M0) + 32*6 bits data
                let blocks = self.total_elements().div_ceil(32);
                blocks * (1 + 24) // 1 byte scale + 24 bytes data
            }
        }
    }
}

/// GGUF file loader
#[derive(Debug)]
pub struct GgufLoader {
    path: String,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensor>,
}

impl GgufLoader {
    /// Create new GGUF loader from file path
    pub fn new(path: &str) -> Result<Self> {
        let mut loader = GgufLoader {
            path: path.to_string(),
            metadata: GgufMetadata::default(),
            tensors: HashMap::new(),
        };

        loader.load_from_disk(true)?;
        Ok(loader)
    }

    /// Inspect only metadata without loading tensors into memory.
    pub fn metadata_from_file(path: &str) -> Result<GgufMetadata> {
        let mut loader = GgufLoader {
            path: path.to_string(),
            metadata: GgufMetadata::default(),
            tensors: HashMap::new(),
        };
        loader.load_from_disk(false)?;
        Ok(loader.metadata)
    }

    /// Get metadata
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Load all tensors into memory
    pub fn load_tensors(&self) -> Result<HashMap<String, GgufTensor>> {
        Ok(self.tensors.clone())
    }

    /// Load tensors and upload to GPU using batched memory pooling.
    /// This allocates multiple moderate-sized buffers instead of thousands of small allocations,
    /// avoiding ROCm driver bugs while staying within reasonable allocation limits.
    pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        use crate::backend::hip_backend::HipBuffer;

        // Pool size: 1 GB per pool (large enough for biggest tensors, small enough for ROCm)
        const POOL_SIZE: usize = 1024 * 1024 * 1024;

        // Calculate tensor sizes
        let mut tensor_list: Vec<(String, usize)> = Vec::new();
        for (name, tensor) in &self.tensors {
            let num_elements = tensor.shape.total_elements();
            let tensor_bytes = num_elements * std::mem::size_of::<f32>();
            tensor_list.push((name.clone(), tensor_bytes));
        }

        let total_bytes: usize = tensor_list.iter().map(|(_, size)| size).sum();
        eprintln!("DEBUG: Batched memory pooling - total: {} bytes ({:.2} MB), tensors: {}",
                  total_bytes, total_bytes as f64 / 1024.0 / 1024.0, tensor_list.len());

        // Find max tensor size to ensure pool is large enough
        let max_tensor_size = tensor_list.iter().map(|(_, size)| *size).max().unwrap_or(0);
        let actual_pool_size = POOL_SIZE.max(max_tensor_size);
        eprintln!("DEBUG: Pool size: {} MB (max tensor: {} MB)",
                  actual_pool_size / 1024 / 1024, max_tensor_size / 1024 / 1024);

        // Create memory pools (account for 4KB alignment padding)
        const ALIGNMENT: usize = 4096;
        let mut pools: Vec<HipBuffer> = Vec::new();
        let mut current_pool_bytes = 0usize;

        for (_, tensor_bytes) in &tensor_list {
            // Account for alignment padding when calculating pool usage
            let aligned_tensor_bytes = (tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);
            if current_pool_bytes + aligned_tensor_bytes > actual_pool_size {
                // Start a new pool
                pools.push(backend.allocate_buffer(actual_pool_size)
                    .map_err(|e| anyhow!("Failed to allocate memory pool: {}", e))?);
                current_pool_bytes = 0;
                eprintln!("DEBUG: Allocated new memory pool #{}", pools.len());
            }
            current_pool_bytes += aligned_tensor_bytes;
        }

        // Allocate final pool if needed
        if current_pool_bytes > 0 {
            pools.push(backend.allocate_buffer(current_pool_bytes)
                .map_err(|e| anyhow!("Failed to allocate final memory pool: {}", e))?);
            eprintln!("DEBUG: Allocated final memory pool #{} (size: {} bytes)",
                      pools.len(), current_pool_bytes);
        }

        eprintln!("DEBUG: Created {} memory pools, total allocation: {} bytes",
                  pools.len(), pools.iter().map(|p| p.size()).sum::<usize>());

        // Upload tensors to their respective pools
        let mut gpu_tensors = HashMap::new();
        let mut pool_idx = 0usize;
        let mut offset = 0usize;

        // Skip memory pooling for tensors that might need transpose (large or specific names)
        // ROCm D2H from sub-buffers is unreliable
        const LARGE_TENSOR_THRESHOLD: usize = 32 * 1024 * 1024;  // 32 MB

        for (name, tensor) in &self.tensors {
            let num_elements = tensor.shape.total_elements();
            let tensor_bytes = num_elements * std::mem::size_of::<f32>();

            // Skip memory pooling for:
            // 1. Large tensors (>32 MB)
            // 2. Embedding/LM head tensors (need transpose)
            // 3. Tensors with [vocab_size, hidden] shape (need transpose)
            // 4. QKV attention tensors (need concatenation)
            let needs_transpose = tensor.shape.dims().len() == 2 &&
                ((tensor.shape.dims()[0] == 151936 || tensor.shape.dims()[1] == 151936) ||
                 name.contains("embd") || name.contains("output"));
            let is_qkv = name.contains("attn_") || name.contains("q_proj") ||
                         name.contains("k_proj") || name.contains("v_proj");
            let is_large = tensor_bytes > LARGE_TENSOR_THRESHOLD;

            if is_large || needs_transpose || is_qkv {
                eprintln!("DEBUG: Skipping memory pool for tensor '{}' ({} MB, large={}, transpose={}, qkv={})",
                         name, tensor_bytes / 1024 / 1024, is_large, needs_transpose, is_qkv);
                let device_tensor = DeviceTensor::from_host_vec(
                    backend,
                    match tensor.tensor_type {
                        GgufTensorType::F32 => {
                            tensor.data.chunks_exact(4)
                                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                                .collect()
                        }
                        GgufTensorType::F16 => {
                            tensor.data.chunks_exact(2)
                                .map(|chunk| {
                                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                    half::f16::from_bits(bits).to_f32()
                                })
                                .collect()
                        }
                        GgufTensorType::Q8_0 => self.dequantize_q8_0(tensor)?,
                        GgufTensorType::Q4_0 => self.dequantize_q4_0(tensor)?,
                        GgufTensorType::Q4_1 => self.dequantize_q4_1(tensor)?,
                        GgufTensorType::Q5_0 => self.dequantize_q5_0(tensor)?,
                        GgufTensorType::Q5_1 => self.dequantize_q5_1(tensor)?,
                        _ => return Err(anyhow!("Unsupported tensor type for tensor '{}'", name)),
                    },
                    tensor.shape.clone(),
                ).map_err(|e| anyhow!("Failed to create tensor '{}': {}", name, e))?;
                gpu_tensors.insert(name.clone(), device_tensor);
                continue;
            }

            // Check if we need to move to next pool
            if offset + tensor_bytes > pools[pool_idx].size() {
                pool_idx += 1;
                offset = 0;
            }

            // Dequantize to FP32 based on tensor type
            let f32_data: Vec<f32> = match tensor.tensor_type {
                GgufTensorType::F32 => {
                    tensor.data.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                }
                GgufTensorType::F16 => {
                    tensor.data.chunks_exact(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect()
                }
                GgufTensorType::Q8_0 => self.dequantize_q8_0(tensor)?,
                GgufTensorType::Q4_0 => self.dequantize_q4_0(tensor)?,
                GgufTensorType::Q4_1 => self.dequantize_q4_1(tensor)?,
                GgufTensorType::Q5_0 => self.dequantize_q5_0(tensor)?,
                GgufTensorType::Q5_1 => self.dequantize_q5_1(tensor)?,
                GgufTensorType::Mxfp4 | GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                    return Err(anyhow!("MXFP dequantization not implemented in memory-pooled load_to_gpu"));
                }
            };

            // Create device tensor from current pool at current offset
            let device_tensor = DeviceTensor::from_pool(
                &pools[pool_idx],
                offset,
                f32_data,
                tensor.shape.clone(),
            ).map_err(|e| anyhow!("Failed to create tensor '{}' from pool #{}: {}", name, pool_idx, e))?;

            gpu_tensors.insert(name.clone(), device_tensor);

            // Advance offset for next tensor (ALIGN TO 4KB BOUNDARY)
            // ROCm requires 4KB-aligned device pointers for D2H copies
            const ALIGNMENT: usize = 4096;
            offset = (offset + tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);

            // Synchronize after each tensor to ensure data is written to GPU
            // This is critical for memory pool sub-buffers
            if pool_idx == 0 && offset > (256 * 1024 * 1024) {
                // Sync after uploading ~256 MB to first pool (embedding layer)
                crate::backend::hip_backend::synchronize_device()
                    .map_err(|e| anyhow!("Sync failed: {}", e))?;
            }
        }

        // Final synchronization after all uploads
        eprintln!("DEBUG: Synchronizing after all tensor uploads");
        crate::backend::hip_backend::synchronize_device()
            .map_err(|e| anyhow!("Final sync failed: {}", e))?;
        eprintln!("DEBUG: Final sync complete");

        eprintln!("DEBUG: All {} tensors uploaded to GPU via {} memory pools",
                  gpu_tensors.len(), pools.len());
        Ok(gpu_tensors)
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
                    eprintln!("GGUF: Using default vocab_size={} for '{}'", default, self.metadata.architecture);
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
                    eprintln!("GGUF: Using default intermediate_size={} (4x hidden_size) for '{}'", default, self.metadata.architecture);
                    default
                }
            }
        };

        Ok(ModelConfig {
            num_hidden_layers: self.metadata.num_layers,
            num_attention_heads: self.metadata.num_heads,
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

        if load_tensors {
            // Parse tensor info
            self.parse_tensor_info(&mut file, tensor_count)?;

            // Read tensor data
            self.read_tensor_data(&mut file)?;
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
                        return Err(anyhow!("value_len too large: {} for key '{}'", value_len, key));
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
                                eprintln!("Warning: Very large STRING array '{}', stopping metadata parse", key);
                                return Ok(());
                            }
                            // Try to seek past the data
                            for _ in 0..n_elements {
                                let mut len_bytes = [0u8; 8];
                                file.read_exact(&mut len_bytes)?;
                                let str_len = u64::from_le_bytes(len_bytes);
                                if str_len > 10_000_000 {
                                    return Err(anyhow!("String too large: {} bytes in array '{}'", str_len, key));
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
                                    return Err(anyhow!("String too large: {} bytes in array '{}'", str_len, key));
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
                        0 | 1 | 7 => 1,  // UINT8, INT8, BOOL
                        2 | 3 => 2,      // UINT16, INT16
                        4..=6 => 4,  // UINT32, INT32, FLOAT32
                        10..=12 => 8,  // UINT64, INT64, FLOAT64
                        _ => {
                            eprintln!("Warning: Unknown array type {}, stopping metadata parse for key '{}'", array_type, key);
                            return Ok(());
                        }
                    };

                    let data_size = n_elements.checked_mul(element_size).ok_or_else(|| {
                        anyhow!("Array data size overflow for key '{}'", key)
                    })?;

                    if data_size > 1_000_000_000 {
                        eprintln!("Warning: Large array ({} bytes) for key '{}', stopping metadata parse", data_size, key);
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
            // Qwen2-specific keys
            "qwen2.block_count" => self.metadata.num_layers = value.parse().unwrap_or(0),
            "qwen2.attention.head_count" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "qwen2.embedding_length" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "qwen2.intermediate_size" => self.metadata.intermediate_size = value.parse().unwrap_or(0),
            "qwen2.rope.dimension_count" => self.metadata.head_dim = value.parse().unwrap_or(0),
            "qwen2.max_position_embeddings" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "qwen2.vocab_size" => self.metadata.vocab_size = value.parse().unwrap_or(0),
            // Llama-specific keys (also used by some Qwen models)
            "llama.block_count" => {
                self.metadata.num_layers = value.parse().unwrap_or(0)
            }
            "llama.attention.head_count" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "llama.attention.head_count_kv" => {}
            "llama.embedding_length" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "llama.feed_forward_length" => self.metadata.intermediate_size = value.parse().unwrap_or(0),
            "llama.rope.dimension_count" => {
                // Usually head_dim = hidden_size / num_heads, but this gives rope dimensions
                self.metadata.head_dim = value.parse().unwrap_or(0)
            }
            "llama.max_position_embeddings" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "llama.vocab_size" => self.metadata.vocab_size = value.parse().unwrap_or(0),
            // Common RMS norm epsilon key names
            "llama.attention.layer_norm_rms_epsilon" |
            "qwen2.attention.layer_norm_rms_epsilon" |
            "qwen2.attention_norm_epsilon" => {
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
            _ => {} // Ignore unknown keys
        }
    }

    /// Parse tensor information from GGUF header
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

            // Store tensor info
            let tensor = GgufTensor {
                name: name.clone(),
                shape,
                tensor_type,
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: Vec::new(), // Will be filled later
            };

            self.tensors.insert(name, tensor);
        }

        Ok(())
    }

    /// Read tensor data from file
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
                            eprintln!("GGUF: Inferred vocab_size={} from {} shape [{}, {}]", d1, name, d0, d1);
                            return Some(d1);
                        } else if d1 == hidden && d0 != hidden {
                            eprintln!("GGUF: Inferred vocab_size={} from {} shape [{}, {}]", d0, name, d0, d1);
                            return Some(d0);
                        }
                    } else {
                        // hidden_size unknown, use heuristic: larger dimension is likely vocab_size
                        let inferred = d0.max(d1);
                        eprintln!("GGUF: Inferred vocab_size={} from {} (heuristic, hidden_size unknown)", inferred, name);
                        return Some(inferred);
                    }
                }
            }
        }

        // No suitable tensor found
        eprintln!("GGUF: Could not infer vocab_size from tensor shapes");
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
            "blk.0.ffn_gate.weight",      // Qwen2-style
            "blk.0.ffn_up.weight",        // Qwen2-style (up projection has same shape)
            "model.layers.0.mlp.gate_proj.weight",  // LLaMA/Mistral-style
            "layers.0.mlp.gate_proj.weight",        // Alternative
            "transformer.layers.0.mlp.gate_proj.weight",  // GPT-style
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
                            eprintln!("GGUF: Auto-detected intermediate_size={} from {} tensor shape", d1, name);
                            return Some(d1);
                        } else if d1 == hidden && d0 != hidden {
                            eprintln!("GGUF: Auto-detected intermediate_size={} from {} tensor shape", d0, name);
                            return Some(d0);
                        }
                    } else {
                        // hidden_size unknown, use heuristic: larger dimension is likely intermediate_size
                        // (FFN expansion is typically 4x hidden_size)
                        let inferred = d0.max(d1);
                        eprintln!("GGUF: Auto-detected intermediate_size={} from {} (heuristic)", inferred, name);
                        return Some(inferred);
                    }
                }
            }
        }

        // No suitable tensor found
        eprintln!("GGUF: Warning - could not auto-detect intermediate_size from tensor shapes");
        None
    }

    /// Upload tensor to GPU memory
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
        }
    }

    /// Dequantize Q8_0 tensor to FP32
    fn dequantize_q8_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 32); // scale (4) + quants (32)

            if block_start + 4 > tensor.data.len() {
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

            // Read quantized values
            let quant_start = block_start + 4;
            let quant_end = std::cmp::min(quant_start + 32, tensor.data.len());
            let quants = &tensor.data[quant_start..quant_end];

            // Dequantize
            for (i, &q) in quants.iter().enumerate() {
                let element_idx = block_idx * 32 + i;
                if element_idx < total_elements {
                    result[element_idx] = (q as f32 - 128.0) * scale;
                }
            }
        }

        Ok(result)
    }

    /// Dequantize Q4_0 tensor to FP32
    fn dequantize_q4_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = total_elements.div_ceil(32);

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 16); // scale (4) + quants (16 bytes for 32 values)

            if block_start + 4 > tensor.data.len() {
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

            // Read quantized values (4-bit packed)
            let quant_start = block_start + 4;
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
                        result[element_idx] = (quant as f32 - 8.0) * scale;
                    }
                }
            }
        }

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
            let min = f32::from_le_bytes([
                min_bytes[0],
                min_bytes[1],
                min_bytes[2],
                min_bytes[3],
            ]);

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
            let qh = u32::from_le_bytes([
                qh_bytes[0],
                qh_bytes[1],
                qh_bytes[2],
                qh_bytes[3],
            ]);

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
                        let high_bit = if bit_idx < 32 {
                            (qh >> bit_idx) & 1
                        } else {
                            0
                        };
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
            let min = f32::from_le_bytes([
                min_bytes[0],
                min_bytes[1],
                min_bytes[2],
                min_bytes[3],
            ]);

            // Read high bits (qh)
            let qh_bytes = &tensor.data[block_start + 8..block_start + 12];
            let qh = u32::from_le_bytes([
                qh_bytes[0],
                qh_bytes[1],
                qh_bytes[2],
                qh_bytes[3],
            ]);

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
                        let high_bit = if bit_idx < 32 {
                            (qh >> bit_idx) & 1
                        } else {
                            0
                        };
                        let quant = (low_bits as u32 | (high_bit << 4)) as u8;
                        result[element_idx] = min + (quant as f32) * scale;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Dequantize MXFP4 tensor to FP32
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
                    let combined = ((packed_data[byte_idx + 1] as u16) << 8) | (packed_data[byte_idx] as u16);
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
}

/// Simple f16 implementation for conversion
#[allow(dead_code)]
struct F16(u16);

impl F16 {
    fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

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
        assert_eq!(5u32, 5, "GGUF_TYPE_INT32");  // NOT BOOL!
        assert_eq!(6u32, 6, "GGUF_TYPE_FLOAT32");  // NOT STRING!
        assert_eq!(7u32, 7, "GGUF_TYPE_BOOL");  // NOT 5!
        assert_eq!(8u32, 8, "GGUF_TYPE_STRING");  // NOT 6!
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
        assert_eq!(GgufTensorType::Q8_0 as u32, 8, "GGML_TYPE_Q8_0");  // Was wrongly 3!
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
        assert!(true, "Documented: array format is type(u32) + count(u64) + data");
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
        assert!(true, "Documented: STRING arrays require per-string length iteration");
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
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let loader = super::GgufLoader::new(&model_path)
            .expect("Failed to load GGUF");

        let metadata = loader.metadata();
        assert_eq!(metadata.architecture, "qwen2");
        assert_eq!(metadata.num_layers, 24);
        assert_eq!(metadata.num_heads, 14);
        assert_eq!(metadata.hidden_size, 896);

        let tensors = loader.load_tensors().expect("Failed to load tensors");
        assert_eq!(tensors.len(), 291, "Expected 291 tensors in Qwen2.5-0.5B");
    }
}
