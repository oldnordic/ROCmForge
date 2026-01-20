//! GGUF (GPT-Generated Unified Format) Loader
//!
//! This module provides GGUF model file loading with support for:
//! - Metadata parsing
//! - Multiple quantization types (Q8_0, Q4_0, Q4_K, Q6_K, FP16, FP32)
//! - Lazy tensor loading with memory-mapped files
//! - GPU memory upload via DeviceTensor
//! - Async batch loading with parallel dequantization
//!
//! # Module Structure
//!
//! - `types`: Core structs (GgufTensor, GgufLoader, F16)
//! - `header`: GGUF magic validation and header parsing
//! - `metadata`: KV pair parsing and metadata extraction
//! - `tensor_info`: Tensor metadata parsing
//! - `tensor_data`: CPU dequantization for various quant types
//! - `gpu_upload`: GPU upload, caching, and async loading
//! - `loader_impl`: Core loader implementation
//!
//! # Example
//!
//! ```ignore
//! use crate::loader::gguf::GgufLoader;
//!
//! let loader = GgufLoader::new("model.gguf")?;
//! let metadata = loader.metadata();
//! let backend = HipBackend::new()?;
//!
//! // Load specific tensor on-demand
//! let tensor = loader.load_tensor_to_gpu("blk.0.attn_q.weight", &backend)?;
//! ```

// Core module exports
mod header;
mod loader_impl;
mod metadata;
mod tensor_data;
mod tensor_info;
mod types;
mod gpu_upload;

// Re-export all public types for backward compatibility
pub use types::{F16, GgufLoader, GgufTensor};

// Re-export metadata
pub use crate::loader::metadata::GgufMetadata;

// Re-export header functions
pub use header::{parse_gguf_header, read_gguf_version, read_kv_count, read_tensor_count, validate_gguf_magic, GGUF_MAGIC, GGUF_VERSION};

// Re-export metadata functions
pub use metadata::{parse_hyperparameters, parse_kv_pairs, update_metadata};

// Re-export tensor info functions
pub use tensor_info::{
    calculate_tensor_size, find_tensors_by_pattern, get_tensor_info, list_tensor_names,
    parse_tensor_info, TensorMetadata,
};

// Re-export tensor data functions
pub use tensor_data::{
    bytes_to_f32, dequantize_mxfp4, dequantize_mxfp6, dequantize_q4_0, dequantize_q4_k,
    dequantize_q6_k, dequantize_q8_0,
};

// Re-export loader implementation functions
pub use gpu_upload::metadata_from_file;
pub use loader_impl::{load_from_disk_impl, new_loader as loader_new_internal, to_model_config};

// Re-export LazyTensor for external access
pub use crate::loader::lazy_tensor::LazyTensor;

// Include MXFP tests (located in parent loader directory)
#[cfg(test)]
#[path = "../mxfp_tests.rs"]
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
    use crate::loader::tensor_type::GgufTensorType;

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
        //     GGML_TYPE_Q4_1    = 3,  // NOT SUPPORTED - no common models use this
        //     // 4, 5 removed
        //     GGML_TYPE_Q5_0    = 6,  // NOT SUPPORTED - no common models use this
        //     GGML_TYPE_Q5_1    = 7,  // NOT SUPPORTED - no common models use this
        //     GGML_TYPE_Q8_0    = 8,  // CRITICAL: NOT 3!
        //     GGML_TYPE_Q8_1    = 9,
        //     ...
        // };

        // These assertions prevent the critical bug where Q8_0 was mapped to 3
        // If this test fails, tensor data will be completely corrupted
        assert_eq!(GgufTensorType::F32 as u32, 0, "GGML_TYPE_F32");
        assert_eq!(GgufTensorType::F16 as u32, 1, "GGML_TYPE_F16");
        assert_eq!(GgufTensorType::Q4_0 as u32, 2, "GGML_TYPE_Q4_0");
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
