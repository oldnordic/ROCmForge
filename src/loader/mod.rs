//! Model loaders module

pub mod gguf;
pub mod lazy_tensor;
pub mod mmap;
pub mod mmap_loader;
pub mod onnx_loader;

// GGUF submodules (03-03: modularization)
mod mxfp;
mod tensor_type;
mod metadata;
mod gguf_tensor;
mod dequant;

// Re-export GGUF types
pub use gguf::{GgufLoader, F16};
pub use mxfp::{E8M0, MxfpBlock};
pub use tensor_type::GgufTensorType;
pub use metadata::GgufMetadata;
pub use gguf_tensor::GgufTensor;

// Re-export dequantization functions for benchmarks
// Note: Q4_1, Q5_0, Q5_1 removed in 23-02 (unused quantization formats)
pub use dequant::{
    dequant_q2_k, dequant_q3_k, dequant_q4_0, dequant_q4_k,
    dequant_q5_k, dequant_q6_k, dequant_q8_0,
    dequant_mxfp4, dequant_mxfp6, dequantize,
};

pub use lazy_tensor::*;
pub use mmap::*;
pub use mmap_loader::*;
pub use onnx_loader::*;
