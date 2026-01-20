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
// Migrated to kernel module in 26-01 - use kernel exports directly
pub use dequant::{
    dequant_q2_k, dequant_q3_k,
    dequant_q5_k,
    dequant_mxfp4, dequant_mxfp6, dequantize,
};

// Re-export CPU dequant functions from kernels module
// Using aliases to preserve local function names for backward compatibility
pub use crate::kernels::quant::dequantize_q4_0_cpu as dequant_q4_0;
pub use crate::kernels::quant::dequantize_q4_k_cpu as dequant_q4_k;
pub use crate::kernels::quant::dequantize_q6_k_cpu as dequant_q6_k;
pub use crate::kernels::quant::dequantize_q8_0 as dequant_q8_0;

pub use lazy_tensor::*;
pub use mmap::*;
pub use mmap_loader::*;
pub use onnx_loader::*;
