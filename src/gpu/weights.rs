//! GPU weight storage in VRAM.
//!
//! Safety-first design:
//! - All hipMalloc/hipMemcpy calls wrapped with error checking
//! - RAII cleanup on Drop prevents VRAM leaks
//! - Never panic, always return GpuError

use super::error::{GpuError, GpuResult};
use super::ffi;
use crate::loader::{GgmlType, GgufFile, TensorDesc};
use std::ptr::NonNull;

// ── Weight Metadata ────────────────────────────────────────────────────────────

/// Metadata for a weight tensor on GPU.
///
/// Same as CPU WeightMeta - quantization type and dimensions.
#[derive(Clone, Debug)]
pub struct WeightMeta {
    /// Quantization type (F32, Q4_0, Q4_1, Q8_0, etc.)
    pub wtype: GgmlType,
    /// Dimensions from GGUF (innermost first)
    pub dims: Vec<u64>,
    /// Whether this weight tensor needs transposed access
    pub needs_transpose: bool,
}

impl WeightMeta {
    /// Create metadata from a GGUF tensor descriptor.
    pub fn from_desc(desc: &TensorDesc, needs_transpose: bool) -> Self {
        Self {
            wtype: desc.ggml_type,
            dims: desc.dims.clone(),
            needs_transpose,
        }
    }

    /// Total size in bytes for this weight tensor.
    pub fn byte_size(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_desc_works() {
        let desc = TensorDesc {
            name: "test.weight".to_string(),
            ggml_type: GgmlType::Q4_0,
            dims: vec![1024, 768],
            offset: 0,
        };
        let meta = WeightMeta::from_desc(&desc, false);
        assert_eq!(meta.wtype, GgmlType::Q4_0);
        assert_eq!(meta.dims, vec![1024, 768]);
        assert_eq!(meta.byte_size(), 1024 * 768);
        assert_eq!(meta.num_elements(), 1024 * 768);
    }

    #[test]
    fn byte_size_calculates_correctly() {
        let meta = WeightMeta {
            wtype: GgmlType::F32,
            dims: vec![100, 200],
            needs_transpose: false,
        };
        assert_eq!(meta.byte_size(), 100 * 200);
    }
}
