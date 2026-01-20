//! GGUF Tensor Information Parsing
//!
//! This module handles tensor metadata parsing from GGUF files:
//! - Reading tensor info (name, shape, type, offset)
//! - Creating LazyTensor handles for lazy loading
//! - Calculating tensor data sizes

use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::tensor_type::GgufTensorType;
use crate::loader::{GgufTensor, TensorShape};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

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
///
/// # Arguments
///
/// * `file` - File handle positioned after KV pairs
/// * `tensor_count` - Number of tensors to parse
///
/// # Returns
///
/// * `HashMap<String, GgufTensor>` - Legacy tensor entries (empty data)
/// * `HashMap<String, LazyTensor>` - Lazy tensor handles
pub fn parse_tensor_info(
    file: &mut File,
    tensor_count: u64,
) -> Result<(HashMap<String, GgufTensor>, HashMap<String, LazyTensor>)> {
    let mut tensors = HashMap::new();
    let mut lazy_tensors = HashMap::new();

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
        let size = calculate_tensor_size(&shape, tensor_type);

        // Create LazyTensor handle (Phase 1: metadata only, no data loaded)
        let lazy_tensor = LazyTensor::unloaded(name.clone(), offset, size, dims.clone(), tensor_type);
        lazy_tensors.insert(name.clone(), lazy_tensor);

        // Store legacy GgufTensor for backward compatibility
        let tensor = GgufTensor {
            name: name.clone(),
            shape,
            tensor_type,
            quant_type: tensor_type.to_string().to_string(),
            offset,
            data: Vec::new(), // Phase 1: NOT filled - data loaded on-demand instead
        };

        tensors.insert(name, tensor);
    }

    Ok((tensors, lazy_tensors))
}

/// Calculate tensor data size in bytes based on type and shape
pub fn calculate_tensor_size(shape: &TensorShape, tensor_type: GgufTensorType) -> usize {
    match tensor_type {
        GgufTensorType::F32 => shape.total_elements() * 4,
        GgufTensorType::F16 => shape.total_elements() * 2,
        GgufTensorType::Q4_0 => {
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
    }
}

/// Tensor metadata extracted during parsing
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: TensorShape,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
    pub size_bytes: usize,
}

impl TensorMetadata {
    pub fn from_gguf_tensor(tensor: &GgufTensor) -> Self {
        Self {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            tensor_type: tensor.tensor_type,
            offset: tensor.offset,
            size_bytes: tensor.data_size(),
        }
    }
}

/// Get tensor info for a specific tensor by name
pub fn get_tensor_info(
    tensors: &HashMap<String, GgufTensor>,
    name: &str,
) -> Option<TensorMetadata> {
    tensors.get(name).map(TensorMetadata::from_gguf_tensor)
}

/// List all tensor names
pub fn list_tensor_names(tensors: &HashMap<String, GgufTensor>) -> Vec<String> {
    let mut names: Vec<String> = tensors.keys().cloned().collect();
    names.sort(); // Sort for predictable ordering
    names
}

/// Get tensors matching a pattern
///
/// Simple glob-style matching with * wildcard
pub fn find_tensors_by_pattern(
    tensors: &HashMap<String, GgufTensor>,
    pattern: &str,
) -> Vec<String> {
    list_tensor_names(tensors)
        .into_iter()
        .filter(|name| match_pattern(name, pattern))
        .collect()
}

/// Simple glob pattern matching
fn match_pattern(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if pattern.ends_with("*") {
        let prefix = &pattern[..pattern.len() - 1];
        return name.starts_with(prefix);
    }
    if pattern.starts_with("*") {
        let suffix = &pattern[1..];
        return name.ends_with(suffix);
    }
    name == pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tensor_size_f32() {
        let shape = TensorShape::from_dims(&[100, 200]);
        let size = calculate_tensor_size(&shape, GgufTensorType::F32);
        assert_eq!(size, 100 * 200 * 4);
    }

    #[test]
    fn test_calculate_tensor_size_f16() {
        let shape = TensorShape::from_dims(&[100, 200]);
        let size = calculate_tensor_size(&shape, GgufTensorType::F16);
        assert_eq!(size, 100 * 200 * 2);
    }

    #[test]
    fn test_calculate_tensor_size_q4_0() {
        let shape = TensorShape::from_dims(&[32]); // Exactly one block
        let size = calculate_tensor_size(&shape, GgufTensorType::Q4_0);
        assert_eq!(size, 4 + 32); // scale (4) + quants (32)
    }

    #[test]
    fn test_calculate_tensor_size_q8_0() {
        let shape = TensorShape::from_dims(&[32]); // Exactly one block
        let size = calculate_tensor_size(&shape, GgufTensorType::Q8_0);
        assert_eq!(size, 4 + 32); // scale (4) + quants (32)
    }

    #[test]
    fn test_calculate_tensor_size_k_quants() {
        let shape = TensorShape::from_dims(&[256]); // Exactly one block
        let size = calculate_tensor_size(&shape, GgufTensorType::Q4_K);
        assert_eq!(size, 256); // Fixed 256 bytes per block
    }

    #[test]
    fn test_match_pattern() {
        assert!(match_pattern("blk.0.attn_q.weight", "blk.0.attn_q.weight"));
        assert!(match_pattern("blk.0.attn_q.weight", "blk.0*"));
        assert!(match_pattern("blk.0.attn_q.weight", "*.weight"));
        assert!(!match_pattern("blk.0.attn_q.weight", "blk.1*"));
        assert!(match_pattern("anything", "*"));
    }

    #[test]
    fn test_find_tensors_by_pattern() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            GgufTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: TensorShape::from_dims(&[100, 200]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 0,
                data: vec![],
            },
        );
        tensors.insert(
            "blk.0.attn_k.weight".to_string(),
            GgufTensor {
                name: "blk.0.attn_k.weight".to_string(),
                shape: TensorShape::from_dims(&[100, 200]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 0,
                data: vec![],
            },
        );
        tensors.insert(
            "blk.1.attn_q.weight".to_string(),
            GgufTensor {
                name: "blk.1.attn_q".to_string(),
                shape: TensorShape::from_dims(&[100, 200]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 0,
                data: vec![],
            },
        );

        let results = find_tensors_by_pattern(&tensors, "blk.0*");
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"blk.0.attn_q.weight".to_string()));
        assert!(results.contains(&"blk.0.attn_k.weight".to_string()));

        let results = find_tensors_by_pattern(&tensors, "*.attn_q.weight");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_list_tensor_names() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "c.weight".to_string(),
            GgufTensor {
                name: "c.weight".to_string(),
                shape: TensorShape::from_dims(&[1]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 0,
                data: vec![],
            },
        );
        tensors.insert(
            "a.weight".to_string(),
            GgufTensor {
                name: "a.weight".to_string(),
                shape: TensorShape::from_dims(&[1]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 0,
                data: vec![],
            },
        );
        tensors.insert(
            "b.weight".to_string(),
            GgufTensor {
                name: "b.weight".to_string(),
                shape: TensorShape::from_dims(&[1]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 0,
                data: vec![],
            },
        );

        let names = list_tensor_names(&tensors);
        assert_eq!(names, vec!["a.weight", "b.weight", "c.weight"]);
    }

    #[test]
    fn test_get_tensor_info() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "test.weight".to_string(),
            GgufTensor {
                name: "test.weight".to_string(),
                shape: TensorShape::from_dims(&[100, 200]),
                tensor_type: GgufTensorType::F32,
                quant_type: "F32".to_string(),
                offset: 12345,
                data: vec![],
            },
        );

        let info = get_tensor_info(&tensors, "test.weight");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "test.weight");
        assert_eq!(info.offset, 12345);
        assert_eq!(info.size_bytes, 100 * 200 * 4);

        let info = get_tensor_info(&tensors, "missing");
        assert!(info.is_none());
    }
}
