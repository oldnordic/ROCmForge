//! GGUF tensor type definitions

use anyhow::Result;
use serde::Serialize;
use std::fmt;

/// GGUF tensor types (ggml_type enum values from ggml.h)
#[derive(Debug, Clone, PartialEq, Copy, Serialize)]
#[repr(u8)]
pub enum GgufTensorType {
    F32 = 0,   // GGML_TYPE_F32
    F16 = 1,   // GGML_TYPE_F16
    Q4_0 = 2,  // GGML_TYPE_Q4_0
    Q4_1 = 3,  // GGML_TYPE_Q4_1
    Q5_0 = 6,  // GGML_TYPE_Q5_0
    Q5_1 = 7,  // GGML_TYPE_Q5_1
    Q8_0 = 8,  // GGML_TYPE_Q8_0
    Q2_K = 10, // GGML_TYPE_Q2_K (K-quants)
    Q3_K = 11, // GGML_TYPE_Q3_K
    Q4_K = 12, // GGML_TYPE_Q4_K
    Q5_K = 13, // GGML_TYPE_Q5_K
    Q6_K = 14, // GGML_TYPE_Q6_K

    // MXFP types (OCP MX Specification v1.0)
    // Using enum values 20-22 to avoid conflicts with future ggml types
    Mxfp4 = 20,     // OCP MXFP4-E2M1 (4-bit)
    Mxfp6E2m3 = 21, // OCP MXFP6-E2M3 (6-bit, recommended)
    Mxfp6E3m2 = 22, // OCP MXFP6-E3M2 (6-bit)
}

impl GgufTensorType {
    /// Parse tensor type from u32 value
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            2 => Ok(GgufTensorType::Q4_0),
            3 => Ok(GgufTensorType::Q4_1),
            6 => Ok(GgufTensorType::Q5_0),
            7 => Ok(GgufTensorType::Q5_1),
            8 => Ok(GgufTensorType::Q8_0),
            10 => Ok(GgufTensorType::Q2_K),
            11 => Ok(GgufTensorType::Q3_K),
            12 => Ok(GgufTensorType::Q4_K),
            13 => Ok(GgufTensorType::Q5_K),
            14 => Ok(GgufTensorType::Q6_K),
            20 => Ok(GgufTensorType::Mxfp4),
            21 => Ok(GgufTensorType::Mxfp6E2m3),
            22 => Ok(GgufTensorType::Mxfp6E3m2),
            _ => Err(anyhow::anyhow!("Unknown tensor type: {}", value)),
        }
    }

    /// Get string representation of this tensor type
    pub fn to_string(&self) -> &'static str {
        match self {
            GgufTensorType::F32 => "FP32",
            GgufTensorType::F16 => "FP16",
            GgufTensorType::Q4_0 => "Q4_0",
            GgufTensorType::Q4_1 => "Q4_1",
            GgufTensorType::Q5_0 => "Q5_0",
            GgufTensorType::Q5_1 => "Q5_1",
            GgufTensorType::Q8_0 => "Q8_0",
            GgufTensorType::Q2_K => "Q2_K",
            GgufTensorType::Q3_K => "Q3_K",
            GgufTensorType::Q4_K => "Q4_K",
            GgufTensorType::Q5_K => "Q5_K",
            GgufTensorType::Q6_K => "Q6_K",
            GgufTensorType::Mxfp4 => "MXFP4",
            GgufTensorType::Mxfp6E2m3 => "MXFP6_E2M3",
            GgufTensorType::Mxfp6E3m2 => "MXFP6_E3M2",
        }
    }

    /// Get the block size (number of elements per block) for block-quantized types
    pub fn block_size(&self) -> usize {
        match self {
            GgufTensorType::Q4_0 => 32,
            GgufTensorType::Q4_1 => 32,
            GgufTensorType::Q5_0 => 32,
            GgufTensorType::Q5_1 => 32,
            GgufTensorType::Q8_0 => 32,
            GgufTensorType::Q2_K => 256,
            GgufTensorType::Q3_K => 256,
            GgufTensorType::Q4_K => 256,
            GgufTensorType::Q5_K => 256,
            GgufTensorType::Q6_K => 256,
            GgufTensorType::Mxfp4 | GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => 32,
            GgufTensorType::F32 | GgufTensorType::F16 => 1,
        }
    }

    /// Check if this is a quantized type
    pub fn is_quantized(&self) -> bool {
        !matches!(self, GgufTensorType::F32 | GgufTensorType::F16)
    }

    /// Get the element size in bytes for this tensor type.
    /// For block-quantized types, returns the block size.
    pub fn element_size(&self) -> usize {
        match self {
            GgufTensorType::F32 => 4,
            GgufTensorType::F16 => 2,
            // Block-based quantization types return their block size
            GgufTensorType::Q4_0 => 32,
            GgufTensorType::Q4_1 => 32,
            GgufTensorType::Q5_0 => 32,
            GgufTensorType::Q5_1 => 32,
            GgufTensorType::Q8_0 => 32,
            GgufTensorType::Q2_K => 256,
            GgufTensorType::Q3_K => 256,
            GgufTensorType::Q4_K => 256,
            GgufTensorType::Q5_K => 256,
            GgufTensorType::Q6_K => 256,
            GgufTensorType::Mxfp4 => 32,
            GgufTensorType::Mxfp6E2m3 => 32,
            GgufTensorType::Mxfp6E3m2 => 32,
        }
    }
}

impl fmt::Display for GgufTensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type_from_u32() {
        assert_eq!(GgufTensorType::from_u32(0).unwrap(), GgufTensorType::F32);
        assert_eq!(GgufTensorType::from_u32(1).unwrap(), GgufTensorType::F16);
        assert_eq!(GgufTensorType::from_u32(2).unwrap(), GgufTensorType::Q4_0);
        assert_eq!(GgufTensorType::from_u32(8).unwrap(), GgufTensorType::Q8_0);
        assert!(GgufTensorType::from_u32(99).is_err());
    }

    #[test]
    fn test_block_size() {
        assert_eq!(GgufTensorType::Q4_0.block_size(), 32);
        assert_eq!(GgufTensorType::Q2_K.block_size(), 256);
        assert_eq!(GgufTensorType::F32.block_size(), 1);
    }

    #[test]
    fn test_is_quantized() {
        assert!(GgufTensorType::Q4_0.is_quantized());
        assert!(GgufTensorType::Mxfp4.is_quantized());
        assert!(!GgufTensorType::F32.is_quantized());
        assert!(!GgufTensorType::F16.is_quantized());
    }
}
