//! GGUF tensor descriptor

use super::tensor_type::GgufTensorType;
use crate::loader::TensorShape;

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
    /// Create a new GGUF tensor
    pub fn new(
        name: String,
        shape: TensorShape,
        tensor_type: GgufTensorType,
        offset: u64,
    ) -> Self {
        Self {
            name,
            shape,
            tensor_type,
            quant_type: tensor_type.to_string().to_string(),
            offset,
            data: Vec::new(),
        }
    }

    /// Calculate total number of elements
    pub fn total_elements(&self) -> usize {
        self.shape.total_elements()
    }

    /// Calculate data size in bytes based on tensor type
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
                let blocks = self.total_elements().div_ceil(256);
                blocks.checked_mul(256).unwrap_or(usize::MAX)
            }
            GgufTensorType::Mxfp4 => {
                // MXFP4: block_size=32, each block has 1 scale (E8M0) + 32*4 bits data
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(1 + 16).unwrap_or(usize::MAX)
            }
            GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => {
                // MXFP6: block_size=32, each block has 1 scale (E8M0) + 32*6 bits data
                let blocks = self.total_elements().div_ceil(32);
                blocks.checked_mul(1 + 24).unwrap_or(usize::MAX)
            }
        }
    }

    /// Load tensor data from file at current offset
    pub fn load_from_file(&mut self, file: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::{Read, Seek, SeekFrom};

        file.seek(SeekFrom::Start(self.offset))?;
        let data_size = self.data_size();
        self.data.resize(data_size, 0);
        file.read_exact(&mut self.data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::from_dims(&[128, 256]);
        let tensor = GgufTensor::new(
            "test.weight".to_string(),
            shape,
            GgufTensorType::F32,
            1024,
        );
        assert_eq!(tensor.name, "test.weight");
        assert_eq!(tensor.total_elements(), 128 * 256);
    }

    #[test]
    fn test_data_size_f32() {
        let shape = TensorShape::from_dims(&[100, 100]);
        let tensor = GgufTensor::new("test".to_string(), shape, GgufTensorType::F32, 0);
        assert_eq!(tensor.data_size(), 100 * 100 * 4);
    }

    #[test]
    fn test_data_size_q4_0() {
        let shape = TensorShape::from_dims(&[32, 32]);
        let tensor = GgufTensor::new("test".to_string(), shape, GgufTensorType::Q4_0, 0);
        // 32*32 = 1024 elements, /32 = 32 blocks, * (4+32) = 1152 bytes
        assert_eq!(tensor.data_size(), 32 * (4 + 32));
    }
}
