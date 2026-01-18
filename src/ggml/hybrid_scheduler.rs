//! Hybrid execution scheduler for automatic CPU/GPU operation selection

use crate::ggml::{GgmlBackend, GgmlResult, Op, DType};

/// Capability descriptor for a backend operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpCapability {
    pub op_type: OpType,
    pub supported_dtypes: Vec<DType>,
    pub max_tensor_size: Option<usize>,
    pub requires_feature: Option<String>, // e.g., "rocm", "simd"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    QuantizedMatMul,
    Add,
    Scale,
    Softmax,
    Attention,
    Dequantize,
    // Add more as needed
}

/// Trait for backends that declare their operation capabilities
pub trait CapableBackend: GgmlBackend {
    /// Get all operations this backend can execute
    fn capabilities(&self) -> Vec<OpCapability>;

    /// Check if this backend can execute a specific operation
    fn can_execute(&self, op: &Op) -> bool {
        self.op_capability(op).is_some()
    }

    /// Get the capability requirement for an operation
    fn op_capability(&self, op: &Op) -> Option<OpCapability>;
}

/// Cost estimate for executing an operation on a backend
#[derive(Debug, Clone, Copy)]
pub struct OpCost {
    pub estimated_us: u64,        // Estimated execution time in microseconds
    pub memory_bytes: usize,       // Estimated memory usage
    pub transfer_cost: Option<u64>, // Data transfer cost (for heterogeneous execution)
}

impl OpCapability {
    /// Create a new capability descriptor
    pub fn new(op_type: OpType) -> Self {
        Self {
            op_type,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: None,
            requires_feature: None,
        }
    }

    /// Add supported data types
    pub fn with_dtypes(mut self, dtypes: Vec<DType>) -> Self {
        self.supported_dtypes = dtypes;
        self
    }

    /// Set maximum tensor size
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_tensor_size = Some(size);
        self
    }

    /// Set required feature
    pub fn with_feature(mut self, feature: &str) -> Self {
        self.requires_feature = Some(feature.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_capability_builder() {
        let cap = OpCapability::new(OpType::MatMul)
            .with_dtypes(vec![DType::F32])
            .with_max_size(1024 * 1024)
            .with_feature("rocm");

        assert_eq!(cap.op_type, OpType::MatMul);
        assert_eq!(cap.supported_dtypes, vec![DType::F32]);
        assert_eq!(cap.max_tensor_size, Some(1024 * 1024));
        assert_eq!(cap.requires_feature, Some("rocm".to_string()));
    }

    #[test]
    fn test_op_capability_vec() {
        let caps = vec![
            OpCapability::new(OpType::MatMul),
            OpCapability::new(OpType::Softmax),
        ];

        assert_eq!(caps.len(), 2);
        assert!(caps.iter().any(|c| c.op_type == OpType::MatMul));
    }
}
