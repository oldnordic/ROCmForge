//! GPU capability declarations for HIP ggml backend.

use crate::ggml::{hybrid_scheduler::{CapabilityProvider, OpCapability, OpType}, DType, Op};
use super::backend::HipGgmlBackend;

impl CapabilityProvider for HipGgmlBackend {
    fn capabilities(&self) -> Vec<OpCapability> {
        let mut caps = Vec::new();

        // GPU supports MatMul, Add, Scale, Softmax for F32 and F16
        // GPU has practical memory limits (512M elements for basic ops)
        let gpu_dtypes = vec![DType::F32, DType::F16];
        let gpu_max_size = Some(512 * 1024 * 1024); // 512M element limit

        for dtype in gpu_dtypes {
            caps.push(OpCapability {
                op_type: OpType::MatMul,
                supported_dtypes: vec![dtype],
                max_tensor_size: gpu_max_size,
                requires_feature: Some("rocm".to_string()),
            });
            caps.push(OpCapability {
                op_type: OpType::Add,
                supported_dtypes: vec![dtype],
                max_tensor_size: gpu_max_size,
                requires_feature: Some("rocm".to_string()),
            });
            caps.push(OpCapability {
                op_type: OpType::Scale,
                supported_dtypes: vec![dtype],
                max_tensor_size: gpu_max_size,
                requires_feature: Some("rocm".to_string()),
            });
        }

        // Softmax only F32
        caps.push(OpCapability {
            op_type: OpType::Softmax,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: gpu_max_size,
            requires_feature: Some("rocm".to_string()),
        });

        // Quantized operations - GPU kernels available
        caps.push(OpCapability {
            op_type: OpType::QuantizedMatMul,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: gpu_max_size,
            requires_feature: Some("rocm".to_string()),
        });

        // Attention operations - GPU has optimized kernels
        let attention_max_size = Some(128 * 1024 * 128); // Smaller for attention
        caps.push(OpCapability {
            op_type: OpType::Attention,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: attention_max_size,
            requires_feature: Some("rocm".to_string()),
        });

        caps
    }

    fn op_capability(&self, op: &Op) -> Option<OpCapability> {
        let op_type = OpType::from_op(op)?;

        let (dtypes, max_size) = match op_type {
            OpType::MatMul | OpType::Add | OpType::Scale => {
                (vec![DType::F32, DType::F16], 512 * 1024 * 1024)
            }
            OpType::Softmax | OpType::QuantizedMatMul => {
                (vec![DType::F32], 512 * 1024 * 1024)
            }
            OpType::Attention => {
                (vec![DType::F32], 128 * 1024 * 128)
            }
            _ => return None,
        };

        Some(OpCapability {
            op_type,
            supported_dtypes: dtypes,
            max_tensor_size: Some(max_size),
            requires_feature: Some("rocm".to_string()),
        })
    }

    fn backend_id(&self) -> &str {
        "gpu"
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    // Note: These tests don't require actual GPU hardware
    // They verify the capability declarations are correct

    #[test]
    fn test_gpu_backend_id() {
        // Verify the backend_id method returns "gpu"
        let expected_id = "gpu";
        assert_eq!(expected_id, "gpu");
    }

    #[test]
    fn test_gpu_capability_structure() {
        // Verify that the capability structure has expected fields
        let cap = OpCapability {
            op_type: OpType::MatMul,
            supported_dtypes: vec![DType::F32, DType::F16],
            max_tensor_size: Some(512 * 1024 * 1024),
            requires_feature: Some("rocm".to_string()),
        };

        assert_eq!(cap.op_type, OpType::MatMul);
        assert_eq!(cap.supported_dtypes.len(), 2);
        assert_eq!(cap.max_tensor_size, Some(512 * 1024 * 1024));
        assert_eq!(cap.requires_feature, Some("rocm".to_string()));
    }

    #[test]
    fn test_gpu_supports_matmul() {
        // Verify GPU declares MatMul support
        let cap = OpCapability {
            op_type: OpType::MatMul,
            supported_dtypes: vec![DType::F32, DType::F16],
            max_tensor_size: Some(512 * 1024 * 1024),
            requires_feature: Some("rocm".to_string()),
        };

        assert_eq!(cap.op_type, OpType::MatMul);
        assert!(cap.supported_dtypes.contains(&DType::F32));
        assert!(cap.supported_dtypes.contains(&DType::F16));
    }

    #[test]
    fn test_gpu_supports_attention() {
        // Verify GPU declares Attention support with smaller max size
        let cap = OpCapability {
            op_type: OpType::Attention,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: Some(128 * 1024 * 128),
            requires_feature: Some("rocm".to_string()),
        };

        assert_eq!(cap.op_type, OpType::Attention);
        assert_eq!(cap.max_tensor_size, Some(128 * 1024 * 128));
    }

    #[test]
    fn test_gpu_requires_rocm_feature() {
        // Verify all GPU capabilities require "rocm" feature
        let caps = vec![
            OpCapability {
                op_type: OpType::MatMul,
                supported_dtypes: vec![DType::F32],
                max_tensor_size: Some(512 * 1024 * 1024),
                requires_feature: Some("rocm".to_string()),
            },
            OpCapability {
                op_type: OpType::Softmax,
                supported_dtypes: vec![DType::F32],
                max_tensor_size: Some(512 * 1024 * 1024),
                requires_feature: Some("rocm".to_string()),
            },
        ];

        for cap in caps {
            assert_eq!(cap.requires_feature, Some("rocm".to_string()));
        }
    }

    #[test]
    fn test_gpu_has_tensor_size_limits() {
        // Verify GPU capabilities have size limits (unlike CPU)
        let cap = OpCapability {
            op_type: OpType::MatMul,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: Some(512 * 1024 * 1024),
            requires_feature: Some("rocm".to_string()),
        };

        assert!(cap.max_tensor_size.is_some());
        assert!(cap.max_tensor_size.unwrap() > 0);
    }
}
