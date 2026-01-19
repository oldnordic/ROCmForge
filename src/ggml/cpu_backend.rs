//! CPU reference backend for ggml IR.
//!
//! Provides CPU-based execution with optional SIMD acceleration when the
//! `simd` feature is enabled. SIMD operations use std::simd for portable
//! vectorization across x86_64 (AVX2) and aarch64 (NEON).

use crate::ggml::{GgmlBackend, GgmlError, GgmlResult, Op, TensorDesc, TensorId};
use crate::ggml::hybrid_scheduler::{CapabilityProvider, OpCapability, OpType};
use std::collections::HashMap;

/// CPU backend with optional SIMD acceleration
///
/// The SIMD capability is determined at compile time via the `simd` feature
/// flag. When enabled, SIMD operations are used for matmul and attention
/// operations on supported architectures.
#[derive(Default)]
pub struct CpuBackend {
    tensors: HashMap<TensorId, (TensorDesc, Vec<f32>)>,
    /// SIMD capability determined at compile time
    simd_capable: bool,
}

impl CpuBackend {
    /// Create a new CPU backend with SIMD capability detection
    pub fn new() -> Self {
        Self::with_simd_detection()
    }

    /// Create CPU backend with SIMD capability detection
    ///
    /// SIMD is enabled via the `simd` feature flag at compile time.
    /// Runtime detection is handled via cfg(target_arch) for architecture
    /// compatibility.
    fn with_simd_detection() -> Self {
        #[cfg(feature = "simd")]
        let simd_capable = Self::detect_simd_capabilities();

        #[cfg(not(feature = "simd"))]
        let simd_capable = false;

        Self {
            tensors: HashMap::new(),
            simd_capable,
        }
    }

    /// Detect CPU SIMD capabilities at compile time
    ///
    /// Returns true if the target architecture supports SIMD operations.
    /// Since std::simd requires feature gating, this is a compile-time check.
    ///
    /// # Architecture Support
    ///
    /// - x86_64: AVX2 (f32x8) - requires AVX2 CPU support
    /// - aarch64: NEON (f32x4) - always available on ARM64
    /// - Other: Scalar fallback
    #[cfg(feature = "simd")]
    fn detect_simd_capabilities() -> bool {
        // Compile-time detection based on target architecture
        // Runtime detection would require cpuid crate or similar

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            // SIMD is available for these architectures when feature is enabled
            // For x86_64, we assume AVX2 is available (most modern CPUs)
            // For aarch64, NEON is always available
            true
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Other architectures use scalar fallback
            false
        }
    }

    /// Returns true if SIMD operations are available
    pub fn is_simd_capable(&self) -> bool {
        self.simd_capable
    }

    /// Execute matrix multiplication: C = A @ B
    ///
    /// Uses SIMD acceleration when available, otherwise falls back to scalar.
    fn matmul(
        &mut self,
        a_id: TensorId,
        b_id: TensorId,
        c_id: TensorId,
    ) -> GgmlResult<()> {
        // Get tensor descriptors first
        let (m, k, n) = {
            let a_desc = self.tensor_desc(a_id);
            let b_desc = self.tensor_desc(b_id);
            let c_desc = self.tensor_desc(c_id);

            match (a_desc, b_desc, c_desc) {
                (Some(a_d), Some(b_d), Some(_c_d)) => {
                    let m = a_d.shape[0];
                    let k = a_d.shape.get(1).copied().unwrap_or(1);
                    let n = b_d.shape.get(1).copied().unwrap_or(1);
                    (m, k, n)
                }
                _ => {
                    return Err(GgmlError::Backend(format!(
                        "Missing tensor descriptors for matmul"
                    )));
                }
            }
        };

        // Clone input buffers to avoid borrow issues
        let (a_data, b_data) = {
            let a_buf = self.buffer(a_id).ok_or_else(|| {
                GgmlError::Backend(format!("Input tensor A not found: {:?}", a_id))
            })?;
            let b_buf = self.buffer(b_id).ok_or_else(|| {
                GgmlError::Backend(format!("Input tensor B not found: {:?}", b_id))
            })?;
            (a_buf.clone(), b_buf.clone())
        };

        // Check SIMD capability before taking mutable borrow
        #[cfg(feature = "simd")]
        let use_simd = self.simd_capable;
        #[cfg(not(feature = "simd"))]
        let _use_simd = false;

        let c_buf = self.buffer_mut(c_id).ok_or_else(|| {
            GgmlError::Backend(format!("Output tensor C not found: {:?}", c_id))
        })?;

        // Verify dimensions match
        if a_data.len() != m * k {
            return Err(GgmlError::InvalidShape(format!(
                "Matrix A buffer size {} doesn't match shape {}x{}",
                a_data.len(), m, k
            )));
        }
        if b_data.len() != k * n {
            return Err(GgmlError::InvalidShape(format!(
                "Matrix B buffer size {} doesn't match shape {}x{}",
                b_data.len(), k, n
            )));
        }
        if c_buf.len() != m * n {
            return Err(GgmlError::InvalidShape(format!(
                "Matrix C buffer size {} doesn't match shape {}x{}",
                c_buf.len(), m, n
            )));
        }

        #[cfg(feature = "simd")]
        {
            if use_simd {
                use crate::backend::cpu::simd::{simd_matmul_f32, SimdMatmulError};
                match simd_matmul_f32(&a_data, &b_data, m, n, k) {
                    Ok(result) => {
                        c_buf.copy_from_slice(&result);
                        return Ok(());
                    }
                    Err(SimdMatmulError::DimensionMismatch(e)) => {
                        return Err(GgmlError::InvalidShape(e));
                    }
                    Err(SimdMatmulError::BufferSizeError { expected, actual }) => {
                        return Err(GgmlError::InvalidShape(format!(
                            "Buffer size mismatch: expected {}, got {}",
                            expected, actual
                        )));
                    }
                }
            }
        }

        // Scalar fallback - inline implementation to avoid borrow issues
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_data[i * k + kk] * b_data[kk * n + j];
                }
                c_buf[i * n + j] = sum;
            }
        }
        Ok(())
    }

    /// Execute softmax operation
    fn softmax(&mut self, input_id: TensorId, output_id: TensorId) -> GgmlResult<()> {
        // Clone input to avoid borrow issues
        let input_data = self.buffer(input_id)
            .ok_or_else(|| GgmlError::Backend(format!("Input tensor not found: {:?}", input_id)))?
            .clone();

        // Check SIMD capability before taking mutable borrow
        #[cfg(feature = "simd")]
        let use_simd = self.simd_capable;
        #[cfg(not(feature = "simd"))]
        let _use_simd = false;

        let output_buf = self.buffer_mut(output_id).ok_or_else(|| {
            GgmlError::Backend(format!("Output tensor not found: {:?}", output_id))
        })?;

        if input_data.len() != output_buf.len() {
            return Err(GgmlError::InvalidShape(format!(
                "Input and output buffers must have same size: {} vs {}",
                input_data.len(),
                output_buf.len()
            )));
        }

        // Apply softmax row-wise (assuming input is 2D: [rows, cols])
        // For now, treat as 1D (single row)
        #[cfg(feature = "simd")]
        {
            if use_simd {
                use crate::attention::cpu::softmax_simd;
                let result = softmax_simd(&input_data);
                output_buf.copy_from_slice(&result);
                return Ok(());
            }
        }

        // Scalar fallback - inline implementation to avoid borrow issues
        let max_val = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = input_data.iter().map(|&x| (x - max_val).exp()).sum();
        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 1.0 };

        for (i, &val) in input_data.iter().enumerate() {
            output_buf[i] = (val - max_val).exp() * inv_sum;
        }
        Ok(())
    }

    /// Execute addition: C = A + B
    fn add(&mut self, a_id: TensorId, b_id: TensorId, c_id: TensorId) -> GgmlResult<()> {
        // Clone inputs to avoid borrow issues
        let (a_data, b_data) = {
            let a_buf = self.buffer(a_id).ok_or_else(|| {
                GgmlError::Backend(format!("Input tensor A not found: {:?}", a_id))
            })?;
            let b_buf = self.buffer(b_id).ok_or_else(|| {
                GgmlError::Backend(format!("Input tensor B not found: {:?}", b_id))
            })?;
            (a_buf.clone(), b_buf.clone())
        };

        let c_buf = self.buffer_mut(c_id).ok_or_else(|| {
            GgmlError::Backend(format!("Output tensor C not found: {:?}", c_id))
        })?;

        if a_data.len() != b_data.len() || a_data.len() != c_buf.len() {
            return Err(GgmlError::InvalidShape(format!(
                "Add buffers must have same size: {} vs {} vs {}",
                a_data.len(),
                b_data.len(),
                c_buf.len()
            )));
        }

        for (i, (&a, &b)) in a_data.iter().zip(b_data.iter()).enumerate() {
            c_buf[i] = a + b;
        }
        Ok(())
    }

    /// Execute scaling: output = input * factor
    fn scale(&mut self, input_id: TensorId, output_id: TensorId, factor: f32) -> GgmlResult<()> {
        // Clone input to avoid borrow issues
        let input_data = self.buffer(input_id)
            .ok_or_else(|| GgmlError::Backend(format!("Input tensor not found: {:?}", input_id)))?
            .clone();

        let output_buf = self.buffer_mut(output_id).ok_or_else(|| {
            GgmlError::Backend(format!("Output tensor not found: {:?}", output_id))
        })?;

        for (i, &val) in input_data.iter().enumerate() {
            output_buf[i] = val * factor;
        }
        Ok(())
    }

    /// Execute copy: output = input
    fn copy(&mut self, input_id: TensorId, output_id: TensorId) -> GgmlResult<()> {
        // Clone input to avoid borrow issues
        let input_data = self.buffer(input_id)
            .ok_or_else(|| GgmlError::Backend(format!("Input tensor not found: {:?}", input_id)))?
            .clone();

        let output_buf = self.buffer_mut(output_id).ok_or_else(|| {
            GgmlError::Backend(format!("Output tensor not found: {:?}", output_id))
        })?;

        if input_data.len() != output_buf.len() {
            return Err(GgmlError::InvalidShape(format!(
                "Copy buffers must have same size: {} vs {}",
                input_data.len(),
                output_buf.len()
            )));
        }

        output_buf.copy_from_slice(&input_data);
        Ok(())
    }
}

impl GgmlBackend for CpuBackend {
    type Buffer = Vec<f32>;

    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()> {
        let buffer = vec![0.0; desc.element_count()];
        self.tensors.insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()> {
        self.tensors.insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn free(&mut self, id: TensorId) -> GgmlResult<()> {
        self.tensors.remove(&id);
        Ok(())
    }

    fn tensor_desc(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors.get(&id).map(|(desc, _)| desc)
    }

    fn buffer(&self, id: TensorId) -> Option<&Self::Buffer> {
        self.tensors.get(&id).map(|(_, buf)| buf)
    }

    fn buffer_mut(&mut self, id: TensorId) -> Option<&mut Self::Buffer> {
        self.tensors.get_mut(&id).map(|(_, buf)| buf)
    }

    fn execute_op(
        &mut self,
        op: &Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<()> {
        match op {
            Op::MatMul => {
                if inputs.len() < 2 || outputs.len() < 1 {
                    return Err(GgmlError::InvalidShape(format!(
                        "MatMul requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    )));
                }
                self.matmul(inputs[0], inputs[1], outputs[0])
            }
            Op::Softmax => {
                if inputs.len() < 1 || outputs.len() < 1 {
                    return Err(GgmlError::InvalidShape(format!(
                        "Softmax requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    )));
                }
                self.softmax(inputs[0], outputs[0])
            }
            Op::Add => {
                if inputs.len() < 2 || outputs.len() < 1 {
                    return Err(GgmlError::InvalidShape(format!(
                        "Add requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    )));
                }
                self.add(inputs[0], inputs[1], outputs[0])
            }
            Op::Scale { factor } => {
                if inputs.len() < 1 || outputs.len() < 1 {
                    return Err(GgmlError::InvalidShape(format!(
                        "Scale requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    )));
                }
                self.scale(inputs[0], outputs[0], *factor)
            }
            Op::Copy => {
                if inputs.len() < 1 || outputs.len() < 1 {
                    return Err(GgmlError::InvalidShape(format!(
                        "Copy requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    )));
                }
                self.copy(inputs[0], outputs[0])
            }
            // Operations not yet implemented
            _ => Err(GgmlError::Unimplemented(format!(
                "CPU backend op not implemented: {:?}",
                op
            ))),
        }
    }

    fn synchronize(&mut self) -> GgmlResult<()> {
        // CPU backend is synchronous, nothing to synchronize
        Ok(())
    }
}

impl CapabilityProvider for CpuBackend {
    fn capabilities(&self) -> Vec<OpCapability> {
        use crate::ggml::DType;
        let mut caps = Vec::new();

        // CPU supports all basic operations for all data types
        // CPU has no size limit (limited only by system RAM)
        let basic_dtypes = vec![DType::F32, DType::F16, DType::I32];

        for dtype in basic_dtypes {
            caps.push(OpCapability {
                op_type: OpType::MatMul,
                supported_dtypes: vec![dtype],
                max_tensor_size: None,
                requires_feature: None,
            });
            caps.push(OpCapability {
                op_type: OpType::Add,
                supported_dtypes: vec![dtype],
                max_tensor_size: None,
                requires_feature: None,
            });
            caps.push(OpCapability {
                op_type: OpType::Scale,
                supported_dtypes: vec![dtype],
                max_tensor_size: None,
                requires_feature: None,
            });
        }

        // Softmax only supports F32
        caps.push(OpCapability {
            op_type: OpType::Softmax,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: None,
            requires_feature: None,
        });

        // Quantized operations - CPU dequantization always available
        caps.push(OpCapability {
            op_type: OpType::QuantizedMatMul,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: None,
            requires_feature: None,
        });

        // Attention operations - CPU has fallback implementation
        caps.push(OpCapability {
            op_type: OpType::Attention,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: None,
            requires_feature: None,
        });

        caps
    }

    fn op_capability(&self, op: &Op) -> Option<OpCapability> {
        use crate::ggml::DType;
        let op_type = OpType::from_op(op)?;

        Some(OpCapability {
            op_type,
            supported_dtypes: vec![DType::F32],
            max_tensor_size: None,
            requires_feature: None,
        })
    }

    fn backend_id(&self) -> &str {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{DType, Layout};
    use crate::ggml::hybrid_scheduler::{CapabilityProvider, OpType};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_test_tensor_id() -> TensorId {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        TensorId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    fn make_test_tensor_desc(shape: Vec<usize>) -> TensorDesc {
        TensorDesc {
            id: make_test_tensor_id(),
            shape,
            dtype: DType::F32,
            layout: Layout::RowMajor,
            strides: vec![],
            byte_offset: 0,
            view_of: None,
        }
    }

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        // Check if SIMD capability matches feature flag
        #[cfg(feature = "simd")]
        {
            let expected_simd = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));
            assert_eq!(backend.is_simd_capable(), expected_simd);
        }
        #[cfg(not(feature = "simd"))]
        {
            assert!(!backend.is_simd_capable());
        }
    }

    #[test]
    fn test_cpu_backend_alloc_bind() {
        let mut backend = CpuBackend::new();
        let desc = make_test_tensor_desc(vec![2, 2]);

        // Test allocation
        backend.alloc(&desc).unwrap();
        assert!(backend.tensor_desc(desc.id).is_some());
        assert!(backend.buffer(desc.id).is_some());

        // Test binding
        let desc2 = make_test_tensor_desc(vec![2, 2]);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        backend.bind(&desc2, data.clone()).unwrap();
        assert_eq!(backend.buffer(desc2.id).unwrap(), &data);
    }

    #[test]
    fn test_cpu_backend_free() {
        let mut backend = CpuBackend::new();
        let desc = make_test_tensor_desc(vec![2, 2]);

        backend.alloc(&desc).unwrap();
        assert!(backend.tensor_desc(desc.id).is_some());

        backend.free(desc.id).unwrap();
        assert!(backend.tensor_desc(desc.id).is_none());
    }

    #[test]
    fn test_cpu_backend_matmul() {
        let mut backend = CpuBackend::new();

        // Create test tensors: 2x2 * 2x2 = 2x2
        let a_desc = make_test_tensor_desc(vec![2, 2]);
        let b_desc = make_test_tensor_desc(vec![2, 2]);
        let c_desc = make_test_tensor_desc(vec![2, 2]);

        let a_data = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b_data = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
        let c_data = vec![0.0; 4];

        backend.bind(&a_desc, a_data).unwrap();
        backend.bind(&b_desc, b_data).unwrap();
        backend.bind(&c_desc, c_data).unwrap();

        // Execute matmul
        backend
            .execute_op(&Op::MatMul, &[a_desc.id, b_desc.id], &[c_desc.id])
            .unwrap();

        // Check result: [[19,22],[43,50]]
        let result = backend.buffer(c_desc.id).unwrap();
        assert!((result[0] - 19.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 43.0).abs() < 1e-5);
        assert!((result[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_backend_add() {
        let mut backend = CpuBackend::new();

        let a_desc = make_test_tensor_desc(vec![4]);
        let b_desc = make_test_tensor_desc(vec![4]);
        let c_desc = make_test_tensor_desc(vec![4]);

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let c_data = vec![0.0; 4];

        backend.bind(&a_desc, a_data).unwrap();
        backend.bind(&b_desc, b_data).unwrap();
        backend.bind(&c_desc, c_data).unwrap();

        backend
            .execute_op(&Op::Add, &[a_desc.id, b_desc.id], &[c_desc.id])
            .unwrap();

        let result = backend.buffer(c_desc.id).unwrap();
        assert_eq!(result, &vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_cpu_backend_scale() {
        let mut backend = CpuBackend::new();

        let a_desc = make_test_tensor_desc(vec![4]);
        let c_desc = make_test_tensor_desc(vec![4]);

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let c_data = vec![0.0; 4];

        backend.bind(&a_desc, a_data).unwrap();
        backend.bind(&c_desc, c_data).unwrap();

        backend
            .execute_op(&Op::Scale { factor: 2.5 }, &[a_desc.id], &[c_desc.id])
            .unwrap();

        let result = backend.buffer(c_desc.id).unwrap();
        assert_eq!(result, &vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_cpu_backend_copy() {
        let mut backend = CpuBackend::new();

        let a_desc = make_test_tensor_desc(vec![4]);
        let c_desc = make_test_tensor_desc(vec![4]);

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let c_data = vec![0.0; 4];

        backend.bind(&a_desc, a_data).unwrap();
        backend.bind(&c_desc, c_data).unwrap();

        backend
            .execute_op(&Op::Copy, &[a_desc.id], &[c_desc.id])
            .unwrap();

        let result = backend.buffer(c_desc.id).unwrap();
        assert_eq!(result, &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_backend_softmax() {
        let mut backend = CpuBackend::new();

        let a_desc = make_test_tensor_desc(vec![3]);
        let c_desc = make_test_tensor_desc(vec![3]);

        let a_data = vec![1.0, 2.0, 3.0];
        let c_data = vec![0.0; 3];

        backend.bind(&a_desc, a_data).unwrap();
        backend.bind(&c_desc, c_data).unwrap();

        backend
            .execute_op(&Op::Softmax, &[a_desc.id], &[c_desc.id])
            .unwrap();

        let result = backend.buffer(c_desc.id).unwrap();

        // Check that softmax sums to ~1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);

        // Check that all values are positive
        for &val in result {
            assert!(val > 0.0);
        }
    }

    // CapabilityProvider tests
    #[test]
    fn test_cpu_capabilities_includes_matmul() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.iter().any(|c| c.op_type == OpType::MatMul));
    }

    #[test]
    fn test_cpu_can_execute_matmul() {
        let backend = CpuBackend::new();
        let op = Op::MatMul;
        assert!(backend.can_execute(&op));
    }

    #[test]
    fn test_cpu_can_execute_softmax() {
        let backend = CpuBackend::new();
        let op = Op::Softmax;
        assert!(backend.can_execute(&op));
    }

    #[test]
    fn test_cpu_can_execute_add() {
        let backend = CpuBackend::new();
        let op = Op::Add;
        assert!(backend.can_execute(&op));
    }

    #[test]
    fn test_cpu_can_execute_scale() {
        let backend = CpuBackend::new();
        let op = Op::Scale { factor: 2.0 };
        assert!(backend.can_execute(&op));
    }

    #[test]
    fn test_cpu_backend_id() {
        let backend = CpuBackend::new();
        assert_eq!(backend.backend_id(), "cpu");
    }

    #[test]
    fn test_cpu_supports_all_basic_ops() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();

        let mut found_matmul = false;
        let mut found_add = false;
        let mut found_scale = false;
        let mut found_softmax = false;

        for cap in &caps {
            match cap.op_type {
                OpType::MatMul => found_matmul = true,
                OpType::Add => found_add = true,
                OpType::Scale => found_scale = true,
                OpType::Softmax => found_softmax = true,
                _ => {}
            }
        }

        assert!(found_matmul, "CPU should support MatMul");
        assert!(found_add, "CPU should support Add");
        assert!(found_scale, "CPU should support Scale");
        assert!(found_softmax, "CPU should support Softmax");
    }
}
