//! Hybrid execution scheduler for automatic CPU/GPU operation selection

use crate::ggml::{GgmlError, GgmlResult, Op, DType};
use std::sync::Arc;

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

impl OpType {
    /// Map from Op to OpType for capability checking
    ///
    /// Returns None for metadata-only operations (Constant, View, Permute, etc.)
    /// that don't require backend execution or capability checking.
    pub fn from_op(op: &Op) -> Option<Self> {
        match op {
            Op::MatMul => Some(OpType::MatMul),
            Op::MatMulQ4_0 | Op::MatMulQ8_0 => Some(OpType::QuantizedMatMul),
            Op::Add => Some(OpType::Add),
            Op::Scale { .. } => Some(OpType::Scale),
            Op::Softmax => Some(OpType::Softmax),
            Op::Attention => Some(OpType::Attention),
            // Metadata ops don't need backend selection
            Op::View | Op::Reshape | Op::Copy | Op::GetRows | Op::Mask
            | Op::LayerNorm { .. } | Op::RmsNorm { .. } | Op::Rope
            | Op::SwiGlu | Op::MlpSwiglu | Op::SplitQkv
            | Op::Accumulate { .. } => None,
        }
    }
}

/// Capability query trait - independent of GgmlBackend to allow dynamic dispatch
pub trait CapabilityProvider {
    /// Get all operations this backend can execute
    fn capabilities(&self) -> Vec<OpCapability>;

    /// Check if this backend can execute a specific operation
    fn can_execute(&self, op: &Op) -> bool {
        self.op_capability(op).is_some()
    }

    /// Get the capability requirement for an operation
    fn op_capability(&self, op: &Op) -> Option<OpCapability>;

    /// Get backend identifier
    fn backend_id(&self) -> &str;
}

/// Cost estimate for executing an operation on a backend
#[derive(Debug, Clone, Copy)]
pub struct OpCost {
    pub estimated_us: u64,        // Estimated execution time in microseconds
    pub memory_bytes: usize,       // Estimated memory usage
    pub transfer_cost: Option<u64>, // Data transfer cost (for heterogeneous execution)
}

/// Execution strategy for operation scheduling
#[derive(Debug, Clone, Copy)]
pub enum ExecutionStrategy {
    /// Always use GPU if available, fallback to CPU
    GpuPreferred,
    /// Always use CPU if available
    CpuPreferred,
    /// Automatically select based on cost model
    Automatic,
    /// Use specified backend only
    BackendOnly(&'static str),
}

/// Selection reason for telemetry
#[derive(Debug, Clone)]
pub enum SelectionReason {
    GpuAvailable,
    GpuUnavailable { reason: String },
    CpuFallback,
    CostModel { gpu_cost: OpCost, cpu_cost: OpCost },
    MemoryConstraint { required: usize, available: usize },
    UserPreference(String),
}

/// Backend selection result
#[derive(Debug, Clone)]
pub struct BackendSelection {
    pub backend_id: String,
    pub reason: SelectionReason,
    pub estimated_cost: OpCost,
}

/// Hybrid execution scheduler
pub struct HybridScheduler {
    cpu_backend: Option<Arc<dyn CapabilityProvider>>,
    gpu_backend: Option<Arc<dyn CapabilityProvider>>,
    strategy: ExecutionStrategy,
    telemetry: Vec<ExecutionEvent>,
}

impl HybridScheduler {
    pub fn new(strategy: ExecutionStrategy) -> Self {
        Self {
            cpu_backend: None,
            gpu_backend: None,
            strategy,
            telemetry: Vec::new(),
        }
    }

    pub fn with_cpu_backend(mut self, backend: Arc<dyn CapabilityProvider>) -> Self {
        self.cpu_backend = Some(backend);
        self
    }

    pub fn with_gpu_backend(mut self, backend: Arc<dyn CapabilityProvider>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    /// Select the best backend for executing an operation
    pub fn select_backend(&self, op: &Op) -> GgmlResult<BackendSelection> {
        match self.strategy {
            ExecutionStrategy::GpuPreferred => self.select_gpu_preferred(op),
            ExecutionStrategy::CpuPreferred => self.select_cpu_preferred(op),
            ExecutionStrategy::Automatic => self.select_automatic(op),
            ExecutionStrategy::BackendOnly(name) => self.select_named(name, op),
        }
    }

    fn select_gpu_preferred(&self, op: &Op) -> GgmlResult<BackendSelection> {
        if let Some(gpu) = &self.gpu_backend {
            if gpu.as_ref().can_execute(op) {
                return Ok(BackendSelection {
                    backend_id: gpu.as_ref().backend_id().to_string(),
                    reason: SelectionReason::GpuAvailable,
                    estimated_cost: self.estimate_cost(gpu.as_ref(), op),
                });
            }
            return Ok(BackendSelection {
                backend_id: gpu.as_ref().backend_id().to_string(),
                reason: SelectionReason::GpuUnavailable {
                    reason: "Operation not supported".to_string(),
                },
                estimated_cost: OpCost { estimated_us: 0, memory_bytes: 0, transfer_cost: None },
            });
        }
        Err(GgmlError::Backend("No GPU backend available".to_string()))
    }

    fn select_cpu_preferred(&self, op: &Op) -> GgmlResult<BackendSelection> {
        if let Some(cpu) = &self.cpu_backend {
            if cpu.as_ref().can_execute(op) {
                return Ok(BackendSelection {
                    backend_id: cpu.as_ref().backend_id().to_string(),
                    reason: SelectionReason::CpuFallback,
                    estimated_cost: self.estimate_cost(cpu.as_ref(), op),
                });
            }
        }
        Err(GgmlError::Backend("No CPU backend available".to_string()))
    }

    fn select_automatic(&self, op: &Op) -> GgmlResult<BackendSelection> {
        // For now, prefer GPU if available and supports the operation
        // Cost modeling will be enhanced in 07-03
        let gpu_available = self.gpu_backend.as_ref()
            .map(|b| b.as_ref().can_execute(op))
            .unwrap_or(false);

        if gpu_available {
            self.select_gpu_preferred(op)
        } else if let Some(cpu) = &self.cpu_backend {
            if cpu.as_ref().can_execute(op) {
                Ok(BackendSelection {
                    backend_id: cpu.as_ref().backend_id().to_string(),
                    reason: SelectionReason::CpuFallback,
                    estimated_cost: self.estimate_cost(cpu.as_ref(), op),
                })
            } else {
                Err(GgmlError::Backend("No backend can execute this operation".to_string()))
            }
        } else {
            Err(GgmlError::Backend("No backends available".to_string()))
        }
    }

    fn select_named(&self, name: &str, op: &Op) -> GgmlResult<BackendSelection> {
        match name {
            "gpu" => self.select_gpu_preferred(op),
            "cpu" => self.select_cpu_preferred(op),
            _ => Err(GgmlError::Backend(format!("Unknown backend: {}", name))),
        }
    }

    fn estimate_cost(&self, backend: &dyn CapabilityProvider, op: &Op) -> OpCost {
        // Basic cost estimation - will be enhanced in 07-03
        OpCost {
            estimated_us: 100, // Placeholder
            memory_bytes: 1024,
            transfer_cost: None,
        }
    }
}

/// Telemetry event for execution tracking
#[derive(Debug, Clone)]
pub struct ExecutionEvent {
    pub timestamp: std::time::Instant,
    pub operation: OpType,
    pub backend: String,
    pub reason: SelectionReason,
    pub actual_duration_us: Option<u64>,
}

impl HybridScheduler {
    pub fn record_execution(&mut self, event: ExecutionEvent) {
        self.telemetry.push(event);
    }

    pub fn get_telemetry(&self) -> &[ExecutionEvent] {
        &self.telemetry
    }

    pub fn clear_telemetry(&mut self) {
        self.telemetry.clear();
    }

    /// Get statistics about backend usage
    pub fn backend_usage_stats(&self) -> BackendStats {
        let mut gpu_count = 0;
        let mut cpu_count = 0;

        for event in &self.telemetry {
            match event.backend.as_str() {
                "gpu" => gpu_count += 1,
                "cpu" => cpu_count += 1,
                _ => {}
            }
        }

        BackendStats {
            total_operations: self.telemetry.len(),
            gpu_operations: gpu_count,
            cpu_operations: cpu_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendStats {
    pub total_operations: usize,
    pub gpu_operations: usize,
    pub cpu_operations: usize,
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_test_tensor_id() -> crate::ggml::TensorId {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        crate::ggml::TensorId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    // Mock capability provider for testing
    struct MockProvider {
        id: String,
        supported_ops: Vec<OpType>,
    }

    impl MockProvider {
        fn new(id: &str, supported_ops: Vec<OpType>) -> Self {
            Self {
                id: id.to_string(),
                supported_ops,
            }
        }
    }

    impl CapabilityProvider for MockProvider {
        fn capabilities(&self) -> Vec<OpCapability> {
            self.supported_ops.iter().map(|&op| OpCapability::new(op)).collect()
        }

        fn op_capability(&self, op: &Op) -> Option<OpCapability> {
            let op_type = OpType::from_op(op)?;
            if self.supported_ops.contains(&op_type) {
                Some(OpCapability::new(op_type))
            } else {
                None
            }
        }

        fn backend_id(&self) -> &str {
            &self.id
        }
    }

    #[test]
    fn test_execution_strategy_variants() {
        // Verify all strategy variants can be created
        let _ = ExecutionStrategy::GpuPreferred;
        let _ = ExecutionStrategy::CpuPreferred;
        let _ = ExecutionStrategy::Automatic;
        let _ = ExecutionStrategy::BackendOnly("cpu");
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);
        let stats = scheduler.backend_usage_stats();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.gpu_operations, 0);
        assert_eq!(stats.cpu_operations, 0);
    }

    #[test]
    fn test_telemetry_tracking() {
        let mut scheduler = HybridScheduler::new(ExecutionStrategy::GpuPreferred);
        let event = ExecutionEvent {
            timestamp: std::time::Instant::now(),
            operation: OpType::MatMul,
            backend: "cpu".to_string(),
            reason: SelectionReason::CpuFallback,
            actual_duration_us: Some(100),
        };
        scheduler.record_execution(event);
        assert_eq!(scheduler.get_telemetry().len(), 1);
    }

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

    #[test]
    fn test_select_gpu_preferred() {
        let gpu = Arc::new(MockProvider::new("gpu", vec![OpType::MatMul]));
        let scheduler = HybridScheduler::new(ExecutionStrategy::GpuPreferred)
            .with_gpu_backend(gpu);

        let selection = scheduler.select_backend(&Op::MatMul).unwrap();
        assert_eq!(selection.backend_id, "gpu");
        assert!(matches!(selection.reason, SelectionReason::GpuAvailable));
    }

    #[test]
    fn test_select_automatic_fallback_to_cpu() {
        let cpu = Arc::new(MockProvider::new("cpu", vec![OpType::MatMul]));
        let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic)
            .with_cpu_backend(cpu);

        // GPU not available, should fall back to CPU
        let selection = scheduler.select_backend(&Op::MatMul).unwrap();
        assert_eq!(selection.backend_id, "cpu");
        assert!(matches!(selection.reason, SelectionReason::CpuFallback));
    }

    #[test]
    fn test_backend_usage_stats() {
        let mut scheduler = HybridScheduler::new(ExecutionStrategy::GpuPreferred);

        scheduler.record_execution(ExecutionEvent {
            timestamp: std::time::Instant::now(),
            operation: OpType::MatMul,
            backend: "gpu".to_string(),
            reason: SelectionReason::GpuAvailable,
            actual_duration_us: Some(100),
        });

        scheduler.record_execution(ExecutionEvent {
            timestamp: std::time::Instant::now(),
            operation: OpType::Softmax,
            backend: "cpu".to_string(),
            reason: SelectionReason::CpuFallback,
            actual_duration_us: Some(50),
        });

        let stats = scheduler.backend_usage_stats();
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.gpu_operations, 1);
        assert_eq!(stats.cpu_operations, 1);
    }
}
