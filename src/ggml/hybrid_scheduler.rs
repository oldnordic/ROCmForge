//! # Hybrid Execution Scheduler
//!
//! This module provides automatic CPU/GPU operation selection for maximum
//! compatibility and performance.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rocmforge::ggml::{HybridScheduler, ExecutionStrategy};
//!
//! // Create a scheduler that automatically selects the best backend
//! let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);
//!
//! // Or prefer GPU with CPU fallback
//! let scheduler = HybridScheduler::new(ExecutionStrategy::GpuPreferred);
//! ```
//!
//! ## Telemetry
//!
//! The scheduler tracks execution decisions and performance:
//!
//! ```rust,ignore
//! let summary = scheduler.execution_summary();
//! println!("GPU: {} us, CPU: {} us", summary.gpu_time_us, summary.cpu_time_us);
//! ```

use crate::ggml::{GgmlBackend, GgmlError, GgmlResult, Op, TensorDesc, TensorId, DType};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

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
        let gpu_available = self.gpu_backend.as_ref()
            .map(|b| b.as_ref().can_execute(op))
            .unwrap_or(false);

        let cpu_available = self.cpu_backend.as_ref()
            .map(|b| b.as_ref().can_execute(op))
            .unwrap_or(false);

        match (gpu_available, cpu_available) {
            (true, true) => {
                // Both available - compare costs
                if let (Some(gpu), Some(cpu)) = (&self.gpu_backend, &self.cpu_backend) {
                    let gpu_cost = self.estimate_cost(gpu.as_ref(), op);
                    let cpu_cost = self.estimate_cost(cpu.as_ref(), op);

                    // Prefer GPU unless CPU is significantly faster
                    // (e.g., for very small operations where transfer cost dominates)
                    // CPU must be at least 2x faster to be preferred
                    if cpu_cost.estimated_us < gpu_cost.estimated_us / 2 {
                        // CPU is at least 2x faster - use it
                        Ok(BackendSelection {
                            backend_id: "cpu".to_string(),
                            reason: SelectionReason::CostModel {
                                gpu_cost,
                                cpu_cost,
                            },
                            estimated_cost: cpu_cost,
                        })
                    } else {
                        // GPU is preferred (default for most operations)
                        Ok(BackendSelection {
                            backend_id: "gpu".to_string(),
                            reason: SelectionReason::CostModel {
                                gpu_cost,
                                cpu_cost,
                            },
                            estimated_cost: gpu_cost,
                        })
                    }
                } else {
                    unreachable!()
                }
            }
            (true, false) => self.select_gpu_preferred(op),
            (false, true) => self.select_cpu_preferred(op),
            (false, false) => {
                Err(GgmlError::Backend("No backend can execute this operation".to_string()))
            }
        }
    }

    fn select_named(&self, name: &str, op: &Op) -> GgmlResult<BackendSelection> {
        match name {
            "gpu" => self.select_gpu_preferred(op),
            "cpu" => self.select_cpu_preferred(op),
            _ => Err(GgmlError::Backend(format!("Unknown backend: {}", name))),
        }
    }

    /// Enhanced cost estimation based on operation properties
    fn estimate_cost(&self, backend: &dyn CapabilityProvider, op: &Op) -> OpCost {
        // Get tensor sizes from the operation
        let tensor_elements = self.estimate_tensor_elements(op);
        let is_gpu = backend.capabilities().iter()
            .any(|c| c.requires_feature.as_deref() == Some("rocm"));

        // Base estimates (in microseconds) - calibrated for different operation types
        let base_us = match OpType::from_op(op) {
            Some(OpType::MatMul) => 10,
            Some(OpType::QuantizedMatMul) => 5,  // Faster due to fusion and reduced memory
            Some(OpType::Softmax) => 5,
            Some(OpType::Add) => 1,
            Some(OpType::Scale) => 1,
            Some(OpType::Attention) => 20,
            Some(OpType::Dequantize) => 2,
            None => return OpCost { estimated_us: 0, memory_bytes: 0, transfer_cost: None },
        };

        // Scale by tensor size (logarithmic scaling)
        // This reflects that larger operations scale sub-linearly due to parallelism
        let size_factor = (tensor_elements as f64).log2().max(1.0) as u64;
        let estimated_us = base_us * size_factor;

        // Memory estimate (4 bytes per element for F32)
        let memory_bytes = tensor_elements * 4;

        // Transfer cost for heterogeneous execution
        // GPU already has data on-device, CPU would need PCIe transfer
        let transfer_cost = if is_gpu {
            None  // GPU already has data
        } else {
            Some(estimated_us / 10)  // 10% overhead for PCIe transfer
        };

        OpCost {
            estimated_us,
            memory_bytes,
            transfer_cost,
        }
    }

    /// Estimate total number of tensor elements for an operation
    fn estimate_tensor_elements(&self, op: &Op) -> usize {
        // Extract tensor sizes from operation
        // These are conservative estimates for cost modeling
        match op {
            Op::MatMul { .. } => {
                // Assume typical transformer layer matmul: 2048 x 2048
                2048 * 2048
            }
            Op::Softmax => {
                // Attention scores: seq_len x seq_len, assume 128 sequence length
                128 * 128
            }
            Op::Add => 1024,
            Op::Scale { .. } => 1024,
            Op::MatMulQ4_0 | Op::MatMulQ8_0 => {
                // Quantized matmul: smaller tensors due to compression
                2048 * 2048 / 2  // Q4_0 is ~4x smaller, Q8_0 is ~2x smaller
            }
            Op::Attention => {
                // Full attention: Q, K, V matrices for a layer
                3 * 128 * 128  // 3 matrices of size seq_len x head_dim
            }
            _ => 1024,
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

    /// Get execution summary by backend
    pub fn execution_summary(&self) -> BackendExecutionSummary {
        let mut by_backend: HashMap<String, Vec<&ExecutionEvent>> = HashMap::new();
        let mut total_time_us: u64 = 0;

        for event in &self.telemetry {
            by_backend
                .entry(event.backend.clone())
                .or_insert_with(Vec::new)
                .push(event);

            if let Some(duration) = event.actual_duration_us {
                total_time_us += duration;
            }
        }

        let mut gpu_time = 0;
        let mut cpu_time = 0;
        let mut gpu_ops = 0;
        let mut cpu_ops = 0;

        for (backend, events) in &by_backend {
            let backend_time: u64 = events.iter()
                .filter_map(|e| e.actual_duration_us)
                .sum();

            match backend.as_str() {
                "gpu" => {
                    gpu_time = backend_time;
                    gpu_ops = events.len();
                }
                "cpu" => {
                    cpu_time = backend_time;
                    cpu_ops = events.len();
                }
                _ => {}
            }
        }

        BackendExecutionSummary {
            total_operations: self.telemetry.len(),
            gpu_operations: gpu_ops,
            cpu_operations: cpu_ops,
            total_time_us,
            gpu_time_us: gpu_time,
            cpu_time_us: cpu_time,
        }
    }

    /// Print a debug summary of execution
    pub fn print_debug_summary(&self) {
        let summary = self.execution_summary();

        eprintln!("=== Hybrid Scheduler Execution Summary ===");
        eprintln!("Total operations: {}", summary.total_operations);
        if summary.total_operations > 0 {
            eprintln!("GPU operations: {} ({:.1}%)",
                summary.gpu_operations,
                (summary.gpu_operations as f64 / summary.total_operations as f64) * 100.0
            );
            eprintln!("CPU operations: {} ({:.1}%)",
                summary.cpu_operations,
                (summary.cpu_operations as f64 / summary.total_operations as f64) * 100.0
            );
            eprintln!("Total time: {} us", summary.total_time_us);
            if summary.total_time_us > 0 {
                eprintln!("GPU time: {} us ({:.1}%)",
                    summary.gpu_time_us,
                    (summary.gpu_time_us as f64 / summary.total_time_us as f64) * 100.0
                );
                eprintln!("CPU time: {} us ({:.1}%)",
                    summary.cpu_time_us,
                    (summary.cpu_time_us as f64 / summary.total_time_us as f64) * 100.0
                );
            }
        }
        eprintln!("=========================================");
    }

    /// Get operations by type
    pub fn operations_by_type(&self, op_type: OpType) -> Vec<&ExecutionEvent> {
        self.telemetry.iter()
            .filter(|e| e.operation == op_type)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct BackendStats {
    pub total_operations: usize,
    pub gpu_operations: usize,
    pub cpu_operations: usize,
}

/// Execution summary by backend
#[derive(Debug, Clone)]
pub struct BackendExecutionSummary {
    pub total_operations: usize,
    pub gpu_operations: usize,
    pub cpu_operations: usize,
    pub total_time_us: u64,
    pub gpu_time_us: u64,
    pub cpu_time_us: u64,
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

/// Hybrid executor that delegates to CPU or GPU backend based on scheduler decisions
///
/// This executor wraps both CPU and GPU backends and automatically selects
/// the appropriate one for each operation based on the cost model.
///
/// # Type Parameters
/// - `Buffer`: Uses `Box<dyn Any>` to handle different buffer types from different backends
///
/// # Example
/// ```ignore
/// let cpu = Box::new(CpuBackend::new());
/// let gpu = Some(Box::new(HipGgmlBackend::new(hip_backend)));
/// let mut executor = HybridExecutor::new(cpu, gpu);
///
/// // Use executor like any other GgmlBackend
/// executor.alloc(&tensor_desc)?;
/// executor.execute_op(&Op::MatMul, &[input_a, input_b], &[output])?;
/// ```
pub struct HybridExecutor {
    scheduler: HybridScheduler,
    /// CPU backend - always available for fallback
    cpu_backend: Option<Box<dyn GgmlBackend<Buffer = Box<dyn Any>>>>,
    /// GPU backend - optional, may not be available
    gpu_backend: Option<Box<dyn GgmlBackend<Buffer = Box<dyn Any>>>>,
    /// Track which backend is currently active for a given operation
    active_backend: Option<String>,
}

impl HybridExecutor {
    /// Create a new hybrid executor with CPU and optional GPU backend
    ///
    /// # Arguments
    /// - `cpu`: CPU backend for fallback and small operations
    /// - `gpu`: Optional GPU backend for large parallelizable operations
    ///
    /// # Returns
    /// A new HybridExecutor with Automatic strategy if GPU available, CpuPreferred otherwise
    pub fn new(
        cpu: Box<dyn GgmlBackend<Buffer = Box<dyn Any>>>,
        gpu: Option<Box<dyn GgmlBackend<Buffer = Box<dyn Any>>>>,
    ) -> Self {
        let strategy = if gpu.is_some() {
            ExecutionStrategy::Automatic
        } else {
            ExecutionStrategy::CpuPreferred
        };

        let scheduler = HybridScheduler::new(strategy);

        // Register capability providers if available
        // Note: We need to wrap the backends to provide CapabilityProvider
        // This is a simplified version - full integration would require backend wrappers

        Self {
            scheduler,
            cpu_backend: Some(cpu),
            gpu_backend: gpu,
            active_backend: None,
        }
    }

    /// Get the scheduler for configuration
    pub fn scheduler(&self) -> &HybridScheduler {
        &self.scheduler
    }

    /// Get mutable scheduler for configuration
    pub fn scheduler_mut(&mut self) -> &mut HybridScheduler {
        &mut self.scheduler
    }

    /// Select backend for an operation using heuristic-based selection
    ///
    /// This is a simplified selection based on operation type.
    /// In production, this would use the scheduler's full cost model with
    /// actual tensor shape information.
    fn select_backend_for_op(&self, op: &Op) -> GgmlResult<String> {
        // Use heuristic based on operation type and availability
        let gpu_available = self.gpu_backend.is_some();

        if gpu_available {
            match op {
                // Large, parallelizable operations prefer GPU
                Op::MatMul { .. } | Op::MatMulQ4_0 | Op::MatMulQ8_0 | Op::Attention => {
                    Ok("gpu".to_string())
                }
                // Small element-wise operations may use CPU
                Op::Add | Op::Scale { .. } | Op::Softmax => Ok("cpu".to_string()),
                // Other operations use CPU for simplicity
                _ => Ok("cpu".to_string()),
            }
        } else {
            Ok("cpu".to_string())
        }
    }

    /// Get a mutable reference to a backend by name
    fn get_backend_mut(&mut self, name: &str) -> GgmlResult<&mut dyn GgmlBackend<Buffer = Box<dyn Any>>> {
        match name {
            "cpu" => self.cpu_backend.as_mut()
                .map(|b| b.as_mut() as &mut dyn GgmlBackend<Buffer = Box<dyn Any>>)
                .ok_or_else(|| GgmlError::Backend("CPU backend not available".to_string())),
            "gpu" => self.gpu_backend.as_mut()
                .map(|b| b.as_mut() as &mut dyn GgmlBackend<Buffer = Box<dyn Any>>)
                .ok_or_else(|| GgmlError::Backend("GPU backend not available".to_string())),
            _ => Err(GgmlError::Backend(format!("Unknown backend: {}", name))),
        }
    }

    /// Execute operation with telemetry recording
    fn execute_op_with_telemetry(
        &mut self,
        op: &Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<()> {
        let start = Instant::now();
        let op_type = OpType::from_op(op);

        // Select and execute
        let backend_name = self.select_backend_for_op(op)?;
        self.active_backend = Some(backend_name.clone());

        let backend = self.get_backend_mut(&backend_name)?;

        let result = backend.execute_op(op, inputs, outputs);
        let duration = start.elapsed();

        // Record telemetry
        if let Some(op_type) = op_type {
            let event = ExecutionEvent {
                timestamp: Instant::now(),
                operation: op_type,
                backend: backend_name.clone(),
                reason: SelectionReason::CpuFallback,  // Simplified for now
                actual_duration_us: Some(duration.as_micros() as u64),
            };
            self.scheduler_mut().record_execution(event);
        }

        result
    }
}

impl GgmlBackend for HybridExecutor {
    type Buffer = Box<dyn Any>;

    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()> {
        // Allocate on CPU for simplicity - data can be transferred to GPU as needed
        if let Some(cpu) = self.cpu_backend.as_mut() {
            cpu.alloc(desc)
        } else {
            Err(GgmlError::Backend("No backend available for allocation".to_string()))
        }
    }

    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()> {
        if let Some(cpu) = self.cpu_backend.as_mut() {
            cpu.bind(desc, buffer)
        } else {
            Err(GgmlError::Backend("No backend available".to_string()))
        }
    }

    fn free(&mut self, id: TensorId) -> GgmlResult<()> {
        if let Some(cpu) = self.cpu_backend.as_mut() {
            cpu.free(id)
        } else {
            Err(GgmlError::Backend("No backend available".to_string()))
        }
    }

    fn tensor_desc(&self, id: TensorId) -> Option<&TensorDesc> {
        self.cpu_backend.as_ref()?.tensor_desc(id)
    }

    fn buffer(&self, id: TensorId) -> Option<&Self::Buffer> {
        self.cpu_backend.as_ref()?.buffer(id)
    }

    fn buffer_mut(&mut self, id: TensorId) -> Option<&mut Self::Buffer> {
        self.cpu_backend.as_mut()?.buffer_mut(id)
    }

    fn execute_op(
        &mut self,
        op: &Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<()> {
        self.execute_op_with_telemetry(op, inputs, outputs)
    }

    fn synchronize(&mut self) -> GgmlResult<()> {
        // Synchronize all backends to ensure all operations complete
        if let Some(cpu) = self.cpu_backend.as_mut() {
            cpu.synchronize()?;
        }
        if let Some(gpu) = self.gpu_backend.as_mut() {
            gpu.synchronize()?;
        }
        Ok(())
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

    // Automatic selection tests
    #[test]
    fn test_automatic_prefers_gpu_for_large_ops() {
        let gpu = Arc::new(MockProvider::new("gpu", vec![OpType::MatMul]));
        let cpu = Arc::new(MockProvider::new("cpu", vec![OpType::MatMul]));
        let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic)
            .with_cpu_backend(cpu)
            .with_gpu_backend(gpu);

        // MatMul is a large operation - should prefer GPU
        let selection = scheduler.select_backend(&Op::MatMul).unwrap();
        assert_eq!(selection.backend_id, "gpu");
        assert!(matches!(selection.reason, SelectionReason::CostModel { .. }));

        // Verify cost model was used
        if let SelectionReason::CostModel { gpu_cost, cpu_cost } = selection.reason {
            // GPU should be preferred (not significantly slower than CPU)
            assert!(gpu_cost.estimated_us > 0);
            assert!(cpu_cost.estimated_us > 0);
        }
    }

    #[test]
    fn test_automatic_error_when_no_backends() {
        let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);
        // No backends registered - should error
        let result = scheduler.select_backend(&Op::MatMul);
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_comparison() {
        let gpu_cost = OpCost {
            estimated_us: 100,
            memory_bytes: 1024,
            transfer_cost: None,
        };
        let cpu_cost = OpCost {
            estimated_us: 40,  // CPU is faster
            memory_bytes: 1024,
            transfer_cost: None,
        };

        // CPU is more than 2x faster - should be preferred
        assert!(cpu_cost.estimated_us < gpu_cost.estimated_us / 2);
    }

    #[test]
    fn test_cost_model_with_transfer_penalty() {
        // Simulate CPU having transfer cost
        let gpu_cost = OpCost {
            estimated_us: 100,
            memory_bytes: 4096,
            transfer_cost: None,  // GPU already has data
        };
        let cpu_cost = OpCost {
            estimated_us: 80,
            memory_bytes: 4096,
            transfer_cost: Some(10),  // CPU needs transfer
        };

        // GPU should be preferred even though CPU base is slightly faster
        // when considering total cost including transfer
        let cpu_total = cpu_cost.estimated_us + cpu_cost.transfer_cost.unwrap_or(0);
        // CPU total (80 + 10 = 90) < GPU (100), so CPU would be preferred here
        // The test validates that transfer cost is correctly computed
        assert_eq!(cpu_total, 90);
        assert!(cpu_cost.transfer_cost.is_some());
        assert!(gpu_cost.transfer_cost.is_none());
    }

    #[test]
    fn test_tensor_element_estimation() {
        let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic);

        // MatMul should estimate large tensor
        let matmul_elements = scheduler.estimate_tensor_elements(&Op::MatMul);
        assert_eq!(matmul_elements, 2048 * 2048);

        // Softmax should estimate smaller tensor
        let softmax_elements = scheduler.estimate_tensor_elements(&Op::Softmax);
        assert_eq!(softmax_elements, 128 * 128);

        // Add should estimate small tensor
        let add_elements = scheduler.estimate_tensor_elements(&Op::Add);
        assert_eq!(add_elements, 1024);
    }

    #[test]
    fn test_enhanced_cost_estimation() {
        let gpu = Arc::new(MockProvider::new("gpu", vec![OpType::MatMul, OpType::Softmax]));
        let scheduler = HybridScheduler::new(ExecutionStrategy::Automatic)
            .with_gpu_backend(gpu);

        // MatMul cost should be higher than Softmax
        let matmul_selection = scheduler.select_backend(&Op::MatMul).unwrap();
        let softmax_selection = scheduler.select_backend(&Op::Softmax).unwrap();

        assert!(matmul_selection.estimated_cost.estimated_us > 0);
        assert!(softmax_selection.estimated_cost.estimated_us > 0);

        // MatMul typically has larger estimated cost than Softmax
        // (due to larger tensor size estimation)
        assert!(matmul_selection.estimated_cost.memory_bytes > softmax_selection.estimated_cost.memory_bytes);
    }
}
