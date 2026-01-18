//! Per-layer execution plan

use crate::loader::lazy_tensor::LazyTensor;
use std::sync::Arc;

/// Execution plan for a single transformer layer
///
/// Contains lazy tensor handles for all weights needed for layer execution:
/// - QKV projection (fused Q, K, V OR separate Q, K, V)
/// - Output projection
/// - MLP layers (gate_proj, up_proj, down_proj for GLM)
/// - Layer normalization weights
#[derive(Debug, Clone)]
pub struct LayerPlan {
    /// Fused QKV projection weight matrix [3 * hidden_size, hidden_size]
    /// NOTE: Some models (e.g., Qwen2) use separate Q, K, V weights instead.
    /// Check q_weight, k_weight, v_weight below - if those are present, use them.
    pub qkv_weight: Arc<LazyTensor>,

    /// Separate Q projection weight [hidden_size, hidden_size]
    /// Present when model uses separate Q, K, V projections (e.g., Qwen2)
    pub q_weight: Option<Arc<LazyTensor>>,
    /// Separate K projection weight [hidden_size, kv_dim]
    pub k_weight: Option<Arc<LazyTensor>>,
    /// Separate V projection weight [hidden_size, kv_dim]
    pub v_weight: Option<Arc<LazyTensor>>,

    /// Optional QKV bias [3 * hidden_size] (for fused) or separate biases
    pub qkv_bias: Option<Arc<LazyTensor>>,
    pub q_bias: Option<Arc<LazyTensor>>,
    pub k_bias: Option<Arc<LazyTensor>>,
    pub v_bias: Option<Arc<LazyTensor>>,

    /// Output projection weight [hidden_size, hidden_size]
    pub o_proj: Arc<LazyTensor>,
    /// Optional output projection bias [hidden_size]
    pub o_proj_bias: Option<Arc<LazyTensor>>,
    /// MLP gate projection weight [intermediate_size, hidden_size] (GLM)
    pub mlp_gate_proj: Arc<LazyTensor>,
    /// MLP up projection weight [intermediate_size, hidden_size] (GLM)
    pub mlp_up_proj: Arc<LazyTensor>,
    /// MLP down projection weight [hidden_size, intermediate_size] (GLM)
    pub mlp_down_proj: Arc<LazyTensor>,
    /// Legacy MLP first layer weight [intermediate_size, hidden_size]
    pub mlp_fc1: Arc<LazyTensor>,
    /// Optional MLP first layer bias [intermediate_size]
    pub mlp_fc1_bias: Option<Arc<LazyTensor>>,
    /// Legacy MLP second layer weight [hidden_size, intermediate_size]
    pub mlp_fc2: Arc<LazyTensor>,
    /// Optional MLP second layer bias [hidden_size]
    pub mlp_fc2_bias: Option<Arc<LazyTensor>>,
    /// First layer norm weight [hidden_size]
    pub norm1_weight: Arc<LazyTensor>,
    /// Optional first layer norm bias [hidden_size]
    pub norm1_bias: Option<Arc<LazyTensor>>,
    /// Second layer norm weight [hidden_size]
    pub norm2_weight: Arc<LazyTensor>,
    /// Optional second layer norm bias [hidden_size]
    pub norm2_bias: Option<Arc<LazyTensor>>,
}

#[allow(dead_code)]
impl LayerPlan {
    /// Create a new layer plan
    ///
    /// **DEPRECATED**: This method is deprecated due to Phase 2 lazy loading.
    /// Layer plans are now created via `ExecutionPlan::from_gguf()` which properly
    /// initializes lazy tensor handles.
    #[deprecated(note = "Layer plans are created by ExecutionPlan::from_gguf()")]
    pub fn new() -> Self {
        // This is a stub - actual creation happens in ExecutionPlan::from_gguf()
        // This is kept for API compatibility but will always fail
        panic!("LayerPlan::new() is deprecated. Use ExecutionPlan::from_gguf() instead.")
    }
}
