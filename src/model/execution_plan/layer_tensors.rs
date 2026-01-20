//! Layer tensor lazy loading and mapping
//!
//! Handles creation of layer plans with lazy tensor handles.

use crate::backend::{HipError, HipResult};
use crate::loader::lazy_tensor::LazyTensor;
use crate::model::config::ModelConfig;
use crate::model::execution_plan::{Architecture, LayerPlan};
use std::collections::HashMap;
use std::sync::Arc;

/// Create layer plan with LazyTensor handles
pub fn create_layer_plan_lazy(
    _config: &ModelConfig,
    lazy_tensors: &HashMap<String, LazyTensor>,
    layer_idx: usize,
    architecture: &Architecture,
) -> HipResult<LayerPlan> {
    eprintln!(
        ">>> CREATE_LAYER_PLAN_LAZY: layer_idx={}, starting...",
        layer_idx
    );
    let prefix = architecture.layer_prefix(layer_idx);
    eprintln!(">>> CREATE_LAYER_PLAN_LAZY: prefix='{}'", prefix);

    // Helper to get lazy tensor or error
    let get_lazy = |name: &str| -> HipResult<Arc<LazyTensor>> {
        lazy_tensors
            .get(name)
            .cloned()
            .map(Arc::new)
            .ok_or_else(|| HipError::GenericError(format!("Tensor '{}' not found", name)))
    };

    let get_lazy_optional = |name: &str| -> Option<Arc<LazyTensor>> {
        lazy_tensors.get(name).cloned().map(Arc::new)
    };

    // Detect attention tensor format:
    // - Some models (LLaMA) use fused QKV: attn_qkv.weight
    // - Some models (Qwen2) use separate: attn_q.weight, attn_k.weight, attn_v.weight
    let qkv_key = &format!("{}.attn_qkv.weight", prefix);
    let q_key = &format!("{}.attn_q.weight", prefix);
    let k_key = &format!("{}.attn_k.weight", prefix);
    let v_key = &format!("{}.attn_v.weight", prefix);

    let has_fused_qkv = lazy_tensors.contains_key(qkv_key);
    let has_q = lazy_tensors.contains_key(q_key);
    let has_k = lazy_tensors.contains_key(k_key);
    let has_v = lazy_tensors.contains_key(v_key);
    let has_separate_qkv = has_q && has_k && has_v;

    eprintln!(">>> Layer {}: Detection - prefix='{}', has_fused_qkv={}, has_separate_qkv={} (q={}, k={}, v={})",
             layer_idx, prefix, has_fused_qkv, has_separate_qkv, has_q, has_k, has_v);

    let (qkv_weight, q_weight, k_weight, v_weight, qkv_bias, q_bias, k_bias, v_bias) =
        if has_fused_qkv {
            // Model uses fused QKV weight
            eprintln!(">>> Layer {}: Using fused QKV weight", layer_idx);
            (
                get_lazy(&format!("{}.attn_qkv.weight", prefix))?,
                None,
                None,
                None,
                get_lazy_optional(&format!("{}.attn_qkv.bias", prefix)),
                None,
                None,
                None,
            )
        } else if has_separate_qkv {
            // Model uses separate Q, K, V weights (e.g., Qwen2 with GQA)
            eprintln!(">>> Layer {}: Using separate Q, K, V weights", layer_idx);
            let q_weight = get_lazy(&format!("{}.attn_q.weight", prefix))?;
            let k_weight = get_lazy(&format!("{}.attn_k.weight", prefix))?;
            let v_weight = get_lazy(&format!("{}.attn_v.weight", prefix))?;
            let q_bias = get_lazy_optional(&format!("{}.attn_q.bias", prefix));
            let k_bias = get_lazy_optional(&format!("{}.attn_k.bias", prefix));
            let v_bias = get_lazy_optional(&format!("{}.attn_v.bias", prefix));

            // Create a placeholder qkv_weight (required by LayerPlan struct)
            // It won't be used since q_weight, k_weight, v_weight are present
            let qkv_weight = q_weight.clone();

            (
                qkv_weight,
                Some(q_weight),
                Some(k_weight),
                Some(v_weight),
                None,
                q_bias,
                k_bias,
                v_bias,
            )
        } else {
            return Err(HipError::GenericError(format!(
            "Layer {}: Neither fused QKV ({0}.attn_qkv.weight) nor separate Q,K,V ({0}.attn_q/k/v.weight) found",
            prefix
        )));
        };

    let o_proj = get_lazy(&format!("{}.attn_output.weight", prefix))?;
    let o_proj_bias = get_lazy_optional(&format!("{}.attn_output.bias", prefix));

    // Map MLP weights
    let mlp_gate = get_lazy(&format!("{}.ffn_gate.weight", prefix))?;
    let mlp_up = get_lazy(&format!("{}.ffn_up.weight", prefix))?;
    let mlp_down = get_lazy(&format!("{}.ffn_down.weight", prefix))?;

    // Map layer norm weights
    let norm1_weight = get_lazy(&format!("{}.attn_norm.weight", prefix))?;
    let norm1_bias = get_lazy_optional(&format!("{}.attn_norm.bias", prefix));
    let norm2_weight = get_lazy(&format!("{}.ffn_norm.weight", prefix))?;
    let norm2_bias = get_lazy_optional(&format!("{}.ffn_norm.bias", prefix));

    Ok(LayerPlan {
        qkv_weight,
        q_weight,
        k_weight,
        v_weight,
        qkv_bias,
        q_bias,
        k_bias,
        v_bias,
        o_proj,
        o_proj_bias,
        mlp_gate_proj: mlp_gate.clone(),
        mlp_up_proj: mlp_up,
        mlp_down_proj: mlp_down.clone(),
        mlp_fc1: mlp_gate.clone(),
        mlp_fc1_bias: None,
        mlp_fc2: mlp_down,
        mlp_fc2_bias: None,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
    })
}
