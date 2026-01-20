// GGML Integration for ExecutionPlan
//
// This file contains GGML-specific methods that are implemented on ExecutionPlan.
// The main ExecutionPlan struct is defined in types.rs.
//
// Phase 25-05: After decomposition, this file only contains GGML graph-based
// execution methods. Core functionality has been moved to focused modules:
// - types.rs: ExecutionPlan struct, LoadingStats
// - embedding.rs: Embedding/LM head operations
// - layer_tensors.rs: Layer tensor lazy loading
// - rope.rs: RoPE caching
// - matmul.rs: Matmul helpers
// - execute.rs: Layer execution logic

use super::types::ExecutionPlan;
use super::ggml_plan::{EmbeddingGgmlPlan, RopeCache, LayerGgmlPlan};
use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::ggml::{Graph, Layout, Op, TensorDesc, DType, GgmlBackend};
use crate::ggml::hip_backend::HipGgmlBackend;
use crate::ggml::executor::execute_graph;
use crate::loader::TensorShape;
use crate::model::kv_cache::KVCache;
use once_cell::sync::OnceCell;
use std::sync::Mutex as StdMutex;
use std::sync::Arc;

impl ExecutionPlan {
    /// Build GGML embedding plan for token lookup (legacy method)
    ///
    /// NOTE: This method is kept for GGML decode path compatibility.
    /// The standard embedding_lookup is now in embedding.rs.
    fn build_embedding_plan(
        &self,
        backend: &HipBackend,
        embedding_weights: &DeviceTensor,
    ) -> HipResult<EmbeddingGgmlPlan> {
        let embed_shape = embedding_weights.shape().dims();
        eprintln!(">>> build_embedding_plan: embed_shape={:?}", embed_shape);
        eprintln!(">>> build_embedding_plan: self.config.hidden_size={}, self.config.vocab_size={}",
            self.config_internal().hidden_size, self.config_internal().vocab_size);

        if embed_shape.len() != 2 {
            return Err(HipError::GenericError(format!(
                "Embedding weight shape must be 2D, got {:?}",
                embed_shape
            )));
        }

        let (n_embd, vocab_size) = (embed_shape[1], embed_shape[0]);
        eprintln!(">>> build_embedding_plan: n_embd={}, vocab_size={}", n_embd, vocab_size);

        if n_embd != self.config_internal().hidden_size {
            return Err(HipError::GenericError(format!(
                "Embedding hidden size mismatch: expected {}, got {}",
                self.config_internal().hidden_size, n_embd
            )));
        }

        let max_seq_len = std::cmp::max(1, self.config_internal().max_position_embeddings);
        let tokens_bytes = max_seq_len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                HipError::GenericError("Token buffer size overflow".to_string())
            })?;
        let output_elems = max_seq_len.checked_mul(n_embd).ok_or_else(|| {
            HipError::GenericError("Output buffer element overflow".to_string())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::GenericError("Output buffer size overflow".to_string()))?;

        let mut graph = Graph::new();
        let weights_id = graph.add_tensor(TensorDesc::new(
            embed_shape.to_vec(),
            DType::F32,
            Layout::RowMajor,
        ));
        let tokens_id = graph.add_tensor(TensorDesc::new(
            vec![max_seq_len],
            DType::U32,
            Layout::RowMajor,
        ));
        let output_id = graph.add_tensor(TensorDesc::new(
            vec![max_seq_len, n_embd],
            DType::F32,
            Layout::RowMajor,
        ));
        graph.add_node(Op::GetRows, vec![weights_id, tokens_id], vec![output_id]);

        let mut ggml_backend = HipGgmlBackend::new(Arc::clone(self.backend()));
        let weights_desc = graph.tensors[weights_id.0].clone();
        ggml_backend
            .bind(&weights_desc, embedding_weights.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("GetRows bind weights failed: {:?}", e)))?;

        let tokens_desc = graph.tensors[tokens_id.0].clone();
        let tokens_buffer = backend.allocate_buffer(tokens_bytes)?;
        ggml_backend
            .bind(&tokens_desc, tokens_buffer.clone())
            .map_err(|e| HipError::GenericError(format!("GetRows bind tokens failed: {:?}", e)))?;

        let output_desc = graph.tensors[output_id.0].clone();
        let output_buffer = backend.allocate_buffer(output_bytes)?;
        ggml_backend
            .bind(&output_desc, output_buffer.clone())
            .map_err(|e| HipError::GenericError(format!("GetRows bind output failed: {:?}", e)))?;

        Ok(EmbeddingGgmlPlan {
            graph,
            backend: StdMutex::new(ggml_backend),
            tokens_buffer,
            output_buffer,
            max_seq_len,
            hidden_size: n_embd,
        })
    }

    /// Get RoPE cache (GGML decode path)
    fn rope_cache_ggml(&self) -> HipResult<Option<&RopeCache>> {
        let Some(ref position_handler) = self.position_handler() else {
            return Ok(None);
        };
        let Some(rope) = position_handler.rope() else {
            return Ok(None);
        };

        let cache = self.rope_cache().get_or_try_init(|| {
            let half_dim = rope.config().head_dim / 2;
            let max_seq_len = rope.config().max_seq_len;
            let cos_shape = TensorShape::from_dims(&[max_seq_len, half_dim]);
            let sin_shape = TensorShape::from_dims(&[max_seq_len, half_dim]);
            let cos_tensor =
                DeviceTensor::from_host_vec(self.backend(), rope.cos().to_vec(), cos_shape)?;
            let sin_tensor =
                DeviceTensor::from_host_vec(self.backend(), rope.sin().to_vec(), sin_shape)?;
            Ok::<RopeCache, HipError>(RopeCache {
                cos: cos_tensor,
                sin: sin_tensor,
                half_dim,
                max_seq_len,
            })
        })?;

        Ok(Some(cache))
    }

    /// Build GGML layer plans for decode path
    fn build_layer_ggml_plans(&self, _backend: &HipBackend) -> HipResult<Vec<LayerGgmlPlan>> {
        let mut plans = Vec::with_capacity(self.layers_internal().len());
        let rope_cache = self.rope_cache_ggml()?;

        for layer_plan in self.layers_internal() {
            let qkv_weight = self.get_or_load_tensor(&layer_plan.qkv_weight)?;
            let q_weight = layer_plan
                .q_weight
                .as_ref()
                .map(|w| self.get_or_load_tensor(w))
                .transpose()?;
            let k_weight = layer_plan
                .k_weight
                .as_ref()
                .map(|w| self.get_or_load_tensor(w))
                .transpose()?;
            let v_weight = layer_plan
                .v_weight
                .as_ref()
                .map(|w| self.get_or_load_tensor(w))
                .transpose()?;
            let qkv_bias = layer_plan
                .qkv_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let q_bias = layer_plan
                .q_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let k_bias = layer_plan
                .k_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let v_bias = layer_plan
                .v_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;

            let o_proj = self.get_or_load_tensor(&layer_plan.o_proj)?;
            let o_proj_bias = layer_plan
                .o_proj_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let mlp_gate_proj = self.get_or_load_tensor(&layer_plan.mlp_gate_proj)?;
            let mlp_up_proj = self.get_or_load_tensor(&layer_plan.mlp_up_proj)?;
            let mlp_down_proj = self.get_or_load_tensor(&layer_plan.mlp_down_proj)?;
            let norm1_weight = self.get_or_load_tensor(&layer_plan.norm1_weight)?;
            let norm1_bias = layer_plan
                .norm1_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;
            let norm2_weight = self.get_or_load_tensor(&layer_plan.norm2_weight)?;
            let norm2_bias = layer_plan
                .norm2_bias
                .as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?;

            let use_separate_qkv = q_weight.is_some() && k_weight.is_some() && v_weight.is_some();

            let num_heads = self.config_internal().num_attention_heads;
            let head_dim = self.config_internal().head_dim;
            let hidden_size = self.config_internal().hidden_size;
            let max_seq_len = self.config_internal().max_position_embeddings.max(1);
            let num_kv_heads = self.config_internal().num_kv_heads.unwrap_or(num_heads);
            let kv_hidden_size = num_kv_heads * head_dim;
            eprintln!(">>> GQA: num_heads={}, num_kv_heads={}, kv_hidden_size={}, head_dim={}",
                     num_heads, num_kv_heads, kv_hidden_size, head_dim);

            let mut graph = Graph::new();

            let input_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let norm1_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let norm2_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));

            let norm1_w_id = graph.add_tensor(TensorDesc::new(
                norm1_weight.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let norm1_bias_id = norm1_bias.as_ref().map(|bias| {
                graph.add_tensor(TensorDesc::new(
                    bias.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ))
            });
            let mut norm1_inputs = vec![input_id, norm1_w_id];
            if let Some(bias_id) = norm1_bias_id {
                norm1_inputs.push(bias_id);
            }
            graph.add_node(Op::LayerNorm { eps: 1e-6 }, norm1_inputs, vec![norm1_id]);

            let q_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let k_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, kv_hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let v_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, kv_hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let q_id = graph.add_tensor(
                TensorDesc::new(vec![1, num_heads, head_dim], DType::F32, Layout::RowMajor)
                    .view_of(q_flat_id, 0),
            );
            let k_id = graph.add_tensor(
                TensorDesc::new(vec![1, num_kv_heads, head_dim], DType::F32, Layout::RowMajor)
                    .view_of(k_flat_id, 0),
            );
            let v_id = graph.add_tensor(
                TensorDesc::new(vec![1, num_kv_heads, head_dim], DType::F32, Layout::RowMajor)
                    .view_of(v_flat_id, 0),
            );

            let mut q_w_id = None;
            let mut k_w_id = None;
            let mut v_w_id = None;
            let mut qkv_w_id = None;
            let mut qkv_bias_id = None;
            let mut q_bias_id = None;
            let mut k_bias_id = None;
            let mut v_bias_id = None;
            let mut o_proj_bias_id = None;

            let (q_src_id, k_src_id, v_src_id) = if use_separate_qkv {
                let q_ref = q_weight.as_ref().expect("q_weight is Some when use_separate_qkv is true");
                let k_ref = k_weight.as_ref().expect("k_weight is Some when use_separate_qkv is true");
                let v_ref = v_weight.as_ref().expect("v_weight is Some when use_separate_qkv is true");

                let q_id_local = graph.add_tensor(TensorDesc::new(
                    q_ref.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                let k_id_local = graph.add_tensor(TensorDesc::new(
                    k_ref.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                let v_id_local = graph.add_tensor(TensorDesc::new(
                    v_ref.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                q_w_id = Some(q_id_local);
                k_w_id = Some(k_id_local);
                v_w_id = Some(v_id_local);
                if q_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    q_bias_id = Some(bias_id);
                    let q_mm_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    graph.add_node(Op::MatMul, vec![norm1_id, q_id_local], vec![q_mm_id]);
                    graph.add_node(Op::Add, vec![q_mm_id, bias_id], vec![q_flat_id]);
                } else {
                    graph.add_node(Op::MatMul, vec![norm1_id, q_id_local], vec![q_flat_id]);
                }

                if k_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, kv_hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    k_bias_id = Some(bias_id);
                    let k_mm_id = graph.add_tensor(TensorDesc::new(
                        vec![1, kv_hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    graph.add_node(Op::MatMul, vec![norm1_id, k_id_local], vec![k_mm_id]);
                    graph.add_node(Op::Add, vec![k_mm_id, bias_id], vec![k_flat_id]);
                } else {
                    graph.add_node(Op::MatMul, vec![norm1_id, k_id_local], vec![k_flat_id]);
                }

                if v_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, kv_hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    v_bias_id = Some(bias_id);
                    let v_mm_id = graph.add_tensor(TensorDesc::new(
                        vec![1, kv_hidden_size],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    graph.add_node(Op::MatMul, vec![norm1_id, v_id_local], vec![v_mm_id]);
                    graph.add_node(Op::Add, vec![v_mm_id, bias_id], vec![v_flat_id]);
                } else {
                    graph.add_node(Op::MatMul, vec![norm1_id, v_id_local], vec![v_flat_id]);
                }

                (q_flat_id, k_flat_id, v_flat_id)
            } else {
                let qkv_id = graph.add_tensor(TensorDesc::new(
                    vec![1, hidden_size * 3],
                    DType::F32,
                    Layout::RowMajor,
                ));
                let qkv_id_local = graph.add_tensor(TensorDesc::new(
                    qkv_weight.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ));
                qkv_w_id = Some(qkv_id_local);
                let qkv_mm_id = if qkv_bias.is_some() {
                    graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size * 3],
                        DType::F32,
                        Layout::RowMajor,
                    ))
                } else {
                    qkv_id
                };
                graph.add_node(Op::MatMul, vec![norm1_id, qkv_id_local], vec![qkv_mm_id]);
                if qkv_bias.is_some() {
                    let bias_id = graph.add_tensor(TensorDesc::new(
                        vec![1, hidden_size * 3],
                        DType::F32,
                        Layout::RowMajor,
                    ));
                    qkv_bias_id = Some(bias_id);
                    graph.add_node(Op::Add, vec![qkv_mm_id, bias_id], vec![qkv_id]);
                }
                graph.add_node(
                    Op::SplitQkv,
                    vec![qkv_id],
                    vec![q_flat_id, k_flat_id, v_flat_id],
                );
                (q_flat_id, k_flat_id, v_flat_id)
            };

            let cos_id = graph.add_tensor(TensorDesc::new(
                vec![1, head_dim / 2],
                DType::F32,
                Layout::RowMajor,
            ));
            let sin_id = graph.add_tensor(TensorDesc::new(
                vec![1, head_dim / 2],
                DType::F32,
                Layout::RowMajor,
            ));

            let q_rope_id = graph.add_tensor(TensorDesc::new(
                vec![1, num_heads, head_dim],
                DType::F32,
                Layout::RowMajor,
            ));
            let k_rope_id = graph.add_tensor(TensorDesc::new(
                vec![1, num_heads, head_dim],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Reshape, vec![q_src_id], vec![q_id]);
            graph.add_node(Op::Reshape, vec![k_src_id], vec![k_id]);
            graph.add_node(Op::Reshape, vec![v_id], vec![v_id]);

            if rope_cache.is_some() {
                graph.add_node(Op::Rope, vec![q_id, cos_id, sin_id], vec![q_rope_id]);
                graph.add_node(Op::Rope, vec![k_id, cos_id, sin_id], vec![k_rope_id]);
            } else {
                graph.add_node(Op::Copy, vec![q_id], vec![q_rope_id]);
                graph.add_node(Op::Copy, vec![k_id], vec![k_rope_id]);
            }

            let kv_read_k_id = graph.add_tensor(
                TensorDesc::new(
                    vec![max_seq_len, num_kv_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );
            let kv_read_v_id = graph.add_tensor(
                TensorDesc::new(
                    vec![max_seq_len, num_kv_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );
            let kv_write_k_id = graph.add_tensor(
                TensorDesc::new(
                    vec![1, num_kv_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );
            let kv_write_v_id = graph.add_tensor(
                TensorDesc::new(
                    vec![1, num_kv_heads, head_dim],
                    DType::F32,
                    Layout::RowMajor,
                )
                .view_of(crate::ggml::TensorId(0), 0),
            );

            graph.add_node(Op::Copy, vec![k_rope_id], vec![kv_write_k_id]);
            graph.add_node(Op::Copy, vec![v_id], vec![kv_write_v_id]);

            let scores_id = graph.add_tensor(TensorDesc::new(
                vec![1, max_seq_len],
                DType::F32,
                Layout::RowMajor,
            ));
            let softmax_id = graph.add_tensor(TensorDesc::new(
                vec![1, max_seq_len],
                DType::F32,
                Layout::RowMajor,
            ));
            let attn_out_id = graph.add_tensor(TensorDesc::new(
                vec![1, num_heads, head_dim],
                DType::F32,
                Layout::RowMajor,
            ));

            graph.add_node(
                Op::Attention,
                vec![q_rope_id, kv_read_k_id, kv_read_v_id, scores_id, softmax_id],
                vec![attn_out_id],
            );

            let attn_flat_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Reshape, vec![attn_out_id], vec![attn_flat_id]);

            let o_proj_id = graph.add_tensor(TensorDesc::new(
                o_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let attn_proj_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            let attn_proj_mm_id = if o_proj_bias.is_some() {
                graph.add_tensor(TensorDesc::new(
                    vec![1, hidden_size],
                    DType::F32,
                    Layout::RowMajor,
                ))
            } else {
                attn_proj_id
            };
            graph.add_node(
                Op::MatMul,
                vec![attn_flat_id, o_proj_id],
                vec![attn_proj_mm_id],
            );
            if o_proj_bias.is_some() {
                let bias_id = graph.add_tensor(TensorDesc::new(
                    vec![1, hidden_size],
                    DType::F32,
                    Layout::RowMajor,
                ));
                o_proj_bias_id = Some(bias_id);
                graph.add_node(Op::Add, vec![attn_proj_mm_id, bias_id], vec![attn_proj_id]);
            }

            let attn_residual_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Add, vec![attn_proj_id, input_id], vec![attn_residual_id]);

            let norm2_w_id = graph.add_tensor(TensorDesc::new(
                norm2_weight.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let norm2_bias_id = norm2_bias.as_ref().map(|bias| {
                graph.add_tensor(TensorDesc::new(
                    bias.shape().dims().to_vec(),
                    DType::F32,
                    Layout::RowMajor,
                ))
            });
            let mut norm2_inputs = vec![attn_residual_id, norm2_w_id];
            if let Some(bias_id) = norm2_bias_id {
                norm2_inputs.push(bias_id);
            }
            graph.add_node(Op::LayerNorm { eps: 1e-6 }, norm2_inputs, vec![norm2_id]);

            let mlp_gate_id = graph.add_tensor(TensorDesc::new(
                mlp_gate_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let mlp_up_id = graph.add_tensor(TensorDesc::new(
                mlp_up_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let mlp_down_id = graph.add_tensor(TensorDesc::new(
                mlp_down_proj.shape().dims().to_vec(),
                DType::F32,
                Layout::RowMajor,
            ));
            let mlp_out_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(
                Op::MlpSwiglu,
                vec![norm2_id, mlp_gate_id, mlp_up_id, mlp_down_id],
                vec![mlp_out_id],
            );

            let output_id = graph.add_tensor(TensorDesc::new(
                vec![1, hidden_size],
                DType::F32,
                Layout::RowMajor,
            ));
            graph.add_node(Op::Add, vec![mlp_out_id, attn_residual_id], vec![output_id]);

            let mut ggml_backend = HipGgmlBackend::new(Arc::clone(self.backend()));
            ggml_backend.bind(&graph.tensors[norm1_w_id.0], norm1_weight.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[norm2_w_id.0], norm2_weight.buffer().clone())?;
            if let (Some(bias), Some(bias_id)) = (norm1_bias.as_ref(), norm1_bias_id) {
                ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
            }
            if let (Some(bias), Some(bias_id)) = (norm2_bias.as_ref(), norm2_bias_id) {
                ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
            }
            ggml_backend.bind(&graph.tensors[o_proj_id.0], o_proj.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[mlp_gate_id.0], mlp_gate_proj.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[mlp_up_id.0], mlp_up_proj.buffer().clone())?;
            ggml_backend.bind(&graph.tensors[mlp_down_id.0], mlp_down_proj.buffer().clone())?;

            if use_separate_qkv {
                let q_buf = q_weight.as_ref().expect("q_weight is Some when use_separate_qkv is true");
                let k_buf = k_weight.as_ref().expect("k_weight is Some when use_separate_qkv is true");
                let v_buf = v_weight.as_ref().expect("v_weight is Some when use_separate_qkv is true");

                ggml_backend.bind(
                    &graph.tensors[q_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing Q weight id".to_string())
                    })?.0],
                    q_buf.buffer().clone(),
                )?;
                ggml_backend.bind(
                    &graph.tensors[k_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing K weight id".to_string())
                    })?.0],
                    k_buf.buffer().clone(),
                )?;
                ggml_backend.bind(
                    &graph.tensors[v_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing V weight id".to_string())
                    })?.0],
                    v_buf.buffer().clone(),
                )?;
                if let (Some(bias), Some(bias_id)) = (q_bias.as_ref(), q_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
                if let (Some(bias), Some(bias_id)) = (k_bias.as_ref(), k_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
                if let (Some(bias), Some(bias_id)) = (v_bias.as_ref(), v_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
            } else {
                ggml_backend.bind(
                    &graph.tensors[qkv_w_id.ok_or_else(|| {
                        HipError::GenericError("Missing QKV weight id".to_string())
                    })?.0],
                    qkv_weight.buffer().clone(),
                )?;
                if let (Some(bias), Some(bias_id)) = (qkv_bias.as_ref(), qkv_bias_id) {
                    ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
                }
            }

            if let Some(rope_cache) = rope_cache {
                ggml_backend.bind(&graph.tensors[cos_id.0], rope_cache.cos.buffer().clone())?;
                ggml_backend.bind(&graph.tensors[sin_id.0], rope_cache.sin.buffer().clone())?;
            }

            if let (Some(bias), Some(bias_id)) = (o_proj_bias.as_ref(), o_proj_bias_id) {
                ggml_backend.bind(&graph.tensors[bias_id.0], bias.buffer().clone())?;
            }

            let scores_bytes = max_seq_len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| HipError::GenericError("Scores buffer size overflow".to_string()))?;
            let scores_buffer = self.backend().allocate_buffer(scores_bytes)?;
            let softmax_buffer = self.backend().allocate_buffer(scores_bytes)?;
            ggml_backend.bind(&graph.tensors[scores_id.0], scores_buffer)?;
            ggml_backend.bind(&graph.tensors[softmax_id.0], softmax_buffer)?;

            plans.push(LayerGgmlPlan {
                graph: StdMutex::new(graph),
                backend: StdMutex::new(ggml_backend),
                input_id,
                output_id,
                kv_read_k_id,
                kv_read_v_id,
                kv_write_k_id,
                kv_write_v_id,
                scores_id,
                softmax_id,
                cos_id,
                sin_id,
                num_heads,
                head_dim,
                hidden_size,
                max_seq_len,
            });
        }

        Ok(plans)
    }

    /// Forward pass through a single layer using GGML decode path
    pub(crate) fn forward_layer_ggml_decode(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        let plans = self
            .layer_ggml_plans()
            .get_or_try_init(|| self.build_layer_ggml_plans(backend))?;
        let plan = plans.get(layer_idx).ok_or_else(|| {
            HipError::GenericError(format!("Missing ggml plan for layer {}", layer_idx))
        })?;

        let current_len = kv_cache.current_seq_len(layer_idx)?;
        if current_len >= plan.max_seq_len {
            return Err(HipError::GenericError(format!(
                "KV cache capacity exceeded at layer {} (len={}, max={})",
                layer_idx, current_len, plan.max_seq_len
            )));
        }
        let new_len = current_len + 1;
        let stride = plan.num_heads * plan.head_dim;
        let elem_bytes = std::mem::size_of::<f32>();
        let write_bytes = stride
            .checked_mul(elem_bytes)
            .ok_or_else(|| HipError::GenericError("KV write size overflow".to_string()))?;
        let _read_bytes = new_len
            .checked_mul(stride)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| HipError::GenericError("KV read size overflow".to_string()))?;
        let write_offset = current_len
            .checked_mul(stride)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or_else(|| HipError::GenericError("KV write offset overflow".to_string()))?;

        let (kv_keys, kv_values) = kv_cache.get(layer_idx)
            .map_err(|e| HipError::GenericError(format!("KV cache get failed: {}", e)))?;

        let graph = plan
            .graph
            .lock()
            .map_err(|_| HipError::GenericError("Layer graph lock poisoned".to_string()))?;

        let mut ggml_backend = plan
            .backend
            .lock()
            .map_err(|_| HipError::GenericError("Layer backend lock poisoned".to_string()))?;

        ggml_backend
            .bind(&graph.tensors[plan.input_id.0], hidden_states.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("Bind input failed: {:?}", e)))?;

        ggml_backend
            .bind(&graph.tensors[plan.kv_read_k_id.0], kv_keys.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("Bind KV read K failed: {:?}", e)))?;
        ggml_backend
            .bind(&graph.tensors[plan.kv_read_v_id.0], kv_values.buffer().clone())
            .map_err(|e| HipError::GenericError(format!("Bind KV read V failed: {:?}", e)))?;

        let (kv_keys_buffer, kv_values_buffer, layer_offset) = kv_cache.get_layer_buffers(layer_idx)
            .map_err(|e| HipError::GenericError(format!("KV cache get_layer_buffers failed: {}", e)))?;

        let kv_write_k_view = kv_keys_buffer.sub_buffer_view(layer_offset + write_offset, write_bytes)?;
        let kv_write_v_view = kv_values_buffer.sub_buffer_view(layer_offset + write_offset, write_bytes)?;
        ggml_backend
            .bind(&graph.tensors[plan.kv_write_k_id.0], kv_write_k_view)
            .map_err(|e| HipError::GenericError(format!("Bind KV write K failed: {:?}", e)))?;
        ggml_backend
            .bind(&graph.tensors[plan.kv_write_v_id.0], kv_write_v_view)
            .map_err(|e| HipError::GenericError(format!("Bind KV write V failed: {:?}", e)))?;

        if let Some(rope_cache) = self.rope_cache_ggml()? {
            let rope_offset = current_len
                .checked_mul(rope_cache.half_dim)
                .and_then(|v| v.checked_mul(elem_bytes))
                .ok_or_else(|| HipError::GenericError("RoPE offset overflow".to_string()))?;
            let rope_bytes = rope_cache
                .half_dim
                .checked_mul(elem_bytes)
                .ok_or_else(|| HipError::GenericError("RoPE view size overflow".to_string()))?;

            let cos_view = rope_cache
                .cos
                .buffer()
                .sub_buffer_view(rope_offset, rope_bytes)?;
            let sin_view = rope_cache
                .sin
                .buffer()
                .sub_buffer_view(rope_offset, rope_bytes)?;

            ggml_backend
                .bind(&graph.tensors[plan.cos_id.0], cos_view)
                .map_err(|e| HipError::GenericError(format!("Bind RoPE cos failed: {:?}", e)))?;
            ggml_backend
                .bind(&graph.tensors[plan.sin_id.0], sin_view)
                .map_err(|e| HipError::GenericError(format!("Bind RoPE sin failed: {:?}", e)))?;
        }

        execute_graph(&mut *ggml_backend, &graph)
            .map_err(|e| HipError::GenericError(format!("GGML layer exec failed: {:?}", e)))?;

        kv_cache.advance(layer_idx, 1)?;

        let output_buf = ggml_backend
            .buffer(plan.output_id)
            .ok_or_else(|| HipError::GenericError("Missing output buffer".to_string()))?
            .clone();
        let output_shape = TensorShape::from_dims(&[1, plan.hidden_size]);
        DeviceTensor::from_buffer(backend, output_buf, output_shape)
    }
}
