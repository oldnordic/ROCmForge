//! GGML graph structures for execution

use crate::backend::DeviceTensor;
use crate::ggml::{Graph, TensorId};
use crate::ggml::hip_backend::HipGgmlBackend;
use std::sync::Mutex as StdMutex;

/// Cached embedding computation graph
#[derive(Debug)]
pub struct EmbeddingGgmlPlan {
    pub graph: Graph,
    pub backend: StdMutex<HipGgmlBackend>,
    pub tokens_buffer: crate::backend::HipBuffer,
    pub output_buffer: crate::backend::HipBuffer,
    pub max_seq_len: usize,
    pub hidden_size: usize,
}

/// Cached RoPE tables on GPU
#[derive(Debug)]
pub struct RopeCache {
    pub cos: DeviceTensor,
    pub sin: DeviceTensor,
    pub half_dim: usize,
    pub max_seq_len: usize,
}

/// Cached layer computation graph
#[derive(Debug)]
pub struct LayerGgmlPlan {
    pub graph: StdMutex<Graph>,
    pub backend: StdMutex<HipGgmlBackend>,
    pub input_id: TensorId,
    pub output_id: TensorId,
    pub kv_read_k_id: TensorId,
    pub kv_read_v_id: TensorId,
    pub kv_write_k_id: TensorId,
    pub kv_write_v_id: TensorId,
    pub scores_id: TensorId,
    pub softmax_id: TensorId,
    pub cos_id: TensorId,
    pub sin_id: TensorId,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub max_seq_len: usize,
}
