//! HIP backend for ggml IR.

pub mod buffer;
pub mod ops;

use crate::backend::HipBackend;
use crate::ggml::{allocator::TensorAllocator, GgmlBackend, GgmlError, GgmlResult, Op, TensorDesc, TensorId};
use std::collections::HashMap;
use std::sync::Arc;

pub struct HipGgmlBackend {
    backend: Arc<HipBackend>,
    tensors: HashMap<TensorId, (TensorDesc, crate::backend::HipBuffer)>,
    /// Optional tensor allocator for buffer reuse
    allocator: Option<TensorAllocator>,
}

impl std::fmt::Debug for HipGgmlBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipGgmlBackend")
            .field("tensor_count", &self.tensors.len())
            .field("allocator_enabled", &self.allocator.is_some())
            .finish()
    }
}

impl HipGgmlBackend {
    pub fn new(backend: Arc<HipBackend>) -> Self {
        Self {
            backend,
            tensors: HashMap::new(),
            allocator: None,
        }
    }

    /// Enable the tensor allocator for buffer reuse.
    pub fn with_allocator(mut self) -> Self {
        self.allocator = Some(TensorAllocator::new());
        self
    }

    /// Enable the tensor allocator with custom max pool size.
    pub fn with_allocator_config(mut self, max_pool_size: usize) -> Self {
        self.allocator = Some(TensorAllocator::new().with_max_pool_size(max_pool_size));
        self
    }

    /// Check if the allocator is enabled.
    pub fn has_allocator(&self) -> bool {
        self.allocator.is_some()
    }

    /// Get allocator statistics.
    pub fn allocator_stats(&self) -> Option<crate::ggml::allocator::AllocatorStats> {
        self.allocator.as_ref().map(|a| a.stats())
    }

    /// Reset the allocator, clearing all free pools.
    /// Call this between graph executions for optimal reuse.
    pub fn reset_allocator(&mut self) {
        if let Some(alloc) = &mut self.allocator {
            alloc.reset();
        }
    }
}

impl GgmlBackend for HipGgmlBackend {
    type Buffer = crate::backend::HipBuffer;

    fn alloc(&mut self, desc: &TensorDesc) -> GgmlResult<()> {
        let bytes = desc.byte_size();

        // Try to allocate from pool if allocator is enabled
        let buffer = if let Some(alloc) = &mut self.allocator {
            alloc.allocate(bytes, |size| {
                self.backend
                    .allocate_buffer(size)
                    .map_err(|e| e.to_string())
            })
            .map_err(|e| GgmlError::Backend(e))?
        } else {
            self.backend
                .allocate_buffer(bytes)
                .map_err(|e| GgmlError::Backend(e.to_string()))?
        };

        self.tensors.insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn bind(&mut self, desc: &TensorDesc, buffer: Self::Buffer) -> GgmlResult<()> {
        self.tensors.insert(desc.id, (desc.clone(), buffer));
        Ok(())
    }

    fn free(&mut self, id: TensorId) -> GgmlResult<()> {
        if let Some((desc, buffer)) = self.tensors.remove(&id) {
            // Return buffer to allocator if enabled
            if let Some(alloc) = &mut self.allocator {
                alloc.free(buffer, desc.byte_size());
            }
            // Otherwise buffer is dropped (deallocated)
        }
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
        _inputs: &[TensorId],
        _outputs: &[TensorId],
    ) -> GgmlResult<()> {
        eprintln!(">>> execute_op: op={:?}", op);
        match op {
            Op::GetRows => {
                eprintln!(">>> execute_op: GetRows START");
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "GetRows expects 2 inputs and 1 output".to_string(),
                    ));
                }
                eprintln!(">>> execute_op: GetRows input/output count validated");

                let weights_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing weights desc".to_string()))?;
                let tokens_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing tokens desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;
                eprintln!(">>> execute_op: GetRows tensor descriptors obtained");

                if weights_desc.shape.len() != 2 {
                    return Err(GgmlError::InvalidShape(
                        "Weights must be 2D".to_string(),
                    ));
                }
                let (n_embd, vocab_size) = match weights_desc.layout {
                    crate::ggml::Layout::RowMajor => (weights_desc.shape[1], weights_desc.shape[0]),
                    crate::ggml::Layout::ColMajor => (weights_desc.shape[0], weights_desc.shape[1]),
                    crate::ggml::Layout::Strided => {
                        return Err(GgmlError::InvalidLayout(
                            "Strided layout not supported for GetRows".to_string(),
                        ));
                    }
                };
                // Use output_desc.shape[0] for actual token count, not tokens_desc.element_count()
                // tokens buffer may be padded to max_seq_len, but output shape reflects actual tokens
                let actual_n_tokens = output_desc.shape[0];
                let buffer_n_tokens = tokens_desc.element_count();
                eprintln!(">>> execute_op: GetRows n_embd={}, vocab_size={}, actual_n_tokens={}, buffer_n_tokens={}, layout={:?}",
                    n_embd, vocab_size, actual_n_tokens, buffer_n_tokens, weights_desc.layout);

                if output_desc.element_count() != n_embd * actual_n_tokens {
                    return Err(GgmlError::InvalidShape(
                        "Output size does not match n_embd * actual_n_tokens".to_string(),
                    ));
                }

                let weights = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing weights buffer".to_string()))?;
                let tokens_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing tokens buffer".to_string()))?;
                let output = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;
                eprintln!(">>> execute_op: GetRows buffers obtained");

                eprintln!(">>> execute_op: GetRows about to copy tokens to host (actual_n_tokens={})...", actual_n_tokens);
                let mut tokens = vec![0u32; actual_n_tokens];
                tokens_buf
                    .copy_to_host(&mut tokens)
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;
                eprintln!(">>> execute_op: GetRow tokens copied to host (full buffer): {:?}", &tokens[..tokens.len().min(10)]);

                // Truncate to actual non-zero tokens (we use 0-padding)
                // This avoids processing 2047 padding tokens unnecessarily
                let actual_tokens: Vec<u32> = tokens.iter()
                    .take_while(|&&t| t != 0)
                    .copied()
                    .collect();
                eprintln!(">>> execute_op: GetRows actual (non-padded) tokens: {:?}, count={}", actual_tokens, actual_tokens.len());

                crate::ggml::hip_backend::ops::get_rows::validate_token_ids(
                    &actual_tokens,
                    vocab_size,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                eprintln!(">>> execute_op: GetRows token IDs validated");

                eprintln!(">>> execute_op: GetRows about to call get_rows with {} tokens...", actual_tokens.len());
                crate::ggml::hip_backend::ops::get_rows::get_rows(
                    &self.backend,
                    weights,
                    &actual_tokens,
                    n_embd,
                    vocab_size,
                    weights_desc.layout,
                    output,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                eprintln!(">>> execute_op: GetRows complete");
                Ok(())
            }
            Op::MatMul => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "MatMul expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let a_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing A desc".to_string()))?;
                let b_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing B desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                if a_desc.shape.len() != 2 || b_desc.shape.len() != 2 {
                    return Err(GgmlError::InvalidShape(
                        "MatMul expects 2D tensors".to_string(),
                    ));
                }

                let m = a_desc.shape[0] as i32;
                let k = a_desc.shape[1] as i32;
                let b_k = b_desc.shape[0] as i32;
                let n = b_desc.shape[1] as i32;
                if k != b_k {
                    return Err(GgmlError::InvalidShape(format!(
                        "MatMul dimension mismatch: k={} b_k={}",
                        k, b_k
                    )));
                }
                if output_desc.element_count() != (m as usize * n as usize) {
                    return Err(GgmlError::InvalidShape(
                        "Output shape does not match matmul result".to_string(),
                    ));
                }

                let a_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing A buffer".to_string()))?;
                let b_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing B buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::matmul::matmul(
                    &self.backend,
                    a_buf,
                    b_buf,
                    m,
                    n,
                    k,
                    out_buf,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Add => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Add expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let a_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing A buffer".to_string()))?;
                let b_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing B buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::add_scale::add(
                    &self.backend,
                    a_buf,
                    b_buf,
                    out_buf,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Scale { factor } => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Scale expects 1 input and 1 output".to_string(),
                    ));
                }

                let in_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::add_scale::scale(
                    &self.backend,
                    in_buf,
                    *factor,
                    out_buf,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::LayerNorm { eps } => {
                let inputs = _inputs;
                let outputs = _outputs;
                if outputs.len() != 1 || (inputs.len() != 2 && inputs.len() != 3) {
                    return Err(GgmlError::InvalidShape(
                        "LayerNorm expects 2 or 3 inputs and 1 output".to_string(),
                    ));
                }

                let input_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                let weight_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing weight desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                if weight_desc.shape.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "LayerNorm weight must be 1D".to_string(),
                    ));
                }
                if output_desc.shape != input_desc.shape {
                    return Err(GgmlError::InvalidShape(
                        "LayerNorm output shape mismatch".to_string(),
                    ));
                }

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let weight_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing weight buffer".to_string()))?;
                let output_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                let input_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&input_desc.shape);
                let weight_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&weight_desc.shape);
                let output_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&output_desc.shape);

                let input_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    input_buf.clone(),
                    input_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let weight_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    weight_buf.clone(),
                    weight_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let mut output_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    output_buf.clone(),
                    output_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;

                let bias_tensor = if inputs.len() == 3 {
                    let bias_desc = self
                        .tensor_desc(inputs[2])
                        .ok_or_else(|| GgmlError::InvalidShape("Missing bias desc".to_string()))?;
                    let bias_buf = self
                        .buffer(inputs[2])
                        .ok_or_else(|| GgmlError::Backend("Missing bias buffer".to_string()))?;
                    let bias_shape =
                        crate::loader::mmap_loader::TensorShape::from_dims(&bias_desc.shape);
                    Some(
                        crate::backend::DeviceTensor::from_buffer(
                            &self.backend,
                            bias_buf.clone(),
                            bias_shape,
                        )
                        .map_err(|e| GgmlError::Backend(e.to_string()))?,
                    )
                } else {
                    None
                };

                self.backend
                    .layernorm(
                        &input_tensor,
                        &weight_tensor,
                        bias_tensor.as_ref(),
                        &mut output_tensor,
                        *eps,
                    )
                    .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::RmsNorm { eps } => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "RmsNorm expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let input_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                let weight_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing weight desc".to_string()))?;

                if input_desc.shape.len() != 2 || weight_desc.shape.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "RmsNorm expects [seq_len, hidden] input and [hidden] weight".to_string(),
                    ));
                }

                let seq_len = input_desc.shape[0] as u32;
                let hidden = input_desc.shape[1] as u32;
                if weight_desc.shape[0] as u32 != hidden {
                    return Err(GgmlError::InvalidShape(
                        "RmsNorm weight hidden size mismatch".to_string(),
                    ));
                }

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let weight_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing weight buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::rms_norm::rms_norm(
                    &self.backend,
                    input_buf,
                    weight_buf,
                    out_buf,
                    seq_len,
                    hidden,
                    *eps,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Rope => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 3 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "RoPE expects 3 inputs and 1 output".to_string(),
                    ));
                }

                let input_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                if input_desc.shape.len() != 3 {
                    return Err(GgmlError::InvalidShape(
                        "RoPE expects input shape [seq_len, heads, head_dim]".to_string(),
                    ));
                }

                let seq_len = input_desc.shape[0] as u32;
                let num_heads = input_desc.shape[1] as u32;
                let head_dim = input_desc.shape[2] as u32;

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let cos_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing cos buffer".to_string()))?;
                let sin_buf = self
                    .buffer(inputs[2])
                    .ok_or_else(|| GgmlError::Backend("Missing sin buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::rope::rope(
                    &self.backend,
                    input_buf,
                    cos_buf,
                    sin_buf,
                    out_buf,
                    seq_len,
                    num_heads,
                    head_dim,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Softmax => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Softmax expects 1 input and 1 output".to_string(),
                    ));
                }

                let input_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                let shape = &input_desc.shape;
                let (batch_size, seq_len) = match shape.len() {
                    4 => {
                        if shape[2] != shape[3] {
                            return Err(GgmlError::InvalidShape(
                                "Softmax expects square last two dims".to_string(),
                            ));
                        }
                        ((shape[0] * shape[1]) as u32, shape[2] as u32)
                    }
                    3 => {
                        if shape[1] != shape[2] {
                            return Err(GgmlError::InvalidShape(
                                "Softmax expects square last two dims".to_string(),
                            ));
                        }
                        (shape[0] as u32, shape[1] as u32)
                    }
                    2 => {
                        if shape[0] != shape[1] {
                            return Err(GgmlError::InvalidShape(
                                "Softmax expects square matrix".to_string(),
                            ));
                        }
                        (1, shape[0] as u32)
                    }
                    _ => {
                        return Err(GgmlError::InvalidShape(
                            "Softmax expects 2D, 3D, or 4D tensor".to_string(),
                        ));
                    }
                };

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::softmax::softmax(
                    &self.backend,
                    input_buf,
                    out_buf,
                    batch_size,
                    seq_len,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Attention => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 5 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Attention expects 5 inputs and 1 output".to_string(),
                    ));
                }

                let q_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing Q desc".to_string()))?;
                let k_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing K desc".to_string()))?;
                let v_desc = self
                    .tensor_desc(inputs[2])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing V desc".to_string()))?;
                let scores_desc = self
                    .tensor_desc(inputs[3])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing scores desc".to_string()))?;
                let softmax_desc = self
                    .tensor_desc(inputs[4])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing softmax desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                if q_desc.shape.len() != 3 || k_desc.shape.len() != 3 || v_desc.shape.len() != 3
                {
                    return Err(GgmlError::InvalidShape(
                        "Q, K, V must be 3D [seq_len, num_heads, head_dim]".to_string(),
                    ));
                }

                let seq_q = q_desc.shape[0];
                let seq_k = k_desc.shape[0];
                if scores_desc.shape != vec![seq_q, seq_k]
                    || softmax_desc.shape != vec![seq_q, seq_k]
                {
                    return Err(GgmlError::InvalidShape(
                        "Attention scores/softmax shape mismatch".to_string(),
                    ));
                }
                if output_desc.shape != q_desc.shape {
                    return Err(GgmlError::InvalidShape(
                        "Attention output shape mismatch".to_string(),
                    ));
                }

                let q_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing Q buffer".to_string()))?;
                let k_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing K buffer".to_string()))?;
                let v_buf = self
                    .buffer(inputs[2])
                    .ok_or_else(|| GgmlError::Backend("Missing V buffer".to_string()))?;
                let scores_buf = self
                    .buffer(inputs[3])
                    .ok_or_else(|| GgmlError::Backend("Missing scores buffer".to_string()))?;
                let softmax_buf = self
                    .buffer(inputs[4])
                    .ok_or_else(|| GgmlError::Backend("Missing softmax buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                let q_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&q_desc.shape);
                let k_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&k_desc.shape);
                let v_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&v_desc.shape);
                let scores_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&scores_desc.shape);
                let softmax_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&softmax_desc.shape);
                let out_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&output_desc.shape);

                let q_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    q_buf.clone(),
                    q_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let k_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    k_buf.clone(),
                    k_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let v_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    v_buf.clone(),
                    v_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let mut scores_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    scores_buf.clone(),
                    scores_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let softmax_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    softmax_buf.clone(),
                    softmax_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let mut out_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    out_buf.clone(),
                    out_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;

                let kernels = crate::ops::attention_gpu::HipAttentionKernels::new(&self.backend)
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;

                kernels
                    .compute_qk_t(&q_tensor, &k_tensor, &mut scores_tensor)
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let head_dim = q_desc.shape[2];
                let scale = 1.0 / (head_dim as f32).sqrt();
                self.backend
                    .scale_inplace(&mut scores_tensor, scale)
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;
                kernels
                    .apply_causal_mask(&mut scores_tensor, seq_q, seq_k)
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;
                kernels
                    .compute_softmax(&mut scores_tensor, &softmax_tensor)
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;
                kernels
                    .compute_attention_weighted_v(&scores_tensor, &v_tensor, &mut out_tensor)
                    .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Mask => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Mask expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let scores_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing scores desc".to_string()))?;
                let mask_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing mask desc".to_string()))?;

                let (batch_size, seq_len) = match scores_desc.shape.len() {
                    4 => {
                        if scores_desc.shape[2] != scores_desc.shape[3] {
                            return Err(GgmlError::InvalidShape(
                                "Mask expects square last two dims".to_string(),
                            ));
                        }
                        (
                            (scores_desc.shape[0] * scores_desc.shape[1]) as u32,
                            scores_desc.shape[2] as u32,
                        )
                    }
                    3 => {
                        if scores_desc.shape[1] != scores_desc.shape[2] {
                            return Err(GgmlError::InvalidShape(
                                "Mask expects square last two dims".to_string(),
                            ));
                        }
                        (scores_desc.shape[0] as u32, scores_desc.shape[1] as u32)
                    }
                    _ => {
                        return Err(GgmlError::InvalidShape(
                            "Mask expects 3D or 4D scores tensor".to_string(),
                        ));
                    }
                };

                let mask_elements = mask_desc.element_count();
                let expected_mask = (seq_len as usize) * (seq_len as usize);
                if mask_elements != expected_mask {
                    return Err(GgmlError::InvalidShape(format!(
                        "Mask expects {} elements, got {}",
                        expected_mask, mask_elements
                    )));
                }

                let scores_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing scores buffer".to_string()))?;
                let mask_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing mask buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::mask::mask(
                    &self.backend,
                    scores_buf,
                    mask_buf,
                    out_buf,
                    batch_size,
                    seq_len,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::SwiGlu => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "SwiGlu expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let gate_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing gate desc".to_string()))?;
                let up_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing up desc".to_string()))?;

                if gate_desc.shape != up_desc.shape || gate_desc.shape.len() != 2 {
                    return Err(GgmlError::InvalidShape(
                        "SwiGlu expects matching 2D gate/up tensors".to_string(),
                    ));
                }

                let seq_len = gate_desc.shape[0] as u32;
                let intermediate_size = gate_desc.shape[1] as u32;

                let gate_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing gate buffer".to_string()))?;
                let up_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing up buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                crate::ggml::hip_backend::ops::swiglu::swiglu(
                    &self.backend,
                    gate_buf,
                    up_buf,
                    out_buf,
                    seq_len,
                    intermediate_size,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::MlpSwiglu => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 4 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "MlpSwiglu expects 4 inputs and 1 output".to_string(),
                    ));
                }

                let input_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                if output_desc.shape != input_desc.shape {
                    return Err(GgmlError::InvalidShape(
                        "MlpSwiglu output shape mismatch".to_string(),
                    ));
                }

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let gate_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing gate buffer".to_string()))?;
                let up_buf = self
                    .buffer(inputs[2])
                    .ok_or_else(|| GgmlError::Backend("Missing up buffer".to_string()))?;
                let down_buf = self
                    .buffer(inputs[3])
                    .ok_or_else(|| GgmlError::Backend("Missing down buffer".to_string()))?;
                let output_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                let input_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&input_desc.shape);
                let output_shape =
                    crate::loader::mmap_loader::TensorShape::from_dims(&output_desc.shape);

                let input_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    input_buf.clone(),
                    input_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let gate_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    gate_buf.clone(),
                    crate::loader::mmap_loader::TensorShape::from_dims(
                        &self.tensor_desc(inputs[1])
                            .ok_or_else(|| {
                                GgmlError::InvalidShape("Missing gate desc".to_string())
                            })?
                            .shape,
                    ),
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let up_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    up_buf.clone(),
                    crate::loader::mmap_loader::TensorShape::from_dims(
                        &self.tensor_desc(inputs[2])
                            .ok_or_else(|| {
                                GgmlError::InvalidShape("Missing up desc".to_string())
                            })?
                            .shape,
                    ),
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let down_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    down_buf.clone(),
                    crate::loader::mmap_loader::TensorShape::from_dims(
                        &self.tensor_desc(inputs[3])
                            .ok_or_else(|| {
                                GgmlError::InvalidShape("Missing down desc".to_string())
                            })?
                            .shape,
                    ),
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;
                let mut output_tensor = crate::backend::DeviceTensor::from_buffer(
                    &self.backend,
                    output_buf.clone(),
                    output_shape,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))?;

                self.backend
                    .mlp_swiglu(
                        &input_tensor,
                        &gate_tensor,
                        &up_tensor,
                        &down_tensor,
                        &mut output_tensor,
                    )
                    .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::SplitQkv => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 1 || outputs.len() != 3 {
                    return Err(GgmlError::InvalidShape(
                        "SplitQkv expects 1 input and 3 outputs".to_string(),
                    ));
                }

                let input_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                if input_desc.shape.len() != 2 {
                    return Err(GgmlError::InvalidShape(
                        "SplitQkv expects 2D input tensor".to_string(),
                    ));
                }
                let seq_len = input_desc.shape[0];
                let total_hidden = input_desc.shape[1];
                if total_hidden % 3 != 0 {
                    return Err(GgmlError::InvalidShape(
                        "SplitQkv expects last dim to be 3 * hidden".to_string(),
                    ));
                }
                let hidden = total_hidden / 3;

                let output_descs: Vec<_> = outputs
                    .iter()
                    .map(|id| {
                        self.tensor_desc(*id).ok_or_else(|| {
                            GgmlError::InvalidShape("Missing output desc".to_string())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                for desc in &output_descs {
                    if desc.shape.len() != 2
                        || desc.shape[0] != seq_len
                        || desc.shape[1] != hidden
                    {
                        return Err(GgmlError::InvalidShape(
                            "SplitQkv output shape mismatch".to_string(),
                        ));
                    }
                }

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let out_q = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing Q buffer".to_string()))?;
                let out_k = self
                    .buffer(outputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing K buffer".to_string()))?;
                let out_v = self
                    .buffer(outputs[2])
                    .ok_or_else(|| GgmlError::Backend("Missing V buffer".to_string()))?;

                crate::ggml::hip_backend::ops::split_qkv::split_qkv(
                    input_buf,
                    out_q,
                    out_k,
                    out_v,
                    seq_len,
                    hidden,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::View | Op::Reshape => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "View/Reshape expects 1 input and 1 output".to_string(),
                    ));
                }

                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;
                let source_id = output_desc.view_of.unwrap_or(inputs[0]);
                if source_id != inputs[0] {
                    return Err(GgmlError::InvalidShape(
                        "View/Reshape source tensor mismatch".to_string(),
                    ));
                }

                let input_buf = self
                    .buffer(source_id)
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let view_buf = input_buf
                    .sub_buffer_view(output_desc.byte_offset, output_desc.byte_size())
                    .map_err(|e| GgmlError::Backend(e.to_string()))?;

                let output_desc = output_desc.clone();
                self.bind(&output_desc, view_buf)?;
                Ok(())
            }
            Op::Copy => {
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Copy expects 1 input and 1 output".to_string(),
                    ));
                }

                let input_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;
                out_buf
                    .copy_from_buffer(input_buf)
                    .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::MatMulQ4_0 => {
                // PHASE 3: Quantized matmul for Q4_0 format
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "MatMulQ4_0 expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let weights_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing weights desc".to_string()))?;
                let input_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                // Validate shapes: weights is 2D, input is 2D
                if weights_desc.shape.len() != 2 || input_desc.shape.len() != 2 {
                    return Err(GgmlError::InvalidShape(
                        "MatMulQ4_0 expects 2D tensors".to_string(),
                    ));
                }

                // Q4_0 format: 20 bytes per 32 elements (4 bytes scale + 16 bytes data)
                let n_rows = weights_desc.shape[0];
                let n_cols = weights_desc.shape[1];
                let expected_q4_bytes = ((n_rows * n_cols + 31) / 32) * 20;
                if weights_desc.byte_size() != expected_q4_bytes {
                    return Err(GgmlError::InvalidShape(format!(
                        "MatMulQ4_0 weights size mismatch: expected {} bytes, got {}",
                        expected_q4_bytes, weights_desc.byte_size()
                    )));
                }

                // Validate matmul dimensions
                let m = input_desc.shape[0] as i32;
                let k = input_desc.shape[1] as i32;
                let weights_k = n_cols as i32;
                let n = n_rows as i32;
                if k != weights_k {
                    return Err(GgmlError::InvalidShape(format!(
                        "MatMulQ4_0 dimension mismatch: k={} weights_k={}",
                        k, weights_k
                    )));
                }

                let weights_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing weights buffer".to_string()))?;
                let input_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                // Read quantized weights from GPU
                let mut quantized_data = vec![0u8; weights_desc.byte_size()];
                weights_buf
                    .copy_to_host(&mut quantized_data)
                    .map_err(|e| GgmlError::Backend(format!("Failed to read weights: {}", e)))?;

                // Perform dequantize + matmul
                crate::ggml::hip_backend::ops::quantized_matmul::matmul_q4_0(
                    &self.backend,
                    &quantized_data,
                    input_buf,
                    n_rows,
                    n_cols,
                    out_buf,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::MatMulQ8_0 => {
                // PHASE 3: Quantized matmul for Q8_0 format
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "MatMulQ8_0 expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let weights_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing weights desc".to_string()))?;
                let input_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing input desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                // Validate shapes: weights is 2D, input is 2D
                if weights_desc.shape.len() != 2 || input_desc.shape.len() != 2 {
                    return Err(GgmlError::InvalidShape(
                        "MatMulQ8_0 expects 2D tensors".to_string(),
                    ));
                }

                // Q8_0 format: 36 bytes per 32 elements (4 bytes scale + 32 bytes data)
                let n_rows = weights_desc.shape[0];
                let n_cols = weights_desc.shape[1];
                let expected_q8_bytes = ((n_rows * n_cols + 31) / 32) * 36;
                if weights_desc.byte_size() != expected_q8_bytes {
                    return Err(GgmlError::InvalidShape(format!(
                        "MatMulQ8_0 weights size mismatch: expected {} bytes, got {}",
                        expected_q8_bytes, weights_desc.byte_size()
                    )));
                }

                // Validate matmul dimensions
                let m = input_desc.shape[0] as i32;
                let k = input_desc.shape[1] as i32;
                let weights_k = n_cols as i32;
                let n = n_rows as i32;
                if k != weights_k {
                    return Err(GgmlError::InvalidShape(format!(
                        "MatMulQ8_0 dimension mismatch: k={} weights_k={}",
                        k, weights_k
                    )));
                }

                let weights_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing weights buffer".to_string()))?;
                let input_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing input buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                // Read quantized weights from GPU
                let mut quantized_data = vec![0u8; weights_desc.byte_size()];
                weights_buf
                    .copy_to_host(&mut quantized_data)
                    .map_err(|e| GgmlError::Backend(format!("Failed to read weights: {}", e)))?;

                // Perform dequantize + matmul
                crate::ggml::hip_backend::ops::quantized_matmul::matmul_q8_0(
                    &self.backend,
                    &quantized_data,
                    input_buf,
                    n_rows,
                    n_cols,
                    out_buf,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            Op::Accumulate { offset } => {
                // PHASE 5: Accumulate op for efficient KV cache writes
                let inputs = _inputs;
                let outputs = _outputs;
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(GgmlError::InvalidShape(
                        "Accumulate expects 2 inputs and 1 output".to_string(),
                    ));
                }

                let src_desc = self
                    .tensor_desc(inputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing src desc".to_string()))?;
                let dst_desc = self
                    .tensor_desc(inputs[1])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing dst desc".to_string()))?;
                let output_desc = self
                    .tensor_desc(outputs[0])
                    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;

                // Validate shapes: src and output should match
                if src_desc.shape != output_desc.shape {
                    return Err(GgmlError::InvalidShape(
                        "Accumulate src shape must match output shape".to_string(),
                    ));
                }

                // Validate dst size: dst must be large enough to hold src at offset
                let src_elements = src_desc.element_count();
                let dst_elements = dst_desc.element_count();
                let byte_offset = offset * std::mem::size_of::<f32>();
                if byte_offset + src_desc.byte_size() > dst_desc.byte_size() {
                    return Err(GgmlError::InvalidShape(format!(
                        "Accumulate offset {} exceeds dst size (src={}, dst={})",
                        offset, src_desc.byte_size(), dst_desc.byte_size()
                    )));
                }

                let src_buf = self
                    .buffer(inputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing src buffer".to_string()))?;
                let dst_buf = self
                    .buffer(inputs[1])
                    .ok_or_else(|| GgmlError::Backend("Missing dst buffer".to_string()))?;
                let out_buf = self
                    .buffer(outputs[0])
                    .ok_or_else(|| GgmlError::Backend("Missing output buffer".to_string()))?;

                // Perform accumulate: dst[offset:offset+src_size] += src
                crate::ggml::hip_backend::ops::accumulate::accumulate(
                    &self.backend,
                    src_buf,
                    dst_buf,
                    out_buf,
                    src_elements,
                    byte_offset,
                )
                .map_err(|e| GgmlError::Backend(e.to_string()))
            }
            _ => Err(GgmlError::Unimplemented(format!(
                "HIP backend op not implemented: {:?}",
                op
            ))),
        }
    }

    fn synchronize(&mut self) -> GgmlResult<()> {
        // PHASE 01 FIX: Actually synchronize instead of being a no-op
        //
        // Previously this was a no-op that just returned Ok(()).
        // This caused hangs because GPU operations queued on the backend's
        // stream weren't completing before the caller tried to read results.
        self.backend
            .synchronize()
            .map_err(|e| GgmlError::Backend(e.to_string()))
    }
}
