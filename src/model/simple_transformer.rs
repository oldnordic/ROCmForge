//! Simple transformer model implementation
//! Minimal transformer blocks for end-to-end testing

use crate::tensor::matmul::cpu_matmul_f32;
use crate::tensor::Tensor;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

/// Model backend for CPU vs GPU computation
#[derive(Debug, Clone, Copy)]
pub enum ModelBackend {
    Cpu,
    #[cfg(feature = "rocm")]
    Gpu,
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Dimension error: {0}")]
    DimensionError(String),
    #[error("Attention error: {0}")]
    AttentionError(#[from] crate::attention::AttentionError),
    #[cfg(feature = "rocm")]
    #[error("GPU error: {0}")]
    GpuError(#[from] crate::tensor::matmul::MatmulError),
}

pub type ModelResult<T> = Result<T, ModelError>;

/// Simple linear layer: y = x * W^T + b
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
    in_features: usize,
    out_features: usize,
    #[cfg(feature = "rocm")]
    gpu_buffer: Option<crate::backend::HipBuffer>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Initialize weights with Xavier initialization
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let mut weight_data = Vec::with_capacity(in_features * out_features);
        for _ in 0..in_features * out_features {
            weight_data.push(rng.gen::<f32>() * scale);
        }

        // Initialize bias to zeros
        let bias_data = vec![0.0f32; out_features];

        let mut linear = Self {
            weight: Tensor { data: weight_data },
            bias: Tensor { data: bias_data },
            in_features,
            out_features,
            #[cfg(feature = "rocm")]
            gpu_buffer: None,
        };

        #[cfg(feature = "rocm")]
        {
            linear = linear.with_gpu_buffer();
        }

        #[cfg(not(feature = "rocm"))]
        {
            // Nothing to do
        }

        linear
    }

    #[cfg(feature = "rocm")]
    fn with_gpu_buffer(mut self) -> Self {
        use crate::backend::HipBackend;

        // Initialize HIP backend and get memory info
        let backend = match HipBackend::new() {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!("Failed to initialize HIP backend: {}", e);
                return self; // Return self without GPU buffer
            }
        };

        // Allocate GPU buffer for weights with memory limit checking
        let weight_buffer = match backend.allocate_buffer(self.weight.data.len()) {
            Ok(buffer) => {
                if let Err(e) = buffer.copy_from_host(&self.weight.data) {
                    tracing::warn!("Failed to copy weights to GPU: {}", e);
                    None
                } else {
                    Some(buffer)
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to allocate GPU buffer for linear layer weights: {}",
                    e
                );
                None
            }
        };

        self.gpu_buffer = weight_buffer;
        self
    }

    pub fn forward(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(ModelError::ShapeMismatch(format!(
                "Expected input length {}, got {}",
                self.in_features,
                input.len()
            )));
        }

        #[cfg(feature = "rocm")]
        if let Some(ref weight_buffer) = self.gpu_buffer {
            self.forward_gpu(input, weight_buffer)
        } else {
            self.forward_cpu(input)
        }

        #[cfg(not(feature = "rocm"))]
        {
            return self.forward_cpu(input);
        }
    }

    fn forward_cpu(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        // Compute output = input * weight^T + bias
        // input is (1, in_features), weight is (in_features, out_features) stored row-major
        // For weight^T, we need (out_features, in_features)
        // So we compute: input (1×k) * weight^T (k×n) = output (1×n)
        let mut output = cpu_matmul_f32(
            input,
            &self.weight.data,
            1,                 // m (rows of input)
            self.out_features, // n (cols of weight^T)
            self.in_features,  // k (cols of input/rows of weight)
        );

        // Add bias
        for (i, &b) in self.bias.data.iter().enumerate() {
            output[i] += b;
        }

        Ok(output)
    }

    #[cfg(feature = "rocm")]
    fn forward_gpu(
        &self,
        input: &[f32],
        weight_buffer: &crate::backend::HipBuffer,
    ) -> ModelResult<Vec<f32>> {
        use crate::backend::{hip_blas, HipBuffer};
        use crate::tensor::matmul::matmul_f32;

        // Debug logging for GPU usage (gated by debug assertions)
        #[cfg(debug_assertions)]
        tracing::debug!("Using GPU-accelerated linear layer forward");

        // Host → Device: Copy input to GPU
        let input_buffer = HipBuffer::new(input.len())
            .map_err(|e| ModelError::GpuError(crate::tensor::matmul::MatmulError::HipError(e)))?;
        input_buffer
            .copy_from_host(input)
            .map_err(|e| ModelError::GpuError(crate::tensor::matmul::MatmulError::HipError(e)))?;

        // Device: Create output buffer on GPU
        let mut output_data = vec![0.0f32; self.out_features];
        let output_buffer = HipBuffer::new(output_data.len())
            .map_err(|e| ModelError::GpuError(crate::tensor::matmul::MatmulError::HipError(e)))?;

        // Device: Perform GPU matrix multiplication
        let backend = crate::backend::HipBackend::new()
            .map_err(|e| ModelError::GpuError(crate::tensor::matmul::MatmulError::HipError(e)))?;
        let handle = hip_blas::HipBlasHandle::new().map_err(|e| ModelError::GpuError(e.into()))?;

        // Convert dimensions to i32 for BLAS API
        // Model dimensions should fit in i32 (>4B dimensions not supported)
        let n: i32 = self.out_features.try_into().map_err(|_| {
            ModelError::DimensionError(format!(
                "out_features {} exceeds i32::MAX, unsupported for BLAS",
                self.out_features
            ))
        })?;
        let k: i32 = self.in_features.try_into().map_err(|_| {
            ModelError::DimensionError(format!(
                "in_features {} exceeds i32::MAX, unsupported for BLAS",
                self.in_features
            ))
        })?;

        let _result = matmul_f32(
            &backend,
            &handle,
            &input_buffer,
            weight_buffer,
            1, // m (rows of input)
            n, // n (cols of weight^T)
            k, // k (cols of input/rows of weight)
        );

        // Device → Host: Copy result back to CPU
        output_buffer
            .copy_to_host(&mut output_data)
            .map_err(|e| ModelError::GpuError(e.into()))?;

        // Host: Add bias (still on CPU for now)
        for (i, &b) in self.bias.data.iter().enumerate() {
            output_data[i] += b;
        }

        Ok(output_data)
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

/// Simple attention wrapper
pub struct SimpleAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    dim: usize,
    backend: ModelBackend,
}

impl SimpleAttention {
    pub fn new(dim: usize, backend: ModelBackend, seed: u64) -> Self {
        Self {
            q_proj: Linear::new(dim, dim, seed),
            k_proj: Linear::new(dim, dim, seed + 1),
            v_proj: Linear::new(dim, dim, seed + 2),
            out_proj: Linear::new(dim, dim, seed + 3),
            dim,
            backend,
        }
    }

    pub fn forward(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        // Input shape: (seq_len, dim)
        let seq_len = input.len() / self.dim;

        if input.len() != seq_len * self.dim {
            return Err(ModelError::ShapeMismatch(format!(
                "Input length {} is not divisible by dim {}",
                input.len(),
                self.dim
            )));
        }

        // Route based on backend
        match self.backend {
            ModelBackend::Cpu => self.forward_cpu(input),
            #[cfg(feature = "rocm")]
            ModelBackend::Gpu => self.forward_gpu(input),
        }
    }

    fn forward_cpu(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        // For now, implement a simple attention mechanism that works
        // This is a simplified version that avoids complex batching issues

        // Project to Q, K, V
        let mut q = Vec::with_capacity(input.len());
        let mut k = Vec::with_capacity(input.len());
        let mut v = Vec::with_capacity(input.len());

        let seq_len = input.len() / self.dim;
        for i in 0..seq_len {
            let start = i * self.dim;
            let token_input = &input[start..start + self.dim];

            q.extend(self.q_proj.forward(token_input)?);
            k.extend(self.k_proj.forward(token_input)?);
            v.extend(self.v_proj.forward(token_input)?);
        }

        // Simple attention: compute QK^T * V manually for single batch
        // Reshape Q, K, V as (seq_len, dim) matrices
        let mut output = Vec::with_capacity(input.len());

        for i in 0..seq_len {
            // Compute attention weights for position i
            let mut attention_weights = Vec::with_capacity(seq_len);
            for j in 0..seq_len {
                // Dot product of Q[i] and K[j]
                let q_start = i * self.dim;
                let k_start = j * self.dim;
                let mut dot_product = 0.0f32;
                for d in 0..self.dim {
                    dot_product += q[q_start + d] * k[k_start + d];
                }
                attention_weights.push(dot_product);
            }

            // Apply softmax
            let max_weight = attention_weights
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_weights = Vec::with_capacity(seq_len);
            let mut exp_sum = 0.0f32;
            for &w in &attention_weights {
                let exp_w = (w - max_weight).exp();
                exp_weights.push(exp_w);
                exp_sum += exp_w;
            }

            // Compute weighted sum of V
            for d in 0..self.dim {
                let mut weighted_sum = 0.0f32;
                for j in 0..seq_len {
                    let v_start = j * self.dim;
                    weighted_sum += (exp_weights[j] / exp_sum) * v[v_start + d];
                }
                output.push(weighted_sum);
            }
        }

        // Project output through output projection
        let mut final_output = Vec::with_capacity(input.len());

        for i in 0..seq_len {
            let start = i * self.dim;
            let token_output = &output[start..start + self.dim];

            final_output.extend(self.out_proj.forward(token_output)?);
        }

        Ok(final_output)
    }

    #[cfg(feature = "rocm")]
    fn forward_gpu(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        // Debug logging for GPU usage (gated by debug assertions)
        #[cfg(debug_assertions)]
        tracing::debug!("Using GPU-accelerated attention (fallback to CPU for now)");

        // For now, fall back to CPU implementation for attention
        // In a full implementation, this would use GPU attention kernels
        self.forward_cpu(input)
    }
}

/// Simple feed-forward network
pub struct SimpleFeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl SimpleFeedForward {
    pub fn new(dim: usize, hidden_dim: usize, seed: u64) -> Self {
        Self {
            linear1: Linear::new(dim, hidden_dim, seed),
            linear2: Linear::new(hidden_dim, dim, seed + 1),
        }
    }

    pub fn forward(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        // First linear + ReLU
        let mut hidden = self.linear1.forward(input)?;
        for x in &mut hidden {
            *x = x.max(0.0); // ReLU
        }

        // Second linear
        self.linear2.forward(&hidden)
    }
}

/// Simple transformer block
pub struct SimpleTransformerBlock {
    attention: SimpleAttention,
    feed_forward: SimpleFeedForward,
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    norm2_weight: Tensor,
    norm2_bias: Tensor,
    dim: usize,
}

impl SimpleTransformerBlock {
    pub fn new(dim: usize, backend: ModelBackend, seed: u64) -> Self {
        Self {
            attention: SimpleAttention::new(dim, backend, seed),
            feed_forward: SimpleFeedForward::new(dim, dim * 4, seed + 4),
            norm1_weight: Tensor {
                data: vec![1.0f32; dim],
            },
            norm1_bias: Tensor {
                data: vec![0.0f32; dim],
            },
            norm2_weight: Tensor {
                data: vec![1.0f32; dim],
            },
            norm2_bias: Tensor {
                data: vec![0.0f32; dim],
            },
            dim,
        }
    }

    fn layer_norm(&self, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());

        for chunk in input.chunks_exact(self.dim) {
            // Compute mean and variance
            let mean: f32 = chunk.iter().sum::<f32>() / self.dim as f32;
            let variance: f32 =
                chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.dim as f32;
            let std = (variance + 1e-6).sqrt();

            // Normalize and scale
            for (i, &x) in chunk.iter().enumerate() {
                let normalized = (x - mean) / std;
                output.push(normalized * weight[i] + bias[i]);
            }
        }

        output
    }

    pub fn forward(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        let seq_len = input.len() / self.dim;

        // Pre-norm + attention + residual
        let normed_input = self.layer_norm(input, &self.norm1_weight.data, &self.norm1_bias.data);
        let attention_out = self.attention.forward(&normed_input)?;

        let mut residual = Vec::with_capacity(input.len());
        for (i, &x) in input.iter().enumerate() {
            residual.push(x + attention_out[i]);
        }

        // Pre-norm + feed-forward + residual
        let normed_residual =
            self.layer_norm(&residual, &self.norm2_weight.data, &self.norm2_bias.data);

        // Apply feed-forward to each token independently
        let mut ff_out = Vec::with_capacity(normed_residual.len());
        for i in 0..seq_len {
            let start = i * self.dim;
            let token_normed = &normed_residual[start..start + self.dim];
            let token_ff_out = self.feed_forward.forward(token_normed)?;
            ff_out.extend(token_ff_out);
        }

        let mut output = Vec::with_capacity(input.len());
        for (i, &x) in residual.iter().enumerate() {
            output.push(x + ff_out[i]);
        }

        Ok(output)
    }
}

/// Simple transformer model
pub struct SimpleModel {
    embedding: Linear,
    blocks: Vec<SimpleTransformerBlock>,
    final_norm_weight: Tensor,
    final_norm_bias: Tensor,
    vocab_size: usize,
    dim: usize,
    max_seq_len: usize,
}

impl SimpleModel {
    pub fn new(
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        max_seq_len: usize,
        backend: ModelBackend,
        seed: u64,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(SimpleTransformerBlock::new(
                dim,
                backend,
                seed + (i as u64) * 10,
            ));
        }

        Self {
            embedding: Linear::new(vocab_size, dim, seed),
            blocks,
            final_norm_weight: Tensor {
                data: vec![1.0f32; dim],
            },
            final_norm_bias: Tensor {
                data: vec![0.0f32; dim],
            },
            vocab_size,
            dim,
            max_seq_len,
        }
    }

    fn layer_norm(&self, input: &[f32]) -> Vec<f32> {
        let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let variance: f32 =
            input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        let std = (variance + 1e-6).sqrt();

        input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                (x - mean) / std * self.final_norm_weight.data[i % self.dim]
                    + self.final_norm_bias.data[i % self.dim]
            })
            .collect()
    }

    pub fn forward(&self, input_tokens: &[u32]) -> ModelResult<Vec<f32>> {
        if input_tokens.len() > self.max_seq_len {
            return Err(ModelError::ShapeMismatch(format!(
                "Input length {} exceeds max sequence length {}",
                input_tokens.len(),
                self.max_seq_len
            )));
        }

        // Create embeddings using embedding lookup
        // Host → Device: Input tokens are on CPU, embedding lookup happens on CPU
        let mut embeddings = Vec::with_capacity(input_tokens.len() * self.dim);
        for &token_id in input_tokens {
            if token_id >= self.vocab_size as u32 {
                return Err(ModelError::ShapeMismatch(format!(
                    "Token ID {} exceeds vocab size {}",
                    token_id, self.vocab_size
                )));
            }

            // For embedding, we can directly index into the weight matrix
            // Each token_id selects a row from the embedding weight matrix
            let start = token_id as usize * self.dim;
            let end = start + self.dim;
            if end <= self.embedding.weight.data.len() {
                let token_embedding = &self.embedding.weight.data[start..end];
                embeddings.extend_from_slice(token_embedding);
            } else {
                return Err(ModelError::ShapeMismatch(format!(
                    "Token ID {} out of bounds for embedding matrix",
                    token_id
                )));
            }
        }

        // Pass through transformer blocks
        // Host → Device → Host: Each block may use GPU internally but returns CPU results
        let mut hidden = embeddings;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // Final layer norm
        // Final computation stays on CPU after all GPU operations complete
        let output = self.layer_norm(&hidden);

        Ok(output)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(3, 2, 42);
        let input = vec![1.0f32, 2.0f32, 3.0f32];
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_simple_attention_forward() {
        let attention = SimpleAttention::new(4, ModelBackend::Cpu, 42);
        let input = vec![1.0f32; 8]; // seq_len=2, dim=4
        let output = attention.forward(&input).unwrap();

        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_simple_model_forward() {
        let model = SimpleModel::new(100, 8, 1, 4, ModelBackend::Cpu, 42);
        let input_tokens = vec![1, 2, 3, 4];
        let output = model.forward(&input_tokens).unwrap();

        assert_eq!(output.len(), 4 * 8); // seq_len * dim
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
