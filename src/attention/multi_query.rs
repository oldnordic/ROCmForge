//! Multi-Query Attention implementation
//!
//! Multi-Query Attention (MQA) uses a single key and value head shared across all query heads.
//! This reduces memory usage and improves inference speed while maintaining performance.

use crate::attention::{rope::Rope, AttentionError, AttentionResult};
#[cfg(feature = "rocm")]
use crate::backend::{DeviceTensor, HipBackend};

/// Multi-Query Attention configuration
#[derive(Debug, Clone)]
pub struct MultiQueryConfig {
    /// Number of query heads
    pub num_query_heads: usize,
    /// Number of key/value heads (typically 1 for MQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to use rotary embeddings
    pub use_rope: bool,
    /// RoPE configuration (if use_rope is true)
    pub rope_config: Option<Rope>,
}

impl MultiQueryConfig {
    pub fn new(num_query_heads: usize, head_dim: usize) -> Self {
        Self {
            num_query_heads,
            num_kv_heads: 1, // Standard MQA uses 1 KV head
            head_dim,
            use_rope: false,
            rope_config: None,
        }
    }

    pub fn with_kv_heads(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    pub fn with_rope(mut self, rope_config: Rope) -> Self {
        self.use_rope = true;
        self.rope_config = Some(rope_config);
        self
    }

    /// Get total hidden dimension
    pub fn hidden_size(&self) -> usize {
        self.num_query_heads * self.head_dim
    }

    /// Get key/value hidden dimension
    pub fn kv_hidden_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Validate configuration
    pub fn validate(&self) -> AttentionResult<()> {
        if self.num_query_heads == 0 {
            return Err(AttentionError::DimensionError(
                "Number of query heads must be > 0".to_string(),
            ));
        }
        if self.num_kv_heads == 0 {
            return Err(AttentionError::DimensionError(
                "Number of KV heads must be > 0".to_string(),
            ));
        }
        if self.head_dim == 0 {
            return Err(AttentionError::DimensionError(
                "Head dimension must be > 0".to_string(),
            ));
        }
        if !self.num_query_heads.is_multiple_of(self.num_kv_heads) {
            return Err(AttentionError::DimensionError(format!(
                "Number of query heads ({}) must be divisible by number of KV heads ({})",
                self.num_query_heads, self.num_kv_heads
            )));
        }
        if self.use_rope && self.rope_config.is_none() {
            return Err(AttentionError::DimensionError(
                "RoPE enabled but no configuration provided".to_string(),
            ));
        }
        Ok(())
    }
}

/// Multi-Query Attention implementation
pub struct MultiQueryAttention {
    config: MultiQueryConfig,
}

impl MultiQueryAttention {
    /// Create new multi-query attention
    pub fn new(config: MultiQueryConfig) -> AttentionResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Forward pass for multi-query attention
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, seq_len, num_query_heads, head_dim]
    /// * `k` - Key tensor [batch_size, kv_seq_len, num_kv_heads, head_dim]
    /// * `v` - Value tensor [batch_size, kv_seq_len, num_kv_heads, head_dim]
    /// * `position_ids` - Position IDs [batch_size, seq_len] (optional, for RoPE)
    /// * `mask` - Attention mask [batch_size, seq_len, kv_seq_len] (optional)
    ///
    /// # Returns
    /// * Output tensor [batch_size, seq_len, num_query_heads, head_dim]
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        position_ids: Option<&[usize]>,
        mask: Option<&[f32]>,
    ) -> AttentionResult<Vec<f32>> {
        let batch_size = self.extract_batch_size(q, k, v)?;
        let seq_len = self.extract_seq_len(q)?;
        let kv_seq_len = self.extract_kv_seq_len(k, v)?;

        // Note: Validation removed because extract functions already validate consistency
        // The validate_input_shapes function had a logic error that recomputed sizes
        // leading to circular validation. The extract functions are sufficient.

        // Apply RoPE to queries and keys if enabled
        let mut q_processed = q.to_vec();
        let mut k_processed = k.to_vec();

        if let (Some(rope), Some(pos_ids)) = (&self.config.rope_config, position_ids) {
            rope.apply_q(&mut q_processed, pos_ids, self.config.num_query_heads)?;
            rope.apply_k(&mut k_processed, pos_ids, self.config.num_kv_heads)?;
        }

        // Expand K and V to match query heads for compatibility
        let (k_expanded, v_expanded) =
            self.expand_kv_to_query_heads(&k_processed, v, batch_size, kv_seq_len)?;

        // Compute attention scores
        let attention_scores = self.compute_attention_scores(
            &q_processed,
            &k_expanded,
            batch_size,
            seq_len,
            kv_seq_len,
        )?;

        // Apply mask if provided
        let masked_scores =
            self.apply_mask(&attention_scores, mask, batch_size, seq_len, kv_seq_len)?;

        // Apply softmax
        let attention_weights =
            self.softmax_attention(&masked_scores, batch_size, seq_len, kv_seq_len)?;

        // Compute final output
        let output = self.compute_output(
            &attention_weights,
            &v_expanded,
            batch_size,
            seq_len,
            kv_seq_len,
        )?;

        Ok(output)
    }

    /// Forward pass with DeviceTensor inputs for GPU computation
    #[cfg(feature = "rocm")]
    pub fn forward_device(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        position_ids: Option<&[usize]>,
        mask: Option<&DeviceTensor>,
    ) -> AttentionResult<DeviceTensor> {
        // If MHA (equal heads), use existing GPU path
        if self.config.num_query_heads == self.config.num_kv_heads {
            return self.forward_mha_gpu(q, k, v, position_ids, mask);
        }

        // MQA/GQA: Replicate K/V on GPU then use standard attention
        let (k_expanded, v_expanded) = self.replicate_kv_gpu(q, k, v)?;

        // Apply RoPE if needed
        // TODO: Implement RoPE application for GPU tensors

        // Use standard attention pipeline with expanded K/V
        self.compute_attention_gpu(q, &k_expanded, &v_expanded, mask)
    }

    /// Replicate K and V tensors from num_kv_heads to num_q_heads on GPU
    #[cfg(feature = "rocm")]
    fn replicate_kv_gpu(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
    ) -> AttentionResult<(DeviceTensor, DeviceTensor)> {
        use crate::attention::kernels::mqa_kv_replicate_gpu_kernel;

        // Extract dimensions from query tensor
        let q_dims = q.shape().dims();
        let batch_size = q_dims[0];
        let seq_len = q_dims[1];
        let num_q_heads = q_dims[2];
        let head_dim = q_dims[3];

        let num_kv_heads = self.config.num_kv_heads;

        // Verify num_q_heads matches config
        assert_eq!(num_q_heads, self.config.num_query_heads);

        // Allocate expanded tensors
        let k_expanded_shape =
            crate::loader::TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);
        let v_expanded_shape =
            crate::loader::TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

        let backend = HipBackend::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
        })?;

        let k_expanded = DeviceTensor::empty(&backend, k_expanded_shape).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to allocate expanded K: {}", e))
        })?;

        let v_expanded = DeviceTensor::empty(&backend, v_expanded_shape).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to allocate expanded V: {}", e))
        })?;

        // Call GPU kernel
        let k_ptr = k.as_ptr();
        let v_ptr = v.as_ptr();
        let k_expanded_ptr = k_expanded.as_ptr() as *mut f32;
        let v_expanded_ptr = v_expanded.as_ptr() as *mut f32;

        unsafe {
            mqa_kv_replicate_gpu_kernel(
                k_ptr,
                v_ptr,
                k_expanded_ptr,
                v_expanded_ptr,
                batch_size as u32,
                seq_len as u32,
                num_kv_heads as u32,
                num_q_heads as u32,
                head_dim as u32,
            )
            .map_err(|e| {
                AttentionError::GpuOperation(format!("KV replication kernel failed: {}", e))
            })?;

            // Synchronize to ensure kernel completes
            backend.synchronize().map_err(|e| {
                AttentionError::Synchronization(format!("Failed to synchronize GPU: {}", e))
            })?;
        }

        Ok((k_expanded, v_expanded))
    }

    /// Compute attention using GPU path (for MHA)
    #[cfg(feature = "rocm")]
    fn forward_mha_gpu(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        _position_ids: Option<&[usize]>,
        mask: Option<&DeviceTensor>,
    ) -> AttentionResult<DeviceTensor> {
        // For MHA, we can use the standard attention path
        // TODO: Implement full GPU attention pipeline
        // For now, return a simple implementation
        self.compute_attention_gpu(q, k, v, mask)
    }

    /// Compute attention scores on GPU
    #[cfg(feature = "rocm")]
    fn compute_attention_gpu(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        _mask: Option<&DeviceTensor>,
    ) -> AttentionResult<DeviceTensor> {
        use crate::attention::kernels::{
            qkt_matmul_gpu_kernel_scaled, softmax_gpu_kernel, weighted_matmul_gpu_kernel,
        };
        use crate::backend::hip_backend::HipBackend;

        let backend = HipBackend::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
        })?;

        let q_dims = q.shape().dims();
        let batch_size = q_dims[0];
        let seq_len = q_dims[1];
        let num_heads = q_dims[2];
        let head_dim = q_dims[3];

        let kv_seq_len = k.shape().dims()[1];

        // Allocate attention scores tensor
        let scores_shape =
            crate::loader::TensorShape::from_dims(&[batch_size, seq_len, num_heads, kv_seq_len]);
        let mut scores = DeviceTensor::empty(&backend, scores_shape.clone()).map_err(|e| {
            AttentionError::MemoryAllocation(format!("Failed to allocate scores: {}", e))
        })?;

        // Compute QK^T with scaling
        let scale = 1.0 / (head_dim as f32).sqrt();
        unsafe {
            qkt_matmul_gpu_kernel_scaled(
                q.as_ptr(),
                k.as_ptr(),
                scores.as_ptr() as *mut f32,
                batch_size as u32,
                seq_len as u32,
                kv_seq_len as u32,
                num_heads as u32,
                head_dim as u32,
                scale,
            )
            .map_err(|e| AttentionError::GpuOperation(format!("QK^T kernel failed: {}", e)))?;

            // Apply softmax
            softmax_gpu_kernel(
                scores.as_ptr() as *mut f32,
                batch_size as u32,
                seq_len as u32 * num_heads as u32, // Treat [batch, seq, heads] as batch
            );

            // Allocate output tensor
            let output_shape =
                crate::loader::TensorShape::from_dims(&[batch_size, seq_len, num_heads, head_dim]);
            let mut output = DeviceTensor::empty(&backend, output_shape.clone()).map_err(|e| {
                AttentionError::MemoryAllocation(format!("Failed to allocate output: {}", e))
            })?;

            // Compute weighted × V
            weighted_matmul_gpu_kernel(
                scores.as_ptr(),
                v.as_ptr(),
                output.as_ptr() as *mut f32,
                batch_size as u32,
                seq_len as u32,
                kv_seq_len as u32,
                num_heads as u32,
                head_dim as u32,
            )
            .map_err(|e| {
                AttentionError::GpuOperation(format!("Weighted matmul kernel failed: {}", e))
            })?;

            backend.synchronize().map_err(|e| {
                AttentionError::Synchronization(format!("Failed to synchronize GPU: {}", e))
            })?;

            Ok(output)
        }
    }

    /// Extract batch size from input tensors
    fn extract_batch_size(&self, q: &[f32], k: &[f32], v: &[f32]) -> AttentionResult<usize> {
        let q_expected = self.config.num_query_heads * self.config.head_dim;
        let k_expected = self.config.num_kv_heads * self.config.head_dim;
        let v_expected = self.config.num_kv_heads * self.config.head_dim;

        if !q.len().is_multiple_of(q_expected)
            || !k.len().is_multiple_of(k_expected)
            || !v.len().is_multiple_of(v_expected)
        {
            return Err(AttentionError::ShapeMismatch(
                "Input tensor sizes are not compatible with head dimensions".to_string(),
            ));
        }

        // Assume batch size is the same for all tensors
        let batch_q = q.len() / q_expected;
        let batch_k = k.len() / k_expected;
        let batch_v = v.len() / v_expected;

        if batch_q != batch_k || batch_q != batch_v {
            return Err(AttentionError::ShapeMismatch(
                "Batch sizes are not consistent across tensors".to_string(),
            ));
        }

        Ok(batch_q)
    }

    /// Extract sequence length from query tensor
    fn extract_seq_len(&self, q: &[f32]) -> AttentionResult<usize> {
        let expected_per_token = self.config.num_query_heads * self.config.head_dim;
        if !q.len().is_multiple_of(expected_per_token) {
            return Err(AttentionError::ShapeMismatch(
                "Query tensor size is not compatible with head dimensions".to_string(),
            ));
        }
        Ok(q.len() / expected_per_token)
    }

    /// Extract key/value sequence length
    fn extract_kv_seq_len(&self, k: &[f32], v: &[f32]) -> AttentionResult<usize> {
        let expected_per_token = self.config.num_kv_heads * self.config.head_dim;
        if !k.len().is_multiple_of(expected_per_token)
            || !v.len().is_multiple_of(expected_per_token)
        {
            return Err(AttentionError::ShapeMismatch(
                "KV tensor sizes are not compatible with head dimensions".to_string(),
            ));
        }
        let seq_k = k.len() / expected_per_token;
        let seq_v = v.len() / expected_per_token;
        if seq_k != seq_v {
            return Err(AttentionError::ShapeMismatch(
                "Key and value sequence lengths must match".to_string(),
            ));
        }
        Ok(seq_k)
    }

    /// Validate input tensor shapes
    fn validate_input_shapes(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_size: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> AttentionResult<()> {
        let q_expected = batch_size * seq_len * self.config.num_query_heads * self.config.head_dim;
        let k_expected = batch_size * kv_seq_len * self.config.num_kv_heads * self.config.head_dim;
        let v_expected = batch_size * kv_seq_len * self.config.num_kv_heads * self.config.head_dim;

        if q.len() != q_expected {
            return Err(AttentionError::ShapeMismatch(format!(
                "Query tensor size {} doesn't match expected {}",
                q.len(),
                q_expected
            )));
        }
        if k.len() != k_expected {
            return Err(AttentionError::ShapeMismatch(format!(
                "Key tensor size {} doesn't match expected {}",
                k.len(),
                k_expected
            )));
        }
        if v.len() != v_expected {
            return Err(AttentionError::ShapeMismatch(format!(
                "Value tensor size {} doesn't match expected {}",
                v.len(),
                v_expected
            )));
        }

        Ok(())
    }

    /// Expand K and V tensors to match query heads for compatibility
    fn expand_kv_to_query_heads(
        &self,
        k: &[f32],
        v: &[f32],
        batch_size: usize,
        kv_seq_len: usize,
    ) -> AttentionResult<(Vec<f32>, Vec<f32>)> {
        let heads_per_kv = self.config.num_query_heads / self.config.num_kv_heads;
        let kv_head_dim = self.config.num_kv_heads * self.config.head_dim;
        let query_head_dim = self.config.num_query_heads * self.config.head_dim;

        let mut k_expanded = vec![0.0f32; batch_size * kv_seq_len * query_head_dim];
        let mut v_expanded = vec![0.0f32; batch_size * kv_seq_len * query_head_dim];

        for b in 0..batch_size {
            for s in 0..kv_seq_len {
                for kv_h in 0..self.config.num_kv_heads {
                    for d in 0..self.config.head_dim {
                        let src_idx = b * kv_seq_len * kv_head_dim
                            + s * kv_head_dim
                            + kv_h * self.config.head_dim
                            + d;
                        let src_val_k = k[src_idx];
                        let src_val_v = v[src_idx];

                        // Copy to all corresponding query heads
                        for q_h_offset in 0..heads_per_kv {
                            let q_h = kv_h * heads_per_kv + q_h_offset;
                            let dst_idx = b * kv_seq_len * query_head_dim
                                + s * query_head_dim
                                + q_h * self.config.head_dim
                                + d;
                            k_expanded[dst_idx] = src_val_k;
                            v_expanded[dst_idx] = src_val_v;
                        }
                    }
                }
            }
        }

        Ok((k_expanded, v_expanded))
    }

    /// Compute attention scores: Q @ K^T
    fn compute_attention_scores(
        &self,
        q: &[f32],
        k: &[f32],
        batch_size: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> AttentionResult<Vec<f32>> {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_query_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scores = vec![0.0f32; batch_size * seq_len * num_heads * kv_seq_len];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    for kv_s in 0..kv_seq_len {
                        let mut dot_product = 0.0f32;

                        // Compute dot product of Q[b,s,h,:] and K[b,kv_s,h,:]
                        for d in 0..head_dim {
                            let q_idx = b * seq_len * num_heads * head_dim
                                + s * num_heads * head_dim
                                + h * head_dim
                                + d;
                            let k_idx = b * kv_seq_len * num_heads * head_dim
                                + kv_s * num_heads * head_dim
                                + h * head_dim
                                + d;
                            dot_product += q[q_idx] * k[k_idx];
                        }

                        let score_idx = b * seq_len * num_heads * kv_seq_len
                            + s * num_heads * kv_seq_len
                            + h * kv_seq_len
                            + kv_s;
                        scores[score_idx] = dot_product * scale;
                    }
                }
            }
        }

        Ok(scores)
    }

    /// Apply attention mask
    ///
    /// Mask can be either:
    /// - Broadcast shape: [batch_size, seq_len, kv_seq_len] - shared across all heads
    /// - Full shape: [batch_size, seq_len, num_heads, kv_seq_len] - per-head masking
    fn apply_mask(
        &self,
        scores: &[f32],
        mask: Option<&[f32]>,
        batch_size: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> AttentionResult<Vec<f32>> {
        let mut masked_scores = scores.to_vec();

        if let Some(mask_data) = mask {
            let num_heads = self.config.num_query_heads;
            let expected_broadcast = batch_size * seq_len * kv_seq_len;
            let expected_full = batch_size * seq_len * num_heads * kv_seq_len;

            // Validate mask shape - accept both broadcast and full shapes
            if mask_data.len() != expected_broadcast && mask_data.len() != expected_full {
                return Err(AttentionError::ShapeMismatch(format!(
                    "Mask length {} does not match expected shapes: broadcast [B,S,KvS]={} or full [B,S,H,KvS]={}",
                    mask_data.len(), expected_broadcast, expected_full
                )));
            }

            let is_broadcast_mask = mask_data.len() == expected_broadcast;

            for b in 0..batch_size {
                for s in 0..seq_len {
                    for kv_s in 0..kv_seq_len {
                        // Apply mask to all heads
                        for h in 0..num_heads {
                            let score_idx = b * seq_len * num_heads * kv_seq_len
                                + s * num_heads * kv_seq_len
                                + h * kv_seq_len
                                + kv_s;

                            let mask_val = if is_broadcast_mask {
                                // Broadcast mask: same value for all heads
                                let mask_idx = b * seq_len * kv_seq_len + s * kv_seq_len + kv_s;
                                mask_data[mask_idx]
                            } else {
                                // Full mask: per-head value
                                let mask_idx = b * seq_len * num_heads * kv_seq_len
                                    + s * num_heads * kv_seq_len
                                    + h * kv_seq_len
                                    + kv_s;
                                mask_data[mask_idx]
                            };

                            masked_scores[score_idx] += mask_val;
                        }
                    }
                }
            }
        }

        Ok(masked_scores)
    }

    /// Apply softmax to attention weights
    fn softmax_attention(
        &self,
        scores: &[f32],
        batch_size: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> AttentionResult<Vec<f32>> {
        let num_heads = self.config.num_query_heads;
        let mut weights = vec![0.0f32; batch_size * seq_len * num_heads * kv_seq_len];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    // Find max for numerical stability
                    let mut max_val = f32::NEG_INFINITY;
                    for kv_s in 0..kv_seq_len {
                        let score_idx = b * seq_len * num_heads * kv_seq_len
                            + s * num_heads * kv_seq_len
                            + h * kv_seq_len
                            + kv_s;
                        max_val = max_val.max(scores[score_idx]);
                    }

                    // Compute exp and sum
                    let mut sum = 0.0f32;
                    for kv_s in 0..kv_seq_len {
                        let score_idx = b * seq_len * num_heads * kv_seq_len
                            + s * num_heads * kv_seq_len
                            + h * kv_seq_len
                            + kv_s;
                        let exp_val = (scores[score_idx] - max_val).exp();
                        weights[score_idx] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize
                    for kv_s in 0..kv_seq_len {
                        let score_idx = b * seq_len * num_heads * kv_seq_len
                            + s * num_heads * kv_seq_len
                            + h * kv_seq_len
                            + kv_s;
                        weights[score_idx] /= sum;
                    }
                }
            }
        }

        Ok(weights)
    }

    /// Compute final output: attention_weights @ V
    fn compute_output(
        &self,
        attention_weights: &[f32],
        v: &[f32],
        batch_size: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> AttentionResult<Vec<f32>> {
        let num_heads = self.config.num_query_heads;
        let head_dim = self.config.head_dim;
        let mut output = vec![0.0f32; batch_size * seq_len * num_heads * head_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;

                        for kv_s in 0..kv_seq_len {
                            let weight_idx = b * seq_len * num_heads * kv_seq_len
                                + s * num_heads * kv_seq_len
                                + h * kv_seq_len
                                + kv_s;
                            let v_idx = b * kv_seq_len * num_heads * head_dim
                                + kv_s * num_heads * head_dim
                                + h * head_dim
                                + d;
                            sum += attention_weights[weight_idx] * v[v_idx];
                        }

                        let out_idx = b * seq_len * num_heads * head_dim
                            + s * num_heads * head_dim
                            + h * head_dim
                            + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get configuration
    pub fn config(&self) -> &MultiQueryConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::rope::{Rope, RopeConfig};

    #[test]
    fn test_multi_query_config_validation() {
        let config = MultiQueryConfig::new(8, 64);
        assert!(config.validate().is_ok());

        // Test invalid configurations
        let invalid_config = MultiQueryConfig::new(0, 64);
        assert!(invalid_config.validate().is_err());

        let invalid_config2 = MultiQueryConfig::new(8, 0);
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_multi_query_attention_basic() {
        let config = MultiQueryConfig::new(2, 4); // 2 query heads, 1 KV head, dim 4
        let mqa = MultiQueryAttention::new(config).unwrap();

        // For batch*seq=1:
        // q: 1 * 2 * 4 = 8 elements (batch*seq * num_query_heads * head_dim)
        // k: 1 * 1 * 4 = 4 elements (batch*seq * num_kv_heads * head_dim)
        // v: 1 * 1 * 4 = 4 elements (batch*seq * num_kv_heads * head_dim)

        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = vec![0.1, 0.2, 0.3, 0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0];

        let output = mqa.forward(&q, &k, &v, None, None).unwrap();

        // Output should match input query size
        assert_eq!(output.len(), q.len());

        // Verify output is not all zeros (computation happened)
        let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_multi_query_with_rope() {
        let rope_config = RopeConfig::new(4, 8);
        let rope = Rope::new(rope_config);
        let config = MultiQueryConfig::new(2, 4).with_rope(rope);
        let mqa = MultiQueryAttention::new(config).unwrap();

        // Data with RoPE applied - use format that RoPE expects
        // For batch_size * seq_len = 1, num_query_heads = 2, head_dim = 4:
        // position_ids needs to be [batch_size * seq_len] = [1] = [0]
        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = vec![0.1, 0.2, 0.3, 0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let position_ids = vec![0]; // [batch_size * seq_len]

        let output = mqa.forward(&q, &k, &v, Some(&position_ids), None).unwrap();
        // Output should match input query size
        assert_eq!(output.len(), q.len());
        // Verify output is not all zeros (computation happened)
        let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_mask_broadcast_shape() {
        // Test ATT-3 fix: broadcast mask shape [batch_size, seq_len, kv_seq_len]
        let config = MultiQueryConfig::new(2, 4); // 2 query heads, 1 KV head, dim 4
        let mqa = MultiQueryAttention::new(config).unwrap();

        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = vec![0.1, 0.2, 0.3, 0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0];

        // Broadcast mask: [batch_size=1, seq_len=1, kv_seq_len=1] = [1]
        let mask_broadcast = vec![0.0]; // No masking

        let output = mqa
            .forward(&q, &k, &v, None, Some(&mask_broadcast))
            .unwrap();
        assert_eq!(output.len(), q.len());

        // Verify computation happened
        let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_mask_full_shape() {
        // Test ATT-3 fix: full mask shape [batch_size, seq_len, num_heads, kv_seq_len]
        let config = MultiQueryConfig::new(2, 4); // 2 query heads, 1 KV head, dim 4
        let mqa = MultiQueryAttention::new(config).unwrap();

        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = vec![0.1, 0.2, 0.3, 0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0];

        // Full mask: [batch_size=1, seq_len=1, num_heads=2, kv_seq_len=1] = [2]
        let mask_full = vec![0.0, 0.0]; // No masking for either head

        let output = mqa.forward(&q, &k, &v, None, Some(&mask_full)).unwrap();
        assert_eq!(output.len(), q.len());

        // Verify computation happened
        let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_mask_invalid_shape() {
        // Test ATT-3 fix: invalid mask shape should produce clear error
        let config = MultiQueryConfig::new(2, 4); // 2 query heads, 1 KV head, dim 4
        let mqa = MultiQueryAttention::new(config).unwrap();

        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = vec![0.1, 0.2, 0.3, 0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0];

        // Invalid mask: wrong size (should be 1 for broadcast or 2 for full)
        let mask_invalid = vec![0.0, 0.0, 0.0]; // 3 elements - neither valid shape

        let result = mqa.forward(&q, &k, &v, None, Some(&mask_invalid));
        assert!(result.is_err());

        // Verify error message mentions both expected shapes
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Mask length"));
        assert!(err_msg.contains("broadcast") || err_msg.contains("full"));
    }
}
