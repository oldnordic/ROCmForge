//! Rotary Positional Embeddings (RoPE) implementation
//!
//! RoPE applies relative position encoding to query and key tensors using rotation matrices.
//! This implementation supports both CPU and GPU computation.

use crate::attention::{AttentionError, AttentionResult};
#[cfg(feature = "rocm")]
use crate::backend::DeviceTensor;

/// RoPE configuration
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Dimension of each head
    pub head_dim: usize,
    /// Maximum sequence length for precomputed frequencies
    pub max_seq_len: usize,
    /// Base frequency for rotary embeddings
    pub base: f32,
    /// Whether to use scaled RoPE (for longer sequences)
    pub scaled: bool,
    /// Scaling factor for scaled RoPE
    pub scale: f32,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            max_seq_len: 2048,
            base: 10000.0,
            scaled: false,
            scale: 8.0,
        }
    }
}

impl RopeConfig {
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base: 10000.0,
            scaled: false,
            scale: 8.0,
        }
    }

    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    pub fn with_scaling(mut self, scale: f32) -> Self {
        self.scaled = true;
        self.scale = scale;
        self
    }
}

/// Rotary Positional Embeddings implementation
#[derive(Debug, Clone)]
pub struct Rope {
    config: RopeConfig,
    /// Precomputed cosine frequencies [max_seq_len, head_dim/2]
    cos: Vec<f32>,
    /// Precomputed sine frequencies [max_seq_len, head_dim/2]
    sin: Vec<f32>,
}

impl Rope {
    /// Create new RoPE with given configuration
    pub fn new(config: RopeConfig) -> Self {
        let head_dim = config.head_dim;
        let max_seq_len = config.max_seq_len;

        assert_eq!(
            head_dim % 2,
            0,
            "Head dimension must be even for RoPE, got {}",
            head_dim
        );

        let mut cos = vec![0.0f32; max_seq_len * (head_dim / 2)];
        let mut sin = vec![0.0f32; max_seq_len * (head_dim / 2)];

        // Precompute frequencies
        for pos in 0..max_seq_len {
            for i in 0..head_dim / 2 {
                let idx = pos * (head_dim / 2) + i;

                // Apply scaling if enabled
                let effective_pos = if config.scaled {
                    (pos as f32 / config.scale).floor() as usize
                } else {
                    pos
                };

                let freq =
                    effective_pos as f32 / config.base.powf(2.0 * i as f32 / head_dim as f32);
                cos[idx] = freq.cos();
                sin[idx] = freq.sin();
            }
        }

        Self { config, cos, sin }
    }

    /// Apply RoPE to query tensor in-place
    ///
    /// # Arguments
    /// * `x` - Query tensor [seq_len, num_heads, head_dim] or [batch_size, seq_len, num_heads, head_dim]
    /// * `position_ids` - Position IDs [seq_len] or [batch_size, seq_len]
    pub fn apply_q(
        &self,
        x: &mut [f32],
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<()> {
        self.apply_rope(x, position_ids, num_heads)
    }

    /// Apply RoPE to key tensor in-place
    ///
    /// # Arguments
    /// * `x` - Key tensor [seq_len, num_heads, head_dim] or [batch_size, seq_len, num_heads, head_dim]
    /// * `position_ids` - Position IDs [seq_len] or [batch_size, seq_len]
    pub fn apply_k(
        &self,
        x: &mut [f32],
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<()> {
        self.apply_rope(x, position_ids, num_heads)
    }

    /// Apply RoPE to tensor in-place
    fn apply_rope(
        &self,
        x: &mut [f32],
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<()> {
        let head_dim = self.config.head_dim;
        let half_dim = head_dim / 2;

        // Determine tensor shape
        let (batch_size, seq_len) = if x.len() == position_ids.len() * num_heads * head_dim {
            (1, position_ids.len())
        } else {
            // Assume batch_size * seq_len * num_heads * head_dim
            let total_elements = x.len();
            let seq_len = position_ids.len() / num_heads;
            let batch_size = total_elements / (seq_len * num_heads * head_dim);
            (batch_size, seq_len)
        };

        if x.len() != batch_size * seq_len * num_heads * head_dim {
            return Err(AttentionError::ShapeMismatch(format!(
                "Tensor size {} doesn't match expected shape [batch_size={}, seq_len={}, num_heads={}, head_dim={}]",
                x.len(), batch_size, seq_len, num_heads, head_dim
            )));
        }

        // Apply RoPE to each sequence position
        for b in 0..batch_size {
            for s in 0..seq_len {
                let pos_id = position_ids[b * seq_len + s];
                if pos_id >= self.config.max_seq_len {
                    return Err(AttentionError::DimensionError(format!(
                        "Position ID {} exceeds maximum sequence length {}",
                        pos_id, self.config.max_seq_len
                    )));
                }

                // Get precomputed cos/sin for this position
                let cos_offset = pos_id * half_dim;
                let sin_offset = pos_id * half_dim;

                for h in 0..num_heads {
                    let tensor_offset = b * seq_len * num_heads * head_dim
                        + s * num_heads * head_dim
                        + h * head_dim;

                    // Apply rotation: x_rotated = x * cos + x_flipped * sin
                    for i in 0..half_dim {
                        let x1 = x[tensor_offset + i];
                        let x2 = x[tensor_offset + i + half_dim];

                        let cos_val = self.cos[cos_offset + i];
                        let sin_val = self.sin[sin_offset + i];

                        x[tensor_offset + i] = x1 * cos_val - x2 * sin_val;
                        x[tensor_offset + i + half_dim] = x1 * sin_val + x2 * cos_val;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply RoPE to DeviceTensor on GPU
    #[cfg(feature = "rocm")]
    pub fn apply_q_device(
        &self,
        x: &mut DeviceTensor,
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<()> {
        self.apply_rope_device(x, position_ids, num_heads)
    }

    /// Apply RoPE to DeviceTensor on GPU
    #[cfg(feature = "rocm")]
    pub fn apply_k_device(
        &self,
        x: &mut DeviceTensor,
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<()> {
        self.apply_rope_device(x, position_ids, num_heads)
    }

    /// Apply RoPE to DeviceTensor on GPU
    #[cfg(feature = "rocm")]
    fn apply_rope_device(
        &self,
        x: &mut DeviceTensor,
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<()> {
        use crate::attention::kernels::rope_gpu_kernel;
        use crate::backend::hip_backend::HipBackend;
        use crate::loader::mmap_loader::TensorShape;

        let head_dim = self.config.head_dim;
        let seq_len = position_ids.len();

        // Validate input shape
        let expected_elements = seq_len * num_heads * head_dim;
        if x.len() != expected_elements {
            return Err(AttentionError::ShapeMismatch(format!(
                "DeviceTensor size {} doesn't match expected shape [seq_len={}, num_heads={}, head_dim={}]",
                x.len(), seq_len, num_heads, head_dim
            )));
        }

        // Validate head_dim is even
        if !head_dim.is_multiple_of(2) {
            return Err(AttentionError::DimensionError(format!(
                "Head dimension must be even for RoPE, got {}",
                head_dim
            )));
        }

        // Create backend for kernel execution
        let backend = HipBackend::new().map_err(|e| {
            AttentionError::HandleCreation(format!("Failed to create HIP backend: {}", e))
        })?;

        // Upload cos/sin to GPU for the positions we need
        // cos/sin shape: [seq_len, head_dim/2]
        let half_dim = head_dim / 2;

        // Check position bounds
        for &pos in position_ids {
            if pos >= self.config.max_seq_len {
                return Err(AttentionError::DimensionError(format!(
                    "Position ID {} exceeds maximum sequence length {}",
                    pos, self.config.max_seq_len
                )));
            }
        }

        // Extract cos/sin for the positions we need
        let mut cos_gpu = Vec::with_capacity(seq_len * half_dim);
        let mut sin_gpu = Vec::with_capacity(seq_len * half_dim);
        for &pos in position_ids {
            let cos_offset = pos * half_dim;
            let sin_offset = pos * half_dim;
            cos_gpu.extend_from_slice(&self.cos[cos_offset..cos_offset + half_dim]);
            sin_gpu.extend_from_slice(&self.sin[sin_offset..sin_offset + half_dim]);
        }

        // Create cos/sin device tensors
        let cos_shape = TensorShape::from_dims(&[seq_len, half_dim]);
        let cos_device =
            DeviceTensor::from_host_vec(&backend, cos_gpu, cos_shape).map_err(|e| {
                AttentionError::MemoryAllocation(format!("Failed to allocate cos tensor: {}", e))
            })?;

        let sin_shape = TensorShape::from_dims(&[seq_len, half_dim]);
        let sin_device =
            DeviceTensor::from_host_vec(&backend, sin_gpu, sin_shape).map_err(|e| {
                AttentionError::MemoryAllocation(format!("Failed to allocate sin tensor: {}", e))
            })?;

        // Get device pointers
        let input_ptr = x.buffer().as_mut_ptr() as *mut f32;
        let cos_ptr = cos_device.as_ptr() as *const f32;
        let sin_ptr = sin_device.as_ptr() as *const f32;

        // Call GPU kernel
        let result = unsafe {
            rope_gpu_kernel(
                input_ptr,
                cos_ptr,
                sin_ptr,
                seq_len as u32,
                num_heads as u32,
                head_dim as u32,
            )
        };

        if result != 0 {
            return Err(AttentionError::GpuOperation(
                "GPU kernel execution failed".to_string(),
            ));
        }

        // Synchronize to ensure kernel completes
        backend.synchronize().map_err(|e| {
            AttentionError::Synchronization(format!("GPU synchronization failed: {}", e))
        })?;

        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &RopeConfig {
        &self.config
    }

    /// Get precomputed cos values
    pub fn cos(&self) -> &[f32] {
        &self.cos
    }

    /// Get precomputed sin values  
    pub fn sin(&self) -> &[f32] {
        &self.sin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let config = RopeConfig::new(128, 1024);
        let rope = Rope::new(config);

        assert_eq!(rope.cos.len(), 1024 * 64); // max_seq_len * head_dim/2
        assert_eq!(rope.sin.len(), 1024 * 64);
    }

    #[test]
    fn test_rope_application() {
        let config = RopeConfig::new(4, 8); // Small dimensions for testing
        let rope = Rope::new(config);

        // Create test tensor: [batch=1, seq_len=2, num_heads=1, head_dim=4]
        let mut x = vec![
            1.0, 2.0, 3.0, 4.0, // position 0 (identity - no change)
            5.0, 6.0, 7.0, 8.0, // position 1 (should rotate)
        ];
        let position_ids = vec![0, 1];

        rope.apply_q(&mut x, &position_ids, 1).unwrap();

        // Position 0 is identity (cos=1, sin=0), so values don't change
        assert_eq!(x[0], 1.0); // Position 0: no rotation
        assert_eq!(x[1], 2.0); // Position 0: no rotation

        // Position 1 should have rotation applied
        assert_ne!(x[4], 5.0); // Position 1: rotated
        assert_ne!(x[5], 6.0); // Position 1: rotated
    }

    #[test]
    fn test_rope_odd_head_dim() {
        let config = RopeConfig::new(5, 8); // Odd dimension should fail
        let result = std::panic::catch_unwind(|| {
            Rope::new(config);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_position_id_overflow() {
        let config = RopeConfig::new(4, 4); // max_seq_len = 4
        let rope = Rope::new(config);

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let position_ids = vec![5]; // Position 5 exceeds max_seq_len

        let result = rope.apply_q(&mut x, &position_ids, 1);
        assert!(result.is_err());
    }
}
