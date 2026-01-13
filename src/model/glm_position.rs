//! GLM-specific position ID handling
//!
//! GLM (General Language Model) uses a unique position encoding scheme that differs from
//! standard transformers. This module provides position ID generation and handling
//! specifically for GLM models.

use crate::attention::rope::{Rope, RopeConfig};
use crate::attention::{AttentionError, AttentionResult};

/// GLM position configuration
#[derive(Debug, Clone, Default)]
pub struct GlmPositionConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Whether to use bidirectional attention
    pub bidirectional: bool,
    /// RoPE configuration (if using rotary embeddings)
    pub rope_config: Option<RopeConfig>,
}

impl GlmPositionConfig {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            max_seq_len,
            bidirectional: false,
            rope_config: None,
        }
    }

    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    pub fn with_rope(mut self, rope_config: RopeConfig) -> Self {
        self.rope_config = Some(rope_config);
        self
    }
}

/// GLM position ID handler
#[derive(Debug, Clone)]
pub struct GlmPositionHandler {
    config: GlmPositionConfig,
    rope: Option<Rope>,
}

impl GlmPositionHandler {
    /// Create new GLM position handler
    pub fn new(config: GlmPositionConfig) -> AttentionResult<Self> {
        let rope = config
            .rope_config
            .as_ref()
            .map(|rope_config| Rope::new(rope_config.clone()));

        Ok(Self { config, rope })
    }

    /// Generate position IDs for GLM attention
    ///
    /// GLM uses a different position scheme than standard transformers:
    /// - For causal attention: positions are [0, 1, 2, ..., seq_len-1]
    /// - For bidirectional: positions can be more complex
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `batch_size` - Batch size (optional, defaults to 1)
    ///
    /// # Returns
    /// * Position IDs [batch_size * seq_len]
    pub fn generate_position_ids(&self, seq_len: usize, batch_size: Option<usize>) -> Vec<usize> {
        let batch_size = batch_size.unwrap_or(1);
        let mut position_ids = Vec::with_capacity(batch_size * seq_len);

        for _b in 0..batch_size {
            if self.config.bidirectional {
                // For bidirectional attention, GLM can use different position schemes
                // One common approach is to use absolute positions
                for pos in 0..seq_len {
                    position_ids.push(pos);
                }
            } else {
                // For causal attention (standard GLM), use sequential positions
                for pos in 0..seq_len {
                    position_ids.push(pos);
                }
            }
        }

        position_ids
    }

    /// Generate position IDs for a specific attention pattern
    ///
    /// GLM supports various attention patterns beyond simple causal/bidirectional
    pub fn generate_pattern_position_ids(
        &self,
        pattern: &GlmAttentionPattern,
        seq_len: usize,
        batch_size: Option<usize>,
    ) -> AttentionResult<Vec<usize>> {
        let batch_size = batch_size.unwrap_or(1);
        let mut position_ids = Vec::with_capacity(batch_size * seq_len);

        match pattern {
            GlmAttentionPattern::Causal => {
                // Standard causal attention: [0, 1, 2, ...]
                for _b in 0..batch_size {
                    for pos in 0..seq_len {
                        position_ids.push(pos);
                    }
                }
            }
            GlmAttentionPattern::Bidirectional => {
                // Bidirectional: [0, 1, 2, ...] (same as causal for position IDs)
                for _b in 0..batch_size {
                    for pos in 0..seq_len {
                        position_ids.push(pos);
                    }
                }
            }
            GlmAttentionPattern::LocalWindow { window_size } => {
                // Local window attention: positions relative to center
                for _b in 0..batch_size {
                    for pos in 0..seq_len {
                        // Use position relative to local window center
                        let _window_center = pos;
                        let relative_pos = if pos >= *window_size {
                            pos - window_size
                        } else {
                            0
                        };
                        position_ids.push(relative_pos);
                    }
                }
            }
            GlmAttentionPattern::GlobalLocal {
                global_positions,
                local_window,
            } => {
                // Mixed global-local attention pattern
                for _b in 0..batch_size {
                    for pos in 0..seq_len {
                        if global_positions.contains(&pos) {
                            // Global positions use absolute position
                            position_ids.push(pos);
                        } else {
                            // Local positions use relative position within window
                            let relative_pos = pos % local_window;
                            position_ids.push(relative_pos);
                        }
                    }
                }
            }
            GlmAttentionPattern::Custom(custom_fn) => {
                // Custom position generation function
                for _b in 0..batch_size {
                    for pos in 0..seq_len {
                        position_ids.push(custom_fn(pos));
                    }
                }
            }
        }

        Ok(position_ids)
    }

    /// Apply position embeddings to query and key tensors
    ///
    /// This handles GLM-specific position encoding, including RoPE if configured
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, seq_len, num_heads, head_dim]
    /// * `k` - Key tensor [batch_size, seq_len, num_heads, head_dim]
    /// * `position_ids` - Position IDs [batch_size, seq_len]
    /// * `num_heads` - Number of attention heads
    ///
    /// # Returns
    /// * (query_with_pos, key_with_pos) - Tensors with position embeddings applied
    pub fn apply_position_embeddings(
        &self,
        mut q: Vec<f32>,
        mut k: Vec<f32>,
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<(Vec<f32>, Vec<f32>)> {
        // Validate inputs
        if position_ids.is_empty() {
            return Err(AttentionError::DimensionError(
                "Position IDs cannot be empty".to_string(),
            ));
        }

        let seq_len = position_ids.len();
        let expected_q_size = seq_len * num_heads * self.get_head_dim();
        let expected_k_size = seq_len * num_heads * self.get_head_dim();

        if q.len() != expected_q_size {
            return Err(AttentionError::ShapeMismatch(format!(
                "Query tensor size {} doesn't match expected {} for seq_len={}, heads={}",
                q.len(),
                expected_q_size,
                seq_len,
                num_heads
            )));
        }

        if k.len() != expected_k_size {
            return Err(AttentionError::ShapeMismatch(format!(
                "Key tensor size {} doesn't match expected {} for seq_len={}, heads={}",
                k.len(),
                expected_k_size,
                seq_len,
                num_heads
            )));
        }

        // Apply RoPE if configured
        if let Some(rope) = &self.rope {
            rope.apply_q(&mut q, position_ids, num_heads)?;
            rope.apply_k(&mut k, position_ids, num_heads)?;
        } else {
            // For GLM without RoPE, we might apply other position encoding schemes
            // For now, leave tensors unchanged (could add learned position embeddings here)
        }

        Ok((q, k))
    }

    /// Apply position embeddings to DeviceTensors
    #[cfg(feature = "rocm")]
    pub fn apply_position_embeddings_device(
        &self,
        q: crate::backend::DeviceTensor,
        k: crate::backend::DeviceTensor,
        position_ids: &[usize],
        num_heads: usize,
    ) -> AttentionResult<(crate::backend::DeviceTensor, crate::backend::DeviceTensor)> {
        use crate::attention::kernels::position_embeddings_gpu_kernel;
        use crate::backend::hip_backend::HipBackend;
        use crate::backend::DeviceTensor;
        use crate::loader::mmap_loader::TensorShape;

        // Validate inputs
        if position_ids.is_empty() {
            return Err(AttentionError::DimensionError(
                "Position IDs cannot be empty".to_string(),
            ));
        }

        let seq_len = position_ids.len();
        let head_dim = self.get_head_dim();

        let expected_q_size = seq_len * num_heads * head_dim;
        let expected_k_size = seq_len * num_heads * head_dim;

        if q.len() != expected_q_size {
            return Err(AttentionError::ShapeMismatch(format!(
                "Query tensor size {} doesn't match expected {} for seq_len={}, heads={}",
                q.len(),
                expected_q_size,
                seq_len,
                num_heads
            )));
        }

        if k.len() != expected_k_size {
            return Err(AttentionError::ShapeMismatch(format!(
                "Key tensor size {} doesn't match expected {} for seq_len={}, heads={}",
                k.len(),
                expected_k_size,
                seq_len,
                num_heads
            )));
        }

        // Apply RoPE if configured
        if let Some(rope) = &self.rope {
            // Try GPU kernel first, fall back to CPU if it fails
            let gpu_success = if let Ok(backend) = HipBackend::new() {
                // Upload cos/sin to GPU for the positions we need
                // cos/sin shape: [seq_len, head_dim/2]
                let half_dim = head_dim / 2;

                // Check position bounds
                for &pos in position_ids {
                    if pos >= rope.config().max_seq_len {
                        return Err(AttentionError::DimensionError(format!(
                            "Position ID {} exceeds maximum sequence length {}",
                            pos,
                            rope.config().max_seq_len
                        )));
                    }
                }

                // Extract cos/sin for the positions we need
                let mut cos_gpu = Vec::with_capacity(seq_len * half_dim);
                let mut sin_gpu = Vec::with_capacity(seq_len * half_dim);
                for &pos in position_ids {
                    let cos_offset = pos * half_dim;
                    let sin_offset = pos * half_dim;
                    cos_gpu.extend_from_slice(&rope.cos()[cos_offset..cos_offset + half_dim]);
                    sin_gpu.extend_from_slice(&rope.sin()[sin_offset..sin_offset + half_dim]);
                }

                // Create cos/sin device tensors
                let cos_shape = TensorShape::from_dims(&[seq_len, half_dim]);
                let cos_device =
                    DeviceTensor::from_host_vec(&backend, cos_gpu, cos_shape).map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to allocate cos tensor: {}",
                            e
                        ))
                    })?;

                let sin_shape = TensorShape::from_dims(&[seq_len, half_dim]);
                let sin_device =
                    DeviceTensor::from_host_vec(&backend, sin_gpu, sin_shape).map_err(|e| {
                        AttentionError::MemoryAllocation(format!(
                            "Failed to allocate sin tensor: {}",
                            e
                        ))
                    })?;

                // Get device pointers
                let q_ptr = q.buffer().as_mut_ptr() as *mut f32;
                let k_ptr = k.buffer().as_mut_ptr() as *mut f32;
                let cos_ptr = cos_device.as_ptr() as *const f32;
                let sin_ptr = sin_device.as_ptr() as *const f32;

                // Call GPU kernel
                let result = unsafe {
                    position_embeddings_gpu_kernel(
                        q_ptr,
                        k_ptr,
                        cos_ptr,
                        sin_ptr,
                        seq_len as u32,
                        num_heads as u32,
                        head_dim as u32,
                    )
                };

                if result == 0 {
                    // Synchronize to ensure kernel completes
                    if backend.synchronize().is_ok() {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            // If GPU kernel failed, fall back to CPU implementation
            if !gpu_success {
                // CPU fallback: download tensors, apply RoPE, upload back
                let q_host = q.to_host_vec()
                    .map_err(|e| AttentionError::MemoryCopy(format!("Failed to download Q tensor: {}", e)))?;
                let k_host = k.to_host_vec()
                    .map_err(|e| AttentionError::MemoryCopy(format!("Failed to download K tensor: {}", e)))?;

                let mut q_with_pos = q_host.clone();
                let mut k_with_pos = k_host.clone();
                rope.apply_q(&mut q_with_pos, position_ids, num_heads)
                    .and_then(|_| rope.apply_k(&mut k_with_pos, position_ids, num_heads))
                    .map_err(|e| AttentionError::GpuOperation(format!("CPU RoPE fallback failed: {}", e)))?;

                // Create new tensors with the modified data
                let backend = HipBackend::new()
                    .map_err(|e| AttentionError::HandleCreation(format!("Failed to create backend for CPU fallback: {}", e)))?;

                let q_shape = q.shape().clone();
                let q_new = DeviceTensor::from_host_vec(&backend, q_with_pos, q_shape)
                    .map_err(|e| AttentionError::MemoryAllocation(format!("Failed to upload Q tensor: {}", e)))?;

                let k_shape = k.shape().clone();
                let k_new = DeviceTensor::from_host_vec(&backend, k_with_pos, k_shape)
                    .map_err(|e| AttentionError::MemoryAllocation(format!("Failed to upload K tensor: {}", e)))?;

                return Ok((q_new, k_new));
            }
        } else {
            // For GLM without RoPE, we might apply other position encoding schemes
            // For now, leave tensors unchanged (could add learned position embeddings here)
        }

        Ok((q, k))
    }

    /// Get head dimension from configuration
    fn get_head_dim(&self) -> usize {
        self.config
            .rope_config
            .as_ref()
            .map(|config| config.head_dim)
            .unwrap_or(128) // Default head dimension
    }

    /// Validate position IDs for GLM constraints
    pub fn validate_position_ids(&self, position_ids: &[usize]) -> AttentionResult<()> {
        if position_ids.is_empty() {
            return Err(AttentionError::DimensionError(
                "Position IDs cannot be empty".to_string(),
            ));
        }

        for (idx, &pos) in position_ids.iter().enumerate() {
            if pos >= self.config.max_seq_len {
                return Err(AttentionError::DimensionError(format!(
                    "Position ID {} at index {} exceeds maximum sequence length {}",
                    pos, idx, self.config.max_seq_len
                )));
            }
        }

        Ok(())
    }

    /// Get attention mask for GLM position pattern
    ///
    /// GLM can use different masking strategies based on position pattern
    pub fn get_attention_mask(
        &self,
        seq_len: usize,
        pattern: &GlmAttentionPattern,
    ) -> AttentionResult<Vec<f32>> {
        let mut mask = vec![0.0f32; seq_len * seq_len];

        match pattern {
            GlmAttentionPattern::Causal => {
                // Causal mask: only attend to previous positions
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if j > i {
                            mask[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            GlmAttentionPattern::Bidirectional => {
                // No masking for bidirectional attention
                // All zeros (no masking)
            }
            GlmAttentionPattern::LocalWindow { window_size } => {
                // Local window mask: only attend within window
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let distance = (i as i32 - j as i32).abs();
                        if distance > *window_size as i32 {
                            mask[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            GlmAttentionPattern::GlobalLocal {
                global_positions,
                local_window,
            } => {
                // Mixed global-local masking
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let can_attend =
                            if global_positions.contains(&i) || global_positions.contains(&j) {
                                // At least one is global: can attend
                                true
                            } else {
                                // Both are local: check window
                                (i as i32 - j as i32).abs() <= *local_window as i32
                            };

                        if !can_attend {
                            mask[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            GlmAttentionPattern::Custom(_) => {
                // Custom masking: user should provide their own mask
                // Return empty mask (no masking)
            }
        }

        Ok(mask)
    }

    /// Get configuration
    pub fn config(&self) -> &GlmPositionConfig {
        &self.config
    }

    /// Get RoPE instance if available
    pub fn rope(&self) -> Option<&Rope> {
        self.rope.as_ref()
    }
}

/// GLM attention patterns
#[derive(Debug, Clone, Default)]
pub enum GlmAttentionPattern {
    /// Standard causal attention (default for GLM)
    #[default]
    Causal,
    /// Bidirectional attention
    Bidirectional,
    /// Local window attention
    LocalWindow { window_size: usize },
    /// Mixed global-local attention
    GlobalLocal {
        global_positions: Vec<usize>,
        local_window: usize,
    },
    /// Custom pattern with user-defined function
    Custom(fn(usize) -> usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm_position_config() {
        let config = GlmPositionConfig::new(1024);
        assert_eq!(config.max_seq_len, 1024);
        assert!(!config.bidirectional);

        let config_bidirectional = config.with_bidirectional(true);
        assert!(config_bidirectional.bidirectional);
    }

    #[test]
    fn test_position_id_generation() {
        let config = GlmPositionConfig::new(8);
        let handler = GlmPositionHandler::new(config).unwrap();

        let position_ids = handler.generate_position_ids(4, Some(2));
        assert_eq!(position_ids.len(), 8); // 2 batches * 4 seq_len
        assert_eq!(position_ids, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn test_causal_mask() {
        let config = GlmPositionConfig::new(4);
        let handler = GlmPositionHandler::new(config).unwrap();

        let mask = handler
            .get_attention_mask(4, &GlmAttentionPattern::Causal)
            .unwrap();

        // Causal mask: position i can only attend to positions <= i
        // Mask uses NEG_INFINITY for positions that CANNOT be attended to

        // Position 0 can only attend to position 0 (self)
        assert_eq!(mask[0 * 4 + 0], 0.0); // Can attend to self
        assert_eq!(mask[0 * 4 + 1], f32::NEG_INFINITY); // Cannot attend to future position 1
        assert_eq!(mask[0 * 4 + 2], f32::NEG_INFINITY); // Cannot attend to future position 2

        // Position 1 can attend to positions 0 and 1
        assert_eq!(mask[1 * 4 + 0], 0.0); // Can attend to past position 0
        assert_eq!(mask[1 * 4 + 1], 0.0); // Can attend to self
        assert_eq!(mask[1 * 4 + 2], f32::NEG_INFINITY); // Cannot attend to future position 2

        // Position 2 can attend to positions 0, 1, and 2
        assert_eq!(mask[2 * 4 + 0], 0.0); // Can attend to past position 0
        assert_eq!(mask[2 * 4 + 1], 0.0); // Can attend to past position 1
        assert_eq!(mask[2 * 4 + 2], 0.0); // Can attend to self
        assert_eq!(mask[2 * 4 + 3], f32::NEG_INFINITY); // Cannot attend to future position 3
    }

    #[test]
    fn test_position_validation() {
        let config = GlmPositionConfig::new(4);
        let handler = GlmPositionHandler::new(config).unwrap();

        // Valid position IDs
        let valid_ids = vec![0, 1, 2, 3];
        assert!(handler.validate_position_ids(&valid_ids).is_ok());

        // Invalid position ID (exceeds max_seq_len)
        let invalid_ids = vec![0, 1, 2, 4]; // 4 >= max_seq_len(4)
        assert!(handler.validate_position_ids(&invalid_ids).is_err());
    }

    #[test]
    fn test_local_window_pattern() {
        let pattern = GlmAttentionPattern::LocalWindow { window_size: 2 };
        let config = GlmPositionConfig::new(4);
        let handler = GlmPositionHandler::new(config).unwrap();

        let position_ids = handler
            .generate_pattern_position_ids(&pattern, 4, None)
            .unwrap();
        assert_eq!(position_ids.len(), 4);

        let mask = handler.get_attention_mask(4, &pattern).unwrap();

        // Check that positions beyond window are masked
        // For position 0, can attend to 0, 1, 2 (within window)
        assert_eq!(mask[0 * 4 + 0], 0.0); // Can attend
        assert_eq!(mask[0 * 4 + 1], 0.0); // Can attend
        assert_eq!(mask[0 * 4 + 2], 0.0); // Can attend
        assert_eq!(mask[0 * 4 + 3], f32::NEG_INFINITY); // Cannot attend (distance 3 > window_size 2)
    }

    #[test]
    fn test_with_rope() {
        let rope_config = RopeConfig::new(64, 1024);
        let config = GlmPositionConfig::new(4).with_rope(rope_config);
        let handler = GlmPositionHandler::new(config).unwrap();

        assert!(handler.rope().is_some());

        let q = vec![1.0; 4 * 8 * 64]; // batch=1, seq=4, heads=8, dim=64
        let k = vec![0.5; 4 * 8 * 64];
        let position_ids = vec![0, 1, 2, 3];

        let q_original = q.clone();
        let k_original = k.clone();

        let (q_with_pos, k_with_pos) = handler
            .apply_position_embeddings(q, k, &position_ids, 8)
            .unwrap();

        // Output should be different from input (RoPE applied)
        assert_ne!(q_with_pos, q_original);
        assert_ne!(k_with_pos, k_original);
    }
}
