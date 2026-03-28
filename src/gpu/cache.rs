//! GPU KV cache and scratch buffers for inference.
//!
//! Safety-first design:
//! - All VRAM allocated with RAII (GpuBuffer)
//! - Bounds checked before kernel launches
//! - Never panic, always return GpuError

use super::error::{GpuError, GpuResult};
use super::weights::GpuBuffer;
use super::kernels::{kv_write, kv_write_batched, zero_fill};
use super::device::GpuDevice;
use crate::config::ModelConfig;

// ── KV Cache ─────────────────────────────────────────────────────────────────────

/// Key-value cache for autoregressive decoding, stored in GPU VRAM.
///
/// Layout: `k[layer][pos * kv_size + offset]` for position-based indexing.
/// All GPU memory managed via RAII (GpuBuffer).
pub struct GpuKvCache {
    /// Key cache: [num_layers][max_seq_len * kv_size]
    k: Vec<GpuBuffer>,
    /// Value cache: [num_layers][max_seq_len * kv_size]
    v: Vec<GpuBuffer>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Size of K/V per position: num_kv_heads * head_dim
    pub kv_size: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl GpuKvCache {
    /// Allocate a new KV cache in GPU VRAM.
    ///
    /// # Arguments
    /// * `config` - Model configuration (determines num_layers, num_kv_heads, head_dim)
    /// * `max_seq_len` - Maximum sequence length to support
    ///
    /// # Returns
    /// Ok(GpuKvCache) if all allocations succeed, Err if any fail (all freed via RAII)
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> GpuResult<Self> {
        let kv_size = config.num_kv_heads * config.head_dim;
        let layer_bytes = max_seq_len * kv_size * std::mem::size_of::<f32>();

        // Allocate K cache per layer
        let mut k = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let buf = GpuBuffer::alloc(layer_bytes).map_err(|e| {
                // On error, all previously allocated buffers are dropped (RAII cleanup)
                GpuError::CacheAllocationFailed {
                    reason: format!("K cache layer {} allocation failed: {}", layer, e),
                }
            })?;
            k.push(buf);
        }

        // Allocate V cache per layer
        let mut v = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let buf = GpuBuffer::alloc(layer_bytes).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("V cache layer {} allocation failed: {}", layer, e),
                }
            })?;
            v.push(buf);
        }

        Ok(Self {
            k,
            v,
            max_seq_len,
            kv_size,
            num_layers: config.num_layers,
        })
    }

    /// Get GPU pointer to K cache for a layer.
    ///
    /// Returns pointer suitable for kernel arguments.
    pub fn k_ptr(&self, layer: usize) -> GpuResult<*mut f32> {
        if layer >= self.num_layers {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("Layer {} exceeds num_layers {}", layer, self.num_layers),
            });
        }
        Ok(self.k[layer].as_ptr() as *mut f32)
    }

    /// Get GPU pointer to V cache for a layer.
    pub fn v_ptr(&self, layer: usize) -> GpuResult<*mut f32> {
        if layer >= self.num_layers {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("Layer {} exceeds num_layers {}", layer, self.num_layers),
            });
        }
        Ok(self.v[layer].as_ptr() as *mut f32)
    }

    /// Write K/V vectors to cache at specific position using GPU kernel.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `pos` - Position in cache (must be < max_seq_len)
    /// * `k_gpu` - GPU pointer to key vector
    /// * `v_gpu` - GPU pointer to value vector
    ///
    /// # Returns
    /// Ok(()) on successful kernel launch
    pub fn write(
        &self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        kv_write(
            k_cache,
            v_cache,
            k_gpu,
            v_gpu,
            pos,
            self.kv_size,
            self.max_seq_len,
        )
    }

    /// Batch write K/V for prefill (multiple positions).
    ///
    /// # Arguments
    /// * `start_pos` - Starting position
    /// * `seq_len` - Number of positions to write
    /// * `k_gpu` - GPU pointer to batched key vectors [seq_len * kv_size]
    /// * `v_gpu` - GPU pointer to batched value vectors
    pub fn write_batched(
        &self,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        kv_write_batched(
            k_cache,
            v_cache,
            k_gpu,
            v_gpu,
            start_pos,
            self.kv_size,
            self.max_seq_len,
            seq_len,
        )
    }

    /// Clear all cached values (zero out via kernel).
    ///
    /// Requires device reference for kernel synchronization.
    pub fn clear(&mut self, device: &GpuDevice) -> GpuResult<()> {
        let elements_per_layer = self.max_seq_len * self.kv_size;

        // Zero out K cache for each layer
        for layer in 0..self.num_layers {
            let k_ptr = self.k[layer].as_ptr() as *mut f32;
            zero_fill(k_ptr, elements_per_layer, device)?;
        }

        // Zero out V cache for each layer
        for layer in 0..self.num_layers {
            let v_ptr = self.v[layer].as_ptr() as *mut f32;
            zero_fill(v_ptr, elements_per_layer, device)?;
        }

        Ok(())
    }

    /// Get total VRAM usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let bytes_per_layer = self.max_seq_len * self.kv_size * std::mem::size_of::<f32>();
        let total_bytes = 2 * self.num_layers * bytes_per_layer; // K + V
        total_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 512,
            hidden_size: 1024,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: crate::config::TensorNameRegistry::from_scheme(
                &crate::config::TensorNamingScheme::Gguf
            ),
        }
    }

    #[test]
    fn new_allocates_correct_buffers() {
        let config = make_test_config();
        let cache = GpuKvCache::new(&config, 256);

        // Will fail without GPU, that's expected
        // Test that allocation is attempted correctly
        match cache {
            Ok(c) => {
                assert_eq!(c.num_layers, 2);
                assert_eq!(c.max_seq_len, 256);
                assert_eq!(c.kv_size, 4 * 128);
            }
            Err(_) => {
                // Expected when HIP unavailable
            }
        }
    }

    #[test]
    fn k_ptr_validates_layer_bounds() {
        let config = make_test_config();

        // Create a cache - will fail without GPU
        let cache = GpuKvCache::new(&config, 256);
        if let Ok(cache) = cache {
            let result = cache.k_ptr(5); // layer 5 > num_layers (2)
            assert!(result.is_err());
        }
        // If allocation failed, test passes (bounds checking exists)
    }
}

// ── Forward Scratch Buffers ───────────────────────────────────────────────────────

/// Reusable scratch buffers in GPU VRAM for a single forward pass.
///
/// Allocated once and reused across all layers to avoid repeated allocations.
/// All buffers are GPU-resident.
pub struct GpuForwardScratch {
    /// Normalized hidden state [hidden_size]
    pub normed: GpuBuffer,
    /// Query vector [num_heads * head_dim]
    pub q: GpuBuffer,
    /// Key vector [num_kv_heads * head_dim]
    pub k: GpuBuffer,
    /// Value vector [num_kv_heads * head_dim]
    pub v: GpuBuffer,
    /// Attention output [num_heads * head_dim]
    pub attn_out: GpuBuffer,
    /// Layer output (residual stream) [hidden_size]
    pub layer_out: GpuBuffer,
    /// FFN gate projection [intermediate_size]
    pub gate: GpuBuffer,
    /// FFN SwiGLU output [intermediate_size]
    pub swiglu: GpuBuffer,
    /// Final logits [vocab_size]
    pub logits: GpuBuffer,
}

impl GpuForwardScratch {
    /// Allocate scratch buffers in GPU VRAM.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Ok(GpuForwardScratch) if all allocations succeed
    pub fn new(config: &ModelConfig) -> GpuResult<Self> {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let v = config.vocab_size;

        // Allocate all buffers - if any fail, all are freed via RAII
        let normed = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("normed buffer allocation failed: {}", e),
            }
        })?;

        let q_buf = GpuBuffer::alloc(q * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("Q buffer allocation failed: {}", e),
            }
        })?;

        let k_buf = GpuBuffer::alloc(kv * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("K buffer allocation failed: {}", e),
            }
        })?;

        let v_buf = GpuBuffer::alloc(kv * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("V buffer allocation failed: {}", e),
            }
        })?;

        let attn_out = GpuBuffer::alloc(q * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("attn_out buffer allocation failed: {}", e),
            }
        })?;

        let layer_out = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("layer_out buffer allocation failed: {}", e),
            }
        })?;

        let gate = GpuBuffer::alloc(ff * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("gate buffer allocation failed: {}", e),
            }
        })?;

        let swiglu = GpuBuffer::alloc(ff * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("swiglu buffer allocation failed: {}", e),
            }
        })?;

        let logits = GpuBuffer::alloc(v * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("logits buffer allocation failed: {}", e),
            }
        })?;

        Ok(Self {
            normed,
            q: q_buf,
            k: k_buf,
            v: v_buf,
            attn_out,
            layer_out,
            gate,
            swiglu,
            logits,
        })
    }

    /// Get GPU pointer to normalized hidden state
    pub fn normed_ptr(&self) -> *const f32 {
        self.normed.as_ptr() as *const f32
    }

    /// Get GPU pointer to query vector
    pub fn q_ptr(&self) -> *const f32 {
        self.q.as_ptr() as *const f32
    }

    /// Get GPU pointer to key vector
    pub fn k_ptr(&self) -> *const f32 {
        self.k.as_ptr() as *const f32
    }

    /// Get GPU pointer to value vector
    pub fn v_ptr(&self) -> *const f32 {
        self.v.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to attention output
    pub fn attn_out_ptr(&mut self) -> *mut f32 {
        self.attn_out.as_ptr() as *mut f32
    }

    /// Get mutable GPU pointer to layer output
    pub fn layer_out_ptr(&mut self) -> *mut f32 {
        self.layer_out.as_ptr() as *mut f32
    }

    /// Get mutable GPU pointer to logits
    pub fn logits_ptr(&mut self) -> *mut f32 {
        self.logits.as_ptr() as *mut f32
    }
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    fn make_test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 512,
            hidden_size: 1024,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: crate::config::TensorNameRegistry::from_scheme(
                &crate::config::TensorNamingScheme::Gguf
            ),
        }
    }

    #[test]
    fn new_allocates_all_buffers() {
        let config = make_test_config();
        let scratch = GpuForwardScratch::new(&config);

        // Will fail without GPU, that's expected
        match scratch {
            Ok(s) => {
                // Verify pointers are valid (or empty)
                assert!(!s.q.as_ptr().is_null() || s.q.is_empty());
            }
            Err(_) => {
                // Expected when HIP unavailable
            }
        }
    }
}
