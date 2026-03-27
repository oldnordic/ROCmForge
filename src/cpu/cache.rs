//! CPU KV cache and scratch buffers for inference.
//!
//! The KV cache stores key/value vectors for all positions seen so far.
//! Scratch buffers are reusable allocations for intermediate computations.

use crate::config::ModelConfig;

// ── KV Cache ─────────────────────────────────────────────────────────────────────

/// Key-value cache for autoregressive decoding.
///
/// Stores K and V vectors for all layers and positions seen so far.
/// Layout: `k[layer][pos * kv_size + offset]` for position-based indexing.
pub struct CpuKvCache {
    /// Key cache: [num_layers][max_seq_len * kv_size]
    pub k: Vec<Vec<f32>>,
    /// Value cache: [num_layers][max_seq_len * kv_size]
    pub v: Vec<Vec<f32>>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Size of K/V per position: num_kv_heads * head_dim
    pub kv_size: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl CpuKvCache {
    /// Allocate a new KV cache.
    ///
    /// # Arguments
    /// * `config` - Model configuration (determines num_layers, num_kv_heads, head_dim)
    /// * `max_seq_len` - Maximum sequence length to support
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> Self {
        let kv_size = config.num_kv_heads * config.head_dim;
        let buf_elems = max_seq_len * kv_size;
        let k = (0..config.num_layers)
            .map(|_| vec![0.0f32; buf_elems])
            .collect();
        let v = (0..config.num_layers)
            .map(|_| vec![0.0f32; buf_elems])
            .collect();
        Self {
            k,
            v,
            max_seq_len,
            kv_size,
            num_layers: config.num_layers,
        }
    }

    /// Get K buffer for a layer (read-only).
    pub fn k_buf(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }

    /// Get V buffer for a layer (read-only).
    pub fn v_buf(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    /// Get K buffer for a layer (mutable).
    pub fn k_buf_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.k[layer]
    }

    /// Get V buffer for a layer (mutable).
    pub fn v_buf_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.v[layer]
    }

    /// Get the K slice for a specific position within a layer.
    pub fn k_at(&self, layer: usize, pos: usize) -> &[f32] {
        let start = pos * self.kv_size;
        &self.k[layer][start..start + self.kv_size]
    }

    /// Get the V slice for a specific position within a layer.
    pub fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        let start = pos * self.kv_size;
        &self.v[layer][start..start + self.kv_size]
    }

    /// Write K values at a specific position.
    pub fn write_k(&mut self, layer: usize, pos: usize, k: &[f32]) {
        let start = pos * self.kv_size;
        self.k[layer][start..start + self.kv_size].copy_from_slice(k);
    }

    /// Write V values at a specific position.
    pub fn write_v(&mut self, layer: usize, pos: usize, v: &[f32]) {
        let start = pos * self.kv_size;
        self.v[layer][start..start + self.kv_size].copy_from_slice(v);
    }

    /// Clear all cached values (zero out).
    pub fn clear(&mut self) {
        for layer in 0..self.num_layers {
            self.k[layer].fill(0.0);
            self.v[layer].fill(0.0);
        }
    }

    /// Get total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let elements_per_layer = self.max_seq_len * self.kv_size;
        let bytes_per_layer = elements_per_layer * std::mem::size_of::<f32>();
        let total_bytes = 2 * self.num_layers * bytes_per_layer; // K + V
        total_bytes
    }
}

// ── Forward Scratch Buffers ───────────────────────────────────────────────────────

/// Reusable scratch buffers for a single forward pass.
///
/// Allocated once and reused across all layers to avoid repeated allocations.
pub struct CpuForwardScratch {
    /// Normalized hidden state [hidden_size]
    pub normed: Vec<f32>,
    /// Query vector [num_heads * head_dim]
    pub q: Vec<f32>,
    /// Key vector [num_kv_heads * head_dim]
    pub k: Vec<f32>,
    /// Value vector [num_kv_heads * head_dim]
    pub v: Vec<f32>,
    /// Attention output [num_heads * head_dim]
    pub attn_out: Vec<f32>,
    /// Layer output (residual stream) [hidden_size]
    pub layer_out: Vec<f32>,
    /// FFN gate projection [intermediate_size]
    pub gate: Vec<f32>,
    /// FFN SwiGLU output [intermediate_size]
    pub swiglu: Vec<f32>,
    /// Final logits [vocab_size]
    pub logits: Vec<f32>,
    /// Q8_0 scratch buffer for GEMV quantization [hidden_size / 32 * 34 bytes]
    /// Reused across all GEMV calls to avoid repeated heap allocations.
    pub q8_scratch: Vec<u8>,
}

impl CpuForwardScratch {
    /// Allocate scratch buffers sized for the given model config.
    pub fn new(config: &ModelConfig) -> Self {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let v = config.vocab_size;

        // Q8_0 scratch buffer for GEMV quantization
        // Size: (hidden_size / 32) * 34 bytes per Q8_0 block
        use super::quant::Q8_BLOCK_ELEMS;
        use super::quant::Q8_BLOCK_BYTES;
        let num_blocks = h / Q8_BLOCK_ELEMS;
        let q8_scratch = vec![0u8; num_blocks * Q8_BLOCK_BYTES];

        Self {
            normed: vec![0.0; h],
            q: vec![0.0; q],
            k: vec![0.0; kv],
            v: vec![0.0; kv],
            attn_out: vec![0.0; q],
            layer_out: vec![0.0; h],
            gate: vec![0.0; ff],
            swiglu: vec![0.0; ff],
            logits: vec![0.0; v],
            q8_scratch,
        }
    }

    /// Get total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.normed.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn_out.len()
            + self.layer_out.len()
            + self.gate.len()
            + self.swiglu.len()
            + self.logits.len())
            * std::mem::size_of::<f32>()
            + self.q8_scratch.len() * std::mem::size_of::<u8>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TensorNameRegistry, TensorNamingScheme};

    fn make_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 4,
            hidden_size: 256,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate_size: 512,
            vocab_size: 1000,
            max_seq_len: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_neox: true,
            use_attention_bias: true,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "qwen2".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
        }
    }

    #[test]
    fn kv_cache_allocates_correct_size() {
        let config = make_test_config();
        let max_seq = 64;
        let kv = CpuKvCache::new(&config, max_seq);

        assert_eq!(kv.num_layers, 4);
        assert_eq!(kv.max_seq_len, 64);
        assert_eq!(kv.kv_size, 2 * 32); // num_kv_heads * head_dim

        // Each layer should have correct buffer size
        let expected_len = max_seq * kv.kv_size;
        for layer in 0..kv.num_layers {
            assert_eq!(kv.k[layer].len(), expected_len);
            assert_eq!(kv.v[layer].len(), expected_len);
        }
    }

    #[test]
    fn kv_cache_write_read() {
        let config = make_test_config();
        let mut kv = CpuKvCache::new(&config, 64);

        let test_k: Vec<f32> = (0..kv.kv_size).map(|i| i as f32).collect();
        let test_v: Vec<f32> = (0..kv.kv_size).map(|i| i as f32 * 2.0).collect();

        kv.write_k(0, 5, &test_k);
        kv.write_v(0, 5, &test_v);

        // Verify read back
        let read_k = kv.k_at(0, 5);
        let read_v = kv.v_at(0, 5);

        assert_eq!(read_k, test_k.as_slice());
        assert_eq!(read_v, test_v.as_slice());
    }

    #[test]
    fn scratch_buffer_sizes() {
        let config = make_test_config();
        let scratch = CpuForwardScratch::new(&config);

        assert_eq!(scratch.normed.len(), 256);
        assert_eq!(scratch.q.len(), 8 * 32); // num_heads * head_dim
        assert_eq!(scratch.k.len(), 2 * 32); // num_kv_heads * head_dim
        assert_eq!(scratch.v.len(), 2 * 32);
        assert_eq!(scratch.attn_out.len(), 8 * 32);
        assert_eq!(scratch.layer_out.len(), 256);
        assert_eq!(scratch.gate.len(), 512);
        assert_eq!(scratch.swiglu.len(), 512);
        assert_eq!(scratch.logits.len(), 1000);
    }
}
