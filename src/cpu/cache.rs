//! CPU KV cache and scratch buffers for inference.
//!
//! The KV cache stores key/value vectors for all positions seen so far.
//! Scratch buffers are reusable allocations for intermediate computations.

use crate::aligned::{AlignedVec, ALIGN_AVX512, ALIGN_GPU_STAGING};
use crate::config::ModelConfig;

// ── KV Cache ─────────────────────────────────────────────────────────────────────

/// Key-value cache for autoregressive decoding.
///
/// Stores K and V vectors for all layers and positions seen so far.
/// Layout: `k[layer][pos * kv_size + offset]` for position-based indexing.
pub struct CpuKvCache {
    /// Key cache: [num_layers][max_seq_len * kv_size]
    pub k: Vec<AlignedVec<f32>>,
    /// Value cache: [num_layers][max_seq_len * kv_size]
    pub v: Vec<AlignedVec<f32>>,
    /// Shortconv state cache: [num_layers][conv_state_size * hidden_size]
    /// Each shortconv layer stores the previous `l_cache - 1` Bx values.
    pub conv_state: Vec<AlignedVec<f32>>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Size of K/V per position: num_kv_heads * head_dim
    pub kv_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Shortconv L_cache (0 if model doesn't use shortconv)
    pub shortconv_l_cache: usize,
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
            .map(|_| AlignedVec::new_zeroed(buf_elems, ALIGN_AVX512))
            .collect();
        let v = (0..config.num_layers)
            .map(|_| AlignedVec::new_zeroed(buf_elems, ALIGN_AVX512))
            .collect();

        let shortconv_l_cache = config.shortconv_l_cache.unwrap_or(0);
        let conv_state_elems = if shortconv_l_cache > 1 {
            (shortconv_l_cache - 1) * config.hidden_size
        } else {
            0
        };
        let conv_state = (0..config.num_layers)
            .map(|_| AlignedVec::new_zeroed(conv_state_elems, ALIGN_AVX512))
            .collect();

        Self {
            k,
            v,
            conv_state,
            max_seq_len,
            kv_size,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            shortconv_l_cache,
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
            self.conv_state[layer].fill(0.0);
        }
    }

    /// Get total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let elements_per_layer = self.max_seq_len * self.kv_size;
        let bytes_per_layer = elements_per_layer * std::mem::size_of::<f32>();
        let kv_bytes = 2 * self.num_layers * bytes_per_layer; // K + V
        let conv_bytes = self.num_layers
            * self.conv_state.first().map(|v| v.len()).unwrap_or(0)
            * std::mem::size_of::<f32>();
        kv_bytes + conv_bytes
    }
}

// ── Forward Scratch Buffers ───────────────────────────────────────────────────────

/// Reusable scratch buffers for a single forward pass.
///
/// Allocated once and reused across all layers to avoid repeated allocations.
#[derive(Debug)]
pub struct CpuForwardScratch {
    /// Normalized hidden state [hidden_size]
    pub normed: AlignedVec<f32>,
    /// Query vector [num_heads * head_dim]
    pub q: AlignedVec<f32>,
    /// Key vector [num_kv_heads * head_dim]
    pub k: AlignedVec<f32>,
    /// Value vector [num_kv_heads * head_dim]
    pub v: AlignedVec<f32>,
    /// Attention output [num_heads * head_dim]
    pub attn_out: AlignedVec<f32>,
    /// Layer output (residual stream) [hidden_size]
    pub layer_out: AlignedVec<f32>,
    /// FFN gate projection [intermediate_size]
    pub gate: AlignedVec<f32>,
    /// FFN SwiGLU output [intermediate_size]
    pub swiglu: AlignedVec<f32>,
    /// Shortconv in_proj output [3 * hidden_size]
    pub shortconv_bcx: AlignedVec<f32>,
    /// Shortconv intermediate [hidden_size]
    pub shortconv_tmp: AlignedVec<f32>,
    /// Fused QKV scratch [num_heads*head_dim + 2*num_kv_heads*head_dim]
    pub qkv: AlignedVec<f32>,
    /// Final logits [vocab_size]
    pub logits: AlignedVec<f32>,
    /// Q8_0 scratch buffer for GEMV quantization [hidden_size / 32 * 34 bytes]
    /// Reused across all GEMV calls to avoid repeated heap allocations.
    pub q8_scratch: AlignedVec<u8>,
    /// Precomputed RoPE sin values for current position [head_dim / 2]
    pub rope_sin: AlignedVec<f32>,
    /// Precomputed RoPE cos values for current position [head_dim / 2]
    pub rope_cos: AlignedVec<f32>,
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
        use super::quant::Q8_BLOCK_BYTES;
        use super::quant::Q8_BLOCK_ELEMS;
        // Size for the largest GEMV in_dim: hidden_size or intermediate_size
        let max_in_dim = h.max(ff);
        let num_blocks = max_in_dim / Q8_BLOCK_ELEMS;
        let q8_scratch = AlignedVec::new_zeroed(num_blocks * Q8_BLOCK_BYTES, ALIGN_GPU_STAGING);
        let half = config.head_dim / 2;

        Self {
            normed: AlignedVec::new_zeroed(h, ALIGN_AVX512),
            q: AlignedVec::new_zeroed(q, ALIGN_AVX512),
            k: AlignedVec::new_zeroed(kv, ALIGN_AVX512),
            v: AlignedVec::new_zeroed(kv, ALIGN_AVX512),
            attn_out: AlignedVec::new_zeroed(q, ALIGN_AVX512),
            layer_out: AlignedVec::new_zeroed(h, ALIGN_AVX512),
            gate: AlignedVec::new_zeroed(ff, ALIGN_AVX512),
            swiglu: AlignedVec::new_zeroed(ff, ALIGN_AVX512),
            shortconv_bcx: AlignedVec::new_zeroed(3 * h, ALIGN_AVX512),
            shortconv_tmp: AlignedVec::new_zeroed(h, ALIGN_AVX512),
            qkv: AlignedVec::new_zeroed(q + 2 * kv, ALIGN_AVX512),
            logits: AlignedVec::new_zeroed(v, ALIGN_AVX512),
            q8_scratch,
            rope_sin: AlignedVec::new_zeroed(half, ALIGN_AVX512),
            rope_cos: AlignedVec::new_zeroed(half, ALIGN_AVX512),
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
            + self.shortconv_bcx.len()
            + self.shortconv_tmp.len()
            + self.qkv.len()
            + self.logits.len()
            + self.rope_sin.len()
            + self.rope_cos.len())
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
            rope_freq: (0..16)
                .map(|i| 1.0 / 1_000_000.0f32.powf((2 * i) as f32 / 32.0f32))
                .collect(),
            rope_neox: true,
            use_attention_bias: true,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            ffn_layout: crate::config::FfnLayout::SwiGLU,
            architecture: "qwen2".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
            shortconv_l_cache: None,
            num_dense_layers: None,
            num_experts_per_tok: None,
            use_expert_bias: false,
            expert_weights_scale: 1.0,
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
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
        assert_eq!(scratch.rope_sin.len(), 16); // head_dim / 2 = 32 / 2
        assert_eq!(scratch.rope_cos.len(), 16);
    }
}
