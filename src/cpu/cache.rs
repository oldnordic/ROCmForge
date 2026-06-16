//! CPU KV cache and scratch buffers for inference.
//!
//! The KV cache stores key/value vectors for all positions seen so far.
//! Scratch buffers are reusable allocations for intermediate computations.

use crate::aligned::{AlignedVec, ALIGN_AVX512, ALIGN_GPU_STAGING};
use crate::config::ModelConfig;
use std::collections::HashMap;

// ── KV Cache ─────────────────────────────────────────────────────────────────────

/// Key-value cache for autoregressive decoding.
///
/// Stores K and V vectors for all layers and positions seen so far.
/// Layout: `k[layer][pos * kv_size + offset]` for position-based indexing.
pub struct CpuKvCache {
    /// Key cache: [num_layers][max_seq_len * kv_size(layer)]
    pub k: Vec<AlignedVec<f32>>,
    /// Value cache: [num_layers][max_seq_len * kv_size(layer)]
    pub v: Vec<AlignedVec<f32>>,
    /// Shortconv state cache: [num_layers][conv_state_size * hidden_size]
    /// Each shortconv layer stores the previous `l_cache - 1` Bx values.
    pub conv_state: Vec<AlignedVec<f32>>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Size of K/V per position for each layer.
    pub per_layer_kv_size: Vec<usize>,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Shortconv L_cache (0 if model doesn't use shortconv)
    pub shortconv_l_cache: usize,
    /// Shared KV states for Gemma4 KV sharing, keyed by layer type.
    pub shared_kv: HashMap<String, (AlignedVec<f32>, AlignedVec<f32>)>,
}

impl CpuKvCache {
    /// K/V size for a specific layer.
    pub fn kv_size(&self, layer: usize) -> usize {
        self.per_layer_kv_size
            .get(layer)
            .copied()
            .unwrap_or(self.hidden_size)
    }
}

impl Clone for CpuKvCache {
    fn clone(&self) -> Self {
        Self {
            k: self.k.to_vec(),
            v: self.v.to_vec(),
            conv_state: self.conv_state.to_vec(),
            max_seq_len: self.max_seq_len,
            per_layer_kv_size: self.per_layer_kv_size.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            shortconv_l_cache: self.shortconv_l_cache,
            shared_kv: self.shared_kv.clone(),
        }
    }
}

impl CpuKvCache {
    /// Allocate a new KV cache.
    ///
    /// # Arguments
    /// * `config` - Model configuration (determines num_layers, num_kv_heads, head_dim)
    /// * `max_seq_len` - Maximum sequence length to support
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> Self {
        let per_layer_kv_size: Vec<usize> = (0..config.num_layers)
            .map(|layer| config.kv_size(layer))
            .collect();

        let mut k = Vec::with_capacity(config.num_layers);
        let mut v = Vec::with_capacity(config.num_layers);
        for kv_size in &per_layer_kv_size {
            let buf_elems = max_seq_len * kv_size;
            k.push(AlignedVec::new_zeroed(buf_elems, ALIGN_AVX512));
            v.push(AlignedVec::new_zeroed(buf_elems, ALIGN_AVX512));
        }

        let shortconv_l_cache = config.shortconv_l_cache.unwrap_or(0);
        let conv_state_elems = if shortconv_l_cache > 1 {
            (shortconv_l_cache - 1) * config.hidden_size
        } else {
            0
        };
        let conv_state = (0..config.num_layers)
            .map(|_| AlignedVec::new_zeroed(conv_state_elems, ALIGN_AVX512))
            .collect();

        // Gemma4: allocate shared KV buffers for the last non-shared layer of each type.
        let mut shared_kv = HashMap::new();
        if config.architecture == "gemma4" && config.num_kv_shared_layers > 0 {
            let first = config.first_kv_shared_layer_idx();
            for layer in 0..first {
                if config.stores_shared_kv(layer) {
                    let ty = config.layer_type_for_layer(layer).to_string();
                    let kv_size = config.kv_size(layer);
                    let buf_elems = max_seq_len * kv_size;
                    shared_kv.insert(
                        ty,
                        (
                            AlignedVec::new_zeroed(buf_elems, ALIGN_AVX512),
                            AlignedVec::new_zeroed(buf_elems, ALIGN_AVX512),
                        ),
                    );
                }
            }
        }

        Self {
            k,
            v,
            conv_state,
            max_seq_len,
            per_layer_kv_size,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            shortconv_l_cache,
            shared_kv,
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
        let kv_size = self.kv_size(layer);
        let start = pos * kv_size;
        &self.k[layer][start..start + kv_size]
    }

    /// Get the V slice for a specific position within a layer.
    pub fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        let kv_size = self.kv_size(layer);
        let start = pos * kv_size;
        &self.v[layer][start..start + kv_size]
    }

    /// Write K values at a specific position.
    pub fn write_k(&mut self, layer: usize, pos: usize, k: &[f32]) {
        let kv_size = self.kv_size(layer);
        let start = pos * kv_size;
        self.k[layer][start..start + kv_size].copy_from_slice(k);
    }

    /// Write V values at a specific position.
    pub fn write_v(&mut self, layer: usize, pos: usize, v: &[f32]) {
        let kv_size = self.kv_size(layer);
        let start = pos * kv_size;
        self.v[layer][start..start + kv_size].copy_from_slice(v);
    }

    /// Read-only shared K/V buffers for a layer type (Gemma4).
    pub fn shared_kv(&self, layer_type: &str) -> Option<(&[f32], &[f32])> {
        self.shared_kv.get(layer_type).map(|(k, v)| (&**k, &**v))
    }

    /// Mutable shared K/V buffers for a layer type (Gemma4).
    pub fn shared_kv_mut(&mut self, layer_type: &str) -> Option<(&mut [f32], &mut [f32])> {
        self.shared_kv
            .get_mut(layer_type)
            .map(|(k, v)| (&mut **k, &mut **v))
    }

    /// Write K values to the shared buffer for a layer type at a position.
    pub fn write_shared_k(&mut self, layer_type: &str, pos: usize, k: &[f32]) {
        if let Some((buf, _)) = self.shared_kv.get_mut(layer_type) {
            let kv_size = k.len();
            let start = pos * kv_size;
            buf[start..start + kv_size].copy_from_slice(k);
        }
    }

    /// Write V values to the shared buffer for a layer type at a position.
    pub fn write_shared_v(&mut self, layer_type: &str, pos: usize, v: &[f32]) {
        if let Some((_, buf)) = self.shared_kv.get_mut(layer_type) {
            let kv_size = v.len();
            let start = pos * kv_size;
            buf[start..start + kv_size].copy_from_slice(v);
        }
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
        let kv_bytes: usize = self
            .per_layer_kv_size
            .iter()
            .map(|kv_size| self.max_seq_len * kv_size * std::mem::size_of::<f32>())
            .sum::<usize>()
            * 2; // K + V
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
    /// Query vector [max per-layer q_size]
    pub q: AlignedVec<f32>,
    /// Key vector [max per-layer kv_size]
    pub k: AlignedVec<f32>,
    /// Value vector [max per-layer kv_size]
    pub v: AlignedVec<f32>,
    /// Attention output [max per-layer q_size]
    pub attn_out: AlignedVec<f32>,
    /// Layer output (residual stream) [hidden_size]
    pub layer_out: AlignedVec<f32>,
    /// FFN gate projection [max per-layer intermediate_size]
    pub gate: AlignedVec<f32>,
    /// FFN SwiGLU output [max per-layer intermediate_size]
    pub swiglu: AlignedVec<f32>,
    /// Shortconv in_proj output [3 * hidden_size]
    pub shortconv_bcx: AlignedVec<f32>,
    /// Shortconv intermediate [hidden_size]
    pub shortconv_tmp: AlignedVec<f32>,
    /// Fused QKV scratch [max q_size + 2*max kv_size]
    pub qkv: AlignedVec<f32>,
    /// Final logits [vocab_size]
    pub logits: AlignedVec<f32>,
    /// Q8_0 scratch buffer for GEMV quantization [hidden_size / 32 * 34 bytes]
    /// Reused across all GEMV calls to avoid repeated heap allocations.
    pub q8_scratch: AlignedVec<u8>,
    /// Precomputed RoPE sin values for current position [max rotated_dims / 2]
    pub rope_sin: AlignedVec<f32>,
    /// Precomputed RoPE cos values for current position [max rotated_dims / 2]
    pub rope_cos: AlignedVec<f32>,
    /// Per-layer embedding (PLE) inputs for Gemma4 [num_layers * ple_dim]
    pub ple_input: AlignedVec<f32>,
    /// Per-layer embedding (PLE) projection scratch for Gemma4 [num_layers * ple_dim]
    pub ple_proj: AlignedVec<f32>,
}

impl CpuForwardScratch {
    /// Allocate scratch buffers sized for the given model config.
    pub fn new(config: &ModelConfig) -> Self {
        let h = config.hidden_size;
        let v = config.vocab_size;

        let max_q_size = (0..config.num_layers)
            .map(|layer| config.q_size(layer))
            .max()
            .unwrap_or(config.num_heads * config.head_dim);
        let max_kv_size = (0..config.num_layers)
            .map(|layer| config.kv_size(layer))
            .max()
            .unwrap_or(config.num_kv_heads * config.head_dim);
        let max_ff_size = (0..config.num_layers)
            .map(|layer| config.intermediate_size_for_layer(layer))
            .max()
            .unwrap_or(config.intermediate_size);
        let max_rotated_half = (0..config.num_layers)
            .map(|layer| {
                let head_dim = config.head_dim_for_layer(layer);
                let factor = config.rope_partial_factor_for_layer(layer);
                ((head_dim as f32 * factor) as usize / 2).max(1)
            })
            .max()
            .unwrap_or(config.head_dim / 2);

        // Q8_0 scratch buffer for GEMV quantization
        // Size: (hidden_size / 32) * 34 bytes per Q8_0 block
        use super::quant::Q8_BLOCK_BYTES;
        use super::quant::Q8_BLOCK_ELEMS;
        // Size for the largest GEMV in_dim: hidden_size or intermediate_size
        let max_in_dim = h.max(max_ff_size);
        let num_blocks = max_in_dim / Q8_BLOCK_ELEMS;
        let q8_scratch = AlignedVec::new_zeroed(num_blocks * Q8_BLOCK_BYTES, ALIGN_GPU_STAGING);

        let ple_elems = if config.hidden_size_per_layer_input > 0 {
            config.num_layers * config.hidden_size_per_layer_input
        } else {
            0
        };

        Self {
            normed: AlignedVec::new_zeroed(h, ALIGN_AVX512),
            q: AlignedVec::new_zeroed(max_q_size, ALIGN_AVX512),
            k: AlignedVec::new_zeroed(max_kv_size, ALIGN_AVX512),
            v: AlignedVec::new_zeroed(max_kv_size, ALIGN_AVX512),
            attn_out: AlignedVec::new_zeroed(max_q_size, ALIGN_AVX512),
            layer_out: AlignedVec::new_zeroed(h, ALIGN_AVX512),
            gate: AlignedVec::new_zeroed(max_ff_size, ALIGN_AVX512),
            swiglu: AlignedVec::new_zeroed(max_ff_size, ALIGN_AVX512),
            shortconv_bcx: AlignedVec::new_zeroed(3 * h, ALIGN_AVX512),
            shortconv_tmp: AlignedVec::new_zeroed(h, ALIGN_AVX512),
            qkv: AlignedVec::new_zeroed(max_q_size + 2 * max_kv_size, ALIGN_AVX512),
            logits: AlignedVec::new_zeroed(v, ALIGN_AVX512),
            q8_scratch,
            rope_sin: AlignedVec::new_zeroed(max_rotated_half, ALIGN_AVX512),
            rope_cos: AlignedVec::new_zeroed(max_rotated_half, ALIGN_AVX512),
            ple_input: AlignedVec::new_zeroed(ple_elems, ALIGN_AVX512),
            ple_proj: AlignedVec::new_zeroed(ple_elems, ALIGN_AVX512),
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
            + self.rope_cos.len()
            + self.ple_input.len()
            + self.ple_proj.len())
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
            ..Default::default()
        }
    }

    #[test]
    fn kv_cache_allocates_correct_size() {
        let config = make_test_config();
        let max_seq = 64;
        let kv = CpuKvCache::new(&config, max_seq);

        assert_eq!(kv.num_layers, 4);
        assert_eq!(kv.max_seq_len, 64);
        assert_eq!(kv.kv_size(0), 2 * 32); // num_kv_heads * head_dim

        // Each layer should have correct buffer size
        let expected_len = max_seq * kv.kv_size(0);
        for layer in 0..kv.num_layers {
            assert_eq!(kv.k[layer].len(), expected_len);
            assert_eq!(kv.v[layer].len(), expected_len);
        }
    }

    #[test]
    fn kv_cache_write_read() {
        let config = make_test_config();
        let mut kv = CpuKvCache::new(&config, 64);

        let test_k: Vec<f32> = (0..kv.kv_size(0)).map(|i| i as f32).collect();
        let test_v: Vec<f32> = (0..kv.kv_size(0)).map(|i| i as f32 * 2.0).collect();

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
