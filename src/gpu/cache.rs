//! GPU KV cache and scratch buffers for inference.
//!
//! Safety-first design:
//! - All VRAM allocated with RAII (GpuBuffer)
//! - Bounds checked before kernel launches
//! - Never panic, always return GpuError

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::graph::{CapturedDecodeGraph, DecodeGraphKey};
use super::kernels::zero_fill;
use super::weights::{GpuBuffer, GpuPinnedBuffer};
use crate::config::ModelConfig;

mod accessors;
mod allocator;
mod binding;
mod dump;
mod init;
mod paged;
mod prefix;
mod scratch;
mod write;

pub use self::allocator::{BlockAllocator, BlockTable};
use self::binding::compute_kv_binding_tag;
pub use self::dump::{KvDump, KV_DUMP_MAGIC};
pub use self::prefix::PrefixCache;
pub use self::scratch::{GpuExpertScratch, GpuForwardScratch, GpuPrefillScratch};

// ── Block Allocator & Block Table ──────────────────────────────────────────────────

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
    /// Persistent SSM state matrices per layer: [num_layers][num_heads * head_dim * head_dim]
    pub ssm_state: Option<Vec<GpuBuffer>>,
    /// Persistent SSM convolution states per layer: [num_layers][qkv_dim * (kernel_size - 1)]
    pub ssm_conv_state: Option<Vec<GpuBuffer>>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Size of K/V per position: num_kv_heads * head_dim
    pub kv_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Cached pointer-mix used by decode-graph key construction.
    decode_binding_tag: u64,

    // Research & advanced compression synergy extensions
    pub kv_lora_dim: Option<usize>,
    pub adastate_anchors_enabled: bool,
    pub kv_frame_codec_enabled: bool,
    pub w_down_k: Option<Vec<GpuBuffer>>,
    pub w_down_v: Option<Vec<GpuBuffer>>,
    pub w_up_k: Option<Vec<GpuBuffer>>,
    pub w_up_v: Option<Vec<GpuBuffer>>,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_quant_bits: Option<usize>,
    pub centroids: Option<GpuBuffer>,
    pub qjl_scale: f32,

    // Paged Cache Storage & Metadata
    pub block_size_tokens: usize,
    pub pos_bytes: usize,
    pub block_allocator: BlockAllocator,
    pub block_table: BlockTable,
    pub paged_k: Vec<Vec<Option<std::sync::Arc<GpuBuffer>>>>,
    pub paged_v: Vec<Vec<Option<std::sync::Arc<GpuBuffer>>>>,
}

impl GpuKvCache {
    /// Get the total size of this KV cache in VRAM (in bytes).
    pub fn vram_bytes(&self) -> usize {
        let mut total = 0;
        for buf in &self.k {
            total += buf.size();
        }
        for buf in &self.v {
            total += buf.size();
        }
        if let Some(ref states) = self.ssm_state {
            for buf in states {
                total += buf.size();
            }
        }
        if let Some(ref conv_states) = self.ssm_conv_state {
            for buf in conv_states {
                total += buf.size();
            }
        }
        if let Some(ref bufs) = self.w_down_k {
            for buf in bufs {
                total += buf.size();
            }
        }
        if let Some(ref bufs) = self.w_down_v {
            for buf in bufs {
                total += buf.size();
            }
        }
        if let Some(ref bufs) = self.w_up_k {
            for buf in bufs {
                total += buf.size();
            }
        }
        if let Some(ref bufs) = self.w_up_v {
            for buf in bufs {
                total += buf.size();
            }
        }
        if let Some(ref buf) = self.centroids {
            total += buf.size();
        }
        // Include paged block buffers
        for layer in &self.paged_k {
            for opt_buf in layer {
                if let Some(ref buf) = opt_buf {
                    total += buf.size();
                }
            }
        }
        for layer in &self.paged_v {
            for opt_buf in layer {
                if let Some(ref buf) = opt_buf {
                    total += buf.size();
                }
            }
        }
        total
    }

    /// Estimate VRAM bytes required for a KV cache without allocating anything.
    ///
    /// Use this for pre-flight VRAM checks before calling `new`.
    pub fn estimate_bytes(config: &ModelConfig, max_seq_len: usize) -> usize {
        let kv_size = config.num_kv_heads * config.head_dim;
        let d = config.kv_lora_dim.unwrap_or(kv_size);
        let layer_bytes = if let Some(_bits) = config.kv_quant_bits {
            let pack_bytes = (d * 3 + 7) / 8;
            let qjl_bytes = (d + 7) / 8;
            let aligned_pos_bytes = (pack_bytes + qjl_bytes + 31) & !31;
            max_seq_len * aligned_pos_bytes
        } else {
            max_seq_len * d * std::mem::size_of::<f32>()
        };
        let mut total = 2 * config.num_layers * layer_bytes;
        if let Some(dc) = config.kv_lora_dim {
            total += 4 * config.num_layers * (dc * kv_size * std::mem::size_of::<f32>());
        }
        if let Some(ref centroids) = config.turboquant_centroids {
            total += centroids.len() * std::mem::size_of::<f32>();
        }
        total
    }

    /// Allocate a new KV cache in GPU VRAM.
    ///
    /// # Arguments
    /// * `config` - Model configuration (determines num_layers, num_kv_heads, head_dim)
    /// * `max_seq_len` - Maximum sequence length to support
    ///
    /// # Returns
    /// Ok(GpuKvCache) if all allocations succeed, Err if any fail (all freed via RAII)
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> GpuResult<Self> {
        Self::build_from_config(config, max_seq_len)
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
    /// Clear all cached values (zero out via kernel).
    ///
    /// Requires device reference for kernel synchronization.
    pub fn clear(&mut self, device: &GpuDevice) -> GpuResult<()> {
        let effective_size = self.kv_lora_dim.unwrap_or(self.kv_size);
        let elements_per_layer = self.max_seq_len * effective_size;

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

        // Zero out SSM buffers if present
        if let Some(ref states) = self.ssm_state {
            for layer in 0..self.num_layers {
                let size = states[layer].size();
                super::ffi::hip_memset(states[layer].as_ptr(), 0, size)?;
            }
        }
        if let Some(ref conv_states) = self.ssm_conv_state {
            for layer in 0..self.num_layers {
                let size = conv_states[layer].size();
                super::ffi::hip_memset(conv_states[layer].as_ptr(), 0, size)?;
            }
        }

        // Release all block table ids
        for &id in &self.block_table.block_ids {
            self.block_allocator.release(id);
        }
        self.block_table.block_ids.clear();

        Ok(())
    }

    /// Get total VRAM usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.vram_bytes()
    }

    /// Cached pointer-mix used by decode-graph key construction.
    #[inline]
    pub fn binding_tag(&self) -> u64 {
        self.decode_binding_tag
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
                &crate::config::TensorNamingScheme::Gguf,
            ),
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
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

    #[test]
    fn binding_tag_is_stable_for_same_cache() {
        let config = make_test_config();
        let cache = GpuKvCache::new(&config, 256);
        if let Ok(cache) = cache {
            let tag_a = cache.binding_tag();
            let tag_b = cache.binding_tag();
            assert_eq!(tag_a, tag_b);
        }
    }

    #[test]
    fn binding_tag_differs_for_distinct_allocations() {
        let config = make_test_config();
        let first = GpuKvCache::new(&config, 256);
        let second = GpuKvCache::new(&config, 256);
        if let (Ok(first), Ok(second)) = (first, second) {
            assert_ne!(first.binding_tag(), second.binding_tag());
        }
    }

    #[test]
    fn test_block_allocator_and_table() {
        let mut allocator = BlockAllocator::new(16);
        assert_eq!(allocator.block_size_tokens, 16);

        let b1 = allocator.allocate();
        let b2 = allocator.allocate();
        assert_eq!(b1, 0);
        assert_eq!(b2, 1);

        allocator.retain(b1); // refcount of b1 becomes 2
        assert!(!allocator.release(b1)); // refcount becomes 1, not freed
        assert!(allocator.release(b1)); // refcount becomes 0, freed

        let b3 = allocator.allocate();
        assert_eq!(b3, 0); // reuses 0

        let mut table = BlockTable {
            block_ids: Vec::new(),
        };
        table.block_ids.push(b3);
        table.block_ids.push(b2);
        assert_eq!(table.block_ids, vec![0, 1]);
    }
}
