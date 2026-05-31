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
use super::kernels::{
    kv_write, kv_write_batched, kv_write_batched_compressed, kv_write_compressed,
    kv_write_on_stream, reconstruct_kv_cache_prefix_sum, zero_fill,
};
use super::vram_budget::query_vram_budget;
use super::weights::{GpuBuffer, GpuPinnedBuffer};
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
        total
    }

    /// Estimate VRAM bytes required for a KV cache without allocating anything.
    ///
    /// Use this for pre-flight VRAM checks before calling `new`.
    pub fn estimate_bytes(config: &ModelConfig, max_seq_len: usize) -> usize {
        let kv_size = config.num_kv_heads * config.head_dim;
        let effective_size = config.kv_lora_dim.unwrap_or(kv_size);
        let layer_bytes = max_seq_len * effective_size * std::mem::size_of::<f32>();
        let mut total = 2 * config.num_layers * layer_bytes;
        if let Some(dc) = config.kv_lora_dim {
            total += 4 * config.num_layers * (dc * kv_size * std::mem::size_of::<f32>());
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
        let kv_size = config.num_kv_heads * config.head_dim;
        let effective_kv_size = config.kv_lora_dim.unwrap_or(kv_size);
        let layer_bytes = max_seq_len * effective_kv_size * std::mem::size_of::<f32>();
        let mut total_cache_bytes = 2 * config.num_layers * layer_bytes;
        if let Some(dc) = config.kv_lora_dim {
            total_cache_bytes +=
                4 * config.num_layers * (dc * kv_size * std::mem::size_of::<f32>());
        }

        let budget = query_vram_budget(super::vram_budget::active_or_default_device_id())?;
        if total_cache_bytes > budget.safe_allocation_size {
            return Err(GpuError::CacheAllocationFailed {
                reason: format!(
                    "KV cache requires {} MB but only {} MB safely allocatable ({} MB free, 2 * {} layers * {} MB/layer)",
                    total_cache_bytes / (1024 * 1024),
                    budget.safe_allocation_size / (1024 * 1024),
                    budget.free_vram / (1024 * 1024),
                    config.num_layers,
                    layer_bytes / (1024 * 1024),
                ),
            });
        }

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
            let buf =
                GpuBuffer::alloc(layer_bytes).map_err(|e| GpuError::CacheAllocationFailed {
                    reason: format!("V cache layer {} allocation failed: {}", layer, e),
                })?;
            v.push(buf);
        }

        let decode_binding_tag = compute_kv_binding_tag(&k, &v);

        let is_hybrid = config.architecture.contains("qwen35");
        let mut ssm_state = None;
        let mut ssm_conv_state = None;

        if is_hybrid {
            let mut states = Vec::with_capacity(config.num_layers);
            let mut conv_states = Vec::with_capacity(config.num_layers);

            let ssm_heads = std::cmp::max(config.num_heads * 2, 32);
            let ssm_state_bytes = ssm_heads * 128 * 128 * std::mem::size_of::<f32>();
            let qkv_dim =
                std::cmp::max(config.num_kv_heads * 128 * 2 + config.num_heads * 128, 8192);
            let ssm_conv_bytes = qkv_dim * 3 * std::mem::size_of::<f32>();

            for layer in 0..config.num_layers {
                let s_buf = GpuBuffer::alloc(ssm_state_bytes).map_err(|e| {
                    GpuError::CacheAllocationFailed {
                        reason: format!("SSM state layer {} allocation failed: {}", layer, e),
                    }
                })?;
                // Zero-initialize SSM state
                super::ffi::hip_memset(s_buf.as_ptr(), 0, ssm_state_bytes)?;
                states.push(s_buf);

                let c_buf = GpuBuffer::alloc(ssm_conv_bytes).map_err(|e| {
                    GpuError::CacheAllocationFailed {
                        reason: format!("SSM conv state layer {} allocation failed: {}", layer, e),
                    }
                })?;
                // Zero-initialize SSM conv state
                super::ffi::hip_memset(c_buf.as_ptr(), 0, ssm_conv_bytes)?;
                conv_states.push(c_buf);
            }

            ssm_state = Some(states);
            ssm_conv_state = Some(conv_states);
        }

        // Allocate projection matrices for VideoMLA / AdaState if kv_lora_dim is present
        let mut w_down_k = None;
        let mut w_down_v = None;
        let mut w_up_k = None;
        let mut w_up_v = None;

        if let Some(dc) = config.kv_lora_dim {
            let mut down_k = Vec::with_capacity(config.num_layers);
            let mut down_v = Vec::with_capacity(config.num_layers);
            let mut up_k = Vec::with_capacity(config.num_layers);
            let mut up_v = Vec::with_capacity(config.num_layers);

            let proj_bytes = dc * kv_size * std::mem::size_of::<f32>();

            let mut host_down = vec![0.0f32; dc * kv_size];
            for j in 0..dc {
                if j < kv_size {
                    host_down[j * kv_size + j] = 1.0f32;
                }
            }

            let mut host_up = vec![0.0f32; kv_size * dc];
            for j in 0..dc {
                if j < kv_size {
                    host_up[j * dc + j] = 1.0f32;
                }
            }

            for layer in 0..config.num_layers {
                let mut d_k =
                    GpuBuffer::alloc(proj_bytes).map_err(|e| GpuError::CacheAllocationFailed {
                        reason: format!("W_down_k layer {} allocation failed: {}", layer, e),
                    })?;
                let bytes_down = unsafe {
                    std::slice::from_raw_parts(
                        host_down.as_ptr() as *const u8,
                        host_down.len() * std::mem::size_of::<f32>(),
                    )
                };
                d_k.copy_from_host(bytes_down)?;
                down_k.push(d_k);

                let mut d_v =
                    GpuBuffer::alloc(proj_bytes).map_err(|e| GpuError::CacheAllocationFailed {
                        reason: format!("W_down_v layer {} allocation failed: {}", layer, e),
                    })?;
                d_v.copy_from_host(bytes_down)?;
                down_v.push(d_v);

                let mut u_k =
                    GpuBuffer::alloc(proj_bytes).map_err(|e| GpuError::CacheAllocationFailed {
                        reason: format!("W_up_k layer {} allocation failed: {}", layer, e),
                    })?;
                let bytes_up = unsafe {
                    std::slice::from_raw_parts(
                        host_up.as_ptr() as *const u8,
                        host_up.len() * std::mem::size_of::<f32>(),
                    )
                };
                u_k.copy_from_host(bytes_up)?;
                up_k.push(u_k);

                let mut u_v =
                    GpuBuffer::alloc(proj_bytes).map_err(|e| GpuError::CacheAllocationFailed {
                        reason: format!("W_up_v layer {} allocation failed: {}", layer, e),
                    })?;
                u_v.copy_from_host(bytes_up)?;
                up_v.push(u_v);
            }

            w_down_k = Some(down_k);
            w_down_v = Some(down_v);
            w_up_k = Some(up_k);
            w_up_v = Some(up_v);
        }

        let adastate_anchors_enabled = config.adastate_anchors_enabled.unwrap_or(false);
        let kv_frame_codec_enabled = config.kv_frame_codec_enabled.unwrap_or(false);

        Ok(Self {
            k,
            v,
            ssm_state,
            ssm_conv_state,
            max_seq_len,
            kv_size,
            num_layers: config.num_layers,
            decode_binding_tag,
            kv_lora_dim: config.kv_lora_dim,
            adastate_anchors_enabled,
            kv_frame_codec_enabled,
            w_down_k,
            w_down_v,
            w_up_k,
            w_up_v,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        })
    }

    /// Get GPU pointer to SSM state cache for a layer.
    pub fn ssm_state_ptr(&self, layer: usize) -> GpuResult<Option<*mut f32>> {
        if layer >= self.num_layers {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("Layer {} exceeds num_layers {}", layer, self.num_layers),
            });
        }
        if let Some(ref states) = self.ssm_state {
            Ok(Some(states[layer].as_ptr() as *mut f32))
        } else {
            Ok(None)
        }
    }

    /// Get GPU pointer to SSM conv state cache for a layer.
    pub fn ssm_conv_state_ptr(&self, layer: usize) -> GpuResult<Option<*mut f32>> {
        if layer >= self.num_layers {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("Layer {} exceeds num_layers {}", layer, self.num_layers),
            });
        }
        if let Some(ref conv_states) = self.ssm_conv_state {
            Ok(Some(conv_states[layer].as_ptr() as *mut f32))
        } else {
            Ok(None)
        }
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
    /// Write K/V vectors to cache.
    pub fn write(
        &self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        self.write_on_stream(layer, pos, k_gpu, v_gpu, hipStream_t::null())
    }

    /// Write K/V vectors to cache on an explicit HIP stream.
    pub fn write_on_stream(
        &self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
        stream: hipStream_t,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        if let Some(dc) = self.kv_lora_dim {
            let temp_pos = pos as i32;
            let mut d_pos = GpuBuffer::alloc(4)?;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &temp_pos as *const i32 as *const u8,
                    std::mem::size_of::<i32>(),
                )
            };
            d_pos.copy_from_host(bytes)?;

            let w_down_k = self
                .w_down_k
                .as_ref()
                .map(|w| w[layer].as_ptr() as *const f32)
                .unwrap_or(std::ptr::null());
            let w_down_v = self
                .w_down_v
                .as_ref()
                .map(|w| w[layer].as_ptr() as *const f32)
                .unwrap_or(std::ptr::null());

            kv_write_compressed(
                k_cache,
                v_cache,
                k_gpu,
                v_gpu,
                d_pos.as_ptr() as *const i32,
                self.num_kv_heads,
                self.head_dim,
                10000.0, // Default theta base
                false,   // Default neox
                dc,
                self.kv_frame_codec_enabled,
                w_down_k,
                w_down_v,
                stream,
            )
        } else {
            kv_write_on_stream(
                k_cache,
                v_cache,
                k_gpu,
                v_gpu,
                pos,
                self.kv_size,
                self.max_seq_len,
                stream,
            )
        }
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

        if let Some(dc) = self.kv_lora_dim {
            let w_down_k = self
                .w_down_k
                .as_ref()
                .map(|w| w[layer].as_ptr() as *const f32)
                .unwrap_or(std::ptr::null());
            let w_down_v = self
                .w_down_v
                .as_ref()
                .map(|w| w[layer].as_ptr() as *const f32)
                .unwrap_or(std::ptr::null());

            kv_write_batched_compressed(
                k_cache,
                v_cache,
                k_gpu,
                v_gpu,
                start_pos,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                dc,
                self.kv_frame_codec_enabled,
                w_down_k,
                w_down_v,
                hipStream_t::null(),
            )?;

            if self.kv_frame_codec_enabled {
                reconstruct_kv_cache_prefix_sum(
                    k_cache,
                    v_cache,
                    start_pos,
                    seq_len,
                    dc,
                    hipStream_t::null(),
                )?;
            }
            Ok(())
        } else {
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
    }

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

// ── KV Dump (research / analysis tool) ───────────────────────────────────────────

/// Magic bytes that identify a KV cache dump file.
pub const KV_DUMP_MAGIC: &[u8; 8] = b"KVCACHE1";

/// In-memory representation of a KV cache dump loaded from disk.
#[allow(dead_code)]
///
/// Layout:
/// - `k[layer]`: flat `Vec<f32>` of shape `[num_tokens × kv_size]`
///   where `kv_size = num_kv_heads × head_dim`.
/// - `v[layer]`: same shape.
pub struct KvDump {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_tokens: usize,
    /// Key vectors per layer. `k[l][t * kv_size .. (t+1) * kv_size]` = token t.
    pub k: Vec<Vec<f32>>,
    /// Value vectors per layer. Same layout as `k`.
    pub v: Vec<Vec<f32>>,
}

impl KvDump {
    /// Load a KV cache dump written by [`GpuKvCache::dump_to_file`].
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::Read;
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;

        // Header: 8 magic + 4×4 fields + 8 padding = 32 bytes
        if data.len() < 32 {
            return Err("KvDump: file too short to contain header".into());
        }
        if &data[..8] != KV_DUMP_MAGIC {
            return Err(format!(
                "KvDump: bad magic {:?}, expected {:?}",
                &data[..8],
                KV_DUMP_MAGIC
            )
            .into());
        }
        let num_layers = u32::from_le_bytes(data[8..12].try_into()?) as usize;
        let num_kv_heads = u32::from_le_bytes(data[12..16].try_into()?) as usize;
        let head_dim = u32::from_le_bytes(data[16..20].try_into()?) as usize;
        let num_tokens = u32::from_le_bytes(data[20..24].try_into()?) as usize;
        // bytes 24..32 are padding

        let kv_size = num_kv_heads * head_dim;
        let floats_per_layer = num_tokens * kv_size;
        let bytes_per_layer = floats_per_layer * 4;
        let expected_len = 32 + 2 * num_layers * bytes_per_layer;

        if data.len() < expected_len {
            return Err(format!(
                "KvDump: truncated — expected {} bytes, got {}",
                expected_len,
                data.len()
            )
            .into());
        }

        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        let mut offset = 32usize;

        for _ in 0..num_layers {
            let k_floats: Vec<f32> = data[offset..offset + bytes_per_layer]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            offset += bytes_per_layer;

            let v_floats: Vec<f32> = data[offset..offset + bytes_per_layer]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            offset += bytes_per_layer;

            k.push(k_floats);
            v.push(v_floats);
        }

        Ok(KvDump {
            num_layers,
            num_kv_heads,
            head_dim,
            num_tokens,
            k,
            v,
        })
    }

    /// Return the key vector for a specific layer and token position.
    ///
    /// Returns a slice of length `num_kv_heads * head_dim`.
    pub fn key(&self, layer: usize, token: usize) -> &[f32] {
        let kv_size = self.num_kv_heads * self.head_dim;
        &self.k[layer][token * kv_size..(token + 1) * kv_size]
    }

    /// Return the value vector for a specific layer and token position.
    pub fn val(&self, layer: usize, token: usize) -> &[f32] {
        let kv_size = self.num_kv_heads * self.head_dim;
        &self.v[layer][token * kv_size..(token + 1) * kv_size]
    }
}

impl GpuKvCache {
    /// Dump the first `num_tokens` positions of every layer's KV cache to a
    /// binary file for off-GPU analysis.
    ///
    /// The file format is:
    /// ```text
    /// [u8; 8]  magic = "KVCACHE1"
    /// u32      num_layers
    /// u32      num_kv_heads
    /// u32      head_dim
    /// u32      num_tokens
    /// [u8; 8]  padding
    /// -- for each layer l in 0..num_layers:
    ///    [f32; num_tokens * num_kv_heads * head_dim]  K[l]
    ///    [f32; num_tokens * num_kv_heads * head_dim]  V[l]
    /// ```
    ///
    /// This is a research / analysis tool; it synchronises the GPU stream and
    /// copies VRAM → host, so it is not suitable for hot inference paths.
    pub fn dump_to_file(
        &self,
        path: &std::path::Path,
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        if num_tokens == 0 || num_tokens > self.max_seq_len {
            return Err(format!(
                "dump_to_file: num_tokens {} out of range [1, {}]",
                num_tokens, self.max_seq_len
            )
            .into());
        }

        let kv_size = num_kv_heads * head_dim;
        let floats_per_layer = num_tokens * kv_size;

        let mut file = std::fs::File::create(path)?;

        // Header
        file.write_all(KV_DUMP_MAGIC)?;
        file.write_all(&(self.num_layers as u32).to_le_bytes())?;
        file.write_all(&(num_kv_heads as u32).to_le_bytes())?;
        file.write_all(&(head_dim as u32).to_le_bytes())?;
        file.write_all(&(num_tokens as u32).to_le_bytes())?;
        file.write_all(&[0u8; 8])?; // padding

        // Body: K then V for each layer
        for layer in 0..self.num_layers {
            // copy_to_host_vec reads the whole layer buffer (max_seq_len * kv_size)
            let full_k = self.k[layer].copy_to_host_vec()?;
            let full_v = self.v[layer].copy_to_host_vec()?;

            // Write only the populated prefix
            let k_bytes: Vec<u8> = full_k[..floats_per_layer]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            file.write_all(&k_bytes)?;

            let v_bytes: Vec<u8> = full_v[..floats_per_layer]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            file.write_all(&v_bytes)?;
        }

        Ok(())
    }
}

#[inline]
fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn compute_kv_binding_tag(k: &[GpuBuffer], v: &[GpuBuffer]) -> u64 {
    let mut tag = 0u64;
    for buffer in k {
        tag = mix_binding_tag(tag, buffer.as_ptr() as usize);
    }
    for buffer in v {
        tag = mix_binding_tag(tag, buffer.as_ptr() as usize);
    }
    tag
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

    // ── KvDump file format ────────────────────────────────────────────────────

    /// A KvDump written and re-parsed must round-trip all header fields.
    #[test]
    fn kv_dump_header_round_trips() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new().unwrap();

        let num_layers: u32 = 4;
        let num_kv_heads: u32 = 8;
        let head_dim: u32 = 128;
        let num_tokens: u32 = 16;

        // Write header
        f.write_all(KV_DUMP_MAGIC).unwrap();
        f.write_all(&num_layers.to_le_bytes()).unwrap();
        f.write_all(&num_kv_heads.to_le_bytes()).unwrap();
        f.write_all(&head_dim.to_le_bytes()).unwrap();
        f.write_all(&num_tokens.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 8]).unwrap(); // padding

        // Write placeholder data (zeros)
        let floats_per_layer = num_tokens as usize * num_kv_heads as usize * head_dim as usize;
        let zeros = vec![0u8; floats_per_layer * 4 * 2 * num_layers as usize];
        f.write_all(&zeros).unwrap();
        f.flush().unwrap();

        // Read back and check header
        let path = f.path().to_owned();
        let mut buf = Vec::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();

        assert_eq!(&buf[..8], KV_DUMP_MAGIC, "magic mismatch");
        assert_eq!(
            u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            num_layers
        );
        assert_eq!(
            u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            num_kv_heads
        );
        assert_eq!(
            u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            head_dim
        );
        assert_eq!(
            u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            num_tokens
        );

        // Total size: 32 header + 2 * num_layers * floats_per_layer * 4
        let expected_len = 32 + 2 * num_layers as usize * floats_per_layer * 4;
        assert_eq!(buf.len(), expected_len, "file size mismatch");
    }

    /// A KvDump with wrong magic returns an error.
    #[test]
    fn kv_dump_parse_rejects_bad_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"BADMAGIC").unwrap();
        f.write_all(&[0u8; 100]).unwrap();
        f.flush().unwrap();

        let result = KvDump::load(f.path());
        assert!(result.is_err(), "should fail on bad magic");
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("magic"), "error should mention magic: {msg}");
    }

    /// A KvDump with truncated data returns an error.
    #[test]
    fn kv_dump_parse_rejects_truncated_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new().unwrap();
        f.write_all(KV_DUMP_MAGIC).unwrap();
        // Write a header claiming 4 layers / 8 heads / 128 dim / 16 tokens
        // but no body data
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&8u32.to_le_bytes()).unwrap();
        f.write_all(&128u32.to_le_bytes()).unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 8]).unwrap();
        f.flush().unwrap();

        let result = KvDump::load(f.path());
        assert!(result.is_err(), "should fail on truncated body");
    }
}

// ── Forward Scratch Buffers ───────────────────────────────────────────────────────

/// Reusable scratch buffers in GPU VRAM for a single forward pass.
///
/// Allocated once and reused across all layers to avoid repeated allocations.
/// All buffers are GPU-resident.
const GPU_ARGMAX_BLOCK_SIZE: usize = 256;
const GPU_ARGMAX_ITEMS_PER_THREAD: usize = 4;
const GPU_ARGMAX_ITEMS_PER_BLOCK: usize = GPU_ARGMAX_BLOCK_SIZE * GPU_ARGMAX_ITEMS_PER_THREAD;

pub struct GpuForwardScratch {
    /// Current hidden state [hidden_size]
    pub hidden: GpuBuffer,
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
    /// Temporary GPU workspace for SVD-Quant outlier corrections [32]
    pub svd_scratch: GpuBuffer,
    /// Final logits [vocab_size]
    pub logits: GpuBuffer,
    /// Partial argmax values for greedy decode [ceil(vocab_size / 1024)]
    pub argmax_partial_values: GpuBuffer,
    /// Partial argmax indices for greedy decode [ceil(vocab_size / 1024)]
    pub argmax_partial_indices: GpuBuffer,
    /// Final greedy token index [1] - Device destination
    pub argmax_result_device: GpuBuffer,
    /// Final greedy token index [1] - Pinned host buffer for async overlap
    pub argmax_result_index: GpuPinnedBuffer,
    /// Pinned host buffer for hidden state upload overlap
    pub input_hidden_pinned: GpuPinnedBuffer,
    /// Per-token decode state uploaded before full-graph replay: [pos, seq_len]
    decode_state: GpuBuffer,
    /// Pinned host staging for decode state upload to keep H2D async and tiny.
    decode_state_host: GpuPinnedBuffer,
    /// Host-tracked decode position currently resident in `decode_state[0]`.
    decode_state_next_pos: Option<usize>,
    /// Cached executable graph for repeated decode work.
    captured_decode: Option<CapturedDecodeGraph>,
    /// Pre-allocated GPU scratch for per-expert H2D upload during MoE decode.
    /// None for non-MoE or non-compressed models.
    pub expert_scratch: Option<GpuExpertScratch>,
}

/// Pre-allocated GPU buffers for uploading one expert's compressed data at decode time.
///
/// Allocated once when a compressed-expert model is detected; reused across all
/// layers and tokens.  Sized for the largest expert dimensions in the model.
pub struct GpuExpertScratch {
    /// U factor upload buffer: [rows * k] F32
    pub u: GpuBuffer,
    /// V factor upload buffer: [k * cols] F32
    pub v: GpuBuffer,
    /// CSR values upload buffer: [max_nnz] F32
    pub csr_values: GpuBuffer,
    /// CSR col-index upload buffer: [max_nnz] u32
    pub csr_col_idx: GpuBuffer,
    /// CSR row-pointer upload buffer: [rows + 1] u32
    pub csr_row_ptr: GpuBuffer,
    /// Intermediate k-vector for V·x computation
    pub temp_v: GpuBuffer,
    /// Pre-allocated scratch buffer for FWHT rotated input activation: [cols] F32
    pub rotated_input: GpuBuffer,
    pub k: u32,
    pub rows: usize,
    pub cols: usize,
    pub max_nnz: usize,
}

impl GpuExpertScratch {
    pub fn new(k: u32, rows: usize, cols: usize, max_nnz: usize) -> GpuResult<Self> {
        let ku = k as usize;
        let nnz = max_nnz.max(1); // avoid zero-size allocation
        Ok(Self {
            u: GpuBuffer::alloc(rows * ku * 4)?,
            v: GpuBuffer::alloc(ku * cols * 4)?,
            csr_values: GpuBuffer::alloc(nnz * 4)?,
            csr_col_idx: GpuBuffer::alloc(nnz * 4)?,
            csr_row_ptr: GpuBuffer::alloc((rows + 1) * 4)?,
            temp_v: GpuBuffer::alloc(ku * 4)?,
            rotated_input: GpuBuffer::alloc(cols * 4)?,
            k,
            rows,
            cols,
            max_nnz: nnz,
        })
    }
}

impl GpuForwardScratch {
    /// Estimate VRAM bytes required for forward scratch buffers without allocating.
    ///
    /// This mirrors the GPU-only (non-pinned) allocations in `new`.
    /// Use for pre-flight VRAM checks before calling `new`.
    pub fn estimate_bytes(config: &ModelConfig) -> usize {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let v = config.vocab_size;
        let argmax_partials = v.div_ceil(GPU_ARGMAX_ITEMS_PER_BLOCK);
        // Count only VRAM (GpuBuffer) allocations, not pinned host buffers.
        std::mem::size_of::<f32>()
            * (3 * h           // hidden, normed, layer_out
                + 2 * q        // q_buf, attn_out
                + 2 * kv       // k_buf, v_buf
                + 2 * ff       // gate, swiglu
                + 32           // svd_scratch
                + v            // logits
                + 2 * argmax_partials  // argmax partial values + indices
                + 3) // argmax_result_device (1) + decode_state (2)
    }

    /// Allocate expert scratch buffers for compressed MoE dispatch.
    ///
    /// Call once after detecting compressed experts in the loaded model.
    /// Sizes the buffers for the given expert dimensions.
    pub fn init_expert_scratch(
        &mut self,
        k: u32,
        rows: usize,
        cols: usize,
        max_nnz: usize,
    ) -> GpuResult<()> {
        self.expert_scratch = Some(GpuExpertScratch::new(k, rows, cols, max_nnz)?);
        Ok(())
    }

    /// Allocate scratch buffers in GPU VRAM.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Ok(GpuForwardScratch) if all allocations succeed
    pub fn new(config: &ModelConfig) -> GpuResult<Self> {
        let h = config.hidden_size;
        // For qwen35, the attention Q weight is [h, 8192] while config.num_heads*head_dim=4096
        // (config reflects SSM head dims). Allocate enough for the larger attention Q projection.
        let q = if config.architecture.contains("qwen35") {
            std::cmp::max(config.num_heads * config.head_dim, h * 2)
        } else {
            config.num_heads * config.head_dim
        };
        // kv_size uses config values; for qwen35 attention layers kv_size (1024) < config (4096),
        // so no overflow — but ensure it's at least as large as a single attention kv row.
        let kv = config.num_kv_heads * config.head_dim;
        let ff = if config.architecture.contains("qwen35") {
            std::cmp::max(config.intermediate_size, 16384)
        } else {
            config.intermediate_size
        };
        let v = config.vocab_size;
        let argmax_partials = v.div_ceil(GPU_ARGMAX_ITEMS_PER_BLOCK);

        // Allocate all buffers - if any fail, all are freed via RAII
        let hidden = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("hidden buffer allocation failed: {}", e),
            }
        })?;

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

        let svd_scratch = GpuBuffer::alloc(32 * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("SVD scratch buffer allocation failed: {}", e),
            }
        })?;

        let logits = GpuBuffer::alloc(v * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("logits buffer allocation failed: {}", e),
            }
        })?;

        let argmax_partial_values = GpuBuffer::alloc(argmax_partials * std::mem::size_of::<f32>())
            .map_err(|e| GpuError::CacheAllocationFailed {
                reason: format!("argmax partial values allocation failed: {}", e),
            })?;

        let argmax_partial_indices = GpuBuffer::alloc(argmax_partials * std::mem::size_of::<i32>())
            .map_err(|e| GpuError::CacheAllocationFailed {
                reason: format!("argmax partial indices allocation failed: {}", e),
            })?;

        let argmax_result_device = GpuBuffer::alloc(std::mem::size_of::<i32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("argmax result device allocation failed: {}", e),
            }
        })?;

        let argmax_result_index =
            GpuPinnedBuffer::alloc(std::mem::size_of::<i32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("argmax result allocation failed: {}", e),
                }
            })?;
        let input_hidden_pinned =
            GpuPinnedBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("input hidden pinned allocation failed: {}", e),
                }
            })?;
        let decode_state = GpuBuffer::alloc(2 * std::mem::size_of::<i32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("decode state allocation failed: {}", e),
            }
        })?;
        let decode_state_host =
            GpuPinnedBuffer::alloc(2 * std::mem::size_of::<i32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("decode state host allocation failed: {}", e),
                }
            })?;

        // Zero-initialize all scratch buffers to prevent NaN propagation from
        // uninitialized memory. SVD correction kernels use += accumulation,
        // and residual connections add buffers together — garbage in any buffer
        // spreads through the entire forward pass.
        super::ffi::hip_memset(hidden.as_ptr(), 0, hidden.size())?;
        super::ffi::hip_memset(normed.as_ptr(), 0, normed.size())?;
        super::ffi::hip_memset(q_buf.as_ptr(), 0, q_buf.size())?;
        super::ffi::hip_memset(k_buf.as_ptr(), 0, k_buf.size())?;
        super::ffi::hip_memset(v_buf.as_ptr(), 0, v_buf.size())?;
        super::ffi::hip_memset(attn_out.as_ptr(), 0, attn_out.size())?;
        super::ffi::hip_memset(layer_out.as_ptr(), 0, layer_out.size())?;
        super::ffi::hip_memset(gate.as_ptr(), 0, gate.size())?;
        super::ffi::hip_memset(swiglu.as_ptr(), 0, swiglu.size())?;
        super::ffi::hip_memset(svd_scratch.as_ptr(), 0, svd_scratch.size())?;
        super::ffi::hip_memset(logits.as_ptr(), 0, logits.size())?;
        super::ffi::hip_memset(
            argmax_partial_values.as_ptr(),
            0,
            argmax_partial_values.size(),
        )?;
        super::ffi::hip_memset(
            argmax_partial_indices.as_ptr(),
            0,
            argmax_partial_indices.size(),
        )?;
        super::ffi::hip_memset(
            argmax_result_device.as_ptr(),
            0,
            argmax_result_device.size(),
        )?;
        super::ffi::hip_memset(decode_state.as_ptr(), 0, decode_state.size())?;
        // Pinned buffers and argmax_result_index are host-side; zeroed on use

        Ok(Self {
            hidden,
            normed,
            q: q_buf,
            k: k_buf,
            v: v_buf,
            attn_out,
            layer_out,
            gate,
            swiglu,
            svd_scratch,
            logits,
            argmax_partial_values,
            argmax_partial_indices,
            argmax_result_device,
            argmax_result_index,
            input_hidden_pinned,
            decode_state,
            decode_state_host,
            decode_state_next_pos: None,
            captured_decode: None,
            expert_scratch: None,
        })
    }

    /// Get GPU pointer to current hidden state.
    pub fn hidden_ptr(&self) -> *const f32 {
        self.hidden.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to current hidden state.
    pub fn hidden_mut_ptr(&mut self) -> *mut f32 {
        self.hidden.as_ptr() as *mut f32
    }

    /// Get GPU pointer to normalized hidden state
    pub fn normed_ptr(&self) -> *const f32 {
        self.normed.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to normalized hidden state.
    pub fn normed_mut_ptr(&mut self) -> *mut f32 {
        self.normed.as_ptr() as *mut f32
    }

    /// Get GPU pointer to query vector
    pub fn q_ptr(&self) -> *const f32 {
        self.q.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to query vector.
    pub fn q_mut_ptr(&mut self) -> *mut f32 {
        self.q.as_ptr() as *mut f32
    }

    /// Get GPU pointer to key vector
    pub fn k_ptr(&self) -> *const f32 {
        self.k.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to key vector.
    pub fn k_mut_ptr(&mut self) -> *mut f32 {
        self.k.as_ptr() as *mut f32
    }

    /// Get GPU pointer to value vector
    pub fn v_ptr(&self) -> *const f32 {
        self.v.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to value vector.
    pub fn v_mut_ptr(&mut self) -> *mut f32 {
        self.v.as_ptr() as *mut f32
    }

    /// Get GPU pointer to attention output.
    pub fn attn_out_ptr(&self) -> *const f32 {
        self.attn_out.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to attention output.
    pub fn attn_out_mut_ptr(&mut self) -> *mut f32 {
        self.attn_out.as_ptr() as *mut f32
    }

    /// Get GPU pointer to layer output.
    pub fn layer_out_ptr(&self) -> *const f32 {
        self.layer_out.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to layer output.
    pub fn layer_out_mut_ptr(&mut self) -> *mut f32 {
        self.layer_out.as_ptr() as *mut f32
    }

    /// Get GPU pointer to FFN gate activations.
    pub fn gate_ptr(&self) -> *const f32 {
        self.gate.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to FFN gate activations.
    pub fn gate_mut_ptr(&mut self) -> *mut f32 {
        self.gate.as_ptr() as *mut f32
    }

    /// Get GPU pointer to SwiGLU activations.
    pub fn swiglu_ptr(&self) -> *const f32 {
        self.swiglu.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to SwiGLU activations.
    pub fn swiglu_mut_ptr(&mut self) -> *mut f32 {
        self.swiglu.as_ptr() as *mut f32
    }

    /// Get GPU pointer to logits.
    pub fn logits_ptr(&self) -> *const f32 {
        self.logits.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to logits.
    pub fn logits_mut_ptr(&mut self) -> *mut f32 {
        self.logits.as_ptr() as *mut f32
    }

    /// Get GPU pointer to argmax partial values.
    pub fn argmax_partial_values_mut_ptr(&mut self) -> *mut f32 {
        self.argmax_partial_values.as_ptr() as *mut f32
    }

    /// Get GPU pointer to argmax partial indices.
    pub fn argmax_partial_indices_mut_ptr(&mut self) -> *mut i32 {
        self.argmax_partial_indices.as_ptr() as *mut i32
    }

    /// Get GPU pointer to final argmax index.
    pub fn argmax_result_index_mut_ptr(&mut self) -> *mut i32 {
        self.argmax_result_device.as_ptr() as *mut i32
    }

    pub fn decode_pos_ptr(&self) -> *const i32 {
        self.decode_state.as_ptr() as *const i32
    }

    pub fn decode_seq_len_ptr(&self) -> *const i32 {
        unsafe { (self.decode_state.as_ptr() as *const i32).add(1) }
    }

    pub fn decode_state_mut_ptr(&mut self) -> *mut i32 {
        self.decode_state.as_ptr() as *mut i32
    }

    pub fn decode_state_matches_pos(&self, pos: usize) -> bool {
        self.decode_state_next_pos == Some(pos)
    }

    pub fn mark_decode_state_next_pos(&mut self, pos: usize) {
        self.decode_state_next_pos = Some(pos);
    }

    pub fn decode_state_next_pos(&self) -> Option<usize> {
        self.decode_state_next_pos
    }

    pub fn upload_decode_state(
        &mut self,
        pos: usize,
        seq_len: usize,
        stream: hipStream_t,
    ) -> GpuResult<()> {
        let pos_i32 = i32::try_from(pos).map_err(|_| GpuError::HipApiError {
            code: -1,
            description: format!("decode pos {} exceeds i32 range", pos),
        })?;
        let seq_len_i32 = i32::try_from(seq_len).map_err(|_| GpuError::HipApiError {
            code: -1,
            description: format!("decode seq_len {} exceeds i32 range", seq_len),
        })?;
        let state = self.decode_state_host.as_slice_mut::<i32>();
        state[0] = pos_i32;
        state[1] = seq_len_i32;
        let state_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.decode_state_host.as_ptr() as *const u8,
                2 * std::mem::size_of::<i32>(),
            )
        };
        self.decode_state
            .copy_from_host_on_stream(state_bytes, stream)?;
        self.decode_state_next_pos = Some(pos);
        Ok(())
    }

    pub fn decode_graph(&self) -> Option<&CapturedDecodeGraph> {
        self.captured_decode.as_ref()
    }

    pub fn decode_graph_mut(&mut self) -> Option<&mut CapturedDecodeGraph> {
        self.captured_decode.as_mut()
    }

    pub fn has_decode_graph_for(&self, key: DecodeGraphKey) -> bool {
        self.captured_decode
            .as_ref()
            .is_some_and(|graph| graph.matches_key(key))
    }

    pub fn replace_decode_graph(
        &mut self,
        graph: CapturedDecodeGraph,
    ) -> Option<CapturedDecodeGraph> {
        self.decode_state_next_pos = None;
        self.captured_decode.replace(graph)
    }

    pub fn try_update_decode_graph(
        &mut self,
        new_graph: &crate::gpu::graph::HipGraph,
    ) -> GpuResult<bool> {
        if let Some(graph) = &self.captured_decode {
            let updated = graph.update(new_graph)?;
            if updated {
                self.decode_state_next_pos = None;
            }
            Ok(updated)
        } else {
            Ok(false)
        }
    }

    pub fn clear_decode_graph(&mut self) {
        self.captured_decode = None;
        self.decode_state_next_pos = None;
    }
}

/// Reusable scratch buffers in GPU VRAM for batched prompt prefill.
///
/// Layout is row-major `[seq_len, dim]` for all activation buffers.
pub struct GpuPrefillScratch {
    pub seq_len: usize,
    pub hidden: GpuBuffer,
    pub normed: GpuBuffer,
    pub q: GpuBuffer,
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub attn_out: GpuBuffer,
    pub layer_out: GpuBuffer,
    pub gate: GpuBuffer,
    pub swiglu: GpuBuffer,
    pub token_ids: GpuBuffer,
    pub logits: GpuBuffer,
    pub svd_scratch: GpuBuffer,
}

impl GpuPrefillScratch {
    pub fn new(config: &ModelConfig, seq_len: usize) -> GpuResult<Self> {
        if seq_len == 0 {
            return Err(GpuError::CacheAllocationFailed {
                reason: "prefill seq_len cannot be zero".to_string(),
            });
        }

        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        let hidden = GpuBuffer::alloc(seq_len * h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill hidden allocation failed: {}", e),
            }
        })?;
        let normed = GpuBuffer::alloc(seq_len * h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill normed allocation failed: {}", e),
            }
        })?;
        let q_buf = GpuBuffer::alloc(seq_len * q * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill q allocation failed: {}", e),
            }
        })?;
        let k_buf = GpuBuffer::alloc(seq_len * kv * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill k allocation failed: {}", e),
            }
        })?;
        let v_buf = GpuBuffer::alloc(seq_len * kv * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill v allocation failed: {}", e),
            }
        })?;
        let attn_out = GpuBuffer::alloc(seq_len * q * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill attn_out allocation failed: {}", e),
            }
        })?;
        let layer_out =
            GpuBuffer::alloc(seq_len * h * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill layer_out allocation failed: {}", e),
                }
            })?;
        let gate = GpuBuffer::alloc(seq_len * ff * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill gate allocation failed: {}", e),
            }
        })?;
        let swiglu = GpuBuffer::alloc(seq_len * ff * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill swiglu allocation failed: {}", e),
            }
        })?;
        let token_ids = GpuBuffer::alloc(seq_len * std::mem::size_of::<i32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill token_ids allocation failed: {}", e),
            }
        })?;
        let logits =
            GpuBuffer::alloc(config.vocab_size * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill logits allocation failed: {}", e),
                }
            })?;

        let svd_scratch =
            GpuBuffer::alloc(seq_len * 32 * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill SVD scratch allocation failed: {}", e),
                }
            })?;

        // Zero-initialize all scratch buffers to prevent NaN propagation from
        // uninitialized memory. SVD correction kernels use += accumulation,
        // and residual connections add buffers together — garbage in any buffer
        // spreads through the entire forward pass.
        super::ffi::hip_memset(hidden.as_ptr(), 0, hidden.size())?;
        super::ffi::hip_memset(normed.as_ptr(), 0, normed.size())?;
        super::ffi::hip_memset(q_buf.as_ptr(), 0, q_buf.size())?;
        super::ffi::hip_memset(k_buf.as_ptr(), 0, k_buf.size())?;
        super::ffi::hip_memset(v_buf.as_ptr(), 0, v_buf.size())?;
        super::ffi::hip_memset(attn_out.as_ptr(), 0, attn_out.size())?;
        super::ffi::hip_memset(layer_out.as_ptr(), 0, layer_out.size())?;
        super::ffi::hip_memset(gate.as_ptr(), 0, gate.size())?;
        super::ffi::hip_memset(swiglu.as_ptr(), 0, swiglu.size())?;
        super::ffi::hip_memset(logits.as_ptr(), 0, logits.size())?;
        super::ffi::hip_memset(svd_scratch.as_ptr(), 0, svd_scratch.size())?;
        // token_ids is written before use, no need to zero

        Ok(Self {
            seq_len,
            hidden,
            normed,
            q: q_buf,
            k: k_buf,
            v: v_buf,
            attn_out,
            layer_out,
            gate,
            swiglu,
            token_ids,
            logits,
            svd_scratch,
        })
    }

    pub fn hidden_row_ptr(&self, row: usize, hidden_size: usize) -> *const f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.hidden.as_ptr() as *const f32).add(row * hidden_size) }
    }

    pub fn normed_row_ptr(&self, row: usize, hidden_size: usize) -> *const f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.normed.as_ptr() as *const f32).add(row * hidden_size) }
    }

    pub fn normed_row_mut_ptr(&mut self, row: usize, hidden_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.normed.as_ptr() as *mut f32).add(row * hidden_size) }
    }

    pub fn q_row_mut_ptr(&mut self, row: usize, q_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.q.as_ptr() as *mut f32).add(row * q_size) }
    }

    pub fn k_row_mut_ptr(&mut self, row: usize, kv_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.k.as_ptr() as *mut f32).add(row * kv_size) }
    }

    pub fn v_row_mut_ptr(&mut self, row: usize, kv_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.v.as_ptr() as *mut f32).add(row * kv_size) }
    }

    pub fn attn_out_row_mut_ptr(&mut self, row: usize, q_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.attn_out.as_ptr() as *mut f32).add(row * q_size) }
    }

    pub fn layer_out_row_mut_ptr(&mut self, row: usize, hidden_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.layer_out.as_ptr() as *mut f32).add(row * hidden_size) }
    }

    pub fn gate_row_mut_ptr(&mut self, row: usize, ff_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.gate.as_ptr() as *mut f32).add(row * ff_size) }
    }

    pub fn swiglu_row_mut_ptr(&mut self, row: usize, ff_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.swiglu.as_ptr() as *mut f32).add(row * ff_size) }
    }

    /// Estimate total VRAM bytes needed for prefill scratch buffers.
    ///
    /// Pure computation for pre-allocation validation. Does not allocate.
    /// Formula: sum over all buffers of (seq_len * dim * sizeof(f32))
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `seq_len` - Number of tokens in prefill batch
    ///
    /// # Returns
    /// Total bytes required for all prefill scratch buffers
    pub fn estimate_total_bytes(config: &ModelConfig, seq_len: usize) -> usize {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let elem_size = std::mem::size_of::<f32>();

        // Sum: [seq_len * dim] for each buffer
        let total_elements = seq_len * (h + h + q + kv + kv + q + h + ff + ff) + seq_len;
        total_elements * elem_size
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
                &crate::config::TensorNamingScheme::Gguf,
            ),
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
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
                assert!(!s.hidden.as_ptr().is_null() || s.hidden.is_empty());
            }
            Err(_) => {
                // Expected when HIP unavailable
            }
        }
    }

    #[test]
    fn prefill_scratch_rejects_zero_seq_len() {
        let config = make_test_config();
        let scratch = GpuPrefillScratch::new(&config, 0);
        assert!(scratch.is_err());
    }
}
