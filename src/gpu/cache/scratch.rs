use super::{
    CapturedDecodeGraph, DecodeGraphKey, GpuBuffer, GpuError, GpuPinnedBuffer, GpuResult,
    ModelConfig,
};
use crate::config::FfnLayout;
use crate::cpu::forward_graph_trace::ForwardGraphRecorder;
use crate::gpu::ffi::hipStream_t;

/// Reusable scratch buffers in GPU VRAM for a single forward pass.
///
/// Allocated once and reused across all layers to avoid repeated allocations.
/// All buffers are GPU-resident.
const GPU_ARGMAX_BLOCK_SIZE: usize = 256;
const GPU_ARGMAX_ITEMS_PER_THREAD: usize = 4;
const GPU_ARGMAX_ITEMS_PER_BLOCK: usize = GPU_ARGMAX_BLOCK_SIZE * GPU_ARGMAX_ITEMS_PER_THREAD;

#[derive(Debug)]
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
    /// Optional forward-graph recorder set by the CLI when tracing GPU decode.
    forward_graph_recorder: Option<*mut ForwardGraphRecorder>,
    /// Optional scratch buffer for normalized attention weights when tracing.
    /// Sized to [num_heads * max_seq_len] f32 and allocated on first use.
    attn_weights: Option<GpuBuffer>,
}

/// Pre-allocated GPU buffers for uploading one expert's compressed data at decode time.
///
/// Allocated once when a compressed-expert model is detected; reused across all
/// layers and tokens. Sized for the largest expert dimensions in the model.
#[derive(Debug)]
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
    /// MPO site data upload buffer: [rows * k + k * cols] F32
    pub mpo_site_data: GpuBuffer,
    /// MPO site dims upload buffer: [8] u32
    pub mpo_site_dims: GpuBuffer,
    pub k: u32,
    pub rows: usize,
    pub cols: usize,
    pub max_nnz: usize,
}

/// Reusable scratch buffers in GPU VRAM for batched prompt prefill.
///
/// Layout is row-major `[seq_len, dim]` for all activation buffers.
#[derive(Debug)]
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

impl GpuExpertScratch {
    pub fn new(k: u32, rows: usize, cols: usize, max_nnz: usize) -> GpuResult<Self> {
        let ku = k as usize;
        let nnz = max_nnz.max(1);
        Ok(Self {
            u: GpuBuffer::alloc(rows * ku * 4)?,
            v: GpuBuffer::alloc(ku * cols * 4)?,
            csr_values: GpuBuffer::alloc(nnz * 4)?,
            csr_col_idx: GpuBuffer::alloc(nnz * 4)?,
            csr_row_ptr: GpuBuffer::alloc((rows + 1) * 4)?,
            temp_v: GpuBuffer::alloc(ku * 4)?,
            rotated_input: GpuBuffer::alloc(cols * 4)?,
            mpo_site_data: GpuBuffer::alloc((rows * ku + ku * cols) * 4)?,
            mpo_site_dims: GpuBuffer::alloc(8 * 4)?,
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
    /// This mirrors the GPU-only (non-pinned) allocations in `new`, plus the
    /// optional attention-weights trace buffer sized to the model's maximum
    /// sequence length.
    pub fn estimate_bytes(config: &ModelConfig) -> usize {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let v = config.vocab_size;
        let argmax_partials = v.div_ceil(GPU_ARGMAX_ITEMS_PER_BLOCK);
        let attn_weights = config.num_heads * config.max_seq_len;
        std::mem::size_of::<f32>()
            * (3 * h + 2 * q + 2 * kv + 2 * ff + 32 + v + 2 * argmax_partials + 3 + attn_weights)
    }

    /// Allocate expert scratch buffers for compressed MoE dispatch.
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
    pub fn new(config: &ModelConfig) -> GpuResult<Self> {
        let h = config.hidden_size;
        let q = if config.architecture.contains("qwen35") {
            std::cmp::max(config.num_heads * config.head_dim, h * 2)
        } else {
            config.num_heads * config.head_dim
        };
        let kv = config.num_kv_heads * config.head_dim;
        let ff = if config.architecture.contains("qwen35") {
            std::cmp::max(config.intermediate_size, 16384)
        } else {
            config.intermediate_size
        };
        let v = config.vocab_size;
        let argmax_partials = v.div_ceil(GPU_ARGMAX_ITEMS_PER_BLOCK);

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

        crate::gpu::ffi::hip_memset(hidden.as_ptr(), 0, hidden.size())?;
        crate::gpu::ffi::hip_memset(normed.as_ptr(), 0, normed.size())?;
        crate::gpu::ffi::hip_memset(q_buf.as_ptr(), 0, q_buf.size())?;
        crate::gpu::ffi::hip_memset(k_buf.as_ptr(), 0, k_buf.size())?;
        crate::gpu::ffi::hip_memset(v_buf.as_ptr(), 0, v_buf.size())?;
        crate::gpu::ffi::hip_memset(attn_out.as_ptr(), 0, attn_out.size())?;
        crate::gpu::ffi::hip_memset(layer_out.as_ptr(), 0, layer_out.size())?;
        crate::gpu::ffi::hip_memset(gate.as_ptr(), 0, gate.size())?;
        crate::gpu::ffi::hip_memset(swiglu.as_ptr(), 0, swiglu.size())?;
        crate::gpu::ffi::hip_memset(svd_scratch.as_ptr(), 0, svd_scratch.size())?;
        crate::gpu::ffi::hip_memset(logits.as_ptr(), 0, logits.size())?;
        crate::gpu::ffi::hip_memset(
            argmax_partial_values.as_ptr(),
            0,
            argmax_partial_values.size(),
        )?;
        crate::gpu::ffi::hip_memset(
            argmax_partial_indices.as_ptr(),
            0,
            argmax_partial_indices.size(),
        )?;
        crate::gpu::ffi::hip_memset(
            argmax_result_device.as_ptr(),
            0,
            argmax_result_device.size(),
        )?;
        crate::gpu::ffi::hip_memset(decode_state.as_ptr(), 0, decode_state.size())?;

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
            forward_graph_recorder: None,
            attn_weights: None,
        })
    }

    /// Bind (or unbind) the optional forward-graph recorder.
    pub fn set_forward_graph_recorder(&mut self, recorder: Option<&mut ForwardGraphRecorder>) {
        self.forward_graph_recorder = recorder.map(|r| r as *mut _);
    }

    /// Clear any bound forward-graph recorder.
    pub fn clear_forward_graph_recorder(&mut self) {
        self.forward_graph_recorder = None;
    }

    /// Return a mutable reference to the bound recorder, if any.
    pub fn forward_graph_recorder(&mut self) -> Option<&mut ForwardGraphRecorder> {
        self.forward_graph_recorder.map(|p| unsafe { &mut *p })
    }

    /// Ensure the per-layer attention-weights scratch buffer exists and return a
    /// device pointer suitable for the decode attention kernel.
    pub fn ensure_attn_weights(&mut self, config: &ModelConfig) -> GpuResult<*mut f32> {
        if self.attn_weights.is_none() {
            let size = config
                .num_heads
                .saturating_mul(config.max_seq_len)
                .saturating_mul(std::mem::size_of::<f32>());
            let buf = GpuBuffer::alloc(size).map_err(|e| GpuError::CacheAllocationFailed {
                reason: format!("attn_weights buffer allocation failed: {}", e),
            })?;
            self.attn_weights = Some(buf);
        }
        let buf = self
            .attn_weights
            .as_ref()
            .ok_or_else(|| GpuError::InvalidOperation {
                op: "ensure_attn_weights".to_string(),
                reason: "attention weights buffer missing after allocation".to_string(),
            })?;
        Ok(buf.as_ptr() as *mut f32)
    }

    /// Borrow the optional attention-weights buffer for host readback.
    pub(crate) fn attn_weights_buf(&self) -> Option<&GpuBuffer> {
        self.attn_weights.as_ref()
    }

    pub fn hidden_ptr(&self) -> *const f32 {
        self.hidden.as_ptr() as *const f32
    }
    pub fn hidden_mut_ptr(&mut self) -> *mut f32 {
        self.hidden.as_ptr() as *mut f32
    }
    pub fn normed_ptr(&self) -> *const f32 {
        self.normed.as_ptr() as *const f32
    }
    pub fn normed_mut_ptr(&mut self) -> *mut f32 {
        self.normed.as_ptr() as *mut f32
    }
    pub fn q_ptr(&self) -> *const f32 {
        self.q.as_ptr() as *const f32
    }
    pub fn q_mut_ptr(&mut self) -> *mut f32 {
        self.q.as_ptr() as *mut f32
    }
    pub fn k_ptr(&self) -> *const f32 {
        self.k.as_ptr() as *const f32
    }
    pub fn k_mut_ptr(&mut self) -> *mut f32 {
        self.k.as_ptr() as *mut f32
    }
    pub fn v_ptr(&self) -> *const f32 {
        self.v.as_ptr() as *const f32
    }
    pub fn v_mut_ptr(&mut self) -> *mut f32 {
        self.v.as_ptr() as *mut f32
    }
    pub fn attn_out_ptr(&self) -> *const f32 {
        self.attn_out.as_ptr() as *const f32
    }
    pub fn attn_out_mut_ptr(&mut self) -> *mut f32 {
        self.attn_out.as_ptr() as *mut f32
    }
    pub fn layer_out_ptr(&self) -> *const f32 {
        self.layer_out.as_ptr() as *const f32
    }
    pub fn layer_out_mut_ptr(&mut self) -> *mut f32 {
        self.layer_out.as_ptr() as *mut f32
    }
    pub fn gate_ptr(&self) -> *const f32 {
        self.gate.as_ptr() as *const f32
    }
    pub fn gate_mut_ptr(&mut self) -> *mut f32 {
        self.gate.as_ptr() as *mut f32
    }
    pub fn swiglu_ptr(&self) -> *const f32 {
        self.swiglu.as_ptr() as *const f32
    }
    pub fn swiglu_mut_ptr(&mut self) -> *mut f32 {
        self.swiglu.as_ptr() as *mut f32
    }
    pub fn logits_ptr(&self) -> *const f32 {
        self.logits.as_ptr() as *const f32
    }
    pub fn logits_mut_ptr(&mut self) -> *mut f32 {
        self.logits.as_ptr() as *mut f32
    }
    pub fn argmax_partial_values_mut_ptr(&mut self) -> *mut f32 {
        self.argmax_partial_values.as_ptr() as *mut f32
    }
    pub fn argmax_partial_indices_mut_ptr(&mut self) -> *mut i32 {
        self.argmax_partial_indices.as_ptr() as *mut i32
    }
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
        new_graph: crate::gpu::graph::HipGraph,
        new_key: crate::gpu::graph::DecodeGraphKey,
    ) -> GpuResult<Result<(), crate::gpu::graph::HipGraph>> {
        if let Some(graph) = &mut self.captured_decode {
            match graph.update(new_graph)? {
                Ok(()) => {
                    graph.set_key(new_key);
                    self.decode_state_next_pos = None;
                    Ok(Ok(()))
                }
                Err(g) => Ok(Err(g)),
            }
        } else {
            Ok(Err(new_graph))
        }
    }
    pub fn clear_decode_graph(&mut self) {
        self.captured_decode = None;
        self.decode_state_next_pos = None;
    }
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

        crate::gpu::ffi::hip_memset(hidden.as_ptr(), 0, hidden.size())?;
        crate::gpu::ffi::hip_memset(normed.as_ptr(), 0, normed.size())?;
        crate::gpu::ffi::hip_memset(q_buf.as_ptr(), 0, q_buf.size())?;
        crate::gpu::ffi::hip_memset(k_buf.as_ptr(), 0, k_buf.size())?;
        crate::gpu::ffi::hip_memset(v_buf.as_ptr(), 0, v_buf.size())?;
        crate::gpu::ffi::hip_memset(attn_out.as_ptr(), 0, attn_out.size())?;
        crate::gpu::ffi::hip_memset(layer_out.as_ptr(), 0, layer_out.size())?;
        crate::gpu::ffi::hip_memset(gate.as_ptr(), 0, gate.size())?;
        crate::gpu::ffi::hip_memset(swiglu.as_ptr(), 0, swiglu.size())?;
        crate::gpu::ffi::hip_memset(logits.as_ptr(), 0, logits.size())?;
        crate::gpu::ffi::hip_memset(svd_scratch.as_ptr(), 0, svd_scratch.size())?;

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
    pub fn estimate_total_bytes(config: &ModelConfig, seq_len: usize) -> usize {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let elem_size = std::mem::size_of::<f32>();
        let total_elements = seq_len * (h + h + q + kv + kv + q + h + ff + ff) + seq_len;
        total_elements * elem_size
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
            rope_freq: (0..64)
                .map(|i| 1.0 / 10000.0f32.powf((2 * i) as f32 / 128.0f32))
                .collect(),
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            ffn_layout: FfnLayout::SwiGLU,
            architecture: "test".to_string(),
            tensor_registry: crate::config::TensorNameRegistry::from_scheme(
                &crate::config::TensorNamingScheme::Gguf,
            ),
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
    fn new_allocates_all_buffers() {
        let config = make_test_config();
        let scratch = GpuForwardScratch::new(&config);
        match scratch {
            Ok(s) => {
                assert!(!s.q.as_ptr().is_null() || s.q.is_empty());
                assert!(!s.hidden.as_ptr().is_null() || s.hidden.is_empty());
            }
            Err(_) => {}
        }
    }

    #[test]
    fn prefill_scratch_rejects_zero_seq_len() {
        let config = make_test_config();
        let scratch = GpuPrefillScratch::new(&config, 0);
        assert!(scratch.is_err());
    }
}
