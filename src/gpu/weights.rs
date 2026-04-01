//! GPU weight storage in VRAM.
//!
//! Safety-first design:
//! - All hipMalloc/hipMemcpy calls wrapped with error checking
//! - RAII cleanup on Drop prevents VRAM leaks
//! - Never panic, always return GpuError

use super::error::{GpuError, GpuResult};
use super::ffi;
use super::ffi::hipStream_t;
use crate::config::{ModelConfig, TensorName, TensorNamingScheme};
use crate::cpu::transpose::compute_transpose_flag;
use crate::loader::{GgmlType, GgufFile, TensorDesc};
use std::ptr::NonNull;

// ── Weight Metadata ────────────────────────────────────────────────────────────

/// Metadata for a weight tensor on GPU.
///
/// Same as CPU WeightMeta - quantization type and dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    Generic,
    LmHead,
    TiedLmHead,
}

#[derive(Clone, Debug)]
pub struct WeightMeta {
    /// Quantization type (F32, Q4_0, Q4_1, Q8_0, etc.)
    pub wtype: GgmlType,
    /// Dimensions from GGUF (innermost first)
    pub dims: Vec<u64>,
    /// Whether this weight tensor needs transposed access
    pub needs_transpose: bool,
    /// Semantic role derived from GGUF/model metadata.
    ///
    /// This allows dispatch to specialize important tensors without
    /// hardcoding model names or architecture-specific assumptions.
    pub role: TensorRole,
}

impl WeightMeta {
    /// Create metadata from a GGUF tensor descriptor.
    pub fn from_desc(desc: &TensorDesc, needs_transpose: bool) -> Self {
        Self {
            wtype: desc.ggml_type,
            dims: desc.dims.clone(),
            needs_transpose,
            role: TensorRole::Generic,
        }
    }

    /// Total size in bytes for this weight tensor.
    pub fn byte_size(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_desc_works() {
        let desc = TensorDesc {
            name: "test.weight".to_string(),
            ggml_type: GgmlType::Q4_0,
            dims: vec![1024, 768],
            offset: 0,
        };
        let meta = WeightMeta::from_desc(&desc, false);
        assert_eq!(meta.wtype, GgmlType::Q4_0);
        assert_eq!(meta.dims, vec![1024, 768]);
        assert_eq!(meta.byte_size(), 1024 * 768);
        assert_eq!(meta.num_elements(), 1024 * 768);
    }

    #[test]
    fn byte_size_calculates_correctly() {
        let meta = WeightMeta {
            wtype: GgmlType::F32,
            dims: vec![100, 200],
            needs_transpose: false,
            role: TensorRole::Generic,
        };
        assert_eq!(meta.byte_size(), 100 * 200);
    }
}

// ── GPU Buffer (RAII) ─────────────────────────────────────────────────────────────

/// RAII wrapper for GPU memory allocation.
///
/// Ensures memory is freed when dropped.
/// Never leaks VRAM, even on panic.
pub struct GpuBuffer {
    /// Pointer to GPU memory (null if empty)
    ptr: Option<NonNull<u8>>,
    /// Size in bytes
    size: usize,
}

impl GpuBuffer {
    /// Allocate GPU memory with safety checking.
    ///
    /// Returns error if allocation fails (OutOfMemory).
    pub fn alloc(size: usize) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        let ptr = ffi::hip_malloc(size)?;

        // Verify allocation succeeded (pointer not null)
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
        })?;

        Ok(Self {
            ptr: Some(nn),
            size,
        })
    }

    /// Create empty buffer (no allocation).
    pub fn empty() -> Self {
        Self { ptr: None, size: 0 }
    }

    /// Get pointer to GPU memory.
    ///
    /// Returns None if buffer is empty.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
            .map(|nn| nn.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Set size in bytes (only for empty or manually managed buffers).
    pub fn set_size(&mut self, size: usize) {
        self.size = size;
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copy data from CPU to this GPU buffer.
    pub fn copy_from_host(&mut self, src: &[u8]) -> GpuResult<()> {
        if src.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    src.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d(self.as_ptr(), src.as_ptr(), self.size)
    }

    /// Copy data from CPU to this GPU buffer on an explicit HIP stream.
    pub fn copy_from_host_on_stream(&mut self, src: &[u8], stream: hipStream_t) -> GpuResult<()> {
        if src.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    src.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_h2d_async(self.as_ptr(), src.as_ptr(), self.size, stream)
    }

    /// Copy data from GPU buffer to CPU.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> GpuResult<()> {
        if dst.len() != self.size {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!(
                    "size mismatch: got {} bytes, expected {}",
                    dst.len(),
                    self.size
                ),
            });
        }
        if self.size == 0 {
            return Ok(());
        }
        ffi::hip_memcpy_d2h(dst.as_mut_ptr(), self.as_ptr(), self.size)
    }
}

// ── GPU Pinned Buffer (RAII) ──────────────────────────────────────────────────────

/// RAII wrapper for Pinned (Page-locked) Host memory.
///
/// Allows for high-speed DMA transfers and zero-copy access from GPU.
pub struct GpuPinnedBuffer {
    ptr: Option<NonNull<u8>>,
    size: usize,
}

impl GpuPinnedBuffer {
    pub fn alloc(size: usize) -> GpuResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: None, size: 0 });
        }

        let ptr = ffi::hip_host_malloc(size)?;
        let nn = NonNull::new(ptr).ok_or_else(|| GpuError::OutOfMemory {
            requested: size,
            available: 0,
        })?;

        Ok(Self {
            ptr: Some(nn),
            size,
        })
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
            .map(|nn| nn.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    pub fn as_slice<T>(&self) -> &[T] {
        if self.size == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        if self.size == 0 {
            return &mut [];
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_ptr() as *mut T,
                self.size / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for GpuPinnedBuffer {
    fn drop(&mut self) {
        if let Some(nn) = self.ptr {
            let _ = ffi::hip_host_free(nn.as_ptr());
        }
    }
}

unsafe impl Send for GpuPinnedBuffer {}
unsafe impl Sync for GpuPinnedBuffer {}

// SAFETY: Send/Sync are safe because this represents owned GPU memory
// Access is only through &mut self for copy operations
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Some(nn) = self.ptr {
            ffi::hip_free(nn.as_ptr());
            // Ignore errors in Drop - can't panic here
            self.ptr = None;
        }
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    #[test]
    fn empty_buffer_has_no_allocation() {
        let buf = GpuBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.size(), 0);
        assert_eq!(buf.as_ptr(), std::ptr::null_mut());
    }

    #[test]
    fn alloc_zero_size_returns_empty() {
        let buf = GpuBuffer::alloc(0).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn copy_from_host_rejects_size_mismatch() {
        let mut buf = GpuBuffer::alloc(100).unwrap();
        let data = vec![1u8; 50]; // Wrong size
        let result = buf.copy_from_host(&data);
        assert!(result.is_err());
    }
}

fn supports_gpu_matrix_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::F32
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q8_0
    )
}

fn derive_tensor_role(is_lm_head: bool, is_tied: bool) -> TensorRole {
    match (is_lm_head, is_tied) {
        (true, true) => TensorRole::TiedLmHead,
        (true, false) => TensorRole::LmHead,
        (false, _) => TensorRole::Generic,
    }
}

fn build_matrix_meta(
    weight_name: &str,
    dims: &[u64],
    wtype: GgmlType,
    config: &ModelConfig,
    is_lm_head: bool,
    is_tied: bool,
) -> GpuResult<WeightMeta> {
    if dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: weight_name.to_string(),
            dims: dims.to_vec(),
            reason: "matrix weights must have at least 2 dimensions".to_string(),
        });
    }

    if !supports_gpu_matrix_type(wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: weight_name.to_string(),
            wtype,
        });
    }

    Ok(WeightMeta {
        wtype,
        dims: dims.to_vec(),
        needs_transpose: compute_transpose_flag(
            weight_name,
            dims,
            wtype,
            config,
            is_lm_head,
            is_tied,
        ),
        role: derive_tensor_role(is_lm_head, is_tied),
    })
}

fn upload_tensor_bytes(data: &[u8]) -> GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(data.len())?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

// ── GPU Layer Weights ─────────────────────────────────────────────────────────────

/// Weights for a single transformer layer, stored in VRAM.
///
/// All weight tensors are stored in their native quantized format.
/// GPU kernels dequantize during inference.
pub struct GpuLayerWeights {
    /// RMS norm weights for attention (always F32)
    pub attn_norm: GpuBuffer,
    /// Query projection weights (quantized)
    pub attn_q: GpuBuffer,
    pub attn_q_meta: WeightMeta,
    /// Query bias (optional, always F32 if present)
    pub attn_q_bias: Option<GpuBuffer>,
    /// Key projection weights (quantized)
    pub attn_k: GpuBuffer,
    pub attn_k_meta: WeightMeta,
    /// Key bias (optional)
    pub attn_k_bias: Option<GpuBuffer>,
    /// Value projection weights (quantized)
    pub attn_v: GpuBuffer,
    pub attn_v_meta: WeightMeta,
    /// Value bias (optional)
    pub attn_v_bias: Option<GpuBuffer>,
    /// Attention output projection (quantized)
    pub attn_o: GpuBuffer,
    pub attn_o_meta: WeightMeta,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: GpuBuffer,
    /// FFN gate projection (SwiGLU gate) (quantized)
    pub ffn_gate: GpuBuffer,
    pub ffn_gate_meta: WeightMeta,
    /// FFN up projection (quantized)
    pub ffn_up: GpuBuffer,
    pub ffn_up_meta: WeightMeta,
    /// FFN down projection (quantized)
    pub ffn_down: GpuBuffer,
    pub ffn_down_meta: WeightMeta,
}

impl GpuLayerWeights {
    /// Load a single layer's weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    pub fn load(file: &GgufFile, layer: usize, config: &ModelConfig) -> GpuResult<Self> {
        // Helper to load weight into GPU buffer with metadata
        let load_weight = |name: &str| -> GpuResult<(GpuBuffer, WeightMeta)> {
            let t = file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", name),
                })?;

            let meta = build_matrix_meta(name, t.dims, t.ggml_type, config, false, false)?;
            let buf = upload_tensor_bytes(t.data)?;

            Ok((buf, meta))
        };

        // Helper to load weight with fallback names (for MoE models)
        let load_weight_fallback = |names: &[&str]| -> GpuResult<(GpuBuffer, WeightMeta)> {
            for name in names {
                match file.tensor(name) {
                    Ok(Some(t)) => {
                        let meta =
                            build_matrix_meta(name, t.dims, t.ggml_type, config, false, false)?;
                        let buf = upload_tensor_bytes(t.data)?;
                        return Ok((buf, meta));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        return Err(GpuError::HipApiError {
                            code: -1,
                            description: format!("tensor lookup failed: {}", e),
                        });
                    }
                }
            }
            Err(GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: tried {:?}", names),
            })
        };

        // Helper to load F32 weight
        let load_f32 = |name: &str| -> GpuResult<GpuBuffer> {
            let t = file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", name),
                })?;

            let data = t.data;
            let mut buf = GpuBuffer::alloc(data.len())?;
            buf.copy_from_host(data)?;
            Ok(buf)
        };

        // Helper to load optional F32 weight
        let load_f32_opt = |name: &str| -> GpuResult<Option<GpuBuffer>> {
            match file.tensor(name) {
                Ok(Some(t)) => {
                    let mut buf = GpuBuffer::alloc(t.data.len())?;
                    buf.copy_from_host(t.data)?;
                    Ok(Some(buf))
                }
                Ok(None) => Ok(None),
                Err(_) => Ok(None), // Missing tensor is OK for optional weights
            }
        };

        // Load all weights - if any fail, this entire struct is dropped (RAII cleanup)
        let attn_norm = load_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?;
        let (attn_q, attn_q_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
        let attn_q_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnQBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_k, attn_k_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
        let attn_k_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnKBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_v, attn_v_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
        let attn_v_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnVBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_o, attn_o_meta) = load_weight(
            &config
                .tensor_registry
                .resolve(TensorName::AttnOutput, layer),
        )?;
        let ffn_norm = load_f32(&config.tensor_registry.resolve(TensorName::FfnNorm, layer))?;

        // For MoE models, try _exps tensors first, then fall back to standard names
        let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
        let ffn_up_name = config.tensor_registry.resolve(TensorName::FfnUp, layer);
        let ffn_down_name = config.tensor_registry.resolve(TensorName::FfnDown, layer);

        let (ffn_gate, ffn_gate_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_gate_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnGateExps, layer);
                load_weight_fallback(&[&ffn_gate_exps_name, &ffn_gate_name])?
            } else {
                load_weight(&ffn_gate_name)?
            };

        let (ffn_up, ffn_up_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_up_exps_name = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
                load_weight_fallback(&[&ffn_up_exps_name, &ffn_up_name])?
            } else {
                load_weight(&ffn_up_name)?
            };

        let (ffn_down, ffn_down_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_down_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnDownExps, layer);
                load_weight_fallback(&[&ffn_down_exps_name, &ffn_down_name])?
            } else {
                load_weight(&ffn_down_name)?
            };

        Ok(Self {
            attn_norm,
            attn_q,
            attn_q_meta,
            attn_q_bias,
            attn_k,
            attn_k_meta,
            attn_k_bias,
            attn_v,
            attn_v_meta,
            attn_v_bias,
            attn_o,
            attn_o_meta,
            ffn_norm,
            ffn_gate,
            ffn_gate_meta,
            ffn_up,
            ffn_up_meta,
            ffn_down,
            ffn_down_meta,
        })
    }
}

// ── GPU Model Weights ─────────────────────────────────────────────────────────────

/// All weights for a transformer model, stored in VRAM.
///
/// Holds token embeddings, all layer weights, output norm, and LM head.
pub struct GpuModelWeights {
    /// Per-layer weights (all in VRAM)
    pub layers: Vec<GpuLayerWeights>,
    /// Token embedding matrix (quantized, in VRAM)
    pub token_emb: GpuBuffer,
    pub token_emb_meta: WeightMeta,
    /// Final RMS norm weights (F32, in VRAM)
    pub output_norm: GpuBuffer,
    /// Language model head / output projection (quantized, in VRAM)
    pub lm_head: GpuBuffer,
    pub lm_head_meta: WeightMeta,
    /// Whether LM head is tied to token embeddings
    pub lm_head_tied: bool,
}

impl GpuModelWeights {
    /// Load all weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    pub fn load(file: &GgufFile, config: &ModelConfig) -> GpuResult<Self> {
        let n = config.num_layers;

        // Helper to load tensor into GPU buffer
        let load_tensor =
            |name: &str, is_lm_head: bool, is_tied: bool| -> GpuResult<(GpuBuffer, WeightMeta)> {
                let t = file
                    .tensor(name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor lookup failed: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", name),
                    })?;

                let meta =
                    build_matrix_meta(name, t.dims, t.ggml_type, config, is_lm_head, is_tied)?;
                let buf = upload_tensor_bytes(t.data)?;

                Ok((buf, meta))
            };

        // Load token embeddings using registry
        let token_emb_name = config.tensor_registry.resolve(TensorName::TokenEmb, 0);
        let (token_emb, token_emb_meta) = load_tensor(&token_emb_name, false, false)?;

        // Load output norm using registry
        let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
        let output_norm_view = file
            .tensor(&output_norm_name)
            .map_err(|_| GpuError::WeightTransferFailed { layer: 0 })?
            .ok_or_else(|| GpuError::WeightTransferFailed { layer: 0 })?;

        let mut output_norm = GpuBuffer::alloc(output_norm_view.data.len())?;
        output_norm.copy_from_host(output_norm_view.data)?;

        // LM head: use lm_head.weight if present, otherwise tie to embeddings
        let lm_head_name = config.tensor_registry.resolve(TensorName::LmHead, 0);
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor(&lm_head_name) {
            let (buf, meta) = load_tensor(&lm_head_name, true, false)?;
            (buf, meta, false)
        } else {
            // Materialize a second GPU buffer for the tied head.
            // This keeps the decode path simple while preserving explicit tied metadata.
            let (buf, _) = load_tensor(&token_emb_name, false, false)?;
            let tied_meta = build_matrix_meta(
                &lm_head_name,
                &token_emb_meta.dims,
                token_emb_meta.wtype,
                config,
                true,
                true,
            )?;
            (buf, tied_meta, true)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            eprintln!("[GPU weights] Loading layer {}/{}", i + 1, n);
            let layer = GpuLayerWeights::load(file, i, config)?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            token_emb,
            token_emb_meta,
            output_norm,
            lm_head,
            lm_head_meta,
            lm_head_tied,
        })
    }

    /// Get weights for a specific layer.
    pub fn layer(&self, i: usize) -> &GpuLayerWeights {
        &self.layers[i]
    }
}

#[cfg(test)]
mod matrix_meta_tests {
    use super::*;
    use crate::config::{AttentionLayout, TensorNameRegistry, TensorNamingScheme};

    fn make_test_config() -> ModelConfig {
        ModelConfig {
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
            attention_layout: AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
        }
    }

    #[test]
    fn explicit_lm_head_matches_cpu_transpose_rule() {
        let config = make_test_config();
        let meta = build_matrix_meta(
            "output.weight",
            &[32000, 1024],
            GgmlType::Q4_0,
            &config,
            true,
            false,
        )
        .unwrap();

        assert!(!meta.needs_transpose);
        assert_eq!(meta.role, TensorRole::LmHead);
    }

    #[test]
    fn tied_lm_head_is_marked_transposed() {
        let config = make_test_config();
        let meta = build_matrix_meta(
            "output.weight",
            &[32000, 1024],
            GgmlType::Q4_0,
            &config,
            true,
            true,
        )
        .unwrap();

        assert!(meta.needs_transpose);
        assert_eq!(meta.role, TensorRole::TiedLmHead);
    }

    #[test]
    fn unsupported_matrix_type_is_rejected() {
        let config = make_test_config();
        let err = build_matrix_meta(
            "blk.0.attn_q.weight",
            &[1024, 1024],
            GgmlType::Q6_K,
            &config,
            false,
            false,
        )
        .unwrap_err();

        assert!(matches!(err, GpuError::UnsupportedWeightType { .. }));
    }

    #[test]
    fn matrix_weights_require_two_dims() {
        let config = make_test_config();
        let err = build_matrix_meta(
            "output.weight",
            &[32000],
            GgmlType::Q4_0,
            &config,
            true,
            false,
        )
        .unwrap_err();

        assert!(matches!(err, GpuError::InvalidWeightLayout { .. }));
    }
}
