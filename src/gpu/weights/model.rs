use super::super::error::{GpuError, GpuResult};
use super::super::ffi;
use super::super::ffi::hipStream_t;
use super::super::vram_budget::{
    active_or_default_device_id, check_model_load_headroom, query_vram_budget,
};
use super::buffer::GpuBuffer;
use super::layer::{GpuLayerWeights, GpuMpoWeights, GpuSparseCsrWeights};
use super::metadata::{TensorRole, WeightMeta};
use super::upload::*;
use crate::config::{ModelConfig, TensorName};
use crate::cpu::transpose::compute_transpose_flag;
use crate::gpu::kernels::quant;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType};

// ── GPU Weight Tensor Abstraction ────────────────────────────────────────────────

/// A GPU-resident weight tensor that may be stored as dense quantized bytes,
/// sparse CSR, or MPO (matrix product operator) compressed format.
///
/// The enum abstracts over the storage so forward paths don't need to
/// branch on every combination of type + role.  It also makes it
/// impossible to accidentally pass a sparse CSR buffer to a dense GEMV
/// kernel, because the dense `GpuBuffer` is only accessible via the
/// `Dense` variant.
#[derive(Debug)]
pub enum GpuWeightTensor {
    /// Standard dense quantized buffer (Q4_0, Q8_0, F32, etc.)
    Dense(GpuBuffer),
    /// Sparse CSR representation — values, column indices, row pointers.
    SparseCsr(GpuSparseCsrWeights),
    /// MPO compressed representation — site data + site dimensions.
    Mpo(GpuMpoWeights),
}

impl GpuWeightTensor {
    /// Return the dense `GpuBuffer` if this is the `Dense` variant.
    pub fn as_dense(&self) -> Option<&GpuBuffer> {
        match self {
            GpuWeightTensor::Dense(buf) => Some(buf),
            _ => None,
        }
    }

    /// Return the sparse CSR weights if this is the `SparseCsr` variant.
    pub fn as_sparse_csr(&self) -> Option<&GpuSparseCsrWeights> {
        match self {
            GpuWeightTensor::SparseCsr(sparse) => Some(sparse),
            _ => None,
        }
    }

    /// Return the MPO weights if this is the `Mpo` variant.
    pub fn as_mpo(&self) -> Option<&GpuMpoWeights> {
        match self {
            GpuWeightTensor::Mpo(mpo) => Some(mpo),
            _ => None,
        }
    }

    /// VRAM bytes allocated for this tensor.
    pub fn size(&self) -> usize {
        match self {
            GpuWeightTensor::Dense(buf) => buf.size(),
            GpuWeightTensor::SparseCsr(sparse) => {
                sparse.values.size() + sparse.col_idx.size() + sparse.row_ptr.size()
            }
            GpuWeightTensor::Mpo(mpo) => mpo.site_data.size() + mpo.site_dims.size(),
        }
    }

    /// Raw GPU pointer of the primary data buffer (for dense variants).
    /// Returns null for sparse/MPO since they have multiple buffers.
    pub fn as_ptr(&self) -> *mut u8 {
        match self {
            GpuWeightTensor::Dense(buf) => buf.as_ptr(),
            _ => std::ptr::null_mut(),
        }
    }
}

// ── GPU Model Weights ─────────────────────────────────────────────────────────────

/// All weights for a transformer model, stored in VRAM.
///
/// Holds token embeddings, all layer weights, output norm, and LM head.
#[derive(Debug)]
pub struct GpuModelWeights {
    /// Per-layer weights (all in VRAM)
    pub layers: Vec<GpuLayerWeights>,
    /// Token embedding matrix (dense, sparse CSR, or MPO)
    pub token_emb: GpuWeightTensor,
    pub token_emb_meta: WeightMeta,
    /// Final RMS norm weights (F32, in VRAM)
    pub output_norm: GpuBuffer,
    /// Language model head / output projection (dense, sparse CSR, or MPO)
    pub lm_head: GpuWeightTensor,
    pub lm_head_meta: WeightMeta,
    /// Whether LM head is tied to token embeddings
    pub lm_head_tied: bool,
    /// Gemma4 Per-Layer Embedding (PLE) tensors (optional, only for gemma4)
    pub per_layer_token_emb: Option<GpuBuffer>,
    pub per_layer_token_emb_meta: Option<WeightMeta>,
    pub per_layer_model_proj: Option<GpuWeightTensor>,
    pub per_layer_model_proj_meta: Option<WeightMeta>,
    pub per_layer_proj_norm: Option<GpuBuffer>,
    /// Cached pointer-mix used by decode-graph key construction.
    decode_binding_tag: u64,
}

impl GpuModelWeights {
    /// Compute total VRAM bytes allocated for this model's weights.
    pub fn vram_bytes(&self) -> usize {
        let mut total = 0;
        total += self.token_emb.size();
        total += self.output_norm.size();
        total += self.lm_head.size();
        for layer in &self.layers {
            total += layer.estimate_vram_usage();
        }
        total
    }

    pub fn has_unsupported_gpu_gemv_weights(&self) -> bool {
        for layer in &self.layers {
            if let Some(ref meta) = layer.attn_qkv_meta {
                if !crate::gpu::ops::supports_gemv_type(meta.wtype) {
                    return true;
                }
            } else {
                if !crate::gpu::ops::supports_gemv_type(layer.attn_q_meta.wtype)
                    || !crate::gpu::ops::supports_gemv_type(layer.attn_k_meta.wtype)
                    || (!layer.attn_v.is_empty()
                        && !crate::gpu::ops::supports_gemv_type(layer.attn_v_meta.wtype))
                {
                    return true;
                }
            }
            if let Some(ref meta) = layer.attn_gate_meta {
                if !crate::gpu::ops::supports_gemv_type(meta.wtype) {
                    return true;
                }
            }
            if !crate::gpu::ops::supports_gemv_type(layer.attn_o_meta.wtype) {
                return true;
            }
            if let Some(ref meta) = layer.ffn_gate_meta {
                if !crate::gpu::ops::supports_gemv_type(meta.wtype) {
                    return true;
                }
            }
            if !crate::gpu::ops::supports_gemv_type(layer.ffn_up_meta.wtype)
                || !crate::gpu::ops::supports_gemv_type(layer.ffn_down_meta.wtype)
            {
                return true;
            }
        }
        false
    }

    /// Load all weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    /// Includes cumulative VRAM tracking to prevent model from exceeding safe limits.
    pub fn load(file: &GgufFile, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_for_device(file, config, active_or_default_device_id())
    }

    pub fn load_for_device(
        file: &GgufFile,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        let n = config.num_layers;
        ffi::hip_set_device(device_id)?;
        let budget = query_vram_budget(device_id)?;

        // Helper to load tensor into GPU buffer without VRAM tracking (done separately)
        fn load_tensor_no_track(
            file: &GgufFile,
            name: &str,
            config: &ModelConfig,
            is_lm_head: bool,
            is_tied: bool,
            device_id: i32,
        ) -> GpuResult<(GpuBuffer, WeightMeta)> {
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

            let meta = build_matrix_meta(name, t.dims, t.ggml_type, config, is_lm_head, is_tied)?;
            let buf = upload_tensor_bytes_for_device(t.data, device_id)?;

            Ok((buf, meta))
        }

        let mut estimated_vram_used = 0usize;

        // Load token embeddings using registry
        let token_emb_name = config.tensor_registry.resolve(TensorName::TokenEmb, 0);
        let token_emb_view = file
            .tensor(&token_emb_name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", token_emb_name),
            })?;
        check_model_load_headroom(budget, estimated_vram_used, token_emb_view.data.len())?;
        let (token_emb, token_emb_meta) =
            load_tensor_no_track(file, &token_emb_name, config, false, false, device_id)?;
        estimated_vram_used += token_emb.size();

        // Load output norm using registry (fallback to token_embd_norm.weight for LFM-style models)
        let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
        let output_norm_view = if let Some(v) = file
            .tensor(&output_norm_name)
            .map_err(|_| GpuError::WeightTransferFailed { layer: 0 })?
        {
            v
        } else {
            file.tensor("token_embd_norm.weight")
                .map_err(|_| GpuError::WeightTransferFailed { layer: 0 })?
                .ok_or(GpuError::WeightTransferFailed { layer: 0 })?
        };

        check_model_load_headroom(budget, estimated_vram_used, output_norm_view.data.len())?;
        let mut output_norm = GpuBuffer::alloc_for_device(output_norm_view.data.len(), device_id)?;
        output_norm.copy_from_host(output_norm_view.data)?;
        estimated_vram_used += output_norm.size();

        // LM head: use lm_head.weight if present, otherwise tie to embeddings
        let lm_head_name = config.tensor_registry.resolve(TensorName::LmHead, 0);
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor(&lm_head_name) {
            let lm_head_view = file
                .tensor(&lm_head_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", lm_head_name),
                })?;
            check_model_load_headroom(budget, estimated_vram_used, lm_head_view.data.len())?;
            let (buf, meta) =
                load_tensor_no_track(file, &lm_head_name, config, true, false, device_id)?;
            estimated_vram_used += buf.size();
            (GpuWeightTensor::Dense(buf), meta, false)
        } else {
            // Materialize a second GPU buffer for the tied head.
            // Instead of keeping it in Q4_0 and using slow CPU transposition fallback,
            // we dequantize it to f32, quantize to standard Q8_0 layout on upload,
            // and run it natively and perfectly on the GPU.
            let (buf, tied_meta) = prepare_tied_lm_head_q8(
                token_emb_view.data,
                token_emb_meta.wtype,
                &token_emb_meta,
                config,
                None,
                device_id,
            )?;
            estimated_vram_used += buf.size();
            (GpuWeightTensor::Dense(buf), tied_meta, true)
        };

        // Gemma4 Per-Layer Embedding (PLE) weights (optional, only for gemma4)
        let (
            per_layer_token_emb,
            per_layer_token_emb_meta,
            per_layer_model_proj,
            per_layer_model_proj_meta,
            per_layer_proj_norm,
        ) = if config.architecture == "gemma4" {
            let per_layer_token_emb_name = config
                .tensor_registry
                .resolve(TensorName::PerLayerTokenEmb, 0);
            let per_layer_model_proj_name = config
                .tensor_registry
                .resolve(TensorName::PerLayerModelProj, 0);
            let per_layer_proj_norm_name = config
                .tensor_registry
                .resolve(TensorName::PerLayerProjNorm, 0);

            let per_layer_token_emb = if file.has_tensor(&per_layer_token_emb_name) {
                let per_layer_token_emb_view = file
                    .tensor(&per_layer_token_emb_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor lookup failed: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", per_layer_token_emb_name),
                    })?;
                check_model_load_headroom(
                    budget,
                    estimated_vram_used,
                    per_layer_token_emb_view.data.len(),
                )?;
                let (buf, meta) = load_tensor_no_track(
                    file,
                    &per_layer_token_emb_name,
                    config,
                    false,
                    false,
                    device_id,
                )?;
                estimated_vram_used += buf.size();
                (Some(buf), Some(meta))
            } else {
                (None, None)
            };

            let per_layer_model_proj = if file.has_tensor(&per_layer_model_proj_name) {
                let per_layer_model_proj_view = file
                    .tensor(&per_layer_model_proj_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor lookup failed: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", per_layer_model_proj_name),
                    })?;
                check_model_load_headroom(
                    budget,
                    estimated_vram_used,
                    per_layer_model_proj_view.data.len(),
                )?;
                let (buf, meta) = load_tensor_no_track(
                    file,
                    &per_layer_model_proj_name,
                    config,
                    false,
                    false,
                    device_id,
                )?;
                estimated_vram_used += buf.size();
                (Some(GpuWeightTensor::Dense(buf)), Some(meta))
            } else {
                (None, None)
            };

            let per_layer_proj_norm = if file.has_tensor(&per_layer_proj_norm_name) {
                let per_layer_proj_norm_view = file
                    .tensor(&per_layer_proj_norm_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor lookup failed: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", per_layer_proj_norm_name),
                    })?;
                check_model_load_headroom(
                    budget,
                    estimated_vram_used,
                    per_layer_proj_norm_view.data.len(),
                )?;
                let mut buf =
                    GpuBuffer::alloc_for_device(per_layer_proj_norm_view.data.len(), device_id)?;
                buf.copy_from_host(per_layer_proj_norm_view.data)?;
                estimated_vram_used += buf.size();
                Some(buf)
            } else {
                None
            };

            (
                per_layer_token_emb.0,
                per_layer_token_emb.1,
                per_layer_model_proj.0,
                per_layer_model_proj.1,
                per_layer_proj_norm,
            )
        } else {
            (None, None, None, None, None)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            eprintln!("[GPU weights] Loading layer {}/{}", i + 1, n);
            let layer_vram = GpuLayerWeights::estimate_vram_usage_from_file(file, i, config)?;
            check_model_load_headroom(budget, estimated_vram_used, layer_vram)?;
            let layer = GpuLayerWeights::load_for_device(file, i, config, device_id)?;
            estimated_vram_used += layer_vram;

            layers.push(layer);
        }

        let decode_binding_tag = compute_model_binding_tag(&layers, &output_norm, &lm_head);

        eprintln!(
            "[GPU weights] Total estimated VRAM usage: {} MB",
            estimated_vram_used / (1024 * 1024)
        );

        Ok(Self {
            layers,
            token_emb: GpuWeightTensor::Dense(token_emb),
            token_emb_meta,
            output_norm,
            lm_head,
            lm_head_meta,
            lm_head_tied,
            per_layer_token_emb,
            per_layer_token_emb_meta,
            per_layer_model_proj,
            per_layer_model_proj_meta,
            per_layer_proj_norm,
            decode_binding_tag,
        })
    }

    /// Get weights for a specific layer.
    pub fn layer(&self, i: usize) -> &GpuLayerWeights {
        &self.layers[i]
    }

    /// Cached pointer-mix used by decode-graph key construction.
    #[inline]
    pub fn binding_tag(&self) -> u64 {
        self.decode_binding_tag
    }

    /// Check if any weights use Q6_K quantization (incompatible with HIP graph capture)
    pub fn uses_q6_k_quantization(&self) -> bool {
        // Check token embedding
        if self.token_emb_meta.wtype == GgmlType::Q6_K {
            return true;
        }

        // Check output layer
        if self.lm_head_meta.wtype == GgmlType::Q6_K {
            return true;
        }

        // Check all layers
        for layer in &self.layers {
            if layer.attn_q_meta.wtype == GgmlType::Q6_K
                || layer.attn_k_meta.wtype == GgmlType::Q6_K
                || (!layer.attn_v.is_empty() && layer.attn_v_meta.wtype == GgmlType::Q6_K)
                || layer.attn_o_meta.wtype == GgmlType::Q6_K
                || layer
                    .attn_qkv_meta
                    .as_ref()
                    .is_some_and(|m| m.wtype == GgmlType::Q6_K)
                || layer
                    .attn_gate_meta
                    .as_ref()
                    .is_some_and(|m| m.wtype == GgmlType::Q6_K)
                || layer.ssm.as_ref().is_some_and(|ssm| {
                    ssm.alpha_meta.wtype == GgmlType::Q6_K
                        || ssm.beta_meta.wtype == GgmlType::Q6_K
                        || ssm.out_meta.wtype == GgmlType::Q6_K
                })
                || layer
                    .ffn_gate_meta
                    .as_ref()
                    .is_some_and(|m| m.wtype == GgmlType::Q6_K)
                || layer.ffn_up_meta.wtype == GgmlType::Q6_K
                || layer.ffn_down_meta.wtype == GgmlType::Q6_K
            {
                return true;
            }
        }

        false
    }

    /// Check if attention weights use Q4_0 quantization (compatible with batched prefill)
    pub fn uses_q4_0_quantization(&self) -> bool {
        // Check all attention layers - must be Q4_0 for batched prefill
        // FFN layers can be other types since we don't use batched kernels for them yet
        for layer in &self.layers {
            if layer.attn_qkv_meta.is_some() {
                return false;
            }
            if layer.attn_q_meta.wtype != GgmlType::Q4_0
                || layer.attn_k_meta.wtype != GgmlType::Q4_0
                || (!layer.attn_v.is_empty() && layer.attn_v_meta.wtype != GgmlType::Q4_0)
                || layer.attn_o_meta.wtype != GgmlType::Q4_0
            {
                return false;
            }
        }

        true
    }

    /// Load all model weights from an RFM model file into GPU memory.
    pub fn load_rfm(file: &RfmFile, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_rfm_for_device(file, config, active_or_default_device_id())
    }

    pub fn load_rfm_for_device(
        file: &RfmFile,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        let n = config.num_layers;
        ffi::hip_set_device(device_id)?;
        let budget = query_vram_budget(device_id)?;

        let mut estimated_vram_used = 0usize;

        // Load token embedding weights
        let token_emb_name = "token_embd.weight";
        let token_emb_view = file
            .tensor(token_emb_name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor error: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", token_emb_name),
            })?;

        let token_emb_wtype = rfm_type_to_ggml(&token_emb_view.wtype);
        let token_emb_unpacked_size = match token_emb_view.wtype {
            RfmType::Q4Split => (token_emb_view.element_count() / 32) * 18,
            _ => token_emb_view.data.len(),
        };

        check_model_load_headroom(budget, estimated_vram_used, token_emb_unpacked_size)?;

        let token_emb = match token_emb_view.wtype {
            RfmType::Q4Split => {
                let raw_gpu_buf = upload_tensor_bytes_for_device(token_emb_view.data, device_id)?;
                let num_blocks = token_emb_view.element_count() / 32;
                let out_gpu_buf = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                quant::gpu_unpack_q4_split(
                    raw_gpu_buf.as_ptr() as *const u8,
                    out_gpu_buf.as_ptr() as *mut u8,
                    num_blocks,
                    hipStream_t::null(),
                )?;
                GpuWeightTensor::Dense(out_gpu_buf)
            }
            RfmType::SparseCsr { .. } => {
                match super::layer::try_load_sparse_csr(file, token_emb_name, device_id)? {
                    Some(sparse) => GpuWeightTensor::SparseCsr(sparse),
                    None => {
                        return Err(GpuError::UnsupportedOperation {
                            operation: format!("load RFM tensor {}", token_emb_name),
                            reason: "sparse CSR tensor parsing failed".to_string(),
                        });
                    }
                }
            }
            RfmType::Mpo { .. } => {
                match super::layer::try_load_mpo(file, token_emb_name, device_id)? {
                    Some(mpo) => GpuWeightTensor::Mpo(mpo),
                    None => {
                        return Err(GpuError::UnsupportedOperation {
                            operation: format!("load RFM tensor {}", token_emb_name),
                            reason: "MPO tensor parsing failed".to_string(),
                        });
                    }
                }
            }
            _ => GpuWeightTensor::Dense(upload_tensor_bytes_for_device(
                token_emb_view.data,
                device_id,
            )?),
        };

        let token_emb_meta = WeightMeta {
            wtype: token_emb_wtype,
            dims: token_emb_view.dims.to_vec(),
            needs_transpose: false,
            role: TensorRole::Generic,
            svd_k: None,
        };
        estimated_vram_used += token_emb.size();

        // Output norm (fallback to token_embd_norm.weight for LFM-style models)
        let output_norm_name = "output_norm.weight";
        let output_norm_view = if let Some(v) =
            file.tensor(output_norm_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })? {
            v
        } else {
            file.tensor("token_embd_norm.weight")
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: "tensor not found: output_norm.weight or token_embd_norm.weight"
                        .to_string(),
                })?
        };

        check_model_load_headroom(budget, estimated_vram_used, output_norm_view.data.len())?;
        let output_norm = upload_tensor_bytes_for_device(output_norm_view.data, device_id)?;
        estimated_vram_used += output_norm.size();

        // LM head
        let lm_head_name = "output.weight";
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor(lm_head_name) {
            let lm_view = file
                .tensor(lm_head_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", lm_head_name),
                })?;
            let lm_wtype = rfm_type_to_ggml(&lm_view.wtype);
            let needs_transpose = compute_transpose_flag(
                TensorRole::from_name(lm_head_name, true, false),
                lm_view.dims,
                lm_wtype,
                config,
            );

            let lm_unpacked_size = match lm_view.wtype {
                RfmType::Q4Split => (lm_view.element_count() / 32) * 18,
                _ => lm_view.data.len(),
            };

            check_model_load_headroom(budget, estimated_vram_used, lm_unpacked_size)?;

            let lm_head = match lm_view.wtype {
                RfmType::Q4Split => {
                    let raw_gpu_buf = upload_tensor_bytes_for_device(lm_view.data, device_id)?;
                    let num_blocks = lm_view.element_count() / 32;
                    let out_gpu_buf = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                    quant::gpu_unpack_q4_split(
                        raw_gpu_buf.as_ptr() as *const u8,
                        out_gpu_buf.as_ptr() as *mut u8,
                        num_blocks,
                        hipStream_t::null(),
                    )?;
                    GpuWeightTensor::Dense(out_gpu_buf)
                }
                RfmType::SparseCsr { .. } => {
                    match super::layer::try_load_sparse_csr(file, lm_head_name, device_id)? {
                        Some(sparse) => GpuWeightTensor::SparseCsr(sparse),
                        None => {
                            return Err(GpuError::UnsupportedOperation {
                                operation: format!("load RFM tensor {}", lm_head_name),
                                reason: "sparse CSR tensor parsing failed".to_string(),
                            });
                        }
                    }
                }
                RfmType::Mpo { .. } => {
                    match super::layer::try_load_mpo(file, lm_head_name, device_id)? {
                        Some(mpo) => GpuWeightTensor::Mpo(mpo),
                        None => {
                            return Err(GpuError::UnsupportedOperation {
                                operation: format!("load RFM tensor {}", lm_head_name),
                                reason: "MPO tensor parsing failed".to_string(),
                            });
                        }
                    }
                }
                _ => {
                    GpuWeightTensor::Dense(upload_tensor_bytes_for_device(lm_view.data, device_id)?)
                }
            };

            let meta = WeightMeta {
                wtype: lm_wtype,
                dims: lm_view.dims.to_vec(),
                needs_transpose,
                role: TensorRole::LmHead,
                svd_k: None,
            };
            estimated_vram_used += lm_head.size();
            (lm_head, meta, false)
        } else {
            // Weight tying
            // Instead of keeping it in Q4_0 and using slow CPU transposition fallback,
            // we dequantize it to f32, quantize to standard Q8_0 layout on upload,
            // and run it natively and perfectly on the GPU.
            let (buf, tied_meta) = prepare_tied_lm_head_q8(
                token_emb_view.data,
                token_emb_meta.wtype,
                &token_emb_meta,
                config,
                Some(token_emb_view.wtype),
                device_id,
            )?;
            estimated_vram_used += buf.size();
            (GpuWeightTensor::Dense(buf), tied_meta, true)
        };

        // Gemma4 Per-Layer Embedding (PLE) weights (optional, only for gemma4)
        let (
            per_layer_token_emb,
            per_layer_token_emb_meta,
            per_layer_model_proj,
            per_layer_model_proj_meta,
            per_layer_proj_norm,
        ) = if config.architecture == "gemma4" {
            let per_layer_token_emb_name = "per_layer_token_embd.weight";
            let per_layer_model_proj_name = "per_layer_model_proj.weight";
            let per_layer_proj_norm_name = "per_layer_proj_norm.weight";

            let per_layer_token_emb = if file.has_tensor(per_layer_token_emb_name) {
                let per_layer_token_emb_view = file
                    .tensor(per_layer_token_emb_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor error: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", per_layer_token_emb_name),
                    })?;

                let per_layer_token_emb_wtype = rfm_type_to_ggml(&per_layer_token_emb_view.wtype);
                let per_layer_token_emb_unpacked_size = match per_layer_token_emb_view.wtype {
                    RfmType::Q4Split => (per_layer_token_emb_view.element_count() / 32) * 18,
                    _ => per_layer_token_emb_view.data.len(),
                };

                check_model_load_headroom(
                    budget,
                    estimated_vram_used,
                    per_layer_token_emb_unpacked_size,
                )?;

                let per_layer_token_emb_buf = match per_layer_token_emb_view.wtype {
                    RfmType::Q4Split => {
                        let raw_gpu_buf = upload_tensor_bytes_for_device(
                            per_layer_token_emb_view.data,
                            device_id,
                        )?;
                        let num_blocks = per_layer_token_emb_view.element_count() / 32;
                        let out_gpu_buf = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                        quant::gpu_unpack_q4_split(
                            raw_gpu_buf.as_ptr() as *const u8,
                            out_gpu_buf.as_ptr() as *mut u8,
                            num_blocks,
                            hipStream_t::null(),
                        )?;
                        out_gpu_buf
                    }
                    _ => upload_tensor_bytes_for_device(per_layer_token_emb_view.data, device_id)?,
                };

                let per_layer_token_emb_meta = WeightMeta {
                    wtype: per_layer_token_emb_wtype,
                    dims: per_layer_token_emb_view.dims.to_vec(),
                    needs_transpose: false,
                    role: TensorRole::Generic,
                    svd_k: None,
                };
                estimated_vram_used += per_layer_token_emb_buf.size();
                (
                    Some(per_layer_token_emb_buf),
                    Some(per_layer_token_emb_meta),
                )
            } else {
                (None, None)
            };

            let per_layer_model_proj = if file.has_tensor(per_layer_model_proj_name) {
                let per_layer_model_proj_view = file
                    .tensor(per_layer_model_proj_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor error: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", per_layer_model_proj_name),
                    })?;

                let per_layer_model_proj_wtype = rfm_type_to_ggml(&per_layer_model_proj_view.wtype);
                let per_layer_model_proj_unpacked_size = match per_layer_model_proj_view.wtype {
                    RfmType::Q4Split => (per_layer_model_proj_view.element_count() / 32) * 18,
                    _ => per_layer_model_proj_view.data.len(),
                };

                check_model_load_headroom(
                    budget,
                    estimated_vram_used,
                    per_layer_model_proj_unpacked_size,
                )?;

                let per_layer_model_proj_buf = match per_layer_model_proj_view.wtype {
                    RfmType::Q4Split => {
                        let raw_gpu_buf = upload_tensor_bytes_for_device(
                            per_layer_model_proj_view.data,
                            device_id,
                        )?;
                        let num_blocks = per_layer_model_proj_view.element_count() / 32;
                        let out_gpu_buf = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                        quant::gpu_unpack_q4_split(
                            raw_gpu_buf.as_ptr() as *const u8,
                            out_gpu_buf.as_ptr() as *mut u8,
                            num_blocks,
                            hipStream_t::null(),
                        )?;
                        out_gpu_buf
                    }
                    _ => upload_tensor_bytes_for_device(per_layer_model_proj_view.data, device_id)?,
                };

                let per_layer_model_proj_meta = WeightMeta {
                    wtype: per_layer_model_proj_wtype,
                    dims: per_layer_model_proj_view.dims.to_vec(),
                    needs_transpose: false,
                    role: TensorRole::Generic,
                    svd_k: None,
                };
                estimated_vram_used += per_layer_model_proj_buf.size();
                (
                    Some(GpuWeightTensor::Dense(per_layer_model_proj_buf)),
                    Some(per_layer_model_proj_meta),
                )
            } else {
                (None, None)
            };

            let per_layer_proj_norm = if file.has_tensor(per_layer_proj_norm_name) {
                let per_layer_proj_norm_view = file
                    .tensor(per_layer_proj_norm_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor error: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", per_layer_proj_norm_name),
                    })?;

                check_model_load_headroom(
                    budget,
                    estimated_vram_used,
                    per_layer_proj_norm_view.data.len(),
                )?;
                let buf = upload_tensor_bytes_for_device(per_layer_proj_norm_view.data, device_id)?;
                estimated_vram_used += buf.size();
                Some(buf)
            } else {
                None
            };

            (
                per_layer_token_emb.0,
                per_layer_token_emb.1,
                per_layer_model_proj.0,
                per_layer_model_proj.1,
                per_layer_proj_norm,
            )
        } else {
            (None, None, None, None, None)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            eprintln!("[GPU weights] Loading layer {}/{} from RFM", i + 1, n);
            let layer_vram = estimate_rfm_layer_vram(file, i)?;
            check_model_load_headroom(budget, estimated_vram_used, layer_vram)?;
            let layer = GpuLayerWeights::load_rfm_for_device(file, i, config, device_id)?;
            estimated_vram_used += layer_vram;
            layers.push(layer);
        }

        let decode_binding_tag = compute_model_binding_tag(&layers, &output_norm, &lm_head);

        eprintln!(
            "[GPU weights] Total estimated VRAM usage: {} MB",
            estimated_vram_used / (1024 * 1024)
        );

        Ok(Self {
            layers,
            token_emb,
            token_emb_meta,
            output_norm,
            lm_head,
            lm_head_meta,
            lm_head_tied,
            per_layer_token_emb,
            per_layer_token_emb_meta,
            per_layer_model_proj,
            per_layer_model_proj_meta,
            per_layer_proj_norm,
            decode_binding_tag,
        })
    }
}

#[inline]
fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn gpu_layer_weights_binding_tag(layer: &GpuLayerWeights) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, layer.attn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.attn_q.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_q_norm
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(
        tag,
        layer
            .attn_q_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_k.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_k_norm
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(
        tag,
        layer
            .attn_k_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_v.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_v_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(
        tag,
        layer
            .attn_qkv
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(
        tag,
        layer
            .attn_gate
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    if let Some(ssm) = &layer.ssm {
        tag = mix_binding_tag(tag, ssm.a.as_ptr() as usize);
        tag = mix_binding_tag(tag, ssm.dt.as_ptr() as usize);
        tag = mix_binding_tag(tag, ssm.norm.as_ptr() as usize);
        tag = mix_binding_tag(tag, ssm.conv1d.as_ptr() as usize);
        tag = mix_binding_tag(tag, ssm.alpha.as_ptr() as usize);
        tag = mix_binding_tag(tag, ssm.beta.as_ptr() as usize);
        tag = mix_binding_tag(tag, ssm.out.as_ptr() as usize);
    }
    tag = mix_binding_tag(tag, layer.attn_o.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_norm.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .ffn_gate
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.ffn_up.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .ffn_gate_up_interleaved
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(
        tag,
        layer
            .ffn_gate_up_interleaved_tile4
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    mix_binding_tag(tag, layer.ffn_down.as_ptr() as usize)
}

fn compute_model_binding_tag(
    layers: &[GpuLayerWeights],
    output_norm: &GpuBuffer,
    lm_head: &GpuWeightTensor,
) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, output_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, lm_head.as_ptr() as usize);
    for layer in layers {
        tag ^= gpu_layer_weights_binding_tag(layer);
    }
    tag
}

fn prepare_tied_lm_head_q8(
    token_emb_data: &[u8],
    token_emb_wtype: GgmlType,
    token_emb_meta: &WeightMeta,
    config: &ModelConfig,
    rfm_type: Option<RfmType>,
    device_id: i32,
) -> GpuResult<(GpuBuffer, WeightMeta)> {
    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;

    // Unpack RfmType::Q4Split to standard Q4_0 layout on CPU if needed
    let mut unpacked_data;
    let data_to_use = match rfm_type {
        Some(RfmType::Q4Split) => {
            let num_elements = vocab_size * hidden_size;
            let num_blocks = num_elements / 32;
            let mut out = Vec::with_capacity(num_blocks * 18);
            let scales_size = num_blocks * 2;
            let zp_size = num_blocks * 2;
            let scales = &token_emb_data[0..scales_size];
            let nibbles = &token_emb_data[scales_size + zp_size..];
            for i in 0..num_blocks {
                out.push(scales[i * 2]);
                out.push(scales[i * 2 + 1]);
                out.extend_from_slice(&nibbles[i * 16..(i + 1) * 16]);
            }
            unpacked_data = out;
            &unpacked_data[..]
        }
        _ => token_emb_data,
    };

    // 1. Dequantize embedding matrix from original format to flat float32 matrix of shape [vocab_size, hidden_size]
    let mut f32_data = vec![0.0f32; vocab_size * hidden_size];
    for id in 0..vocab_size {
        let out_row = &mut f32_data[id * hidden_size..(id + 1) * hidden_size];
        match token_emb_wtype {
            GgmlType::Q4_0 => {
                crate::cpu::quant::embed_q4_0(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q4_1 => {
                crate::cpu::quant::embed_q4_1(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q8_0 => {
                crate::cpu::quant::embed_q8_0(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q6_K => {
                crate::cpu::quant::embed_q6_k(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q4_K => {
                crate::cpu::quant::embed_q4_k(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q5_K => {
                crate::cpu::quant::embed_q5_k(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q3_K => {
                crate::cpu::quant::embed_q3_k(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q5_0 => {
                crate::cpu::quant::embed_q5_0(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::Q5_1 => {
                crate::cpu::quant::embed_q5_1(id, data_to_use, out_row, hidden_size);
            }
            GgmlType::F32 => {
                let f32_emb = unsafe {
                    std::slice::from_raw_parts(
                        data_to_use.as_ptr() as *const f32,
                        data_to_use.len() / 4,
                    )
                };
                crate::cpu::quant::embed_f32(id, f32_emb, out_row);
            }
            GgmlType::F16 => {
                let start_idx = id * hidden_size;
                for i in 0..hidden_size {
                    let offset = (start_idx + i) * 2;
                    let bits = u16::from_le_bytes([data_to_use[offset], data_to_use[offset + 1]]);
                    out_row[i] = half::f16::from_bits(bits).to_f32();
                }
            }
            GgmlType::BF16 => {
                let start_idx = id * hidden_size;
                for i in 0..hidden_size {
                    let offset = (start_idx + i) * 2;
                    let bits = u16::from_le_bytes([data_to_use[offset], data_to_use[offset + 1]]);
                    out_row[i] = half::bf16::to_f32(half::bf16::from_bits(bits));
                }
            }
            other => {
                return Err(GpuError::UnsupportedWeightType {
                    tensor: "tied_lm_head_dequant".to_string(),
                    wtype: other,
                });
            }
        }
    }

    // 2. Quantize the float32 matrix to standard Q8_0 format (34 bytes per block) row-by-row
    let num_blocks_per_row = hidden_size / 32;
    let mut q8_data = vec![0u8; vocab_size * num_blocks_per_row * 34];

    for id in 0..vocab_size {
        let f32_row = &f32_data[id * hidden_size..(id + 1) * hidden_size];
        let q8_row = &mut q8_data[id * num_blocks_per_row * 34..(id + 1) * num_blocks_per_row * 34];

        for b in 0..num_blocks_per_row {
            let src_block = &f32_row[b * 32..(b + 1) * 32];
            let dst_block = &mut q8_row[b * 34..(b + 1) * 34];

            let scale = crate::cpu::quant::quantize_f32_to_q8_0(src_block, &mut dst_block[2..34]);
            let scale_bytes = half::f16::from_f32(scale).to_bits().to_le_bytes();
            dst_block[0..2].copy_from_slice(&scale_bytes);
        }
    }

    // 3. Upload the standard Q8_0 weights to the GPU
    let buf = upload_tensor_bytes_for_device(&q8_data, device_id)?;

    // 4. Create standard Q8_0 WeightMeta (no transpose needed now)
    let meta = WeightMeta {
        wtype: GgmlType::Q8_0,
        dims: token_emb_meta.dims.clone(),
        needs_transpose: false,
        role: TensorRole::TiedLmHead,
        svd_k: None,
    };

    Ok((buf, meta))
}
