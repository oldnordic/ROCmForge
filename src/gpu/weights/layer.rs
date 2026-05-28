use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;
use super::super::vram_budget::active_or_default_device_id;
use super::buffer::GpuBuffer;
use super::metadata::{TensorRole, WeightMeta};
use super::upload::*;
use crate::config::{AttentionLayout, ModelConfig, TensorName, TensorNamingScheme};
use crate::cpu::transpose::compute_transpose_flag;
use crate::gpu::kernels::quant;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType};

// ── GPU Layer Weights ─────────────────────────────────────────────────────────────

/// SVD low-rank outlier correction matrices stored in VRAM.
pub struct SvdCorrection {
    /// Left singular vectors (N_out x k) scaled by singular values
    pub u: GpuBuffer,
    /// Right singular vectors (k x N_in)
    pub v: GpuBuffer,
    /// SVD rank k
    pub k: u32,
}

/// Sparse CSR weight representation for GPU execution.
pub struct GpuSparseCsrWeights {
    pub values: GpuBuffer,
    pub col_idx: GpuBuffer,
    pub row_ptr: GpuBuffer,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

/// MPO (Matrix Product Operator) weight representation for GPU execution.
pub struct GpuMpoWeights {
    pub site_data: GpuBuffer,
    pub site_dims: GpuBuffer,
    pub n_sites: u32,
}

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
    pub attn_q_svd: Option<SvdCorrection>,
    /// Optional Q RMSNorm for Qwen35 full-attention layers.
    pub attn_q_norm: Option<GpuBuffer>,
    /// Query bias (optional, always F32 if present)
    pub attn_q_bias: Option<GpuBuffer>,
    /// Key projection weights (quantized)
    pub attn_k: GpuBuffer,
    pub attn_k_meta: WeightMeta,
    pub attn_k_svd: Option<SvdCorrection>,
    /// Optional K RMSNorm for Qwen35 full-attention layers.
    pub attn_k_norm: Option<GpuBuffer>,
    /// Key bias (optional)
    pub attn_k_bias: Option<GpuBuffer>,
    /// Value projection weights (quantized)
    pub attn_v: GpuBuffer,
    pub attn_v_meta: WeightMeta,
    pub attn_v_svd: Option<SvdCorrection>,
    /// Value bias (optional)
    pub attn_v_bias: Option<GpuBuffer>,
    /// Fused QKV projection used by Qwen35-style hybrid layers.
    pub attn_qkv: Option<GpuBuffer>,
    pub attn_qkv_meta: Option<WeightMeta>,
    /// Attention gate projection used by Qwen35 hybrid attention/SSM mixing.
    pub attn_gate: Option<GpuBuffer>,
    pub attn_gate_meta: Option<WeightMeta>,
    /// SSM tensors used by Qwen35 hybrid layers.
    pub ssm: Option<GpuSsmWeights>,
    /// Attention output projection (quantized)
    pub attn_o: GpuBuffer,
    pub attn_o_meta: WeightMeta,
    pub attn_o_svd: Option<SvdCorrection>,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: GpuBuffer,
    /// FFN gate projection (SwiGLU gate) (quantized)
    pub ffn_gate: GpuBuffer,
    pub ffn_gate_meta: WeightMeta,
    pub ffn_gate_svd: Option<SvdCorrection>,
    /// FFN up projection (quantized)
    pub ffn_up: GpuBuffer,
    pub ffn_up_meta: WeightMeta,
    pub ffn_up_svd: Option<SvdCorrection>,
    /// Optional decode-friendly interleaved Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved: Option<GpuBuffer>,
    /// Optional decode-friendly 4-column tiled Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved_tile4: Option<GpuBuffer>,
    /// FFN down projection (quantized)
    pub ffn_down: GpuBuffer,
    pub ffn_down_meta: WeightMeta,
    pub ffn_down_svd: Option<SvdCorrection>,
    /// Optional MoE router and shared-expert weights.
    pub moe: Option<GpuMoeWeights>,
    /// Optional sparse CSR variants for FFN weights.
    pub ffn_gate_sparse: Option<GpuSparseCsrWeights>,
    pub ffn_up_sparse: Option<GpuSparseCsrWeights>,
    pub ffn_down_sparse: Option<GpuSparseCsrWeights>,
    /// Optional MPO variants for FFN weights.
    pub ffn_gate_mpo: Option<GpuMpoWeights>,
    pub ffn_up_mpo: Option<GpuMpoWeights>,
    pub ffn_down_mpo: Option<GpuMpoWeights>,
}

/// Mixture-of-Experts side weights for Qwen-style MoE layers.
pub struct GpuMoeWeights {
    pub router: GpuBuffer,
    pub router_meta: WeightMeta,
    pub router_svd: Option<SvdCorrection>,
    pub shared_gate: Option<GpuBuffer>,
    pub shared_gate_meta: Option<WeightMeta>,
    pub shared_gate_svd: Option<SvdCorrection>,
    pub shared_up: Option<GpuBuffer>,
    pub shared_up_meta: Option<WeightMeta>,
    pub shared_up_svd: Option<SvdCorrection>,
    pub shared_down: Option<GpuBuffer>,
    pub shared_down_meta: Option<WeightMeta>,
    pub shared_down_svd: Option<SvdCorrection>,
    pub shared_gate_inp: Option<GpuBuffer>,
    pub shared_gate_inp_meta: Option<WeightMeta>,
}

/// Native Qwen35 SSM tensors for one layer, resident in VRAM.
pub struct GpuSsmWeights {
    pub a: GpuBuffer,
    pub dt: GpuBuffer,
    pub norm: GpuBuffer,
    pub conv1d: GpuBuffer,
    pub alpha: GpuBuffer,
    pub alpha_meta: WeightMeta,
    pub beta: GpuBuffer,
    pub beta_meta: WeightMeta,
    pub out: GpuBuffer,
    pub out_meta: WeightMeta,
}

fn qwen35_ssm_meta(name: &str, dims: &[u64], wtype: GgmlType, config: &ModelConfig) -> WeightMeta {
    WeightMeta {
        wtype,
        dims: dims.to_vec(),
        needs_transpose: compute_transpose_flag(name, dims, wtype, config, false, false),
        role: TensorRole::Generic,
        svd_k: None,
    }
}

fn qwen35_post_attention_norm_name(config: &ModelConfig, layer: usize) -> Option<String> {
    if config.architecture.contains("qwen35") {
        Some(format!("blk.{}.post_attention_norm.weight", layer))
    } else {
        None
    }
}

fn load_qwen35_ssm_gguf(
    file: &GgufFile,
    layer: usize,
    config: &ModelConfig,
    device_id: i32,
) -> GpuResult<GpuSsmWeights> {
    let load_f32 = |suffix: &str| -> GpuResult<GpuBuffer> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = file
            .tensor(&name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", name),
            })?;
        upload_tensor_bytes_for_device(tensor.data, device_id)
    };
    let load_weight = |suffix: &str| -> GpuResult<(GpuBuffer, WeightMeta)> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = file
            .tensor(&name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", name),
            })?;
        let meta = qwen35_ssm_meta(&name, tensor.dims, tensor.ggml_type, config);
        let buffer = upload_tensor_bytes_for_device(tensor.data, device_id)?;
        Ok((buffer, meta))
    };

    let (alpha, alpha_meta) = load_weight("ssm_alpha.weight")?;
    let (beta, beta_meta) = load_weight("ssm_beta.weight")?;
    let (out, out_meta) = load_weight("ssm_out.weight")?;

    Ok(GpuSsmWeights {
        a: load_f32("ssm_a")?,
        dt: load_f32("ssm_dt")?,
        norm: load_f32("ssm_norm.weight")?,
        conv1d: load_f32("ssm_conv1d.weight")?,
        alpha,
        alpha_meta,
        beta,
        beta_meta,
        out,
        out_meta,
    })
}

fn load_qwen35_ssm_rfm(
    file: &RfmFile,
    layer: usize,
    config: &ModelConfig,
    device_id: i32,
) -> GpuResult<GpuSsmWeights> {
    let load_tensor = |name: &str| -> GpuResult<crate::loader::RfmTensorView<'_>> {
        file.tensor(name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", name),
            })
    };
    let load_f32 = |suffix: &str| -> GpuResult<GpuBuffer> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = load_tensor(&name)?;
        upload_tensor_bytes_for_device(tensor.data, device_id)
    };
    let load_weight = |suffix: &str| -> GpuResult<(GpuBuffer, WeightMeta)> {
        let name = format!("blk.{}.{}", layer, suffix);
        let tensor = load_tensor(&name)?;
        let wtype = rfm_type_to_ggml(&tensor.wtype);
        let mut meta = qwen35_ssm_meta(&name, tensor.dims, wtype, config);
        meta.svd_k = match tensor.wtype {
            RfmType::Q4SvdQuant { k } => Some(k),
            _ => None,
        };
        let buffer = upload_tensor_bytes_for_device(tensor.data, device_id)?;
        Ok((buffer, meta))
    };

    let (alpha, alpha_meta) = load_weight("ssm_alpha.weight")?;
    let (beta, beta_meta) = load_weight("ssm_beta.weight")?;
    let (out, out_meta) = load_weight("ssm_out.weight")?;

    Ok(GpuSsmWeights {
        a: load_f32("ssm_a")?,
        dt: load_f32("ssm_dt")?,
        norm: load_f32("ssm_norm.weight")?,
        conv1d: load_f32("ssm_conv1d.weight")?,
        alpha,
        alpha_meta,
        beta,
        beta_meta,
        out,
        out_meta,
    })
}

fn try_load_sparse_csr(
    file: &RfmFile,
    name: &str,
    device_id: i32,
) -> GpuResult<Option<GpuSparseCsrWeights>> {
    let t = match file.tensor(name) {
        Ok(Some(t)) => t,
        _ => return Ok(None),
    };
    let csr = match t.as_sparse_csr() {
        Some(csr) => csr,
        None => return Ok(None),
    };

    let row_ptr_buf = upload_tensor_bytes_for_device(csr.row_offsets, device_id)?;
    let col_idx_buf = upload_tensor_bytes_for_device(csr.col_indices, device_id)?;
    let values_buf = upload_tensor_bytes_for_device(csr.values, device_id)?;

    Ok(Some(GpuSparseCsrWeights {
        values: values_buf,
        col_idx: col_idx_buf,
        row_ptr: row_ptr_buf,
        rows: csr.rows,
        cols: csr.cols,
        nnz: csr.nnz,
    }))
}

fn try_load_mpo(file: &RfmFile, name: &str, device_id: i32) -> GpuResult<Option<GpuMpoWeights>> {
    let t = match file.tensor(name) {
        Ok(Some(t)) => t,
        _ => return Ok(None),
    };
    let mpo = match t.as_mpo() {
        Some(mpo) => mpo,
        None => return Ok(None),
    };

    let site_data = upload_tensor_bytes_for_device(mpo.data, device_id)?;
    let site_dims_host: Vec<u32> = mpo.site_dims.iter().map(|d| *d as u32).collect();
    let mut site_dims = GpuBuffer::alloc(site_dims_host.len() * std::mem::size_of::<u32>())?;
    let site_dims_bytes = unsafe {
        std::slice::from_raw_parts(
            site_dims_host.as_ptr() as *const u8,
            site_dims_host.len() * std::mem::size_of::<u32>(),
        )
    };
    site_dims.copy_from_host(site_dims_bytes)?;

    Ok(Some(GpuMpoWeights {
        site_data,
        site_dims,
        n_sites: mpo.n_sites as u32,
    }))
}

impl GpuLayerWeights {
    /// Load a single layer's weights from GGUF file into GPU memory.
    ///
    /// Returns error if any allocation or transfer fails.
    /// On error, all allocated memory is freed via Drop.
    pub fn load(file: &GgufFile, layer: usize, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_for_device(file, layer, config, active_or_default_device_id())
    }

    pub fn load_for_device(
        file: &GgufFile,
        layer: usize,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
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
            let buf = upload_tensor_bytes_for_device(t.data, device_id)?;

            Ok((buf, meta))
        };

        // Helper to load weight with fallback names (for MoE models)
        let load_weight_fallback = |names: &[&str]| -> GpuResult<(GpuBuffer, WeightMeta)> {
            for name in names {
                match file.tensor(name) {
                    Ok(Some(t)) => {
                        let meta =
                            build_matrix_meta(name, t.dims, t.ggml_type, config, false, false)?;
                        let buf = upload_tensor_bytes_for_device(t.data, device_id)?;
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
            let mut buf = GpuBuffer::alloc_for_device(data.len(), device_id)?;
            buf.copy_from_host(data)?;
            Ok(buf)
        };

        // Helper to load optional F32 weight
        let load_f32_opt = |name: &str| -> GpuResult<Option<GpuBuffer>> {
            match file.tensor(name) {
                Ok(Some(t)) => {
                    let mut buf = GpuBuffer::alloc_for_device(t.data.len(), device_id)?;
                    buf.copy_from_host(t.data)?;
                    Ok(Some(buf))
                }
                Ok(None) => Ok(None),
                Err(_) => Ok(None), // Missing tensor is OK for optional weights
            }
        };

        // Load all weights - if any fail, this entire struct is dropped (RAII cleanup)
        let attn_norm = load_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?;
        let qkv_name = format!("blk.{}.attn_qkv.weight", layer);
        let layer_has_fused_qkv = matches!(config.attention_layout, AttentionLayout::FusedQkv)
            && file.has_tensor(&qkv_name);
        let (
            attn_q,
            attn_q_meta,
            attn_k,
            attn_k_meta,
            attn_v,
            attn_v_meta,
            attn_qkv,
            attn_qkv_meta,
        ) = if layer_has_fused_qkv {
            let (qkv, qkv_meta) = load_weight(&qkv_name)?;
            (
                GpuBuffer::empty(),
                qkv_meta.clone(),
                GpuBuffer::empty(),
                qkv_meta.clone(),
                GpuBuffer::empty(),
                qkv_meta.clone(),
                Some(qkv),
                Some(qkv_meta),
            )
        } else {
            let (attn_q, attn_q_meta) =
                load_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
            let (attn_k, attn_k_meta) =
                load_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
            let (attn_v, attn_v_meta) =
                load_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
            (
                attn_q,
                attn_q_meta,
                attn_k,
                attn_k_meta,
                attn_v,
                attn_v_meta,
                None,
                None,
            )
        };
        let attn_q_norm = load_f32_opt(&format!("blk.{}.attn_q_norm.weight", layer))?;
        let attn_k_norm = load_f32_opt(&format!("blk.{}.attn_k_norm.weight", layer))?;
        let attn_q_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnQBias, layer)
                .unwrap_or_default(),
        )?;
        let attn_k_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnKBias, layer)
                .unwrap_or_default(),
        )?;
        let attn_v_bias = load_f32_opt(
            &config
                .tensor_registry
                .resolve_optional(TensorName::AttnVBias, layer)
                .unwrap_or_default(),
        )?;
        let (attn_o, attn_o_meta) = if layer_has_fused_qkv {
            let meta = attn_qkv_meta.as_ref().ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "attn_qkv_meta missing for fused QKV layer".to_string(),
            })?;
            (GpuBuffer::empty(), meta.clone())
        } else {
            load_weight(
                &config
                    .tensor_registry
                    .resolve(TensorName::AttnOutput, layer),
            )?
        };
        let (attn_gate, attn_gate_meta, ssm) = if layer_has_fused_qkv {
            let attn_gate_name = format!("blk.{}.attn_gate.weight", layer);
            let (attn_gate, attn_gate_meta) = load_weight(&attn_gate_name)?;
            let ssm = load_qwen35_ssm_gguf(file, layer, config, device_id)?;
            (Some(attn_gate), Some(attn_gate_meta), Some(ssm))
        } else {
            (None, None, None)
        };
        let ffn_norm_name = qwen35_post_attention_norm_name(config, layer)
            .unwrap_or_else(|| config.tensor_registry.resolve(TensorName::FfnNorm, layer));
        let ffn_norm = load_f32(&ffn_norm_name)?;

        // For MoE models, try _exps tensors first, then fall back to standard names
        let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
        let ffn_up_name = config.tensor_registry.resolve(TensorName::FfnUp, layer);
        let ffn_down_name = config.tensor_registry.resolve(TensorName::FfnDown, layer);

        let (ffn_gate_name_used, ffn_gate, ffn_gate_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_gate_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnGateExps, layer);
                let (buf, meta) = load_weight_fallback(&[&ffn_gate_exps_name, &ffn_gate_name])?;
                let chosen = if file.has_tensor(&ffn_gate_exps_name) {
                    ffn_gate_exps_name
                } else {
                    ffn_gate_name.clone()
                };
                (chosen, buf, meta)
            } else {
                let (buf, meta) = load_weight(&ffn_gate_name)?;
                (ffn_gate_name.clone(), buf, meta)
            };

        let (ffn_up_name_used, ffn_up, ffn_up_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_up_exps_name = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
                let (buf, meta) = load_weight_fallback(&[&ffn_up_exps_name, &ffn_up_name])?;
                let chosen = if file.has_tensor(&ffn_up_exps_name) {
                    ffn_up_exps_name
                } else {
                    ffn_up_name.clone()
                };
                (chosen, buf, meta)
            } else {
                let (buf, meta) = load_weight(&ffn_up_name)?;
                (ffn_up_name.clone(), buf, meta)
            };

        let ffn_gate_up_interleaved = match (
            file.tensor(&ffn_gate_name_used).ok().and_then(|t| t),
            file.tensor(&ffn_up_name_used).ok().and_then(|t| t),
        ) {
            (Some(gate_t), Some(up_t)) => try_build_q4_0_gate_up_interleaved(
                gate_t.data,
                &ffn_gate_meta,
                up_t.data,
                &ffn_up_meta,
            )
            .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
            .transpose()?,
            _ => None,
        };
        let ffn_gate_up_interleaved_tile4 = match (
            file.tensor(&ffn_gate_name_used).ok().and_then(|t| t),
            file.tensor(&ffn_up_name_used).ok().and_then(|t| t),
        ) {
            (Some(gate_t), Some(up_t)) => try_build_q4_0_gate_up_interleaved_tile4(
                gate_t.data,
                &ffn_gate_meta,
                up_t.data,
                &ffn_up_meta,
            )
            .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
            .transpose()?,
            _ => None,
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

        let moe = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
            let router_name = config
                .tensor_registry
                .resolve(TensorName::FfnGateInp, layer);
            if file.has_tensor(&router_name) {
                let (router, router_meta) = load_weight(&router_name)?;
                let load_moe_weight_opt =
                    |name: &str| -> GpuResult<Option<(GpuBuffer, WeightMeta)>> {
                        if file.has_tensor(name) {
                            load_weight(name).map(Some)
                        } else {
                            Ok(None)
                        }
                    };

                let shared_gate_name = format!("blk.{}.ffn_gate_shexp.weight", layer);
                let shared_up_name = format!("blk.{}.ffn_up_shexp.weight", layer);
                let shared_down_name = format!("blk.{}.ffn_down_shexp.weight", layer);
                let shared_gate_inp_name = format!("blk.{}.ffn_gate_inp_shexp.weight", layer);

                let shared_gate = load_moe_weight_opt(&shared_gate_name)?;
                let shared_up = load_moe_weight_opt(&shared_up_name)?;
                let shared_down = load_moe_weight_opt(&shared_down_name)?;
                let shared_gate_inp = load_moe_weight_opt(&shared_gate_inp_name)?;
                let (shared_gate, shared_gate_meta) = match shared_gate {
                    Some((buf, meta)) => (Some(buf), Some(meta)),
                    None => (None, None),
                };
                let (shared_up, shared_up_meta) = match shared_up {
                    Some((buf, meta)) => (Some(buf), Some(meta)),
                    None => (None, None),
                };
                let (shared_down, shared_down_meta) = match shared_down {
                    Some((buf, meta)) => (Some(buf), Some(meta)),
                    None => (None, None),
                };
                let (shared_gate_inp, shared_gate_inp_meta) = match shared_gate_inp {
                    Some((buf, meta)) => (Some(buf), Some(meta)),
                    None => (None, None),
                };

                Some(GpuMoeWeights {
                    router,
                    router_meta,
                    router_svd: None,
                    shared_gate,
                    shared_gate_meta,
                    shared_gate_svd: None,
                    shared_up,
                    shared_up_meta,
                    shared_up_svd: None,
                    shared_down,
                    shared_down_meta,
                    shared_down_svd: None,
                    shared_gate_inp,
                    shared_gate_inp_meta,
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            attn_norm,
            attn_q,
            attn_q_meta,
            attn_q_svd: None,
            attn_q_norm,
            attn_q_bias,
            attn_k,
            attn_k_meta,
            attn_k_svd: None,
            attn_k_norm,
            attn_k_bias,
            attn_v,
            attn_v_meta,
            attn_v_svd: None,
            attn_v_bias,
            attn_qkv,
            attn_qkv_meta,
            attn_gate,
            attn_gate_meta,
            ssm,
            attn_o,
            attn_o_meta,
            attn_o_svd: None,
            ffn_norm,
            ffn_gate,
            ffn_gate_meta,
            ffn_gate_svd: None,
            ffn_up,
            ffn_up_meta,
            ffn_up_svd: None,
            ffn_gate_up_interleaved,
            ffn_gate_up_interleaved_tile4,
            ffn_down,
            ffn_down_meta,
            ffn_down_svd: None,
            moe,
            ffn_gate_sparse: None,
            ffn_up_sparse: None,
            ffn_down_sparse: None,
            ffn_gate_mpo: None,
            ffn_up_mpo: None,
            ffn_down_mpo: None,
        })
    }

    pub(super) fn estimate_vram_usage_from_file(
        file: &GgufFile,
        layer: usize,
        config: &ModelConfig,
    ) -> GpuResult<usize> {
        let tensor_bytes = |name: &str| -> GpuResult<usize> {
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
            Ok(t.data.len())
        };
        let tensor_bytes_optional = |name: &str| -> GpuResult<usize> {
            if name.is_empty() {
                return Ok(0);
            }
            Ok(file
                .tensor(name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor lookup failed: {}", e),
                })?
                .map(|t| t.data.len())
                .unwrap_or(0))
        };
        let choose_ffn_tensor = |primary: &str, fallback: &str| -> GpuResult<(String, usize)> {
            if file.has_tensor(primary) {
                Ok((primary.to_string(), tensor_bytes(primary)?))
            } else {
                Ok((fallback.to_string(), tensor_bytes(fallback)?))
            }
        };

        let attn_norm_name = config.tensor_registry.resolve(TensorName::AttnNorm, layer);
        let attn_q_name = config.tensor_registry.resolve(TensorName::AttnQ, layer);
        let attn_k_name = config.tensor_registry.resolve(TensorName::AttnK, layer);
        let attn_v_name = config.tensor_registry.resolve(TensorName::AttnV, layer);
        let attn_o_name = config
            .tensor_registry
            .resolve(TensorName::AttnOutput, layer);
        let qkv_name = format!("blk.{}.attn_qkv.weight", layer);
        let layer_has_fused_qkv = matches!(config.attention_layout, AttentionLayout::FusedQkv)
            && file.has_tensor(&qkv_name);
        let ffn_norm_name = qwen35_post_attention_norm_name(config, layer)
            .unwrap_or_else(|| config.tensor_registry.resolve(TensorName::FfnNorm, layer));
        let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
        let ffn_up_name = config.tensor_registry.resolve(TensorName::FfnUp, layer);
        let ffn_down_name = config.tensor_registry.resolve(TensorName::FfnDown, layer);

        let (ffn_gate_name_used, ffn_gate_bytes) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let primary = config
                    .tensor_registry
                    .resolve(TensorName::FfnGateExps, layer);
                choose_ffn_tensor(&primary, &ffn_gate_name)?
            } else {
                (ffn_gate_name.clone(), tensor_bytes(&ffn_gate_name)?)
            };
        let (ffn_up_name_used, ffn_up_bytes) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let primary = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
                choose_ffn_tensor(&primary, &ffn_up_name)?
            } else {
                (ffn_up_name.clone(), tensor_bytes(&ffn_up_name)?)
            };
        let ffn_down_bytes = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE)
        {
            let primary = config
                .tensor_registry
                .resolve(TensorName::FfnDownExps, layer);
            choose_ffn_tensor(&primary, &ffn_down_name)?.1
        } else {
            tensor_bytes(&ffn_down_name)?
        };
        let moe_extra_bytes =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                tensor_bytes_optional(
                    &config
                        .tensor_registry
                        .resolve(TensorName::FfnGateInp, layer),
                )? + tensor_bytes_optional(&format!("blk.{}.ffn_gate_shexp.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ffn_up_shexp.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ffn_down_shexp.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ffn_gate_inp_shexp.weight", layer))?
            } else {
                0
            };

        let gate_tensor = file
            .tensor(&ffn_gate_name_used)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", ffn_gate_name_used),
            })?;
        let up_tensor = file
            .tensor(&ffn_up_name_used)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor lookup failed: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", ffn_up_name_used),
            })?;
        let ffn_gate_meta = build_matrix_meta(
            &ffn_gate_name_used,
            gate_tensor.dims,
            gate_tensor.ggml_type,
            config,
            false,
            false,
        )?;
        let ffn_up_meta = build_matrix_meta(
            &ffn_up_name_used,
            up_tensor.dims,
            up_tensor.ggml_type,
            config,
            false,
            false,
        )?;
        let interleaved_bytes = try_build_q4_0_gate_up_interleaved(
            gate_tensor.data,
            &ffn_gate_meta,
            up_tensor.data,
            &ffn_up_meta,
        )
        .map_or(0, |bytes| bytes.len());
        let interleaved_tile4_bytes = try_build_q4_0_gate_up_interleaved_tile4(
            gate_tensor.data,
            &ffn_gate_meta,
            up_tensor.data,
            &ffn_up_meta,
        )
        .map_or(0, |bytes| bytes.len());

        let attention_bytes = if layer_has_fused_qkv {
            tensor_bytes(&qkv_name)?
                + tensor_bytes(&format!("blk.{}.attn_gate.weight", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_a", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_dt", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_norm.weight", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_conv1d.weight", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_alpha.weight", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_beta.weight", layer))?
                + tensor_bytes(&format!("blk.{}.ssm_out.weight", layer))?
        } else {
            tensor_bytes(&attn_q_name)?
                + tensor_bytes_optional(&format!("blk.{}.attn_q_norm.weight", layer))?
                + tensor_bytes_optional(
                    &config
                        .tensor_registry
                        .resolve_optional(TensorName::AttnQBias, layer)
                        .unwrap_or_default(),
                )?
                + tensor_bytes(&attn_k_name)?
                + tensor_bytes_optional(&format!("blk.{}.attn_k_norm.weight", layer))?
                + tensor_bytes_optional(
                    &config
                        .tensor_registry
                        .resolve_optional(TensorName::AttnKBias, layer)
                        .unwrap_or_default(),
                )?
                + tensor_bytes(&attn_v_name)?
                + tensor_bytes_optional(
                    &config
                        .tensor_registry
                        .resolve_optional(TensorName::AttnVBias, layer)
                        .unwrap_or_default(),
                )?
                + tensor_bytes(&attn_o_name)?
        };

        Ok(tensor_bytes(&attn_norm_name)?
            + attention_bytes
            + tensor_bytes(&ffn_norm_name)?
            + ffn_gate_bytes
            + ffn_up_bytes
            + interleaved_bytes
            + interleaved_tile4_bytes
            + ffn_down_bytes
            + moe_extra_bytes)
    }

    /// Estimate total VRAM usage for this layer in bytes.
    ///
    /// This is a conservative estimate that sums all buffer sizes.
    pub fn estimate_vram_usage(&self) -> usize {
        let mut total = 0;

        // Mandatory buffers
        total += self.attn_norm.size();
        total += self.attn_q.size();
        total += self.attn_k.size();
        total += self.attn_v.size();
        total += self.attn_o.size();
        total += self.ffn_norm.size();
        total += self.ffn_gate.size();
        total += self.ffn_up.size();
        total += self.ffn_down.size();

        // Optional buffers
        if let Some(ref buf) = self.attn_q_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_q_norm {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_k_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_k_norm {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_v_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_qkv {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_gate {
            total += buf.size();
        }
        if let Some(ref ssm) = self.ssm {
            total += ssm.a.size();
            total += ssm.dt.size();
            total += ssm.norm.size();
            total += ssm.conv1d.size();
            total += ssm.alpha.size();
            total += ssm.beta.size();
            total += ssm.out.size();
        }
        if let Some(ref buf) = self.ffn_gate_up_interleaved {
            total += buf.size();
        }
        if let Some(ref buf) = self.ffn_gate_up_interleaved_tile4 {
            total += buf.size();
        }
        if let Some(ref moe) = self.moe {
            total += moe.router.size();
            if let Some(ref buf) = moe.shared_gate {
                total += buf.size();
            }
            if let Some(ref buf) = moe.shared_up {
                total += buf.size();
            }
            if let Some(ref buf) = moe.shared_down {
                total += buf.size();
            }
            if let Some(ref buf) = moe.shared_gate_inp {
                total += buf.size();
            }
        }

        if let Some(ref svd) = self.attn_q_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref svd) = self.attn_k_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref svd) = self.attn_v_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref svd) = self.attn_o_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref svd) = self.ffn_gate_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref svd) = self.ffn_up_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref svd) = self.ffn_down_svd {
            total += svd.u.size() + svd.v.size();
        }
        if let Some(ref moe) = self.moe {
            if let Some(ref svd) = moe.router_svd {
                total += svd.u.size() + svd.v.size();
            }
            if let Some(ref svd) = moe.shared_gate_svd {
                total += svd.u.size() + svd.v.size();
            }
            if let Some(ref svd) = moe.shared_up_svd {
                total += svd.u.size() + svd.v.size();
            }
            if let Some(ref svd) = moe.shared_down_svd {
                total += svd.u.size() + svd.v.size();
            }
        }

        total
    }

    /// Load a single layer's weights from an RFM model file into GPU memory.
    pub fn load_rfm(file: &RfmFile, layer: usize, config: &ModelConfig) -> GpuResult<Self> {
        Self::load_rfm_for_device(file, layer, config, active_or_default_device_id())
    }

    pub fn load_rfm_for_device(
        file: &RfmFile,
        layer: usize,
        config: &ModelConfig,
        device_id: i32,
    ) -> GpuResult<Self> {
        // Helper to load and unpack an RFM tensor into standard Q4_0 layout
        let load_rfm_weight =
            |name: &str,
             needs_transpose: bool|
             -> GpuResult<(GpuBuffer, WeightMeta, Option<SvdCorrection>)> {
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

                let wtype = rfm_type_to_ggml(&t.wtype);
                let svd_k = match t.wtype {
                    RfmType::Q4SvdQuant { k } => Some(k),
                    _ => None,
                };
                let meta = WeightMeta {
                    wtype,
                    dims: t.dims.to_vec(),
                    needs_transpose,
                    role: TensorRole::Generic,
                    svd_k,
                };

                let base_buf = match t.wtype {
                    RfmType::Q4Split | RfmType::Q4SvdQuant { .. } => {
                        let raw_gpu_buf = upload_tensor_bytes_for_device(t.data, device_id)?;
                        let num_blocks = t.element_count() / 32;
                        let out_gpu_buf = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                        quant::gpu_unpack_q4_split(
                            raw_gpu_buf.as_ptr() as *const u8,
                            out_gpu_buf.as_ptr() as *mut u8,
                            num_blocks,
                            hipStream_t::null(),
                        )?;
                        out_gpu_buf
                    }
                    RfmType::SparseCsr { .. } | RfmType::Mpo { .. } => {
                        // Upload raw data as-is; sparse/MPO dispatch will interpret it
                        upload_tensor_bytes_for_device(t.data, device_id)?
                    }
                    _ => upload_tensor_bytes_for_device(t.data, device_id)?,
                };

                let svd_corr = if let RfmType::Q4SvdQuant { k } = t.wtype {
                    let u_name = format!("{}.svd_u", name);
                    let v_name = format!("{}.svd_v", name);
                    let u_t = file
                        .tensor(&u_name)
                        .map_err(|e| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD U lookup failed for {}: {}", name, e),
                        })?
                        .ok_or_else(|| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD U tensor not found: {}", u_name),
                        })?;
                    let v_t = file
                        .tensor(&v_name)
                        .map_err(|e| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD V lookup failed for {}: {}", name, e),
                        })?
                        .ok_or_else(|| GpuError::HipApiError {
                            code: -1,
                            description: format!("SVD V tensor not found: {}", v_name),
                        })?;

                    let u_buf = upload_tensor_bytes_for_device(u_t.data, device_id)?;
                    let v_buf = upload_tensor_bytes_for_device(v_t.data, device_id)?;
                    Some(SvdCorrection {
                        u: u_buf,
                        v: v_buf,
                        k,
                    })
                } else {
                    None
                };

                Ok((base_buf, meta, svd_corr))
            };

        // Helper to load RMS Norm weights (which are F32)
        let load_rfm_norm = |name: &str| -> GpuResult<GpuBuffer> {
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
            upload_tensor_bytes_for_device(t.data, device_id)
        };

        let q_name = format!("blk.{}.attn_q.weight", layer);
        let k_name = format!("blk.{}.attn_k.weight", layer);
        let v_name = format!("blk.{}.attn_v.weight", layer);
        let o_name = format!("blk.{}.attn_output.weight", layer);
        let qkv_name = format!("blk.{}.attn_qkv.weight", layer);
        let layer_has_fused_qkv = matches!(config.attention_layout, AttentionLayout::FusedQkv)
            && file.has_tensor(&qkv_name);

        let (
            attn_q,
            attn_q_meta,
            attn_q_svd,
            attn_k,
            attn_k_meta,
            attn_k_svd,
            attn_v,
            attn_v_meta,
            attn_v_svd,
            attn_qkv,
            attn_qkv_meta,
        ) = if layer_has_fused_qkv {
            let qkv_view = file
                .tensor(&qkv_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", qkv_name),
                })?;
            let qkv_tr = compute_transpose_flag(
                &qkv_name,
                qkv_view.dims,
                rfm_type_to_ggml(&qkv_view.wtype),
                config,
                false,
                false,
            );
            let (qkv, qkv_meta, qkv_svd) = load_rfm_weight(&qkv_name, qkv_tr)?;
            (
                GpuBuffer::empty(),
                qkv_meta.clone(),
                None,
                GpuBuffer::empty(),
                qkv_meta.clone(),
                None,
                GpuBuffer::empty(),
                qkv_meta.clone(),
                None,
                Some(qkv),
                Some(qkv_meta),
            )
        } else {
            let q_view = file
                .tensor(&q_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", q_name),
                })?;
            let k_view = file
                .tensor(&k_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", k_name),
                })?;
            let v_view = file
                .tensor(&v_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", v_name),
                })?;
            let q_tr = compute_transpose_flag(
                &q_name,
                q_view.dims,
                rfm_type_to_ggml(&q_view.wtype),
                config,
                false,
                false,
            );
            let k_tr = compute_transpose_flag(
                &k_name,
                k_view.dims,
                rfm_type_to_ggml(&k_view.wtype),
                config,
                false,
                false,
            );
            let v_tr = compute_transpose_flag(
                &v_name,
                v_view.dims,
                rfm_type_to_ggml(&v_view.wtype),
                config,
                false,
                false,
            );
            let (attn_q, attn_q_meta, attn_q_svd) = load_rfm_weight(&q_name, q_tr)?;
            let (attn_k, attn_k_meta, attn_k_svd) = load_rfm_weight(&k_name, k_tr)?;
            let (attn_v, attn_v_meta, attn_v_svd) = load_rfm_weight(&v_name, v_tr)?;
            (
                attn_q,
                attn_q_meta,
                attn_q_svd,
                attn_k,
                attn_k_meta,
                attn_k_svd,
                attn_v,
                attn_v_meta,
                attn_v_svd,
                None,
                None,
            )
        };
        let (attn_o, attn_o_meta, attn_o_svd) = if layer_has_fused_qkv {
            let meta = attn_qkv_meta.as_ref().ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: "attn_qkv_meta missing for fused QKV layer".to_string(),
            })?;
            (GpuBuffer::empty(), meta.clone(), None)
        } else {
            let o_view = file
                .tensor(&o_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", o_name),
                })?;
            let o_tr = compute_transpose_flag(
                &o_name,
                o_view.dims,
                rfm_type_to_ggml(&o_view.wtype),
                config,
                false,
                false,
            );
            load_rfm_weight(&o_name, o_tr)?
        };

        let attn_q_bias = None;
        let attn_k_bias = None;
        let attn_v_bias = None;

        let attn_q_norm = if file.has_tensor(&format!("blk.{}.attn_q_norm.weight", layer)) {
            Some(load_rfm_norm(&format!("blk.{}.attn_q_norm.weight", layer))?)
        } else {
            None
        };
        let attn_k_norm = if file.has_tensor(&format!("blk.{}.attn_k_norm.weight", layer)) {
            Some(load_rfm_norm(&format!("blk.{}.attn_k_norm.weight", layer))?)
        } else {
            None
        };

        let (attn_gate, attn_gate_meta, ssm) = if layer_has_fused_qkv {
            let attn_gate_name = format!("blk.{}.attn_gate.weight", layer);
            let attn_gate_view = file
                .tensor(&attn_gate_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", attn_gate_name),
                })?;
            let attn_gate_tr = compute_transpose_flag(
                &attn_gate_name,
                attn_gate_view.dims,
                rfm_type_to_ggml(&attn_gate_view.wtype),
                config,
                false,
                false,
            );
            let (attn_gate, attn_gate_meta, _) = load_rfm_weight(&attn_gate_name, attn_gate_tr)?;
            let ssm = load_qwen35_ssm_rfm(file, layer, config, device_id)?;
            (Some(attn_gate), Some(attn_gate_meta), Some(ssm))
        } else {
            (None, None, None)
        };

        // FFN gate+up fusion or separate SVD-Quant loading
        let ffn_gate_up_name = format!("blk.{}.ffn_gate_up.weight", layer);
        let ffn_gate_name = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE)
        {
            let exp_name = config
                .tensor_registry
                .resolve(TensorName::FfnGateExps, layer);
            if file.has_tensor(&exp_name) {
                exp_name
            } else {
                config.tensor_registry.resolve(TensorName::FfnGate, layer)
            }
        } else {
            config.tensor_registry.resolve(TensorName::FfnGate, layer)
        };
        let ffn_up_name = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
            let exp_name = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
            if file.has_tensor(&exp_name) {
                exp_name
            } else {
                config.tensor_registry.resolve(TensorName::FfnUp, layer)
            }
        } else {
            config.tensor_registry.resolve(TensorName::FfnUp, layer)
        };

        let (
            ffn_gate,
            ffn_gate_meta,
            ffn_gate_svd,
            ffn_up,
            ffn_up_meta,
            ffn_up_svd,
            ffn_gate_up_interleaved,
            ffn_gate_up_interleaved_tile4,
        ) = if file.has_tensor(&ffn_gate_up_name) {
            let ffn_gate_up_view = file
                .tensor(&ffn_gate_up_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", ffn_gate_up_name),
                })?;

            let gate_tr = compute_transpose_flag(
                &ffn_gate_name,
                ffn_gate_up_view.dims,
                GgmlType::Q4_0,
                config,
                false,
                false,
            );
            let up_tr = compute_transpose_flag(
                &ffn_up_name,
                ffn_gate_up_view.dims,
                GgmlType::Q4_0,
                config,
                false,
                false,
            );

            let ffn_gate_meta = WeightMeta {
                wtype: GgmlType::Q4_0,
                dims: ffn_gate_up_view.dims.to_vec(),
                needs_transpose: gate_tr,
                role: TensorRole::Generic,
                svd_k: None,
            };
            let ffn_up_meta = WeightMeta {
                wtype: GgmlType::Q4_0,
                dims: ffn_gate_up_view.dims.to_vec(),
                needs_transpose: up_tr,
                role: TensorRole::Generic,
                svd_k: None,
            };

            let (ffn_gate, ffn_up, ffn_gate_up_interleaved, ffn_gate_up_interleaved_tile4) =
                match ffn_gate_up_view.wtype {
                    RfmType::Q4FusedGateUp => {
                        let num_blocks = ffn_gate_up_view.element_count() / 32;
                        let hidden_size = ffn_gate_up_view.dims[0] as usize;
                        let intermediate_size = ffn_gate_up_view.dims[1] as usize;

                        let raw_gpu_buf =
                            upload_tensor_bytes_for_device(ffn_gate_up_view.data, device_id)?;

                        let ffn_gate = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                        let ffn_up = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                        let ffn_gate_up_interleaved =
                            GpuBuffer::alloc_for_device(num_blocks * 36, device_id)?;
                        let ffn_gate_up_interleaved_tile4 =
                            GpuBuffer::alloc_for_device(num_blocks * 36, device_id)?;

                        quant::gpu_unpack_q4_fused_gate_up(
                            raw_gpu_buf.as_ptr() as *const u8,
                            ffn_gate.as_ptr() as *mut u8,
                            ffn_up.as_ptr() as *mut u8,
                            ffn_gate_up_interleaved.as_ptr() as *mut u8,
                            ffn_gate_up_interleaved_tile4.as_ptr() as *mut u8,
                            intermediate_size,
                            hidden_size,
                            hipStream_t::null(),
                        )?;

                        (
                            ffn_gate,
                            ffn_up,
                            Some(ffn_gate_up_interleaved),
                            Some(ffn_gate_up_interleaved_tile4),
                        )
                    }
                    _ => {
                        let (gate_data, up_data) = unpack_q4_fused_gate_up(
                            ffn_gate_up_view.data,
                            ffn_gate_up_view.element_count(),
                        );
                        let ffn_gate = upload_tensor_bytes_for_device(&gate_data, device_id)?;
                        let ffn_up = upload_tensor_bytes_for_device(&up_data, device_id)?;

                        let ffn_gate_up_interleaved = try_build_q4_0_gate_up_interleaved(
                            &gate_data,
                            &ffn_gate_meta,
                            &up_data,
                            &ffn_up_meta,
                        )
                        .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
                        .transpose()?;

                        let ffn_gate_up_interleaved_tile4 =
                            try_build_q4_0_gate_up_interleaved_tile4(
                                &gate_data,
                                &ffn_gate_meta,
                                &up_data,
                                &ffn_up_meta,
                            )
                            .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
                            .transpose()?;

                        (
                            ffn_gate,
                            ffn_up,
                            ffn_gate_up_interleaved,
                            ffn_gate_up_interleaved_tile4,
                        )
                    }
                };
            (
                ffn_gate,
                ffn_gate_meta,
                None,
                ffn_up,
                ffn_up_meta,
                None,
                ffn_gate_up_interleaved,
                ffn_gate_up_interleaved_tile4,
            )
        } else {
            // If fused weight not present, load gate and up separately (which will load their SVD corrections!)
            let gate_view = file
                .tensor(&ffn_gate_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", ffn_gate_name),
                })?;
            let up_view = file
                .tensor(&ffn_up_name)
                .map_err(|e| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor error: {}", e),
                })?
                .ok_or_else(|| GpuError::HipApiError {
                    code: -1,
                    description: format!("tensor not found: {}", ffn_up_name),
                })?;

            let gate_tr = compute_transpose_flag(
                &ffn_gate_name,
                gate_view.dims,
                rfm_type_to_ggml(&gate_view.wtype),
                config,
                false,
                false,
            );
            let up_tr = compute_transpose_flag(
                &ffn_up_name,
                up_view.dims,
                rfm_type_to_ggml(&up_view.wtype),
                config,
                false,
                false,
            );

            let (ffn_gate, ffn_gate_meta, ffn_gate_svd) = load_rfm_weight(&ffn_gate_name, gate_tr)?;
            let (ffn_up, ffn_up_meta, ffn_up_svd) = load_rfm_weight(&ffn_up_name, up_tr)?;

            (
                ffn_gate,
                ffn_gate_meta,
                ffn_gate_svd,
                ffn_up,
                ffn_up_meta,
                ffn_up_svd,
                None,
                None,
            )
        };

        // FFN down
        let ffn_down_name = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE)
        {
            let exp_name = config
                .tensor_registry
                .resolve(TensorName::FfnDownExps, layer);
            if file.has_tensor(&exp_name) {
                exp_name
            } else {
                config.tensor_registry.resolve(TensorName::FfnDown, layer)
            }
        } else {
            config.tensor_registry.resolve(TensorName::FfnDown, layer)
        };
        let ffn_down_view = file
            .tensor(&ffn_down_name)
            .map_err(|e| GpuError::HipApiError {
                code: -1,
                description: format!("tensor error: {}", e),
            })?
            .ok_or_else(|| GpuError::HipApiError {
                code: -1,
                description: format!("tensor not found: {}", ffn_down_name),
            })?;
        let down_tr = compute_transpose_flag(
            &ffn_down_name,
            ffn_down_view.dims,
            rfm_type_to_ggml(&ffn_down_view.wtype),
            config,
            false,
            false,
        );
        let (ffn_down, ffn_down_meta, ffn_down_svd) = load_rfm_weight(&ffn_down_name, down_tr)?;

        let moe = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
            let router_name = config
                .tensor_registry
                .resolve(TensorName::FfnGateInp, layer);
            if file.has_tensor(&router_name) {
                let router_view = file
                    .tensor(&router_name)
                    .map_err(|e| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor error: {}", e),
                    })?
                    .ok_or_else(|| GpuError::HipApiError {
                        code: -1,
                        description: format!("tensor not found: {}", router_name),
                    })?;
                let router_tr = compute_transpose_flag(
                    &router_name,
                    router_view.dims,
                    rfm_type_to_ggml(&router_view.wtype),
                    config,
                    false,
                    false,
                );
                let (router, router_meta, router_svd) = load_rfm_weight(&router_name, router_tr)?;

                let load_optional = |name: &str| -> GpuResult<(
                    Option<GpuBuffer>,
                    Option<WeightMeta>,
                    Option<SvdCorrection>,
                )> {
                    if !file.has_tensor(name) {
                        return Ok((None, None, None));
                    }
                    let view = file
                        .tensor(name)
                        .map_err(|e| GpuError::HipApiError {
                            code: -1,
                            description: format!("tensor error: {}", e),
                        })?
                        .ok_or_else(|| GpuError::HipApiError {
                            code: -1,
                            description: format!("tensor not found: {}", name),
                        })?;
                    let tr = compute_transpose_flag(
                        name,
                        view.dims,
                        rfm_type_to_ggml(&view.wtype),
                        config,
                        false,
                        false,
                    );
                    let (buf, meta, svd) = load_rfm_weight(name, tr)?;
                    Ok((Some(buf), Some(meta), svd))
                };

                let (shared_gate, shared_gate_meta, shared_gate_svd) =
                    load_optional(&format!("blk.{}.ffn_gate_shexp.weight", layer))?;
                let (shared_up, shared_up_meta, shared_up_svd) =
                    load_optional(&format!("blk.{}.ffn_up_shexp.weight", layer))?;
                let (shared_down, shared_down_meta, shared_down_svd) =
                    load_optional(&format!("blk.{}.ffn_down_shexp.weight", layer))?;
                let (shared_gate_inp, shared_gate_inp_meta, _) =
                    load_optional(&format!("blk.{}.ffn_gate_inp_shexp.weight", layer))?;

                Some(GpuMoeWeights {
                    router,
                    router_meta,
                    router_svd,
                    shared_gate,
                    shared_gate_meta,
                    shared_gate_svd,
                    shared_up,
                    shared_up_meta,
                    shared_up_svd,
                    shared_down,
                    shared_down_meta,
                    shared_down_svd,
                    shared_gate_inp,
                    shared_gate_inp_meta,
                })
            } else {
                None
            }
        } else {
            None
        };

        // RMS norms
        let attn_norm_name = format!("blk.{}.attn_norm.weight", layer);
        let ffn_norm_name = qwen35_post_attention_norm_name(config, layer)
            .unwrap_or_else(|| format!("blk.{}.ffn_norm.weight", layer));

        let attn_norm = load_rfm_norm(&attn_norm_name)?;
        let ffn_norm = load_rfm_norm(&ffn_norm_name)?;

        // Extract sparse CSR / MPO metadata from loaded buffers for FFN weights
        let ffn_gate_sparse = try_load_sparse_csr(&file, &ffn_gate_name, device_id)?;
        let ffn_up_sparse = try_load_sparse_csr(&file, &ffn_up_name, device_id)?;
        let ffn_down_sparse = try_load_sparse_csr(&file, &ffn_down_name, device_id)?;
        let ffn_gate_mpo = try_load_mpo(&file, &ffn_gate_name, device_id)?;
        let ffn_up_mpo = try_load_mpo(&file, &ffn_up_name, device_id)?;
        let ffn_down_mpo = try_load_mpo(&file, &ffn_down_name, device_id)?;

        Ok(Self {
            attn_norm,
            attn_q,
            attn_q_meta,
            attn_q_svd,
            attn_q_norm,
            attn_q_bias,
            attn_k,
            attn_k_meta,
            attn_k_svd,
            attn_k_norm,
            attn_k_bias,
            attn_v,
            attn_v_meta,
            attn_v_svd,
            attn_v_bias,
            attn_qkv,
            attn_qkv_meta,
            attn_gate,
            attn_gate_meta,
            ssm,
            attn_o,
            attn_o_meta,
            attn_o_svd,
            ffn_norm,
            ffn_gate,
            ffn_gate_meta,
            ffn_gate_svd,
            ffn_up,
            ffn_up_meta,
            ffn_up_svd,
            ffn_gate_up_interleaved,
            ffn_gate_up_interleaved_tile4,
            ffn_down,
            ffn_down_meta,
            ffn_down_svd,
            moe,
            ffn_gate_sparse,
            ffn_up_sparse,
            ffn_down_sparse,
            ffn_gate_mpo,
            ffn_up_mpo,
            ffn_down_mpo,
        })
    }
}
