use super::super::error::{GpuError, GpuResult};
use super::super::ffi::hipStream_t;
use super::super::vram_budget::active_or_default_device_id;
use super::buffer::GpuBuffer;
use super::metadata::{TensorRole, WeightMeta};
use super::upload::*;
use crate::config::{ModelConfig, TensorName, TensorNamingScheme};
use crate::cpu::transpose::compute_transpose_flag;
use crate::gpu::kernels::quant;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType};

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
    /// Optional decode-friendly interleaved Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved: Option<GpuBuffer>,
    /// Optional decode-friendly 4-column tiled Q4_0 layout for fused gate/up kernels.
    pub ffn_gate_up_interleaved_tile4: Option<GpuBuffer>,
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
            ffn_gate_up_interleaved,
            ffn_gate_up_interleaved_tile4,
            ffn_down,
            ffn_down_meta,
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
        let ffn_norm_name = config.tensor_registry.resolve(TensorName::FfnNorm, layer);
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

        Ok(tensor_bytes(&attn_norm_name)?
            + tensor_bytes(&attn_q_name)?
            + tensor_bytes_optional(
                &config
                    .tensor_registry
                    .resolve_optional(TensorName::AttnQBias, layer)
                    .unwrap_or_default(),
            )?
            + tensor_bytes(&attn_k_name)?
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
            + tensor_bytes(&ffn_norm_name)?
            + ffn_gate_bytes
            + ffn_up_bytes
            + interleaved_bytes
            + interleaved_tile4_bytes
            + ffn_down_bytes)
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
        if let Some(ref buf) = self.attn_k_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.attn_v_bias {
            total += buf.size();
        }
        if let Some(ref buf) = self.ffn_gate_up_interleaved {
            total += buf.size();
        }
        if let Some(ref buf) = self.ffn_gate_up_interleaved_tile4 {
            total += buf.size();
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
            |name: &str, needs_transpose: bool| -> GpuResult<(GpuBuffer, WeightMeta)> {
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
                let meta = WeightMeta {
                    wtype,
                    dims: t.dims.to_vec(),
                    needs_transpose,
                    role: TensorRole::Generic,
                };

                match t.wtype {
                    RfmType::Q4Split => {
                        let raw_gpu_buf = upload_tensor_bytes_for_device(t.data, device_id)?;
                        let num_blocks = t.element_count() / 32;
                        let out_gpu_buf = GpuBuffer::alloc_for_device(num_blocks * 18, device_id)?;
                        quant::gpu_unpack_q4_split(
                            raw_gpu_buf.as_ptr() as *const u8,
                            out_gpu_buf.as_ptr() as *mut u8,
                            num_blocks,
                            hipStream_t::null(),
                        )?;
                        Ok((out_gpu_buf, meta))
                    }
                    _ => {
                        let buf = upload_tensor_bytes_for_device(t.data, device_id)?;
                        Ok((buf, meta))
                    }
                }
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
        let o_tr = compute_transpose_flag(
            &o_name,
            o_view.dims,
            rfm_type_to_ggml(&o_view.wtype),
            config,
            false,
            false,
        );

        let (attn_q, attn_q_meta) = load_rfm_weight(&q_name, q_tr)?;
        let (attn_k, attn_k_meta) = load_rfm_weight(&k_name, k_tr)?;
        let (attn_v, attn_v_meta) = load_rfm_weight(&v_name, v_tr)?;
        let (attn_o, attn_o_meta) = load_rfm_weight(&o_name, o_tr)?;

        let attn_q_bias = None;
        let attn_k_bias = None;
        let attn_v_bias = None;

        // FFN gate+up fusion
        let ffn_gate_up_name = format!("blk.{}.ffn_gate_up.weight", layer);
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

        let ffn_gate_name = format!("blk.{}.ffn_gate.weight", layer);
        let ffn_up_name = format!("blk.{}.ffn_up.weight", layer);

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
        };
        let ffn_up_meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: ffn_gate_up_view.dims.to_vec(),
            needs_transpose: up_tr,
            role: TensorRole::Generic,
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

                    let ffn_gate_up_interleaved_tile4 = try_build_q4_0_gate_up_interleaved_tile4(
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

        // FFN down
        let ffn_down_name = format!("blk.{}.ffn_down.weight", layer);
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
        let (ffn_down, ffn_down_meta) = load_rfm_weight(&ffn_down_name, down_tr)?;

        // RMS norms
        let attn_norm_name = format!("blk.{}.attn_norm.weight", layer);
        let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer);

        let attn_norm = load_rfm_norm(&attn_norm_name)?;
        let ffn_norm = load_rfm_norm(&ffn_norm_name)?;

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
            ffn_gate_up_interleaved,
            ffn_gate_up_interleaved_tile4,
            ffn_down,
            ffn_down_meta,
        })
    }
}
