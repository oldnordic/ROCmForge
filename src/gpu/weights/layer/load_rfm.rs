use super::super::super::error::{GpuError, GpuResult};
use super::super::super::ffi::hipStream_t;
use super::super::buffer::GpuBuffer;
use super::super::metadata::{TensorRole, WeightMeta};
use super::super::upload::{
    rfm_type_to_ggml, try_build_q4_0_gate_up_interleaved, try_build_q4_0_gate_up_interleaved_tile4,
    unpack_q4_fused_gate_up, upload_tensor_bytes_for_device,
};
use super::support::{
    load_qwen35_ssm_rfm, qwen35_post_attention_norm_name, try_load_compressed_experts,
};
use super::{
    try_load_mpo, try_load_sparse_csr, CpuCompressedExperts, GpuLayerWeights, GpuMoeWeights,
    GpuMpoWeights, GpuSparseCsrWeights, SvdCorrection,
};
use crate::config::{AttentionLayout, ModelConfig, TensorName, TensorNamingScheme};
use crate::cpu::transpose::compute_transpose_flag;
use crate::gpu::kernels::quant;
use crate::loader::{GgmlType, RfmFile, RfmType};

pub(super) fn load_for_device(
    file: &RfmFile,
    layer: usize,
    config: &ModelConfig,
    device_id: i32,
) -> GpuResult<GpuLayerWeights> {
    let load_rfm_weight = |name: &str,
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
            RfmType::Q4SvdQuant { k } | RfmType::SvdSparseCsr { k, .. } => Some(k),
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
            RfmType::SparseCsr { .. } | RfmType::SvdSparseCsr { .. } | RfmType::Mpo { .. } => {
                upload_tensor_bytes_for_device(t.data, device_id)?
            }
            _ => upload_tensor_bytes_for_device(t.data, device_id)?,
        };

        let svd_corr = match t.wtype {
            RfmType::Q4SvdQuant { k } | RfmType::SvdSparseCsr { k, .. } => {
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
            }
            _ => None,
        };

        Ok((base_buf, meta, svd_corr))
    };

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
    let layer_has_fused_qkv =
        matches!(config.attention_layout, AttentionLayout::FusedQkv) && file.has_tensor(&qkv_name);

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
        attn_qkv_svd,
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
            qkv_svd,
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
            None,
        )
    };
    let (attn_o, attn_o_meta, attn_o_svd) = if file.has_tensor(&o_name) {
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
    } else {
        (
            GpuBuffer::empty(),
            WeightMeta {
                wtype: GgmlType::F32,
                dims: vec![0, 0],
                needs_transpose: false,
                role: TensorRole::Generic,
                svd_k: None,
            },
            None,
        )
    };
    let load_rfm_bias_opt = |name_opt: Option<String>| -> GpuResult<Option<GpuBuffer>> {
        match name_opt {
            Some(name) => {
                if file.has_tensor(&name) {
                    Ok(Some(load_rfm_norm(&name)?))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    };

    let attn_q_bias = load_rfm_bias_opt(
        config
            .tensor_registry
            .resolve_optional(TensorName::AttnQBias, layer),
    )?;
    let attn_k_bias = load_rfm_bias_opt(
        config
            .tensor_registry
            .resolve_optional(TensorName::AttnKBias, layer),
    )?;
    let attn_v_bias = load_rfm_bias_opt(
        config
            .tensor_registry
            .resolve_optional(TensorName::AttnVBias, layer),
    )?;

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

    let (attn_gate, attn_gate_meta, attn_gate_svd, ssm) = if layer_has_fused_qkv {
        let attn_gate_name = format!("blk.{}.attn_gate.weight", layer);
        if file.has_tensor(&attn_gate_name) {
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
            let (attn_gate, attn_gate_meta, attn_gate_svd) =
                load_rfm_weight(&attn_gate_name, attn_gate_tr)?;
            let ssm = load_qwen35_ssm_rfm(file, layer, config, device_id)?;
            (
                Some(attn_gate),
                Some(attn_gate_meta),
                attn_gate_svd,
                Some(ssm),
            )
        } else {
            (None, None, None, None)
        }
    } else {
        (None, None, None, None)
    };

    let ffn_gate_up_name = format!("blk.{}.ffn_gate_up.weight", layer);
    let ffn_gate_name = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
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

    let ffn_down_name = if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
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

    let attn_norm_name = format!("blk.{}.attn_norm.weight", layer);
    let ffn_norm_name = qwen35_post_attention_norm_name(config, layer)
        .unwrap_or_else(|| format!("blk.{}.ffn_norm.weight", layer));

    let attn_norm = load_rfm_norm(&attn_norm_name)?;
    let ffn_norm = load_rfm_norm(&ffn_norm_name)?;

    let ffn_gate_sparse = try_load_sparse_csr(file, &ffn_gate_name, device_id)?;
    let ffn_up_sparse = try_load_sparse_csr(file, &ffn_up_name, device_id)?;
    let ffn_down_sparse = try_load_sparse_csr(file, &ffn_down_name, device_id)?;
    let ffn_gate_mpo = try_load_mpo(file, &ffn_gate_name, device_id)?;
    let ffn_up_mpo = try_load_mpo(file, &ffn_up_name, device_id)?;
    let ffn_down_mpo = try_load_mpo(file, &ffn_down_name, device_id)?;
    let ffn_gate_compressed = try_load_compressed_experts(file, &ffn_gate_name)?;
    let ffn_up_compressed = try_load_compressed_experts(file, &ffn_up_name)?;
    let ffn_down_compressed = try_load_compressed_experts(file, &ffn_down_name)?;

    Ok(GpuLayerWeights {
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
        attn_qkv_svd,
        attn_gate,
        attn_gate_meta,
        attn_gate_svd,
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
        ffn_gate_compressed,
        ffn_up_compressed,
        ffn_down_compressed,
    })
}
