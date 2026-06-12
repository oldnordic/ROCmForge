use super::super::super::error::{GpuError, GpuResult};
use super::super::buffer::GpuBuffer;
use super::super::metadata::WeightMeta;
use super::super::upload::{
    build_matrix_meta, try_build_q4_0_gate_up_interleaved,
    try_build_q4_0_gate_up_interleaved_tile4, upload_tensor_bytes_for_device,
};
use super::support::{load_qwen35_ssm_gguf, qwen35_post_attention_norm_name};
use super::{
    CpuCompressedExperts, CpuMpoExperts, GpuLayerWeights, GpuMoeWeights, GpuMpoWeights,
    GpuSparseCsrWeights,
};
use crate::config::{AttentionLayout, ModelConfig, TensorName, TensorNamingScheme};
use crate::loader::GgufFile;

pub(super) fn load_for_device(
    file: &GgufFile,
    layer: usize,
    config: &ModelConfig,
    device_id: i32,
) -> GpuResult<GpuLayerWeights> {
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

    let load_weight_fallback = |names: &[&str]| -> GpuResult<(GpuBuffer, WeightMeta)> {
        for name in names {
            match file.tensor(name) {
                Ok(Some(t)) => {
                    let meta = build_matrix_meta(name, t.dims, t.ggml_type, config, false, false)?;
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

    let load_f32_opt = |name: &str| -> GpuResult<Option<GpuBuffer>> {
        match file.tensor(name) {
            Ok(Some(t)) => {
                let mut buf = GpuBuffer::alloc_for_device(t.data.len(), device_id)?;
                buf.copy_from_host(t.data)?;
                Ok(Some(buf))
            }
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    };

    let attn_norm = load_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?;
    let qkv_name = format!("blk.{}.attn_qkv.weight", layer);
    let layer_has_fused_qkv =
        matches!(config.attention_layout, AttentionLayout::FusedQkv) && file.has_tensor(&qkv_name);
    let (attn_q, attn_q_meta, attn_k, attn_k_meta, attn_v, attn_v_meta, attn_qkv, attn_qkv_meta) =
        if layer_has_fused_qkv {
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
        let meta = attn_qkv_meta
            .as_ref()
            .ok_or_else(|| GpuError::HipApiError {
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
        if file.has_tensor(&attn_gate_name) {
            let (attn_gate, attn_gate_meta) = load_weight(&attn_gate_name)?;
            let ssm = load_qwen35_ssm_gguf(file, layer, config, device_id)?;
            (Some(attn_gate), Some(attn_gate_meta), Some(ssm))
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };
    let ffn_norm_name = qwen35_post_attention_norm_name(config, layer)
        .unwrap_or_else(|| config.tensor_registry.resolve(TensorName::FfnNorm, layer));
    let ffn_norm = load_f32(&ffn_norm_name)?;

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
        (Some(gate_t), Some(up_t)) => {
            try_build_q4_0_gate_up_interleaved(gate_t.data, &ffn_gate_meta, up_t.data, &ffn_up_meta)
                .map(|bytes| upload_tensor_bytes_for_device(&bytes, device_id))
                .transpose()?
        }
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
            let load_moe_weight_opt = |name: &str| -> GpuResult<Option<(GpuBuffer, WeightMeta)>> {
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

    Ok(GpuLayerWeights {
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
        attn_qkv_svd: None,
        attn_gate,
        attn_gate_meta,
        attn_gate_svd: None,
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
        ffn_gate_sparse: None::<GpuSparseCsrWeights>,
        ffn_up_sparse: None::<GpuSparseCsrWeights>,
        ffn_down_sparse: None::<GpuSparseCsrWeights>,
        ffn_gate_mpo: None::<GpuMpoWeights>,
        ffn_up_mpo: None::<GpuMpoWeights>,
        ffn_down_mpo: None::<GpuMpoWeights>,
        ffn_gate_mpo_experts: None::<CpuMpoExperts>,
        ffn_up_mpo_experts: None::<CpuMpoExperts>,
        ffn_down_mpo_experts: None::<CpuMpoExperts>,
        ffn_gate_compressed: None::<CpuCompressedExperts>,
        ffn_up_compressed: None::<CpuCompressedExperts>,
        ffn_down_compressed: None::<CpuCompressedExperts>,
    })
}
