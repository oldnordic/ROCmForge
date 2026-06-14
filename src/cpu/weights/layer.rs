use super::helpers::{copy_f32, rfm_type_to_ggml, rfm_weight_meta, unpack_q4_split};
use super::meta::{WeightError, WeightMeta};
use super::shortconv_moe::{
    load_moe_gguf, load_moe_rfm, load_shortconv_gguf, load_shortconv_rfm, CpuMoeWeights,
    CpuShortconvWeights,
};
use super::ssm::{
    load_qwen35_ssm_gguf, load_qwen35_ssm_rfm, qwen35_post_attention_norm_name, CpuSsmWeights,
};
use crate::config::{ModelConfig, TensorName, TensorRole};
use crate::cpu::transpose::compute_transpose_flag;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType};

#[derive(Clone, Debug)]
pub struct CpuLayerWeights {
    /// Whether this layer uses attention (true) or shortconv (false).
    pub is_attention_layer: bool,
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<u8>,
    pub attn_q_meta: WeightMeta,
    pub attn_k: Vec<u8>,
    pub attn_k_meta: WeightMeta,
    pub attn_v: Vec<u8>,
    pub attn_v_meta: WeightMeta,
    pub attn_o: Vec<u8>,
    pub attn_o_meta: WeightMeta,
    pub attn_qkv: Option<Vec<u8>>,
    pub attn_qkv_meta: Option<WeightMeta>,
    pub attn_gate: Option<Vec<u8>>,
    pub attn_gate_meta: Option<WeightMeta>,
    pub attn_q_bias: Option<Vec<f32>>,
    pub attn_k_bias: Option<Vec<f32>>,
    pub attn_v_bias: Option<Vec<f32>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Option<Vec<u8>>,
    pub ffn_gate_meta: Option<WeightMeta>,
    pub ffn_up: Vec<u8>,
    pub ffn_up_meta: WeightMeta,
    pub ffn_down: Vec<u8>,
    pub ffn_down_meta: WeightMeta,
    pub ssm: Option<CpuSsmWeights>,
    pub shortconv: Option<CpuShortconvWeights>,
    pub moe: Option<CpuMoeWeights>,
    pub weight_type: GgmlType,
}

impl CpuLayerWeights {
    /// Load layer weights from an open GGUF model file.
    pub fn load(file: &GgufFile, layer: usize, config: &ModelConfig) -> Result<Self, WeightError> {
        let load_weight = |name_enum: TensorName| -> Result<(Vec<u8>, WeightMeta), WeightError> {
            let name = config.tensor_registry.resolve(name_enum, layer);
            let t = file
                .tensor(&name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            let role = TensorRole::from_name(&name, false, false);
            let needs_transpose = compute_transpose_flag(role, t.dims, t.ggml_type, config);
            Ok((
                t.data.to_vec(),
                WeightMeta::from_view_with_role(&t, needs_transpose, role),
            ))
        };

        let load_opt_name = |name: &str| -> Result<Option<(Vec<u8>, WeightMeta)>, WeightError> {
            if let Some(t) = file.tensor(name).map_err(WeightError::Load)? {
                let role = TensorRole::from_name(name, false, false);
                let needs_transpose = compute_transpose_flag(role, t.dims, t.ggml_type, config);
                Ok(Some((
                    t.data.to_vec(),
                    WeightMeta::from_view_with_role(&t, needs_transpose, role),
                )))
            } else {
                Ok(None)
            }
        };

        let load_opt =
            |name_enum: TensorName| -> Result<Option<(Vec<u8>, WeightMeta)>, WeightError> {
                let name = config.tensor_registry.resolve(name_enum, layer);
                load_opt_name(&name)
            };

        let copy_f32_opt = |name_enum: TensorName| -> Result<Option<Vec<f32>>, WeightError> {
            let name_opt = config.tensor_registry.resolve_optional(name_enum, layer);
            if let Some(name) = name_opt {
                if file.tensor(&name).map_err(WeightError::Load)?.is_some() {
                    copy_f32(file, &name).map(Some)
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        };

        // ── Layer type detection ────────────────────────────────────────────────
        // Attention layer if attn_k tensor exists; otherwise shortconv (LFM2).
        let is_attention_layer = file
            .tensor(&config.tensor_registry.resolve(TensorName::AttnK, layer))
            .map_err(WeightError::Load)?
            .is_some();

        // ── Attention / Shortconv weights ─────────────────────────────────────────
        let (
            attn_q,
            attn_q_meta,
            attn_k,
            attn_k_meta,
            attn_v,
            attn_v_meta,
            attn_qkv,
            attn_qkv_meta,
            attn_o,
            attn_o_meta,
            shortconv,
        ) = if is_attention_layer {
            let qkv_name = format!("blk.{}.attn_qkv.weight", layer);
            let layer_has_fused_qkv = matches!(
                config.attention_layout,
                crate::config::AttentionLayout::FusedQkv
            ) && file.has_tensor(&qkv_name);

            if layer_has_fused_qkv {
                let (qkv_buf, qkv_meta) = load_opt_name(&qkv_name)?.ok_or_else(|| {
                    WeightError::TensorNotFound(format!(
                        "Fused QKV tensor {} not found despite has_tensor check",
                        qkv_name
                    ))
                })?;
                (
                    vec![],
                    qkv_meta.clone(),
                    vec![],
                    qkv_meta.clone(),
                    vec![],
                    qkv_meta.clone(),
                    Some(qkv_buf),
                    Some(qkv_meta),
                    vec![],
                    WeightMeta::default(),
                    None,
                )
            } else {
                let (aq, aq_meta) = load_weight(TensorName::AttnQ)?;
                let (ak, ak_meta) = load_weight(TensorName::AttnK)?;
                let (av, av_meta) = load_weight(TensorName::AttnV)?;
                let (ao, ao_meta) = load_weight(TensorName::AttnOutput)?;
                (
                    aq, aq_meta, ak, ak_meta, av, av_meta, None, None, ao, ao_meta, None,
                )
            }
        } else {
            (
                vec![],
                WeightMeta::default(),
                vec![],
                WeightMeta::default(),
                vec![],
                WeightMeta::default(),
                None,
                None,
                vec![],
                WeightMeta::default(),
                Some(load_shortconv_gguf(file, layer)?),
            )
        };

        // ── FFN weights (dense vs MoE) ────────────────────────────────────────────
        let is_moe_layer = if let Some(name) = config
            .tensor_registry
            .resolve_optional(TensorName::FfnGateExps, layer)
        {
            file.tensor(&name).map_err(WeightError::Load)?.is_some()
        } else {
            false
        };

        let (ffn_gate, ffn_gate_meta, ffn_up, ffn_up_meta, ffn_down, ffn_down_meta, moe) =
            if is_moe_layer {
                let moe_weights = load_moe_gguf(file, layer, config)?;
                (
                    None,
                    None,
                    vec![],
                    WeightMeta::default(),
                    vec![],
                    WeightMeta::default(),
                    Some(moe_weights),
                )
            } else {
                let ffn_gate_opt = load_opt(TensorName::FfnGate)?;
                let (fu, fu_meta) = load_weight(TensorName::FfnUp)?;
                let (fd, fd_meta) = load_weight(TensorName::FfnDown)?;
                (
                    ffn_gate_opt.as_ref().map(|(d, _)| d.clone()),
                    ffn_gate_opt.as_ref().map(|(_, m)| m.clone()),
                    fu,
                    fu_meta,
                    fd,
                    fd_meta,
                    None,
                )
            };

        let weight_type = if is_attention_layer && !attn_q_meta.is_empty() {
            attn_q_meta.wtype
        } else if let Some(ref sc) = shortconv {
            sc.in_proj_meta.wtype
        } else {
            ffn_up_meta.wtype
        };

        let ssm = if file
            .tensor(&format!("blk.{}.ssm_conv1d.weight", layer))
            .map_err(WeightError::Load)?
            .is_some()
        {
            Some(load_qwen35_ssm_gguf(file, layer)?)
        } else {
            None
        };

        let gate = load_opt_name(&format!("blk.{}.attn_gate.weight", layer))?;

        Ok(CpuLayerWeights {
            is_attention_layer,
            ssm,
            shortconv,
            moe,
            attn_norm: copy_f32(
                file,
                &config.tensor_registry.resolve(TensorName::AttnNorm, layer),
            )?,
            attn_q,
            attn_q_meta,
            attn_k,
            attn_k_meta,
            attn_v,
            attn_v_meta,
            attn_o,
            attn_o_meta,
            attn_qkv,
            attn_qkv_meta,
            attn_gate: gate.as_ref().map(|(d, _)| d.clone()),
            attn_gate_meta: gate.as_ref().map(|(_, m)| m.clone()),
            attn_q_bias: copy_f32_opt(TensorName::AttnQBias)?,
            attn_k_bias: copy_f32_opt(TensorName::AttnKBias)?,
            attn_v_bias: copy_f32_opt(TensorName::AttnVBias)?,
            attn_q_norm: copy_f32_opt(TensorName::AttnQNorm)?,
            attn_k_norm: copy_f32_opt(TensorName::AttnKNorm)?,
            ffn_norm: copy_f32(
                file,
                &qwen35_post_attention_norm_name(config, layer)
                    .unwrap_or_else(|| config.tensor_registry.resolve(TensorName::FfnNorm, layer)),
            )?,
            ffn_gate,
            ffn_gate_meta,
            ffn_up,
            ffn_up_meta,
            ffn_down,
            ffn_down_meta,
            weight_type,
        })
    }

    /// Load layer weights from an open RFM model file.
    pub fn load_rfm(
        file: &RfmFile,
        layer: usize,
        config: &ModelConfig,
    ) -> Result<Self, WeightError> {
        let load_rfm_weight = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
            let t = file
                .tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;

            let data = match t.wtype {
                RfmType::F32 | RfmType::Mq4 | RfmType::Mq6 => t.data.to_vec(),
                RfmType::Q4Split | RfmType::Q4SvdQuant { .. } => {
                    unpack_q4_split(t.data, t.element_count())
                }
                RfmType::GgufPassthrough(_) => t.data.to_vec(),
                RfmType::MoeExpertSvdSparse { rows, cols, .. }
                | RfmType::MoeExpertSvdFwhtSparse { rows, cols, .. }
                | RfmType::MoeExpertMpo { rows, cols, .. } => {
                    vec![0u8; (rows * cols) as usize * 4]
                }
                _ => {
                    return Err(WeightError::Load(
                        crate::loader::LoadError::UnknownTensorType(999),
                    ))
                }
            };

            let role = TensorRole::from_name(t.name, false, false);
            let needs_transpose =
                compute_transpose_flag(role, t.dims, rfm_type_to_ggml(&t.wtype), config);
            let mut meta = rfm_weight_meta(&t, needs_transpose);
            meta.role = role;
            Ok((data, meta))
        };

        let load_rfm_opt = |name: &str| -> Result<Option<(Vec<u8>, WeightMeta)>, WeightError> {
            if file.tensor(name).map_err(WeightError::Load)?.is_some() {
                load_rfm_weight(name).map(Some)
            } else {
                Ok(None)
            }
        };

        let load_rfm_f32 = |name: &str| -> Result<Vec<f32>, WeightError> {
            let t = file
                .tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            match t.wtype {
                RfmType::F32 => Ok(super::helpers::copy_f32_from_bytes(t.data)),
                _ => Err(WeightError::Load(
                    crate::loader::LoadError::UnknownTensorType(0),
                )),
            }
        };

        let load_rfm_f32_opt = |name_enum: TensorName| -> Result<Option<Vec<f32>>, WeightError> {
            let name_opt = config.tensor_registry.resolve_optional(name_enum, layer);
            if let Some(name) = name_opt {
                if file.tensor(&name).map_err(WeightError::Load)?.is_some() {
                    load_rfm_f32(&name).map(Some)
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        };

        // ── Layer type detection ────────────────────────────────────────────────
        let is_attention_layer = file
            .tensor(&config.tensor_registry.resolve(TensorName::AttnK, layer))
            .map_err(WeightError::Load)?
            .is_some();

        // ── Attention / Shortconv weights ───────────────────────────────────────
        let (
            attn_q,
            attn_q_meta,
            attn_k,
            attn_k_meta,
            attn_v,
            attn_v_meta,
            attn_qkv,
            attn_qkv_meta,
            attn_o,
            attn_o_meta,
            shortconv,
        ) = if is_attention_layer {
            let (aq, aq_meta) =
                load_rfm_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
            let (ak, ak_meta) =
                load_rfm_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
            let (av, av_meta) =
                load_rfm_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
            let (ao, ao_meta) = load_rfm_weight(
                &config
                    .tensor_registry
                    .resolve(TensorName::AttnOutput, layer),
            )?;
            let qkv = load_rfm_opt(&format!("blk.{}.attn_qkv.weight", layer))?;
            (
                aq,
                aq_meta,
                ak,
                ak_meta,
                av,
                av_meta,
                qkv.as_ref().map(|(d, _)| d.clone()),
                qkv.as_ref().map(|(_, m)| m.clone()),
                ao,
                ao_meta,
                None,
            )
        } else {
            (
                vec![],
                WeightMeta::default(),
                vec![],
                WeightMeta::default(),
                vec![],
                WeightMeta::default(),
                None,
                None,
                vec![],
                WeightMeta::default(),
                Some(load_shortconv_rfm(file, layer)?),
            )
        };

        // ── FFN weights (dense vs MoE) ──────────────────────────────────────────
        let is_moe_layer = if let Some(name) = config
            .tensor_registry
            .resolve_optional(TensorName::FfnGateExps, layer)
        {
            file.tensor(&name).map_err(WeightError::Load)?.is_some()
        } else {
            false
        };

        let (ffn_gate, ffn_gate_meta, ffn_up, ffn_up_meta, ffn_down, ffn_down_meta, moe) =
            if is_moe_layer {
                let moe_weights = load_moe_rfm(file, layer, config)?;
                (
                    None,
                    None,
                    vec![],
                    WeightMeta::default(),
                    vec![],
                    WeightMeta::default(),
                    Some(moe_weights),
                )
            } else {
                let ffn_gate_opt =
                    load_rfm_opt(&config.tensor_registry.resolve(TensorName::FfnGate, layer))?;
                let (fu, fu_meta) =
                    load_rfm_weight(&config.tensor_registry.resolve(TensorName::FfnUp, layer))?;
                let (fd, fd_meta) =
                    load_rfm_weight(&config.tensor_registry.resolve(TensorName::FfnDown, layer))?;
                (
                    ffn_gate_opt.as_ref().map(|(d, _)| d.clone()),
                    ffn_gate_opt.as_ref().map(|(_, m)| m.clone()),
                    fu,
                    fu_meta,
                    fd,
                    fd_meta,
                    None,
                )
            };

        let weight_type = if is_attention_layer && !attn_q_meta.is_empty() {
            attn_q_meta.wtype
        } else if let Some(ref sc) = shortconv {
            sc.in_proj_meta.wtype
        } else {
            ffn_up_meta.wtype
        };

        let ssm = if file
            .tensor(&format!("blk.{}.ssm_conv1d.weight", layer))
            .map_err(WeightError::Load)?
            .is_some()
        {
            Some(load_qwen35_ssm_rfm(file, layer)?)
        } else {
            None
        };

        let gate = load_rfm_opt(&format!("blk.{}.attn_gate.weight", layer))?;

        Ok(CpuLayerWeights {
            is_attention_layer,
            ssm,
            shortconv,
            moe,
            attn_norm: load_rfm_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?,
            attn_q,
            attn_q_meta: attn_q_meta.clone(),
            attn_k,
            attn_k_meta,
            attn_v,
            attn_v_meta,
            attn_o,
            attn_o_meta,
            attn_qkv,
            attn_qkv_meta,
            attn_gate: gate.as_ref().map(|(d, _)| d.clone()),
            attn_gate_meta: gate.as_ref().map(|(_, m)| m.clone()),
            attn_q_bias: load_rfm_f32_opt(TensorName::AttnQBias)?,
            attn_k_bias: load_rfm_f32_opt(TensorName::AttnKBias)?,
            attn_v_bias: load_rfm_f32_opt(TensorName::AttnVBias)?,
            attn_q_norm: load_rfm_f32_opt(TensorName::AttnQNorm)?,
            attn_k_norm: load_rfm_f32_opt(TensorName::AttnKNorm)?,
            ffn_norm: load_rfm_f32(
                &qwen35_post_attention_norm_name(config, layer)
                    .unwrap_or_else(|| config.tensor_registry.resolve(TensorName::FfnNorm, layer)),
            )?,
            ffn_gate,
            ffn_gate_meta,
            ffn_up,
            ffn_up_meta,
            ffn_down,
            ffn_down_meta,
            weight_type,
        })
    }
}
