use super::helpers::{
    copy_f32, rfm_type_to_ggml,
    rfm_weight_meta, unpack_q4_split,
};
use super::meta::{WeightError, WeightMeta};
use super::ssm::{
    load_qwen35_ssm_gguf, load_qwen35_ssm_rfm, qwen35_post_attention_norm_name, CpuSsmWeights,
};
use crate::config::{ModelConfig, TensorName};
use crate::cpu::transpose::compute_transpose_flag;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType};

#[derive(Clone, Debug)]
pub struct CpuLayerWeights {
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
    pub ffn_gate: Vec<u8>,
    pub ffn_gate_meta: WeightMeta,
    pub ffn_up: Vec<u8>,
    pub ffn_up_meta: WeightMeta,
    pub ffn_down: Vec<u8>,
    pub ffn_down_meta: WeightMeta,
    pub ssm: Option<CpuSsmWeights>,
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
            let needs_transpose =
                compute_transpose_flag(&name, t.dims, t.ggml_type, config, false, false);
            Ok((t.data.to_vec(), WeightMeta::from_view(&t, needs_transpose)))
        };

        let load_opt = |name: &str| -> Result<Option<(Vec<u8>, WeightMeta)>, WeightError> {
            if let Some(t) = file.tensor(name).map_err(WeightError::Load)? {
                let needs_transpose =
                    compute_transpose_flag(name, t.dims, t.ggml_type, config, false, false);
                Ok(Some((
                    t.data.to_vec(),
                    WeightMeta::from_view(&t, needs_transpose),
                )))
            } else {
                Ok(None)
            }
        };

        let (attn_q, attn_q_meta) = load_weight(TensorName::AttnQ)?;
        let (attn_k, attn_k_meta) = load_weight(TensorName::AttnK)?;
        let (attn_v, attn_v_meta) = load_weight(TensorName::AttnV)?;
        let (attn_o, attn_o_meta) = load_weight(TensorName::AttnOutput)?;

        let (ffn_gate, ffn_gate_meta) = load_weight(TensorName::FfnGate)?;
        let (ffn_up, ffn_up_meta) = load_weight(TensorName::FfnUp)?;
        let (ffn_down, ffn_down_meta) = load_weight(TensorName::FfnDown)?;

        let weight_type = attn_q_meta.wtype;

        let ssm = if file
            .tensor(&format!("blk.{}.ssm_conv1d.weight", layer))
            .map_err(WeightError::Load)?
            .is_some()
        {
            Some(load_qwen35_ssm_gguf(file, layer)?)
        } else {
            None
        };

        let copy_f32_opt = |name_enum: TensorName| -> Result<Option<Vec<f32>>, WeightError> {
            let name = config.tensor_registry.resolve(name_enum, layer);
            if file.tensor(&name).map_err(WeightError::Load)?.is_some() {
                copy_f32(file, &name).map(Some)
            } else {
                Ok(None)
            }
        };

        let qkv = load_opt(&format!("blk.{}.attn_qkv.weight", layer))?;
        let gate = load_opt(&format!("blk.{}.attn_gate.weight", layer))?;

        Ok(CpuLayerWeights {
            ssm,
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
            attn_qkv: qkv.as_ref().map(|(d, _)| d.clone()),
            attn_qkv_meta: qkv.as_ref().map(|(_, m)| m.clone()),
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
                | RfmType::MoeExpertSvdFwhtSparse { rows, cols, .. } => {
                    vec![0u8; (rows * cols) as usize * 4]
                }
                _ => {
                    return Err(WeightError::Load(
                        crate::loader::LoadError::UnknownTensorType(999),
                    ))
                }
            };

            let needs_transpose = compute_transpose_flag(
                t.name,
                t.dims,
                rfm_type_to_ggml(&t.wtype),
                config,
                false,
                false,
            );
            let meta = rfm_weight_meta(&t, needs_transpose);
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
                RfmType::F32 => {
                    let n = t.data.len() / 4;
                    let mut out = vec![0.0f32; n];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            t.data.as_ptr() as *const f32,
                            out.as_mut_ptr(),
                            n,
                        );
                    }
                    Ok(out)
                }
                _ => Err(WeightError::Load(
                    crate::loader::LoadError::UnknownTensorType(0),
                )),
            }
        };

        let load_rfm_f32_opt = |name_enum: TensorName| -> Result<Option<Vec<f32>>, WeightError> {
            let name = config.tensor_registry.resolve(name_enum, layer);
            if file.tensor(&name).map_err(WeightError::Load)?.is_some() {
                load_rfm_f32(&name).map(Some)
            } else {
                Ok(None)
            }
        };

        let (attn_q, attn_q_meta) =
            load_rfm_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
        let (attn_k, attn_k_meta) =
            load_rfm_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
        let (attn_v, attn_v_meta) =
            load_rfm_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
        let (attn_o, attn_o_meta) = load_rfm_weight(
            &config
                .tensor_registry
                .resolve(TensorName::AttnOutput, layer),
        )?;

        let (ffn_gate, ffn_gate_meta) =
            load_rfm_weight(&config.tensor_registry.resolve(TensorName::FfnGate, layer))?;
        let (ffn_up, ffn_up_meta) =
            load_rfm_weight(&config.tensor_registry.resolve(TensorName::FfnUp, layer))?;
        let (ffn_down, ffn_down_meta) =
            load_rfm_weight(&config.tensor_registry.resolve(TensorName::FfnDown, layer))?;

        let weight_type = attn_q_meta.wtype;

        let ssm = if file
            .tensor(&format!("blk.{}.ssm_conv1d.weight", layer))
            .map_err(WeightError::Load)?
            .is_some()
        {
            Some(load_qwen35_ssm_rfm(file, layer)?)
        } else {
            None
        };

        let qkv = load_rfm_opt(&format!("blk.{}.attn_qkv.weight", layer))?;
        let gate = load_rfm_opt(&format!("blk.{}.attn_gate.weight", layer))?;

        Ok(CpuLayerWeights {
            ssm,
            attn_norm: load_rfm_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?,
            attn_q,
            attn_q_meta: attn_q_meta.clone(),
            attn_k,
            attn_k_meta,
            attn_v,
            attn_v_meta,
            attn_o,
            attn_o_meta,
            attn_qkv: qkv.as_ref().map(|(d, _)| d.clone()),
            attn_qkv_meta: qkv.as_ref().map(|(_, m)| m.clone()),
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
