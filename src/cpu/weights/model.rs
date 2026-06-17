use super::helpers::{
    copy_f32, copy_f32_from_bytes, copy_tensor_with_meta, rfm_type_to_ggml, rfm_weight_meta,
};
use super::layer::CpuLayerWeights;
use super::meta::{WeightError, WeightMeta};
use crate::config::{ModelConfig, TensorName, TensorRole};
use crate::loader::{GgufFile, RfmFile};

#[derive(Clone, Debug)]
pub struct CpuModelWeights {
    pub token_emb: Vec<u8>,
    pub token_emb_meta: WeightMeta,
    pub output_norm: Vec<f32>,
    pub output: Vec<u8>,
    pub output_meta: WeightMeta,
    pub layers: Vec<CpuLayerWeights>,
    pub lm_head: Vec<u8>,
    pub lm_head_meta: WeightMeta,
    pub lm_head_tied: bool,
    // Gemma4 Per-Layer Embedding (PLE) model-level tensors (optional, only for gemma4)
    pub per_layer_token_emb: Option<(Vec<u8>, WeightMeta)>,
    pub per_layer_proj_norm: Option<Vec<f32>>,
    pub per_layer_model_proj: Option<(Vec<u8>, WeightMeta)>,
}

impl CpuModelWeights {
    /// Load all model weights from an open GGUF model file.
    pub fn load(file: &GgufFile, config: &ModelConfig) -> Result<Self, WeightError> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(CpuLayerWeights::load(file, i, config)?);
        }

        let (token_emb, token_emb_meta) = copy_tensor_with_meta(
            file,
            &config.tensor_registry.resolve(TensorName::TokenEmb, 0),
            false,
        )?;

        let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
        let output_norm = copy_f32(file, &output_norm_name)
            .or_else(|_| copy_f32(file, "token_embd_norm.weight"))
            .map_err(|_| WeightError::TensorNotFound(output_norm_name))?;

        // lm_head
        let (lm_head, lm_head_meta, lm_head_tied) = if let Some(view) = file
            .tensor(&config.tensor_registry.resolve(TensorName::LmHead, 0))
            .map_err(WeightError::Load)?
        {
            (
                view.data.to_vec(),
                WeightMeta::from_view_with_role(&view, false, TensorRole::LmHead),
                false,
            )
        } else {
            (token_emb.clone(), token_emb_meta.clone(), true)
        };

        // Gemma4-specific model-level tensors
        let (per_layer_token_emb, per_layer_proj_norm, per_layer_model_proj) =
            if config.architecture == "gemma4" {
                let ple_tok = copy_tensor_with_meta(
                    file,
                    &config
                        .tensor_registry
                        .resolve(TensorName::PerLayerTokenEmb, 0),
                    false,
                )?;
                let ple_norm = copy_f32(
                    file,
                    &config
                        .tensor_registry
                        .resolve(TensorName::PerLayerProjNorm, 0),
                )?;
                let ple_proj = copy_tensor_with_meta(
                    file,
                    &config
                        .tensor_registry
                        .resolve(TensorName::PerLayerModelProj, 0),
                    false,
                )?;
                (Some(ple_tok), Some(ple_norm), Some(ple_proj))
            } else {
                (None, None, None)
            };

        Ok(CpuModelWeights {
            token_emb,
            token_emb_meta,
            output_norm,
            output: lm_head.clone(),
            output_meta: lm_head_meta.clone(),
            layers,
            lm_head,
            lm_head_meta,
            lm_head_tied,
            per_layer_token_emb,
            per_layer_proj_norm,
            per_layer_model_proj,
        })
    }

    pub fn layer(&self, layer: usize) -> &CpuLayerWeights {
        &self.layers[layer]
    }

    /// Load all model weights from an open RFM model file.
    pub fn load_rfm(file: &RfmFile, config: &ModelConfig) -> Result<Self, WeightError> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(CpuLayerWeights::load_rfm(file, i, config)?);
        }

        let load_rfm_f32 = |name: &str| -> Result<Vec<f32>, WeightError> {
            let t = file
                .tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            Ok(crate::cpu::weights::helpers::copy_f32_from_bytes(t.data))
        };

        let load_rfm_u8_meta = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
            let t = file
                .tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            let role = TensorRole::from_name(t.name, false, false);
            let mut meta = WeightMeta {
                wtype: rfm_type_to_ggml(&t.wtype),
                dims: t.dims.to_vec(),
                needs_transpose: false,
                role,
                svd_k: None,
            };
            if let crate::loader::RfmType::Q4SvdQuant { k, .. } = t.wtype {
                meta.svd_k = Some(k);
            }
            Ok((t.data.to_vec(), meta))
        };

        let (token_emb, token_emb_meta) =
            load_rfm_u8_meta(&config.tensor_registry.resolve(TensorName::TokenEmb, 0))?;

        let (lm_head, lm_head_meta, lm_head_tied) = if let Some(t) = file
            .tensor(&config.tensor_registry.resolve(TensorName::LmHead, 0))
            .map_err(WeightError::Load)?
        {
            let (data, meta) = load_rfm_u8_meta(t.name)?;
            (data, meta, false)
        } else {
            (token_emb.clone(), token_emb_meta.clone(), true)
        };

        // Gemma4-specific model-level tensors
        let (per_layer_token_emb, per_layer_proj_norm, per_layer_model_proj) =
            if config.architecture == "gemma4" {
                let ple_tok = load_rfm_u8_meta(
                    &config
                        .tensor_registry
                        .resolve(TensorName::PerLayerTokenEmb, 0),
                )?;
                let ple_norm = load_rfm_f32(
                    &config
                        .tensor_registry
                        .resolve(TensorName::PerLayerProjNorm, 0),
                )?;
                let ple_proj_t = file
                    .tensor(
                        &config
                            .tensor_registry
                            .resolve(TensorName::PerLayerModelProj, 0),
                    )
                    .map_err(WeightError::Load)?
                    .ok_or_else(|| {
                        WeightError::TensorNotFound(
                            config
                                .tensor_registry
                                .resolve(TensorName::PerLayerModelProj, 0),
                        )
                    })?;
                let mut ple_proj_meta = rfm_weight_meta(&ple_proj_t, false);
                let ple_proj = match ple_proj_t.wtype {
                    crate::loader::RfmType::F32 => ple_proj_t.data.to_vec(),
                    crate::loader::RfmType::GgufPassthrough(_) => ple_proj_t.data.to_vec(),
                    _ => {
                        // Fallback: dequantize to f32 and store as bytes.
                        ple_proj_meta.wtype = crate::loader::GgmlType::F32;
                        let f32s = copy_f32_from_bytes(ple_proj_t.data);
                        let mut bytes = Vec::with_capacity(f32s.len() * 4);
                        for v in f32s {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        bytes
                    }
                };
                (
                    Some(ple_tok),
                    Some(ple_norm),
                    Some((ple_proj, ple_proj_meta)),
                )
            } else {
                (None, None, None)
            };

        Ok(CpuModelWeights {
            token_emb,
            token_emb_meta,
            output_norm: load_rfm_f32(&config.tensor_registry.resolve(TensorName::OutputNorm, 0))
                .or_else(|_| load_rfm_f32("token_embd_norm.weight"))
                .map_err(|_| {
                    WeightError::TensorNotFound(
                        config.tensor_registry.resolve(TensorName::OutputNorm, 0),
                    )
                })?,
            output: lm_head.clone(),
            output_meta: lm_head_meta.clone(),
            layers,
            lm_head,
            lm_head_meta,
            lm_head_tied,
            per_layer_token_emb,
            per_layer_proj_norm,
            per_layer_model_proj,
        })
    }
}
