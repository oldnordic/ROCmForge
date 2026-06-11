use super::helpers::{copy_f32, copy_tensor_with_meta, rfm_type_to_ggml};
use super::layer::CpuLayerWeights;
use super::meta::{WeightError, WeightMeta};
use crate::config::{ModelConfig, TensorName};
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

        let output_norm = copy_f32(
            file,
            &config.tensor_registry.resolve(TensorName::OutputNorm, 0),
        )?;

        // lm_head
        let (lm_head, lm_head_meta, lm_head_tied) = if let Some(view) = file
            .tensor(&config.tensor_registry.resolve(TensorName::LmHead, 0))
            .map_err(WeightError::Load)?
        {
            (
                view.data.to_vec(),
                WeightMeta::from_view(&view, false),
                false,
            )
        } else {
            (token_emb.clone(), token_emb_meta.clone(), true)
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
            let n = t.data.len() / 4;
            let mut out = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(t.data.as_ptr() as *const f32, out.as_mut_ptr(), n);
            }
            Ok(out)
        };

        let load_rfm_u8_meta = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
            let t = file
                .tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            let mut meta = WeightMeta {
                wtype: rfm_type_to_ggml(&t.wtype),
                dims: t.dims.to_vec(),
                needs_transpose: false,
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

        Ok(CpuModelWeights {
            token_emb,
            token_emb_meta,
            output_norm: load_rfm_f32(&config.tensor_registry.resolve(TensorName::OutputNorm, 0))?,
            output: lm_head.clone(),
            output_meta: lm_head_meta.clone(),
            layers,
            lm_head,
            lm_head_meta,
            lm_head_tied,
        })
    }
}
