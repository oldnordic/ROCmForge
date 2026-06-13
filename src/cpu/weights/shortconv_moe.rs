use super::helpers::{copy_f32, copy_f32_from_bytes, rfm_weight_meta};
use super::meta::{WeightError, WeightMeta};
use crate::config::ModelConfig;
use crate::loader::{GgufFile, LoadError, RfmFile, RfmType};

/// Shortconv (depthwise causal conv1d with double gating) weights for LFM2 layers.
#[derive(Clone, Debug)]
pub struct CpuShortconvWeights {
    pub in_proj: Vec<u8>,
    pub in_proj_meta: WeightMeta,
    pub conv: Vec<u8>,
    pub conv_meta: WeightMeta,
    pub out_proj: Vec<u8>,
    pub out_proj_meta: WeightMeta,
}

/// Mixture-of-Experts FFN weights for a single layer (CPU-resident).
#[derive(Clone, Debug)]
pub struct CpuMoeWeights {
    pub gate_exps: Vec<u8>,
    pub gate_exps_meta: WeightMeta,
    pub up_exps: Vec<u8>,
    pub up_exps_meta: WeightMeta,
    pub down_exps: Vec<u8>,
    pub down_exps_meta: WeightMeta,
    pub gate_inp: Vec<u8>,
    pub gate_inp_meta: WeightMeta,
    pub exp_probs_b_bias: Option<Vec<f32>>,
    pub num_experts: usize,
    pub ff_size: usize,
}

pub(crate) fn load_shortconv_gguf(
    file: &GgufFile,
    layer: usize,
) -> Result<CpuShortconvWeights, WeightError> {
    let load = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
        let t = file
            .tensor(name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        Ok((t.data.to_vec(), WeightMeta::from_view(&t, false)))
    };

    Ok(CpuShortconvWeights {
        in_proj: load(&format!("blk.{}.shortconv.in_proj.weight", layer))?.0,
        in_proj_meta: load(&format!("blk.{}.shortconv.in_proj.weight", layer))?.1,
        conv: load(&format!("blk.{}.shortconv.conv.weight", layer))?.0,
        conv_meta: load(&format!("blk.{}.shortconv.conv.weight", layer))?.1,
        out_proj: load(&format!("blk.{}.shortconv.out_proj.weight", layer))?.0,
        out_proj_meta: load(&format!("blk.{}.shortconv.out_proj.weight", layer))?.1,
    })
}

pub(crate) fn load_shortconv_rfm(
    file: &RfmFile,
    layer: usize,
) -> Result<CpuShortconvWeights, WeightError> {
    let load = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
        let t = file
            .tensor(name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        Ok((t.data.to_vec(), rfm_weight_meta(&t, false)))
    };

    Ok(CpuShortconvWeights {
        in_proj: load(&format!("blk.{}.shortconv.in_proj.weight", layer))?.0,
        in_proj_meta: load(&format!("blk.{}.shortconv.in_proj.weight", layer))?.1,
        conv: load(&format!("blk.{}.shortconv.conv.weight", layer))?.0,
        conv_meta: load(&format!("blk.{}.shortconv.conv.weight", layer))?.1,
        out_proj: load(&format!("blk.{}.shortconv.out_proj.weight", layer))?.0,
        out_proj_meta: load(&format!("blk.{}.shortconv.out_proj.weight", layer))?.1,
    })
}

pub(crate) fn load_moe_gguf(
    file: &GgufFile,
    layer: usize,
    _config: &ModelConfig,
) -> Result<CpuMoeWeights, WeightError> {
    let load = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
        let t = file
            .tensor(name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        Ok((t.data.to_vec(), WeightMeta::from_view(&t, false)))
    };

    let gate_exps = load(&format!("blk.{}.ffn_gate_exps.weight", layer))?;
    let up_exps = load(&format!("blk.{}.ffn_up_exps.weight", layer))?;
    let down_exps = load(&format!("blk.{}.ffn_down_exps.weight", layer))?;
    let gate_inp = load(&format!("blk.{}.ffn_gate_inp.weight", layer))?;

    // Infer num_experts from gate_exps dims: [hidden, ff_size, num_experts]
    let num_experts = gate_exps.1.dims.last().copied().unwrap_or(1) as usize;
    let ff_size = gate_exps.1.dims[1] as usize;

    let exp_probs_b_name = format!("blk.{}.exp_probs_b.bias", layer);
    let exp_probs_b_bias = if file.tensor(&exp_probs_b_name).map_err(WeightError::Load)?.is_some() {
        Some(copy_f32(file, &exp_probs_b_name)?)
    } else {
        None
    };

    Ok(CpuMoeWeights {
        gate_exps: gate_exps.0,
        gate_exps_meta: gate_exps.1,
        up_exps: up_exps.0,
        up_exps_meta: up_exps.1,
        down_exps: down_exps.0,
        down_exps_meta: down_exps.1,
        gate_inp: gate_inp.0,
        gate_inp_meta: gate_inp.1,
        exp_probs_b_bias,
        num_experts,
        ff_size,
    })
}

pub(crate) fn load_moe_rfm(
    file: &RfmFile,
    layer: usize,
    _config: &ModelConfig,
) -> Result<CpuMoeWeights, WeightError> {
    let load = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
        let t = file
            .tensor(name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        Ok((t.data.to_vec(), rfm_weight_meta(&t, false)))
    };

    let load_f32 = |name: &str| -> Result<Vec<f32>, WeightError> {
        let t = file
            .tensor(name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        if !matches!(t.wtype, RfmType::F32) {
            return Err(WeightError::Load(LoadError::UnknownTensorType(0)));
        }
        Ok(copy_f32_from_bytes(t.data))
    };

    let gate_exps = load(&format!("blk.{}.ffn_gate_exps.weight", layer))?;
    let up_exps = load(&format!("blk.{}.ffn_up_exps.weight", layer))?;
    let down_exps = load(&format!("blk.{}.ffn_down_exps.weight", layer))?;
    let gate_inp = load(&format!("blk.{}.ffn_gate_inp.weight", layer))?;

    let num_experts = gate_exps.1.dims.last().copied().unwrap_or(1) as usize;
    let ff_size = gate_exps.1.dims[1] as usize;

    let exp_probs_b_name = format!("blk.{}.exp_probs_b.bias", layer);
    let exp_probs_b_bias = if file.tensor(&exp_probs_b_name).map_err(WeightError::Load)?.is_some() {
        Some(load_f32(&exp_probs_b_name)?)
    } else {
        None
    };

    Ok(CpuMoeWeights {
        gate_exps: gate_exps.0,
        gate_exps_meta: gate_exps.1,
        up_exps: up_exps.0,
        up_exps_meta: up_exps.1,
        down_exps: down_exps.0,
        down_exps_meta: down_exps.1,
        gate_inp: gate_inp.0,
        gate_inp_meta: gate_inp.1,
        exp_probs_b_bias,
        num_experts,
        ff_size,
    })
}
