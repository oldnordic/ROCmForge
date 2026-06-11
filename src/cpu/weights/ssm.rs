use super::helpers::{
    copy_f32, copy_f32_from_bytes, rfm_weight_meta,
};
use super::meta::{WeightError, WeightMeta};
use crate::config::ModelConfig;
use crate::loader::{GgufFile, LoadError, RfmFile, RfmType};

#[derive(Clone, Debug)]
pub struct CpuSsmWeights {
    pub conv1d: Vec<u8>,
    pub conv1d_meta: WeightMeta,
    pub conv1d_bias: Vec<f32>,
    pub x_proj: Vec<u8>,
    pub x_proj_meta: WeightMeta,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub d: Vec<f32>,
    pub out_proj: Vec<u8>,
    pub out_proj_meta: WeightMeta,
    // Add missing fields for Qwen 3.5
    pub beta: Vec<u8>,
    pub beta_meta: WeightMeta,
    pub alpha: Vec<u8>,
    pub alpha_meta: WeightMeta,
    pub dt: Vec<f32>,
    pub a: Vec<f32>,
    pub out: Vec<u8>,
    pub out_meta: WeightMeta,
}

pub(crate) fn qwen35_post_attention_norm_name(
    config: &ModelConfig,
    layer: usize,
) -> Option<String> {
    if config.num_layers == 40 {
        Some(format!("blk.{}.post_attention_norm.weight", layer))
    } else {
        None
    }
}

pub(crate) fn load_qwen35_ssm_rfm(
    file: &RfmFile,
    layer: usize,
) -> Result<CpuSsmWeights, WeightError> {
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

    Ok(CpuSsmWeights {
        conv1d: load(&format!("blk.{}.ssm_conv1d.weight", layer))?.0,
        conv1d_meta: load(&format!("blk.{}.ssm_conv1d.weight", layer))?.1,
        conv1d_bias: load_f32(&format!("blk.{}.ssm_conv1d.bias", layer))?,
        x_proj: load(&format!("blk.{}.ssm_x_proj.weight", layer))?.0,
        x_proj_meta: load(&format!("blk.{}.ssm_x_proj.weight", layer))?.1,
        dt_bias: load_f32(&format!("blk.{}.ssm_dt_bias", layer))?,
        a_log: load_f32(&format!("blk.{}.ssm_a_log", layer))?,
        d: load_f32(&format!("blk.{}.ssm_d", layer))?,
        out_proj: load(&format!("blk.{}.ssm_out_proj.weight", layer))?.0,
        out_proj_meta: load(&format!("blk.{}.ssm_out_proj.weight", layer))?.1,
        beta: load(&format!("blk.{}.ssm_beta.weight", layer))?.0,
        beta_meta: load(&format!("blk.{}.ssm_beta.weight", layer))?.1,
        alpha: load(&format!("blk.{}.ssm_alpha.weight", layer))?.0,
        alpha_meta: load(&format!("blk.{}.ssm_alpha.weight", layer))?.1,
        dt: load_f32(&format!("blk.{}.ssm_dt", layer))?,
        a: load_f32(&format!("blk.{}.ssm_a", layer))?,
        out: load(&format!("blk.{}.ssm_out.weight", layer))?.0,
        out_meta: load(&format!("blk.{}.ssm_out.weight", layer))?.1,
    })
}

pub(crate) fn load_qwen35_ssm_gguf(
    file: &GgufFile,
    layer: usize,
) -> Result<CpuSsmWeights, WeightError> {
    let load = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
        let t = file
            .tensor(name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        Ok((t.data.to_vec(), WeightMeta::from_view(&t, false)))
    };

    Ok(CpuSsmWeights {
        conv1d: load(&format!("blk.{}.ssm_conv1d.weight", layer))?.0,
        conv1d_meta: load(&format!("blk.{}.ssm_conv1d.weight", layer))?.1,
        conv1d_bias: copy_f32(file, &format!("blk.{}.ssm_conv1d.bias", layer))?,
        x_proj: load(&format!("blk.{}.ssm_x_proj.weight", layer))?.0,
        x_proj_meta: load(&format!("blk.{}.ssm_x_proj.weight", layer))?.1,
        dt_bias: copy_f32(file, &format!("blk.{}.ssm_dt_bias", layer))?,
        a_log: copy_f32(file, &format!("blk.{}.ssm_a_log", layer))?,
        d: copy_f32(file, &format!("blk.{}.ssm_d", layer))?,
        out_proj: load(&format!("blk.{}.ssm_out_proj.weight", layer))?.0,
        out_proj_meta: load(&format!("blk.{}.ssm_out_proj.weight", layer))?.1,
        beta: load(&format!("blk.{}.ssm_beta.weight", layer))?.0,
        beta_meta: load(&format!("blk.{}.ssm_beta.weight", layer))?.1,
        alpha: load(&format!("blk.{}.ssm_alpha.weight", layer))?.0,
        alpha_meta: load(&format!("blk.{}.ssm_alpha.weight", layer))?.1,
        dt: copy_f32(file, &format!("blk.{}.ssm_dt", layer))?,
        a: copy_f32(file, &format!("blk.{}.ssm_a", layer))?,
        out: load(&format!("blk.{}.ssm_out.weight", layer))?.0,
        out_meta: load(&format!("blk.{}.ssm_out.weight", layer))?.1,
    })
}
