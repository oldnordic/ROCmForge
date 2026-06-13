use super::super::super::error::{GpuError, GpuResult};
use super::super::upload::{
    build_matrix_meta, try_build_q4_0_gate_up_interleaved, try_build_q4_0_gate_up_interleaved_tile4,
};
use super::support::{qwen35_post_attention_norm_name, SvdCorrection};
use super::GpuLayerWeights;
use crate::config::{AttentionLayout, ModelConfig, TensorName, TensorNamingScheme};
use crate::loader::GgufFile;

impl GpuLayerWeights {
    /// Estimate total VRAM bytes used by this layer's weights.
    pub fn estimate_vram_usage(&self) -> usize {
        let mut total = 0;
        total += self.attn_norm.size();
        total += self.attn_q.size();
        total += self.attn_k.size();
        total += self.attn_v.size();
        total += self.attn_o.size();
        total += self.ffn_norm.size();
        total += self.ffn_up.size();
        total += self.ffn_down.size();

        if let Some(ref qkv) = self.attn_qkv {
            total += qkv.size();
        }
        if let Some(ref gate) = self.attn_gate {
            total += gate.size();
        }
        if let Some(ref ssm) = self.ssm {
            total += ssm.a.size() + ssm.dt.size() + ssm.norm.size() + ssm.conv1d.size();
            total += ssm.alpha.size() + ssm.beta.size() + ssm.out.size();
        }
        if let Some(ref sc) = self.shortconv {
            total += sc.in_proj.size() + sc.conv.size() + sc.out_proj.size();
        }
        if let Some(ref ffn_gate) = self.ffn_gate {
            total += ffn_gate.size();
        }
        if let Some(ref q_norm) = self.attn_q_norm {
            total += q_norm.size();
        }
        if let Some(ref k_norm) = self.attn_k_norm {
            total += k_norm.size();
        }
        if let Some(ref q_bias) = self.attn_q_bias {
            total += q_bias.size();
        }
        if let Some(ref k_bias) = self.attn_k_bias {
            total += k_bias.size();
        }
        if let Some(ref v_bias) = self.attn_v_bias {
            total += v_bias.size();
        }
        if let Some(ref g_inter) = self.ffn_gate_up_interleaved {
            total += g_inter.size();
        }
        if let Some(ref g_tile4) = self.ffn_gate_up_interleaved_tile4 {
            total += g_tile4.size();
        }
        if let Some(ref moe) = self.moe {
            total += moe.router.size();
            if let Some(ref b) = moe.router_bias {
                total += b.size();
            }
            if let Some(ref g) = moe.shared_gate {
                total += g.size();
            }
            if let Some(ref u) = moe.shared_up {
                total += u.size();
            }
            if let Some(ref d) = moe.shared_down {
                total += d.size();
            }
            if let Some(ref gi) = moe.shared_gate_inp {
                total += gi.size();
            }
        }

        total
    }

    pub(in crate::gpu::weights) fn estimate_vram_usage_from_file(
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

        let is_attention_layer = file.has_tensor(&attn_k_name);

        let attention_bytes = if is_attention_layer {
            if layer_has_fused_qkv {
                tensor_bytes(&qkv_name)?
                    + tensor_bytes_optional(&format!("blk.{}.attn_gate.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_a", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_dt", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_norm.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_conv1d.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_alpha.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_beta.weight", layer))?
                    + tensor_bytes_optional(&format!("blk.{}.ssm_out.weight", layer))?
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
            }
        } else {
            // Shortconv layer
            tensor_bytes(&format!("blk.{}.shortconv.in_proj.weight", layer))?
                + tensor_bytes(&format!("blk.{}.shortconv.conv.weight", layer))?
                + tensor_bytes(&format!("blk.{}.shortconv.out_proj.weight", layer))?
        };

        let total = tensor_bytes(&attn_norm_name)?
            + attention_bytes
            + tensor_bytes(&ffn_norm_name)?
            + ffn_gate_bytes
            + ffn_up_bytes
            + ffn_down_bytes
            + moe_extra_bytes
            + interleaved_bytes
            + interleaved_tile4_bytes;

        Ok(total)
    }
}
