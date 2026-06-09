use super::super::error::{GpuError, GpuResult};
use super::buffer::GpuBuffer;
use super::metadata::{TensorRole, WeightMeta};
use crate::config::ModelConfig;
use crate::cpu::transpose::compute_transpose_flag;
use crate::loader::{GgmlType, GgufFile, RfmFile, RfmType, TensorDesc};

pub(super) fn supports_gpu_matrix_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
    )
}

pub(super) fn derive_tensor_role(is_lm_head: bool, is_tied: bool) -> TensorRole {
    match (is_lm_head, is_tied) {
        (true, true) => TensorRole::TiedLmHead,
        (true, false) => TensorRole::LmHead,
        (false, _) => TensorRole::Generic,
    }
}

pub(super) fn build_matrix_meta(
    weight_name: &str,
    dims: &[u64],
    wtype: GgmlType,
    config: &ModelConfig,
    is_lm_head: bool,
    is_tied: bool,
) -> GpuResult<WeightMeta> {
    if dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: weight_name.to_string(),
            dims: dims.to_vec(),
            reason: "matrix weights must have at least 2 dimensions".to_string(),
        });
    }

    if !supports_gpu_matrix_type(wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: weight_name.to_string(),
            wtype,
        });
    }

    Ok(WeightMeta {
        wtype,
        dims: dims.to_vec(),
        needs_transpose: compute_transpose_flag(
            weight_name,
            dims,
            wtype,
            config,
            is_lm_head,
            is_tied,
        ),
        role: derive_tensor_role(is_lm_head, is_tied),
        svd_k: None,
    })
}

pub(super) fn upload_tensor_bytes(data: &[u8]) -> GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc(data.len())?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

pub(super) fn upload_tensor_bytes_for_device(data: &[u8], device_id: i32) -> GpuResult<GpuBuffer> {
    let mut buf = GpuBuffer::alloc_for_device(data.len(), device_id)?;
    buf.copy_from_host(data)?;
    Ok(buf)
}

pub(super) fn try_build_q4_0_gate_up_interleaved(
    gate_data: &[u8],
    gate_meta: &WeightMeta,
    up_data: &[u8],
    up_meta: &WeightMeta,
) -> Option<Vec<u8>> {
    const QK4_0: usize = 32;
    const Q4_0_BLOCK_SIZE: usize = 18;

    if gate_meta.wtype != GgmlType::Q4_0 || up_meta.wtype != GgmlType::Q4_0 {
        return None;
    }
    if gate_meta.dims != up_meta.dims || gate_meta.dims.len() < 2 {
        return None;
    }

    let n_rows = gate_meta.dims[0] as usize;
    let n_ff = gate_meta.dims[1] as usize;
    if n_rows == 0 || n_ff == 0 || !n_rows.is_multiple_of(QK4_0) {
        return None;
    }

    let n_blocks_total = n_rows / QK4_0;
    let expected_len = n_ff
        .checked_mul(n_blocks_total)?
        .checked_mul(Q4_0_BLOCK_SIZE)?;
    if gate_data.len() != expected_len || up_data.len() != expected_len {
        return None;
    }

    let mut interleaved = Vec::with_capacity(expected_len * 2);
    for ff_idx in 0..n_ff {
        for block_idx in 0..n_blocks_total {
            let offset = (ff_idx * n_blocks_total + block_idx) * Q4_0_BLOCK_SIZE;
            interleaved.extend_from_slice(&gate_data[offset..offset + Q4_0_BLOCK_SIZE]);
            interleaved.extend_from_slice(&up_data[offset..offset + Q4_0_BLOCK_SIZE]);
        }
    }

    Some(interleaved)
}

pub(super) fn try_build_q4_0_gate_up_interleaved_tile4(
    gate_data: &[u8],
    gate_meta: &WeightMeta,
    up_data: &[u8],
    up_meta: &WeightMeta,
) -> Option<Vec<u8>> {
    const QK4_0: usize = 32;
    const Q4_0_BLOCK_SIZE: usize = 18;
    const TILE_FF: usize = 4;

    if gate_meta.wtype != GgmlType::Q4_0 || up_meta.wtype != GgmlType::Q4_0 {
        return None;
    }
    if gate_meta.dims != up_meta.dims || gate_meta.dims.len() < 2 {
        return None;
    }

    let n_rows = gate_meta.dims[0] as usize;
    let n_ff = gate_meta.dims[1] as usize;
    if n_rows == 0 || n_ff == 0 || !n_rows.is_multiple_of(QK4_0) || !n_ff.is_multiple_of(TILE_FF) {
        return None;
    }

    let n_blocks_total = n_rows / QK4_0;
    let expected_len = n_ff
        .checked_mul(n_blocks_total)?
        .checked_mul(Q4_0_BLOCK_SIZE)?;
    if gate_data.len() != expected_len || up_data.len() != expected_len {
        return None;
    }

    let mut interleaved = Vec::with_capacity(expected_len * 2);
    for ff_base in (0..n_ff).step_by(TILE_FF) {
        for block_idx in 0..n_blocks_total {
            for tile_ff in 0..TILE_FF {
                let ff_idx = ff_base + tile_ff;
                let offset = (ff_idx * n_blocks_total + block_idx) * Q4_0_BLOCK_SIZE;
                interleaved.extend_from_slice(&gate_data[offset..offset + Q4_0_BLOCK_SIZE]);
                interleaved.extend_from_slice(&up_data[offset..offset + Q4_0_BLOCK_SIZE]);
            }
        }
    }

    Some(interleaved)
}

pub(super) fn rfm_type_to_ggml(t: &RfmType) -> GgmlType {
    match t {
        RfmType::F32 => GgmlType::F32,
        RfmType::Q4Split | RfmType::Q4FusedGateUp | RfmType::Q4SvdQuant { .. } | RfmType::Mq4 => {
            GgmlType::Q4_0
        }
        RfmType::Mq6 => GgmlType::Q6_K,
        RfmType::GgufPassthrough(v) => GgmlType::from_u32(*v).unwrap_or(GgmlType::Q4_0),
        RfmType::SparseCsr { value_type, .. }
        | RfmType::Mpo { value_type, .. }
        | RfmType::SvdSparseCsr { value_type, .. }
        | RfmType::MoeExpertSvdSparse { value_type, .. }
        | RfmType::MoeExpertSvdFwhtSparse { value_type, .. } => {
            GgmlType::from_u32(*value_type).unwrap_or(GgmlType::F32)
        }
    }
}

pub(super) fn unpack_q4_fused_gate_up(data: &[u8], gate_elements: usize) -> (Vec<u8>, Vec<u8>) {
    let num_blocks = gate_elements / 32;
    let rfm_blocks = num_blocks / 8;

    let mut gate_out = Vec::with_capacity(num_blocks * 18);
    let mut up_out = Vec::with_capacity(num_blocks * 18);

    let scales_total = rfm_blocks * 32;
    let zps_total = rfm_blocks * 32;

    let scales_offset = 0;
    let nibbles_offset = scales_total + zps_total;

    let scales = &data[scales_offset..scales_offset + scales_total];
    let nibbles = &data[nibbles_offset..];

    for b in 0..rfm_blocks {
        let gate_scale_chunk = &scales[b * 32..b * 32 + 16];
        let up_scale_chunk = &scales[b * 32 + 16..b * 32 + 32];

        let gate_nibble_chunk = &nibbles[b * 256..b * 256 + 128];
        let up_nibble_chunk = &nibbles[b * 256 + 128..b * 256 + 256];

        for i in 0..8 {
            gate_out.push(gate_scale_chunk[i * 2]);
            gate_out.push(gate_scale_chunk[i * 2 + 1]);
            gate_out.extend_from_slice(&gate_nibble_chunk[i * 16..(i + 1) * 16]);

            up_out.push(up_scale_chunk[i * 2]);
            up_out.push(up_scale_chunk[i * 2 + 1]);
            up_out.extend_from_slice(&up_nibble_chunk[i * 16..(i + 1) * 16]);
        }
    }
    (gate_out, up_out)
}

pub(super) fn estimate_rfm_layer_vram(file: &RfmFile, layer: usize) -> GpuResult<usize> {
    let mut total = 0;
    let tensors = vec![
        format!("blk.{}.attn_q.weight", layer),
        format!("blk.{}.attn_k.weight", layer),
        format!("blk.{}.attn_v.weight", layer),
        format!("blk.{}.attn_output.weight", layer),
        format!("blk.{}.ffn_gate_up.weight", layer),
        format!("blk.{}.ffn_down.weight", layer),
        format!("blk.{}.attn_norm.weight", layer),
        format!("blk.{}.ffn_norm.weight", layer),
    ];
    for name in &tensors {
        if let Some(t) = file.tensor(name).map_err(|e| GpuError::HipApiError {
            code: -1,
            description: format!("tensor lookup failed: {}", e),
        })? {
            match &t.wtype {
                RfmType::F32 => total += t.data.len(),
                RfmType::Q4Split => {
                    let num_blocks = t.element_count() / 32;
                    total += num_blocks * 18;
                }
                RfmType::Q4FusedGateUp => {
                    let num_blocks = t.element_count() / 32;
                    total += num_blocks * 18 * 2; // gate + up
                    total += num_blocks * 18 * 2; // ffn_gate_up_interleaved
                    total += num_blocks * 18 * 2; // ffn_gate_up_interleaved_tile4
                }
                RfmType::Q4SvdQuant { k } => {
                    let num_blocks = t.element_count() / 32;
                    total += num_blocks * 18;
                    // SVD corrections: U (out_dim x k) + V (k x in_dim) F32
                    if t.dims.len() >= 2 {
                        let in_dim = t.dims[0] as usize;
                        let out_dim = t.dims[1] as usize;
                        total += (in_dim + out_dim) * (*k as usize) * 4;
                    }
                }
                RfmType::GgufPassthrough(_) | RfmType::Mq4 | RfmType::Mq6 => total += t.data.len(),
                RfmType::SparseCsr { .. } | RfmType::Mpo { .. } => total += t.data.len(),
                RfmType::SvdSparseCsr { k, .. } => {
                    // Sparse residual + SVD U + SVD V
                    total += t.data.len();
                    if t.dims.len() >= 2 {
                        let in_dim = t.dims[0] as usize;
                        let out_dim = t.dims[1] as usize;
                        total += (in_dim + out_dim) * (*k as usize) * 4;
                    }
                }
                RfmType::MoeExpertSvdSparse { .. } | RfmType::MoeExpertSvdFwhtSparse { .. } => {
                    // Experts are CPU-resident; they contribute 0 bytes to GPU VRAM.
                }
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod matrix_meta_tests {
    use super::*;
    use crate::config::{AttentionLayout, TensorNameRegistry, TensorNamingScheme};

    fn make_test_config() -> ModelConfig {
        ModelConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 512,
            hidden_size: 1024,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
        }
    }

    #[test]
    fn explicit_lm_head_matches_cpu_transpose_rule() {
        let config = make_test_config();
        let meta = build_matrix_meta(
            "output.weight",
            &[32000, 1024],
            GgmlType::Q4_0,
            &config,
            true,
            false,
        )
        .unwrap();

        assert!(!meta.needs_transpose);
        assert_eq!(meta.role, TensorRole::LmHead);
    }

    #[test]
    fn tied_lm_head_is_not_transposed() {
        let config = make_test_config();
        let meta = build_matrix_meta(
            "output.weight",
            &[32000, 1024],
            GgmlType::Q4_0,
            &config,
            true,
            true,
        )
        .unwrap();

        assert!(!meta.needs_transpose);
        assert_eq!(meta.role, TensorRole::TiedLmHead);
    }

    #[test]
    fn unsupported_matrix_type_is_rejected() {
        let config = make_test_config();
        let err = build_matrix_meta(
            "blk.0.attn_q.weight",
            &[1024, 1024],
            GgmlType::Q5_0,
            &config,
            false,
            false,
        )
        .unwrap_err();

        assert!(matches!(err, GpuError::UnsupportedWeightType { .. }));
    }

    #[test]
    fn matrix_weights_require_two_dims() {
        let config = make_test_config();
        let err = build_matrix_meta(
            "output.weight",
            &[32000],
            GgmlType::Q4_0,
            &config,
            true,
            false,
        )
        .unwrap_err();

        assert!(matches!(err, GpuError::InvalidWeightLayout { .. }));
    }

    #[test]
    fn supports_q2_k_and_q3_k_matrix_types() -> Result<(), Box<dyn std::error::Error>> {
        let config = make_test_config();
        let meta_q2 = build_matrix_meta(
            "blk.0.attn_q.weight",
            &[1024, 1024],
            GgmlType::Q2_K,
            &config,
            false,
            false,
        )?;
        assert_eq!(meta_q2.wtype, GgmlType::Q2_K);

        let meta_q3 = build_matrix_meta(
            "blk.0.attn_q.weight",
            &[1024, 1024],
            GgmlType::Q3_K,
            &config,
            false,
            false,
        )?;
        assert_eq!(meta_q3.wtype, GgmlType::Q3_K);
        Ok(())
    }
}
