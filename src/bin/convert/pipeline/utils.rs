use rocmforge::loader::{GgmlType, TensorView};
use std::io::Write;

use super::super::quant::{
    bytes_to_f32, dequantize_q4_0_to_f32, dequantize_q6_k_to_f32,
};

pub(crate) fn should_compress_tensor(name: &str, tensor: &TensorView) -> bool {
    if tensor.dims.len() != 2 {
        return false;
    }
    if tensor.dims.iter().any(|&d| d < 64) {
        return false;
    }
    matches!(
        tensor.ggml_type,
        GgmlType::F32 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K
    ) && name.ends_with(".weight")
        && (name.contains("ffn_gate") || name.contains("ffn_up") || name.contains("ffn_down"))
}

pub(crate) fn estimate_nnz_ratio(tensor: &TensorView) -> f32 {
    let count = tensor.element_count();
    let sample_size = count.min(4096);
    let step = if count > sample_size {
        count / sample_size
    } else {
        1
    };

    let w_f32 = match tensor.ggml_type {
        GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, count),
        GgmlType::Q4_K => {
            let mut out = vec![0.0f32; count];
            rocmforge::cpu::quant::embed_q4_k(0, tensor.data, &mut out, count);
            out
        }
        GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, count),
        GgmlType::F32 => bytes_to_f32(tensor.data),
        _ => return 1.0f32,
    };

    let mut nnz = 0usize;
    for i in 0..sample_size {
        let idx = i * step;
        if idx < w_f32.len() && w_f32[idx].abs() > 1e-6 {
            nnz += 1;
        }
    }

    (nnz as f32) / (sample_size as f32)
}

pub(crate) fn parse_layer_idx(name: &str) -> Option<usize> {
    if !name.starts_with("blk.") {
        return None;
    }
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    parts[1].parse().ok()
}

pub(crate) fn should_svd_tensor(name: &str, tensor: &TensorView, svd_attn_only: bool) -> bool {
    if tensor.dims.len() != 2 {
        return false;
    }
    if svd_attn_only {
        return name.contains("attn_q.weight")
            || name.contains("attn_k.weight")
            || name.contains("attn_v.weight")
            || name.contains("attn_o.weight");
    }
    name.ends_with(".weight")
        && (name.contains("attn") || name.contains("ffn") || name.contains("shortconv"))
}

pub(crate) fn align_to_256(writer: &mut dyn Write, offset: &mut u64) -> Result<(), std::io::Error> {
    let padding = (256 - (*offset % 256)) % 256;
    if padding > 0 {
        writer.write_all(&vec![0u8; padding as usize])?;
        *offset += padding;
    }
    Ok(())
}
