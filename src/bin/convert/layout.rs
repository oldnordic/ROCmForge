use std::fs::File;
use std::io::Write;

use rocmforge::loader::RfmType;
use rocmforge::loader::{GgmlType, TensorView};

use super::quant::{
    bytes_to_f32, dequantize_f16_to_f32, dequantize_q4_0_to_f32, dequantize_q6_k_to_f32,
    dequantize_q8_0_to_f32, quantize_matrix_q4_0,
};

fn rotate_tensor_inplace(data: &mut [f32], rows: usize, cols: usize) {
    let scale = 1.0 / (cols as f32).sqrt();
    for r in 0..rows {
        let row_slice = &mut data[r * cols..(r + 1) * cols];
        super::math::fwht_inplace(row_slice);
        for val in row_slice.iter_mut() {
            *val *= scale;
        }
    }
}

fn quantize_q6_k_block(block: &[f32]) -> [u8; 210] {
    let mut max_abs = 0.0f32;
    for &x in block {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    let d = max_abs / 32.0;
    let d_f16 = half::f16::from_f32(d);
    let d_f32 = d_f16.to_f32();
    let inv_d = if d_f32 > 1e-10 { 1.0 / d_f32 } else { 0.0 };

    let mut q = [0i8; 256];
    for i in 0..256 {
        let val = block[i] * inv_d;
        q[i] = val.round().clamp(-32.0, 31.0) as i8;
    }

    let mut out = [0u8; 210];

    let d_bytes = d_f16.to_bits().to_le_bytes();
    out[208] = d_bytes[0];
    out[209] = d_bytes[1];

    for byte in &mut out[192..208] {
        *byte = 1;
    }

    for l in 0..32 {
        let qi1 = (q[l] as i32 + 32).clamp(0, 63) as u8;
        let qi2 = (q[l + 32] as i32 + 32).clamp(0, 63) as u8;
        let qi3 = (q[l + 64] as i32 + 32).clamp(0, 63) as u8;
        let qi4 = (q[l + 96] as i32 + 32).clamp(0, 63) as u8;

        out[l] = (qi1 & 0x0F) | ((qi3 & 0x0F) << 4);
        out[l + 32] = (qi2 & 0x0F) | ((qi4 & 0x0F) << 4);
        out[128 + l] = ((qi1 >> 4) & 3)
            | (((qi2 >> 4) & 3) << 2)
            | (((qi3 >> 4) & 3) << 4)
            | (((qi4 >> 4) & 3) << 6);
    }

    for l in 0..32 {
        let qi1 = (q[128 + l] as i32 + 32).clamp(0, 63) as u8;
        let qi2 = (q[128 + l + 32] as i32 + 32).clamp(0, 63) as u8;
        let qi3 = (q[128 + l + 64] as i32 + 32).clamp(0, 63) as u8;
        let qi4 = (q[128 + l + 96] as i32 + 32).clamp(0, 63) as u8;

        out[64 + l] = (qi1 & 0x0F) | ((qi3 & 0x0F) << 4);
        out[96 + l] = (qi2 & 0x0F) | ((qi4 & 0x0F) << 4);
        out[160 + l] = ((qi1 >> 4) & 3)
            | (((qi2 >> 4) & 3) << 2)
            | (((qi3 >> 4) & 3) << 4)
            | (((qi4 >> 4) & 3) << 6);
    }

    out
}

fn quantize_matrix_q6_k(data: &[f32]) -> Vec<u8> {
    let num_blocks = data.len() / 256;
    let mut out = Vec::with_capacity(num_blocks * 210);
    for i in 0..num_blocks {
        let block = &data[i * 256..(i + 1) * 256];
        let q_block = quantize_q6_k_block(block);
        out.extend_from_slice(&q_block);
    }
    out
}

pub(super) fn pack_tensor(
    tensor: &TensorView,
    writer: &mut dyn Write,
    wtype: RfmType,
) -> Result<u64, Box<dyn std::error::Error>> {
    match wtype {
        RfmType::F32 => {
            writer.write_all(tensor.data)?;
            Ok(tensor.data.len() as u64)
        }
        RfmType::GgufPassthrough(_) => {
            writer.write_all(tensor.data)?;
            Ok(tensor.data.len() as u64)
        }
        RfmType::Mq4 => {
            let count = tensor.element_count();
            let mut f32_data = match tensor.ggml_type {
                GgmlType::F32 => bytes_to_f32(tensor.data),
                GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, count),
                GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, count),
                GgmlType::Q8_0 => dequantize_q8_0_to_f32(tensor.data, count),
                GgmlType::F16 => dequantize_f16_to_f32(tensor.data, count),
                other => return Err(format!("Unsupported source type for MQ4: {:?}", other).into()),
            };

            let cols = tensor.dims[0] as usize;
            let rows = if tensor.dims.len() > 1 {
                tensor.dims[1] as usize
            } else {
                1
            };
            if cols.is_power_of_two() {
                rotate_tensor_inplace(&mut f32_data, rows, cols);
            } else {
                println!(
                    "⚠️ Warning: tensor cols {} is not a power of two, skipping pre-rotation",
                    cols
                );
            }

            let q_data = quantize_matrix_q4_0(&f32_data);
            writer.write_all(&q_data)?;
            Ok(q_data.len() as u64)
        }
        RfmType::Mq6 => {
            let count = tensor.element_count();
            let mut f32_data = match tensor.ggml_type {
                GgmlType::F32 => bytes_to_f32(tensor.data),
                GgmlType::Q4_0 => dequantize_q4_0_to_f32(tensor.data, count),
                GgmlType::Q6_K => dequantize_q6_k_to_f32(tensor.data, count),
                GgmlType::Q8_0 => dequantize_q8_0_to_f32(tensor.data, count),
                GgmlType::F16 => dequantize_f16_to_f32(tensor.data, count),
                other => return Err(format!("Unsupported source type for MQ6: {:?}", other).into()),
            };

            let cols = tensor.dims[0] as usize;
            let rows = if tensor.dims.len() > 1 {
                tensor.dims[1] as usize
            } else {
                1
            };
            if cols.is_power_of_two() {
                rotate_tensor_inplace(&mut f32_data, rows, cols);
            } else {
                println!(
                    "⚠️ Warning: tensor cols {} is not a power of two, skipping pre-rotation",
                    cols
                );
            }

            let q_data = quantize_matrix_q6_k(&f32_data);
            writer.write_all(&q_data)?;
            Ok(q_data.len() as u64)
        }
        RfmType::Q4Split => {
            if tensor.ggml_type != GgmlType::Q4_0 {
                return Err(format!(
                    "Unsupported GGUF quant type for split conversion: {:?}",
                    tensor.ggml_type
                )
                .into());
            }

            let num_gguf_blocks = tensor.data.len() / 18;
            let rfm_blocks = num_gguf_blocks / 8;
            if num_gguf_blocks % 8 != 0 {
                return Err(format!(
                    "Tensor {} blocks count is not divisible by 8: {}",
                    tensor.name, num_gguf_blocks
                )
                .into());
            }

            let mut scales = Vec::with_capacity(rfm_blocks * 8 * 2);
            let zero_points = vec![0u8; rfm_blocks * 16];
            let mut nibbles = Vec::with_capacity(rfm_blocks * 128);

            for b in 0..rfm_blocks {
                let base_idx = b * 8;
                for i in 0..8 {
                    let g_block = &tensor.data[(base_idx + i) * 18..(base_idx + i + 1) * 18];
                    scales.push(g_block[0]);
                    scales.push(g_block[1]);
                    nibbles.extend_from_slice(&g_block[2..18]);
                }
            }

            writer.write_all(&scales)?;
            writer.write_all(&zero_points)?;
            writer.write_all(&nibbles)?;

            let total_size = (scales.len() + zero_points.len() + nibbles.len()) as u64;
            Ok(total_size)
        }
        _ => Err("Invalid tensor packing layout selected".into()),
    }
}

pub(super) fn rfm_type_for_tensor(tensor: &TensorView, mq4: bool, mq6: bool) -> RfmType {
    let is_weight = tensor.dims.len() > 1 && tensor.name.contains(".weight");
    if mq4 && is_weight {
        RfmType::Mq4
    } else if mq6 && is_weight {
        RfmType::Mq6
    } else {
        match tensor.ggml_type {
            GgmlType::F32 => RfmType::F32,
            GgmlType::Q4_0 => RfmType::Q4Split,
            other => RfmType::GgufPassthrough(other as u32),
        }
    }
}

pub(super) fn pack_gate_up_fused(
    gate: &TensorView,
    up: &TensorView,
    writer: &mut File,
) -> Result<u64, Box<dyn std::error::Error>> {
    if gate.ggml_type != GgmlType::Q4_0 || up.ggml_type != GgmlType::Q4_0 {
        return Err("Only Q4_0 GGUF tensors can be fused into Gate-Up layout".into());
    }

    if gate.dims != up.dims {
        return Err("Gate and Up tensor dimensions must match exactly for fusion".into());
    }

    let intermediate_size = gate.dims[1] as usize;
    let hidden_size = gate.dims[0] as usize;
    let num_gguf_blocks_row = hidden_size / 32;
    let rfm_blocks_row = num_gguf_blocks_row / 8;

    if num_gguf_blocks_row % 8 != 0 {
        return Err(format!(
            "Hidden size {} is not a multiple of 256 elements",
            hidden_size
        )
        .into());
    }

    let mut scales = Vec::new();
    let mut zero_points = Vec::new();
    let mut nibbles = Vec::new();

    for r in 0..intermediate_size {
        let gate_row_offset = r * num_gguf_blocks_row * 18;
        let up_row_offset = r * num_gguf_blocks_row * 18;

        for b in 0..rfm_blocks_row {
            let base_gguf_blk = b * 8;

            let mut gate_scales = [0u8; 16];
            let mut gate_nibbles = [0u8; 128];
            for i in 0..8 {
                let blk_bytes = &gate.data[gate_row_offset + (base_gguf_blk + i) * 18
                    ..gate_row_offset + (base_gguf_blk + i + 1) * 18];
                gate_scales[i * 2] = blk_bytes[0];
                gate_scales[i * 2 + 1] = blk_bytes[1];
                gate_nibbles[i * 16..(i + 1) * 16].copy_from_slice(&blk_bytes[2..18]);
            }

            let mut up_scales = [0u8; 16];
            let mut up_nibbles = [0u8; 128];
            for i in 0..8 {
                let blk_bytes = &up.data[up_row_offset + (base_gguf_blk + i) * 18
                    ..up_row_offset + (base_gguf_blk + i + 1) * 18];
                up_scales[i * 2] = blk_bytes[0];
                up_scales[i * 2 + 1] = blk_bytes[1];
                up_nibbles[i * 16..(i + 1) * 16].copy_from_slice(&blk_bytes[2..18]);
            }

            scales.extend_from_slice(&gate_scales);
            scales.extend_from_slice(&up_scales);
            zero_points.extend_from_slice(&[0u8; 32]);
            nibbles.extend_from_slice(&gate_nibbles);
            nibbles.extend_from_slice(&up_nibbles);
        }
    }

    writer.write_all(&scales)?;
    writer.write_all(&zero_points)?;
    writer.write_all(&nibbles)?;

    let total_size = (scales.len() + zero_points.len() + nibbles.len()) as u64;
    Ok(total_size)
}
