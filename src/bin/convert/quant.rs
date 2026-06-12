pub fn dequantize_q4_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    let num_blocks = num_elements / 32;
    use rayon::prelude::*;
    out.par_chunks_mut(32).enumerate().for_each(|(i, block_out)| {
        let block_data = &data[i * 18..(i + 1) * 18];
        rocmforge::cpu::quant::dequant_q4_0_block(block_data, block_out);
    });
    out
}

pub fn dequantize_q6_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    rocmforge::cpu::quant::embed_q6_k(0, data, &mut out, num_elements);
    out
}

pub fn dequantize_q8_0_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    rocmforge::cpu::quant::embed_q8_0(0, data, &mut out, num_elements);
    out
}

pub fn dequantize_f16_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_elements];
    for i in 0..num_elements {
        let offset = i * 2;
        let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        out[i] = half::f16::from_bits(bits).to_f32();
    }
    out
}

pub fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len() / 4];
    for i in 0..out.len() {
        out[i] = f32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
    }
    out
}

fn quantize_q4_0_block(block: &[f32]) -> [u8; 18] {
    let mut max_abs = 0.0f32;
    for &x in block {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    let scale = max_abs / 8.0;
    let scale_f16 = half::f16::from_f32(scale);
    let scale_f32 = scale_f16.to_f32();
    let inv_scale = if scale_f32 > 1e-10 {
        1.0 / scale_f32
    } else {
        0.0
    };

    let mut q = [0i8; 32];
    for j in 0..32 {
        let val = block[j] * inv_scale;
        q[j] = val.round().clamp(-8.0, 7.0) as i8;
    }

    let mut out = [0u8; 18];
    let scale_bytes = scale_f16.to_bits().to_le_bytes();
    out[0] = scale_bytes[0];
    out[1] = scale_bytes[1];

    for i in 0..16 {
        let low = (q[2 * i] + 8) as u8 & 0x0F;
        let high = (q[2 * i + 1] + 8) as u8 & 0x0F;
        out[2 + i] = low | (high << 4);
    }
    out
}

pub fn quantize_matrix_q4_0(data: &[f32]) -> Vec<u8> {
    let num_blocks = data.len() / 32;
    let mut out = vec![0u8; num_blocks * 18];
    use rayon::prelude::*;
    out.par_chunks_mut(18).enumerate().for_each(|(i, chunk)| {
        let block = &data[i * 32..(i + 1) * 32];
        let q_block = quantize_q4_0_block(block);
        chunk.copy_from_slice(&q_block);
    });
    out
}

#[cfg(test)]
mod tests {
    use super::{dequantize_q6_k_to_f32, quantize_matrix_q4_0};

    #[test]
    fn test_dequantize_q6_k_zero_block() {
        let data = vec![0u8; rocmforge::cpu::quant::Q6_K_BLOCK_BYTES];
        let out = dequantize_q6_k_to_f32(&data, rocmforge::cpu::quant::Q6_K_BLOCK_ELEMS);
        assert_eq!(out.len(), rocmforge::cpu::quant::Q6_K_BLOCK_ELEMS);
        assert!(out.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_quantize_matrix_q4_0_produces_expected_block_count() {
        let data = vec![0.0f32; 64];
        let out = quantize_matrix_q4_0(&data);
        assert_eq!(out.len(), 36);
    }
}
