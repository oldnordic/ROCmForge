use crate::cpu::quant::load_f16_scale;
use rayon::prelude::*;
use super::super::gemm::{gemm_q2_k_fallback, gemm_q3_k_fallback, gemm_q5_k_fallback, gemm_q6_k_fallback};

/// Q3_K GEMV: dequant on-the-fly (fallback, slower but works).
pub fn gemv_q3_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    gemm_q3_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q2_K GEMV: dequant on-the-fly (fallback, slower but works).
pub fn gemv_q2_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    gemm_q2_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q6_K GEMV: dequant on-the-fly (fallback, slower but works).
pub fn gemv_q6_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q5_K GEMV: dequantize weights on the fly during matrix-vector multiplication.
pub fn gemv_q5_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
}

/// Q2_K GEMV transposed for tied embeddings.
pub(crate) fn gemv_q2_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use crate::cpu::kernels::BlockQ2K;
    use crate::cpu::quant::{Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q2_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q2_K_BLOCK_BYTES;

    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q2_K_BLOCK_BYTES) as *const BlockQ2K) };
            let mut deq = [0.0f32; Q2_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q2_K_BLOCK_ELEMS..(b + 1) * Q2_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        *out = acc;
    });
    let _ = out_dim;
}

/// Q3_K GEMV transposed for tied embeddings.
pub(crate) fn gemv_q3_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use crate::cpu::kernels::BlockQ3K;
    use crate::cpu::quant::{Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q3_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q3_K_BLOCK_BYTES;

    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q3_K_BLOCK_BYTES) as *const BlockQ3K) };
            let mut deq = [0.0f32; Q3_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q3_K_BLOCK_ELEMS..(b + 1) * Q3_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        *out = acc;
    });
    let _ = out_dim;
}

/// Q5_K GEMV transposed for tied embeddings.
pub(crate) fn gemv_q5_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use crate::cpu::kernels::BlockQ5K;
    use crate::cpu::quant::{Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q5_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q5_K_BLOCK_BYTES;

    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q5_K_BLOCK_BYTES) as *const BlockQ5K) };
            let mut deq = [0.0f32; Q5_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q5_K_BLOCK_ELEMS..(b + 1) * Q5_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        *out = acc;
    });
    let _ = out_dim;
}

/// Q6_K GEMV transposed for tied embeddings.
pub(crate) fn gemv_q6_k_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use crate::cpu::kernels::BlockQ6K;
    use crate::cpu::quant::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q6_K_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q6_K_BLOCK_BYTES;

    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block =
                unsafe { &*(w.as_ptr().add(col_offset + b * Q6_K_BLOCK_BYTES) as *const BlockQ6K) };
            let mut deq = [0.0f32; Q6_K_BLOCK_ELEMS];
            block.dequantize(&mut deq);
            let xb = &x[b * Q6_K_BLOCK_ELEMS..(b + 1) * Q6_K_BLOCK_ELEMS];
            acc += deq.iter().zip(xb.iter()).map(|(d, xi)| d * xi).sum::<f32>();
        }
        *out = acc;
    });
    let _ = out_dim;
}

/// Q4_K GEMV for transposed weights (tied embeddings).
pub(crate) fn gemv_q4_k_transposed_fallback(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    _out_dim: usize,
    in_dim: usize,
) {
    use crate::cpu::kernels::q4::BlockQ4K;

    let num_blocks_k = in_dim / 256;
    let row_bytes = num_blocks_k * BlockQ4K::SIZE;

    y.par_iter_mut().enumerate().for_each(|(vocab_idx, out)| {
        let mut acc = 0.0f32;

        for block_idx in 0..num_blocks_k {
            let block_ptr = unsafe {
                w.as_ptr()
                    .add(vocab_idx * row_bytes + block_idx * BlockQ4K::SIZE)
                    as *const BlockQ4K
            };
            let block = unsafe { &*block_ptr };

            let block_start = block_idx * 256;
            for i in 0..256 {
                if block_start + i < in_dim {
                    let q4_value = if i < 128 {
                        (block.qs[i / 2] >> (4 * (i % 2))) & 0x0F
                    } else {
                        (block.qs[64 + (i - 128) / 2] >> (4 * ((i - 128) % 2))) & 0x0F
                    };
                    
                    let d = load_f16_scale(&block.d);
                    let m = load_f16_scale(&block.dmin);
                    
                    let is_scale = i / 64;
                    let is_group = (i % 64) / 32;
                    let sc_val = (block.scales[is_scale] >> (4 * is_group)) & 0x0F;
                    
                    let val = d * (sc_val as f32) * ((q4_value as i32 - 8) as f32) + m;
                    acc += val * x[block_start + i];
                }
            }
        }
        *out = acc;
    });
}
