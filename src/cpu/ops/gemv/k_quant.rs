use super::super::gemm::{
    gemm_q2_k_fallback, gemm_q3_k_fallback, gemm_q5_k_fallback, gemm_q6_k_fallback,
};
use rayon::prelude::*;

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
pub(crate) fn gemv_q2_k_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
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
pub(crate) fn gemv_q3_k_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
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
pub(crate) fn gemv_q5_k_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
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
pub(crate) fn gemv_q6_k_transposed(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
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
    out_dim: usize,
    in_dim: usize,
) {
    y.par_iter_mut().enumerate().for_each(|(vocab_idx, out)| {
        let mut deq = vec![0.0f32; in_dim];
        crate::cpu::quant::embed_q4_k(vocab_idx, w, &mut deq, in_dim);
        *out = deq.iter().zip(x.iter()).map(|(d, xi)| d * xi).sum::<f32>();
    });
    let _ = out_dim;
}
