//! GEMV (matrix-vector multiply for decode).

use crate::cpu::quant::{
    load_f16_scale, validate_block_size, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q4_BLOCK_BYTES,
    Q4_BLOCK_ELEMS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q6_K_BLOCK_ELEMS, Q8_0_MAX, Q8_BLOCK_BYTES,
    Q8_BLOCK_ELEMS,
};
use crate::cpu::weights::{try_as_f32_slice, WeightMeta};
use crate::loader::GgmlType;
use rayon::prelude::*;

use super::avx2::{
    dot_q4_0_block_avx2, dot_q4_0_q8_0_block_avx2, dot_q4_0_q8_0_block_scalar,
    dot_q4_1_q8_0_block_avx2, dot_q4_1_q8_0_block_scalar,
};

mod dense;
mod k_quant;

pub use dense::*;
pub use k_quant::*;

/// Quantize a single f32 vector to Q8_0 (one block per Q8_BLOCK_ELEMS).
fn quantize_q8_0_single(x: &[f32], out: &mut [u8], in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    debug_assert_eq!(out.len(), num_blocks * Q8_BLOCK_BYTES);

    for b in 0..num_blocks {
        let xb = &x[b * Q8_BLOCK_ELEMS..(b + 1) * Q8_BLOCK_ELEMS];
        let off = b * Q8_BLOCK_BYTES;
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / Q8_0_MAX } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let scale_bytes = half::f16::from_f32(scale).to_bits().to_le_bytes();
        out[off] = scale_bytes[0];
        out[off + 1] = scale_bytes[1];
        for i in 0..Q8_BLOCK_ELEMS {
            let q = (xb[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out[off + 2 + i] = q as u8;
        }
    }
}

pub fn gemv_q4_0(w: &[u8], x: &[f32], y: &mut [f32], _out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;
    let features = crate::cpu::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..];
            let scale = load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_block_avx2(qs, xb, scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..16 {
                        let q0 = (qs[i] & 0x0F) as i32 - 8;
                        let q1 = (qs[i] >> 4) as i32 - 8;
                        acc += scale * (q0 as f32) * xb[i] + scale * (q1 as f32) * xb[i + 16];
                    }
                }
            } else {
                for i in 0..16 {
                    let q0 = (qs[i] & 0x0F) as i32 - 8;
                    let q1 = (qs[i] >> 4) as i32 - 8;
                    acc += scale * (q0 as f32) * xb[i] + scale * (q1 as f32) * xb[i + 16];
                }
            }
        }
        *out = acc;
    });
}

pub fn gemv_q4_0_q8_0(w: &[u8], x_q8: &[u8], y: &mut [f32], _out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;
    let features = crate::cpu::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q4_BLOCK_BYTES..];
            let w_scale = load_f16_scale(&block[0..2]);
            let x_scale = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale = w_scale * x_scale;
            let qs = &block[2..18];
            let q8 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe { dot_q4_0_q8_0_block_avx2(qs, q8, combined_scale) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_0_q8_0_block_scalar(qs, q8, combined_scale);
                }
            } else {
                acc += dot_q4_0_q8_0_block_scalar(qs, q8, combined_scale);
            }
        }
        *out = acc;
    });
}

pub fn gemv_q4_1_q8_0(w: &[u8], x_q8: &[u8], y: &mut [f32], _out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_1_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_1_BLOCK_BYTES;
    let features = crate::cpu::features::CpuFeatures::get();
    let use_avx2 = features.has_avx2;

    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q4_1_BLOCK_BYTES..];
            let w_scale = load_f16_scale(&block[0..2]);
            let w_min = load_f16_scale(&block[2..4]);
            let x_scale = load_f16_scale(&x_q8[b * Q8_BLOCK_BYTES..][0..2]);
            let combined_scale = w_scale * x_scale;
            let qs = &block[4..20];
            let q8 = &x_q8[b * Q8_BLOCK_BYTES + 2..][..Q8_BLOCK_ELEMS];
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    acc += unsafe {
                        dot_q4_1_q8_0_block_avx2(qs, q8, combined_scale, w_min * x_scale)
                    };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    acc += dot_q4_1_q8_0_block_scalar(qs, q8, combined_scale, w_min * x_scale);
                }
            } else {
                acc += dot_q4_1_q8_0_block_scalar(qs, q8, combined_scale, w_min * x_scale);
            }
        }
        *out = acc;
    });
}

pub fn gemv_q5_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q5_0_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_0_BLOCK_BYTES;
    for o in 0..out_dim {
        let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
            let d = load_f16_scale(&block[0..2]);
            let qh = &block[2..6];
            let qs = &block[6..22];
            let xb = &x[b * Q5_0_BLOCK_ELEMS..];
            for i in 0..16 {
                let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                let low_bits_0 = qs[i] & 0x0F;
                let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;
                let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                let low_bits_1 = (qs[i] >> 4) & 0x0F;
                let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;
                acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
            }
        }
        y[o] = acc;
    }
}

pub fn gemv_q8_0(w: &[u8], x: &[f32], y: &mut [f32], _out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q8_BLOCK_BYTES;
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_w = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..num_blocks {
            let block = &row_w[b * Q8_BLOCK_BYTES..(b + 1) * Q8_BLOCK_BYTES];
            let scale = load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];
            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }
        *out = acc;
    });
}

pub fn gemv_q8_0_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q8_BLOCK_BYTES;
    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block = &w[col_offset + b * Q8_BLOCK_BYTES..col_offset + (b + 1) * Q8_BLOCK_BYTES];
            let scale = load_f16_scale(&block[0..2]);
            let qs = &block[2..34];
            let xb = &x[b * Q8_BLOCK_ELEMS..];
            for i in 0..Q8_BLOCK_ELEMS {
                acc += (qs[i] as i8) as f32 * scale * xb[i];
            }
        }
        *out = acc;
    });
    let _ = out_dim;
}

pub fn gemv_q4_0_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks * Q4_BLOCK_BYTES;
    y.par_iter_mut().enumerate().for_each(|(v, out)| {
        let mut acc = 0.0f32;
        let col_offset = v * col_bytes;
        for b in 0..num_blocks {
            let block = &w[col_offset + b * Q4_BLOCK_BYTES..col_offset + (b + 1) * Q4_BLOCK_BYTES];
            let scale = load_f16_scale(&block[0..2]);
            let qs = &block[2..18];
            let xb = &x[b * Q4_BLOCK_ELEMS..];
            for i in 0..16 {
                let q0 = (qs[i] & 0x0f) as i32 - 8;
                let q1 = (qs[i] >> 4) as i32 - 8;
                acc += scale * (q0 as f32) * xb[i] + scale * (q1 as f32) * xb[i + 16];
            }
        }
        *out = acc;
    });
    let _ = out_dim;
}

pub fn gemv_q4_1_transposed(
    _w: &[u8],
    _x: &[f32],
    _y: &mut [f32],
    _out_dim: usize,
    _in_dim: usize,
) {
    // Placeholder for symmetry, usually not needed for tied embeddings
}

pub fn dispatch_gemv_transposed(
    w: &[u8],
    meta: &WeightMeta,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), crate::cpu::CpuError> {
    let mut meta_copy = meta.clone();
    meta_copy.needs_transpose = true;
    dispatch_gemv(w, &meta_copy, x, y, out_dim, in_dim, None)
}

pub fn dispatch_gemv(
    w: &[u8],
    meta: &WeightMeta,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    q8_scratch: Option<&mut [u8]>,
) -> Result<(), crate::cpu::CpuError> {
    // Validate block size for quantized types
    match meta.wtype {
        GgmlType::Q4_0 | GgmlType::Q4_1 => validate_block_size(in_dim, Q4_BLOCK_ELEMS, "in_dim")
            .map_err(|e| crate::cpu::CpuError::InvalidOperation(e.to_string()))?,
        GgmlType::Q6_K => validate_block_size(in_dim, Q6_K_BLOCK_ELEMS, "in_dim")
            .map_err(|e| crate::cpu::CpuError::InvalidOperation(e.to_string()))?,
        _ => {}
    }

    match meta.wtype {
        GgmlType::F32 => {
            if let Some(w_f32) = try_as_f32_slice(w) {
                if meta.needs_transpose {
                    gemv_f32_transposed(w_f32, x, y, out_dim, in_dim);
                } else {
                    gemv_f32(w_f32, x, y);
                }
            } else {
                if meta.needs_transpose {
                    gemv_f32_transposed_bytes(w, x, y, out_dim, in_dim);
                } else {
                    gemv_f32_bytes(w, x, y, out_dim, in_dim);
                }
            }
        }
        GgmlType::F16 => {
            if meta.needs_transpose {
                gemv_f16_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_f16(w, x, y);
            }
        }
        GgmlType::Q4_0 => {
            if meta.needs_transpose {
                gemv_q4_0_transposed(w, x, y, out_dim, in_dim);
            } else if let Some(scratch) = q8_scratch {
                let required = in_dim / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES;
                if scratch.len() < required {
                    return Err(crate::cpu::CpuError::InvalidOperation(
                        "scratch too small".to_string(),
                    ));
                }
                quantize_q8_0_single(x, scratch, in_dim);
                gemv_q4_0_q8_0(w, scratch, y, out_dim, in_dim);
            } else {
                gemv_q4_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_1 => {
            if let Some(scratch) = q8_scratch {
                let required = in_dim / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES;
                if scratch.len() < required {
                    return Err(crate::cpu::CpuError::InvalidOperation(
                        "scratch too small".to_string(),
                    ));
                }
                quantize_q8_0_single(x, scratch, in_dim);
                gemv_q4_1_q8_0(w, scratch, y, out_dim, in_dim);
            } else {
                return Err(crate::cpu::CpuError::UnsupportedWeightType(GgmlType::Q4_1));
            }
        }
        GgmlType::Q5_0 => {
            gemv_q5_0(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q3_K => {
            if meta.needs_transpose {
                gemv_q3_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q3_k(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q2_K => {
            if meta.needs_transpose {
                gemv_q2_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q2_k(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q8_0 => {
            if meta.needs_transpose {
                gemv_q8_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_K => {
            if meta.needs_transpose {
                gemv_q4_k_transposed_fallback(w, x, y, out_dim, in_dim);
            } else {
                crate::cpu::kernels::gemm_q4k_q8::gemv_q4_k_q8_k_dispatch(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q6_K => {
            if meta.needs_transpose {
                gemv_q6_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q6_k(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q5_K => {
            if meta.needs_transpose {
                gemv_q5_k_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemv_q5_k(w, x, y, out_dim, in_dim);
            }
        }
        other => return Err(crate::cpu::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}
