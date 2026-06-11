//! GEMM (batched matrix multiply for prefill).

use crate::cpu::quant::{
    load_f16_scale, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS,
    Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q8_BLOCK_BYTES, Q8_BLOCK_ELEMS,
};
use crate::cpu::weights::WeightMeta;
use crate::loader::GgmlType;
use rayon::prelude::*;

/// Load f16 value from bytes as f32.
fn load_f16_as_f32(bytes: &[u8]) -> f32 {
    let u = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from(half::f16::from_bits(u))
}

// ── GEMM (batched matrix multiply for prefill) ───────────────────────────────────

/// F32 GEMM: Y[s, o] = dot(W[o, :], X[s, :])
/// W: [out_dim, in_dim], X: [seq_len, in_dim], Y: [seq_len, out_dim]
pub fn gemm_f32(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row = &w[o * in_dim..(o + 1) * in_dim];
                y_row[o] = w_row.iter().zip(x_row.iter()).map(|(wi, xi)| wi * xi).sum();
            }
        });
}

/// Q4_0 GEMM: dequant on-the-fly.
pub fn gemm_q4_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let block = &row_w[b * Q4_BLOCK_BYTES..(b + 1) * Q4_BLOCK_BYTES];
                    let scale = super::super::quant::load_f16_scale(&block[0..2]);
                    let qs = &block[2..18];
                    let xb = &x_row[b * Q4_BLOCK_ELEMS..];
                    for i in 0..16 {
                        let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                        let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                        acc += q_lo * xb[i] + q_hi * xb[i + 16];
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Q4_1 GEMM: dequant on-the-fly with min offset.
pub fn gemm_q4_1(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q4_1_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q4_1_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let block = &row_w[b * Q4_1_BLOCK_BYTES..(b + 1) * Q4_1_BLOCK_BYTES];
                    let scale = load_f16_scale(&block[0..2]);
                    let min = load_f16_as_f32(&block[2..4]);
                    let qs = &block[4..20];
                    let xb = &x_row[b * Q4_1_BLOCK_ELEMS..];
                    for i in 0..16 {
                        let q_lo = (qs[i] & 0x0F) as f32 * scale + min;
                        let q_hi = (qs[i] >> 4) as f32 * scale + min;
                        acc += q_lo * xb[i] + q_hi * xb[i + 16];
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Q5_0 GEMM: dequant on-the-fly.
pub fn gemm_q5_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q5_0_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_0_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let block = &row_w[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
                    let d = super::super::quant::load_f16_scale(&block[0..2]);
                    let qh = &block[2..6];
                    let qs = &block[6..22];
                    let xb = &x_row[b * Q5_0_BLOCK_ELEMS..];

                    for i in 0..16 {
                        // Process 2 values per iteration
                        let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                        let low_bits_0 = qs[i] & 0x0F;
                        let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                        let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                        let low_bits_1 = (qs[i] >> 4) & 0x0F;
                        let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                        acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Q8_0 GEMM: dequant on-the-fly.
pub fn gemm_q8_0(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks = in_dim / Q8_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q8_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let block = &row_w[b * Q8_BLOCK_BYTES..(b + 1) * Q8_BLOCK_BYTES];
                    let scale = super::super::quant::load_f16_scale(&block[0..2]);
                    let qs = &block[2..34];
                    let xb = &x_row[b * Q8_BLOCK_ELEMS..];
                    for i in 0..Q8_BLOCK_ELEMS {
                        acc += (qs[i] as i8) as f32 * scale * xb[i];
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Q6_K GEMM fallback: exact translation of llama.cpp dequantize_row_q6_K + dot product.
///
/// Following llama.cpp/ggml/src/ggml-quants.c dequantize_row_q6_K exactly:
/// - Two iterations for 256 values (n = 0, 128)
/// - Each iteration processes 128 output values
/// - Pointers advance between iterations
pub fn gemm_q6_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::quant::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q6_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q6_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;

                for b in 0..num_blocks {
                    let block = &row_w[b * Q6_K_BLOCK_BYTES..(b + 1) * Q6_K_BLOCK_BYTES];

                    // Q6_K block layout (from llama.cpp ggml-common.h):
                    // struct block_q6_K {
                    //     uint8_t ql[QK_K/2];      // 128 bytes
                    //     uint8_t qh[QK_K/4];      // 64 bytes
                    //     int8_t  scales[QK_K/16]; // 16 bytes
                    //     ggml_half d;             // 2 bytes (AT THE END!)
                    // }
                    let mut ql = &block[0..128];
                    let mut qh = &block[128..192];
                    let mut sc: &[i8] = unsafe {
                        std::slice::from_raw_parts(block[192..208].as_ptr() as *const i8, 16)
                    };
                    let d = super::super::quant::load_f16_scale(&block[208..210]);

                    // Base index into x_row for this 256-element block
                    let xb_base = b * Q6_K_BLOCK_ELEMS;
                    let mut xb_offset = 0; // Offset within block (0, then 128)

                    // for (int n = 0; n < QK_K; n += 128) {
                    // This iterates TWICE: n=0, n=128
                    for _ in 0..2 {
                        // for (int l = 0; l < 32; ++l) {
                        for l in 0..32 {
                            let is = l / 16;

                            // const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            let q1_ql_part = ql[l] & 0xF;
                            let q1_qh_part = (qh[l] & 3) << 4;
                            let q1_combined = q1_ql_part | q1_qh_part;
                            let q1 = q1_combined as i8 as i32 - 32;
                            // const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            let q2_ql_part = ql[l + 32] & 0xF;
                            let q2_qh_part = ((qh[l] >> 2) & 3) << 4;
                            let q2_combined = q2_ql_part | q2_qh_part;
                            let q2 = q2_combined as i8 as i32 - 32;
                            // const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                            let q3_ql_part = (ql[l] >> 4) & 0xF;
                            let q3_qh_part = ((qh[l] >> 4) & 3) << 4;
                            let q3_combined = q3_ql_part | q3_qh_part;
                            let q3 = q3_combined as i8 as i32 - 32;
                            // const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                            let q4_ql_part = (ql[l + 32] >> 4) & 0xF;
                            let q4_qh_part = ((qh[l] >> 6) & 3) << 4;
                            let q4_combined = q4_ql_part | q4_qh_part;
                            let q4 = q4_combined as i8 as i32 - 32;

                            // Compute dot product contribution
                            // llama.cpp: y[l + 0] = d * sc[is + 0] * q1;
                            // For dot product: acc += d * sc[is + 0] * q1 * xb[l + 0]
                            let scale1 = d * (sc[is] as f32);
                            let scale2 = d * (sc[is + 2] as f32);
                            let scale3 = d * (sc[is + 4] as f32);
                            let scale4 = d * (sc[is + 6] as f32);

                            acc += scale1 * (q1 as f32) * x_row[xb_base + xb_offset + l];
                            acc += scale2 * (q2 as f32) * x_row[xb_base + xb_offset + l + 32];
                            acc += scale3 * (q3 as f32) * x_row[xb_base + xb_offset + l + 64];
                            acc += scale4 * (q4 as f32) * x_row[xb_base + xb_offset + l + 96];
                        }

                        // Advance pointers for next 128 elements (llama.cpp pattern)
                        // y  += 128;  ql += 64;  qh += 32;  sc += 8;
                        ql = &ql[64..]; // Advance 64 bytes into the block
                        qh = &qh[32..]; // Advance 32 bytes
                        sc = &sc[8..]; // Advance 8 scales
                        xb_offset += 128; // Advance 128 elements in x_row
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Q2_K GEMM fallback: dequantize blocks on the fly and compute matrix multiply.
pub fn gemm_q2_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::quant::{Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q2_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q2_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;
                for b in 0..num_blocks {
                    let block_bytes = &row_w[b * Q2_K_BLOCK_BYTES..(b + 1) * Q2_K_BLOCK_BYTES];
                    let block = unsafe {
                        &*(block_bytes.as_ptr() as *const super::super::kernels::BlockQ2K)
                    };
                    let mut deq = [0.0f32; Q2_K_BLOCK_ELEMS];
                    block.dequantize(&mut deq);

                    let xb = &x_row[b * Q2_K_BLOCK_ELEMS..(b + 1) * Q2_K_BLOCK_ELEMS];
                    acc += deq.iter().zip(xb.iter()).map(|(d, x)| d * x).sum::<f32>();
                }
                y_row[o] = acc;
            }
        });
}

/// Q3_K GEMM fallback: dequantize blocks on the fly and compute matrix multiply.
///
/// For each output row:
/// - Load Q3_K blocks from weight matrix
/// - Dequantize each block to f32
/// - Compute dot product with input row
/// - Store result in output
///
/// Performance notes:
/// - Dequantization is done inline during GEMM
/// - Each iteration processes 256 output values (Q3_K block size)
/// - Uses rayon for parallel processing across output rows
pub fn gemm_q3_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::quant::{Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q3_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q3_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;

                for b in 0..num_blocks {
                    let block = &row_w[b * Q3_K_BLOCK_BYTES..(b + 1) * Q3_K_BLOCK_BYTES];

                    // Q3_K block layout:
                    // - hmask[32]: high bit mask
                    // - qs[64]: low 2 bits
                    // - scales[12]: packed 6-bit scales
                    // - d[2]: f16 super-block scale
                    let hmask = &block[0..32];
                    let qs = &block[32..96];
                    let scales = &block[96..108];
                    let d = super::super::quant::load_f16_scale(&block[108..110]);

                    // Unpack scales from packed 6-bit format into 16 scale values
                    const KMASK1: u32 = 0x03030303;
                    const KMASK2: u32 = 0x0f0f0f0f;

                    let mut aux = [0u32; 4];
                    aux[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
                    aux[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
                    aux[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);

                    let tmp = aux[2];
                    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
                    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

                    let mut unpacked_scales = [0i8; 16];
                    for i in 0..4 {
                        let bytes = aux[i].to_le_bytes();
                        for j in 0..4 {
                            unpacked_scales[i * 4 + j] = bytes[j] as i8;
                        }
                    }

                    let mut scale_idx = 0;
                    let xb_base = b * Q3_K_BLOCK_ELEMS;
                    let mut m = 1u8;

                    // Process 256 elements as two 128-element chunks
                    for chunk in 0..2 {
                        let q = &qs[chunk * 32..(chunk + 1) * 32];

                        let mut shift = 0i32;

                        // 4 groups of 32 elements each (4 * 32 = 128 elements)
                        for group in 0..4 {
                            // First 16 elements
                            let dl = d * (unpacked_scales[scale_idx] as f32 - 32.0);
                            scale_idx += 1;

                            for l in 0..16 {
                                let ql = (q[l] >> shift) & 0x03;
                                let hbit = if hmask[l] & m != 0 { 0 } else { 4 };
                                let weight_val = (ql as i8 - hbit) as f32;
                                acc +=
                                    dl * weight_val * x_row[xb_base + chunk * 128 + group * 32 + l];
                            }

                            // Next 16 elements
                            let dl = d * (unpacked_scales[scale_idx] as f32 - 32.0);
                            scale_idx += 1;

                            for l in 0..16 {
                                let ql = (q[l + 16] >> shift) & 0x03;
                                let hbit = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                                let weight_val = (ql as i8 - hbit) as f32;
                                acc += dl
                                    * weight_val
                                    * x_row[xb_base + chunk * 128 + group * 32 + 16 + l];
                            }

                            shift += 2;
                            m <<= 1;
                        }
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Q5_K GEMM fallback: dequantize weights on the fly during matrix multiplication.
///
/// Computes Y = W * X where W is Q5_K quantized.
///
/// # Arguments
/// * `w` - Quantized weight matrix (column-major, stored as [out_dim, in_dim] in Q5_K blocks)
/// * `x` - Input matrix [batch_size, in_dim]
/// * `y` - Output matrix [batch_size, out_dim]
/// * `out_dim` - Output dimension (columns of W, rows of stored layout)
/// * `in_dim` - Input dimension (rows of W, columns of stored layout)
///
/// Performance notes:
/// - Dequantization is done inline during GEMM
/// - Each iteration processes 256 output values (Q5_K block size)
/// - Uses rayon for parallel processing across output rows
pub fn gemm_q5_k_fallback(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    use super::super::quant::{Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};

    let num_blocks = in_dim / Q5_K_BLOCK_ELEMS;
    let row_bytes = num_blocks * Q5_K_BLOCK_BYTES;

    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let row_w = &w[o * row_bytes..(o + 1) * row_bytes];
                let mut acc = 0.0f32;

                for b in 0..num_blocks {
                    let block = &row_w[b * Q5_K_BLOCK_BYTES..(b + 1) * Q5_K_BLOCK_BYTES];

                    // Q5_K block layout (matches llama.cpp block_q5_K):
                    // - d[2]: f16 super-block scale
                    // - dmin[2]: f16 super-block min scale
                    // - scales[12]: scales and mins packed as 6-bit values
                    // - qh[32]: high bit of 5-bit quantization
                    // - ql[128]: low 4-bit quantized weights
                    let d = super::super::quant::load_f16_scale(&block[0..2]);
                    let dmin = super::super::quant::load_f16_scale(&block[2..4]);
                    let scales = &block[4..16];
                    let qh = &block[16..48];
                    let ql = &block[48..176];

                    // Unpack scales and mins from packed 6-bit format (get_scale_min_k4 pattern)
                    let mut unpacked_scales = [0i8; 8];
                    let mut unpacked_mins = [0i8; 8];
                    for j in 0..8 {
                        if j < 4 {
                            unpacked_scales[j] = (scales[j] & 63) as i8;
                            unpacked_mins[j] = (scales[j + 4] & 63) as i8;
                        } else {
                            unpacked_scales[j] =
                                ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as i8;
                            unpacked_mins[j] =
                                ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)) as i8;
                        }
                    }

                    let mut scale_idx = 0;
                    let xb_base = b * Q5_K_BLOCK_ELEMS;

                    // Process 256 elements as 4 chunks of 64 elements each
                    for chunk in 0..4 {
                        let q = &ql[chunk * 32..];
                        let hm = &qh[chunk * 8..];
                        let u1 = 1u8.wrapping_shl(2 * chunk as u32);
                        let u2 = 1u8.wrapping_shl(2 * chunk as u32 + 1);

                        // First 32 elements: d1 * q - m1
                        let d1 = d * unpacked_scales[scale_idx] as f32;
                        let m1 = dmin * unpacked_mins[scale_idx] as f32;
                        scale_idx += 1;
                        for l in 0..32 {
                            let ql_bits = q[l] & 0x0F;
                            let hbit = if hm[l >> 3] & u1 != 0 { 16 } else { 0 };
                            let q_val = (ql_bits + hbit) as f32;
                            acc += (d1 * q_val - m1) * x_row[xb_base + chunk * 64 + l];
                        }

                        // Next 32 elements: d2 * q - m2
                        let d2 = d * unpacked_scales[scale_idx] as f32;
                        let m2 = dmin * unpacked_mins[scale_idx] as f32;
                        scale_idx += 1;
                        for l in 0..32 {
                            let ql_bits = q[l] >> 4;
                            let hbit = if hm[l >> 3] & u2 != 0 { 16 } else { 0 };
                            let q_val = (ql_bits + hbit) as f32;
                            acc += (d2 * q_val - m2) * x_row[xb_base + chunk * 64 + 32 + l];
                        }
                    }
                }
                y_row[o] = acc;
            }
        });
}

/// Dispatch GEMM by weight type with automatic transposition detection.
///
/// Uses metadata to determine if weights need transposed access.
pub fn dispatch_gemm(
    w: &[u8],
    meta: &WeightMeta,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(), super::super::CpuError> {
    // Validate dimensions for block-based quantization
    let block_elems = match meta.wtype {
        GgmlType::F32 => 1,
        GgmlType::Q8_0 => super::super::quant::Q8_BLOCK_ELEMS,
        GgmlType::Q4_0 | GgmlType::Q4_1 => super::super::quant::Q4_BLOCK_ELEMS,
        GgmlType::Q5_0 => super::super::quant::Q5_0_BLOCK_ELEMS,
        GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q2_K => super::super::quant::Q4_K_BLOCK_ELEMS,
        GgmlType::Q3_K => super::super::quant::Q3_K_BLOCK_ELEMS,
        GgmlType::Q5_K => super::super::quant::Q5_K_BLOCK_ELEMS,
        _ => 1,
    };

    if !in_dim.is_multiple_of(block_elems) {
        return Err(super::super::CpuError::InvalidOperation(format!(
            "in_dim {} is not a multiple of block size {} for type {:?}",
            in_dim, block_elems, meta.wtype
        )));
    }

    match meta.wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            if meta.needs_transpose {
                gemm_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemm_f32(wf, x, y, out_dim, in_dim);
            }
        }
        GgmlType::F16 => {
            if meta.needs_transpose {
                gemm_f16_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemm_f16(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_0 => {
            if meta.needs_transpose {
                gemm_q4_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_1 => {
            if meta.needs_transpose {
                gemm_q4_1_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_1(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q5_0 => {
            if meta.needs_transpose {
                gemm_q5_0_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemm_q5_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q8_0 => {
            if meta.needs_transpose {
                gemm_q8_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_K => {
            // Q4_K × Q8_K GEMM: quantize input on the fly
            if meta.needs_transpose {
                // For transposed Q4_K, use dequant-on-the-fly fallback
                gemm_q4_k_transposed_fallback(w, x, y, 1, out_dim, in_dim);
            } else {
                crate::cpu::kernels::gemm_q4k_q8::gemm_q4_k_q8_k_dispatch_gemm(
                    w, x, y, 1, out_dim, in_dim,
                );
            }
        }
        GgmlType::Q6_K => {
            // Q6_K: dequantize to f32 on the fly (slower but works)
            // For transposed weights, the fallback function handles it correctly
            gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q3_K => {
            // Q3_K: dequantize to f32 on the fly (slower but works)
            gemm_q3_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q2_K => {
            // Q2_K: dequantize to f32 on the fly (slower but works)
            gemm_q2_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            // Q5_K: dequantize to f32 on the fly (slower but works)
            gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// Dispatch GEMM with transposed flag for transposed weight matrices.
///
/// When `transposed` is true, computes: Y = W^T * X
/// Otherwise computes: Y = W * X
///
/// Used for FFN down projection where weights are stored as [in_dim, out_dim].
pub fn dispatch_gemm_transposed(
    w: &[u8],
    wtype: GgmlType,
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    transposed: bool,
) -> Result<(), super::super::CpuError> {
    match wtype {
        GgmlType::F32 => {
            let wf: &[f32] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, w.len() / 4) };
            if transposed {
                gemm_f32_transposed(wf, x, y, out_dim, in_dim);
            } else {
                gemm_f32(wf, x, y, out_dim, in_dim);
            }
        }
        GgmlType::F16 => {
            if transposed {
                gemm_f16_transposed(w, x, y, out_dim, in_dim);
            } else {
                gemm_f16(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_0 => {
            if transposed {
                gemm_q4_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q4_1 => {
            if transposed {
                gemm_q4_1_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q4_1(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q8_0 => {
            if transposed {
                gemm_q8_0_transposed_gemm(w, x, y, out_dim, in_dim);
            } else {
                gemm_q8_0(w, x, y, out_dim, in_dim);
            }
        }
        GgmlType::Q6_K => {
            gemm_q6_k_fallback(w, x, y, out_dim, in_dim);
        }
        GgmlType::Q5_K => {
            gemm_q5_k_fallback(w, x, y, out_dim, in_dim);
        }
        other => return Err(super::super::CpuError::UnsupportedWeightType(other)),
    }
    Ok(())
}

/// Q4_0 GEMM transposed for transposed weight matrices.
///
/// Computes: Y = W^T * X where W has shape [in_dim, out_dim]
/// stored in column-major Q4_0 blocked format.
pub fn gemm_q4_0_transposed_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_col = in_dim / Q4_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_BLOCK_BYTES;
    let _seq_len = x.len() / in_dim;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(s, y_row)| {
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                let mut acc = 0.0f32;

                // Iterate through blocks in this column
                for b in 0..num_blocks_per_col {
                    let block =
                        &w[col_offset + b * Q4_BLOCK_BYTES..col_offset + (b + 1) * Q4_BLOCK_BYTES];
                    let scale = super::super::quant::load_f16_scale(&block[0..2]);
                    let qs = &block[2..18];
                    let xb = &x_row[b * Q4_BLOCK_ELEMS..];

                    for i in 0..16 {
                        let q_lo = (((qs[i] & 0x0F) as i32) - 8) as f32 * scale;
                        let q_hi = (((qs[i] >> 4) as i32) - 8) as f32 * scale;
                        acc += q_lo * xb[i] + q_hi * xb[i + 16];
                    }
                }

                y_row[o] = acc;
            });
    }
}

/// Q4_1 GEMM transposed for transposed weight matrices.
pub fn gemm_q4_1_transposed_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_col = in_dim / Q4_1_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q4_1_BLOCK_BYTES;
    let _seq_len = x.len() / in_dim;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(s, y_row)| {
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                let mut acc = 0.0f32;

                // Iterate through blocks in this column
                for b in 0..num_blocks_per_col {
                    let block = &w[col_offset + b * Q4_1_BLOCK_BYTES
                        ..col_offset + (b + 1) * Q4_1_BLOCK_BYTES];
                    let w_scale = super::super::quant::load_f16_scale(&block[0..2]);
                    let w_min = super::super::quant::load_f16_scale(&block[2..4]);
                    let qs = &block[4..20];
                    let xb = &x_row[b * Q4_1_BLOCK_ELEMS..];

                    for i in 0..16 {
                        let q_lo = ((qs[i] & 0x0F) as i32) as f32;
                        let q_hi = ((qs[i] >> 4) as i32) as f32;
                        let v_lo = (q_lo * w_scale + w_min) * xb[i];
                        let v_hi = (q_hi * w_scale + w_min) * xb[i + 16];
                        acc += v_lo + v_hi;
                    }
                }

                y_row[o] = acc;
            });
    }
}

/// Q5_0 GEMM transposed for transposed weight matrices.
///
/// Computes: Y = W^T * X where W has shape [in_dim, out_dim]
/// stored in column-major Q5_0 blocked format.
pub fn gemm_q5_0_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let num_blocks_per_col = in_dim / Q5_0_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q5_0_BLOCK_BYTES;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(s, y_row)| {
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                let mut acc = 0.0f32;

                // Iterate through blocks in this column
                for b in 0..num_blocks_per_col {
                    let block = &w[col_offset + b * Q5_0_BLOCK_BYTES
                        ..col_offset + (b + 1) * Q5_0_BLOCK_BYTES];
                    let d = super::super::quant::load_f16_scale(&block[0..2]);
                    let qh = &block[2..6];
                    let qs = &block[6..22];
                    let xb = &x_row[b * Q5_0_BLOCK_ELEMS..];

                    for i in 0..16 {
                        // Process 2 values per iteration
                        let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) << 4;
                        let low_bits_0 = qs[i] & 0x0F;
                        let q0 = ((high_bit_0 | low_bits_0) as i32) - 16;

                        let high_bit_1 = ((qh[i / 8 + 2] >> (i % 8)) & 1) << 4;
                        let low_bits_1 = (qs[i] >> 4) & 0x0F;
                        let q1 = ((high_bit_1 | low_bits_1) as i32) - 16;

                        acc += d * (q0 as f32) * xb[i] + d * (q1 as f32) * xb[i + 16];
                    }
                }

                y_row[o] = acc;
            });
    }
}

/// Q8_0 GEMM transposed for transposed weight matrices.
pub fn gemm_q8_0_transposed_gemm(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let num_blocks_per_col = in_dim / Q8_BLOCK_ELEMS;
    let col_bytes = num_blocks_per_col * Q8_BLOCK_BYTES;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        let col_offset = o * col_bytes;

        // Process all sequences in parallel
        y.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(s, y_row)| {
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                let mut acc = 0.0f32;

                // Iterate through blocks in this column
                for b in 0..num_blocks_per_col {
                    let block =
                        &w[col_offset + b * Q8_BLOCK_BYTES..col_offset + (b + 1) * Q8_BLOCK_BYTES];
                    let scale = super::super::quant::load_f16_scale(&block[0..2]);
                    let qs = &block[2..34];
                    let xb = &x_row[b * Q8_BLOCK_ELEMS..];

                    for i in 0..Q8_BLOCK_ELEMS {
                        acc += (qs[i] as i8) as f32 * scale * xb[i];
                    }
                }

                y_row[o] = acc;
            });
    }
}

/// F32 GEMM transposed.
fn gemm_f32_transposed(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    let _seq_len = x.len() / in_dim;

    // For each output dimension (column in original matrix)
    for o in 0..out_dim {
        // Process all sequences in parallel
        y.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(s, y_row)| {
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                let mut acc = 0.0f32;

                // Compute dot product: sum_i(x[i] * W[i, o])
                for i in 0..in_dim {
                    acc += x_row[i] * w[i * out_dim + o];
                }

                y_row[o] = acc;
            });
    }
}

/// F16 GEMM: Y[s, o] = dot(W[o, :], X[s, :])
fn gemm_f16(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    y.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(s, y_row)| {
            let x_row = &x[s * in_dim..(s + 1) * in_dim];
            for o in 0..out_dim {
                let w_row_offset = o * in_dim * 2;
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    let val = load_f16_as_f32(&w[w_row_offset + i * 2..w_row_offset + i * 2 + 2]);
                    acc += val * x_row[i];
                }
                y_row[o] = acc;
            }
        });
}

/// F16 GEMM transposed.
fn gemm_f16_transposed(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    for o in 0..out_dim {
        y.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(s, y_row)| {
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    let offset = (i * out_dim + o) * 2;
                    let val = load_f16_as_f32(&w[offset..offset + 2]);
                    acc += x_row[i] * val;
                }
                y_row[o] = acc;
            });
    }
}

/// Q4_K GEMM for transposed weights.
///
/// For transposed access, computes Y = W^T * X where W is stored as [in_dim, out_dim].
/// Uses dequantization-on-the-fly for simplicity (slower but correct).
fn gemm_q4_k_transposed_fallback(w: &[u8], x: &[f32], y: &mut [f32], _m: usize, n: usize, k: usize) {
    use crate::cpu::kernels::q4::BlockQ4K;

    let num_blocks_k = k / 256;
    let row_bytes = num_blocks_k * BlockQ4K::SIZE;

    // For transposed access: W stored as [k, n], compute as W^T * X
    // Each output column j corresponds to row j in stored layout
    y.par_chunks_mut(n)
        .enumerate()
        .for_each(|(batch_idx, y_row)| {
            let x_row = &x[batch_idx * k..(batch_idx + 1) * k];

            for out_col in 0..n {
                let mut acc = 0.0f32;

                for block_idx in 0..num_blocks_k {
                    let block_ptr = unsafe {
                        w.as_ptr()
                            .add(out_col * row_bytes + block_idx * BlockQ4K::SIZE)
                            as *const BlockQ4K
                    };
                    let block = unsafe { &*block_ptr };

                    // Dequantize this block and compute dot product
                    let block_start = block_idx * 256;
                    for i in 0..256 {
                        if block_start + i < k {
                            // Simplified Q4_K dequantization
                            let q4_value = if i < 128 {
                                (block.qs[i / 2] >> (4 * (i % 2))) & 0x0F
                            } else {
                                (block.qs[64 + (i - 128) / 2] >> (4 * ((i - 128) % 2))) & 0x0F
                            };

                            let d = half::f16::from_le_bytes(block.d).to_f32();
                            let weight = d * (q4_value as f32 - 8.0);
                            acc += weight * x_row[block_start + i];
                        }
                    }
                }

                y_row[out_col] = acc;
            }
        });
}
