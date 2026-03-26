//! Quantization support for GGML weight types.
//!
//! Supports Q4_0, Q4_1, Q8_0 dequantization and embedding table lookup.

use half::f16;

// ── Block sizes ────────────────────────────────────────────────────────────────

/// Q4_0: 32 elements per block, 18 bytes (2 scale + 16 nibbles)
pub const Q4_BLOCK_ELEMS: usize = 32;
pub const Q4_BLOCK_BYTES: usize = 18;

/// Q4_1: 32 elements per block, 20 bytes (2 scale + 2 min + 16 nibbles)
pub const Q4_1_BLOCK_ELEMS: usize = 32;
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Q4_K: 256 elements per block, 144 bytes (2 d + 2 dmin + 12 scales + 128 qs)
pub const Q4_K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Q5_0: 32 elements per block, 22 bytes (2 scale + 4 qh + 16 qs)
/// Q5_0 block format:
/// - d: f16 scale (2 bytes)
/// - qh: 4 bytes of high bits (1 bit per value)
/// - qs: 16 bytes of low 4 bits (2 values per byte)
/// Total: 22 bytes for 32 values (5.5 bits per weight)
pub const Q5_0_BLOCK_ELEMS: usize = 32;
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Q6_K: 256 elements per block, 210 bytes (128 ql + 64 qh + 16 scales + 2 d)
pub const Q6_K_BLOCK_ELEMS: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Q8_0: 32 elements per block, 34 bytes (2 scale + 32 int8)
pub const Q8_BLOCK_ELEMS: usize = 32;
pub const Q8_BLOCK_BYTES: usize = 34;

/// Q8_0 max value for quantization
pub const Q8_0_MAX: f32 = 127.0;

// ── f16 helpers ─────────────────────────────────────────────────────────────────

/// Load little-endian f16 from bytes and convert to f32.
#[inline(always)]
pub fn load_f16_scale(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f16::from_bits(bits).to_f32()
}

/// Store f32 as little-endian f16 bytes.
#[inline(always)]
pub fn store_f16_scale(scale: f32) -> [u8; 2] {
    f16::from_f32(scale).to_bits().to_le_bytes()
}

// ── F32 embedding lookup ─────────────────────────────────────────────────────────

/// Copy embedding row from F32 table: out = emb[token_id]
pub fn embed_f32(token_id: usize, emb: &[f32], out: &mut [f32]) {
    let hidden = out.len();
    let offset = token_id * hidden;
    out.copy_from_slice(&emb[offset..offset + hidden]);
}

/// Batch embed from F32: out[s] = emb[ids[s]]
pub fn embed_f32_batch(ids: &[u32], emb: &[f32], out: &mut [f32], hidden_size: usize) {
    for (s, &id) in ids.iter().enumerate() {
        let or = &mut out[s * hidden_size..(s + 1) * hidden_size];
        embed_f32(id as usize, emb, or);
    }
}

// ── Q4_0 embedding lookup ─────────────────────────────────────────────────────────

/// Dequantize Q4_0 embedding row: out = dequant(emb[token_id])
///
/// Q4_0 block: [f16 scale | 16 bytes of 4-bit pairs] = 18 bytes for 32 values
pub fn embed_q4_0(token_id: usize, emb: &[u8], out: &mut [f32], hidden_size: usize) {
    let num_blocks = hidden_size / Q4_BLOCK_ELEMS;
    let row_offset = token_id * num_blocks * Q4_BLOCK_BYTES;

    for b in 0..num_blocks {
        let block = &emb[row_offset + b * Q4_BLOCK_BYTES..row_offset + (b + 1) * Q4_BLOCK_BYTES];
        let scale = load_f16_scale(&block[0..2]);
        let qs = &block[2..18];
        let base = b * Q4_BLOCK_ELEMS;

        // Each byte holds two 4-bit values: lo nibble and hi nibble
        for i in 0..16 {
            // lo nibble -> element i, hi nibble -> element i+16
            let lo = ((qs[i] & 0x0F) as i32) - 8; // -8 to +7 range
            let hi = ((qs[i] >> 4) as i32) - 8;
            out[base + i] = lo as f32 * scale;
            out[base + i + 16] = hi as f32 * scale;
        }
    }
}

/// Batch embed from Q4_0
pub fn embed_q4_0_batch(ids: &[u32], emb: &[u8], out: &mut [f32], hidden_size: usize) {
    for (s, &id) in ids.iter().enumerate() {
        let or = &mut out[s * hidden_size..(s + 1) * hidden_size];
        embed_q4_0(id as usize, emb, or, hidden_size);
    }
}

/// Dequantize Q4_1 embedding row: out = dequant(emb[token_id])
///
/// Q4_1 block: [f16 scale | f16 min | 16 bytes of 4-bit pairs] = 20 bytes for 32 values
/// Values are in range [min, min + 15*scale]
pub fn embed_q4_1(token_id: usize, emb: &[u8], out: &mut [f32], hidden_size: usize) {
    let num_blocks = hidden_size / Q4_1_BLOCK_ELEMS;
    let row_offset = token_id * num_blocks * Q4_1_BLOCK_BYTES;

    for b in 0..num_blocks {
        let block = &emb[row_offset + b * Q4_1_BLOCK_BYTES..row_offset + (b + 1) * Q4_1_BLOCK_BYTES];
        let scale = load_f16_scale(&block[0..2]);
        let min = load_f16_scale(&block[2..4]);
        let qs = &block[4..20];
        let base = b * Q4_1_BLOCK_ELEMS;

        // Each byte holds two 4-bit values: lo nibble and hi nibble
        for i in 0..16 {
            // lo nibble -> element i, hi nibble -> element i+16
            let lo = (qs[i] & 0x0F) as f32; // 0 to 15
            let hi = (qs[i] >> 4) as f32;
            out[base + i] = lo * scale + min;
            out[base + i + 16] = hi * scale + min;
        }
    }
}

/// Batch embed from Q4_1
pub fn embed_q4_1_batch(ids: &[u32], emb: &[u8], out: &mut [f32], hidden_size: usize) {
    for (s, &id) in ids.iter().enumerate() {
        let or = &mut out[s * hidden_size..(s + 1) * hidden_size];
        embed_q4_1(id as usize, emb, or, hidden_size);
    }
}

/// Dequantize Q4_K embedding row: out = dequant(emb[token_id])
///
/// Q4_K block: [d f16 | dmin f16 | scales[12] | qs[128]] = 144 bytes for 256 values
/// Uses 4.5-bit quantization with multiple scales and mins per block.
pub fn embed_q4_k(token_id: usize, emb: &[u8], out: &mut [f32], hidden_size: usize) {
    let num_blocks = hidden_size / Q4_K_BLOCK_ELEMS;
    let row_offset = token_id * num_blocks * Q4_K_BLOCK_BYTES;

    for b in 0..num_blocks {
        let block = &emb[row_offset + b * Q4_K_BLOCK_BYTES..row_offset + (b + 1) * Q4_K_BLOCK_BYTES];
        let d = load_f16_scale(&block[0..2]);
        let dmin = load_f16_scale(&block[2..4]);
        let scales = &block[4..16]; // 12 bytes of packed 6-bit scales + mins
        let qs = &block[16..144]; // 128 bytes of 4-bit quants
        let base = b * Q4_K_BLOCK_ELEMS;

        // Helper to unpack scale and min (following llama.cpp's get_scale_min_k4)
        let get_scale_min = |j: usize| -> (i8, i8) {
            if j < 4 {
                let sc = ((scales[j] & 63) as i8).wrapping_sub(32);
                let m = ((scales[j + 4] & 63) as i8).wrapping_sub(32);
                (sc, m)
            } else {
                let sc = ((scales[j + 4] & 0xF) as i8 | (((scales[j - 4] >> 6) as i8) << 4)).wrapping_sub(32);
                let m = ((scales[j + 4] >> 4) as i8 | (((scales[j] >> 6) as i8) << 4)).wrapping_sub(32);
                (sc, m)
            }
        };

        // Dequantize 256 values, 32 at a time (8 groups of 32)
        for j in 0..8 {
            let offset = j * 32;
            for i in 0..32 {
                let q = (qs[(offset + i) / 2] >> (((offset + i) % 2) * 4)) & 0x0F;
                let (sc, m) = get_scale_min(j);
                let ls = (d * (sc as f32)) * (q as f32);
                let lm = dmin * (m as f32);
                out[base + offset + i] = ls + lm;
            }
        }
    }
}

/// Batch embed from Q4_K
pub fn embed_q4_k_batch(ids: &[u32], emb: &[u8], out: &mut [f32], hidden_size: usize) {
    for (s, &id) in ids.iter().enumerate() {
        let or = &mut out[s * hidden_size..(s + 1) * hidden_size];
        embed_q4_k(id as usize, emb, or, hidden_size);
    }
}

/// Dequantize Q6_K embedding row: out = dequant(emb[token_id])
///
/// Q6_K block: [128 ql | 64 qh | 16 scales | 2 f16 d] = 210 bytes for 256 values
/// Uses 2-bit quantization with multiple scales per block.
pub fn embed_q6_k(token_id: usize, emb: &[u8], out: &mut [f32], hidden_size: usize) {
    let num_blocks = hidden_size / Q6_K_BLOCK_ELEMS;
    let row_offset = token_id * num_blocks * Q6_K_BLOCK_BYTES;

    for b in 0..num_blocks {
        let block = &emb[row_offset + b * Q6_K_BLOCK_BYTES..row_offset + (b + 1) * Q6_K_BLOCK_BYTES];
        let d = load_f16_scale(&block[0..2]);
        let ql = &block[2..130]; // 128 bytes
        let qh = &block[130..194]; // 64 bytes
        let scales = &block[194..210]; // 16 bytes
        let base = b * Q6_K_BLOCK_ELEMS;

        // Process 128 values at a time
        for g in 0..2 {
            let offset = g * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from(((ql[l + offset] & 0xF) | (((qh[l] >> (is * 2)) & 3) << 4)) as i8) - 32;
                let q2 = i32::from(((ql[l + 32 + offset] & 0xF) | (((qh[l] >> (is * 2 + 2)) & 3) << 4)) as i8) - 32;
                let q3 = i32::from((((ql[l + offset] >> 4) & 0xF) | (((qh[l] >> (is * 2 + 4)) & 3) << 4)) as i8) - 32;
                let q4 = i32::from((((ql[l + 32 + offset] >> 4) & 0xF) | (((qh[l] >> (is * 2 + 6)) & 3) << 4)) as i8) - 32;

                let sc = scales[is * 4 + (l / 8)] as i8;

                out[base + l + offset] = d * (sc as f32) * (q1 as f32);
                out[base + l + 32 + offset] = d * (sc as f32) * (q2 as f32);
                out[base + l + 64 + offset] = d * (sc as f32) * (q3 as f32);
                out[base + l + 96 + offset] = d * (sc as f32) * (q4 as f32);
            }
        }
    }
}

/// Embed Q8_0 token: out = dequant(emb[token_id])
///
/// The embedding tensor has shape [hidden_size, vocab_size] = [896, 151936].
/// To embed token_id, we need the column vocab_id of the matrix.
/// Each column has hidden_size elements, stored in Q8_0 blocks.
pub fn embed_q8_0(token_id: usize, emb: &[u8], out: &mut [f32], hidden_size: usize) {
    let num_blocks = hidden_size / Q8_BLOCK_ELEMS;
    // For [hidden_size, vocab_size] layout, column vocab_id is at offset vocab_id * hidden_size
    let col_offset = token_id * num_blocks * Q8_BLOCK_BYTES;

    for b in 0..num_blocks {
        let block = &emb[col_offset + b * Q8_BLOCK_BYTES..col_offset + (b + 1) * Q8_BLOCK_BYTES];
        let scale = load_f16_scale(&block[0..2]);
        let qs = &block[2..34];
        let base = b * Q8_BLOCK_ELEMS;

        for i in 0..Q8_BLOCK_ELEMS {
            out[base + i] = (qs[i] as i8) as f32 * scale;
        }
    }
}

/// Batch embed from Q8_0
pub fn embed_q8_0_batch(ids: &[u32], emb: &[u8], out: &mut [f32], hidden_size: usize) {
    for (s, &id) in ids.iter().enumerate() {
        let or = &mut out[s * hidden_size..(s + 1) * hidden_size];
        embed_q8_0(id as usize, emb, or, hidden_size);
    }
}

/// Batch embed from Q6_K
pub fn embed_q6_k_batch(ids: &[u32], emb: &[u8], out: &mut [f32], hidden_size: usize) {
    for (s, &id) in ids.iter().enumerate() {
        let or = &mut out[s * hidden_size..(s + 1) * hidden_size];
        embed_q6_k(id as usize, emb, or, hidden_size);
    }
}

// ── Dequantization helpers for GEMV/GEMM ─────────────────────────────────────────

/// Dequantize one Q4_0 block to f32.
pub fn dequant_q4_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= Q4_BLOCK_BYTES, "block too small for Q4_0");
    debug_assert!(out.len() >= Q4_BLOCK_ELEMS, "output too small for Q4_0 block");

    let scale = load_f16_scale(&block[0..2]);
    let qs = &block[2..18];

    for i in 0..16 {
        let lo = ((qs[i] & 0x0F) as i32) - 8;
        let hi = ((qs[i] >> 4) as i32) - 8;
        out[i] = lo as f32 * scale;
        out[i + 16] = hi as f32 * scale;
    }
}

/// Dequantize one Q8_0 block to f32.
pub fn dequant_q8_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= Q8_BLOCK_BYTES, "block too small for Q8_0");
    debug_assert!(out.len() >= Q8_BLOCK_ELEMS, "output too small for Q8_0 block");

    let scale = load_f16_scale(&block[0..2]);
    let qs = &block[2..34];

    for i in 0..Q8_BLOCK_ELEMS {
        out[i] = (qs[i] as i8) as f32 * scale;
    }
}

// ── Q8_0 quantization for activations ────────────────────────────────────────

/// Quantize f32 values to Q8_0 bytes.
///
/// Returns quantized bytes and scale (f16).
pub fn quantize_f32_to_q8_0(src: &[f32], dst: &mut [u8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len(), "src and dst must have same length");

    // Find max absolute value to determine scale
    let max_val = src.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_val > 0.0 { max_val / Q8_0_MAX } else { 1.0 };
    let inv_scale = if scale > 0.0 { Q8_0_MAX / max_val } else { 0.0 };

    // Quantize: round to nearest, clamp to [-127, 127]
    for (i, &val) in src.iter().enumerate() {
        let q = (val * inv_scale).round() as i32;
        dst[i] = q.clamp(-127, 127) as u8 as i8 as u8;
    }

    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 100.0, 0.001];
        for &v in &values {
            let bytes = store_f16_scale(v);
            let loaded = load_f16_scale(&bytes);
            // f16 has ~3 decimal digits of precision
            assert!((v - loaded).abs() < v.abs() * 0.001 + 0.001, "f16 roundtrip failed for {}", v);
        }
    }

    #[test]
    fn embed_f32_copies_row() {
        let emb: Vec<f32> = (0..6).map(|i| i as f32).collect(); // 2 rows of 3
        let mut out = vec![0.0f32; 3];

        embed_f32(1, &emb, &mut out);

        assert_eq!(out, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn embed_q4_0_dequantizes_correctly() {
        // Create one Q4_0 block: scale=2.0
        // Each nibble stores 0..15, which dequantizes to (nibble - 8) * scale = -8..+7 * 2.0
        let mut block = [0u8; Q4_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x40; // f16 2.0

        // Pack: each byte has lo nibble for element i, hi nibble for element i+16
        for i in 0..16 {
            let lo = i as u8; // 0..15 for elements 0..15
            let hi = (15 - i) as u8; // 15..0 for elements 16..31 (reversed for variety)
            block[2 + i] = lo | (hi << 4);
        }

        let emb = &block[..];
        let mut out = vec![0.0f32; Q4_BLOCK_ELEMS];
        embed_q4_0(0, emb, &mut out, Q4_BLOCK_ELEMS);

        // Dequantized: (nibble - 8) * 2.0
        for i in 0..16 {
            let expected_lo = ((i as i32) - 8) as f32 * 2.0;
            let expected_hi = (((15 - i) as i32) - 8) as f32 * 2.0;
            assert!((out[i] - expected_lo).abs() < 0.01, "lo mismatch at {}: got {} expected {}", i, out[i], expected_lo);
            assert!((out[i + 16] - expected_hi).abs() < 0.01, "hi mismatch at {}: got {} expected {}", i, out[i + 16], expected_hi);
        }
    }
}
