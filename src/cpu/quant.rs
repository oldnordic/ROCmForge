//! Quantization support for GGML weight types.
//!
//! Supports Q4_0, Q4_1, Q8_0 dequantization and embedding table lookup.

use half::f16;

// ── Block sizes ────────────────────────────────────────────────────────────────

/// Q4_0: 32 elements per block, 18 bytes (2 scale + 16 nibbles)
pub const Q4_BLOCK_ELEMS: usize = 32;
pub const Q4_BLOCK_BYTES: usize = 18;

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
        let block = &emb[row_offset + b * Q4_BLOCK_BYTES..];
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
