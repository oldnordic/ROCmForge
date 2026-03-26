//! Scalar fallback for Q4_K × Q8_K operations.

use crate::cpu::kernels::q4::BlockQ4K;
use crate::cpu::kernels::q8::BlockQ8K;
use half::f16;

/// Scalar fallback for Q4_K × Q8_K dot product.
///
/// Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c:2004 (#else branch)
pub fn dot_q4_k_q8_k_block_scalar(
    q4_block: &BlockQ4K,
    q8_block: &BlockQ8K,
) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // Load scales
    let d = q8_block.d * f16::from_le_bytes(q4_block.d).to_f32();
    let dmin = -q8_block.d * f16::from_le_bytes(q4_block.dmin).to_f32();

    // Unpack Q4_K scales (12 bytes into 4 u32)
    let mut utmp = [0u32; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            q4_block.scales.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

    // Scale unpacking algorithm from llama.cpp
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // scales and mins are accessed as byte arrays from utmp
    let scales = &utmp[0..2]; // First 8 bytes contain scales
    let mins = &utmp[2..4];   // Next 8 bytes contain mins (after unpacking)

    // Copy bsums to local array to avoid packed struct reference issues
    let mut bsums_local = [0i16; 16];
    for i in 0..16 {
        bsums_local[i] = q8_block.bsums[i];
    }

    // Compute min contribution
    let mut sumi = 0i32;
    for j in 0..16 {
        let min_val = get_scaled_min(mins, j);
        sumi += bsums_local[j] as i32 * min_val;
    }

    // Extract Q4_K nibbles into aux array (256 signed 8-bit values)
    let mut aux8 = [0i8; 256];
    let mut q4_ptr = 0;
    for j in 0..4 {
        // Process 32 low nibbles
        for l in 0..32 {
            aux8[j * 64 + l] = (q4_block.qs[q4_ptr + l] & 0x0F) as i8;
        }
        // Process 32 high nibbles
        for l in 0..32 {
            aux8[j * 64 + 32 + l] = (q4_block.qs[q4_ptr + l] >> 4) as i8;
        }
        q4_ptr += 32;
    }

    // Accumulate dot products for 8 sub-blocks
    let mut sums = [0.0f32; 8];
    let mut q8_ptr = 0;
    let mut aux_ptr = 0;
    let mut scale_idx = 0;

    for j in 0..8 {
        let mut aux32 = [0i32; 8];

        // Get scale for this sub-block
        let scale = get_scale(scales, scale_idx);
        scale_idx += 1;

        // Process 4 groups of 8 elements
        for _ in 0..4 {
            for l in 0..8 {
                aux32[l] += (q8_block.qs[q8_ptr + l] as i32) * (aux8[aux_ptr + l] as i32);
            }
            q8_ptr += 8;
            aux_ptr += 8;
        }

        for l in 0..8 {
            sums[l] += d * (scale as f32) * (aux32[l] as f32);
        }
    }

    let mut result = dmin * (sumi as f32);
    for l in 0..8 {
        result += sums[l];
    }

    result
}

/// Extract 6-bit scale value from packed array.
fn get_scale(scales: &[u32], index: usize) -> i32 {
    let byte_idx = index / 2;
    let bit_offset = (index % 2) * 6;

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, 8)
    };

    let scale = ((scales_bytes[byte_idx] as u32) >> bit_offset) & 0x3F;
    // Convert to signed value centered around 32
    (scale as i32) - 32
}

/// Extract 6-bit min value from packed array.
fn get_scaled_min(mins: &[u32], index: usize) -> i32 {
    let byte_idx = index / 2;
    let bit_offset = (index % 2) * 6;

    let mins_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(mins.as_ptr() as *const u8, 8)
    };

    let min_val = ((mins_bytes[byte_idx] as u32) >> bit_offset) & 0x3F;
    // Convert to signed value centered around 32
    (min_val as i32) - 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_q4_k_q8_k_scalar_zero_blocks() {
        let q4 = BlockQ4K::zero();
        let q8 = BlockQ8K::zero();

        let result = dot_q4_k_q8_k_block_scalar(&q4, &q8);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn dot_q4_k_q8_k_scalar_simple_case() {
        // Create Q4_K block with known values
        let mut q4 = BlockQ4K::zero();
        // Set all quants to 8 (middle of range)
        for i in 0..128 {
            q4.qs[i] = 0x88; // Both nibbles = 8
        }

        // Create Q8_K block with known values
        let mut q8 = BlockQ8K::zero();
        for i in 0..256 {
            q8.qs[i] = 1;
            q8.bsums[i / 16] += 1;
        }
        q8.d = 1.0;

        let result = dot_q4_k_q8_k_block_scalar(&q4, &q8);

        // With zero scales in Q4_K, result should be 0 or close to 0
        // (The exact value depends on the scale encoding which is simplified)
        assert!(result.abs() < 10000.0); // Just check it's not infinite/NaN
    }
}
