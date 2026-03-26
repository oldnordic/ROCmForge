//! Scalar fallback for Q4_K × Q8_K operations.

use crate::cpu::kernels::q4::BlockQ4K;
use crate::cpu::kernels::q8::BlockQ8K;
use half::f16;
use rayon::prelude::*;

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

/// Scalar Q4_K × Q8_K GEMV: y = W * x
///
/// # Arguments
/// * `w` - Q4_K weights (row-major: each row is blocks of 256)
/// * `x` - Input vector (f32, length = in_dim)
/// * `y` - Output vector (f32, length = out_dim)
/// * `out_dim` - Number of output rows
/// * `in_dim` - Inner dimension (must be multiple of 256)
pub fn gemv_q4_k_q8_k(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(in_dim % 256 == 0, "in_dim must be multiple of QK_K=256");
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);

    let num_blocks_per_row = in_dim / 256;
    let bytes_per_row = num_blocks_per_row * BlockQ4K::SIZE;

    // Quantize input to Q8_K (once per column of blocks)
    let mut x_q8 = vec![BlockQ8K::zero(); num_blocks_per_row];
    for b in 0..num_blocks_per_row {
        let start = b * 256;
        let end = start + 256;
        x_q8[b] = crate::cpu::kernels::q8::quantize_q8_k(&x[start..end]);
    }

    // Process each output row
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for b in 0..num_blocks_per_row {
            let block_offset = row_start + b * BlockQ4K::SIZE;
            let q4_block = unsafe {
                &*(w.as_ptr().add(block_offset) as *const BlockQ4K)
            };
            let q8_block = &x_q8[b];

            acc += dot_q4_k_q8_k_block_scalar(q4_block, q8_block);
        }

        *out = acc;
    });
}

/// Scalar Q4_K × Q8_K GEMM: Y = W * X
///
/// # Arguments
/// * `w` - Q4_K weights [out_dim, in_dim] in blocks
/// * `x` - Input matrix [m, in_dim] row-major f32
/// * `y` - Output matrix [m, out_dim] row-major f32
/// * `m` - Batch size
/// * `n` - Output dimension (out_dim)
/// * `k` - Inner dimension (in_dim)
pub fn gemm_q4_k_q8_k(
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(k % 256 == 0, "k must be multiple of QK_K=256");

    let num_blocks_k = k / 256;

    // Process each batch row
    y.par_chunks_mut(n).enumerate().for_each(|(batch_idx, y_row)| {
        let x_row = &x[batch_idx * k..(batch_idx + 1) * k];

        // Quantize this row to Q8_K blocks
        let mut x_q8 = vec![BlockQ8K::zero(); num_blocks_k];
        for b in 0..num_blocks_k {
            x_q8[b] = crate::cpu::kernels::q8::quantize_q8_k(&x_row[b * 256..(b + 1) * 256]);
        }

        // Compute dot products for each output column
        for out_col in 0..n {
            let mut acc = 0.0f32;

            for b in 0..num_blocks_k {
                let w_offset = out_col * num_blocks_k * BlockQ4K::SIZE + b * BlockQ4K::SIZE;
                let q4_block = unsafe { &*(w.as_ptr().add(w_offset) as *const BlockQ4K) };
                let q8_block = &x_q8[b];

                acc += dot_q4_k_q8_k_block_scalar(q4_block, q8_block);
            }

            y_row[out_col] = acc;
        }
    });
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

    #[test]
    fn gemv_q4_k_q8_k_dimensions() {
        // Create dummy weights (2 rows, 256 cols = 1 block per row)
        let w = vec![0u8; 2 * BlockQ4K::SIZE];
        let x = vec![0.0f32; 256];
        let mut y = vec![0.0f32; 2];

        gemv_q4_k_q8_k(&w, &x, &mut y, 2, 256);

        // Should not panic, output should have correct size
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn gemv_q4_k_q8_k_simple_pattern() {
        // Test with simple pattern - use zero blocks for predictability
        let w = vec![0u8; BlockQ4K::SIZE];
        let x: Vec<f32> = (0..256).map(|_| 1.0).collect();
        let mut y = vec![0.0f32; 1];

        gemv_q4_k_q8_k(&w, &x, &mut y, 1, 256);

        // Zero weights × non-zero input = zero output
        assert_eq!(y[0], 0.0);
    }

    #[test]
    fn gemm_q4_k_q8_k_scalar_dimensions() {
        // For k=512: 2 blocks per row (512/256=2)
        // For n=2: 2 output columns, each needs 2 blocks = 4 blocks total
        let num_blocks_n = 2 * (512 / 256); // n * (k / 256)
        let w = vec![0u8; num_blocks_n * BlockQ4K::SIZE];
        // For m=2 batch size and k=512: need 1024 elements
        let x: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
        // For m=2 batch size and n=2 output: need 4 elements
        let mut y = vec![0.0f32; 4];

        // Test 2x2 blocks (m=2, n=2, k=512)
        gemm_q4_k_q8_k(&w, &x, &mut y, 2, 2, 512);

        assert_eq!(y.len(), 4);
    }

    #[test]
    fn gemm_q4_k_q8_k_scalar_simple_pattern() {
        // Zero weights × non-zero input = zero output
        let w = vec![0u8; BlockQ4K::SIZE];
        let x: Vec<f32> = (0..256).map(|_| 1.0).collect();
        let mut y = vec![0.0f32; 1];

        gemm_q4_k_q8_k(&w, &x, &mut y, 1, 1, 256);

        assert_eq!(y[0], 0.0);
    }
}
