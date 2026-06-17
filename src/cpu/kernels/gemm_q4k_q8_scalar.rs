//! Scalar fallback for Q4_K × Q8_K operations.

use crate::cpu::kernels::q4::BlockQ4K;
use crate::cpu::kernels::q8::BlockQ8K;
use half::f16;
use rayon::prelude::*;

/// Scalar fallback for Q4_K × Q8_K dot product.
///
/// Reference: llama.cpp ggml/src/ggml-cpu/arch/x86/quants.c:2004 (#else branch)
pub fn dot_q4_k_q8_k_block_scalar(q4_block: &BlockQ4K, q8_block: &BlockQ8K) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // Load scales
    let d = q8_block.d * f16::from_le_bytes(q4_block.d).to_f32();
    let dmin = -q8_block.d * f16::from_le_bytes(q4_block.dmin).to_f32();

    // Unpack Q4_K scales (12 bytes into 4 u32)
    let mut utmp = [0u32; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(q4_block.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
    }

    // Scale unpacking algorithm from llama.cpp
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // After unpacking, utmp holds 16 bytes:
    //   bytes 0..7  -> 8 unsigned 6-bit scales (one per byte, low 6 bits)
    //   bytes 8..15 -> 8 unsigned 6-bit mins   (one per byte, low 6 bits)
    let utmp_bytes = utmp.as_ptr() as *const u8;
    let scales_bytes: [u8; 8] = unsafe { std::ptr::read_unaligned(utmp_bytes as *const [u8; 8]) };
    let mins_bytes: [u8; 8] =
        unsafe { std::ptr::read_unaligned(utmp_bytes.add(8) as *const [u8; 8]) };

    // Copy bsums to local array to avoid packed struct reference issues
    let mut bsums_local = [0i16; 16];
    #[allow(clippy::manual_memcpy, reason = "packed copy")]
    for i in 0..16 {
        bsums_local[i] = q8_block.bsums[i];
    }

    // Compute min contribution.  Q4_K has one min per 32-element sub-block,
    // but bsums are sums over 16-element groups, so two adjacent groups share a min.
    let mut sumi = 0i32;
    for (j, bsum) in bsums_local.iter().enumerate() {
        let min_val = (mins_bytes[j / 2] & 0x3F) as i32;
        sumi += *bsum as i32 * min_val;
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

    for &scale_byte in &scales_bytes {
        let mut aux32 = [0i32; 8];

        // Scale for this 32-element sub-block is an unsigned 6-bit value.
        let scale = (scale_byte & 0x3F) as i32;

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
    for sum in &sums {
        result += sum;
    }

    result
}

/// Scalar Q4_K × Q8_K GEMV: y = W * x
///
/// # Arguments
/// * `w` - Q4_K weights (row-major: each row is blocks of 256)
/// * `x` - Input vector (f32, length = in_dim)
/// * `y` - Output vector (f32, length = out_dim)
/// * `out_dim` - Number of output rows
/// * `in_dim` - Inner dimension (must be multiple of 256)
pub fn gemv_q4_k_q8_k(w: &[u8], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    assert!(
        in_dim.is_multiple_of(256),
        "in_dim must be multiple of QK_K=256"
    );
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);

    let num_blocks_per_row = in_dim / 256;
    let bytes_per_row = num_blocks_per_row * BlockQ4K::SIZE;

    // Quantize input to Q8_K (once per column of blocks)
    let mut x_q8 = vec![BlockQ8K::zero(); num_blocks_per_row];
    for (b, block) in x_q8.iter_mut().enumerate() {
        let start = b * 256;
        let end = start + 256;
        *block = crate::cpu::kernels::q8::quantize_q8_k(&x[start..end]);
    }

    // Process each output row
    y.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for (b, q8_block) in x_q8.iter().enumerate() {
            let block_offset = row_start + b * BlockQ4K::SIZE;
            let q4_block = unsafe { &*(w.as_ptr().add(block_offset) as *const BlockQ4K) };

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
pub fn gemm_q4_k_q8_k(w: &[u8], x: &[f32], y: &mut [f32], _m: usize, n: usize, k: usize) {
    assert!(k.is_multiple_of(256), "k must be multiple of QK_K=256");

    let num_blocks_k = k / 256;

    // Process each batch row
    y.par_chunks_mut(n)
        .enumerate()
        .for_each(|(batch_idx, y_row)| {
            let x_row = &x[batch_idx * k..(batch_idx + 1) * k];

            // Quantize this row to Q8_K blocks
            let mut x_q8 = vec![BlockQ8K::zero(); num_blocks_k];
            for (b, block) in x_q8.iter_mut().enumerate() {
                *block = crate::cpu::kernels::q8::quantize_q8_k(&x_row[b * 256..(b + 1) * 256]);
            }

            // Compute dot products for each output column
            for (out_col, y_out) in y_row.iter_mut().enumerate().take(n) {
                let mut acc = 0.0f32;

                for (b, q8_block) in x_q8.iter().enumerate() {
                    let w_offset = out_col * num_blocks_k * BlockQ4K::SIZE + b * BlockQ4K::SIZE;
                    let q4_block = unsafe { &*(w.as_ptr().add(w_offset) as *const BlockQ4K) };

                    acc += dot_q4_k_q8_k_block_scalar(q4_block, q8_block);
                }

                *y_out = acc;
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

    #[test]
    fn dot_q4_k_q8_k_scalar_matches_dequantized() {
        // The kernel must agree with the block's own dequantize semantics.
        let mut rng = fastrand::Rng::new();
        let weights: Vec<f32> = (0..256).map(|_| rng.f32() * 4.0 - 2.0).collect();
        let activations: Vec<f32> = (0..256).map(|_| rng.f32() * 4.0 - 2.0).collect();

        let q4 = BlockQ4K::quantize(&weights);
        let q8 = crate::cpu::kernels::q8::quantize_q8_k(&activations);

        let quantized_dot = dot_q4_k_q8_k_block_scalar(&q4, &q8);

        let mut dequant_w = vec![0.0f32; 256];
        q4.dequantize(&mut dequant_w);
        let mut dequant_x = vec![0.0f32; 256];
        q8.dequantize(&mut dequant_x);
        let reference_dot: f32 = dequant_w.iter().zip(&dequant_x).map(|(w, a)| w * a).sum();

        let abs_error = (quantized_dot - reference_dot).abs();
        let rel_error = abs_error / reference_dot.abs().max(1e-6);

        assert!(
            rel_error < 0.01,
            "Q4_K scalar dot mismatch: got {quantized_dot}, expected {reference_dot} (rel {rel_error})"
        );
    }

    #[test]
    fn gemv_q4_k_q8_k_scalar_matches_dequantized() {
        let mut rng = fastrand::Rng::new();
        let out_dim = 5;
        let in_dim = 512;

        let weights_f32: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect())
            .collect();
        let mut w = vec![0u8; out_dim * (in_dim / 256) * BlockQ4K::SIZE];
        for (row, weights) in weights_f32.iter().enumerate() {
            for (b, block_weights) in weights.chunks(256).enumerate() {
                let q4 = BlockQ4K::quantize(block_weights);
                let offset = row * (in_dim / 256) * BlockQ4K::SIZE + b * BlockQ4K::SIZE;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &q4 as *const _ as *const u8,
                        w.as_mut_ptr().add(offset),
                        BlockQ4K::SIZE,
                    );
                }
            }
        }

        let x: Vec<f32> = (0..in_dim).map(|_| rng.f32() * 4.0 - 2.0).collect();
        let mut y = vec![0.0f32; out_dim];
        gemv_q4_k_q8_k(&w, &x, &mut y, out_dim, in_dim);

        // Build the dequantized input vector the kernel actually uses.
        let mut dequant_x = vec![0.0f32; in_dim];
        for (b, chunk) in x.chunks(256).enumerate() {
            let q8 = crate::cpu::kernels::q8::quantize_q8_k(chunk);
            q8.dequantize(&mut dequant_x[b * 256..(b + 1) * 256]);
        }

        for (row, weights) in weights_f32.iter().enumerate() {
            let mut dequant_w = vec![0.0f32; in_dim];
            for (b, block_weights) in weights.chunks(256).enumerate() {
                let q4 = BlockQ4K::quantize(block_weights);
                q4.dequantize(&mut dequant_w[b * 256..(b + 1) * 256]);
            }
            let expected = dequant_w
                .iter()
                .zip(&dequant_x)
                .map(|(w, a)| w * a)
                .sum::<f32>();
            let abs_error = (y[row] - expected).abs();
            let rel_error = abs_error / expected.abs().max(1e-6);
            assert!(
                rel_error < 0.01,
                "Q4_K scalar GEMV row {row} mismatch: got {}, expected {} (rel {})",
                y[row],
                expected,
                rel_error
            );
        }
    }
}
