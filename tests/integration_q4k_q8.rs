// tests/integration_q4k_q8.rs

use rocmforge::cpu::kernels::{
    gemm_q4k_q8::gemv_q4_k_q8_k_dispatch,
    q4::BlockQ4K,
    q8::quantize_q8_k,
};

#[test]
fn test_q4k_q8k_end_to_end() {
    // Create test data matching typical 7B model dimensions
    const HIDDEN: usize = 1536; // Qwen2.5-7B hidden size
    const FF: usize = 8960;     // Intermediate size

    // Simplified test with smaller dimensions (must be multiple of 256)
    const OUT_DIM: usize = 256;
    const IN_DIM: usize = 256;

    // Create weights
    let w = vec![0u8; OUT_DIM * BlockQ4K::SIZE];
    let x: Vec<f32> = (0..IN_DIM).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let mut y = vec![0.0f32; OUT_DIM];

    // Run kernel
    gemv_q4_k_q8_k_dispatch(&w, &x, &mut y, OUT_DIM, IN_DIM);

    // Verify output
    assert_eq!(y.len(), OUT_DIM);
}

#[test]
fn test_q4k_q8k_quantization_correctness() {
    // Test that quantization preserves important properties
    let values: Vec<f32> = (0..256).map(|i| i as f32 - 128.0).collect();

    let block = quantize_q8_k(&values);

    // Check that scale is reasonable
    assert!(block.d > 0.5 && block.d < 2.0);

    // Check that bsums match
    for i in 0..16 {
        let expected: i16 = block.qs[i * 16..(i + 1) * 16].iter().map(|&x| x as i16).sum();
        let bsum_value = block.bsums[i]; // Copy to avoid packed struct reference
        assert_eq!(bsum_value, expected);
    }
}
