// examples/test_real_weights.rs
// Test Q4_K × Q8_K kernels with real model weights

use rocmforge::cpu::kernels::{
    gemm_q4k_q8::gemv_q4_k_q8_k_dispatch, q4::BlockQ4K, q8::quantize_q8_k,
};
use std::time::Instant;

fn main() {
    println!("Q4_K × Q8_K Kernel Test with Real-ish Weights");
    println!("==============================================\n");

    // Check CPU features at runtime
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            println!("✓ AVX2+FMA detected - using optimized kernels\n");
        } else {
            println!("✗ AVX2 not available - using scalar fallback\n");
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("? Not x86_64 - using available SIMD\n");
    }

    // Simulate a real Q4_K weight matrix (1024 x 1024 for testing)
    const OUT_DIM: usize = 1024;
    const IN_DIM: usize = 1024; // Must be multiple of 256 for current impl
    const NUM_BLOCKS: usize = IN_DIM / 256;

    println!("Testing with dimensions: {} x {}", OUT_DIM, IN_DIM);
    println!("Number of blocks: {}\n", NUM_BLOCKS * OUT_DIM);

    // Create Q4_K weight blocks with some pattern
    let mut w = vec![0u8; OUT_DIM * NUM_BLOCKS * BlockQ4K::SIZE];

    // Fill with some pattern to simulate real weights
    for i in 0..w.len() {
        w[i] = ((i as u32).wrapping_mul(0x9E3779B9) & 0xFF) as u8;
    }

    // Create input with pattern (simulating activation values)
    let x: Vec<f32> = (0..IN_DIM)
        .map(|i| {
            // Simulate activations with some variation
            let angle = i as f32 * 0.1;
            angle.sin() * 0.5 + angle.cos() * 0.3
        })
        .collect();

    let mut y = vec![0.0f32; OUT_DIM];

    println!(
        "Input range: {:.3} to {:.3}",
        x.iter().cloned().fold(f32::INFINITY, f32::min),
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Warmup
    gemv_q4_k_q8_k_dispatch(&w, &x, &mut y, OUT_DIM, IN_DIM);

    // Benchmark Q4_K × Q8_K GEMV
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        gemv_q4_k_q8_k_dispatch(&w, &x, &mut y, OUT_DIM, IN_DIM);
    }
    let elapsed = start.elapsed();

    let ms_per_call = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let calls_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("\nQ4_K × Q8_K GEMV Performance:");
    println!("  {:.3} ms per call", ms_per_call);
    println!("  {:.1} calls/sec", calls_per_sec);

    // Calculate theoretical tokens/sec for decode
    // Each token generation requires ~2 * num_layers matmuls
    // For 24 layers: 48 matmuls per token
    let num_layers = 24;
    let ms_per_token = ms_per_call * 2.0 * num_layers as f64;
    let tokens_per_sec = 1000.0 / ms_per_token;

    println!("\nEstimated Decode Performance ({} layers):", num_layers);
    println!("  {:.2} ms per token", ms_per_token);
    println!("  {:.1} tokens/sec", tokens_per_sec);

    // Verify output is reasonable (not all zeros, not NaN)
    let y_min = y.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_sum: f32 = y.iter().sum();

    println!("\nOutput verification:");
    println!("  Range: [{:.3}, {:.3}]", y_min, y_max);
    println!("  Sum: {:.3}", y_sum);
    println!("  All valid: {}", y.iter().all(|v| v.is_finite()));
}
