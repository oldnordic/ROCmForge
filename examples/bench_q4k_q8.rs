// examples/bench_q4k_q8.rs

use rocmforge::cpu::kernels::{gemm_q4k_q8::gemv_q4_k_q8_k_dispatch, q4::BlockQ4K};
use std::time::Instant;

fn main() {
    println!("Q4_K × Q8_K Kernel Benchmarks");
    println!("==============================\n");

    // Benchmark GEMV
    bench_gemv();

    // Benchmark GEMM
    bench_gemm();
}

fn bench_gemv() {
    const OUT_DIM: usize = 1024;  // Typical hidden size (must be multiple of 256)
    const IN_DIM: usize = 1024;   // Must be multiple of 256

    // Create dummy weights
    let num_blocks = IN_DIM / 256;
    let w = vec![0u8; OUT_DIM * num_blocks * BlockQ4K::SIZE];

    // Create random input
    let x: Vec<f32> = (0..IN_DIM).map(|i| i as f32 * 0.01).collect();
    let mut y = vec![0.0f32; OUT_DIM];

    // Warmup
    gemv_q4_k_q8_k_dispatch(&w, &x, &mut y, OUT_DIM, IN_DIM);

    // Benchmark
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        gemv_q4_k_q8_k_dispatch(&w, &x, &mut y, OUT_DIM, IN_DIM);
    }
    let elapsed = start.elapsed();

    let ms_per_call = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let calls_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("GEMV ({} x {}):", OUT_DIM, IN_DIM);
    println!("  {:.3} ms per call", ms_per_call);
    println!("  {:.1} calls/sec", calls_per_sec);
    println!();
}

fn bench_gemm() {
    const M: usize = 16;   // Batch size
    const N: usize = 1024; // Hidden size (must be multiple of 256)
    const K: usize = 1024; // Must be multiple of 256

    let num_blocks_k = K / 256;
    let w = vec![0u8; N * num_blocks_k * BlockQ4K::SIZE];
    let x: Vec<f32> = (0..M * K).map(|i| i as f32 * 0.01).collect();
    let mut y = vec![0.0f32; M * N];

    // Warmup
    rocmforge::cpu::kernels::gemm_q4k_q8::gemm_q4_k_q8_k_dispatch_gemm(&w, &x, &mut y, M, N, K);

    // Benchmark
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        rocmforge::cpu::kernels::gemm_q4k_q8::gemm_q4_k_q8_k_dispatch_gemm(&w, &x, &mut y, M, N, K);
    }
    let elapsed = start.elapsed();

    let ms_per_call = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let calls_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("GEMM ({} x {} x {}):", M, N, K);
    println!("  {:.3} ms per call", ms_per_call);
    println!("  {:.1} calls/sec", calls_per_sec);
}
