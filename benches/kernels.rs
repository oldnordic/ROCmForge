//! Criterion kernel benchmarks.
//!
//! Microbenchmarks for GEMV and GEMM kernels with statistical analysis.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rocmforge::cpu::kernels::{
    gemm_q4k_q8::{gemm_q4_k_q8_k_dispatch_gemm, gemv_q4_k_q8_k_dispatch},
    gemm_q4k_q8_scalar::gemv_q4_k_q8_k as gemv_q4_k_q8_k_scalar,
    q4::BlockQ4K,
};

fn bench_gemv_q4k_q8(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_q4k_q8");

    for size in [256, 512, 768, 1024].iter() {
        let out_dim = *size;
        let in_dim = *size;
        let num_blocks = in_dim / 256;

        // Create test data
        let w = vec![0u8; out_dim * num_blocks * BlockQ4K::SIZE];
        let x: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.01).collect();
        let mut y = vec![0.0f32; out_dim];

        group.bench_with_input(BenchmarkId::new("gemv", size), size, |b, &_size| {
            b.iter(|| {
                gemv_q4_k_q8_k_dispatch(
                    black_box(&w),
                    black_box(&x),
                    black_box(&mut y),
                    out_dim,
                    in_dim,
                );
                black_box(&y);
            });
        });
    }

    group.finish();
}

fn bench_gemv_scalar_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_scalar_comparison");

    for size in [256, 512].iter() {
        let out_dim = *size;
        let in_dim = *size;
        let num_blocks = in_dim / 256;

        let w = vec![0u8; out_dim * num_blocks * BlockQ4K::SIZE];
        let x: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.01).collect();
        let mut y_dispatch = vec![0.0f32; out_dim];
        let mut y_scalar = vec![0.0f32; out_dim];

        group.bench_with_input(BenchmarkId::new("dispatch", size), size, |b, &_size| {
            b.iter(|| {
                gemv_q4_k_q8_k_dispatch(
                    black_box(&w),
                    black_box(&x),
                    black_box(&mut y_dispatch),
                    out_dim,
                    in_dim,
                );
                black_box(&y_dispatch);
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &_size| {
            b.iter(|| {
                gemv_q4_k_q8_k_scalar(
                    black_box(&w),
                    black_box(&x),
                    black_box(&mut y_scalar),
                    out_dim,
                    in_dim,
                );
                black_box(&y_scalar);
            });
        });
    }

    group.finish();
}

fn bench_gemm_q4k_q8(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_q4k_q8");

    for size in [256, 512, 768].iter() {
        let num_blocks_k = size / 256;
        let m = 16; // batch size
        let n = *size;
        let k = *size;

        let w = vec![0u8; n * num_blocks_k * BlockQ4K::SIZE];
        let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let mut y = vec![0.0f32; m * n];

        group.bench_with_input(BenchmarkId::new("gemv", size), size, |b, &_size| {
            b.iter(|| {
                gemm_q4_k_q8_k_dispatch_gemm(
                    black_box(&w),
                    black_box(&x),
                    black_box(&mut y),
                    m,
                    n,
                    k,
                );
                black_box(&y);
            });
        });
    }

    group.finish();
}

criterion_group!(
    kernels,
    bench_gemv_q4k_q8,
    bench_gemv_scalar_comparison,
    bench_gemm_q4k_q8
);
criterion_main!(kernels);
