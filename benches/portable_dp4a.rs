//! Portable DP4A performance benchmarks
//!
//! Measures performance difference between hardware DP4A (RDNA2)
//! and software fallback (RDNA3).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rocmforge::gpu::kernels::quant::{bench_dot4_hardware, bench_dot4_manual};
use rocmforge::gpu::GpuDevice;

#[cfg(feature = "gpu")]
fn bench_dot4_performance(c: &mut Criterion) {
    let ctx = GpuDevice::init(0).unwrap();
    let device_name = ctx.get_name().unwrap_or_default();

    let mut group = c.benchmark_group("dot4_implementation");

    // Test various packed int8 vectors
    let test_cases: Vec<(&str, i32, i32)> = vec![
        ("simple_positive", 0x00010203, 0x01010101),
        ("simple_negative", (0xFFFFFFFF_u32) as i32, 0x01010101),
        ("mixed", 0x7F808080, 0x01010101),
        ("random_1", 0x12345678, (0x9ABCDEF0_u32) as i32),
        ("random_2", (0xFEDCBA98_u32) as i32, 0x76543210),
    ];

    for (name, a_packed, b_packed) in test_cases {
        // Benchmark hardware DP4A (RDNA2 only)
        if device_name.contains("gfx1030") {
            group.bench_with_input(
                BenchmarkId::new("hardware", name),
                &(a_packed, b_packed),
                |bencher, &(a, b)| {
                    bencher.iter(|| {
                        black_box(unsafe { bench_dot4_hardware(black_box(a), black_box(b), 0) })
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("manual_rdna2", name),
                &(a_packed, b_packed),
                |bencher, &(a, b)| {
                    bencher.iter(|| {
                        black_box(unsafe { bench_dot4_manual(black_box(a), black_box(b), 0) })
                    })
                },
            );
        }

        // Benchmark manual on RDNA3
        if device_name.contains("gfx1100") {
            group.bench_with_input(
                BenchmarkId::new("manual_rdna3", name),
                &(a_packed, b_packed),
                |bencher, &(a, b)| {
                    bencher.iter(|| {
                        black_box(unsafe { bench_dot4_manual(black_box(a), black_box(b), 0) })
                    })
                },
            );
        }
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_full_kernel_throughput(c: &mut Criterion) {
    let ctx = GpuDevice::init(0).unwrap();
    let device_name = ctx.get_name().unwrap_or_default();

    let mut group = c.benchmark_group("dp4a_kernel_throughput");

    // Benchmark actual fused kernel with different sizes
    for seq_len in [128, 256, 512, 1024, 2048] {
        let bench_name = format!("seq_len_{}", seq_len);

        if device_name.contains("gfx1030") || device_name.contains("gfx1100") {
            group.bench_with_input(
                BenchmarkId::from_parameter(bench_name),
                &seq_len,
                |b, &seq_len| {
                    b.iter(|| {
                        // Launch full fused kernel
                        // This is a placeholder - actual implementation will call the real kernel
                        black_box(seq_len);
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_dot4_performance, bench_full_kernel_throughput
}

criterion_main!(benches);
