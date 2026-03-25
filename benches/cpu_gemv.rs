use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rocmforge::cpu::ops::{gemv_q4_0_q8_0, gemv_q4_0};

/// Benchmark Q4_0 × Q8_0 GEMV.
fn bench_q4_0_q8_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("Q4_0×Q8_0");

    // Test with hidden size matching Qwen2.5-0.5B (2048)
    let hidden_size = 2048;
    let out_dim = 2048;
    let num_blocks = hidden_size / 32; // Q4_0 blocks

    // Generate fake weights
    let mut w = vec![0u8; out_dim * num_blocks * 18];

    for row in 0..out_dim {
        let row_offset = row * num_blocks * 18;
        for b in 0..num_blocks {
            let block_offset = row_offset + b * 18;
            w[block_offset] = 0x00; // scale byte 1
            w[block_offset + 1] = 0x40; // scale byte 2 (scale = 2.0)
            for j in 2..18 {
                // pack two 4-bit values: lo nibble for element j, hi nibble for element j+16
                // value j in row = (j % 32) -> index in block
                let lo = j; // 0..15
                let hi = 15 - j; // 15..0 reversed
                w[block_offset + j] = (lo | (hi << 4)) as u8;
            }
        }
    }

    // Generate fake input
    let x: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.01).collect();

    // Warmup
    gemv_q4_0_q8_0(&w, &x, &mut vec![0.0f32; out_dim], out_dim, hidden_size);

    group.bench_function("Q4_0×Q8_0", |b, mut output| {
        gemv_q4_0_q8_0(&w, &x, &mut *output, out_dim, hidden_size);
        black_box(output)
    });
}

fn bench_q4_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("Q4_0×f32");

    let mut w = vec![0u8; out_dim * num_blocks * 18];

    for row in 0..out_dim {
        let row_offset = row * num_blocks * 18;
        for b in 0..num_blocks {
            let block_offset = row_offset + b * 18;
            w[block_offset] = 0x00; // scale byte 1
            w[block_offset + 1] = 0x40; // scale byte 2 (scale = 2.0)
            for j in 2..18 {
                w[block_offset + j] = (j - 2) as u8; // values 0..15
            }
        }
    }

    let x: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.01).collect();

    group.bench_function("Q4_0×f32", |b, mut output| {
        gemv_q4_0(&w, &x, &mut *output, out_dim, hidden_size);
        black_box(output)
    });
}

criterion_main!(c);
