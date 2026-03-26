/// Simple benchmark for GEMV performance comparison.

use rocmforge::cpu::ops::{gemv_q4_0, gemv_q4_0_q8_0};
use rocmforge::cpu::quant::{load_f16_scale, store_f16_scale};

fn main() {
    const H: usize = 896;
    const OUT_DIM: usize = 896;
    const NUM_BLOCKS: usize = H / 32;

    // Create fake Q4_0 weights
    let mut w = vec![0u8; OUT_DIM * NUM_BLOCKS * 18];
    for i in 0..OUT_DIM {
        for b in 0..NUM_BLOCKS {
            let offset = i * NUM_BLOCKS * 18 + b * 18;
            w[offset] = 0x00; // scale f16
            w[offset + 1] = 0x40; // scale = 2.0
            for j in 0..16 {
                w[offset + 2 + j] = (j % 16) as u8 | (((15 - j) % 16) as u8) << 4;
            }
        }
    }

    // Create fake input
    let x: Vec<f32> = (0..H).map(|i| i as f32 * 0.01).collect();
    let mut y_f32 = vec![0.0f32; OUT_DIM];
    let mut y_q8 = vec![0.0f32; OUT_DIM];

    // Warmup
    gemv_q4_0(&w, &x, &mut y_f32, OUT_DIM, H);
    gemv_q4_0_q8_0(&w, &x, &mut y_q8, OUT_DIM, H, None);

    // Benchmark Q4_0 × f32
    let start = std::time::Instant::now();
    let iterations = 1000;
    for _ in 0..iterations {
        gemv_q4_0(&w, &x, &mut y_f32, OUT_DIM, H);
    }
    let elapsed_f32 = start.elapsed();
    let tps_f32 = (iterations as f64 * 1000.0) / elapsed_f32.as_secs_f64();

    // Benchmark Q4_0 × Q8_0
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        gemv_q4_0_q8_0(&w, &x, &mut y_q8, OUT_DIM, H, None);
    }
    let elapsed_q8 = start.elapsed();
    let tps_q8 = (iterations as f64 * 1000.0) / elapsed_q8.as_secs_f64();

    println!("Q4_0 × f32: {:.2} ms per GEMV ({:.1} calls/s)", elapsed_f32.as_millis() as f64 / iterations as f64, tps_f32);
    println!("Q4_0 × Q8_0: {:.2} ms per GEMV ({:.1} calls/s)", elapsed_q8.as_millis() as f64 / iterations as f64, tps_q8);
    println!("Speedup: {:.2}x", tps_q8 / tps_f32);

    // Verify correctness
    let max_diff = y_f32.iter().zip(y_q8.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, |m, d| m.max(d));
    println!("Max difference: {:.6}", max_diff);
}
