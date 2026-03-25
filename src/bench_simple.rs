use std::time::Instant;

/// Simple benchmark of Q4_0 × Q8_0 GEMV
/// This bypasses criterion complexity to get a quick measurement
fn main() {
    let hidden_size = 2048; // Qwen2.5-0.5B model
    let out_dim = 2048;
    let num_blocks = hidden_size / 32; // Q4_0 blocks

    // Generate fake Q4_0 weights
    let mut w = vec![0u8; out_dim * num_blocks * 18];
    for row in 0..out_dim {
        let row_offset = row * num_blocks * 18;
        for b in 0..num_blocks {
            let block_offset = row_offset + b * 18;
            w[block_offset] = 0x00; // scale = 2.0
            w[block_offset + 1] = 0x40; // scale = 2.0
            for j in 0..16 {
                let lo = j as u8;      // 0..15
                let hi = 15 - j;    // 15..0
                w[block_offset + j] = lo | (hi << 4);
            }
        }
    }

    // Generate fake input
    let x: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.01).collect();

    let start = Instant::now();

    // Warmup (run once to cache)
    rocmforge::cpu::ops::gemv_q4_0_q8_0(&w, &x, &mut vec![0.0f32; out_dim], hidden_size);

    for _ in 0..100 {
        rocmforge::cpu::ops::gemv_q4_0_q8_0(&w, &x, &mut vec![0.0f32; out_dim], hidden_size);
    }

    let elapsed = start.elapsed();
    let warmup_ms = elapsed.as_millis_f64();

    // Benchmark (run 10 times)
    let total_ms = 0.0;
    for i in 0..10 {
        let iter_start = Instant::now();
        rocmforge::cpu::ops::gemv_q4_0_q8_0(&w, &x, &mut vec![0.0f32; out_dim], hidden_size);
        let iter_time = iter_start.elapsed().as_millis_f64();
        total_ms += iter_time;
    }

    let avg_ms = total_ms / 10.0;
    let calls_per_sec = 1000.0 / (avg_ms / 1000.0);

    println!("Q4_0 × Q8_0 GEMV Benchmark:");
    println!("  Warmup: {:.0} ms", warmup_ms);
    println!("  Average:  {:.2} ms", avg_ms);
    println!("  Calls/sec: {:.1}", calls_per_sec);
}
