use std::time::Instant;

/// Simple direct benchmark of Q4_0 × Q8_0 GEMV
/// This bypasses criterion complexity to get a quick measurement
fn main() {
    let hidden_size = 2048; // Qwen2.5-0.5B model
    let out_dim = 2048;
    let num_blocks = hidden_size / 32; // Q4_0 blocks
    let num_iterations = 1000;

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

    // Warmup
    rocmforge::cpu::ops::gemv_q4_0_q8_0(&w, &x, &mut vec![0.0f32; out_dim], hidden_size);

    for _ in 0..num_iterations {
        rocmforge::cpu::ops::gemv_q4_0_q8_0(&w, &x, &mut vec![0.0f32; out_dim], hidden_size);
    }

    let elapsed = start.elapsed();

    let ms_per_call = elapsed.as_millis_f64() as f64;
    let calls_per_sec = (num_iterations as f64) / elapsed.as_secs_f64();

    println!("Q4_0 × Q8_0 GEMV:");
    println!("  Time per call: {:.2} ms", ms_per_call);
    println!("  Calls per second: {:.1}", calls_per_sec);
    println!("  Hidden: {}, Out: {}, Blocks: {}", hidden_size, out_dim, num_blocks);
}
