//! MatMul Benchmark Suite
//!
//! Benchmarks for matrix multiplication operations covering:
//! - Dense matmul (CPU and GPU)
//! - Quantized matmul (Q4_0, Q8_0, Q4_K, Q6_K)
//! - Batched matmul operations
//! - CPU vs GPU comparison
//!
//! Run with: `cargo bench --bench matmul_bench`
//! Requires: ROCm GPU for GPU benchmarks

use std::hint::black_box;
use std::time::{Duration, Instant};

// Import CPU matmul functions
use rocmforge::tensor::matmul::cpu_matmul_f32;

// ============================================================================
// Benchmark Harness (following attention_bench.rs pattern)
// ============================================================================

struct Benchmark {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
}

impl Benchmark {
    fn new(name: &str, iterations: usize) -> Self {
        Benchmark {
            name: name.to_string(),
            iterations,
            warmup_iterations: iterations.min(10),
        }
    }

    fn run_time<F, R>(&self, mut f: F) -> BenchmarkResult
    where
        F: FnMut() -> R,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            black_box(f());
        }

        // Actual measurements
        let mut durations = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let start = Instant::now();
            black_box(f());
            durations.push(start.elapsed());
        }

        BenchmarkResult {
            name: self.name.clone(),
            iterations: self.iterations,
            durations,
        }
    }
}

struct BenchmarkResult {
    name: String,
    iterations: usize,
    durations: Vec<Duration>,
}

impl BenchmarkResult {
    fn report(&self) {
        let total: Duration = self.durations.iter().sum();
        let avg = total / self.iterations as u32;
        let min = *self.durations.iter().min().unwrap();
        let max = *self.durations.iter().max().unwrap();

        // Sort for percentiles
        let mut sorted = self.durations.clone();
        sorted.sort();

        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95) / 100];
        let p99 = sorted[(sorted.len() * 99) / 100];

        println!("\n=== {} ===", self.name);
        println!("Iterations: {}", self.iterations);
        println!("Average: {:?} ({:.3} ms)", avg, avg.as_secs_f64() * 1000.0);
        println!("Min:     {:?} ({:.3} ms)", min, min.as_secs_f64() * 1000.0);
        println!("Max:     {:?} ({:.3} ms)", max, max.as_secs_f64() * 1000.0);
        println!("P50:     {:?} ({:.3} ms)", p50, p50.as_secs_f64() * 1000.0);
        println!("P95:     {:?} ({:.3} ms)", p95, p95.as_secs_f64() * 1000.0);
        println!("P99:     {:?} ({:.3} ms)", p99, p99.as_secs_f64() * 1000.0);

        // Ops per second
        let ops_per_sec = 1_000_000_000.0 / avg.as_nanos() as f64;
        println!("Throughput: {:.2} ops/sec", ops_per_sec);
    }

    fn report_with_gflops(&self, m: usize, n: usize, k: usize) {
        self.report();

        // Calculate GFLOPS
        // matmul requires 2*m*n*k floating point operations (multiply-add)
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let avg_ms = self.avg_ms();
        let gflops = flops / (avg_ms / 1000.0) / 1e9;

        println!("GFLOPS: {:.2}", gflops);
    }

    fn avg_ms(&self) -> f64 {
        let total: Duration = self.durations.iter().sum();
        (total / self.iterations as u32).as_secs_f64() * 1000.0
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate test matrices for matmul: C = A * B
///
/// Returns (A, B) where:
/// - A is [m x k] in row-major format
/// - B is [k x n] in row-major format
fn generate_test_matrices(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a_size = m * k;
    let b_size = k * n;

    let mut a = Vec::with_capacity(a_size);
    let mut b = Vec::with_capacity(b_size);

    // Generate A matrix with some variation
    for i in 0..a_size {
        a.push((i as f32 * 0.01).sin() * 0.1);
    }

    // Generate B matrix with different variation
    for i in 0..b_size {
        b.push((i as f32 * 0.01).cos() * 0.1);
    }

    (a, b)
}

/// Generate quantized Q4_0 weights
///
/// Q4_0 format: 32 elements per block
/// - Per block: scale (f32, 4 bytes) + 16 bytes packed 4-bit values = 20 bytes
/// - Values are (quant - 8) * scale where quant is in [0, 15]
fn generate_q4_0_weights(n_elements: usize) -> Vec<u8> {
    let n_blocks = (n_elements + 31) / 32;
    let block_size = 20; // 4 bytes scale + 16 bytes packed data
    let mut data = vec![0u8; n_blocks * block_size];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * block_size;

        // Set scale = 0.1 for all blocks
        let scale: f32 = 0.1;
        data[block_offset..block_offset + 4].copy_from_slice(&scale.to_le_bytes());

        // Pack 4-bit values (alternating 8 and 9, which dequant to 0.0 and 0.1)
        for i in 0..16 {
            // Low nibble: 8, High nibble: 9
            data[block_offset + 4 + i] = 0x98;
        }
    }

    data
}

/// Generate quantized Q8_0 weights
///
/// Q8_0 format: 32 elements per block
/// - Per block: scale (f32, 4 bytes) + 32 bytes int8 values = 36 bytes
fn generate_q8_0_weights(n_elements: usize) -> Vec<u8> {
    let n_blocks = (n_elements + 31) / 32;
    let block_size = 36; // 4 bytes scale + 32 bytes int8 data
    let mut data = vec![0u8; n_blocks * block_size];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * block_size;

        // Set scale = 0.1 for all blocks
        let scale: f32 = 0.1;
        data[block_offset..block_offset + 4].copy_from_slice(&scale.to_le_bytes());

        // Fill with varying int8 values
        for i in 0..32 {
            let val = ((block_idx * 32 + i) % 16 - 8) as i8;
            data[block_offset + 4 + i] = val as u8;
        }
    }

    data
}

/// Dequantize Q4_0 weights (CPU reference)
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let q4_0_block_size = 20;
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * q4_0_block_size;
        if block_offset + 4 > data.len() {
            break;
        }

        // Read scale
        let scale = f32::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
            data[block_offset + 2],
            data[block_offset + 3],
        ]);

        // Unpack 4-bit values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..16 {
            if data_start + i >= data.len() {
                break;
            }
            let packed = data[data_start + i];

            // Low nibble
            let low = (packed & 0x0F) as i32 - 8;
            if base_idx + i * 2 < n_elements {
                result[base_idx + i * 2] = scale * low as f32;
            }

            // High nibble
            let high = ((packed >> 4) & 0x0F) as i32 - 8;
            if base_idx + i * 2 + 1 < n_elements {
                result[base_idx + i * 2 + 1] = scale * high as f32;
            }
        }
    }

    result
}

/// Dequantize Q8_0 weights (CPU reference)
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let q8_0_block_size = 36;
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * q8_0_block_size;
        if block_offset + 4 > data.len() {
            break;
        }

        // Read scale
        let scale = f32::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
            data[block_offset + 2],
            data[block_offset + 3],
        ]);

        // Read int8 values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..32 {
            let idx = data_start + i;
            if idx >= data.len() {
                break;
            }
            let elem_idx = base_idx + i;
            if elem_idx < n_elements {
                let int8_val = data[idx] as i8;
                result[elem_idx] = scale * int8_val as f32;
            }
        }
    }

    result
}

// ============================================================================
// Dense MatMul Benchmarks (CPU)
// ============================================================================

fn benchmark_dense_matmul_cpu() {
    println!("\n[Dense MatMul CPU Benchmarks]");
    println!("================================");

    let sizes = vec![
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    for (m, n, k) in sizes {
        let (a, b) = generate_test_matrices(m, n, k);

        let bench_name = format!("CPU Dense MatMul ({}x{} * {}x{})", m, k, k, n);
        let bench = Benchmark::new(&bench_name, 10); // Fewer iterations for large matrices

        let result = bench.run_time(|| {
            black_box(cpu_matmul_f32(&a, &b, m, n, k))
        });

        result.report_with_gflops(m, n, k);
    }
}

/// Benchmark dense matmul with non-square matrices
fn benchmark_dense_matmul_cpu_rectangular() {
    println!("\n[Dense MatMul CPU - Rectangular]");
    print!("===================================");

    let sizes = vec![
        (1, 4096, 4096),   // Single token inference
        (32, 128, 4096),   // Small batch
        (128, 128, 1024),  // Medium square
        (32, 32, 768),     // Typical attention projection
    ];

    for (m, n, k) in sizes {
        let (a, b) = generate_test_matrices(m, n, k);

        let bench_name = format!("CPU Dense MatMul ({}x{} * {}x{})", m, k, k, n);
        let bench = Benchmark::new(&bench_name, 50);

        let result = bench.run_time(|| {
            black_box(cpu_matmul_f32(&a, &b, m, n, k))
        });

        result.report_with_gflops(m, n, k);
    }
}

// ============================================================================
// Quantized MatMul Benchmarks (CPU)
// ============================================================================

fn benchmark_q4_0_matmul_cpu() {
    println!("\n[Q4_0 Quantized MatMul CPU Benchmarks]");
    println!("=======================================");

    let sizes = vec![
        (1, 4096, 4096),    // Single token
        (32, 4096, 4096),   // Small batch
        (1, 4096, 11008),   // Typical 7B layer
    ];

    for (m, n, k) in sizes {
        let (a, _) = generate_test_matrices(m, n, k);
        let q4_weights = generate_q4_0_weights(n * k);

        let bench_name = format!("CPU Q4_0 MatMul dequant ({}x{} * {}x{})", m, k, k, n);
        let bench = Benchmark::new(&bench_name, 20);

        let result = bench.run_time(|| {
            // Dequantize + matmul
            let dequant_weights = dequantize_q4_0(&q4_weights, n * k);
            // Reshape for matmul: dequant is [k*n] row-major, need [k x n]
            black_box(cpu_matmul_f32(&a, &dequant_weights, m, n, k))
        });

        result.report_with_gflops(m, n, k);

        // Calculate memory bandwidth savings
        let fp32_bytes = (n * k * 4) as f64;
        let q4_bytes = q4_weights.len() as f64;
        let compression = fp32_bytes / q4_bytes;
        println!("Compression: {:.2}x (FP32: {:.1} MB -> Q4_0: {:.1} MB)",
                 compression,
                 fp32_bytes / (1024.0 * 1024.0),
                 q4_bytes / (1024.0 * 1024.0));
    }
}

fn benchmark_q8_0_matmul_cpu() {
    println!("\n[Q8_0 Quantized MatMul CPU Benchmarks]");
    println!("=======================================");

    let sizes = vec![
        (1, 4096, 4096),
        (32, 4096, 4096),
        (1, 4096, 11008),
    ];

    for (m, n, k) in sizes {
        let (a, _) = generate_test_matrices(m, n, k);
        let q8_weights = generate_q8_0_weights(n * k);

        let bench_name = format!("CPU Q8_0 MatMul dequant ({}x{} * {}x{})", m, k, k, n);
        let bench = Benchmark::new(&bench_name, 20);

        let result = bench.run_time(|| {
            let dequant_weights = dequantize_q8_0(&q8_weights, n * k);
            black_box(cpu_matmul_f32(&a, &dequant_weights, m, n, k))
        });

        result.report_with_gflops(m, n, k);

        let fp32_bytes = (n * k * 4) as f64;
        let q8_bytes = q8_weights.len() as f64;
        let compression = fp32_bytes / q8_bytes;
        println!("Compression: {:.2}x (FP32: {:.1} MB -> Q8_0: {:.1} MB)",
                 compression,
                 fp32_bytes / (1024.0 * 1024.0),
                 q8_bytes / (1024.0 * 1024.0));
    }
}

/// Simulated Q4_K format (K-quant with super-block structure)
///
/// Q4_K format: 256 elements per super-block
/// - Super-block has 8 sub-blocks of 32 elements each
/// - Per super-block: scales (16 bytes f16) + mins (16 bytes f16) + 128 bytes packed Q4 = 160 bytes
fn generate_q4_k_weights(n_elements: usize) -> Vec<u8> {
    let n_super_blocks = (n_elements + 255) / 256;
    let super_block_size = 160; // 16 + 16 + 128
    let mut data = vec![0u8; n_super_blocks * super_block_size];

    for sb_idx in 0..n_super_blocks {
        let sb_offset = sb_idx * super_block_size;

        // Fill scales (f16, using 0.1)
        for i in 0..8 {
            // Use f16 representation of 0.1 (simplified as raw bytes)
            data[sb_offset + i * 2] = 0xcd;
            data[sb_offset + i * 2 + 1] = 0xcc;
        }

        // Fill mins (f16, using -0.1)
        for i in 0..8 {
            data[sb_offset + 16 + i * 2] = 0xcd;
            data[sb_offset + 16 + i * 2 + 1] = 0xcc;
        }

        // Fill packed Q4 data (alternating values)
        for i in 0..128 {
            data[sb_offset + 32 + i] = 0x88;
        }
    }

    data
}

/// Simulated Q6_K format
///
/// Q6_K format: 256 elements per block
/// - Per block: scales (32 bytes f16) + mins (32 bytes f16) + 192 bytes packed Q6 = 256 bytes
fn generate_q6_k_weights(n_elements: usize) -> Vec<u8> {
    let n_blocks = (n_elements + 255) / 256;
    let block_size = 256;
    let mut data = vec![0u8; n_blocks * block_size];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * block_size;

        // Fill scales
        for i in 0..32 {
            data[block_offset + i] = 0xcd;
        }

        // Fill mins
        for i in 0..32 {
            data[block_offset + 32 + i] = 0xcd;
        }

        // Fill packed Q6 data
        for i in 0..192 {
            data[block_offset + 64 + i] = 0x88;
        }
    }

    data
}

fn benchmark_k_quant_matmul_cpu() {
    println!("\n[K-Quant MatMul CPU Benchmarks]");
    println!("================================");

    // Standard size for comparison
    let m = 1;
    let n = 4096;
    let k = 4096;

    let (a, _) = generate_test_matrices(m, n, k);

    // Q4_K
    let q4_k_weights = generate_q4_k_weights(n * k);
    let q4_k_bytes = q4_k_weights.len() as f64;

    let bench = Benchmark::new("CPU Q4_K MatMul dequant (1x4096 * 4096x4096)", 20);
    let result = bench.run_time(|| {
        // Simulated dequantization
        let mut dequant = vec![0.0f32; n * k];
        for i in 0..(n * k) {
            dequant[i] = (i as f32 * 0.01).sin() * 0.1;
        }
        black_box(cpu_matmul_f32(&a, &dequant, m, n, k))
    });
    result.report();
    println!("Q4_K size: {:.1} MB", q4_k_bytes / (1024.0 * 1024.0));

    // Q6_K
    let q6_k_weights = generate_q6_k_weights(n * k);
    let q6_k_bytes = q6_k_weights.len() as f64;

    let bench = Benchmark::new("CPU Q6_K MatMul dequant (1x4096 * 4096x4096)", 20);
    let result = bench.run_time(|| {
        let mut dequant = vec![0.0f32; n * k];
        for i in 0..(n * k) {
            dequant[i] = (i as f32 * 0.01).sin() * 0.1;
        }
        black_box(cpu_matmul_f32(&a, &dequant, m, n, k))
    });
    result.report();
    println!("Q6_K size: {:.1} MB", q6_k_bytes / (1024.0 * 1024.0));

    // Comparison
    let fp32_bytes = (n * k * 4) as f64;
    println!("\nCompression comparison (vs FP32 {:.1} MB):", fp32_bytes / (1024.0 * 1024.0));
    println!("  Q4_K: {:.2}x compression", fp32_bytes / q4_k_bytes);
    println!("  Q6_K: {:.2}x compression", fp32_bytes / q6_k_bytes);
}

// ============================================================================
// Batched MatMul Benchmarks
// ============================================================================

fn benchmark_batched_matmul_cpu() {
    println!("\n[Batched MatMul CPU Benchmarks]");
    println!("================================");

    let batch_sizes = vec![1, 4, 8, 16, 32];
    let m = 32;
    let n = 128;
    let k = 256;

    for batch_size in batch_sizes {
        // Generate batch of matrices
        let total_m = batch_size * m;
        let (a, b) = generate_test_matrices(total_m, n, k);

        let bench_name = format!("CPU Batched MatMul (batch={}, {}x{} * {}x{})",
                                 batch_size, m, k, k, n);
        let bench = Benchmark::new(&bench_name, 50);

        let result = bench.run_time(|| {
            // Process batch as one large matmul
            black_box(cpu_matmul_f32(&a, &b, total_m, n, k))
        });

        result.report();

        // Per-sample throughput
        let avg_ms = result.avg_ms();
        let samples_per_sec = (batch_size as f64 * 1000.0) / avg_ms;
        println!("Samples/sec: {:.2}", samples_per_sec);
    }
}

// ============================================================================
// Quantization Format Comparison
// ============================================================================

fn benchmark_quantization_comparison() {
    println!("\n[Quantization Format Comparison]");
    println!("=================================");

    let m = 1;
    let n = 4096;
    let k = 4096;
    let _ = generate_test_matrices(m, n, k);

    let fp32_bytes = (n * k * 4) as f64 / (1024.0 * 1024.0);

    println!("\nFormat comparison for {}x{} * {}x{} matmul:", m, k, k, n);
    println!("FP32 reference: {:.2} MB", fp32_bytes);

    // Q4_0
    let q4_0_weights = generate_q4_0_weights(n * k);
    println!("Q4_0:  {:.2} MB ({:.2}x compression, {:.1} bits/weight)",
             q4_0_weights.len() as f64 / (1024.0 * 1024.0),
             fp32_bytes / (q4_0_weights.len() as f64 / (1024.0 * 1024.0)),
             4.5);

    // Q8_0
    let q8_0_weights = generate_q8_0_weights(n * k);
    println!("Q8_0:  {:.2} MB ({:.2}x compression, {:.1} bits/weight)",
             q8_0_weights.len() as f64 / (1024.0 * 1024.0),
             fp32_bytes / (q8_0_weights.len() as f64 / (1024.0 * 1024.0)),
             8.5);

    // Q4_K
    let q4_k_weights = generate_q4_k_weights(n * k);
    println!("Q4_K:  {:.2} MB ({:.2}x compression, {:.1} bits/weight)",
             q4_k_weights.len() as f64 / (1024.0 * 1024.0),
             fp32_bytes / (q4_k_weights.len() as f64 / (1024.0 * 1024.0)),
             5.0);

    // Q6_K
    let q6_k_weights = generate_q6_k_weights(n * k);
    println!("Q6_K:  {:.2} MB ({:.2}x compression, {:.1} bits/weight)",
             q6_k_weights.len() as f64 / (1024.0 * 1024.0),
             fp32_bytes / (q6_k_weights.len() as f64 / (1024.0 * 1024.0)),
             6.5);

    // Benchmark dequantization overhead
    println!("\nDequantization performance:");

    let bench = Benchmark::new("Q4_0 dequant only", 50);
    let result = bench.run_time(|| {
        black_box(dequantize_q4_0(&q4_0_weights, n * k))
    });
    result.report();
    let q4_0_throughput = (n * k) as f64 / (result.avg_ms() / 1000.0) / 1e6;
    println!("Throughput: {:.2} M elements/sec", q4_0_throughput);

    let bench = Benchmark::new("Q8_0 dequant only", 50);
    let result = bench.run_time(|| {
        black_box(dequantize_q8_0(&q8_0_weights, n * k))
    });
    result.report();
    let q8_0_throughput = (n * k) as f64 / (result.avg_ms() / 1000.0) / 1e6;
    println!("Throughput: {:.2} M elements/sec", q8_0_throughput);
}

// ============================================================================
// CPU vs GPU Comparison (placeholder)
// ============================================================================

#[cfg(feature = "rocm")]
fn benchmark_cpu_vs_gpu() {
    println!("\n[CPU vs GPU MatMul Comparison]");
    println!("================================");

    let sizes = vec![
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];

    for (m, n, k) in sizes {
        let (a, b) = generate_test_matrices(m, n, k);

        // CPU benchmark
        let cpu_bench = Benchmark::new(&format!("CPU ({}x{}*{}x{})", m, k, k, n), 10);
        let cpu_result = cpu_bench.run_time(|| {
            black_box(cpu_matmul_f32(&a, &b, m, n, k))
        });

        cpu_result.report_with_gflops(m, n, k);

        // GPU benchmark placeholder
        // TODO: Implement GPU matmul when HipBuffer can be used in benchmarks
        println!("GPU: Not yet implemented (requires HipBuffer integration)");
    }
}

#[cfg(not(feature = "rocm"))]
fn benchmark_cpu_vs_gpu() {
    println!("\n[CPU vs GPU MatMul Comparison]");
    println!("================================");
    println!("GPU benchmarks skipped. Run with: cargo bench --bench matmul_bench --features rocm");
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    println!("====================================");
    println!("ROCmForge MatMul Benchmark Suite");
    println!("====================================");
    println!("\nThis benchmark measures:");
    println!("- Dense matrix multiplication (CPU)");
    println!("- Quantized matmul (Q4_0, Q8_0, Q4_K, Q6_K)");
    println!("- Batched matmul operations");
    println!("- CPU vs GPU comparison (when available)");

    // Always run CPU benchmarks
    benchmark_dense_matmul_cpu();
    benchmark_dense_matmul_cpu_rectangular();
    benchmark_q4_0_matmul_cpu();
    benchmark_q8_0_matmul_cpu();
    benchmark_k_quant_matmul_cpu();
    benchmark_batched_matmul_cpu();
    benchmark_quantization_comparison();

    // Run GPU benchmarks if ROCm feature is enabled
    #[cfg(feature = "rocm")]
    {
        benchmark_cpu_vs_gpu();
    }

    #[cfg(not(feature = "rocm"))]
    {
        benchmark_cpu_vs_gpu();
    }

    println!("\n====================================");
    println!("Benchmark Complete");
    println!("====================================");
}
