//! Attention Benchmark Suite
//!
//! Benchmarks for comparing different attention backends:
//! - Standard attention (CPU backend)
//! - Flash attention (GPU backend)
//!
//! Run with: `cargo bench --bench attention_bench`
//! Requires: ROCm GPU for GPU benchmarks

use std::hint::black_box;
use std::time::{Duration, Instant};

// Import backend registry types (always available for CPU)
use rocmforge::attention::backend_registry::{AttentionBackendRegistry, AttentionConfig};

// ============================================================================
// Benchmark Harness
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

    fn avg_ms(&self) -> f64 {
        let total: Duration = self.durations.iter().sum();
        (total / self.iterations as u32).as_secs_f64() * 1000.0
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_test_data(dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Generate test data for [batch_size=1, seq_len=dim, dim] shape
    // CPU backend expects: q.len() = batch_size * seq_len * dim = 1 * dim * dim
    let total_size = dim * dim;

    // Generate test data with some variation (not all zeros)
    let mut q = Vec::with_capacity(total_size);
    let mut k = Vec::with_capacity(total_size);
    let mut v = Vec::with_capacity(total_size);

    for i in 0..total_size {
        q.push((i as f32 * 0.01).sin() * 0.1);
        k.push((i as f32 * 0.01).cos() * 0.1);
        v.push((i as f32 * 0.01).tan() * 0.1);
    }

    (q, k, v)
}

// ============================================================================
// CPU Benchmarks (Always Available)
// ============================================================================

fn benchmark_cpu_attention() {
    println!("\n[CPU Attention Benchmarks]");
    println!("==========================");

    let registry = AttentionBackendRegistry::new();
    let cpu_backend = registry.get_backend("cpu").unwrap();

    // Test configurations: different dimensions
    // Shape is [batch_size=1, seq_len=dim, dim]
    let dims = vec![32, 64, 128, 256, 512];

    for dim in dims {
        let (q, k, v) = generate_test_data(dim);

        let bench_name = format!("CPU Attention (seq_len={}, dim={})", dim, dim);
        let bench = Benchmark::new(&bench_name, 100);

        let result = bench.run_time(|| {
            match cpu_backend.forward(
                &AttentionConfig::new(dim, 1, dim),
                &q, &k, &v, None
            ) {
                Ok(output) => black_box(output),
                Err(e) => {
                    eprintln!("CPU forward failed: {}", e);
                    Vec::new()
                }
            }
        });

        result.report();

        // Calculate tokens per second
        let avg_ms = result.avg_ms();
        let tokens_per_sec = (dim as f64 * 1000.0) / avg_ms;
        println!("Tokens/sec: {:.2}", tokens_per_sec);
    }
}

// ============================================================================
// GPU Benchmarks (ROCm Feature Required)
// ============================================================================

fn generate_test_data_flash(seq_len: usize, num_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // FlashAttention expects: [batch_size, seq_len, num_heads * head_dim]
    // But BackendImplementation provides: [batch_size, seq_len, dim] where dim = num_heads * head_dim
    let dim = num_heads * head_dim;
    let total_size = seq_len * dim;

    let mut q = Vec::with_capacity(total_size);
    let mut k = Vec::with_capacity(total_size);
    let mut v = Vec::with_capacity(total_size);

    for i in 0..total_size {
        q.push((i as f32 * 0.01).sin() * 0.1);
        k.push((i as f32 * 0.01).cos() * 0.1);
        v.push((i as f32 * 0.01).tan() * 0.1);
    }

    (q, k, v)
}

fn benchmark_flash_attention() {
    println!("\n[Flash Attention Benchmarks]");
    println!("============================");

    let registry = AttentionBackendRegistry::new();

    // Check if flash_attention backend is available
    let flash_backend = match registry.get_backend("flash_attention") {
        Ok(backend) => backend,
        Err(_) => {
            println!("FlashAttention backend not available (check rocm feature)");
            return;
        }
    };

    // Test configurations: (seq_len, num_heads, head_dim)
    let configs = vec![
        (128, 4, 32),
        (256, 8, 32),
        (512, 8, 64),
        (1024, 16, 64),
    ];

    for (seq_len, num_heads, head_dim) in configs {
        let dim = num_heads * head_dim;

        let config = AttentionConfig::new(dim, num_heads, head_dim)
            .with_max_sequence_length(seq_len);

        if !flash_backend.supports(&config) {
            println!("Config (seq={}, heads={}, dim={}) not supported by FlashAttention", seq_len, num_heads, head_dim);
            continue;
        }

        let (q, k, v) = generate_test_data_flash(seq_len, num_heads, head_dim);

        let bench_name = format!("Flash Attention (seq={}, heads={}, dim={})", seq_len, num_heads, head_dim);
        let bench = Benchmark::new(&bench_name, 50); // Fewer iterations for GPU

        let result = bench.run_time(|| {
            match flash_backend.forward(&config, &q, &k, &v, None) {
                Ok(output) => black_box(output),
                Err(e) => {
                    eprintln!("Flash attention failed: {}", e);
                    Vec::new()
                }
            }
        });

        result.report();

        // Calculate tokens per second
        let avg_ms = result.avg_ms();
        let tokens_per_sec = (seq_len as f64 * 1000.0) / avg_ms;
        println!("Tokens/sec: {:.2}", tokens_per_sec);
    }
}

fn benchmark_cpu_vs_flash() {
    println!("\n[CPU vs Flash Attention Comparison]");
    println!("====================================");

    let registry = AttentionBackendRegistry::new();
    let cpu_backend = registry.get_backend("cpu").unwrap();
    let flash_backend = registry.get_backend("flash_attention");

    if flash_backend.is_err() {
        println!("FlashAttention backend not available, skipping comparison");
        return;
    }
    let flash_backend = flash_backend.unwrap();

    // Test configuration that works for both
    // Use smaller size since CPU backend uses seq_len = dim
    let dim = 64;
    let (q, k, v) = generate_test_data(dim);

    let cpu_config = AttentionConfig::new(dim, 1, dim);

    // Benchmark CPU
    let cpu_bench = Benchmark::new("CPU (comparison)", 100);
    let cpu_result = cpu_bench.run_time(|| {
        match cpu_backend.forward(&cpu_config, &q, &k, &v, None) {
            Ok(output) => black_box(output),
            Err(e) => {
                eprintln!("CPU forward failed: {}", e);
                Vec::new()
            }
        }
    });

    cpu_result.report();

    // Note: FlashAttention comparison skipped due to layout mismatch
    // CPU backend uses [batch, seq_len, dim]
    // FlashAttention backend uses [batch, seq_len, num_heads * head_dim] but expects different internal layout
    println!("\n>>> Flash Attention comparison skipped (layout mismatch between backends)");
    println!(">>> See 06-03-SUMMARY.md for known layout issues");
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    println!("====================================");
    println!("ROCmForge Attention Benchmark Suite");
    println!("====================================");
    println!("\nThis benchmark measures:");
    println!("- CPU attention performance");
    println!("- Flash Attention GPU performance (if available)");
    println!("- Comparison between backends");

    // Always run CPU benchmarks
    benchmark_cpu_attention();

    // Run GPU benchmarks if ROCm feature is enabled
    {
        benchmark_flash_attention();
        benchmark_cpu_vs_flash();
    }

    {
        println!("\n[GPU Benchmarks Skipped]");
        println!("ROCm feature not enabled. Run with: cargo bench --bench attention_bench --features rocm");
    }

    println!("\n====================================");
    println!("Benchmark Complete");
    println!("====================================");
}
