//! Dequantization Benchmark Suite
//!
//! Benchmarks for comparing dequantization performance across all GGUF quantization formats.
//! Tests all 15 formats: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, MXFP6E2m3, MXFP6E3m2
//!
//! Run with: `cargo bench --bench dequant_bench`

use std::hint::black_box;
use std::time::{Duration, Instant};

// Import quantization and dequantization types
use rocmforge::loader::{
    dequant_q2_k, dequant_q3_k, dequant_q4_0, dequant_q4_1, dequant_q4_k,
    dequant_q5_0, dequant_q5_1, dequant_q5_k, dequant_q6_k, dequant_q8_0,
    dequant_mxfp4, dequant_mxfp6, dequantize,
    GgufTensor, GgufTensorType, TensorShape,
};

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
    }

    fn avg_ms(&self) -> f64 {
        let total: Duration = self.durations.iter().sum();
        (total / self.iterations as u32).as_secs_f64() * 1000.0
    }

    /// Report with throughput metrics
    fn report_throughput(&self, element_count: usize, compressed_bytes: usize) {
        self.report();

        let avg_ms = self.avg_ms();
        let avg_sec = avg_ms / 1000.0;

        // Calculate throughput
        let elements_per_sec = element_count as f64 / avg_sec;
        let uncompressed_bytes = element_count * 4; // f32 output
        let bandwidth_out_gb = (uncompressed_bytes as f64 / avg_sec) / 1e9;
        let bandwidth_in_gb = (compressed_bytes as f64 / avg_sec) / 1e9;

        // Compression ratio
        let compression_ratio = uncompressed_bytes as f64 / compressed_bytes as f64;

        println!("Throughput:     {:.2} million elements/sec", elements_per_sec / 1e6);
        println!("Output Bandwidth: {:.2} GB/sec", bandwidth_out_gb);
        println!("Input Bandwidth:  {:.2} GB/sec", bandwidth_in_gb);
        println!("Compression Ratio: {:.2}x", compression_ratio);
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

const TENSOR_SIZE: usize = 4096; // Elements per dimension
const TOTAL_ELEMENTS: usize = TENSOR_SIZE * TENSOR_SIZE; // ~16.8M elements

fn create_test_tensor(tensor_type: GgufTensorType, data: Vec<u8>) -> GgufTensor {
    GgufTensor {
        name: "bench_tensor".to_string(),
        shape: TensorShape::from_dims(&[TENSOR_SIZE, TENSOR_SIZE]),
        tensor_type,
        quant_type: tensor_type.to_string().to_string(),
        offset: 0,
        data,
    }
}

// Format 01: F32 (baseline - no actual dequantization)
fn create_f32_tensor() -> GgufTensor {
    let total_bytes = TOTAL_ELEMENTS * 4;
    let mut data = vec![0u8; total_bytes];

    // Fill with some pattern
    for i in 0..TOTAL_ELEMENTS {
        let val: f32 = ((i as f32) * 0.001).sin();
        let bytes = val.to_le_bytes();
        data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }

    create_test_tensor(GgufTensorType::F32, data)
}

// Format 02: F16 (baseline - minimal conversion)
fn create_f16_tensor() -> GgufTensor {
    let total_bytes = TOTAL_ELEMENTS * 2;
    let mut data = vec![0u8; total_bytes];

    // Fill with f16 values
    for i in 0..TOTAL_ELEMENTS {
        let val: f32 = ((i as f32) * 0.001).sin();
        let f16 = half::f16::from_f32(val);
        let bits = f16.to_bits();
        let bytes = bits.to_le_bytes();
        data[i * 2..i * 2 + 2].copy_from_slice(&bytes);
    }

    create_test_tensor(GgufTensorType::F16, data)
}

// Format 03: Q4_0 - 32 values per block, scale (4 bytes) + 16 bytes quants
fn create_q4_0_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * (4 + 16);
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 16);

        // Set scale = 0.01
        let scale: f32 = 0.01;
        let scale_bytes = scale.to_le_bytes();
        data[block_start..block_start + 4].copy_from_slice(&scale_bytes);

        // Quants already zero-initialized
    }

    create_test_tensor(GgufTensorType::Q4_0, data)
}

// Format 04: Q4_1 - 32 values per block, scale + min + 16 bytes quants
fn create_q4_1_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * (4 + 4 + 16);
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 16);

        // Set scale = 0.01
        let scale: f32 = 0.01;
        let scale_bytes = scale.to_le_bytes();
        data[block_start..block_start + 4].copy_from_slice(&scale_bytes);

        // Set min = 0.0
        let min: f32 = 0.0;
        let min_bytes = min.to_le_bytes();
        data[block_start + 4..block_start + 8].copy_from_slice(&min_bytes);
    }

    create_test_tensor(GgufTensorType::Q4_1, data)
}

// Format 05: Q5_0 - 32 values per block, scale + qh + 20 bytes quants
fn create_q5_0_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * (4 + 4 + 20);
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 20);

        // Set scale = 0.01
        let scale: f32 = 0.01;
        let scale_bytes = scale.to_le_bytes();
        data[block_start..block_start + 4].copy_from_slice(&scale_bytes);

        // qh = 0
        data[block_start + 4..block_start + 8].copy_from_slice(&[0u8; 4]);
    }

    create_test_tensor(GgufTensorType::Q5_0, data)
}

// Format 06: Q5_1 - 32 values per block, scale + min + qh + 20 bytes quants
fn create_q5_1_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * (4 + 4 + 4 + 20);
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 4 + 20);

        // Set scale = 0.01
        let scale: f32 = 0.01;
        let scale_bytes = scale.to_le_bytes();
        data[block_start..block_start + 4].copy_from_slice(&scale_bytes);

        // Set min = 0.0
        let min: f32 = 0.0;
        let min_bytes = min.to_le_bytes();
        data[block_start + 4..block_start + 8].copy_from_slice(&min_bytes);

        // qh = 0
        data[block_start + 8..block_start + 12].copy_from_slice(&[0u8; 4]);
    }

    create_test_tensor(GgufTensorType::Q5_1, data)
}

// Format 07: Q8_0 - 32 values per block, scale (4 bytes) + 32 bytes quants
fn create_q8_0_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * (4 + 32);
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 32);

        // Set scale = 0.01
        let scale: f32 = 0.01;
        let scale_bytes = scale.to_le_bytes();
        data[block_start..block_start + 4].copy_from_slice(&scale_bytes);

        // Quants already zero-initialized
    }

    create_test_tensor(GgufTensorType::Q8_0, data)
}

// Format 08: Q2_K - 256 elements per super-block (256 bytes total)
fn create_q2_k_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(256);
    let total_bytes = blocks * 256;
    let mut data = vec![0u8; total_bytes];

    // Data is already zero-initialized which represents all zeros
    // Set a scale for non-zero behavior
    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        // Set first scale to 0.01 in half precision
        data[block_start] = 0x01;
        data[block_start + 1] = 0x3C; // ~0.01 in f16

        // Set first min to 0
        data[block_start + 32] = 0;
        data[block_start + 33] = 0;
    }

    create_test_tensor(GgufTensorType::Q2_K, data)
}

// Format 09: Q3_K - 256 elements per super-block (256 bytes total)
fn create_q3_k_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(256);
    let total_bytes = blocks * 256;
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        // Set first scale to 0.01 in half precision
        data[block_start] = 0x01;
        data[block_start + 1] = 0x3C;
    }

    create_test_tensor(GgufTensorType::Q3_K, data)
}

// Format 10: Q4_K - 256 elements per super-block (256 bytes total)
fn create_q4_k_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(256);
    let total_bytes = blocks * 256;
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        // Set first scale to 0.01 in half precision
        data[block_start] = 0x01;
        data[block_start + 1] = 0x3C;

        // Set first min to 0
        data[block_start + 16] = 0;
    }

    create_test_tensor(GgufTensorType::Q4_K, data)
}

// Format 11: Q5_K - 256 elements per super-block (256 bytes total)
fn create_q5_k_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(256);
    let total_bytes = blocks * 256;
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        // Set first scale to 0.01 in half precision
        data[block_start] = 0x01;
        data[block_start + 1] = 0x3C;

        // Set first min to 0
        data[block_start + 32] = 0;
    }

    create_test_tensor(GgufTensorType::Q5_K, data)
}

// Format 12: Q6_K - 256 elements per super-block (256 bytes total)
fn create_q6_k_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(256);
    let total_bytes = blocks * 256;
    let mut data = vec![0u8; total_bytes];

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        // Set first scale to 0.01 in half precision
        data[block_start] = 0x01;
        data[block_start + 1] = 0x3C;
    }

    create_test_tensor(GgufTensorType::Q6_K, data)
}

// Format 13: MXFP4 - 32 values per block, 1 scale byte + 16 data bytes
fn create_mxfp4_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * 17;
    let mut data = vec![0u8; total_bytes];

    // Set scale exponent to 0 (scale = 1.0)
    for block_idx in 0..blocks {
        let block_start = block_idx * 17;
        data[block_start] = 0; // E8M0 scale exponent = 0
    }

    create_test_tensor(GgufTensorType::Mxfp4, data)
}

// Format 14: MXFP6E2M3 - 32 values per block, 1 scale byte + 24 data bytes
fn create_mxfp6_e2m3_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * 25;
    let mut data = vec![0u8; total_bytes];

    // Set scale exponent to 0 (scale = 1.0)
    for block_idx in 0..blocks {
        let block_start = block_idx * 25;
        data[block_start] = 0; // E8M0 scale exponent = 0
    }

    create_test_tensor(GgufTensorType::Mxfp6E2m3, data)
}

// Format 15: MXFP6E3M2 - 32 values per block, 1 scale byte + 24 data bytes
fn create_mxfp6_e3m2_tensor() -> GgufTensor {
    let blocks = TOTAL_ELEMENTS.div_ceil(32);
    let total_bytes = blocks * 25;
    let mut data = vec![0u8; total_bytes];

    // Set scale exponent to 0 (scale = 1.0)
    for block_idx in 0..blocks {
        let block_start = block_idx * 25;
        data[block_start] = 0; // E8M0 scale exponent = 0
    }

    create_test_tensor(GgufTensorType::Mxfp6E3m2, data)
}

// ============================================================================
// CPU Dequantization Benchmarks
// ============================================================================

/// Benchmark F32 (baseline - no actual dequantization, just copy/conversion)
fn benchmark_f32() {
    println!("\n[F32 Dequantization (Baseline)]");
    println!("=================================");

    let tensor = create_f32_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("F32 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequantize(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark F16 conversion
fn benchmark_f16() {
    println!("\n[F16 Dequantization]");
    println!("=====================");

    let tensor = create_f16_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("F16 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequantize(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q4_0 dequantization
fn benchmark_q4_0() {
    println!("\n[Q4_0 Dequantization]");
    println!("======================");

    let tensor = create_q4_0_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q4_0 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q4_0(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q4_1 dequantization
fn benchmark_q4_1() {
    println!("\n[Q4_1 Dequantization]");
    println!("======================");

    let tensor = create_q4_1_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q4_1 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q4_1(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q5_0 dequantization
fn benchmark_q5_0() {
    println!("\n[Q5_0 Dequantization]");
    println!("======================");

    let tensor = create_q5_0_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q5_0 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q5_0(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q5_1 dequantization
fn benchmark_q5_1() {
    println!("\n[Q5_1 Dequantization]");
    println!("======================");

    let tensor = create_q5_1_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q5_1 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q5_1(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q8_0 dequantization
fn benchmark_q8_0() {
    println!("\n[Q8_0 Dequantization]");
    println!("======================");

    let tensor = create_q8_0_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q8_0 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q8_0(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q2_K dequantization
fn benchmark_q2_k() {
    println!("\n[Q2_K Dequantization]");
    println!("======================");

    let tensor = create_q2_k_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q2_K -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q2_k(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q3_K dequantization
fn benchmark_q3_k() {
    println!("\n[Q3_K Dequantization]");
    println!("======================");

    let tensor = create_q3_k_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q3_K -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q3_k(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q4_K dequantization
fn benchmark_q4_k() {
    println!("\n[Q4_K Dequantization]");
    println!("======================");

    let tensor = create_q4_k_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q4_K -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q4_k(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q5_K dequantization
fn benchmark_q5_k() {
    println!("\n[Q5_K Dequantization]");
    println!("======================");

    let tensor = create_q5_k_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q5_K -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q5_k(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark Q6_K dequantization
fn benchmark_q6_k() {
    println!("\n[Q6_K Dequantization]");
    println!("======================");

    let tensor = create_q6_k_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("Q6_K -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_q6_k(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark MXFP4 dequantization
fn benchmark_mxfp4() {
    println!("\n[MXFP4 Dequantization]");
    println!("=======================");

    let tensor = create_mxfp4_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("MXFP4 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_mxfp4(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark MXFP6E2M3 dequantization
fn benchmark_mxfp6_e2m3() {
    println!("\n[MXFP6E2M3 Dequantization]");
    println!("===========================");

    let tensor = create_mxfp6_e2m3_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("MXFP6E2M3 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_mxfp6(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Benchmark MXFP6E3M2 dequantization
fn benchmark_mxfp6_e3m2() {
    println!("\n[MXFP6E3M2 Dequantization]");
    println!("===========================");

    let tensor = create_mxfp6_e3m2_tensor();
    let input_bytes = tensor.data.len();

    let bench = Benchmark::new("MXFP6E3M2 -> FP32", 50);
    let result = bench.run_time(|| {
        black_box(dequant_mxfp6(&tensor).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

/// Generic format benchmark
fn benchmark_generic(format_name: &str, tensor_type: GgufTensorType, input_bytes: usize) {
    let tensor = create_test_tensor(tensor_type, vec![0u8; input_bytes]);

    // Set some non-zero data for realistic benchmarking
    let mut data = tensor.data;
    if !data.is_empty() {
        data[0] = 1;
        if data.len() > 1 {
            data[1] = 0x3C; // f16 scale
        }
    }

    let tensor_with_data = GgufTensor { data, ..tensor };

    let bench = Benchmark::new(format_name, 50);
    let result = bench.run_time(|| {
        black_box(dequantize(&tensor_with_data).unwrap())
    });

    result.report_throughput(TOTAL_ELEMENTS, input_bytes);
}

// ============================================================================
// Format Comparison Summary
// ============================================================================

struct FormatMetrics {
    name: String,
    avg_ms: f64,
    elements_per_sec: f64,
    bandwidth_gb: f64,
    compression_ratio: f64,
    input_bytes: usize,
}

fn print_comparison_table(metrics: &[FormatMetrics]) {
    println!("\n");
    println!("================================================================================");
    println!("                       DEQUANTIZATION FORMAT COMPARISON");
    println!("================================================================================");
    println!("| Format         | Avg (ms) | Throughput (M/s) | Bandwidth (GB/s) | Ratio |");
    println!("|----------------|----------|-------------------|------------------|-------|");

    for m in metrics {
        println!(
            "| {:14} | {:8.3} | {:17.2} | {:16.2} | {:5.2}x |",
            m.name,
            m.avg_ms,
            m.elements_per_sec / 1e6,
            m.bandwidth_gb,
            m.compression_ratio
        );
    }

    println!("================================================================================");
    println!("Tensor size: {}x{} = {} elements (~67 MB uncompressed)",
             TENSOR_SIZE, TENSOR_SIZE, TOTAL_ELEMENTS);

    // Identify fastest and slowest
    let fastest = metrics.iter().min_by(|a, b| a.avg_ms.partial_cmp(&b.avg_ms).unwrap()).unwrap();
    let slowest = metrics.iter().max_by(|a, b| a.avg_ms.partial_cmp(&b.avg_ms).unwrap()).unwrap();

    println!("\nFastest format: {} ({:.3} ms, {:.2} M elements/sec)",
             fastest.name, fastest.avg_ms, fastest.elements_per_sec / 1e6);
    println!("Slowest format: {} ({:.3} ms, {:.2} M elements/sec)",
             slowest.name, slowest.avg_ms, slowest.elements_per_sec / 1e6);

    // Best compression
    let best_compression = metrics.iter().max_by(|a, b| a.compression_ratio.partial_cmp(&b.compression_ratio).unwrap()).unwrap();
    println!("Best compression: {} ({:.2}x)", best_compression.name, best_compression.compression_ratio);
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    println!("========================================");
    println!("ROCmForge Dequantization Benchmark Suite");
    println!("========================================");
    println!("\nTensor size: {}x{} = {} elements", TENSOR_SIZE, TENSOR_SIZE, TOTAL_ELEMENTS);
    println!("\nThis benchmark measures:");
    println!("- Dequantization time for all 15 GGUF formats");
    println!("- Throughput (elements/sec)");
    println!("- Memory bandwidth (GB/s)");
    println!("- Compression ratio vs FP32");

    // Collect metrics for comparison table
    let mut metrics = Vec::new();

    // Baseline formats
    println!("\n========================================");
    println!("BASELINE FORMATS (No quantization)");
    println!("========================================");

    benchmark_f32();
    metrics.push(FormatMetrics {
        name: "F32".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 1.0,
        input_bytes: TOTAL_ELEMENTS * 4,
    });

    benchmark_f16();
    metrics.push(FormatMetrics {
        name: "F16".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 2.0,
        input_bytes: TOTAL_ELEMENTS * 2,
    });

    // Q-format (4-bit variants)
    println!("\n========================================");
    println!("4-Bit QUANTIZED FORMATS");
    println!("========================================");

    benchmark_q4_0();
    benchmark_q4_1();

    metrics.push(FormatMetrics {
        name: "Q4_0".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 8.0,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * (4 + 16),
    });
    metrics.push(FormatMetrics {
        name: "Q4_1".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 7.1,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * (4 + 4 + 16),
    });

    // Q-format (5-bit variants)
    println!("\n========================================");
    println!("5-Bit QUANTIZED FORMATS");
    println!("========================================");

    benchmark_q5_0();
    benchmark_q5_1();

    metrics.push(FormatMetrics {
        name: "Q5_0".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 6.4,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * (4 + 4 + 20),
    });
    metrics.push(FormatMetrics {
        name: "Q5_1".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 5.8,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * (4 + 4 + 4 + 20),
    });

    // Q-format (8-bit)
    println!("\n========================================");
    println!("8-Bit QUANTIZED FORMAT");
    println!("========================================");

    benchmark_q8_0();

    metrics.push(FormatMetrics {
        name: "Q8_0".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 4.0,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * (4 + 32),
    });

    // K-quant formats
    println!("\n========================================");
    println!("K-QUANT FORMATS (Super-block)");
    println!("========================================");

    benchmark_q2_k();
    benchmark_q3_k();
    benchmark_q4_k();
    benchmark_q5_k();
    benchmark_q6_k();

    metrics.push(FormatMetrics {
        name: "Q2_K".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 8.0,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(256)) * 256,
    });
    metrics.push(FormatMetrics {
        name: "Q3_K".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 6.4,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(256)) * 256,
    });
    metrics.push(FormatMetrics {
        name: "Q4_K".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 5.3,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(256)) * 256,
    });
    metrics.push(FormatMetrics {
        name: "Q5_K".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 4.4,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(256)) * 256,
    });
    metrics.push(FormatMetrics {
        name: "Q6_K".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 3.8,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(256)) * 256,
    });

    // MXFP formats
    println!("\n========================================");
    println!("MXFP FORMATS (OCP MX Specification)");
    println!("========================================");

    benchmark_mxfp4();
    benchmark_mxfp6_e2m3();
    benchmark_mxfp6_e3m2();

    metrics.push(FormatMetrics {
        name: "MXFP4".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 8.0,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * 17,
    });
    metrics.push(FormatMetrics {
        name: "MXFP6_E2M3".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 5.3,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * 25,
    });
    metrics.push(FormatMetrics {
        name: "MXFP6_E3M2".to_string(),
        avg_ms: 0.0, elements_per_sec: 0.0, bandwidth_gb: 0.0,
        compression_ratio: 5.3,
        input_bytes: (TOTAL_ELEMENTS.div_ceil(32)) * 25,
    });

    // Note: Actual timing metrics would be collected during benchmark execution
    // This is a placeholder for the comparison table
    println!("\n========================================");
    println!("Benchmark Complete");
    println!("========================================");
    println!("\nNote: Comparison table shows theoretical compression ratios.");
    println!("Actual throughput metrics are reported per-format above.");
}
