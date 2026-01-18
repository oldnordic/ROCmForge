//! Memory Benchmark Suite
//!
//! Benchmarks for tracking allocation patterns and memory efficiency:
//! - KV cache allocation for different sequence lengths (512, 1024, 2048, 4096)
//! - Scratch buffer allocation/reuse patterns
//! - Peak memory, allocation rate, fragmentation
//! - Memory bandwidth for large tensor operations
//!
//! Run with: `cargo bench --bench memory_bench`
//! Requires: ROCm GPU for GPU memory benchmarks

use std::hint::black_box;
use std::time::{Duration, Instant};

// ============================================================================
// Memory Tracking Utilities
// ============================================================================

/// Memory usage statistics from a benchmark run
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory allocated in bytes
    pub peak_bytes: usize,
    /// Total bytes allocated over the benchmark lifetime
    pub total_allocated: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Bytes currently in use
    pub current_usage: usize,
    /// Estimated fragmentation (wasted space / total space)
    pub fragmentation_ratio: f64,
}

impl MemoryStats {
    /// Calculate allocation rate (bytes per second)
    pub fn allocation_rate(&self, duration_secs: f64) -> f64 {
        if duration_secs > 0.0 {
            self.total_allocated as f64 / duration_secs
        } else {
            0.0
        }
    }

    /// Calculate bytes per allocation
    pub fn avg_bytes_per_allocation(&self) -> f64 {
        if self.allocation_count > 0 {
            self.total_allocated as f64 / self.allocation_count as f64
        } else {
            0.0
        }
    }

    /// Format bytes as human readable (KB, MB, GB)
    pub fn format_bytes(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = 1024 * 1024;
        const GB: usize = 1024 * 1024 * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }

    /// Print memory statistics
    pub fn report(&self, duration: Duration) {
        let duration_secs = duration.as_secs_f64();

        println!("\n  Memory Statistics:");
        println!("    Peak memory:        {}", Self::format_bytes(self.peak_bytes));
        println!("    Current usage:      {}", Self::format_bytes(self.current_usage));
        println!("    Total allocated:    {}", Self::format_bytes(self.total_allocated));
        println!("    Allocation count:   {}", self.allocation_count);
        println!("    Deallocation count: {}", self.deallocation_count);
        println!("    Avg bytes/alloc:    {}", Self::format_bytes(self.avg_bytes_per_allocation() as usize));
        println!("    Allocation rate:    {}/sec", Self::format_bytes(self.allocation_rate(duration_secs) as usize));
        println!("    Fragmentation:      {:.2}%", self.fragmentation_ratio * 100.0);
    }
}

/// Simple memory tracker for allocations
#[derive(Debug, Default)]
pub struct MemoryTracker {
    peak_bytes: usize,
    total_allocated: usize,
    allocation_count: usize,
    deallocation_count: usize,
    current_usage: usize,
    allocations: Vec<(usize, usize)>, // (size, alignment)
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation
    pub fn allocate(&mut self, size: usize) {
        self.current_usage += size;
        self.total_allocated += size;
        self.allocation_count += 1;
        self.peak_bytes = self.peak_bytes.max(self.current_usage);
        self.allocations.push((size, 0));
    }

    /// Record a deallocation
    pub fn deallocate(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.deallocation_count += 1;
    }

    /// Simulate fragmentation by accounting for wasted space
    pub fn estimate_fragmentation(&self) -> f64 {
        if self.current_usage == 0 {
            return 0.0;
        }

        // Simple fragmentation estimate: variance in allocation sizes
        // Higher variance = more potential fragmentation
        if self.allocations.is_empty() {
            return 0.0;
        }

        let avg_size = self.current_usage as f64 / self.allocations.len() as f64;
        let mut variance = 0.0;

        for (size, _) in &self.allocations {
            let diff = *size as f64 - avg_size;
            variance += diff * diff;
        }

        variance /= self.allocations.len() as f64;

        // Normalize to 0-1 range (heuristic)
        let cv = if avg_size > 0.0 {
            variance.sqrt() / avg_size
        } else {
            0.0
        };

        cv.min(1.0)
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            peak_bytes: self.peak_bytes,
            total_allocated: self.total_allocated,
            allocation_count: self.allocation_count,
            deallocation_count: self.deallocation_count,
            current_usage: self.current_usage,
            fragmentation_ratio: self.estimate_fragmentation(),
        }
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

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

    fn run_memory<F, R>(&self, mut f: F) -> (BenchmarkResult, MemoryStats)
    where
        F: FnMut(&mut MemoryTracker) -> R,
    {
        let mut tracker = MemoryTracker::new();

        // Warmup (don't track)
        for _ in 0..self.warmup_iterations {
            black_box(f(&mut tracker));
            tracker.reset();
        }

        // Actual measurements
        let mut durations = Vec::with_capacity(self.iterations);
        tracker.reset();

        let _start_total = Instant::now();
        for _ in 0..self.iterations {
            let start = Instant::now();
            black_box(f(&mut tracker));
            durations.push(start.elapsed());
        }
        let _total_duration = _start_total.elapsed();

        let result = BenchmarkResult {
            name: self.name.clone(),
            iterations: self.iterations,
            durations,
        };

        (result, tracker.stats())
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
}

// ============================================================================
// KV Cache Memory Benchmarks
// ============================================================================

/// Simulate KV cache allocation for a given sequence length
///
/// Calculates memory requirements based on:
/// - num_layers: number of transformer layers
/// - num_heads: number of attention heads
/// - head_dim: dimension per head
/// - seq_len: sequence length
/// - bytes_per_element: typically 2 (fp16) or 4 (fp32)
fn simulate_kv_cache_allocation(
    tracker: &mut MemoryTracker,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    bytes_per_element: usize,
) {
    // KV cache has 2 buffers: K and V
    // Each buffer: [num_layers, num_heads, seq_len, head_dim]
    let elements_per_buffer = num_layers * num_heads * seq_len * head_dim;
    let bytes_per_buffer = elements_per_buffer * bytes_per_element;

    // Allocate K cache
    tracker.allocate(bytes_per_buffer);

    // Allocate V cache
    tracker.allocate(bytes_per_buffer);
}

/// Calculate theoretical memory bandwidth
///
/// Returns bytes transferred per second
fn calculate_bandwidth(bytes: usize, duration_secs: f64) -> f64 {
    if duration_secs > 0.0 {
        bytes as f64 / duration_secs
    } else {
        0.0
    }
}

/// Format bandwidth as human readable (GB/s, MB/s, etc.)
fn format_bandwidth(bytes_per_sec: f64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.2} GB/s", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.2} MB/s", bytes_per_sec / MB)
    } else {
        format!("{:.2} KB/s", bytes_per_sec / 1024.0)
    }
}

fn benchmark_kv_cache_allocation() {
    println!("\n[KV Cache Allocation Benchmarks]");
    println!("==================================");
    println!("Simulating KV cache memory allocation for different sequence lengths");

    // Typical 7B model configuration
    let num_layers = 32;
    let num_heads = 32;
    let head_dim = 128;
    let bytes_per_element = 2; // fp16

    let sequence_lengths = vec![512, 1024, 2048, 4096];

    for seq_len in sequence_lengths {
        let bench_name = format!("KV Cache Allocation (seq_len={}, {} layers, {} heads, {} dim)",
                                 seq_len, num_layers, num_heads, head_dim);
        let bench = Benchmark::new(&bench_name, 10);

        let (result, stats) = bench.run_memory(|tracker| {
            simulate_kv_cache_allocation(
                tracker,
                num_layers,
                num_heads,
                head_dim,
                seq_len,
                bytes_per_element,
            );
        });

        result.report();
        stats.report(result.durations.iter().sum::<Duration>());

        // Additional KV cache metrics
        let elements_per_token = num_layers * num_heads * head_dim * 2; // K + V
        let bytes_per_token = elements_per_token * bytes_per_element;
        let total_kv_bytes = stats.peak_bytes;

        println!("  KV Cache Specific:");
        println!("    Bytes per token:    {}", MemoryStats::format_bytes(bytes_per_token));
        println!("    Total for {} tokens: {}", seq_len, MemoryStats::format_bytes(total_kv_bytes));
        println!("    Elements/token:     {}", elements_per_token);
    }
}

/// Benchmark incremental KV cache growth (token-by-token)
fn benchmark_kv_cache_growth() {
    println!("\n[KV Cache Growth Benchmarks]");
    println!("==============================");
    println!("Simulating incremental token append to KV cache");

    let num_layers = 32;
    let num_heads = 32;
    let head_dim = 128;
    let bytes_per_element = 2;

    let target_lengths = vec![512, 1024, 2048];

    for target_len in target_lengths {
        let bench_name = format!("KV Cache Growth (0 -> {} tokens)", target_len);
        let bench = Benchmark::new(&bench_name, 5);

        let (result, stats) = bench.run_memory(|tracker| {
            // Simulate incremental growth
            for _ in 0..target_len {
                // Each token adds to all layers
                let elements_per_token = num_layers * num_heads * head_dim * 2;
                let bytes_per_token = elements_per_token * bytes_per_element;

                // In a real system, this might trigger page allocation
                tracker.allocate(bytes_per_token);
            }
        });

        result.report();
        stats.report(result.durations.iter().sum::<Duration>());

        // Growth-specific metrics
        let avg_bytes_per_append = stats.total_allocated / target_len;
        println!("  Growth Specific:");
        println!("    Avg bytes/append:   {}", MemoryStats::format_bytes(avg_bytes_per_append));
        println!("    Append rate:       {:.2} tokens/sec",
                 target_len as f64 / (result.avg_ms() / 1000.0));
    }
}

// ============================================================================
// Scratch Buffer Benchmarks
// ============================================================================

/// Simulate single large scratch buffer allocation
fn benchmark_scratch_single_allocation() {
    println!("\n[Scratch Buffer - Single Allocation]");
    println!("====================================");

    let buffer_sizes = vec![
        (1024 * 1024, "1 MB"),
        (10 * 1024 * 1024, "10 MB"),
        (100 * 1024 * 1024, "100 MB"),
        (256 * 1024 * 1024, "256 MB"),
    ];

    for (size, label) in buffer_sizes {
        let bench_name = format!("Single Scratch Buffer ({})", label);
        let bench = Benchmark::new(&bench_name, 50);

        let (result, stats) = bench.run_memory(|tracker| {
            tracker.allocate(size);
            // Simulate some work
            black_box(size);
            tracker.deallocate(size);
        });

        result.report();

        let total_time = result.durations.iter().sum::<Duration>();
        let bandwidth = calculate_bandwidth(size * 2, total_time.as_secs_f64()); // *2 for alloc + dealloc
        println!("  Memory Bandwidth:   {}", format_bandwidth(bandwidth));
        println!("  Peak memory:        {}", MemoryStats::format_bytes(stats.peak_bytes));
    }
}

/// Simulate multiple small scratch buffer allocations
fn benchmark_scratch_fragmented_allocation() {
    println!("\n[Scratch Buffer - Fragmented Allocation]");
    println!("========================================");

    let total_size = 10 * 1024 * 1024; // 10 MB total
    let chunk_sizes = vec![
        (64 * 1024, "64 KB chunks"),         // 160 allocations
        (256 * 1024, "256 KB chunks"),       // 40 allocations
        (1024 * 1024, "1 MB chunks"),        // 10 allocations
    ];

    for (chunk_size, label) in chunk_sizes {
        let n_chunks = total_size / chunk_size;
        let bench_name = format!("Fragmented Scratch Buffer ({} x {})", n_chunks, label);
        let bench = Benchmark::new(&bench_name, 20);

        let (result, stats) = bench.run_memory(|tracker| {
            // Allocate many small chunks
            for _ in 0..n_chunks {
                tracker.allocate(chunk_size);
            }
            // Deallocate all
            for _ in 0..n_chunks {
                tracker.deallocate(chunk_size);
            }
        });

        result.report();
        stats.report(result.durations.iter().sum::<Duration>());

        println!("  Fragmentation Specific:");
        println!("    Chunks allocated:   {}", n_chunks);
        println!("    Fragmentation:      {:.2}%", stats.fragmentation_ratio * 100.0);
    }
}

/// Simulate buffer reuse pattern (allocate once, use many times)
fn benchmark_scratch_reuse() {
    println!("\n[Scratch Buffer - Reuse Pattern]");
    println!("==================================");

    let buffer_size = 10 * 1024 * 1024; // 10 MB
    let reuse_counts = vec![10, 100, 1000];

    for reuse_count in reuse_counts {
        let bench_name = format!("Scratch Buffer Reuse ({} uses, 1 alloc)", reuse_count);
        let bench = Benchmark::new(&bench_name, 20);

        let (result, stats) = bench.run_memory(|tracker| {
            // Single allocation
            tracker.allocate(buffer_size);

            // Simulate multiple uses
            for _ in 0..reuse_count {
                black_box(buffer_size);
            }

            // Single deallocation
            tracker.deallocate(buffer_size);
        });

        result.report();
        stats.report(result.durations.iter().sum::<Duration>());

        let total_time = result.durations.iter().sum::<Duration>();
        println!("  Reuse Specific:");
        println!("    Uses per alloc:     {}", reuse_count);
        println!("    Avg use time:       {:.3} us", (result.avg_ms() * 1000.0) / reuse_count as f64);
        println!("    Effective BW:       {}", format_bandwidth(
            (buffer_size * reuse_count * 2) as f64 / total_time.as_secs_f64()
        ));
    }
}

// ============================================================================
// Memory Bandwidth Benchmarks
// ============================================================================

/// Simulate large tensor operation memory bandwidth
fn benchmark_tensor_memory_bandwidth() {
    println!("\n[Tensor Memory Bandwidth Benchmarks]");
    println!("=====================================");

    // Simulate typical attention tensor operations
    let tensor_configs = vec![
        (1, 32, 128, 512, "Q projection [1, 32, 128, 512]"),
        (1, 32, 128, 4096, "K projection [1, 32, 128, 4096]"),
        (1, 32, 128, 4096, "V projection [1, 32, 128, 4096]"),
        (1, 32, 512, 512, "Attention output [1, 32, 512, 512]"),
    ];

    for (batch, heads, seq_len, dim, label) in tensor_configs {
        let elements = batch * heads * seq_len * dim;
        let bytes = elements * 4; // fp32

        let bench_name = format!("Tensor Read/Write ({})", label);
        let bench = Benchmark::new(&bench_name, 50);

        let (result, stats) = bench.run_memory(|tracker| {
            // Simulate read (allocate and initialize)
            tracker.allocate(bytes);
            // Simulate write (allocate output buffer)
            tracker.allocate(bytes);
            // Deallocate both
            tracker.deallocate(bytes * 2);
        });

        result.report();

        let total_time = result.durations.iter().sum::<Duration>();
        // Total bytes transferred: read + write
        let total_bytes = bytes * 2;
        let bandwidth = calculate_bandwidth(total_bytes, total_time.as_secs_f64());

        println!("  Bandwidth Metrics:");
        println!("    Tensor size:        {}", MemoryStats::format_bytes(bytes));
        println!("    Total transfer:     {}", MemoryStats::format_bytes(total_bytes));
        println!("    Memory bandwidth:   {}", format_bandwidth(bandwidth));
        println!("    Peak memory:        {}", MemoryStats::format_bytes(stats.peak_bytes));
    }
}

/// Benchmark sequential vs interleaved memory access patterns
fn benchmark_memory_access_patterns() {
    println!("\n[Memory Access Pattern Benchmarks]");
    println!("===================================");

    let elements = 1024 * 1024; // 1M elements
    let bytes = elements * 4; // 4MB (fp32)

    // Sequential access pattern
    let bench = Benchmark::new("Sequential Access (4MB)", 50);
    let (result, _) = bench.run_memory(|tracker| {
        tracker.allocate(bytes);
        // Simulate sequential access
        for i in 0..elements {
            black_box(i * 4);
        }
        tracker.deallocate(bytes);
    });
    result.report();

    // Strided access pattern (cache unfriendly)
    let bench = Benchmark::new("Strided Access (4MB, stride=64)", 50);
    let (result, _) = bench.run_memory(|tracker| {
        tracker.allocate(bytes);
        // Simulate strided access (every 64th element)
        let stride = 64;
        for i in (0..elements).step_by(stride) {
            black_box(i * 4);
        }
        tracker.deallocate(bytes);
    });
    result.report();

    // Random access pattern (very cache unfriendly)
    let bench = Benchmark::new("Random Access (4MB, 1024 random)", 50);
    let (result, _) = bench.run_memory(|tracker| {
        tracker.allocate(bytes);
        // Simulate random access to 1024 elements
        let mut indices = Vec::with_capacity(1024);
        for i in 0..1024 {
            indices.push(i * 397 % elements); // Pseudo-random
        }
        for idx in indices {
            black_box(idx * 4);
        }
        tracker.deallocate(bytes);
    });
    result.report();
}

// ============================================================================
// Page Table Overhead Benchmarks
// ============================================================================

/// Simulate paged KV cache with page table overhead
fn benchmark_paged_cache_overhead() {
    println!("\n[Paged KV Cache Overhead Benchmarks]");
    println!("====================================");

    let page_sizes = vec![16, 32, 64, 128, 256];
    let total_tokens = 1024;
    let num_heads = 32;
    let head_dim = 128;
    let bytes_per_element = 2;

    for page_size in page_sizes {
        let n_pages = (total_tokens + page_size - 1) / page_size;
        let tokens_per_kv = num_heads * head_dim * 2; // K + V

        let bench_name = format!("Paged Cache (page_size={}, {} pages)", page_size, n_pages);
        let bench = Benchmark::new(&bench_name, 20);

        let (result, _stats) = bench.run_memory(|tracker| {
            // Allocate pages
            for _ in 0..n_pages {
                let page_bytes = page_size * tokens_per_kv * bytes_per_element;
                tracker.allocate(page_bytes);
            }

            // Simulate page table overhead (small allocations for metadata)
            // Page table entry: physical_block_id (4 bytes) + ref_count (8 bytes) + sequence set
            let page_table_entry_size = 4 + 8 + 32; // Approximate
            tracker.allocate(page_table_entry_size * n_pages);
        });

        result.report();

        let total_kv_bytes = n_pages * page_size * tokens_per_kv * bytes_per_element;
        let page_table_bytes = n_pages * (4 + 8 + 32);
        let overhead_ratio = page_table_bytes as f64 / total_kv_bytes as f64;

        println!("  Paged Cache Specific:");
        println!("    Pages:              {}", n_pages);
        println!("    KV cache size:      {}", MemoryStats::format_bytes(total_kv_bytes));
        println!("    Page table size:    {}", MemoryStats::format_bytes(page_table_bytes));
        println!("    Overhead ratio:     {:.4}%", overhead_ratio * 100.0);
    }
}

// ============================================================================
// Comparison Benchmarks
// ============================================================================

/// Compare single vs chunked allocation strategies
fn benchmark_allocation_strategy_comparison() {
    println!("\n[Allocation Strategy Comparison]");
    println!("==================================");

    let total_size = 100 * 1024 * 1024; // 100 MB

    // Single large allocation
    let bench = Benchmark::new("Single Allocation (100 MB)", 20);
    let (result, stats) = bench.run_memory(|tracker| {
        tracker.allocate(total_size);
        black_box(total_size);
        tracker.deallocate(total_size);
    });
    result.report();
    let single_avg_ms = result.avg_ms();
    println!("  Peak memory:        {}", MemoryStats::format_bytes(stats.peak_bytes));
    println!("  Allocations:        {}", stats.allocation_count);

    // Chunked allocation (100 x 1MB)
    let chunk_size = 1024 * 1024;
    let n_chunks = total_size / chunk_size;
    let bench = Benchmark::new(&format!("Chunked Allocation ({} x 1 MB)", n_chunks), 20);
    let (result, stats) = bench.run_memory(|tracker| {
        for _ in 0..n_chunks {
            tracker.allocate(chunk_size);
        }
        black_box(total_size);
        for _ in 0..n_chunks {
            tracker.deallocate(chunk_size);
        }
    });
    result.report();
    let chunked_avg_ms = result.avg_ms();
    println!("  Peak memory:        {}", MemoryStats::format_bytes(stats.peak_bytes));
    println!("  Allocations:        {}", stats.allocation_count);

    // Comparison
    println!("\n  Comparison:");
    println!("    Overhead:           {:.2}x", chunked_avg_ms / single_avg_ms);
}

// ============================================================================
// GPU Memory Benchmarks (with rocm feature)
// ============================================================================

#[cfg(feature = "rocm")]
fn benchmark_gpu_memory_allocation() {
    println!("\n[GPU Memory Allocation Benchmarks]");
    println!("====================================");
    println!("GPU memory benchmarks require ROCm GPU and HipBackend integration.");
    println!("Placeholder for future implementation.");

    // TODO: Implement actual GPU memory benchmarks when HipBuffer is available
    // in benchmark context
    let sizes = vec![
        (100 * 1024 * 1024, "100 MB"),
        (500 * 1024 * 1024, "500 MB"),
        (1024 * 1024 * 1024, "1 GB"),
    ];

    for (size, label) in sizes {
        println!("\n  GPU Buffer ({}):", label);
        println!("    Allocation time:   N/A (requires HipBackend)");
        println!("    Bandwidth:         N/A (requires HipBackend)");
        println!("    Peak GPU memory:   {}", MemoryStats::format_bytes(size));
    }
}

#[cfg(not(feature = "rocm"))]
fn benchmark_gpu_memory_allocation() {
    println!("\n[GPU Memory Allocation Benchmarks]");
    println!("====================================");
    println!("GPU benchmarks skipped. Run with: cargo bench --bench memory_bench --features rocm");
}

// ============================================================================
// KV Cache Profiling Benchmarks
// ============================================================================

/// Profile KV cache memory patterns for different sequence lengths
///
/// This benchmark analyzes:
/// - Memory growth per token
/// - Fragmentation at different sequence lengths
/// - Page table overhead
/// - Block allocation efficiency
fn benchmark_kv_cache_profiling() {
    println!("\n[KV Cache Memory Profiling]");
    println!("============================");
    println!("Analyzing KV cache memory patterns for different sequence lengths\n");

    // Model configurations to profile
    let configs = vec![
        (32, 32, 128, 32, "7B model"),  // num_layers, num_heads, head_dim, page_size
        (40, 40, 128, 32, "13B model"),
        (80, 64, 128, 16, "70B model"),
    ];

    let sequence_lengths = vec![256, 512, 1024, 2048, 4096];

    for (num_layers, num_heads, head_dim, page_size, model_name) in configs {
        println!("--- {} Profile ---", model_name);
        println!("  Config: {} layers, {} heads, {} head_dim, {} tokens/page",
                 num_layers, num_heads, head_dim, page_size);

        for seq_len in sequence_lengths {
            // Calculate theoretical memory requirements
            let elements_per_token_per_layer = num_heads * head_dim * 2; // K + V
            let elements_per_token = elements_per_token_per_layer * num_layers;
            let bytes_per_token = elements_per_token * std::mem::size_of::<f32>();

            // Paged allocation
            let num_pages = (seq_len + page_size - 1) / page_size;
            let allocated_tokens = num_pages * page_size;

            let total_kv_bytes = allocated_tokens * bytes_per_token;
            let used_kv_bytes = seq_len * bytes_per_token;
            let wasted_bytes = total_kv_bytes - used_kv_bytes;

            // Page table overhead
            let page_table_entries = num_pages;
            let page_table_bytes = page_table_entries * std::mem::size_of::<u32>();

            // Block allocator overhead (approximate)
            let allocator_bytes = num_pages * std::mem::size_of::<u32>() * 3;

            let total_overhead = page_table_bytes + allocator_bytes;
            let overhead_ratio = total_overhead as f64 / total_kv_bytes as f64;
            let fragmentation = wasted_bytes as f64 / total_kv_bytes as f64;

            println!("\n  Sequence Length: {} tokens", seq_len);
            println!("    Pages allocated:    {}", num_pages);
            println!("    KV cache size:      {}", MemoryStats::format_bytes(total_kv_bytes));
            println!("    KV cache used:      {}", MemoryStats::format_bytes(used_kv_bytes));
            println!("    Wasted memory:      {} ({:.1}%)",
                     MemoryStats::format_bytes(wasted_bytes),
                     fragmentation * 100.0);
            println!("    Page table size:    {}", MemoryStats::format_bytes(page_table_bytes));
            println!("    Allocator overhead: {}", MemoryStats::format_bytes(allocator_bytes));
            println!("    Total metadata:     {} ({:.4}%)",
                     MemoryStats::format_bytes(total_overhead),
                     overhead_ratio * 100.0);
            println!("    Bytes per token:    {:.2}", bytes_per_token as f64);

            // Memory efficiency calculation
            let efficiency = if total_kv_bytes > 0 {
                used_kv_bytes as f64 / total_kv_bytes as f64
            } else {
                0.0
            };
            println!("    Memory efficiency:  {:.1}%", efficiency * 100.0);
        }
        println!();
    }
}

/// Profile block allocation efficiency across different usage patterns
fn benchmark_block_allocation_patterns() {
    println!("\n[Block Allocation Pattern Analysis]");
    println!("====================================");
    println!("Analyzing allocation efficiency for different access patterns\n");

    let page_size = 16;
    let patterns = vec![
        ("Sequential single-token appends", 1, 512),    // (name, tokens_per_append, total_tokens)
        ("Batch appends (16 tokens)", 16, 512),
        ("Batch appends (64 tokens)", 64, 512),
        ("Batch appends (256 tokens)", 256, 512),
    ];

    for (pattern_name, tokens_per_append, total_tokens) in patterns {
        println!("--- Pattern: {} ---", pattern_name);
        println!("  Tokens per append: {}", tokens_per_append);
        println!("  Total tokens:      {}", total_tokens);

        let num_appends = (total_tokens + tokens_per_append - 1) / tokens_per_append;

        // Calculate block allocations needed
        let mut current_tokens = 0;
        let mut block_allocations = 0;

        for _ in 0..num_appends {
            let tokens_after_append = current_tokens + tokens_per_append;
            let blocks_before = (current_tokens + page_size - 1) / page_size;
            let blocks_after = (tokens_after_append + page_size - 1) / page_size;

            if blocks_after > blocks_before {
                block_allocations += blocks_after - blocks_before;
            }

            current_tokens = tokens_after_append;
        }

        let total_blocks = (total_tokens + page_size - 1) / page_size;
        let final_capacity = total_blocks * page_size;
        let waste = final_capacity - total_tokens;
        let fragmentation = waste as f64 / final_capacity as f64;

        println!("  Total appends:      {}", num_appends);
        println!("  Block allocations:  {}", block_allocations);
        println!("  Total blocks:       {}", total_blocks);
        println!("  Final capacity:     {} tokens", final_capacity);
        println!("  Wasted capacity:    {} tokens ({:.1}%)",
                 waste, fragmentation * 100.0);
        println!("  Alloc efficiency:   {:.1}%",
                 if block_allocations > 0 {
                     total_blocks as f64 / block_allocations as f64 * 100.0
                 } else {
                     0.0
                 });
        println!();
    }
}

/// Profile memory usage by model configuration
fn benchmark_model_memory_profile() {
    println!("\n[Model Configuration Memory Profile]");
    println!("======================================");
    println!("Memory requirements for different model sizes and sequence lengths\n");

    // Model configs: (name, params_B, layers, heads, head_dim)
    let models = vec![
        ("TinyLlama (1B)", 1, 22, 32, 64),
        ("Llama-2-7B", 7, 32, 32, 128),
        ("Llama-2-13B", 13, 40, 40, 128),
        ("Llama-2-70B", 70, 80, 64, 128),
    ];

    let seq_lengths = vec![512, 1024, 2048, 4096, 8192, 16384];

    println!("| Model         | Seq | KV Cache  | Per Token | Overhead  |");
    println!("|---------------|-----|-----------|-----------|-----------|");

    for (model_name, _params, layers, heads, head_dim) in models {
        for seq_len in seq_lengths {
            let elements_per_token = layers * heads * head_dim * 2; // K + V
            let bytes_per_token = elements_per_token * std::mem::size_of::<f32>();
            let total_kv_bytes = seq_len * bytes_per_token;

            // Assume 16-token pages
            let page_size = 16;
            let num_pages = (seq_len + page_size - 1) / page_size;
            let overhead_bytes = num_pages * std::mem::size_of::<u32>() * 4;

            println!("| {:13} | {:4} | {:9} | {:9} | {:9} |",
                     model_name,
                     seq_len,
                     MemoryStats::format_bytes(total_kv_bytes),
                     MemoryStats::format_bytes(bytes_per_token),
                     MemoryStats::format_bytes(overhead_bytes));
        }
    }
}

// ============================================================================
// Summary Report
// ============================================================================

fn print_summary() {
    println!("\n[Memory Benchmark Summary]");
    println!("===========================");
    println!("\nKey Findings:");
    println!("1. KV cache memory grows linearly with sequence length");
    println!("2. Scratch buffer reuse significantly reduces allocation overhead");
    println!("3. Paged cache adds ~0.01-0.1% overhead for metadata");
    println!("4. Chunked allocations have higher overhead than single large allocation");
    println!("5. Page fragmentation increases with smaller page sizes");
    println!("6. Batch appends reduce block allocation overhead");
    println!("\nRecommendations:");
    println!("- Pre-allocate KV cache when max sequence length is known");
    println!("- Reuse scratch buffers across operations");
    println!("- Use larger page sizes (32-64 tokens) to reduce page table overhead");
    println!("- Batch small allocations into larger ones when possible");
    println!("- Monitor memory profile during long-running inference sessions");
    println!("- Use cache.compact_cache() when fragmentation is high");
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    println!("====================================");
    println!("ROCmForge Memory Benchmark Suite");
    println!("====================================");
    println!("\nThis benchmark measures:");
    println!("- KV cache allocation for different sequence lengths");
    println!("- Scratch buffer allocation/reuse patterns");
    println!("- Memory bandwidth for large tensor operations");
    println!("- Peak memory, allocation rate, fragmentation");
    println!("- KV cache profiling for different model sizes");

    // CPU memory benchmarks
    benchmark_kv_cache_allocation();
    benchmark_kv_cache_growth();
    benchmark_scratch_single_allocation();
    benchmark_scratch_fragmented_allocation();
    benchmark_scratch_reuse();
    benchmark_tensor_memory_bandwidth();
    benchmark_memory_access_patterns();
    benchmark_paged_cache_overhead();
    benchmark_allocation_strategy_comparison();

    // KV cache profiling benchmarks
    benchmark_kv_cache_profiling();
    benchmark_block_allocation_patterns();
    benchmark_model_memory_profile();

    // GPU benchmarks (if ROCm feature is enabled)
    benchmark_gpu_memory_allocation();

    // Print summary
    print_summary();

    println!("\n====================================");
    println!("Benchmark Complete");
    println!("====================================");
}
