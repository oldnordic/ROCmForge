# Plan: CPU Hardware Detection for Cache-Aware Prefill Batching

## Overview

Implement CPU capability detection to automatically derive optimal prefill batch sizes based on actual hardware (L3 cache size, physical cores, memory). This eliminates hardcoded assumptions and ensures optimal performance across different CPUs.

## Motivation

Current prefill allocates buffers for the entire prompt sequence. For large prompts (128+ tokens), this can cause:
- Cache thrashing when working set exceeds L3 capacity
- RAM spills that degrade performance
- Suboptimal core utilization (using all cores including hyperthreads)

Hardware-aware batching keeps each batch within L3 cache, avoiding RAM traffic.

## Startup Sequence

```
1. Hardware detection (NEW: src/hardware/)
   ├─ CPU topology (physical cores, hyperthreading)
   ├─ Cache sizes (L3, L2, L1)
   └─ Available memory

2. Derive BatchConfig from capabilities
   ├─ memory_per_token() → compute from ModelConfig
   ├─ max_tokens_per_batch = L3_cache / memory_per_token
   └─ optimal_cores = physical_cores (ignore hyperthreading)

3. Load model (existing: src/loader/)
   └─ CpuModelWeights::load()

4. Run inference with batched prefill
   └─ cpu_prefill_forward() splits into batches
```

## Files to Create (src/hardware/)

### src/hardware/mod.rs (< 1K LOC)
```rust
// Public API
pub use caps::CpuCapabilities;
pub use config::{BatchConfig, memory_per_token};

// Re-export for convenience
pub fn detect() -> CpuCapabilities { ... }
pub fn derive_batch_config(caps: &CpuCapabilities, config: &ModelConfig) -> BatchConfig { ... }
```

### src/hardware/caps.rs (< 1K LOC)
```rust
/// Detected CPU hardware capabilities
pub struct CpuCapabilities {
    /// Physical CPU cores (NOT including hyperthreads)
    pub physical_cores: usize,
    /// Logical CPUs (physical_cores * threads_per_core)
    pub logical_cpus: usize,
    /// L3 cache size in bytes (0 if undetectable)
    pub l3_cache_bytes: usize,
    /// L2 cache size in bytes (0 if undetectable)
    pub l2_cache_bytes: usize,
    /// Total system memory in bytes
    pub total_memory_bytes: usize,
}

impl CpuCapabilities {
    /// Detect capabilities using sysinfo crate
    pub fn detect() -> Result<Self, HardwareError> { ... }

    /// Get safe number of cores for compute (physical only, not hyperthreads)
    pub fn compute_cores(&self) -> usize { ... }
}
```

### src/hardware/config.rs (< 1K LOC)
```rust
/// Derived prefill batching configuration
pub struct BatchConfig {
    /// Maximum tokens to process in one batch (fits in L3)
    pub max_tokens_per_batch: usize,
    /// Number of cores to use for parallelism (physical cores only)
    pub num_cores: usize,
}

impl BatchConfig {
    /// Derive from CPU capabilities and model configuration
    pub fn from_capabilities(
        caps: &CpuCapabilities,
        config: &ModelConfig,
    ) -> Self {
        let mem_per_tok = memory_per_token(config);
        let max_batch = if caps.l3_cache_bytes > 0 {
            // Use 80% of L3 to leave room for OS/other processes
            (caps.l3_cache_bytes * 8 / 10) / mem_per_tok
        } else {
            // Fallback: no L3 info, use 4MB per batch
            (4 * 1024 * 1024) / mem_per_tok
        };
        let max_batch = max_batch.max(1).min(256); // Clamp: at least 1, at most 256

        Self {
            max_tokens_per_batch: max_batch,
            num_cores: caps.physical_cores,
        }
    }
}

/// Compute memory footprint per token during prefill
/// (activations + KV cache writes per layer)
pub fn memory_per_token(config: &ModelConfig) -> usize {
    // Per layer: hidden + normed + q + k + v + attn_out + layer_out + gate + swiglu
    let per_layer = config.hidden_size * 4  // hidden, normed, layer_out (f32 each)
        + (config.num_heads * config.head_dim) // q
        + (config.num_kv_heads * config.head_dim) * 2 // k, v
        + config.intermediate_size * 2; // gate, swiglu

    // All layers share buffers, so max of per-layer
    let activations = per_layer * std::mem::size_of::<f32>();

    // Add KV cache writes (per layer, k + v for this position)
    let kv_per_layer = 2 * (config.num_kv_heads * config.head_dim)
        * std::mem::size_of::<f32>();
    let kv_write = kv_per_layer * config.num_layers;

    activations + kv_write
}
```

### src/hardware/error.rs (< 1K LOC)
```rust
#[derive(Debug)]
pub enum HardwareError {
    DetectionFailed(String),
    UnsupportedPlatform(String),
}

impl std::fmt::Display for HardwareError { ... }
impl std::error::Error for HardwareError { ... }
```

## Files to Modify

### Cargo.toml
```toml
[dependencies]
sysinfo = "0.38"  # NEW: CPU/cache detection
```

### src/cpu/mod.rs
```rust
// Add hardware module to CPU backend
pub mod hardware;

// Re-export for convenience
pub use hardware::{CpuCapabilities, BatchConfig, detect, derive_batch_config};
```

### src/cpu/prefill.rs
```rust
// Modify cpu_prefill_forward() to accept BatchConfig
pub fn cpu_prefill_forward(
    tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    start_pos: usize,
    config: &ModelConfig,
    batch_config: &BatchConfig,  // NEW
) -> Result<(), CpuError> {
    let seq_len = tokens.len();
    let batch_size = batch_config.max_tokens_per_batch;

    // Process in batches that fit in L3 cache
    for (batch_idx, batch_start) in (0..seq_len).step_by(batch_size).enumerate() {
        let batch_end = (batch_start + batch_size).min(seq_len);
        let batch_tokens = &tokens[batch_start..batch_end];
        let batch_pos = start_pos + batch_start;

        // Run prefill for this batch
        cpu_prefill_forward_batch(
            batch_tokens,
            weights,
            kv,
            scratch,
            batch_pos,
            config,
            batch_idx == 0,  // first_batch: allocate scratch
        )?;
    }

    Ok(())
}

// New internal function for single batch
fn cpu_prefill_forward_batch(
    tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    start_pos: usize,
    config: &ModelConfig,
    first_batch: bool,
) -> Result<(), CpuError> { ... }
```

### src/main.rs
```rust
use rocmforge::hardware::{detect, derive_batch_config};

fn run_cpu_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Detect hardware (BEFORE model loading)
    eprint!("Detecting CPU capabilities... ");
    let caps = detect().map_err(|e| format!("hardware detection: {}", e))?;
    eprintln!("done");
    eprintln!("  Physical cores: {}", caps.physical_cores);
    eprintln!("  L3 cache: {} MB", caps.l3_cache_bytes / (1024 * 1024));

    // Load GGUF file
    let file = GgufFile::open(&args.model)?;
    let config = ModelConfig::from_gguf(&file)?;

    // 2. Derive batch config from hardware + model
    let batch_config = derive_batch_config(&caps, &config);
    eprintln!("Batch config: max {} tokens/batch, use {} cores",
        batch_config.max_tokens_per_batch, batch_config.num_cores);

    // ... rest of loading (tokenizer, weights) ...

    // 3. Prefill with batch config
    cpu_prefill_forward(
        &prompt_tokens,
        &weights,
        &mut kv,
        &mut scratch,
        0,
        &config,
        &batch_config,  // NEW
    )?;

    // ... decode loop ...
}
```

## Testing Strategy

### Unit Tests (src/hardware/caps.rs tests)
```rust
#[test]
fn detect_returns_sensible_values() {
    let caps = CpuCapabilities::detect().unwrap();
    assert!(caps.physical_cores > 0);
    assert!(caps.logical_cpus >= caps.physical_cores);
    assert!(caps.total_memory_bytes > 0);
}

#[test]
fn compute_cores_returns_physical_only() {
    let caps = CpuCapabilities::detect().unwrap();
    let compute = caps.compute_cores();
    // Should NOT be affected by hyperthreading
    assert_eq!(compute, caps.physical_cores);
}
```

### Integration Tests (src/hardware/config.rs tests)
```rust
#[test]
fn memory_per_token_is_positive() {
    let config = make_test_config();
    let bytes = memory_per_token(&config);
    assert!(bytes > 0);
}

#[test]
fn batch_config_clamps_to_sensible_range() {
    let caps = CpuCapabilities {
        physical_cores: 8,
        logical_cpus: 16,
        l3_cache_bytes: 96 * 1024 * 1024,
        l2_cache_bytes: 2 * 1024 * 1024,
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
    };
    let config = make_test_config();
    let batch = BatchConfig::from_capabilities(&caps, &config);

    assert!(batch.max_tokens_per_batch >= 1);
    assert!(batch.max_tokens_per_batch <= 256);
    assert_eq!(batch.num_cores, 8); // Physical cores only
}
```

### Manual Verification
```bash
# Build and run
cargo build --release
./target/release/rocmforge --model qwen2.5-7b.gguf --prompt "Hello, world!" --debug

# Expected output:
# [Hardware] Physical cores: 8
# [Hardware] L3 cache: 96 MB
# [Batch config] max 32 tokens/batch, use 8 cores
# Prefill: 10.5ms (95.2 tok/s)  <- Should be faster than before
```

## Dependencies

- **sysinfo 0.38**: Cross-platform CPU/memory/cache detection
  - `System::physical_core_count()`
  - `System::total_memory()`
  - `Cpu::cache_sizes()` for L3/L2 cache

## Rollout Plan

1. Add `src/hardware/` module (caps.rs, config.rs, error.rs, mod.rs)
2. Add `sysinfo` to Cargo.toml
3. Write unit tests for detection and config derivation
4. Update `src/cpu/mod.rs` to export hardware module
5. Modify `src/cpu/prefill.rs` to accept BatchConfig and process in batches
6. Update `src/main.rs` to detect hardware before model load
7. Test with actual model and verify performance improvement
8. Add --show-hardware flag to display detected capabilities

## Success Criteria

- [ ] Hardware detection works on Linux, macOS, Windows
- [ ] Batch config derived correctly from actual L3 cache size
- [ ] Prefill processes tokens in batches that fit in cache
- [ ] Performance improves for prompts > 64 tokens (lower prefill latency)
- [ ] No regressions for small prompts (single token, 1-4 tokens)
- [ ] Fallback behavior when L3 cache info unavailable
- [ ] All existing tests still pass
