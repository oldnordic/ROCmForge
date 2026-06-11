# Performance Optimization Opportunities: Porting from rs3gw to rocmforge

## Executive Summary

rs3gw is a high-performance S3 gateway with several Rust optimization patterns that are directly applicable to rocmforge's model loading, inference server, and conversion pipeline. This document maps each technique to rocmforge's architecture.

---

## 1. Release Profile Optimizations

### Current State (rocmforge)
```toml
[profile.release]
opt-level = 3
lto = "thin"
```

### rs3gw Pattern
```toml
[profile.release]
opt-level = 3
lto = true          # Full LTO (not thin)
codegen-units = 1    # Single codegen unit for max optimization
panic = "abort"      # Smaller binary, no unwinding overhead
strip = true         # Remove debug symbols from release
```

### Applicability: ✅ HIGH
**What to port:** Add `codegen-units = 1`, `panic = "abort"`, and `strip = true` to rocmforge's release profile. Full LTO may increase build time significantly but can improve inference throughput by 5-15% through better inlining and dead code elimination.

**Risk:** `panic = "abort"` means no stack traces on panic — acceptable for release inference but may hinder debugging. Consider a `release-with-debug` profile (rs3gw has this).

**Implementation:**
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
strip = true

[profile.release-with-debug]
inherits = "release"
debug = true
strip = false
```

---

## 2. Zero-Copy Buffer Sharing with `bytes`

### rs3gw Pattern
rs3gw uses the `bytes` crate (`bytes = "1.11"`) for zero-copy sharing of buffers across async boundaries. `Bytes` is reference-counted and can be cloned in O(1) time without copying the underlying data.

**Where rs3gw uses it:**
- Streaming object data through the HTTP pipeline
- Cache entries (`MlCacheEntry { data: Bytes }`)
- gRPC/protobuf serialization

### Current State (rocmforge)
rocmforge copies `Vec<u8>` extensively:
- Tokenizer BPE keys: `BytesKey(Vec<u8>)` with `.clone()` on every lookup
- GGUF tensor views: `&[u8]` slices (good — zero-copy) but copied when converting to CPU/GPU weights
- Server inference: `prompt_tokens.clone()` per request, `String` clones for responses
- Model loading: tensors are copied from mmap into `Vec` for CPU weights, then again for GPU upload

### Applicability: ✅ HIGH
**What to port:**

1. **Tokenizer BPE:** Replace `BytesKey(Vec<u8>)` with `Bytes` keys. The tokenizer's `token_to_id` and `merges` HashMaps clone `Vec<u8>` on every encode operation. With `Bytes`, clone is O(1).

2. **Server response streaming:** Use `Bytes` for SSE event payloads instead of `String`→`Vec<u8>` conversion. The `tokio::sync::mpsc::unbounded_channel` in `handlers.rs` currently sends `String` which gets converted.

3. **Model tensor views:** Keep mmap-backed `&[u8]` for GGUF but wrap conversion pipeline buffers in `Bytes` to avoid copies between pipeline stages.

**Estimated impact:** 10-20% reduction in memory bandwidth during tokenization and server response generation.

---

## 3. Aligned Buffer Allocation (`AlignedBuffer`)

### rs3gw Pattern
rs3gw implements `AlignedBuffer` — a manually allocated buffer with 512-byte alignment (required for Linux `O_DIRECT`):

```rust
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        const ALIGNMENT: usize = 512;
        let aligned_size = (size + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let layout = Layout::from_size_align(aligned_size, ALIGNMENT)
            .expect("Invalid layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        // ...
    }
}
```

### Current State (rocmforge)
rocmforge needs alignment for SIMD kernels (AVX2 needs 32-byte, AVX512 needs 64-byte) but uses standard `vec![0.0f32; n]` allocations. The GPU upload path requires 256-byte alignment for HIP memory copy but does not guarantee it for CPU-side staging buffers.

### Applicability: ✅ MEDIUM-HIGH
**What to port:**

1. **CPU kernel scratch buffers:** Use `AlignedBuffer` (with 64-byte alignment for AVX-512) for `CpuForwardScratch` and `CpuKvCache` allocations. This avoids the `is_aligned` runtime checks in `src/gpu/kernels/mod.rs` and improves cache line behavior.

2. **GPU staging buffers:** The `rfm.rs` conversion pipeline already aligns to 256 bytes for GPU, but the CPU→GPU copy staging area uses `Vec<u8>` without guaranteed alignment.

3. **Direct I/O for model files:** For loading very large models (7B+ parameters), `O_DIRECT` with `AlignedBuffer` bypasses the Linux page cache, reducing memory pressure when multiple models are loaded sequentially.

**Implementation sketch:**
```rust
// In cpu/cache.rs or a new utils/aligned.rs
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    layout: std::alloc::Layout,
}

impl<T> AlignedVec<T> {
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        let size = std::mem::size_of::<T>() * len;
        let aligned_size = (size + align - 1) & !(align - 1);
        let layout = Layout::from_size_align(aligned_size, align).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        Self { ptr: ptr as *mut T, len, layout }
    }
}
```

---

## 4. ML-Based Smart Cache & Prefetching

### rs3gw Pattern
rs3gw implements `SmartCacheManager` with:
- Access pattern tracking (periodic, bursty, trending, declining, random)
- Exponential moving average (EMA) of access frequency
- Predictive prefetching with confidence scoring
- Adaptive TTL based on pattern type
- Cache warming strategies

### Current State (rocmforge)
rocmforge has no model-level caching strategy:
- Each inference request acquires a semaphore and runs from scratch
- No inference result caching (repeated prompts are re-computed)
- No KV cache warming between sessions
- No predictive loading (model weights are loaded on first request)

### Applicability: ✅ MEDIUM
**What to port:**

1. **Inference result cache:** For deterministic prompts (system prompts, common queries), cache the generated output keyed by (model_hash, prompt_hash, temperature, top_p). This avoids recomputation for identical requests.

2. **KV cache warming:** The `SmartCacheManager` pattern could track which conversation patterns lead to long contexts and pre-warm the KV cache with common prefixes.

3. **Model access prediction:** Track which models are loaded/unloaded and predictively keep hot models in memory. Currently `ModelManager` unloads immediately on request — an EMA-based priority score could delay eviction.

**Note:** This is lower priority than I/O and memory optimizations because rocmforge's inference time is dominated by GPU kernel execution, not cache misses. But for CPU-only deployments serving multiple models, it matters.

---

## 5. tokio-uring for Async I/O (Linux-only)

### rs3gw Pattern
rs3gw optionally uses `tokio-uring` for io_uring-based async I/O on Linux:
```toml
[target.'cfg(target_os = "linux")'.dependencies]
tokio-uring = { workspace = true, optional = true }
```

### Current State (rocmforge)
rocmforge uses standard `tokio::fs` operations for all file I/O:
- `File::open(path)?` in `loader/file.rs` (synchronous)
- `tokio::fs::copy` in `bin/convert/pipeline.rs`
- `std::fs::File` in `rfm.rs` write path

### Applicability: ✅ MEDIUM
**What to port:**

1. **Model conversion pipeline:** The `convert.rs` and `pipeline.rs` tools read/write multi-GB GGUF files. io_uring can overlap reads and writes, improving throughput by 20-40% for large sequential I/O.

2. **Server model loading:** When the API server loads a model on demand, io_uring can reduce the blocking time by overlapping tensor reads.

**Risk:** `tokio-uring` is Linux-only and requires the `io_uring` kernel feature. Feature-gate it appropriately.

---

## 6. Direct I/O (O_DIRECT) for Large Model Files

### rs3gw Pattern
rs3gw opens files with `O_DIRECT` (bypassing page cache) for objects above a configurable threshold (default 1MB). It pairs this with `AlignedBuffer` (512-byte alignment) since `O_DIRECT` requires aligned buffers.

```rust
#[cfg(target_os = "linux")]
pub async fn open_direct_io(path: &Path, write: bool) -> Result<File, ZeroCopyError> {
    let mut opts = StdOpenOptions::new();
    opts.custom_flags(0x4000); // O_DIRECT
    let file = opts.open(path)?;
    Ok(File::from_std(file))
}
```

### Current State (rocmforge)
rocmforge loads models through `memmap2` (which uses the page cache) or `BufReader` (which also uses page cache). For very large models (70B+), the page cache can evict other processes' memory or cause thrashing.

### Applicability: ✅ MEDIUM
**What to port:**

1. **RFM conversion output:** When writing the converted `.rfm` file, use `DirectIoWriter` for files > 1GB. This avoids polluting the page cache with write data that will never be read again by the OS.

2. **Sequential model loading:** For models loaded once and then mapped, `O_DIRECT` is counterproductive (you want the page cache). But for the conversion tool that reads one GGUF and writes one RFM, direct I/O avoids double-caching.

---

## 7. Zero-Copy File Transfer (splice/sendfile)

### rs3gw Pattern
rs3gw uses Linux `splice()` and `sendfile()` syscalls for kernel-level zero-copy file copying:
```rust
pub fn splice_copy(fd_in: RawFd, fd_out: RawFd, len: usize) -> Result<usize, ZeroCopyError> {
    unsafe {
        libc::splice(fd_in, null_mut(), fd_out, null_mut(), remaining, SPLICE_F_MOVE | SPLICE_F_MORE)
    }
}
```

### Current State (rocmforge)
rocmforge uses `tokio::fs::copy` (which copies through userspace buffers) or reads into `Vec<u8>` and writes back.

### Applicability: ✅ LOW-MEDIUM
**What to port:**

1. **Model file copying:** The `convert.rs` tool essentially copies tensor data from GGUF to RFM. Using `splice` or `sendfile` could skip the userspace copy for tensors that don't need format conversion.

2. **Server model download/streaming:** If rocmforge ever supports model download via HTTP, zero-copy file send is valuable.

---

## 8. Compression Optimizations

### rs3gw Pattern
rs3gw uses `oxiarc-zstd`, `oxiarc-lz4`, and `oxiarc-deflate` — high-performance compression crates. It benchmarks compression/decompression throughput.

### Current State (rocmforge)
rocmforge does not compress model files. The `.rfm` format stores raw quantized weights.

### Applicability: ✅ LOW
Not directly applicable — model weights are already quantized (Q4_0, Q5_K, etc.) which is a form of compression. Adding zstd/lz4 on top would hurt GPU upload speed. Skip this unless doing network transfer of models.

---

## Priority Ranking

| # | Optimization | Impact | Effort | Risk | Priority |
|---|-------------|--------|--------|------|----------|
| 1 | Release profile (`lto=true`, `codegen-units=1`, `panic=abort`) | High | 5 min | Low | **P0** |
| 2 | `bytes` crate for tokenizer/server | High | 2-4 hrs | Low | **P0** |
| 3 | `AlignedBuffer` for SIMD/GPU staging | Medium-High | 4-6 hrs | Low | **P1** |
| 4 | tokio-uring for conversion I/O | Medium | 1-2 days | Medium | **P2** |
| 5 | Direct I/O for conversion output | Medium | 4-6 hrs | Low | **P2** |
| 6 | ML-based cache/prefetch | Medium | 1-2 days | Medium | **P3** |
| 7 | splice/sendfile | Low-Medium | 2-4 hrs | Low | **P3** |
| 8 | Compression | Low | N/A | High | **Skip** |

---

## Implementation Plan

### Phase 1: Quick Wins (30 minutes)
1. Update `Cargo.toml` release profile
2. Add `bytes = "1"` to dependencies

### Phase 2: Tokenizer & Server Zero-Copy (1 day)
1. Replace `BytesKey(Vec<u8>)` with `Bytes`-based keys
2. Use `Bytes` in server SSE channel
3. Add `release-with-debug` profile

### Phase 3: SIMD Alignment (1-2 days)
1. Port `AlignedBuffer` from rs3gw (adapt to 64-byte for AVX-512)
2. Replace `CpuForwardScratch` and `CpuKvCache` `Vec` allocations with aligned buffers
3. Remove runtime alignment checks in GPU kernel dispatch

### Phase 4: I/O Optimization (2-3 days)
1. Feature-gate `tokio-uring` for Linux
2. Implement `DirectIoWriter` for RFM conversion
3. Benchmark conversion pipeline before/after

---

## Verification Commands

After implementing Phase 1:
```bash
cargo build --release
ls -lh target/release/rocmforge  # Should be smaller
```

After Phase 2:
```bash
cargo test tokenizer::tests  # Ensure correctness
cargo bench  # Compare throughput
```

After Phase 3:
```bash
cargo test cpu::ops_tests  # Ensure SIMD kernels still work
```

After Phase 4:
```bash
cargo run --bin convert -- --model qwen2.5-7b.gguf --output model.rfm
time cargo run --bin convert --features io_uring ...
```
