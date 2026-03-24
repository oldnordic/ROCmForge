# rocmforge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build AMD GPU/CPU inference engine for Qwen2.5 with metadata-driven configuration, TDD approach, and HTTP server.

**Architecture:** Direct port from Memoria with modularization for 1K LOC limit. Two explicit execution paths (CPU/GPU), no fallback. Metadata-driven config from GGUF files.

**Tech Stack:** Rust 2021, HIP (AMD GPU), memmap2, half, regex, rayon, axum (HTTP server)

---

## Phase 1: Project Scaffold

### Task 1.1: Cargo.toml and Directory Structure

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/main.rs` (minimal)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "rocmforge"
version = "0.1.0"
edition = "2021"
description = "AMD-first LLM inference engine for Qwen model family"

[[bin]]
name = "rocmforge"
path = "src/main.rs"

[dependencies]
# GGUF loading
memmap2 = "0.9"
half = "2"
# Tokenizer
regex = "1"
once_cell = "1"
# GPU runtime loading
libloading = "0.8"
# CPU backend parallelism
rayon = "1"
num_cpus = "1"
# HTTP server
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
# Random for sampling
rand = "0.8"

[dev-dependencies]
serial_test = "3"

[profile.release]
opt-level = 3
lto = "thin"
```

**Step 2: Create src/lib.rs**

```rust
pub mod config;
pub mod cpu;
pub mod gpu;
pub mod loader;
pub mod tokenizer;
pub mod server;
```

**Step 3: Create minimal src/main.rs**

```rust
fn main() {
    println!("rocmforge - AMD LLM inference engine");
}
```

**Step 4: Verify build**

Run: `cargo build`
Expected: Compiles without errors

**Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/main.rs
git commit -m "feat: initialize project scaffold

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Phase 2: Loader (GGUF Parsing)

### Task 2.1: Loader Error Types

**Files:**
- Create: `src/loader/mod.rs`
- Create: `src/loader/error.rs`

**Step 1: Create src/loader/error.rs**

```rust
/// Errors from GGUF file loading and parsing.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    InvalidMagic { expected: u32, found: u32 },
    InvalidVersion { supported: u32, found: u32 },
    MissingTensor(String),
    MissingMetadata(String),
    InvalidTensorShape { name: String, expected: usize, found: usize },
    Utf8Error(std::str::Utf8Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "IO error: {}", e),
            LoadError::InvalidMagic { expected, found } => {
                write!(f, "invalid GGUF magic: expected 0x{:08x}, found 0x{:08x}", expected, found)
            }
            LoadError::InvalidVersion { supported, found } => {
                write!(f, "unsupported GGUF version: supported {}, found {}", supported, found)
            }
            LoadError::MissingTensor(name) => write!(f, "missing tensor: {}", name),
            LoadError::MissingMetadata(key) => write!(f, "missing metadata: {}", key),
            LoadError::InvalidTensorShape { name, expected, found } => {
                write!(f, "invalid tensor shape for '{}': expected {} dims, found {}", name, expected, found)
            }
            LoadError::Utf8Error(e) => write!(f, "UTF-8 error: {}", e),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<std::str::Utf8Error> for LoadError {
    fn from(e: std::str::Utf8Error) -> Self {
        LoadError::Utf8Error(e)
    }
}
```

**Step 2: Create src/loader/mod.rs**

```rust
mod error;

pub use error::LoadError;
```

**Step 3: Verify build**

Run: `cargo build`
Expected: Compiles without errors

**Step 4: Commit**

```bash
git add src/loader/mod.rs src/loader/error.rs
git commit -m "feat(loader): add error types

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 2.2: GGML Quantization Types

**Files:**
- Create: `src/loader/ggml_type.rs`
- Modify: `src/loader/mod.rs`

**Step 1: Create src/loader/ggml_type.rs**

```rust
/// GGML quantization types from GGUF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2S = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2Ss = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Bf16 = 29,
}

impl GgmlType {
    /// Convert from GGUF type code.
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(GgmlType::F32),
            1 => Some(GgmlType::F16),
            2 => Some(GgmlType::Q4_0),
            3 => Some(GgmlType::Q4_1),
            6 => Some(GgmlType::Q5_0),
            7 => Some(GgmlType::Q5_1),
            8 => Some(GgmlType::Q8_0),
            10 => Some(GgmlType::Q2K),
            11 => Some(GgmlType::Q3K),
            12 => Some(GgmlType::Q4K),
            13 => Some(GgmlType::Q5K),
            14 => Some(GgmlType::Q6K),
            15 => Some(GgmlType::Q8K),
            _ => None,
        }
    }

    /// Size in bytes per block for quantized types, or per element for F32/F16.
    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 20,  // 32 values: 2 bytes scale + 16 bytes quants
            GgmlType::Q4_1 => 24,  // 32 values: 2 bytes scale + 2 bytes min + 16 bytes quants
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 26,
            GgmlType::Q8_0 => 34,  // 32 values: 2 bytes scale + 32 bytes quants
            GgmlType::Q2K => 256,
            GgmlType::Q3K => 256,
            GgmlType::Q4K => 256,
            GgmlType::Q5K => 256,
            GgmlType::Q6K => 256,
            GgmlType::Q8K => 256,
            _ => 4,
        }
    }

    /// Number of elements per block for quantized types.
    pub fn elements_per_block(&self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => 32,
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8K => 256,
            _ => 1,
        }
    }

    /// Check if this is a quantized type requiring dequantization.
    pub fn is_quantized(&self) -> bool {
        matches!(self,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 |
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8K
        )
    }
}
```

**Step 2: Update src/loader/mod.rs**

```rust
mod error;
mod ggml_type;

pub use error::LoadError;
pub use ggml_type::GgmlType;
```

**Step 3: Verify build**

Run: `cargo build`
Expected: Compiles without errors

**Step 4: Commit**

```bash
git add src/loader/ggml_type.rs src/loader/mod.rs
git commit -m "feat(loader): add GGML quantization types

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 2.3: GGUF Metadata Parsing

**Files:**
- Create: `src/loader/metadata.rs`
- Modify: `src/loader/mod.rs`

**Step 1: Create src/loader/metadata.rs**

Port from Memoria's `src/loader/metadata.rs` - contains:
- `GgufMetadata` struct with all GGUF KV pairs
- `GgufValueType` enum
- Parsed metadata accessors (block_count, embedding_length, etc.)

**Step 2: Update src/loader/mod.rs**

Add `mod metadata;` and export `GgufMetadata`.

**Step 3: Add tests for metadata parsing**

Create tests that verify metadata extraction from GGUF files.

**Step 4: Commit**

```bash
git add src/loader/metadata.rs src/loader/mod.rs
git commit -m "feat(loader): add GGUF metadata parsing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 2.4: GGUF File and Tensor Access

**Files:**
- Create: `src/loader/file.rs`
- Create: `src/loader/parse.rs`
- Modify: `src/loader/mod.rs`

**Step 1: Create src/loader/file.rs**

Port from Memoria's `src/loader/file.rs` - contains:
- `GgufFile` struct with memmap access
- `Tensor` struct with pointer into mmap
- `tokenizer_data()` accessor
- `tensor_names()` iterator

**Step 2: Create src/loader/parse.rs**

Port from Memoria's `src/loader/parse.rs` - contains:
- GGUF header parsing (magic, version, counts)
- KV pair parsing
- Tensor info parsing

**Step 3: Update exports in mod.rs**

**Step 4: Verify with test GGUF file**

**Step 5: Commit**

---

## Phase 3: Config (Model Configuration)

### Task 3.1: ModelConfig and ModelTraits

**Files:**
- Create: `src/config.rs`

Port from Memoria's `src/config.rs`:
- `RopeStyle` enum (Normal, NeoX)
- `AttentionLayout` enum (SplitQkv, FusedQkv)
- `ModelTraits` struct with registry
- `ModelConfig` struct with all dimensions
- `ConfigError` type
- `ChatTemplate` enum
- `detect_chat_template()` function

Key: Registry-based traits lookup, metadata-driven values.

---

## Phase 4: Tokenizer (BPE)

### Task 4.1: BPE Tokenizer

**Files:**
- Create: `src/tokenizer/mod.rs`
- Create: `src/tokenizer/bpe.rs`

Port from Memoria's `src/tokenizer/bpe.rs`:
- `BpeTokenizer` struct
- `encode()` method
- `decode()` method
- `decode_token()` for streaming
- `is_eog()` for EOS detection
- Merge rule application

---

## Phase 5: CPU Backend

### Task 5.1: CPU Weights

**Files:**
- Create: `src/cpu/mod.rs`
- Create: `src/cpu/weights.rs`

Port from Memoria:
- `CpuModelWeights` with dequantized f32 weights
- `LayerWeights` per-layer struct
- Weight loading from GGUF tensors

### Task 5.2: CPU KV Cache

**Files:**
- Create: `src/cpu/cache.rs`

Port from Memoria:
- `CpuKvCache` struct
- Cache allocation based on config
- Position indexing

### Task 5.3: CPU Forward Pass

**Files:**
- Create: `src/cpu/forward.rs`
- Create: `src/cpu/ops.rs`

Port from Memoria:
- `cpu_embed_token()` - embedding lookup
- `layer_forward()` - single layer
- `cpu_full_forward()` - all layers
- RMS norm, matmul, RoPE ops

### Task 5.4: CPU Prefill

**Files:**
- Create: `src/cpu/prefill.rs`

Port from Memoria:
- `cpu_prefill_forward()` - batched prompt processing

### Task 5.5: CPU Sampler

**Files:**
- Create: `src/cpu/sampler.rs`

Port from Memoria:
- `cpu_sample_greedy()` - argmax
- `cpu_sample_top_p()` - nucleus sampling

---

## Phase 6: GPU Backend (HIP)

### Task 6.1: HIP Context

**Files:**
- Create: `src/gpu/mod.rs`
- Create: `src/gpu/context.rs`

Port from Memoria:
- `GpuContext` struct with HIP device
- `DeviceBuffer` for VRAM allocation
- `is_available()` check
- Device info (name, VRAM)

### Task 6.2: HIP Kernels

**Files:**
- Create: `gpu/libgpu.hip`

Port from Memoria's HIP kernels:
- `gpu_init()`, `gpu_cleanup()`
- `gpu_available()`, `gpu_device_name()`
- `gpu_vram_free()`, `gpu_vram_total()`
- `rms_norm()` kernel
- `rope_neox()` kernel
- `matmul_f16_f32()` kernel
- `softmax()` kernel
- Sampling kernels

Build: `hipcc -O3 -fPIC -shared gpu/libgpu.hip -o gpu/libgpu.so`

### Task 6.3: GPU Weights

**Files:**
- Create: `src/gpu/weights.rs`

Port from Memoria:
- `ModelWeights` with VRAM allocation
- Weight upload to GPU
- Dequantize-on-GPU support

### Task 6.4: GPU KV Cache

**Files:**
- Create: `src/gpu/cache.rs`

Port from Memoria:
- `KvCache` with VRAM allocation

### Task 6.5: GPU Forward/Prefill/Sampler

**Files:**
- Create: `src/gpu/forward.rs`
- Create: `src/gpu/prefill.rs`
- Create: `src/gpu/sampler.rs`

---

## Phase 7: CLI Entry Point

### Task 7.1: Main CLI

**Files:**
- Modify: `src/main.rs`

Port from Memoria's main.rs:
- Argument parsing
- Device selection (`--device cpu|gpu`)
- `run_cpu()` function
- `run_gpu()` function
- Inference loop
- Output streaming

---

## Phase 8: HTTP Server

### Task 8.1: Server Scaffold

**Files:**
- Create: `src/server/mod.rs`
- Create: `src/server/router.rs`
- Create: `src/server/model.rs`

### Task 8.2: OpenAI API

**Files:**
- Create: `src/server/openai.rs`

### Task 8.3: Claude API

**Files:**
- Create: `src/server/claude.rs`

---

## Execution Order

1. Phase 1: Project Scaffold
2. Phase 2: Loader (direct port from Memoria)
3. Phase 3: Config (direct port)
4. Phase 4: Tokenizer (direct port)
5. Phase 5: CPU Backend (port + modularize)
6. Phase 6: GPU Backend (port CUDA→HIP)
7. Phase 7: CLI Entry Point
8. Phase 8: HTTP Server

Each phase: TDD, <1K LOC per file, commit after each task.
