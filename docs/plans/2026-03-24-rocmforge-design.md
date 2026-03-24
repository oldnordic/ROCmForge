# rocmforge Design Document

**Date:** 2026-03-24
**Purpose:** AMD GPU/CPU inference engine, ported from Memoria

---

## Overview

rocmforge is an AMD-first LLM inference engine for the Qwen model family, starting with Qwen2.5.
It provides two execution hotpaths (CPU and GPU) and an HTTP server with OpenAI and Claude APIs.

**Key Constraints:**
- Max 1K LOC per file
- TDD approach (tests written first)
- No guessing/no assumptions — code truth only
- Metadata-driven: all config from GGUF, no hardcoded values

---

## Architecture

### High-Level Structure

```
rocmforge/
├── Cargo.toml
├── src/
│   ├── main.rs                   # CLI entry (device selection)
│   ├── lib.rs                    # Module exports
│   ├── config.rs                 # ModelConfig, traits for Qwen2.5
│   ├── loader/                   # GGUF loading (port directly)
│   │   ├── mod.rs
│   │   ├── error.rs
│   │   ├── file.rs
│   │   ├── metadata.rs
│   │   ├── ggml_type.rs
│   │   └── parse.rs
│   ├── tokenizer/                # BPE tokenizer (port directly)
│   │   ├── mod.rs
│   │   └── bpe.rs
│   ├── cpu/                     # CPU backend (port + modularize)
│   │   ├── mod.rs
│   │   ├── forward.rs            # Layer forward pass
│   │   ├── prefill.rs           # Prefill (batched)
│   │   ├── sampler.rs           # Greedy + top-p
│   │   ├── weights.rs           # CpuModelWeights, LayerWeights
│   │   ├── cache.rs             # CpuKvCache
│   │   └── ops.rs              # CPU ops
│   ├── gpu/                     # HIP backend (port CUDA → HIP)
│   │   ├── mod.rs
│   │   ├── context.rs            # GpuContext, DeviceBuffer
│   │   ├── forward.rs            # GPU layer forward
│   │   ├── prefill.rs           # GPU prefill
│   │   ├── sampler.rs           # GPU sampling
│   │   ├── weights.rs           # GpuModelWeights
│   │   └── cache.rs             # GpuKvCache
│   └── server/                  # HTTP server
│       ├── mod.rs
│       ├── openai.rs            # OpenAI-compatible API
│       ├── claude.rs            # Claude-compatible API
│       ├── router.rs            # Request routing
│       └── model.rs             # Shared model state
└── gpu/
    └── libgpu.hip              # HIP kernels (max 1K LOC)
```

### Execution Paths

1. **CLI Mode**: `--device cpu|gpu` → routes to `run_cpu()` or `run_gpu()`
2. **Server Mode**: `--server --port N` → HTTP API (OpenAI + Claude)
3. **Explicit Selection**: No runtime fallback between CPU/GPU
4. **Batch-Ready**: API designed for batching, single-request initially

---

## Core Components

### 1. Model Config (`config.rs`)

- `ModelConfig`: num_layers, hidden_size, num_heads, rope_neox, etc.
- `RopeStyle`: Normal, NeoX
- `AttentionLayout`: SplitQkv, FusedQkv
- `ChatTemplate`: None, ChatML, LLaMA3, etc.
- `ModelTraits`: Registry-based, architecture string lookup

**Qwen2.5-specific traits:**
```rust
ModelTraits {
    rope_style: NeoX,          // Split-half pairs
    attention_layout: SplitQkv,  // Separate Q/K/V
    use_attention_bias: true,    // Qwen2/2.5 has bias
    default_rope_theta: 1_000_000.0,
    default_norm_eps: 1e-6,
}
```

### 2. Loader (`loader/`)

Direct port from Memoria, metadata-driven:
- `GgufFile`: memmap'd GGUF with metadata parsing
- `Tensor`: Raw weight tensor + dims + GGML type
- `LoadError`: Parsing errors

### 3. Tokenizer (`tokenizer/bpe.rs`)

Direct port from Memoria:
- `BpeTokenizer`: encode/decode with merge rules
- `TokenizerData`: from GGUF `tokenizer.json`
- ChatML template for Qwen2.5

### 4. CPU Backend (`cpu/`)

| Module | Purpose | Key Types |
|---------|----------|-----------|
| `weights.rs` | Dequantized f32 weights in RAM | `CpuModelWeights`, `LayerWeights` |
| `cache.rs` | KV cache in RAM | `CpuKvCache`, `[f32; ...]` buffers |
| `forward.rs` | Single-token layer forward | `layer_forward()`, `cpu_full_forward()` |
| `prefill.rs` | Batched prompt processing | `cpu_prefill_forward()` |
| `sampler.rs` | Greedy + top-p | `cpu_sample_greedy()`, `cpu_sample_top_p()` |
| `ops.rs` | Vector ops (rms_norm, matmul) | Pure f32 ops |

### 5. GPU Backend (`gpu/`)

| Module | Purpose | Key Types |
|---------|----------|-----------|
| `context.rs` | HIP device + buffer mgmt | `GpuContext`, `DeviceBuffer` |
| `weights.rs` | Weights copied to VRAM | `GpuModelWeights`, dequantize kernels |
| `cache.rs` | KV cache in VRAM | `GpuKvCache`, pinned buffers |
| `forward.rs` | GPU layer forward | `layer_forward()`, `full_forward()` |
| `prefill.rs` | Batched GPU prefill | `prefill_forward()` |
| `sampler.rs` | GPU sampling | `Sampler::sample_greedy()`, `sample_top_p()` |

### 6. HIP Kernels (`gpu/libgpu.hip`)

Required kernels (MVP only):
- `rms_norm()` - RMS normalization
- `rope_neox()` - Qwen2.5 RoPE (NeoX style)
- `matmul_f16_f32()` - GEMM for dequantized weights
- `softmax()` - Attention softmax
- `sample_greedy()`, `sample_top_p()` - Sampling

Max 1K LOC → likely 2-3 kernel files: `ops.hip`, `attention.hip`, `sampler.hip`

### 7. HTTP Server (`src/server/`)

#### OpenAI-compatible API
```
POST /v1/chat/completions     # Chat completion
POST /v1/completions         # Text completion
GET  /v1/models              # Model list
GET  /health                 # Health check
```

#### Claude-compatible API
```
POST /v1/messages               # Message API
GET  /v1/models                # Model list
GET  /health                   # Health check
```

#### Model State
```rust
pub struct InferenceModel {
    backend: Backend,
    tokenizer: BpeTokenizer,
    config: ModelConfig,
    max_batch: usize,
    queue: RequestQueue,
}

pub enum Backend {
    Cpu(CpuModelWeights, CpuKvCache, ...),
    Gpu(GpuContext, GpuModelWeights, GpuKvCache, ...),
}
```

---

## Data Flow

### CLI Inference Flow

```
user prompt
    ↓
[main.rs] --device selection (explicit: cpu OR gpu)
    ├─→ CPU Path
    │   ↓
    │ [loader] GgufFile::open()
    │   ↓
    │ [config] ModelConfig::from_gguf() (metadata-driven)
    │   ↓
    │ [tokenizer] BpeTokenizer::from_gguf()
    │   ↓
    │ [weights] CpuModelWeights::load()
    │   ↓
    │ [cache] CpuKvCache::new()
    │   ↓
    │ [prefill] cpu_prefill_forward() (batched)
    │   ↓
    │ [sampler] cpu_sample_greedy/top_p()
    │   ↓
    │ DECODE LOOP: embed_token() → cpu_full_forward() → sample()
    │   ↓
    │ [repeat until EOS/max_tokens]
    │
    └─→ GPU Path
        ↓
        [context] GpuContext::init()
        ↓
        [weights] ModelWeights::load() (dequantize on GPU)
        ↓
        [cache] KvCache::new()
        ↓
        [prefill] prefill_forward() (batched, GPU)
        ↓
        [sampler] Sampler::sample()
        ↓
        DECODE LOOP: embed_token() → full_forward() → sample()
        ↓
        [repeat until EOS/max_tokens]
```

### HTTP Server Flow

```
HTTP Request (OpenAI or Claude API)
    ↓
[server/router.rs] route()
    ├─→ /v1/chat/completions → [server/openai.rs] → parse OpenAI format
    ├─→ /v1/messages → [server/claude.rs] → parse Claude format
    └─→ [server/model.rs] → InternalInferenceRequest
        ↓
        Backend::Cpu OR Backend::Gpu
        ↓
        [same flow as CLI: prefill → decode loop]
        ↓
        InternalInferenceResponse
        ↓
        ├─→ [openai.rs] → format OpenAI response
        └─→ [claude.rs] → format Claude response
            ↓
            HTTP 200 + JSON
```

---

## Metadata-Driven Design (Critical Principle)

**All configuration derived from GGUF metadata, no hardcoded assumptions.**

```
GGUF File → Metadata Detection → All Config

Model architecture     → from ggml.architecture (e.g., "qwen2")
Model dimensions       → from ggml.block_count, ggml.embedding_length, etc.
Tokenizer type         → from ggml.tokenizer.model
Vocabulary            → from ggml.tokenizer.tokens (not ggml.vocab_size!)
RoPE style            → from ModelTraits registry (based on architecture)
Attention layout       → from ModelTraits registry (based on architecture)
Quantization type      → from tensor.ggml_type
```

**Key Loader Behaviors:**

1. **Vocab Size**: Taken from `tokenizer_data.tokens.len()`, NOT from GGUF `vocab_size` key
   - Qwen2.5 GGUF reports vocab_size=0, but tokenizer has full vocab

2. **Intermediate Size**: Inferred from tensor shape if metadata missing
   - Looks for `blk.0.ffn_gate.weight` or similar pattern

3. **Architecture Traits**: Registry-based, string lookup
   - `ModelTraits::for_arch("qwen2")` → specific traits
   - `ModelTraits::for_arch("unknown")` → falls back to LLaMA defaults

4. **No Model-Specific Code in Inference Paths**
   - CPU/GPU forward uses config.rope_neox flag, NOT `if model == "qwen"`
   - Kernels take parameters, NOT model-specific #ifdefs
   - KV cache allocates based on config, NOT model name

---

## Error Handling

### Error Types

```rust
// loader/error.rs
pub enum LoadError {
    Io(std::io::Error),
    InvalidGguf(String),
    MissingTensor(String),
    InvalidMetadata(String),
}

// cpu/mod.rs
pub enum CpuError {
    UnsupportedWeightType(GgmlType),
    DimensionMismatch(&'static str),
    InvalidOperation(String),
}

// gpu/mod.rs
pub enum GpuError {
    InitFailed(String),
    DeviceNotFound,
    VramExceeded { required: usize, available: usize },
    KernelLaunchFailed(String),
}

// config.rs
pub enum ConfigError {
    Missing(&'static str),
    Invalid(String),
    Load(LoadError),
}

// tokenizer/mod.rs
pub enum TokenizerError {
    InvalidToken(u32),
    EncodingFailed(String),
    DecodingFailed(String),
}

// server/mod.rs
pub enum ServerError {
    InvalidRequest(String),
    ModelNotLoaded,
    InferenceFailed(String),
    BadRequest(String),
}
```

### Error Propagation

```
CLI: LoadError/ConfigError/CpuError/GpuError → Box<dyn Error> → exit(1)
Server: Backend errors → ServerError → HTTP 500 / 400
```

### No Fallback (Explicit Device)

- GPU init fails → error, no retry on CPU
- User must restart with correct `--device` flag
- No runtime switching

---

## Testing (TDD Approach)

### Test Structure

```
tests/
├── common/
│   └── mod.rs              # Shared utilities, fixtures
├── integration/
│   ├── loader.rs           # GGUF loading tests
│   ├── tokenizer.rs        # Tokenization tests
│   ├── config.rs          # Config + traits tests
│   ├── cpu_forward.rs     # CPU forward pass tests
│   ├── cpu_prefill.rs     # CPU prefill tests
│   ├── cpu_sampler.rs     # CPU sampling tests
│   ├── gpu_forward.rs     # GPU forward pass tests
│   ├── gpu_prefill.rs     # GPU prefill tests
│   ├── gpu_sampler.rs     # GPU sampling tests
│   └── server.rs         # HTTP API tests
└── unit/
    ├── ops.rs             # CPU ops unit tests
    └── kernel.rs          # Kernel correctness tests
```

### TDD Workflow

1. Write failing test first
2. Implement minimal code to pass
3. Refactor (keep < 1K LOC)
4. Run all tests (no regression)

### Test Fixtures

- `fixtures/qwen2.5-tiny.gguf` - Minimal model for fast tests
- `fixtures/tokenizer.json` - Standalone tokenizer

---

## Dependencies (from Memoria)

```toml
memmap2 = "0.9"
half = "2"
regex = "1"
once_cell = "1"
libloading = "0.8"
rayon = "1"
num_cpus = "1"
rand = "0.8"

# HTTP server (new)
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
criterion = "0.5"
serial_test = "3"
```

---

## Implementation Phases

1. **Project Scaffold** - Cargo.toml, directory structure
2. **Loader + Config** - GGUF loading, metadata-driven config
3. **Tokenizer** - BPE implementation
4. **CPU Backend** - forward, prefill, sampler, cache
5. **GPU Backend** - HIP context, kernels
6. **CLI** - device routing, inference loop
7. **HTTP Server** - OpenAI + Claude APIs

Each phase: TDD first, ensure < 1K LOC per file, all tests pass.
