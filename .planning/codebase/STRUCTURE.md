# Codebase Structure

**Analysis Date:** 2026-01-20

## Directory Layout

```
ROCmForge/
├── src/                      # Main library source
│   ├── attention/            # Multi-head attention (CPU/GPU kernels)
│   ├── backend/              # CPU and HIP/ROCm backends
│   │   ├── cpu/              # CPU SIMD operations
│   │   ├── hip_backend/      # HIP wrapper and GPU execution
│   ├── ggml/                 # GGML graph IR and backends
│   │   ├── hip_backend/      # GGML HIP ops (matmul, copy, etc.)
│   ├── gguf/                 # GGUF loader (merged into loader/)
│   ├── http/                 # HTTP API server
│   ├── kv_cache/             # Paged KV cache implementation
│   ├── loader/               # GGUF/ONNX model loaders
│   ├── mlp/                  # MLP GPU kernels (SwiGLU, RMSNorm)
│   ├── model/                # Model configs and execution plans
│   │   └── execution_plan/   # Layer-wise execution mapping
│   ├── ops/                  # High-level GPU operations
│   ├── prompt/               # Prompt processing optimization
│   ├── sampler/              # Token sampling (top-k, top-p, temperature)
│   ├── scheduler/            # Continuous batching scheduler
│   ├── tensor/               # Basic tensor operations
│   ├── bin/                  # Binary entry points
│   ├── context/              # SQLiteGraph context engine (feature-gated)
│   ├── logging/              # Tracing/logging setup
│   ├── metrics/              # Prometheus metrics
│   ├── otel_traces.rs        # OpenTelemetry span export
│   ├── profiling/            # ROCprof integration
│   ├── tokenizer.rs          # Tokenizer wrapper
│   ├── models.rs             # Model registry
│   ├── error.rs              # Unified error types
│   ├── engine.rs             # Main inference engine
│   └── lib.rs                # Library root, public exports
├── tests/                    # Integration tests
│   ├── common/               # Shared test fixtures
│   ├── attention/            # Attention-specific tests
│   └── data/                 # Test data files
├── benches/                  # Criterion benchmarks
├── docs/                     # Project documentation
├── models/                   # Downloaded GGUF models (gitignored)
├── .planning/                # Phase plans and codebase analysis
├── Cargo.toml                # Project manifest
├── CLAUDE.md                 # Development rules
└── README.md                 # Project overview
```

## Directory Purposes

**`src/attention/`:**
- Purpose: Multi-head attention computation with CPU/GPU backends
- Contains: Kernels (QK^T, softmax, weighted matmul), FlashAttention, PagedAttention, RoPE, causal masks
- Key files: `src/attention/flash_attention.rs`, `src/attention/paged_kernel.rs`, `src/attention/rope.rs`

**`src/backend/`:**
- Purpose: Low-level GPU/CPU abstraction
- Contains: `HipBackend` (main GPU wrapper), `DeviceTensor`, `HipBuffer`, `ModelRuntime`, CPU SIMD
- Key files: `src/backend/hip_backend/backend.rs`, `src/backend/gpu_executor.rs`

**`src/ggml/`:**
- Purpose: GGML graph computation and backend abstraction
- Contains: `Graph`, `Node`, `Op`, `GgmlBackend` trait, hybrid scheduler
- Key files: `src/ggml/graph.rs`, `src/ggml/hybrid_scheduler.rs`

**`src/http/`:**
- Purpose: HTTP API server (OpenAI-compatible)
- Contains: Route handlers, SSE streaming, request/response models
- Key files: `src/http/mod.rs`

**`src/kv_cache/`:**
- Purpose: Paged key-value cache for efficient attention
- Contains: `KvCache`, `PageTable`, `BlockAllocator`, LRU eviction
- Key files: `src/kv_cache/kv_cache.rs`, `src/kv_cache/page_table.rs`, `src/kv_cache/block_allocator.rs`

**`src/loader/`:**
- Purpose: Parse GGUF/ONNX models and load weights
- Contains: `GgufLoader`, dequantization (Q2-K through Q8_0), MXFP support, lazy tensors
- Key files: `src/loader/mod.rs`, `src/loader/gguf.rs`, `src/loader/dequant.rs`

**`src/mlp/`:**
- Purpose: Feed-forward layer GPU kernels
- Contains: SwiGLU activation, RMSNorm kernel implementations
- Key files: `src/mlp/kernels.rs`

**`src/model/`:**
- Purpose: Model configuration and execution planning
- Contains: `SimpleTransformer`, `ExecutionPlan`, architecture detection, weight mapping
- Key files: `src/model/simple_transformer.rs`, `src/model/execution_plan/layer_plan.rs`, `src/model/execution_plan/architecture.rs`

**`src/sampler/`:**
- Purpose: Token sampling from logits
- Contains: Temperature scaling, top-k filtering, top-p (nucleus) sampling, repetition penalty
- Key files: `src/sampler/sampler.rs`

**`src/scheduler/`:**
- Purpose: Continuous batching for high GPU utilization
- Contains: `Scheduler`, `GenerationRequest`, `IterationBatch`, queue management
- Key files: `src/scheduler/scheduler.rs`

**`src/prompt/`:**
- Purpose: Prompt processing optimizations
- Contains: Chunking strategies, prefix caching, batch attention
- Key files: `src/prompt/chunking.rs`, `src/prompt/cache.rs`, `src/prompt/batch_attention.rs`

**`src/tensor/`:**
- Purpose: Basic tensor data structures and operations
- Contains: `Tensor` (host-side), matmul operations
- Key files: `src/tensor/matmul.rs`

**`src/ops/`:**
- Purpose: High-level GPU operations
- Contains: QKV projection, attention GPU wrappers
- Key files: `src/ops/attention_gpu.rs`, `src/ops/qkv.rs`

**`src/bin/`:**
- Purpose: Executable entry points
- Contains: CLI, server, debug tools
- Key files: `src/bin/rocmforge_cli.rs`, `src/bin/inspect_gguf.rs`

**`tests/`:**
- Purpose: Integration and end-to-end tests
- Contains: Model loading, inference pipeline, GPU kernels
- Key files: `tests/glm_model_tests.rs`, `tests/transformer_integration_tests.rs`, `tests/sampling_gpu_tests.rs`

**`benches/`:**
- Purpose: Performance benchmarks
- Contains: Attention, matmul, dequant, inference, memory benchmarks
- Key files: `benches/attention_bench.rs`, `benches/matmul_bench.rs`

## Key File Locations

**Entry Points:**
- `src/lib.rs`: Library root, module declarations, public API exports
- `src/bin/rocmforge_cli.rs`: Main CLI (models, generate, serve commands)
- `src/engine.rs`: `InferenceEngine` - main inference orchestration

**Configuration:**
- `Cargo.toml`: Dependencies, features, build configuration
- `CLAUDE.md`: Development rules and conventions

**Core Logic:**
- `src/attention/flash_attention.rs`: Fused attention kernel
- `src/kv_cache/kv_cache.rs`: Paged KV cache implementation
- `src/scheduler/scheduler.rs`: Continuous batching scheduler
- `src/sampler/sampler.rs`: Token sampling algorithms
- `src/loader/gguf.rs`: GGUF format parser
- `src/model/simple_transformer.rs`: Minimal transformer model

**GPU Backend:**
- `src/backend/hip_backend/backend.rs`: HIP wrapper, device management
- `src/backend/gpu_executor.rs`: GPU kernel execution
- `src/backend/scratch.rs`: Scratch buffer management

**Error Handling:**
- `src/error.rs`: Unified `RocmForgeError` with categorization

**Testing:**
- `tests/glm_model_tests.rs`: GLM model integration tests
- `tests/transformer_integration_tests.rs`: End-to-end transformer tests
- `tests/decode_step_integration_tests.rs`: Single-token decode tests
- `tests/sampling_gpu_tests.rs`: GPU sampling verification

## Naming Conventions

**Files:**
- `mod.rs`: Module exports and re-exports
- `{module}_tests.rs`: Unit tests co-located with module (under `src/`)
- `{feature}_tests.rs`: Integration tests (under `tests/`)
- `{module}.rs`: Single-type modules (e.g., `error.rs`, `engine.rs`)

**Directories:**
- `src/`: Library source code
- `tests/`: Integration tests (compiled as separate crates)
- `benches/`: Criterion benchmarks
- `src/bin/`: Executable entry points

**Functions:**
- Public API: `snake_case`
- FFI bindings: `hip` prefix (e.g., `hipGetDeviceCount`)
- Tests: `test_{feature}` or `test_{scenario}_{expected_result}`

**Types:**
- Structs: `PascalCase`
- Enums: `PascalCase`
- Type aliases: `{Module}Result<T>`, `{Module}Error`

**Modules:**
- `snake_case` matching directory name
- Test modules: `{module}_tests` (feature-gated with `#[cfg(test)]`)

**Constants:**
- `SCREAMING_SNAKE_CASE`

## Where to Add New Code

**New Feature (Inference):**
- Primary code: `src/model/` or `src/attention/` or `src/mlp/`
- Tests: `tests/` with `{feature}_tests.rs` naming
- Kernels: `src/attention/kernels.rs` or `src/mlp/kernels.rs`

**New Model Architecture:**
- Detection: `src/model/execution_plan/architecture.rs` (add `Architecture` enum variant)
- Weight mapping: `src/model/execution_plan/layer_plan.rs`
- Tests: `tests/{architecture}_model_tests.rs`

**New GPU Kernel:**
- Implementation: `src/attention/` or `src/mlp/`
- Tests: Co-located `{module}_kernel_tests.rs` under `src/` with `#[cfg(test)]` and `#[cfg(feature = "rocm")]`
- Use `#[serial]` attribute for GPU isolation

**New HTTP Endpoint:**
- Handler: `src/http/mod.rs`
- Tests: Create `tests/http_{feature}_tests.rs` (needs integration test setup)

**New CLI Command:**
- Implementation: `src/bin/rocmforge_cli.rs` (add subcommand to `Cli` enum)
- Tests: `tests/cli_{command}_tests.rs` (need to test CLI via assert_cmd or similar)

**New Sampling Method:**
- Implementation: `src/sampler/sampler.rs`
- Tests: `tests/sampling_{method}_tests.rs`

**Utilities:**
- Shared helpers: `src/` with appropriate module (e.g., `src/utils.rs` or domain-specific module)
- Test fixtures: `tests/common/mod.rs`

## Special Directories

**`src/.codemcp/`:**
- Purpose: Internal CodeMCP state
- Generated: Yes
- Committed: No

**`src/bin/`:**
- Purpose: Executable entry points (binaries declared in `Cargo.toml`)
- Generated: No
- Committed: Yes

**`tests/common/`:**
- Purpose: Shared test fixtures and utilities
- Generated: No
- Committed: Yes

**`models/`:**
- Purpose: Downloaded GGUF models for local testing
- Generated: Yes (by CLI or manual download)
- Committed: No (gitignored)

**`target/`:**
- Purpose: Cargo build artifacts
- Generated: Yes
- Committed: No

**`.planning/`:**
- Purpose: Development plans and codebase analysis
- Generated: No
- Committed: Yes

---
*Structure analysis: 2026-01-20*
