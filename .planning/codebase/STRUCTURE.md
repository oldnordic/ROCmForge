# Codebase Structure

**Analysis Date:** 2026-01-20

## Directory Layout

```
ROCmForge/
├── src/                    # Main library source code
│   ├── attention/          # Attention mechanism (CPU/GPU kernels)
│   ├── backend/            # GPU/HIP backend abstraction
│   │   ├── hip_backend/    # HIP implementation details
│   │   └── cpu/            # CPU SIMD implementation
│   ├── bin/                # CLI binaries
│   ├── ggml/               # IR-based execution graph
│   │   └── hip_backend/    # GGML HIP backend (ops implementation)
│   ├── http/               # REST API server
│   ├── kv_cache/           # Paged KV cache
│   ├── loader/             # GGUF/ONNX model loading
│   ├── mlp/                # MLP operations (SwiGLU, RMSNorm)
│   ├── model/              # Model implementations
│   │   └── execution_plan/ # Static execution plans
│   ├── ops/                # GPU computation ops
│   ├── profiling/          # Performance profiling
│   ├── prompt/             # Prompt processing optimization
│   ├── sampler/            # Token sampling
│   ├── scheduler/          # Continuous batching
│   ├── tensor/             # Basic tensor operations
│   ├── context/            # Context engine (feature-gated)
│   └── logging/            # Logging configuration
├── tests/                  # Integration tests
│   ├── common/             # Test fixtures
│   └── attention/          # Attention test data
├── kernels/                # HIP kernel sources (.hip files)
├── benches/                # Criterion benchmarks
├── build/                  # Build artifacts (HIP compilation)
├── docs/                   # Documentation
├── models/                 # Downloaded model files
├── .planning/              # Planning artifacts
└── Cargo.toml              # Package manifest
```

## Directory Purposes

**`src/attention/`:**
- Purpose: Attention computation with pluggable CPU/GPU backends
- Contains: Attention implementations, RoPE, causal masking, FlashAttention, paged attention
- Key files: `src/attention/mod.rs`, `src/attention/gpu.rs`, `src/attention/flash_attention.rs`, `src/attention/paged_kernel.rs`

**`src/backend/`:**
- Purpose: GPU abstraction layer providing HIP FFI and device memory management
- Contains: `HipBackend`, `DeviceTensor`, `HipBuffer`, FFI bindings, stream management
- Key files: `src/backend/mod.rs`, `src/backend/hip_backend/backend.rs`, `src/backend/gpu_executor.rs`

**`src/backend/cpu/`:**
- Purpose: CPU fallback implementation with SIMD support
- Contains: CPU feature detection, SIMD matmul operations
- Key files: `src/backend/cpu/mod.rs`, `src/backend/cpu/simd_ops.rs`

**`src/bin/`:**
- Purpose: Executable binaries for CLI and utilities
- Contains: `rocmforge_cli` (main CLI), GGUF inspector, test harnesses
- Key files: `src/bin/rocmforge_cli.rs`, `src/bin/inspect_gguf.rs`, `src/bin/test_gguf_load.rs`

**`src/ggml/`:**
- Purpose: IR-based execution graph for model operations
- Contains: Graph representation, operations, optimizer, executor, hybrid scheduler
- Key files: `src/ggml/mod.rs`, `src/ggml/graph.rs`, `src/ggml/executor.rs`, `src/ggml/hybrid_scheduler.rs`

**`src/ggml/hip_backend/`:**
- Purpose: HIP backend implementation for GGML IR operations
- Contains: Op implementations (matmul, softmax, RoPE, attention, quantized ops)
- Key files: `src/ggml/hip_backend/mod.rs`, `src/ggml/hip_backend/ops/matmul.rs`, `src/ggml/hip_backend/ops/softmax.rs`

**`src/http/`:**
- Purpose: HTTP server for inference API
- Contains: Axum server, SSE streaming, request handlers
- Key files: `src/http/server.rs`, `src/http/mod.rs`

**`src/kv_cache/`:**
- Purpose: Paged KV cache with block sharing and LRU eviction
- Contains: `KvCache`, `PageTable`, `BlockAllocator`, sequence management
- Key files: `src/kv_cache/mod.rs`, `src/kv_cache/kv_cache.rs`, `src/kv_cache/page_table.rs`

**`src/loader/`:**
- Purpose: Model file parsing (GGUF/ONNX) and lazy weight loading
- Contains: `GgufLoader`, `MmapGguf`, `LazyTensor`, dequantization
- Key files: `src/loader/mod.rs`, `src/loader/gguf.rs`, `src/loader/lazy_tensor.rs`, `src/loader/mmap_loader.rs`

**`src/model/`:**
- Purpose: Model implementations and configuration
- Contains: `SimpleTransformer`, model config detection, position embeddings
- Key files: `src/model/mod.rs`, `src/model/config.rs`, `src/model/simple_transformer.rs`

**`src/model/execution_plan/`:**
- Purpose: Static execution plans for transformer layers
- Contains: `ExecutionPlan`, `LayerPlan`, `Architecture` detection
- Key files: `src/model/execution_plan/mod.rs`, `src/model/execution_plan/execution_plan_src.rs`

**`src/mlp/`:**
- Purpose: MLP layer operations (SwiGLU activation, RMSNorm)
- Contains: GPU kernels for MLP operations
- Key files: `src/mlp/mod.rs`, `src/mlp/kernels.rs`

**`src/ops/`:**
- Purpose: GPU computation operations
- Contains: Attention kernels, QKV operations
- Key files: `src/ops/mod.rs`, `src/ops/attention_gpu.rs`, `src/ops/qkv.rs`

**`src/profiling/`:**
- Purpose: Performance profiling and timing utilities
- Contains: Kernel timers, TTFT profiling, ROCm tool integration
- Key files: `src/profiling/mod.rs`, `src/profiling/kernel_timer.rs`, `src/profiling/ttft.rs`

**`src/prompt/`:**
- Purpose: Prompt processing optimizations
- Contains: Chunking, prefix caching, batch attention
- Key files: `src/prompt/mod.rs`, `src/prompt/chunking.rs`, `src/prompt/cache.rs`

**`src/sampler/`:**
- Purpose: Token sampling (top-k, top-p, temperature)
- Contains: `Sampler`, `SamplingConfig`
- Key files: `src/sampler/mod.rs`, `src/sampler/sampler.rs`

**`src/scheduler/`:**
- Purpose: Continuous batching scheduler
- Contains: `Scheduler`, `GenerationRequest`, `IterationBatch`
- Key files: `src/scheduler/mod.rs`, `src/scheduler/scheduler.rs`

**`src/tensor/`:**
- Purpose: Basic tensor operations (matmul)
- Contains: CPU matmul, tensor structure
- Key files: `src/tensor/mod.rs`, `src/tensor/matmul.rs`

**`tests/`:**
- Purpose: Integration tests (not unit tests in src/)
- Contains: End-to-end tests, model loading tests, GPU tests
- Key files: `tests/e2e_suite.rs`, `tests/attention_gpu_tests.rs`, `tests/glm_model_tests.rs`

**`kernels/`:**
- Purpose: HIP kernel source files
- Contains: `.hip` and `.hsaco` (compiled) kernel files
- Key files: `kernels/flash_attention.hip`, `kernels/q4_k_matmul.hip`, `kernels/topk_topp_sampling.hip`

**`benches/`:**
- Purpose: Criterion benchmarks
- Contains: Performance benchmarks for attention, matmul, inference
- Key files: `benches/attention_bench.rs`, `benches/matmul_bench.rs`, `benches/inference_bench.rs`

## Key File Locations

**Entry Points:**
- `src/lib.rs`: Library entry, module declarations, public API exports
- `src/engine.rs`: `InferenceEngine` - main orchestration
- `src/bin/rocmforge_cli.rs`: CLI entry point
- `src/http/server.rs`: HTTP server entry
- `test_inference.rs`: Standalone inference test

**Configuration:**
- `Cargo.toml`: Dependencies, features, binary definitions
- `build.rs`: Build script for HIP kernel compilation
- `.env.example`: Environment variable template

**Core Logic:**
- `src/backend/hip_backend/backend.rs`: HIP backend implementation
- `src/model/execution_plan/execution_plan_src.rs`: Execution plan
- `src/loader/gguf.rs`: GGUF model loading
- `src/scheduler/scheduler.rs`: Request scheduling

**Testing:**
- `src/hip_backend_debug_tests.rs`: Backend debugging tests
- `tests/`: Integration tests
- Unit tests co-located in source files (e.g., `src/engine.rs` has `#[cfg(test)]` module)

## Naming Conventions

**Files:**
- Modules: `mod_name.rs` (snake_case)
- Tests: `{module}_tests.rs` or integrated in `mod.rs` with `#[cfg(test)]`
- HIP kernels: `{operation}.hip` (e.g., `flash_attention.hip`)
- Benchmarks: `{name}_bench.rs` in `benches/`

**Directories:**
- Feature modules: `src/feature_name/` (e.g., `src/attention/`, `src/scheduler/`)
- Backend variants: `src/backend/{variant}/` (e.g., `src/backend/hip_backend/`)
- Test data: `tests/{feature}/` (e.g., `tests/attention/`)

**Types:**
- Structs/Enums: `PascalCase` (e.g., `InferenceEngine`, `AttentionBackend`)
- Error types: `{Module}Error` (e.g., `HipError`, `SchedulerError`)
- Result types: `{Module}Result<T>` (e.g., `HipResult<T>`, `EngineResult<T>`)

**Functions:**
- Public: `snake_case` (e.g., `submit_request`, `run_forward_pass`)
- Async: `async fn snake_case` (e.g., `async fn from_gguf`)
- FFI: `hipFunctionName` (matches HIP API)

**Constants:**
- Upper snake case: `MAX_BATCH_SIZE`, `HIP_SUCCESS`

## Where to Add New Code

**New Feature:**
- Primary code: `src/{feature_name}/`
- Tests: `tests/{feature_name}_tests.rs` or co-located in `mod.rs`
- GPU kernels: `kernels/{feature_name}.hip`

**New Model Architecture:**
- Implementation: `src/model/{architecture}.rs`
- Config detection: `src/model/config.rs` (add to `detect_architecture()`)
- Execution plan: `src/model/execution_plan/` (may need new `Architecture` variant)

**New Attention Variant:**
- Implementation: `src/attention/{variant}.rs`
- GPU kernel: `kernels/{variant}.hip`
- Backend registry: `src/attention/backend_registry.rs` (register variant)

**New GGUF Operation:**
- Op definition: `src/ggml/op.rs` (add to `Op` enum)
- CPU impl: `src/ggml/cpu_backend.rs`
- GPU impl: `src/ggml/hip_backend/mod.rs` (add to `execute_op()` match)

**New HTTP Endpoint:**
- Handler: `src/http/server.rs` (add route)
- Request/Response types: `src/http/mod.rs` or inline in server

**Utilities:**
- Shared helpers: `src/{util_name}.rs` (top-level if generic)
- Backend helpers: `src/backend/{util_name}.rs`

## Special Directories

**`src/.codemcp/`:**
- Purpose: CodeMCP operation backups
- Generated: Yes
- Committed: Yes (gitignored backups in `.codemcp/backups/`)

**`build/`:**
- Purpose: Build-time HIP kernel compilation outputs
- Generated: Yes
- Committed: No (in .gitignore)

**`kernels/`:**
- Purpose: HIP kernel source files (text) and compiled binaries (.hsaco)
- Generated: Partially (`.hsaco` files are compiled from `.hip`)
- Committed: Yes (both source and compiled kernels)

**`models/`:**
- Purpose: Downloaded model files (GGUF, tokenizers)
- Generated: Yes
- Committed: No (large files, gitignored)

**`.planning/`:**
- Purpose: Development planning documents
- Generated: Yes (by GSD commands)
- Committed: Yes (version controlled planning artifacts)

**`docs/`:**
- Purpose: Project documentation (manual, API docs, etc.)
- Generated: No (manual edits)
- Committed: Yes

---

*Structure analysis: 2026-01-20*
