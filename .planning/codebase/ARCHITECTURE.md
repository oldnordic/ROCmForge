# Architecture

**Analysis Date:** 2026-01-20

## Pattern Overview

**Overall:** Layered inference engine with pluggable backends

**Key Characteristics:**
- GPU-first design with CPU fallback paths
- Lazy tensor loading for memory efficiency
- Paged KV cache for variable-length sequences
- Continuous batching scheduler for throughput
- IR-based execution (ggml-inspired) with backend abstraction

## Layers

**Engine Layer (`src/engine.rs`):**
- Purpose: Top-level orchestration of inference, request lifecycle management
- Location: `src/engine.rs`
- Contains: `InferenceEngine`, `EngineConfig`, `HealthStatus`, retry logic
- Depends on: Backend, Scheduler, KvCache, Sampler, ModelRuntime
- Used by: HTTP server, CLI

**Backend Layer (`src/backend/`):**
- Purpose: GPU abstraction and kernel execution
- Location: `src/backend/`
- Contains: HIP FFI bindings, DeviceTensor, HipBuffer, stream management
- Depends on: ROCm HIP runtime (amdhip64)
- Used by: All GPU operations, ModelRuntime

**Model Execution Layer (`src/model/`, `src/ggml/`):**
- Purpose: Model representation and execution planning
- Location: `src/model/execution_plan/`, `src/ggml/`
- Contains: `ExecutionPlan`, `LayerPlan`, `Graph`, `Node`, `Op` enum
- Depends on: Backend, Loader (for weights)
- Used by: Engine for forward passes

**Attention Layer (`src/attention/`):**
- Purpose: Attention computation with pluggable backends
- Location: `src/attention/`
- Contains: `Attention`, `AttentionBackend`, CPU/GPU implementations, FlashAttention
- Depends on: Backend (for GPU), tensor operations
- Used by: Model layers

**Loader Layer (`src/loader/`):**
- Purpose: Model file parsing and lazy weight loading
- Location: `src/loader/`
- Contains: `GgufLoader`, `MmapGguf`, `LazyTensor`, dequantization
- Depends on: memmap2, byteorder (for file parsing)
- Used by: ModelRuntime initialization

**Scheduler Layer (`src/scheduler/`):**
- Purpose: Continuous batching and request management
- Location: `src/scheduler/scheduler.rs`
- Contains: `Scheduler`, `GenerationRequest`, `RequestState`, `IterationBatch`
- Depends on: tokio (async primitives)
- Used by: Engine inference loop

**KV Cache Layer (`src/kv_cache/`):**
- Purpose: Paged KV cache with block sharing and eviction
- Location: `src/kv_cache/`
- Contains: `KvCache`, `PageTable`, `BlockAllocator`, `SequenceCache`
- Depends on: Backend (GPU memory)
- Used by: Engine for multi-token generation

**Sampler Layer (`src/sampler/`):**
- Purpose: Token sampling (top-k, top-p, temperature)
- Location: `src/sampler/sampler.rs`
- Contains: `Sampler`, `SamplingConfig`, `TokenScore`
- Depends on: rand (for sampling)
- Used by: Engine for token selection

**MLP Layer (`src/mlp/`):**
- Purpose: Feed-forward network operations (SwiGLU, RMSNorm)
- Location: `src/mlp/`
- Contains: GPU kernels for MLP operations
- Depends on: Backend
- Used by: Model execution

**HTTP Layer (`src/http/`):**
- Purpose: REST API for inference requests
- Location: `src/http/`
- Contains: Axum server, SSE streaming, health endpoints
- Depends on: Engine, tokio, axum
- Used by: External clients

## Data Flow

**Request Submission Flow:**

1. HTTP request received at `/generate` endpoint (`src/http/server.rs`)
2. Request tokenized via `TokenizerAdapter` (`src/tokenizer.rs`)
3. `InferenceEngine::submit_request()` queues request in `Scheduler`
4. Request ID returned to client for streaming

**Inference Loop Flow:**

1. `InferenceEngine::inference_loop()` runs as background task
2. `Scheduler::get_next_iteration_batch()` creates batch from pending/processing requests
3. For each request in batch:
   - `run_forward_pass()` executes model using `ModelRuntime::decode_step()`
   - `Sampler::sample()` selects next token from logits
   - `KvCache::append_token()` stores KV pairs for next iteration
4. Completed requests notify waiters via `Notify`

**Model Loading Flow:**

1. `InferenceEngine::from_gguf()` parses GGUF metadata
2. `ModelRuntime::load_from_gguf_with_loader()` creates execution plan
3. Weights stored as `Arc<LazyTensor>` (not loaded yet)
4. On first forward pass, weights loaded on-demand via `get_or_load_tensor()`

**State Management:**
- Per-engine: `Arc<HipBackend>`, `Arc<RwLock<ModelRuntime>>`, `Arc<RwLock<KvCache>>`
- Per-request: `RequestRuntimeState` (logical progress only, no GPU resources)
- Request lifecycle tracked in `Scheduler` with `RequestState` enum

## Key Abstractions

**DeviceTensor:**
- Purpose: GPU buffer with shape information
- Examples: `src/backend/hip_backend/backend.rs:DeviceTensor`
- Pattern: Smart pointer around `HipBuffer` with shape tracking
- Methods: `from_host_vec()`, `copy_to_host()`, `shape()`, `len()`

**LazyTensor:**
- Purpose: Deferred weight loading to reduce RAM usage
- Examples: `src/loader/lazy_tensor.rs:LazyTensor`
- Pattern: `Arc<LazyTensor>` with `OnceCell` for GPU cache
- Loads on first access via `get_or_load_tensor(&backend)`

**AttentionBackend (pluggable):**
- Purpose: Runtime selection of CPU vs GPU attention
- Examples: `src/attention/backend.rs:AttentionBackend`
- Pattern: Enum with variants for each backend implementation
- Registry pattern via `AttentionBackendRegistry`

**Ggml IR:**
- Purpose: Graph-based model representation
- Examples: `src/ggml/graph.rs:Graph`, `src/ggml/op.rs:Op`
- Pattern: Nodes = tensors, Edges = operations
- Operations include: MatMul, Add, Scale, Softmax, RoPE, Attention, SwiGlu

**ExecutionPlan:**
- Purpose: Static description of layer execution
- Examples: `src/model/execution_plan/execution_plan_src.rs:ExecutionPlan`
- Pattern: Stores `Arc<LazyTensor>` weights, provides `embedding_lookup()` and `layer_forward()`

## Entry Points

**HTTP Server:**
- Location: `src/http/server.rs:run_server()`
- Triggers: `rocmforge serve` CLI command, `InferenceEngine::from_gguf().await` then `run_server()`
- Responsibilities: Bind Axum routes, handle SSE streaming, health checks

**CLI (`rocmforge-cli`):**
- Location: `src/bin/rocmforge_cli.rs`
- Triggers: User command execution
- Responsibilities: Parse args, dispatch to server or local inference

**Library Entry:**
- Location: `src/lib.rs`
- Triggers: External crate usage
- Responsibilities: Re-exports public API, module organization

**Test Binaries:**
- `src/bin/inspect_gguf.rs`: GGUF file inspection utility
- `src/bin/test_gguf_load.rs`: Model loading test harness
- `test_inference.rs`: Standalone inference test

## Error Handling

**Strategy:** Result types with thiserror for domain-specific errors

**Patterns:**
- `RocmForgeError` (unified): Centralized error type in `src/error.rs`
- Module-level errors: `AttentionError`, `SchedulerError`, `SamplerError`, `KvCacheError`
- HIP errors: `HipError` with HIP error code translation
- Category-based error handling via `ErrorCategory` enum

**Error Propagation:**
- Backend errors return `HipResult<T> = Result<T, HipError>`
- Model operations return `ForgeResult<T> = Result<T, RocmForgeError>`
- Engine operations return `EngineResult<T> = Result<T, EngineError>`

## Cross-Cutting Concerns

**Logging:** tracing-based with structured output (JSON/human), configured via `src/logging/mod.rs`
**Validation:** Shape validation in tensor operations, token range checking in scheduler
**Authentication:** Not implemented (development/testing only)
**Metrics:** Prometheus metrics via `src/metrics.rs`, OpenTelemetry traces via `src/otel_traces.rs`
**Profiling:** Kernel timing, TTFT breakdown, ROCm tool integration in `src/profiling/`

---

*Architecture analysis: 2026-01-20*
