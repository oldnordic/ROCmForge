# Architecture

**Analysis Date:** 2026-01-20

## Pattern Overview

**Overall:** Layered inference engine with pluggable backends

**Key Characteristics:**
- Async request processing with continuous batching
- Paged KV cache for memory efficiency
- Pluggable CPU/GPU backends via trait abstraction
- GGUF model format for quantized weights
- HTTP API for inference serving

## Layers

**Presentation Layer (HTTP API):**
- Purpose: Expose OpenAI-compatible HTTP endpoints
- Location: `src/http/`
- Contains: Route handlers, request/response models
- Depends on: Engine layer
- Used by: External clients

**Engine Layer:**
- Purpose: Orchestrate inference workflow, manage lifecycle
- Location: `src/engine.rs`
- Contains: `InferenceEngine`, request routing, retry logic
- Depends on: Scheduler, KV cache, Model runtime, Sampler, Backend
- Used by: HTTP layer, CLI

**Scheduler Layer:**
- Purpose: Continuous batching for efficient GPU utilization
- Location: `src/scheduler/scheduler.rs`
- Contains: `Scheduler`, `GenerationRequest`, `IterationBatch`, `Batch`
- Depends on: None (core state machine)
- Used by: Engine

**Model Runtime Layer:**
- Purpose: Load models and execute transformer layers
- Location: `src/model/`, `src/backend/hip_backend/backend.rs` (ModelRuntime)
- Contains: `SimpleTransformer`, `ExecutionPlan`, `ModelRuntime`
- Depends on: Backend, Attention, MLP, KV cache, Loader
- Used by: Engine

**KV Cache Layer:**
- Purpose: Paged key-value storage for attention
- Location: `src/kv_cache/`
- Contains: `KvCache`, `PageTable`, `BlockAllocator`
- Depends on: Backend (for GPU memory)
- Used by: Model runtime, Scheduler

**Attention Layer:**
- Purpose: Multi-head attention computation
- Location: `src/attention/`
- Contains: `Attention`, `FlashAttention`, `PagedAttention`, RoPE
- Depends on: Backend (GPU kernels), Tensor operations
- Used by: Model runtime

**MLP Layer:**
- Purpose: Feed-forward network computation
- Location: `src/mlp/`
- Contains: SwiGLU, RMSNorm GPU kernels
- Depends on: Backend (GPU kernels)
- Used by: Model runtime

**Sampler Layer:**
- Purpose: Token sampling from logits
- Location: `src/sampler/sampler.rs`
- Contains: `Sampler` (temperature, top-k, top-p, repetition penalty)
- Depends on: None (pure computation)
- Used by: Engine

**Loader Layer:**
- Purpose: Parse and load GGUF models
- Location: `src/loader/`
- Contains: `GgufLoader`, `OnnxLoader`, dequantization, MXFP support
- Depends on: Mmap, Tensor types
- Used by: Model runtime

**Backend Layer:**
- Purpose: Abstract CPU/GPU computation
- Location: `src/backend/`
- Contains: `HipBackend` (GPU), CPU backend, `DeviceTensor`, `HipBuffer`
- Depends on: ROCm/HIP FFI
- Used by: All computation layers

**GGML IR Layer:**
- Purpose: Graph-based computation abstraction
- Location: `src/ggml/`
- Contains: `Graph`, `Node`, `Op`, `GgmlBackend` trait
- Depends on: Backend layer
- Used by: Model runtime (alternative execution path)

## Data Flow

**Inference Request Flow:**

1. HTTP request arrives at `/v1/completions`
2. `InferenceEngine::submit_request()` queues to `Scheduler`
3. `Scheduler::get_next_iteration_batch()` forms continuous batch
4. For each request in batch:
   - `ModelRuntime::decode_step()` runs one transformer layer
   - `ExecutionPlan` orchestrates: embedding -> attention -> MLP -> lm_head
   - `KvCache` stores KV pairs paged by sequence
   - `Sampler::sample_with_history()` selects next token
5. `Scheduler::update_iteration_batch()` updates state, removes completed
6. Response streamed via Server-Sent Events

**Model Loading Flow:**

1. `InferenceEngine::from_gguf()` parses GGUF metadata
2. `GgufLoader` reads tensor names, detects architecture (Qwen2/LLaMA/Mistral)
3. `ExecutionPlan` maps tensors to layers (attention weights, MLP weights)
4. Weights loaded lazily via `LazyTensor` (mmap + dequant on demand)
5. `ModelRuntime` initialized with backend, scratch buffers, KV cache

**State Management:**
- Per-engine: `Arc<HipBackend>`, `Arc<RwLock<KvCache>>`, `Arc<RwLock<ModelRuntime>>`
- Per-request: `RequestRuntimeState` (token position only, no GPU resources)
- Shared: `Arc<RwLock<Scheduler>>`, `Arc<RwLock<Sampler>>`

## Key Abstractions

**Backend Trait (`GgmlBackend`):**
- Purpose: Pluggable execution backends
- Examples: `src/ggml/hip_backend/`, `src/ggml/cpu_backend.rs`
- Pattern: Trait object with async kernel execution

**Attention Backend (`AttentionBackend`):**
- Purpose: CPU vs GPU selection for attention
- Examples: `src/attention/backend.rs`, `src/attention/cpu.rs`, `src/attention/gpu.rs`
- Pattern: Enum-based dispatch to CPU or GPU kernels

**Device Abstraction (`DeviceTensor`):**
- Purpose: GPU memory with safe copy operations
- Examples: `src/backend/hip_backend/backend.rs`
- Pattern: RAII wrapper around HIP buffer with stream-aware copies

**Model Architecture Detection:**
- Purpose: Support multiple transformer architectures
- Examples: `src/model/execution_plan/architecture.rs` (Qwen2, LLaMA, Mistral, Yi, Mixtral)
- Pattern: Tensor name prefix detection (`blk.N.`, `transformer.layers.N.`, `model.layers.N.`)

## Entry Points

**Binary Entry Points:**
- Location: `src/bin/`
- Triggers: Command-line execution
- Responsibilities:
  - `rocmforge_cli.rs`: CLI for model management, inference, server
  - `inspect_gguf.rs`: Debug tool for GGUF inspection
  - `run_simple_model.rs`: Simple model execution
  - `test_gguf_load.rs`: GGUF loader test utility

**HTTP Server Entry:**
- Location: `src/http/mod.rs`
- Triggers: HTTP request to `:8080`
- Responsibilities: OpenAI-compatible API, SSE streaming

**Library Entry Point:**
- Location: `src/lib.rs`
- Triggers: `use rocmforge::*`
- Responsibilities: Public API exports

**Engine Entry Point:**
- Location: `src/engine.rs::InferenceEngine`
- Triggers: `InferenceEngine::new()` or `InferenceEngine::from_gguf()`
- Responsibilities: Initialize backend, cache, scheduler, model runtime

## Error Handling

**Strategy:** Centralized error type with categorization

**Patterns:**
- `RocmForgeError` (unified enum) in `src/error.rs`
- Categories: User, Recoverable, Internal, Backend, Model
- Result types: `ForgeResult<T>`, `HipResult<T>`, `SchedulerResult<T>`, etc.
- Conversion: `From<E> for RocmForgeError` for interoperability
- Macros: `user_error!()`, `internal_error!()`, `backend_error!()`, `model_error!()`

## Cross-Cutting Concerns

**Logging:** `tracing` framework with `tracing-subscriber`
- `src/logging/mod.rs`: Initialization helpers
- Levels: trace, debug, info, warn, error
- Format: JSON (structured) or text

**Metrics:** Prometheus client
- `src/metrics.rs`: Metric registry
- Exposed at `/metrics` HTTP endpoint

**Tracing:** OpenTelemetry span export
- `src/otel_traces.rs`: Span recording for inference operations

**Profiling:** ROCprof integration
- `src/profiling/`: Kernel timing, baseline capture

**Prompt Optimization:**
- `src/prompt/`: Chunking, prefix caching, batch attention

**Testing:** `#[serial]` for GPU tests, `serial_test` crate
- GPU isolation to prevent device reset between tests

---
*Architecture analysis: 2026-01-20*
