# Architecture

**Analysis Date:** 2026-01-18

## Pattern Overview

**Overall:** Monolithic Rust Application with GPU-centric modular design

**Key Characteristics:**
- Single binary with modular component architecture
- GPU-accelerated LLM inference on AMD hardware
- Async/await concurrency via Tokio
- Layered abstraction from API to hardware

## Layers

**API Layer:**
- Purpose: HTTP interface and request handling
- Contains: OpenAI-compatible REST API, SSE streaming
- Location: `src/http/`
- Depends on: Service layer (Engine, Scheduler)
- Used by: External HTTP clients

**Service Layer:**
- Purpose: Core business logic coordination
- Contains: Scheduler (`src/scheduler/`), Sampler (`src/sampler/`), Attention (`src/attention/`)
- Depends on: Core Engine, Data Layer
- Used by: API Layer

**Core Engine:**
- Purpose: Central inference orchestrator
- Contains: `src/engine.rs`
- Depends on: Model, Backend, KV Cache, Tokenizer
- Used by: All layers

**Data Layer:**
- Purpose: Model storage and GPU abstraction
- Contains: KV Cache (`src/kv_cache/`), Model Loader (`src/loader/`), Backend (`src/backend/`)
- Depends on: GPU runtime (HIP/ROCm)
- Used by: Core Engine

**Kernel Layer:**
- Purpose: Low-level GPU operations
- Contains: Tensor ops (`src/tensor/`), GPU kernels (`src/ggml/hip_backend/ops/`), Attention ops (`src/ops/attention_gpu.rs`)
- Depends on: HIP runtime
- Used by: Backend layer

## Data Flow

**Inference Request:**

1. HTTP request received at `src/http/server.rs`
2. Request queued in Scheduler (`src/scheduler/`)
3. Engine coordinates execution (`src/engine.rs`)
4. Model execution via GPU Backend (`src/backend/hip_backend.rs`)
5. Token sampling and KV cache update
6. Results streamed via SSE to client

**State Management:**
- Stateless request handling (each request independent)
- KV cache maintains session state across inference steps
- Model weights loaded once and reused

## Key Abstractions

**Config Objects:**
- Purpose: Type-safe configuration for all components
- Examples: `ModelConfig`, `EngineConfig`, `CacheConfig` (all in `src/model/config.rs`)
- Pattern: `pub struct {Name}Config { ... }`

**Trait-based Backends:**
- Purpose: Polymorphic hardware implementations
- Examples: `AttentionBackend`, `HipBackend`
- Pattern: `pub trait {Name} { ... }`

**Result Types:**
- Purpose: Consistent error handling
- Pattern: `Result<T, Error>` throughout

## Entry Points

**CLI Binary:**
- Location: `src/bin/rocmforge_cli.rs`
- Triggers: User runs `rocmforge_cli` command
- Responsibilities: Parse args, route to serve/generate/inspect commands

**Utility Binaries:**
- `src/bin/inspect_gguf.rs` - GGUF metadata inspection
- `src/bin/test_gguf_load.rs` - Model loading tests
- `src/bin/run_simple_model.rs` - Simple inference runner

## Error Handling

**Strategy:** Result types with structured errors

**Patterns:**
- `Result<T, Error>` return types
- `anyhow::Error` for application errors
- Custom error types with `thiserror`
- `unwrap()` used extensively in tests (not production)

## Cross-Cutting Concerns

**Logging:**
- Transitioning from `eprintln!` to `tracing` framework
- Structured logging via `tracing` and `tracing-subscriber`

**Validation:**
- Dimension validation in tensor operations (`src/tensor/matmul.rs`)
- Model configuration validation in `src/model/config.rs`

**GPU Management:**
- HIP stream management for async operations
- GPU memory allocation and cleanup

---

*Architecture analysis: 2026-01-18*
*Update when major patterns change*
