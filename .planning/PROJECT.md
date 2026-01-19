# ROCmForge

## What This Is

A production-ready LLM inference engine for AMD GPUs, written in Rust. ROCmForge loads GGUF-format models and runs them on ROCm/HIP with optimized CPU fallback using SIMD. It provides an OpenAI-compatible HTTP API for seamless integration with existing LLM applications.

## Core Value

**Reliable, fast inference on AMD GPUs with transparent CPU fallback.**

If ROCm is available, use it. If not, fall back to optimized CPU execution seamlessly. Any GGUF model should just work.

## Requirements

### Validated

*Capabilities shipped in v1.0:*

- ✓ GGUF model loading — 15 quantization formats supported
- ✓ HuggingFace tokenizer integration — `src/tokenizer.rs`
- ✓ OpenAI-compatible HTTP API with SSE streaming — `src/http/server.rs`
- ✓ Token sampling (top-k, top-p, temperature) — `src/sampler/`
- ✓ Multi-head attention (CPU + GPU backends) — `src/attention/`
- ✓ Flash attention GPU implementation — `src/attention/flash_attention.rs`
- ✓ Paged KV cache — `src/kv_cache/`
- ✓ CPU SIMD backend — AVX-512/AVX2/NEON with runtime detection (56 operations)
- ✓ Native HIP quantization kernels — Fused dequant+matmul, Q4_0/Q8_0/Q4_K/Q6_K
- ✓ Hybrid CPU/GPU scheduler — Automatic backend selection via CapabilityProvider
- ✓ Tensor operations — Matmul, RMSNorm, RoPE, activations (SiLU, SwiGLU, GELU)
- ✓ GPU kernels: matmul, softmax, rope, swiglu — `src/ggml/hip_backend/ops/`
- ✓ CLI tools: serve, generate, context — `src/bin/rocmforge_cli.rs`
- ✓ Model configs: LLaMA, Qwen, Mistral, Yi, Mixtral — `src/model/config.rs`
- ✓ Error handling — Proper Result types throughout, no unwrap() in prod paths
- ✓ Observability — Tracing, metrics (/metrics), health endpoints (/health, /ready)
- ✓ SQLiteGraph context engine — Feature-gated semantic context storage

**v1.0 Requirements (All Complete):**
- ✓ Fix inference hangs (GPU stream synchronization bug) — v1.0
- ✓ Complete quantized matmul with native HIP dequantization kernel — v1.0
- ✓ Implement flash attention detection and GPU kernels — v1.0
- ✓ Add CPU SIMD backend for all tensor operations — v1.0
- ✓ Hybrid execution scheduler (automatic CPU/GPU op selection) — v1.0
- ✓ Universal GGUF compatibility (all architectures, quantizations) — v1.0
- ✓ Performance optimization (balanced: throughput, latency, memory) — v1.0
- ✓ Production-ready reliability and error handling — v1.0

### Active

*What we're building toward next:*

(None - all v1.0 requirements satisfied. Run `/gsd:new-milestone` to define v1.1 goals.)

### Out of Scope

- **Training features** (LoRA adapters, fine-tuning, training modes) — Focus is inference-only
- **Non-text modalities** (vision, audio, multimodal models) — Text-only for v1
- **Multi-GPU/distributed execution** — Single GPU focus for v1
- **Non-AMD GPU support** — ROCm/HIP only (CPU fallback covers non-GPU systems)

## Current State

**Shipped v1.0 (2026-01-19):**
- 620 tests passing (unit + integration + E2E)
- ~64,900 lines of Rust code
- 13 phases complete (96 plans)
- 8/8 requirements satisfied
- Complete documentation: user guide, CLI reference, API docs, deployment guide
- Production-ready: error handling, tracing, metrics, graceful degradation

**Technical Environment:**
- Rust 2021 edition with Tokio async runtime
- ROCm/HIP for AMD GPU (gfx1100 / RX 7900 XT + broader support)
- Axum web framework for HTTP API
- GGUF format for model weights (llama.cpp compatibility)

**Known Limitations (Acceptable for v1.0):**
- GPU sampler kernels: CPU fallback (optimization deferred)
- MQA GPU optimization: Partial CPU fallback (works but not optimal)
- SIMD feature: Requires nightly Rust (documented limitation)
- AVX-512: Opt-in feature flag to avoid CPU throttling
- ~82 compiler warnings (cosmetic: unused imports, variables)

## Context

<details>
<summary>Original Project State (Pre-v1.0)</summary>

**Existing Codebase State (Initial):**
- Monolithic Rust application with layered architecture (API → Service → Engine → Data → Kernel)
- 96 test files with good coverage, but 20+ tests commented out pending API rewrite
- 3 files exceed 3000 LOC (execution_plan.rs, hip_backend.rs, gguf.rs) — modularization needed
- Known GPU synchronization bugs causing inference hangs (hipBLAS vs hipMemcpy stream mismatch)
- Transition in progress: eprintln! → tracing for logging

**Known Issues (All Resolved in v1.0):**
- ✅ GPU stream synchronization bug — FIXED in Phase 01-01
- ✅ Race condition in inference loop — FIXED in Phase 01-02
- ✅ Engine cleanup issues in CLI — FIXED in Phase 01-03
- ✅ Missing .env.example — FIXED in Phase 10-12
- ✅ Test compilation errors — FIXED in Phase 11
</details>

## Constraints

- **Platform**: ROCm on Linux only — AMD GPU driver limitation
- **Model Format**: GGUF only — Leverage llama.cpp ecosystem
- **Hardware**: AMD GPU or CPU SIMD — No NVIDIA, no other accelerators
- **Architecture**: Single binary, self-contained — No external runtime dependencies beyond ROCm

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rust implementation | Performance, safety, GPU FFI control | ✅ Successful - 64.9k LOC, production-ready |
| GGUF format only | llama.cpp ecosystem compatibility | ✅ 15 formats supported |
| Hybrid CPU/GPU execution | Maximum compatibility, graceful degradation | ✅ Working - CapabilityProvider trait |
| OpenAI-compatible API | Drop-in replacement for existing apps | ✅ Implemented - SSE streaming |
| Modular architecture with trait backends | Easy CPU/GPU switching, testability | ✅ Achieved - BackendImplementation trait |
| AVX-512 opt-in feature flag | Avoid CPU throttling on older hardware | ✅ Implemented |
| SQLiteGraph as optional dependency | Keep core lean while enabling context features | ✅ Feature-gated |

---

*Last updated: 2026-01-19 after v1.0 milestone*
