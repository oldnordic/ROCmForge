# ROCmForge

## What This Is

A production-ready LLM inference engine for AMD GPUs, written in Rust. ROCmForge loads GGUF-format models and runs them on ROCm/HIP with optimized CPU fallback using SIMD. It provides an OpenAI-compatible HTTP API for seamless integration with existing LLM applications.

## Core Value

**Reliable, fast inference on AMD GPUs with transparent CPU fallback.**

If ROCm is available, use it. If not, fall back to optimized CPU execution seamlessly. Any GGUF model should just work.

## Requirements

### Validated

*Capabilities shipped through v1.1:*

**v1.0 Capabilities:**
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

**v1.1 Capabilities (Bug Fixes):**
- ✓ **Qwen2/Qwen2.5 support** — Fixed `head_dim = 0` bug using `hidden_size / num_heads` calculation
- ✓ **Documentation accuracy** — Corrected "selective pooling COMPLETE" to "NOT IMPLEMENTED"
- ✓ **Code quality** — Reduced compiler warnings from 406 to 27 (93% reduction)

**v1.0 Requirements (All Complete):**
- ✓ Fix inference hangs (GPU stream synchronization bug) — v1.0
- ✓ Complete quantized matmul with native HIP dequantization kernel — v1.0
- ✓ Implement flash attention detection and GPU kernels — v1.0
- ✓ Add CPU SIMD backend for all tensor operations — v1.0
- ✓ Hybrid execution scheduler (automatic CPU/GPU op selection) — v1.0
- ✓ Universal GGUF compatibility (all architectures, quantizations) — v1.0
- ✓ Performance optimization (balanced: throughput, latency, memory) — v1.0
- ✓ Production-ready reliability and error handling — v1.0

**v1.1 Requirements (All Complete):**
- ✓ **QWEN-01 through QWEN-06**: Qwen2 head_dim fix — Calculate `head_dim = hidden_size / num_heads` before GGUF parsing
- ✓ **ROCM-01 through ROCM-04**: Memory allocation validation — Confirmed direct allocation approach avoids D2H bug
- ✓ **CLEAN-01 through CLEAN-03**: Dead code removal — Reduced warnings 93%, reviewed all dead_code markers

### Active

*What we're building toward next:*

**Current Milestone: v1.6 FFI Device Props Fix**

**Goal:** Fix FFI device properties bug causing "block.y exceeds limit 0" errors during kernel launch.

**Target features:**
- **Sanity check strengthening**: Validate ALL 3 dimensions (X, Y, Z), not just [0]
- **Delete duplicate code**: Remove duplicate DeviceLimits assignment that undermines fixes
- **Bindgen infrastructure**: Add compile-time offset verification to catch ROCm version changes
- **Offset verification test**: Assert manual offsets match bindgen-generated values

### Out of Scope

- **Training features** (LoRA adapters, fine-tuning, training modes) — Focus is inference-only
- **Non-text modalities** (vision, audio, multimodal models) — Text-only for v1
- **Multi-GPU/distributed execution** — Single GPU focus for v1
- **Non-AMD GPU support** — ROCm/HIP only (CPU fallback covers non-GPU systems)

## Current State

**Shipped v1.1 (2026-01-19):**
- 572 tests passing
- ~65,000 lines of Rust code
- 16 phases complete (101 plans)
- All v1.0 + v1.1 requirements satisfied
- Code quality: 27 lib warnings (93% reduction from baseline)

**Shipped v1.0 (2026-01-19):**
- 620 tests passing (unit + integration + E2E)
- ~64,900 lines of Rust code
- 13 phases complete (96 plans)
- Complete documentation: user guide, CLI reference, API docs, deployment guide

**Technical Environment:**
- Rust 2021 edition with Tokio async runtime
- ROCm/HIP for AMD GPU (gfx1100 / RX 7900 XT + broader support)
- Axum web framework for HTTP API
- GGUF format for model weights (llama.cpp compatibility)

**Known Limitations (Acceptable for v1.1):**
- GPU sampler kernels: CPU fallback (optimization deferred)
- MQA GPU optimization: Partial CPU fallback (works but not optimal)
- SIMD feature: Requires nightly Rust (documented limitation)
- AVX-512: Opt-in feature flag to avoid CPU throttling
- 27 compiler warnings (FFI dead_code markers, deprecated to_host_vec, Q*_K naming)

**Technical Debt:**
- Duplicate `GgufMetadata` structs exist (one in metadata.rs, one in gguf.rs) — both have `calculate_default_head_dim()` method
- 10 tests marked #[ignore] using deprecated `ExecutionPlan::new()` — need rewriting with actual GGUF files
- `DeviceTensor::to_host_vec()` deprecated but 100+ usages remain

## Context

<details>
<summary>v1.1 Milestone Summary (Archived)</summary>

**Phases:** 13-01, 13-02, 13-03 (6 plans total)
**Timeline:** 1 day (2026-01-19)
**Commits:** 31 commits (651c25c to f0e1cdb)

**Phase 13-01: Qwen2 head_dim Fix**
- Implemented `calculate_default_head_dim()` following llama.cpp pattern
- Fixed `rope.dimension_count` parsing with safe if-let instead of `unwrap_or(0)`
- All Qwen2/Qwen2.5 models now load without `buffer_size=0` errors

**Phase 13-02: Memory Pooling Documentation**
- Discovered selective pooling was never implemented (no LARGE_TENSOR_THRESHOLD, from_pool never called)
- Corrected misleading "Status: COMPLETE" documentation
- Created MEMORY_ARCHITECTURE.md documenting actual direct-allocation approach

**Phase 13-03: Dead Code Removal**
- Removed 4 unused functions, fixed 4 incorrect dead_code markers
- Replaced 30+ deprecated method calls
- Removed 93 unused imports, suppressed GGUF naming warnings
- Reduced compiler warnings from 406 to 27 (93% reduction)

**Full details:** `.planning/milestones/v1.1-ROADMAP.md`
</details>

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
| Rust implementation | Performance, safety, GPU FFI control | ✅ Successful - 65k LOC, production-ready |
| GGUF format only | llama.cpp ecosystem compatibility | ✅ 15 formats supported |
| Hybrid CPU/GPU execution | Maximum compatibility, graceful degradation | ✅ Working - CapabilityProvider trait |
| OpenAI-compatible API | Drop-in replacement for existing apps | ✅ Implemented - SSE streaming |
| Modular architecture with trait backends | Easy CPU/GPU switching, testability | ✅ Achieved - BackendImplementation trait |
| AVX-512 opt-in feature flag | Avoid CPU throttling on older hardware | ✅ Implemented |
| SQLiteGraph as optional dependency | Keep core lean while enabling context features | ✅ Feature-gated |
| Qwen2 head_dim default calculation | Prevents zero-init when GGUF key missing | ✅ v1.1 - llama.cpp pattern |
| Direct allocation for D2H tensors | Simpler than pooling, avoids ROCm 7.1 bug | ✅ v1.1 - verified approach |
| Honest documentation > false claims | Prevents wasted time searching for non-existent code | ✅ v1.1 - corrected docs |

---

*Last updated: 2026-01-21 after v1.6 milestone initialization*
