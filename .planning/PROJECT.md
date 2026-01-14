# ROCmForge

## What This Is

AMD GPU inference engine for Large Language Models using ROCm and HIP. Provides LLM inference capabilities on AMD hardware with GPU-accelerated kernels and ggml-style IR for efficient execution.

## Core Value

**Performance parity with llama.cpp on AMD GPUs.** The ggml IR must execute efficiently on AMD hardware (RX 7900 XT / gfx1100 target), with token generation speed matching reference implementations while maintaining clean architecture.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ GPU Kernels (Phases 1-4) — scale, mask, softmax, RoPE, FlashAttention, SwiGLU, RMSNorm
- ✓ GGUF Loader — F32, F16, Q8_0, Q4_0, MXFP4/MXFP6 support
- ✓ MXFP Quantization (Phase 5) — OCP MX Spec v1.0
- ✓ KV Cache — Paged attention cache
- ✓ HTTP Server — OpenAI-compatible API
- ✓ Async GPU Loading (Phase 17) — Multi-stream concurrent uploads

### Active

<!-- Current scope. Building toward these. -->

- [ ] Fix graph rebuilding every token — Use fixed-shape tensors with offset views
- [ ] Add quantized matmul ops — Q4_0, Q8_0 dequantization in ggml IR
- [ ] Single-pass GGUF loading — Parse once, reuse for config and weights
- [ ] Bind weights once — Separate static weight graphs from dynamic decode graphs
- [ ] Complete missing ggml ops — Accumulate, quantized matmul variants
- [ ] End-to-end integration tests — Real model validation

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Multi-GPU support — Focus on single-GPU efficiency first
- HTTP server optimization — Existing server functional, defer tuning
- New model architectures — Focus on existing LLaMA/Qwen2/GLM support
- Production deployment — Alpha software, testing focus only

## Context

**Existing Codebase State:**
- Alpha software, Phase 26 (GQA Scaffolding)
- ~50 compiler warnings remaining
- 6 test files have compilation errors
- End-to-end inference not fully tested

**Recent Analysis (2026-01-14):**
See: `docs/CLI_AND_MODEL_LOADING_ANALYSIS.md`

Identified critical issues:
1. Triple GGUF parsing — wasteful startup overhead
2. Graph shape mutation every decode token — O(tokens) rebuilds instead of O(1) updates
3. Weights bound per-decode-step — unnecessary overhead
4. Missing quantization ops — Q4_0/Q8_0 matmul kernels
5. Inefficient KV cache access — View reshaping vs offset-based

**ggml IR Status:**
- Core IR complete (Graph, Node, TensorDesc, Op enum, Layout)
- HIP backend implements ~20 operations
- Missing: Quantized matmul, Accumulate op, tensor pool/allocator, graph optimizer

**Reference:**
- llama.cpp source at `/home/feanor/Projects/llama.cpp`
- GGUF spec: `ggml/include/gguf.h`
- ggml ops: `ggml/include/ggml.h`

## Constraints

- **Hardware**: AMD GPU (RX 7900 XT / gfx1100 target) — ROCm 5.x+ required
- **Platform**: Linux only — ROCm requirement
- **Memory**: 16GB+ recommended for 7B models
- **Implementation**: Follow llama.cpp patterns — Reference proven approach
- **Compatibility**: Maintain existing CLI functionality — No breaking changes

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use ggml IR architecture | Proven by llama.cpp, clean separation of concerns | — Pending |
| Fixed-shape tensors with offsets | Avoid O(tokens) graph rebuilds | — Pending |
| Single-pass GGUF loading | Eliminate redundant parsing overhead | — Pending |
| Quantized ops in ggml backend | Follow llama.cpp quantization strategy | — Pending |

---
*Last updated: 2026-01-14 after initialization*
