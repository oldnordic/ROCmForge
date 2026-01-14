# ROCmForge Roadmap

**Goal**: Achieve performance parity with llama.cpp on AMD GPUs through ggml IR completion

**Target Hardware**: AMD RX 7900 XT (gfx1100)

**Reference Implementation**: `/home/feanor/Projects/llama.cpp`

---

## Phase 1: Single-Pass GGUF Loading

**Issue**: Triple GGUF parsing - file parsed twice (once for config, once for weights)

**Impact**: 20-30% faster startup, reduced memory overhead

### Plans
- Parse GGUF once, cache metadata in struct
- Reuse loader for both config extraction and weight loading
- Add `load_gguf_model_with_loader(Arc<GgufLoader>)` variant
- Update `InferenceEngine::from_gguf()` to use single-pass pattern

**Files**: `src/engine.rs`, `src/loader/gguf.rs`

---

## Phase 2: Fixed-Shape Tensors with Offset Views

**Issue**: Graph shape mutation every decode token causes O(tokens) rebuilds

**Impact**: 10-15% faster token generation

### Plans
- Pre-allocate max-size tensors for KV cache and attention
- Implement offset-based views instead of reshape operations
- Add `View` op variant for position-based tensor access
- Update `forward_layer_ggml_decode()` to use fixed buffers

**Files**: `src/model/execution_plan.rs`, `src/ggml/op.rs`

**Reference**: `ggml/src/ggml.c` - `ggml_view_1d()` pattern

---

## Phase 3: Quantized MatMul Operations

**Issue**: No quantized matmul ops (Q4_0, Q8_0) - must dequantize to F32 first

**Impact**: Enables efficient quantized model inference

### Plans
- Add `MatMulQ4_0`, `MatMulQ8_0` variants to `Op` enum
- Implement HIP dequantization kernels
- Add block-level quantization support
- Integrate with ggml executor

**Files**: `src/ggml/op.rs`, `src/ggml/hip_backend/mod.rs`

**Reference**: `ggml/src/ggml-cpu/quants.h`

---

## Phase 4: Static Weight Binding

**Issue**: Weights rebound every decode step despite never changing

**Impact**: 5-10% faster token generation

### Plans
- Separate static weight graphs from dynamic decode graphs
- Bind weight tensors once at graph construction
- Cache weight buffer bindings
- Update graph execution to skip redundant binds

**Files**: `src/model/execution_plan.rs`, `src/ggml/executor.rs`

---

## Phase 5: Complete Missing ggml Ops

**Issue**: Accumulate op, tensor pool/allocator, graph optimizer missing

**Impact**: Full ggml IR compatibility, cleaner architecture

### Plans
- Implement `Accumulate` op for KV cache writes
- Add tensor pool/allocator for buffer reuse
- Implement CSE (common subexpression elimination)
- Add dead code elimination pass

**Files**: `src/ggml/op.rs`, `src/ggml/allocator.rs` (new)

---

## Phase 6: End-to-End Integration Tests

**Issue**: No validation that full pipeline works with real models

**Impact**: Confidence in production readiness

### Plans
- Test with real GGUF models (Qwen2 0.5B, LLaMA 7B)
- Validate output against llama.cpp reference
- Benchmark token generation speed
- Profile memory usage

**Files**: `tests/integration/` (new directory)

---

## Order of Execution

1. **Phase 1** (Single-pass loading) - Easiest win, eliminates waste
2. **Phase 2** (Fixed-shape tensors) - Biggest performance gain
3. **Phase 4** (Static binding) - Builds on Phase 2
4. **Phase 3** (Quantized ops) - Enables quantized models
5. **Phase 5** (Complete ops) - Polish and optimization
6. **Phase 6** (Integration tests) - Validate everything

---

*Last updated: 2026-01-14*
