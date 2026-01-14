# CLI and Model Loading Analysis

**Date**: 2026-01-14
**Analyzer**: Claude (CodeMCP semantic index)
**Scope**: CLI flow, GGUF loading, ggml IR implementation

## Executive Summary

This analysis examines the CLI inference flow and model loading mechanism, comparing ROCmForge's implementation with llama.cpp's proven approach. The ggml IR implementation started yesterday is structurally sound but has several efficiency issues that prevent optimal performance.

## Architecture Overview

### Current Flow

```
CLI (run_local_generate)
  └─→ create_engine()
      └─→ InferenceEngine::from_gguf()
          ├─→ GgufLoader::new() #1 - to get config
          ├─→ InferenceEngine::new(config)
          └─→ load_gguf_model()
              └─→ ModelRuntime::load_from_gguf()
                  └─→ GgufLoader::new() #2 - actual loading
                  └─→ ExecutionPlan::from_gguf()
                  └─→ decode_step() (inference)
                      └─→ forward_layer_ggml_decode()
```

### Key Files

| File | Purpose |
|------|---------|
| `src/bin/rocmforge_cli.rs` | CLI entry point |
| `src/engine.rs` | `InferenceEngine::from_gguf()` |
| `src/backend/hip_backend.rs` | `ModelRuntime` (lines ~93121-111830) |
| `src/model/execution_plan.rs` | `ExecutionPlan`, `forward_layer_ggml_decode()` |
| `src/loader/gguf.rs` | GGUF file parsing |
| `src/ggml/` | ggml IR implementation |

---

## Critical Issues

### Issue #1: Triple GGUF Parsing

**Location**: `src/engine.rs:163-197`

```rust
pub async fn from_gguf<P: AsRef<std::path::Path>>(path: P) -> EngineResult<Self> {
    // GgufLoader #1 - just to extract config
    let loader = GgufLoader::new(&path_string)?;
    let config = loader.to_model_config()?;

    let mut engine = Self::new(engine_config)?;
    engine.load_gguf_model(path).await?;  // Creates GgufLoader #2 inside
    // ...
}
```

**Problem**: The GGUF file is parsed twice:
1. First to extract model config
2. Second during actual weight loading in `load_gguf_model()`

**Impact**: For large models (7B+), parsing metadata twice adds significant startup latency.

**llama.cpp Approach**: Single `gguf_init_from_file()` call with `no_alloc` flag for metadata-only pass.

**Fix**:
```rust
// Parse once, reuse loader
let loader = Arc::new(GgufLoader::new(&path_string)?);
let config = loader.to_model_config()?;
// Pass loader to load_gguf_model instead of path
engine.load_gguf_model_with_loader(loader).await?;
```

---

### Issue #2: Graph Shape Mutation Every Decode

**Location**: `src/model/execution_plan.rs:1578-1680`

```rust
pub(crate) fn forward_layer_ggml_decode(...) {
    let plans = self.layer_ggml_plans.get_or_try_init(|| self.build_layer_ggml_plans(backend))?;
    let plan = plans.get(layer_idx)?;

    // Shape mutation EVERY token:
    let new_len = current_len + 1;
    graph.tensors[plan.kv_read_k_id.0].set_shape(vec![new_len, num_heads, plan.head_dim]);
    graph.tensors[plan.scores_id.0].set_shape(vec![1, new_len]);
    graph.tensors[plan.softmax_id.0].set_shape(vec![1, new_len]);
    // ...
}
```

**Problem**: Tensor shapes are mutated for each token generated. This triggers graph recalculations and buffer reallocations.

**Impact**: O(tokens) graph rebuilds instead of O(1) setup + O(tokens) simple updates.

**llama.cpp Approach**: Fixed-size pre-allocated buffers with position tracking:
```c
// llama.cpp pattern
struct ggml_tensor * K = ggml_new_tensor(..., max_seq_len, n_heads, head_dim);
// Use ggml_set_1d() to create views with offset:
struct ggml_tensor * Kcur = ggml_view_1d(K, ..., pos * n_heads * head_dim);
```

**Fix**: Pre-allocate max-size tensors, use offset-based views instead of reshaping.

---

### Issue #3: Weights Bound Per-Decode-Step

**Location**: `src/model/execution_plan.rs:1027-1576` (build_layer_ggml_plans)

```rust
fn build_layer_ggml_plans(&self, _backend: &HipBackend) -> HipResult<Vec<LayerGgmlPlan>> {
    for layer_plan in &self.layers {
        let qkv_weight = self.get_or_load_tensor(&layer_plan.qkv_weight)?;
        let o_proj = self.get_or_load_tensor(&layer_plan.o_proj)?;
        // ... all weights loaded and added to graph

        // But these bindings happen EVERY decode step through execute_graph()
    }
}
```

**Problem**: Weight tensors are added to graphs and bound on every decode pass, even though they never change.

**Impact**: Unnecessary buffer binding overhead per token.

**llama.cpp Approach**: Weights are bound once at graph construction time and reused:
```c
// Weights are part of the static graph structure
struct ggml_tensor * wq = ggml_new_tensor(..., GGML_TYPE_F16, ...);
// Bind once, reuse forever
ggml_set_tensor(wq, weight_data);
```

**Fix**: Separate static weight bindings from dynamic per-token bindings.

---

### Issue #4: Missing Quantization Ops

**Location**: `src/ggml/op.rs`

```rust
pub enum Op {
    GetRows,
    MatMul,              // ← Only F32 matmul!
    Add,
    Mask,
    Scale { factor: f32 },
    LayerNorm { eps: f32 },
    RmsNorm { eps: f32 },
    Rope,
    Softmax,
    Attention,
    SwiGlu,
    MlpSwiglu,
    SplitQkv,
    Reshape,
    View,
    Copy,
    // Missing: Quantized matmul variants
}
```

**Problem**: No quantized matrix multiplication operations (Q4_0, Q4_K, Q8_0, etc.). The `MatMul` op only handles F32.

**Impact**: Quantized models (Q4_0, Q4_K, Q8_0) must dequantize to F32 before matmul, negating memory/compute savings.

**llama.cpp Approach**: Extensive quantization support:
```c
enum ggml_op {
    GGML_OP_MUL_MAT,
    GGML_OP_MUL_MAT_ID,  // For expert models
    // Many dequantize ops:
    GGML_OP_DEQUANTIZE_Q4_0,
    GGML_OP_DEQUANTIZE_Q4_K,
    GGML_OP_DEQUANTIZE_Q8_0,
    // ...
}
```

**Fix**: Add quantized matmul ops to ggml IR and implement HIP kernels.

---

### Issue #5: Inefficient KV Cache Access Pattern

**Location**: `src/model/execution_plan.rs:1633-1640`

```rust
// Current approach: create new buffer view for write
let kv_write_k_view = kv_keys.buffer().sub_buffer_view(write_offset, write_bytes)?;
ggml_backend.bind(&graph.tensors[plan.kv_write_k_id.0], kv_write_k_view)?;
```

**Problem**: While using offset correctly, the pattern requires creating new view objects and re-binding every token.

**llama.cpp Approach**: Pointer arithmetic on pre-allocated buffers:
```c
// Pre-allocated KV cache
struct ggml_tensor * Kcache = ggml_new_tensor(..., max_seq_len, n_heads, head_dim);
// View with offset - NO bind every time
struct ggml_tensor * Kcur = ggml_view_1d(Kcache, n_heads * head_dim, pos * head_size);
```

**Impact**: Extra allocations and bindings per token.

---

## Comparison: ROCmForge vs llama.cpp

| Aspect | llama.cpp | ROCmForge | Status |
|--------|-----------|-----------|--------|
| GGUF Loading | Single pass with mmap | Multiple parses | ⚠️ Inefficient |
| Graph Building | Once at init | Per-layer lazy init | ⚠️ Acceptable |
| Graph Reuse | Fixed shapes, offset-based | Shape mutation | ❌ Rebuilds every token |
| KV Cache | Fixed buffer, pos-based | View reshaping | ⚠️ Functional but slow |
| Quantization | Full ggml_quants support | Limited (F32/F16 only in ggml) | ❌ Missing ops |
| Ops | 50+ ggml ops | ~20 ops | ⚠️ Incomplete |
| Weight Binding | Once at graph build | Every decode step | ❌ Overhead |
| Memory Mapping | mmap for fast access | Full load into memory | ⚠️ Startup time |

---

## ggml IR Implementation Status

### What's Complete ✅

1. **Core IR Structure** (`src/ggml/`)
   - `Graph`, `Node`, `TensorDesc`
   - `Op` enum with basic operations
   - `Layout` (RowMajor, ColMajor, Strided)

2. **Backend Trait** (`src/ggml/backend.rs`)
   - `GgmlBackend` trait with alloc, bind, execute_op, synchronize
   - Clean separation of concerns

3. **HIP Backend** (`src/ggml/hip_backend/mod.rs`)
   - `HipGgmlBackend` implements all required traits
   - Operations: GetRows, MatMul, Add, Scale, LayerNorm, RmsNorm, RoPE, Softmax, Attention, Mask, SwiGlu, MlpSwiglu, SplitQkv, Reshape, View, Copy

4. **Executor** (`src/ggml/executor.rs`)
   - Simple graph execution loop
   - Buffer allocation and op dispatch

### What's Missing ❌

1. **Quantization Kernels**
   - Q4_0, Q4_K, Q8_0 dequantization
   - Quantized matmul (MulMatQ)

2. **Accumulate Op**
   - For KV cache writes in single pass
   - Currently using Copy + manual offset management

3. **Tensor Pool/Allocator**
   - llama.cpp has `ggml_allocr` for efficient buffer reuse
   - Current implementation allocates per-tensor

4. **Graph Optimization**
   - CSE (common subexpression elimination)
   - Dead code elimination
   - Layout optimization

---

## Recommendations

### High Priority (Performance)

1. **Cache GGUF Metadata**
   - Parse once, store in struct
   - Reuse for both config and weight loading
   - Estimated gain: 20-30% faster startup

2. **Fix Graph Rebuilding**
   - Pre-allocate max-size tensors
   - Use offset-based views (ggml_view_1d pattern)
   - Estimated gain: 10-15% faster token generation

3. **Bind Weights Once**
   - Separate static weight graphs from dynamic decode graphs
   - Estimated gain: 5-10% faster token generation

### Medium Priority (Functionality)

4. **Add Quantized MatMul**
   - Implement Q4_0, Q8_0 matmul kernels
   - Add to `Op` enum and HIP backend
   - Required for efficient quantized model inference

5. **Integrate Lazy Loading**
   - `LazyTensor` exists but not fully utilized in ggml path
   - Load weights on-demand during first use

### Low Priority (Optimization)

6. **Memory Mapping**
   - Use mmap for GGUF tensor data
   - Load weights lazily from file
   - Reduces startup memory for large models

7. **Graph Optimizer**
   - CSE for shared subgraphs
   - Layout optimization for better cache utilization

---

## Code References

### llama.cpp Files for Reference

| File | Purpose |
|------|---------|
| `ggml/include/ggml.h` | Core ggml API |
| `ggml/include/gguf.h` | GGUF file format |
| `src/llama-model.h` | Model loading logic |
| `src/llama-graph.cpp` | Graph construction |
| `ggml/src/ggml-cpu/quants.h` | Quantization kernels |

### ROCmForge Files to Modify

| File | Changes Needed |
|------|---------------|
| `src/engine.rs` | Single-pass GGUF loading |
| `src/model/execution_plan.rs` | Fixed-shape tensors, offset views |
| `src/ggml/op.rs` | Add quantized ops |
| `src/ggml/hip_backend/mod.rs` | Add quantized kernels |
| `src/loader/gguf.rs` | Add mmap support |

---

## Testing Recommendations

1. **Benchmark GGUF loading time** - Before/after single-pass fix
2. **Profile token generation** - Identify graph rebuild overhead
3. **Memory usage tracking** - Verify lazy loading effectiveness
4. **Quantized model testing** - Q4_0, Q8_0 with new ops

---

## Conclusion

The ggml IR implementation is **structurally sound** and follows good design patterns. The main issues are efficiency-related:

1. **Triple parsing** is wasteful but easy to fix
2. **Graph rebuilding** every token is the biggest performance issue
3. **Missing quantization** limits model compatibility

The implementation is a solid foundation that needs optimization work rather than architectural changes. Following llama.cpp's patterns for:
- Single-pass GGUF loading
- Fixed-size tensor buffers with offset views
- Quantized matmul kernels

...will bring performance to par with reference implementations.

---

**Next Steps**: Prioritize Issue #2 (graph rebuilding) for maximum performance gain, or Issue #1 (triple parsing) for easiest win.
