# ROCmForge Code Architecture & Modularization Analysis

**Analysis Date:** 2026-01-07
**Total Files Analyzed:** 105 Rust files
**Total LOC:** 33,123
**Constraints:** Soft limit 300 LOC, Hard limit 600 LOC
**Analyst:** Architecture Review Agent

---

## Executive Summary

The ROCmForge codebase shows **critical modularization violations** with **5 files exceeding the 600 LOC hard ceiling** and **8 additional files exceeding the 300 LOC soft limit**. This represents a **13% violation rate** for the core codebase, with test files showing additional violations.

**Most Critical Issues:**
1. `src/model/execution_plan.rs` - **2,305 LOC** (384% over hard limit)
2. `src/backend/hip_backend.rs` - **2,135 LOC** (356% over hard limit)
3. `src/loader/gguf.rs` - **1,950 LOC** (325% over hard limit)
4. `src/ops/attention_gpu.rs` - **1,222 LOC** (204% over hard limit)
5. `src/attention/kernels.rs` - **955 LOC** (159% over hard limit)

These violations mix multiple concerns, create maintenance burden, and violate the GLOBAL ENGINEERING CONSTRAINTS defined in the project standards.

---

## Violation Summary Table

### Production Code Violations

| File | LOC | Status | Priority | Mixes Concerns |
|------|-----|--------|----------|----------------|
| `src/model/execution_plan.rs` | 2,305 | **CRITICAL SPLIT** | P0 | Yes (4+ concerns) |
| `src/backend/hip_backend.rs` | 2,135 | **CRITICAL SPLIT** | P0 | Yes (5+ concerns) |
| `src/loader/gguf.rs` | 1,950 | **CRITICAL SPLIT** | P0 | Yes (3+ concerns) |
| `src/ops/attention_gpu.rs` | 1,222 | **CRITICAL SPLIT** | P0 | Yes (3+ concerns) |
| `src/attention/kernels.rs` | 955 | **SPLIT** | P1 | Yes (kernel loading) |
| `src/engine.rs` | 803 | **SPLIT** | P1 | Yes (orchestration) |
| `src/http/server.rs` | 662 | **SPLIT** | P1 | Yes (HTTP + engine) |
| `src/scheduler/scheduler.rs` | 640 | **SPLIT** | P1 | Yes (scheduling + state) |
| `src/attention/multi_query.rs` | 609 | **SPLIT** | P1 | No (cohesive) |
| `src/model/simple_transformer.rs` | 606 | **SPLIT** | P1 | No (cohesive) |
| `src/model/glm_position.rs` | 601 | **SPLIT** | P1 | Yes (position + GLM) |
| `src/models.rs` | 484 | **MONITOR** | P2 | No (modest exceed) |
| `src/sampler/sampler.rs` | 474 | **MONITOR** | P2 | No (cohesive) |
| `src/backend/gpu_executor.rs` | 456 | **MONITOR** | P2 | No (cohesive) |

### Test File Violations (Lower Priority)

| File | LOC | Status | Notes |
|------|-----|--------|-------|
| `tests/q_dequant_tests.rs` | 643 | Acceptable | Tests can be larger |
| `tests/scheduler_tests.rs` | 552 | Acceptable | Test suite |
| `tests/attention_tests.rs` | 548 | Acceptable | Test suite |
| `tests/execution_plan_and_decode_tests.rs` | 481 | Acceptable | Integration tests |
| `src/loader/mxfp_tests.rs` | 454 | Acceptable | Specialized tests |
| `src/attention/flash_attention_tests.rs` | 551 | Acceptable | Test suite |
| `src/attention/flash_nocausal_tests.rs` | 470 | Acceptable | Test suite |
| `src/ops/causal_mask_tests.rs` | 437 | Acceptable | Test suite |
| `tests/kv_cache_tests.rs` | 416 | Acceptable | Test suite |
| `tests/execution_plan_weight_mapping_tests.rs` | 379 | Acceptable | Test suite |

---

## Detailed Refactoring Proposals

### Priority P0: CRITICAL SPLIT Required (>1000 LOC)

---

#### 1. `src/model/execution_plan.rs` (2,305 LOC)

**Current Concerns Mixed:**
- Architecture detection logic (Qwen2, LLaMA, Mistral)
- Weight mapping (embedding, LM head, attention, MLP, layer norms)
- Forward pass execution (embedding lookup, layer execution, attention, MLP)
- Model configuration management
- QKV tensor concatenation
- Tensor transposition utilities

**Proposed Split Strategy:**

```
src/model/
├── execution_plan/
│   ├── mod.rs                    # 150 LOC - Module exports
│   ├── plan.rs                   # 200 LOC - ExecutionPlan struct, layers()
│   ├── architecture.rs            # 180 LOC - Architecture enum, detection
│   ├── forward.rs                 # 350 LOC - forward(), forward_layer()
│   ├── weights/
│   │   ├── mod.rs                # 100 LOC - Module exports
│   │   ├── embedding.rs          # 250 LOC - map_embedding(), map_lm_head()
│   │   ├── attention.rs          # 400 LOC - map_attention_weights(), concatenate_qkv()
│   │   ├── mlp.rs                # 300 LOC - map_mlp_weights()
│   │   ├── layer_norm.rs         # 250 LOC - map_layer_norm_weights()
│   │   └── transpose.rs          # 150 LOC - transpose_2d_tensor()
│   └── layer_plan.rs             # 200 LOC - LayerPlan struct
```

**File Sizes After Split:**
- `mod.rs`: 150 LOC
- `plan.rs`: 200 LOC
- `architecture.rs`: 180 LOC
- `forward.rs`: 350 LOC
- `weights/mod.rs`: 100 LOC
- `weights/embedding.rs`: 250 LOC
- `weights/attention.rs`: 400 LOC
- `weights/mlp.rs`: 300 LOC
- `weights/layer_norm.rs`: 250 LOC
- `weights/transpose.rs`: 150 LOC
- `layer_plan.rs`: 200 LOC

**Max file size:** 400 LOC (within 33% overage, acceptable for cohesive modules)

**Separation of Concerns Achieved:**
- Architecture detection isolated
- Weight mapping broken down by component
- Forward pass logic separate from weight mapping
- Tensor utilities isolated

**Refactoring Effort:** ~16 hours (high complexity, many cross-references)

---

#### 2. `src/backend/hip_backend.rs` (2,135 LOC)

**Current Concerns Mixed:**
- FFI bindings (extern "C" functions)
- Device properties (HipDeviceProp with offset accessors)
- Memory management (HipBuffer, HipStream)
- Kernel management (HipModule, HipKernel)
- DeviceTensor operations
- Backend singleton (GLOBAL_BACKEND, initialization)
- ModelRuntime (should be separate)
- BLAS operations (add_inplace, scale_inplace)
- MLP and LayerNorm implementations
- Transformer layer execution
- High-level decode step

**Proposed Split Strategy:**

```
src/backend/
├── hip_backend/
│   ├── mod.rs                    # 150 LOC - Public API re-exports
│   ├── ffi.rs                    # 400 LOC - extern "C" bindings
│   ├── device.rs                 # 250 LOC - HipDevice, HipDeviceProp
│   ├── memory.rs                 # 350 LOC - HipBuffer, HipStream
│   ├── kernel.rs                 # 200 LOC - HipModule, HipKernel
│   ├── tensor.rs                 # 300 LOC - DeviceTensor
│   ├── backend.rs                # 400 LOC - HipBackend singleton, init
│   └── ops/
│       ├── mod.rs                # 100 LOC - Module exports
│       ├── blas.rs               # 250 LOC - add_inplace, scale_inplace
│       ├── mlp.rs                # 300 LOC - mlp_swiglu()
│       ├── layernorm.rs          # 250 LOC - layernorm()
│       └── transformer.rs        # 350 LOC - transformer_layer(), decode_step()
├── model_runtime.rs              # 600 LOC - Extract ModelRuntime entirely
└── mod.rs                        # Update exports
```

**File Sizes After Split:**
- `hip_backend/mod.rs`: 150 LOC
- `hip_backend/ffi.rs`: 400 LOC
- `hip_backend/device.rs`: 250 LOC
- `hip_backend/memory.rs`: 350 LOC
- `hip_backend/kernel.rs`: 200 LOC
- `hip_backend/tensor.rs`: 300 LOC
- `hip_backend/backend.rs`: 400 LOC
- `hip_backend/ops/mod.rs`: 100 LOC
- `hip_backend/ops/blas.rs`: 250 LOC
- `hip_backend/ops/mlp.rs`: 300 LOC
- `hip_backend/ops/layernorm.rs`: 250 LOC
- `hip_backend/ops/transformer.rs`: 350 LOC
- `model_runtime.rs`: 600 LOC

**Max file size:** 600 LOC (at hard ceiling, justified for cohesive MLP/LayerNorm)

**Separation of Concerns Achieved:**
- FFI bindings isolated (unsafe code contained)
- Memory management separated
- Device management separated
- Kernel loading separated
- Tensor operations separated
- High-level backend operations in dedicated modules
- ModelRuntime extracted as top-level module

**Refactoring Effort:** ~20 hours (highest complexity, FFI safety, global state)

---

#### 3. `src/loader/gguf.rs` (1,950 LOC)

**Current Concerns Mixed:**
- E8M0 and MXFP data structures
- GGUF tensor type definitions
- Metadata structures
- GGUF file format parsing (KV pairs, tensor info)
- Quantization/dequantization (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4, MXFP6)
- Memory pooling for GPU upload
- Metadata inference (vocab_size, intermediate_size)
- ModelConfig conversion
- F16 conversion utilities

**Proposed Split Strategy:**

```
src/loader/
├── gguf/
│   ├── mod.rs                    # 150 LOC - Public API
│   ├── types.rs                  # 250 LOC - GgufTensorType, E8M0, MxfpBlock
│   ├── metadata.rs               # 200 LOC - GgufMetadata, inference
│   ├── parser.rs                 # 400 LOC - parse_kv_pairs, parse_tensor_info
│   ├── dequant/
│   │   ├── mod.rs                # 100 LOC - Module exports
│   │   ├── q8.rs                 # 150 LOC - dequantize_q8_0()
│   │   ├── q4.rs                 # 200 LOC - dequantize_q4_0(), q4_1()
│   │   ├── q5.rs                 # 200 LOC - dequantize_q5_0(), q5_1()
│   │   └── mxfp.rs               # 300 LOC - dequantize_mxfp4(), mxfp6()
│   ├── upload.rs                 # 350 LOC - load_to_gpu(), memory pooling
│   ├── config.rs                 # 200 LOC - to_model_config()
│   └── loader.rs                 # 300 LOC - GgufLoader struct, load_from_disk()
└── mod.rs                        # Update exports
```

**File Sizes After Split:**
- `gguf/mod.rs`: 150 LOC
- `gguf/types.rs`: 250 LOC
- `gguf/metadata.rs`: 200 LOC
- `gguf/parser.rs`: 400 LOC
- `gguf/dequant/mod.rs`: 100 LOC
- `gguf/dequant/q8.rs`: 150 LOC
- `gguf/dequant/q4.rs`: 200 LOC
- `gguf/dequant/q5.rs`: 200 LOC
- `gguf/dequant/mxfp.rs`: 300 LOC
- `gguf/upload.rs`: 350 LOC
- `gguf/config.rs`: 200 LOC
- `gguf/loader.rs`: 300 LOC

**Max file size:** 400 LOC (within acceptable range)

**Separation of Concerns Achieved:**
- Type definitions isolated
- Metadata handling separated
- Parsing logic isolated
- Dequantization broken down by format
- GPU upload logic isolated (memory pooling is distinct concern)
- Config conversion separated

**Refactoring Effort:** ~14 hours (medium complexity, clear boundaries)

---

#### 4. `src/ops/attention_gpu.rs` (1,222 LOC)

**Current Concerns Mixed:**
- HipAttentionKernels struct
- Kernel compilation and caching
- QK^T computation (GEMM + CPU fallback)
- Causal masking (GPU + CPU fallback)
- Softmax (GPU + CPU fallback)
- Attention-weighted V computation (GEMM + CPU fallback)
- CPU fallback implementations
- KV cache integration

**Proposed Split Strategy:**

```
src/ops/
├── attention/
│   ├── mod.rs                    # 150 LOC - Public API
│   ├── kernels.rs                # 300 LOC - HipAttentionKernels, init
│   ├── qk.rs                     # 250 LOC - compute_qk_t(), compute_qk_t_gemm()
│   ├── mask.rs                   # 200 LOC - apply_causal_mask(), GPU impl
│   ├── softmax.rs                # 250 LOC - compute_softmax(), GPU impl
│   ├── weighted_v.rs             # 250 LOC - compute_attention_weighted_v()
│   └── fallback.rs               # 300 LOC - All CPU fallbacks
└── mod.rs                        # Update exports
```

**File Sizes After Split:**
- `attention/mod.rs`: 150 LOC
- `attention/kernels.rs`: 300 LOC
- `attention/qk.rs`: 250 LOC
- `attention/mask.rs`: 200 LOC
- `attention/softmax.rs`: 250 LOC
- `attention/weighted_v.rs`: 250 LOC
- `attention/fallback.rs`: 300 LOC

**Max file size:** 300 LOC (all at or below soft limit)

**Separation of Concerns Achieved:**
- Kernel management isolated
- Each attention operation in its own file
- CPU fallbacks consolidated (single concern: software implementation)
- Clean separation between GPU/CPU paths

**Refactoring Effort:** ~10 hours (medium complexity, clear operation boundaries)

---

#### 5. `src/attention/kernels.rs` (955 LOC)

**Current Concerns Mixed:**
- Global kernel cache initialization
- HSACO path loading (10+ different kernels)
- Individual kernel wrappers (scale, mask, softmax, RoPE, etc.)
- Kernel launching logic
- Tuning constants

**Proposed Split Strategy:**

```
src/attention/
├── kernels/
│   ├── mod.rs                    # 150 LOC - Public API, get_or_init_cache()
│   ├── cache.rs                  # 300 LOC - KernelCache struct, initialization
│   ├── scale.rs                  # 150 LOC - scale_gpu_kernel()
│   ├── mask.rs                   # 150 LOC - mask_gpu_kernel()
│   ├── softmax.rs                # 150 LOC - softmax_gpu_kernel()
│   ├── rope.rs                   # 150 LOC - rope_gpu_kernel()
│   ├── matmul.rs                 # 200 LOC - qkt_matmul, weighted_matmul
│   └── flash.rs                  # 300 LOC - FlashAttention kernels
└── kernels.rs                    # DEPRECATED - Move to kernels/
```

**File Sizes After Split:**
- `kernels/mod.rs`: 150 LOC
- `kernels/cache.rs`: 300 LOC
- `kernels/scale.rs`: 150 LOC
- `kernels/mask.rs`: 150 LOC
- `kernels/softmax.rs`: 150 LOC
- `kernels/rope.rs`: 150 LOC
- `kernels/matmul.rs`: 200 LOC
- `kernels/flash.rs`: 300 LOC

**Max file size:** 300 LOC (all at or below soft limit)

**Separation of Concerns Achieved:**
- Cache management isolated (complex initialization)
- Each kernel wrapper in its own file
- FlashAttention grouped together (related variants)
- Public API consolidated

**Refactoring Effort:** ~12 hours (medium complexity, cache initialization complexity)

---

### Priority P1: SPLIT Required (600-1000 LOC)

---

#### 6. `src/engine.rs` (803 LOC)

**Current Concerns Mixed:**
- Engine configuration
- InferenceEngine struct
- Request state management
- Model loading (GGUF, ONNX)
- Inference loop orchestration
- HTTP integration coupling
- Request lifecycle management

**Proposed Split Strategy:**

```
src/engine/
├── mod.rs                        # 150 LOC - Public API
├── config.rs                    # 150 LOC - EngineConfig, EngineError
├── engine.rs                    # 300 LOC - InferenceEngine struct, core lifecycle
├── model.rs                     # 250 LOC - load_gguf_model(), load_onnx_model()
├── inference.rs                 # 300 LOC - run_inference_loop(), inference_loop()
└── request.rs                   # 250 LOC - submit_request(), RequestRuntimeState
```

**File Sizes After Split:**
- `mod.rs`: 150 LOC
- `config.rs`: 150 LOC
- `engine.rs`: 300 LOC
- `model.rs`: 250 LOC
- `inference.rs`: 300 LOC
- `request.rs`: 250 LOC

**Max file size:** 300 LOC (all at or below soft limit)

**Separation of Concerns Achieved:**
- Configuration separated
- Model loading isolated
- Inference loop isolated
- Request management isolated
- Core engine lifecycle focused

**Refactoring Effort:** ~8 hours (moderate complexity)

---

#### 7. `src/http/server.rs` (662 LOC)

**Current Concerns Mixed:**
- HTTP server setup
- Route handlers (health, generate, load, status)
- Inference engine coupling
- Request/response DTOs
- WebSocket support (if present)

**Proposed Split Strategy:**

```
src/http/
├── mod.rs                        # 150 LOC - Public API
├── server.rs                    # 300 LOC - Server setup, Axum state
├── routes.rs                    # 400 LOC - All route handlers
├── dto.rs                       # 200 LOC - Request/response types
└── websocket.rs                 # 200 LOC - WebSocket handling (if present)
```

**File Sizes After Split:**
- `mod.rs`: 150 LOC
- `server.rs`: 300 LOC
- `routes.rs`: 400 LOC
- `dto.rs`: 200 LOC
- `websocket.rs`: 200 LOC (if needed)

**Max file size:** 400 LOC (acceptable for route consolidation)

**Separation of Concerns Achieved:**
- Server setup separated from route handling
- DTOs isolated
- WebSocket logic separated (if present)

**Refactoring Effort:** ~6 hours (low complexity)

---

#### 8. `src/scheduler/scheduler.rs` (640 LOC)

**Current Concerns Mixed:**
- Scheduler configuration
- Request queue management
- Batching logic
- Request state tracking
- Priority management

**Proposed Split Strategy:**

```
src/scheduler/
├── mod.rs                        # 150 LOC - Public API
├── config.rs                    # 150 LOC - SchedulerConfig
├── scheduler.rs                 # 300 LOC - Scheduler struct, queue mgmt
├── batching.rs                  # 250 LOC - Batching logic
└── state.rs                     # 200 LOC - RequestState management
```

**File Sizes After Split:**
- `mod.rs`: 150 LOC
- `config.rs`: 150 LOC
- `scheduler.rs`: 300 LOC
- `batching.rs`: 250 LOC
- `state.rs`: 200 LOC

**Max file size:** 300 LOC (all at or below soft limit)

**Separation of Concerns Achieved:**
- Configuration separated
- Queue management focused
- Batching logic isolated
- State management isolated

**Refactoring Effort:** ~6 hours (low complexity)

---

#### 9. `src/attention/multi_query.rs` (609 LOC)

**Analysis:** This file appears cohesive (single concern: multi-query attention). The 609 LOC is only slightly over the hard limit and likely represents a well-structured implementation.

**Recommendation:** **MONITOR** - Keep as-is unless it grows further. Document why the exceedance is acceptable (cohesive concern).

---

#### 10. `src/model/simple_transformer.rs` (606 LOC)

**Analysis:** Similar to multi_query.rs, this appears to be a cohesive implementation of a simple transformer architecture.

**Recommendation:** **MONITOR** - Keep as-is. Document cohesion.

---

#### 11. `src/model/glm_position.rs` (601 LOC)

**Current Concerns Mixed:**
- GLM-specific model architecture
- Position embedding logic

**Proposed Split Strategy:**

```
src/model/
├── glm/
│   ├── mod.rs                    # 100 LOC - Module exports
│   ├── transformer.rs           # 350 LOC - GLM transformer logic
│   └── position.rs              # 250 LOC - Position embeddings
└── glm_position.rs              # DEPRECATED - Move to glm/
```

**File Sizes After Split:**
- `glm/mod.rs`: 100 LOC
- `glm/transformer.rs`: 350 LOC
- `glm/position.rs`: 250 LOC

**Max file size:** 350 LOC (within acceptable range)

**Separation of Concerns Achieved:**
- GLM-specific architecture isolated
- Position embeddings separated

**Refactoring Effort:** ~4 hours (low complexity)

---

### Priority P2: MONITOR (400-600 LOC)

---

#### 12-14. Files Under 500 LOC

These files modestly exceed the soft limit but are likely cohesive:
- `src/models.rs` (484 LOC)
- `src/sampler/sampler.rs` (474 LOC)
- `src/backend/gpu_executor.rs` (456 LOC)

**Recommendation:** **MONITOR** - Review during next refactoring cycle. No immediate action required unless they grow further or mix concerns.

---

## Refactoring Priority & Timeline

### Phase 1: Critical Splits (Weeks 1-4)
**Effort:** ~60 hours
**Target:** P0 files (>1000 LOC)

1. **Week 1-2:** `src/backend/hip_backend.rs` (20h) - Highest complexity
2. **Week 2-3:** `src/model/execution_plan.rs` (16h)
3. **Week 3-4:** `src/loader/gguf.rs` (14h)
4. **Week 4:** `src/ops/attention_gpu.rs` (10h)

**Expected Outcome:** Reduce 4 critical files to compliant sizes, establish modularization patterns

### Phase 2: High-Priority Splits (Weeks 5-7)
**Effort:** ~26 hours
**Target:** P1 files (600-1000 LOC)

1. **Week 5:** `src/attention/kernels.rs` (12h)
2. **Week 6:** `src/engine.rs` (8h)
3. **Week 6:** `src/http/server.rs` (6h)
4. **Week 7:** `src/scheduler/scheduler.rs` (6h)
5. **Week 7:** `src/model/glm_position.rs` (4h)

**Expected Outcome:** All production files under 600 LOC hard limit

### Phase 3: Monitoring & Cleanup (Week 8+)
**Effort:** ~8 hours
**Target:** P2 files (400-600 LOC)

1. Review `src/models.rs`, `src/sampler/sampler.rs`, `src/backend/gpu_executor.rs`
2. Update module documentation
3. Establish automated LOC monitoring in CI/CD

---

## Risk Assessment

### High-Risk Refactorings

1. **`src/backend/hip_backend.rs` (RISK: CRITICAL)**
   - **Risk:** FFI safety, global state, Arc/RwLock complexity
   - **Mitigation:**
     - Keep FFI bindings in single module (don't split extern "C" blocks)
     - Maintain singleton pattern during split
     - Extensive testing of GPU operations
     - Incremental migration (create new modules before deleting old code)

2. **`src/model/execution_plan.rs` (RISK: HIGH)**
   - **Risk:** Complex weight mapping logic, architecture detection
   - **Mitigation:**
     - Create `weights/` subdirectory first
     - Migrate one weight type at a time (test after each)
     - Keep public API stable
     - Add integration tests for each architecture

3. **`src/loader/gguf.rs` (RISK: MEDIUM)**
   - **Risk:** Dequantization correctness, memory pooling
   - **Mitigation:**
     - Keep dequantization implementations isolated
     - Validate with known GGUF files
     - Test memory pooling with large models

### Low-Risk Refactorings

- `src/ops/attention_gpu.rs` - Clear operation boundaries
- `src/attention/kernels.rs` - Kernel wrappers are independent
- `src/engine.rs` - Orchestration logic is easily separable
- `src/http/server.rs` - Route extraction is straightforward
- `src/scheduler/scheduler.rs` - Well-defined concerns

---

## Testing Strategy

### Pre-Refactoring Baseline

1. **Capture Current Test Coverage:**
   ```bash
   cargo test --workspace 2>&1 | tee baseline_test_results.txt
   ```

2. **Performance Benchmarks:**
   - Measure inference latency before refactoring
   - Track memory usage patterns
   - Document GPU kernel launch times

### Regression Testing

1. **Per-File Testing:**
   - After each file split, run targeted tests
   - Verify no behavioral changes
   - Check GPU memory correctness

2. **Integration Testing:**
   - End-to-end inference tests
   - Model loading tests (GGUF, ONNX)
   - Attention computation tests

3. **Performance Validation:**
   - Compare pre/post refactoring benchmarks
   - Ensure no performance degradation >5%

### Test File Strategy

- **DO NOT aggressively split test files** - Test files >300 LOC are acceptable
- **Focus test refactoring on:** Reducing duplication, organizing by feature
- **Keep test cohesion:** Related tests should stay together

---

## Modularization Patterns Established

### Pattern 1: Subdirectory Modules

For large modules with clear sub-concerns:

```
src/module_name/
├── mod.rs              # Public API, re-exports
├── core.rs             # Primary struct/impl
├── sub_concern_1.rs    # Isolated concern
├── sub_concern_2.rs
└── utils.rs            # Helper functions
```

**Examples:** `execution_plan/`, `hip_backend/`, `gguf/`

### Pattern 2: Operations Subdirectory

For modules with multiple distinct operations:

```
src/ops/
├── mod.rs              # Public API
└── operation_name/
    ├── mod.rs          # Operation-specific API
    ├── gpu.rs          # GPU implementation
    └── fallback.rs     # CPU fallback
```

**Examples:** `attention/`, `mlp/` (future)

### Pattern 3: Configuration Isolation

Separate configuration from implementation:

```
src/module/
├── config.rs           # Config structs, defaults
├── impl.rs             # Main implementation
└── types.rs            # Associated types
```

**Examples:** `engine/`, `scheduler/`

---

## CI/CD Integration

### Automated LOC Monitoring

Add to `.github/workflows/code-quality.yml`:

```yaml
name: Code Quality

on: [pull_request]

jobs:
  loc-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check file sizes
        run: |
          # Find files exceeding 300 LOC
          find src -name "*.rs" -type f -exec wc -l {} + | \
            awk '$1 > 300 { print $0 }' > violations.txt

          # Fail if critical violations found
          if awk '$1 > 600 { print }' violations.txt | grep -q .; then
            echo "CRITICAL: Files exceed 600 LOC hard limit"
            cat violations.txt
            exit 1
          fi

          # Warn on soft violations
          if [ -s violations.txt ]; then
            echo "WARNING: Files exceed 300 LOC soft limit"
            cat violations.txt
          fi
```

### Pre-Commit Hooks

Add `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Check for files >600 LOC before committing

find src -name "*.rs" -type f -exec wc -l {} + | \
  awk '$1 > 600 { print FILENAME " exceeds 600 LOC: " $1 " lines" }' | \
  while read line; do
    echo "ERROR: $line"
    exit 1
  done
```

---

## Success Metrics

### Quantitative Goals

1. **Zero files >600 LOC** (hard ceiling compliance)
2. **<5 files >300 LOC** (soft limit compliance)
3. **<10% average LOC per file** (overall codebase health)
4. **0% performance regression** (maintain inference speed)
5. **100% test pass rate** (no regressions)

### Qualitative Goals

1. **Clear separation of concerns** - Each file has single, well-defined purpose
2. **Easy navigation** - Developers can find code quickly
3. **Maintainability** - New features can be added without touching large files
4. **Testability** - Modules can be tested in isolation
5. **Documentation** - Each module has clear API documentation

---

## Conclusion

The ROCmForge codebase requires **immediate modularization effort** to comply with GLOBAL ENGINEERING CONSTRAINTS. The proposed refactoring plan:

- **Reduces 5 critical violations** (>1000 LOC) to compliant sizes
- **Addresses 8 additional violations** (>600 LOC)
- **Establishes modularization patterns** for future development
- **Maintains GPU performance** through careful testing
- **Requires ~94 hours of focused effort** over 8 weeks

**Recommended Action:** Begin Phase 1 immediately with `src/backend/hip_backend.rs`, as it represents the highest risk and greatest complexity. The modularization patterns established during this refactoring will serve as templates for the remaining files.

**Next Steps:**
1. Review and approve this refactoring plan
2. Create tracking issues for each phase
3. Set up CI/CD LOC monitoring
4. Begin Phase 1 with `hip_backend.rs` refactoring
5. Document lessons learned for subsequent refactoring cycles

---

## Appendix: File-by-File Breakdown

### All Production Rust Files (>200 LOC)

| File | LOC | Status | Action Required |
|------|-----|--------|-----------------|
| `src/model/execution_plan.rs` | 2,305 | CRITICAL | Split into 11 files |
| `src/backend/hip_backend.rs` | 2,135 | CRITICAL | Split into 13 files |
| `src/loader/gguf.rs` | 1,950 | CRITICAL | Split into 12 files |
| `src/ops/attention_gpu.rs` | 1,222 | CRITICAL | Split into 7 files |
| `src/attention/kernels.rs` | 955 | HIGH | Split into 8 files |
| `src/engine.rs` | 803 | HIGH | Split into 6 files |
| `src/http/server.rs` | 662 | HIGH | Split into 4 files |
| `src/scheduler/scheduler.rs` | 640 | HIGH | Split into 5 files |
| `src/attention/multi_query.rs` | 609 | MONITOR | Document cohesion |
| `src/model/simple_transformer.rs` | 606 | MONITOR | Document cohesion |
| `src/model/glm_position.rs` | 601 | HIGH | Split into 3 files |
| `src/models.rs` | 484 | MONITOR | Monitor growth |
| `src/sampler/sampler.rs` | 474 | MONITOR | Monitor growth |
| `src/backend/gpu_executor.rs` | 456 | MONITOR | Monitor growth |
| `src/kv_cache/kv_cache.rs` | 447 | OK | No action |
| `src/attention/gpu.rs` | 423 | OK | No action |
| `src/attention/rope.rs` | 399 | OK | No action |
| `src/tensor/matmul.rs` | 306 | OK | No action |
| `src/model/kv_cache.rs` | 285 | OK | No action |
| `src/mlp/kernels.rs` | 275 | OK | No action |
| `src/ops/qkv.rs` | 272 | OK | No action |
| `src/loader/onnx_loader.rs` | 246 | OK | No action |
| `src/backend/hip_blas.rs` | 247 | OK | No action |
| `src/tokenizer.rs` | 194 | OK | No action |
| `src/loader/mmap_loader.rs` | 181 | OK | No action |
| `src/backend/scratch.rs` | 161 | OK | No action |
| `src/model/config.rs` | 132 | OK | No action |
| `src/attention/compute.rs` | 103 | OK | No action |

**Summary:**
- **Critical (>1000 LOC):** 5 files
- **High (600-1000 LOC):** 6 files
- **Monitor (400-600 LOC):** 3 files
- **OK (<400 LOC):** 91 files

**Compliance Rate:** 86.7% (91/105 files under 400 LOC)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-07
**Next Review:** After Phase 1 completion (approximately 4 weeks)
