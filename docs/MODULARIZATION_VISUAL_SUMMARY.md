# ROCmForge Modularization Visual Summary

## File Size Distribution

```
LOC Count    │ Files  │ Status
─────────────┼────────┼──────────────────────────────────────
>2000 LOC    │   2    │ ████████████████████████████████ CRITICAL
1500-2000    │   1    │ ██████████████████████ CRITICAL
1000-1500    │   2    │ ████████████████ CRITICAL
 600-1000    │   6    │ ██████████ HIGH PRIORITY
 300-600     │   2    │ ████ MONITOR
 <300        │  92    │ ✓ COMPLIANT
─────────────┼────────┼──────────────────────────────────────
TOTAL        │ 105    │ 86.7% COMPLIANT
```

## Critical Files Requiring Immediate Splits

```
┌─────────────────────────────────────────────────────────────────┐
│  src/model/execution_plan.rs          2,305 LOC (384% over)     │
│  ├─ Architecture detection                                       │
│  ├─ Weight mapping (5 types)                                     │
│  ├─ Forward pass execution                                       │
│  └─ Tensor utilities                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  src/backend/hip_backend.rs            2,135 LOC (356% over)     │
│  ├─ FFI bindings (extern "C")                                    │
│  ├─ Memory management (HipBuffer, HipStream)                      │
│  ├─ Kernel loading (HipModule, HipKernel)                        │
│  ├─ DeviceTensor operations                                      │
│  ├─ BLAS operations                                               │
│  ├─ MLP, LayerNorm, Transformer                                  │
│  └─ ModelRuntime (should be separate)                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  src/loader/gguf.rs                   1,950 LOC (325% over)     │
│  ├─ E8M0, MXFP data structures                                  │
│  ├─ GGUF parsing (KV pairs, tensor info)                         │
│  ├─ Dequantization (7 formats)                                   │
│  ├─ GPU upload with memory pooling                               │
│  └─ ModelConfig conversion                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  src/ops/attention_gpu.rs              1,222 LOC (204% over)     │
│  ├─ Kernel compilation & caching                                 │
│  ├─ QK^T computation (GEMM + CPU fallback)                       │
│  ├─ Causal masking (GPU + CPU fallback)                          │
│  ├─ Softmax (GPU + CPU fallback)                                 │
│  ├─ Attention-weighted V (GEMM + CPU fallback)                   │
│  └─ KV cache integration                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  src/attention/kernels.rs                955 LOC (159% over)     │
│  ├─ Global kernel cache                                           │
│  ├─ HSACO loading (10+ kernels)                                  │
│  └─ Individual kernel wrappers                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Refactoring Roadmap

```
Phase 1: CRITICAL (Weeks 1-4)
┌─────────────────────────────────────────────────────────────┐
│ Week 1-2  hip_backend.rs         ████████████████████ 20h    │
│ Week 2-3  execution_plan.rs      ████████████████ 16h       │
│ Week 3-4  gguf.rs                ██████████████ 14h         │
│ Week 4    attention_gpu.rs       ██████████ 10h             │
└─────────────────────────────────────────────────────────────┘

Phase 2: HIGH PRIORITY (Weeks 5-7)
┌─────────────────────────────────────────────────────────────┐
│ Week 5    kernels.rs             ████████████ 12h           │
│ Week 6    engine.rs              ████████ 8h                │
│ Week 6    server.rs              ██████ 6h                  │
│ Week 7    scheduler.rs           ██████ 6h                  │
│ Week 7    glm_position.rs        ████ 4h                    │
└─────────────────────────────────────────────────────────────┘

Phase 3: MONITORING (Week 8+)
┌─────────────────────────────────────────────────────────────┐
│ Review P2 files, documentation, CI/CD integration  8h       │
└─────────────────────────────────────────────────────────────┘
```

## Before & After Comparison

### BEFORE: Current State
```
src/
├── model/
│   ├── execution_plan.rs          2,305 LOC ✗ CRITICAL
│   ├── simple_transformer.rs       606 LOC ⚠ MONITOR
│   └── glm_position.rs             601 LOC ⚠ MONITOR
├── backend/
│   ├── hip_backend.rs            2,135 LOC ✗ CRITICAL
│   └── gpu_executor.rs             456 LOC ✓ OK
├── loader/
│   ├── gguf.rs                   1,950 LOC ✗ CRITICAL
│   └── onnx_loader.rs              246 LOC ✓ OK
├── ops/
│   └── attention_gpu.rs          1,222 LOC ✗ CRITICAL
├── attention/
│   ├── kernels.rs                   955 LOC ✗ CRITICAL
│   └── multi_query.rs               609 LOC ⚠ MONITOR
├── engine.rs                        803 LOC ✗ HIGH
├── http/
│   └── server.rs                    662 LOC ✗ HIGH
└── scheduler/
    └── scheduler.rs                 640 LOC ✗ HIGH

COMPLIANCE: 86.7% (92/105 files under 400 LOC)
```

### AFTER: Target State
```
src/
├── model/
│   ├── execution_plan/
│   │   ├── mod.rs                    150 LOC ✓
│   │   ├── plan.rs                   200 LOC ✓
│   │   ├── architecture.rs            180 LOC ✓
│   │   ├── forward.rs                 350 LOC ✓
│   │   ├── weights/
│   │   │   ├── mod.rs                100 LOC ✓
│   │   │   ├── embedding.rs          250 LOC ✓
│   │   │   ├── attention.rs          400 LOC ✓
│   │   │   ├── mlp.rs                300 LOC ✓
│   │   │   ├── layer_norm.rs         250 LOC ✓
│   │   │   └── transpose.rs          150 LOC ✓
│   │   └── layer_plan.rs             200 LOC ✓
│   ├── glm/
│   │   ├── transformer.rs           350 LOC ✓
│   │   └── position.rs              250 LOC ✓
│   └── simple_transformer.rs         606 LOC ⚠ Documented exception
├── backend/
│   ├── hip_backend/
│   │   ├── mod.rs                    150 LOC ✓
│   │   ├── ffi.rs                    400 LOC ✓
│   │   ├── device.rs                 250 LOC ✓
│   │   ├── memory.rs                 350 LOC ✓
│   │   ├── kernel.rs                 200 LOC ✓
│   │   ├── tensor.rs                 300 LOC ✓
│   │   ├── backend.rs                400 LOC ✓
│   │   └── ops/
│   │       ├── blas.rs               250 LOC ✓
│   │       ├── mlp.rs                300 LOC ✓
│   │       ├── layernorm.rs          250 LOC ✓
│   │       └── transformer.rs        350 LOC ✓
│   ├── model_runtime.rs               600 LOC ⚠ At hard limit
│   └── gpu_executor.rs                456 LOC ✓
├── loader/
│   ├── gguf/
│   │   ├── mod.rs                    150 LOC ✓
│   │   ├── types.rs                  250 LOC ✓
│   │   ├── metadata.rs               200 LOC ✓
│   │   ├── parser.rs                 400 LOC ✓
│   │   ├── dequant/
│   │   │   ├── mod.rs                100 LOC ✓
│   │   │   ├── q8.rs                 150 LOC ✓
│   │   │   ├── q4.rs                 200 LOC ✓
│   │   │   ├── q5.rs                 200 LOC ✓
│   │   │   └── mxfp.rs               300 LOC ✓
│   │   ├── upload.rs                 350 LOC ✓
│   │   ├── config.rs                 200 LOC ✓
│   │   └── loader.rs                 300 LOC ✓
│   └── onnx_loader.rs                 246 LOC ✓
├── ops/
│   ├── attention/
│   │   ├── mod.rs                    150 LOC ✓
│   │   ├── kernels.rs                300 LOC ✓
│   │   ├── qk.rs                     250 LOC ✓
│   │   ├── mask.rs                   200 LOC ✓
│   │   ├── softmax.rs                250 LOC ✓
│   │   ├── weighted_v.rs             250 LOC ✓
│   │   └── fallback.rs               300 LOC ✓
│   └── mod.rs                          14 LOC ✓
├── attention/
│   ├── kernels/
│   │   ├── mod.rs                    150 LOC ✓
│   │   ├── cache.rs                  300 LOC ✓
│   │   ├── scale.rs                  150 LOC ✓
│   │   ├── mask.rs                   150 LOC ✓
│   │   ├── softmax.rs                150 LOC ✓
│   │   ├── rope.rs                   150 LOC ✓
│   │   ├── matmul.rs                 200 LOC ✓
│   │   └── flash.rs                  300 LOC ✓
│   └── multi_query.rs                 609 LOC ⚠ Documented exception
├── engine/
│   ├── mod.rs                          150 LOC ✓
│   ├── config.rs                       150 LOC ✓
│   ├── engine.rs                       300 LOC ✓
│   ├── model.rs                        250 LOC ✓
│   ├── inference.rs                    300 LOC ✓
│   └── request.rs                      250 LOC ✓
├── http/
│   ├── mod.rs                          150 LOC ✓
│   ├── server.rs                       300 LOC ✓
│   ├── routes.rs                       400 LOC ✓
│   └── dto.rs                          200 LOC ✓
├── scheduler/
│   ├── mod.rs                            5 LOC ✓
│   ├── config.rs                       150 LOC ✓
│   ├── scheduler.rs                    300 LOC ✓
│   ├── batching.rs                     250 LOC ✓
│   └── state.rs                        200 LOC ✓
└── ... (other files unchanged)

COMPLIANCE TARGET: 95%+ (all files under 600 LOC, most under 400 LOC)
```

## Risk Heatmap

```
Risk Level    Files  Hours
───────────── ─────  ─────
CRITICAL         1     20  (hip_backend.rs - FFI, GPU memory)
HIGH             2     30  (execution_plan.rs, gguf.rs)
MEDIUM           3     34  (attention_gpu.rs, kernels.rs, +1 other)
LOW              5     24  (engine.rs, server.rs, scheduler.rs, etc.)
───────────── ─────  ─────
TOTAL           11    108  (excluding monitoring)
```

## Key Success Metrics

```
Before Refactoring          After Refactoring (Target)
─────────────────          ──────────────────────────
5 files >1000 LOC          0 files >600 LOC
6 files >600 LOC           <5 files >300 LOC
86.7% compliance          95%+ compliance
Unclear concerns          Clear separation of concerns
Hard to navigate          Easy to find code
Difficult to test         Modules testable in isolation
```

---

**Generated:** 2026-01-07
**Full Report:** ARCHITECTURE_MODULARIZATION_ANALYSIS.md
