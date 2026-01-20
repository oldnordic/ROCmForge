# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models.

## Milestones

- **v1.0 Production-Ready** — Phases 1-12 + 12.1A + 12.1B (shipped 2026-01-19)
- **v1.1 Bug Fix Release** — Phases 13-01, 13-02, 13-03 (shipped 2026-01-19)
- **v1.2 Technical Debt Cleanup + Performance** — Phases 14-18 (shipped 2026-01-19)
- **v1.3 Test Health & Performance Validation** — Phases 19-21 (shipped 2026-01-20)
- **v1.4 Memory Safety + Code Restructure** — Phases 22-26 (shipped 2026-01-20)
- **v1.5 GPU Transpose Fix** — Phase 27 (shipped 2026-01-20)
- **v1.6 ROCm Feature Fix** — Phase 28 (in progress 2026-01-20)

## Phases

<details>
<summary>v1.0 Production-Ready (Phases 1-12 + 12.1A + 12.1B) — SHIPPED 2026-01-19</summary>

**Full details archived in:** [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

- [x] Phase 1: Critical Bug Fixes (3/3 plans)
- [x] Phase 2: Test Infrastructure (4/4 plans)
- [x] Phase 3: Codebase Modularization (4/4 plans)
- [x] Phase 4: CPU SIMD Backend (4/4 plans)
- [x] Phase 5: Quantized Operations (4/4 plans)
- [x] Phase 6: Attention Optimization (4/4 plans)
- [x] Phase 7: Hybrid Execution Scheduler (4/4 plans)
- [x] Phase 8: GGUF Compatibility (11/11 plans)
- [x] Phase 9: Performance Optimization (18/18 plans)
- [x] Phase 10: Production Hardening (20/20 plans)
- [x] Phase 11: Fix Test Suite & Verify E2E (2/2 plans)
- [x] Phase 12: Complete CPU SIMD Attention (4/4 plans)
- [x] Phase 12.1A: CPU SIMD Completion (2/2 plans)
- [x] Phase 12.1B: Context Engine Integration (1/1 plan)

**Total:** 96 plans across 13 phases

</details>

<details>
<summary>v1.1 Bug Fix Release (Phases 13-01, 13-02, 13-03) — SHIPPED 2026-01-19</summary>

**Full details archived in:** [.planning/milestones/v1.1-ROADMAP.md](.planning/milestones/v1.1-ROADMAP.md)

- [x] Phase 13-01: Qwen2 head_dim Fix (1/1 plans)
- [x] Phase 13-02: Memory Pooling Documentation (1/1 plans)
- [x] Phase 13-03: Dead Code Removal (4/4 plans)

**Total:** 6 plans across 3 phases

**Summary:** Fixed Qwen2 model loading, corrected documentation, reduced compiler warnings 93%

</details>

<details>
<summary>v1.2 Technical Debt Cleanup + Performance (Phases 14-18) — SHIPPED 2026-01-19</summary>

**Full details archived in:** [.planning/milestones/v1.2-ROADMAP.md](.planning/milestones/v1.2-ROADMAP.md)

- [x] Phase 14: Scheduler Clone Bug Fix (2/2 plans)
- [x] Phase 15: GPU Sampling Kernels (7/7 plans)
- [x] Phase 16: GPU RoPE Implementation (2/2 plans)
- [x] Phase 17: GPU Quantization (3/3 plans)
- [x] Phase 18: GPU Attention Completion (3/3 plans)

**Total:** 17 plans across 5 phases

**Summary:** GPU sampling (top-k, top-p, temperature), GPU RoPE, GPU quantization (Q4_0, Q4_K, Q6_K), GPU attention (FlashAttention, MQA, GQA) - all running on GPU with pure HIP-native kernels.

</details>

<details>
<summary>v1.3 Test Health & Performance Validation (Phases 19-21) — SHIPPED 2026-01-20</summary>

**Full details archived in:** [.planning/milestones/v1.3-ROADMAP.md](.planning/milestones/v1.3-ROADMAP.md)

- [x] Phase 19: Wavefront-Native Quantized Matmul (4/4 plans)
- [x] Phase 20: Code Hygiene Completion (8/8 plans)
- [x] Phase 21: Test Health & Performance (5/6 plans, 21-06 skipped)

**Total:** 17 plans across 3 phases

**Summary:** Eliminated all CUDA intrinsics from quantized kernels, achieved zero compiler warnings baseline, fixed all broken tests, implemented graceful skip pattern for CI/CD without GPU kernels. Performance validation (21-06) skipped by user.

</details>

---

## Progress

**Execution Order:** Phases execute in numeric order

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-12 + 12.1A + 12.1B | v1.0 | 96/96 | Complete | 2026-01-18 |
| 13-01, 13-02, 13-03 | v1.1 | 6/6 | Complete | 2026-01-18 |
| 14 | v1.2 | 2/2 | Complete | 2026-01-19 |
| 15 | v1.2 | 7/7 | Complete | 2026-01-19 |
| 16 | v1.2 | 2/2 | Complete | 2026-01-19 |
| 17 | v1.2 | 3/3 | Complete | 2026-01-19 |
| 18 | v1.2 | 3/3 | Complete | 2026-01-19 |
| 19 | v1.3 | 4/4 | Complete | 2026-01-19 |
| 20 | v1.3 | 8/8 | Complete | 2026-01-19 |
| 21 | v1.3 | 5/6 | Complete* | 2026-01-20 |
| 22 | v1.4 | 5/5 | Complete | 2026-01-20 |
| 23 | v1.4 | 5/5 | Complete | 2026-01-20 |
| 24 | v1.4 | 6/6 | Complete | 2026-01-20 |
| 25 | v1.4 | 17/17 | Complete | 2026-01-20 |
| 26 | v1.4 | 4/4 | Complete | 2026-01-20 |
| 27 | v1.5 | 4/4 | Complete | 2026-01-20 |
| 28 | v1.6 | 0/3 | In Progress | 2026-01-20 |

**Total Progress:** 171/174 plans complete (98%)

**Note:** Phase 21-06 (Performance Validation) skipped by user request. All test health goals (TEST-01 through TEST-06) achieved.

---

**v1.6**: ROCm Feature Fix — IN PROGRESS 2026-01-20

### v1.6 Summary

**Focus:** Remove incorrect `rocm` feature flag - ROCm/HIP is core to ROCmForge

**Issue:** 155 compilation errors when `--features rocm` is enabled. GPU code paths accumulated bit-rot from lack of compilation.

**Root Cause:** Codebase was built/tested without `rocm` feature enabled. GPU-specific code wrapped in `#[cfg(feature = "rocm")]` was never compiled.

**Solution:** Remove the `rocm` feature entirely. ROCm/HIP is always enabled.

**Plans:**
- Phase 28: ROCm Feature Fix (3 plans)

<details>
<summary>Phase 28: ROCm Compilation Fix (3 plans)</summary>

**Goal:** Remove the incorrect `rocm` feature flag entirely - ROCm/HIP is core to ROCmForge, NOT optional.

**Plans:**
- [ ] 28-01: Fix missing imports and parameter names
- [ ] 28-02: Remove all #[cfg(feature = "rocm")] gates
- [ ] 28-03: Remove rocm feature from Cargo.toml and build.rs

**Implementation Order (from research):**
1. Fix missing imports (c_void, HipError, Path, Mutex)
2. Fix underscore-prefixed parameters
3. Fix type annotations
4. Fix missing mut keyword
5. Remove #[cfg(feature = "rocm")] gates
6. Remove rocm feature from Cargo.toml
7. Update build.rs to always compile kernels

</details>

---

**v1.5**: GPU Transpose Fix — COMPLETE ✓ 2026-01-20

### v1.5 Summary

**Focus:** Fix GPU transpose crash that prevents inference

**Issue:** After Phase 22 memory pool implementation, GPU→CPU copy fails during embedding transpose with `hipMemcpyDtoH failed with code 1`

**Root Cause:** `transpose_2d_tensor()` uses CPU transpose (GPU→CPU→GPU round-trip) which fails with arena sub-buffers at large offsets

**Solution:** Implemented pure GPU transpose kernel based on AMD ROCm Examples

**Files Changed:**
- `src/kernels/transpose/mod.rs` (472 LOC) - TransposeKernel with lazy HSACO loading
- `kernels/transpose.hip` (131 LOC) - HIP kernel with TILE_DIM=64, shared memory padding
- `build.rs` - TRANSPOSE_HSACO compilation entry
- `src/model/execution_plan/types.rs:273` - Uses GPU transpose

**Before:**
```
ERROR: hipMemcpyDtoH failed with code 1 (offset=1.97GB into arena)
```

**After:**
```
INFO: Transposing embedding tensor from [896, 151936] to [151936, 896] on GPU
INFO: Embedding weights loaded successfully
```

**Phases:**
- Phase 27: GPU Transpose Kernel Implementation (COMPLETE - 4/4 plans)

---

**v1.4**: Memory Safety + Code Restructure — COMPLETE ✓ 2026-01-20

### v1.4 Overview

**Focus**: Fix GPU memory allocation issues, remove dead/duplicate code, decompose all monolithic files, eliminate compiler warnings

**Phases:**
- Phase 22: Memory Pool Implementation (Complete - fixes GPU hang on desktop)
- Phase 23: Dead/Duplicate Code Removal (Complete)
- Phase 24: Kernel-Centric Restructure (Complete)
- Phase 25: Architectural Decomposition (Complete - all 17 plans executed)
- Phase 26: Compiler Warning Cleanup (Complete - zero warnings baseline)

**Total:** 5 phases, 37 plans (all complete)

**Rationale:** Research on 2026-01-20 identified critical issues:
1. **GPU hang risk**: Model loading uses 200-300 individual `hipMalloc` calls instead of memory pooling (llama.cpp analysis)
2. **Duplicate MXFP code**: `MxfpBlock` and `E8M0` defined in both `gguf.rs` and `mxfp.rs`
3. **File size issues**: `gguf.rs` is 2850 lines, loader total ~5000 lines - needs modularization
4. **AMD-only focus**: Confirm pure AMD HIP + CPU architecture, no CUDA dependencies
5. **Compiler warnings**: Phase 25 decomposition introduced 42 compiler warnings across 5 categories

<details>
<summary>v1.4 Memory Safety + Code Restructure (Phase 22-26) — COMPLETE ✓ 2026-01-20</summary>

- [x] Phase 22: Memory Pool Implementation (5/5 plans) — **COMPLETE** 2026-01-20
- [x] Phase 23: Dead/Duplicate Code Removal (5/5 plans) — **COMPLETE** 2026-01-20
- [x] Phase 24: Kernel-Centric Restructure (6/6 plans) — **COMPLETE** 2026-01-20
- [x] Phase 25: Architectural Decomposition (17/17 plans) — **COMPLETE** 2026-01-20
- [x] Phase 26: Compiler Warning Cleanup (4/4 plans) — **COMPLETE** 2026-01-20

**Phase 22: Memory Pool Implementation**
- [x] 22-01: Implement `ModelWeightArena` for single-allocation model loading
- [x] 22-02: Memory requirements calculation before GPU allocation
- [x] 22-03: Replace 200+ `HipBuffer::new()` calls with arena pattern
- [x] 22-04: Update `load_to_gpu_async()` to use memory pool
- [x] 22-05: Verification: Model loads without GPU hang

**Phase 23: Dead/Duplicate Code Removal**
- [x] 23-01: Remove duplicate MXFP code from `gguf.rs` (keep `mxfp.rs` version)
- [x] 23-02: Remove unused quantization formats (Q4_1, Q5_0, Q5_1)
- [x] 23-03: Consolidate `GgufMetadata` structs
- [x] 23-04: Clean up `ParallelResult` and unused async loading code
- [x] 23-05: Verification: No duplicate structures, all tests pass

**Phase 24: Kernel-Centric Restructure**
- [x] 24-01: Create new `src/kernels/` directory layout
- [x] 24-02: Split quantization into separate modules (q4_0.rs, q4_k.rs, q6_k.rs, fp16.rs)
- [x] 24-03: Move attention kernels to `src/kernels/attention/`
- [x] 24-04: Move matmul kernels to `src/kernels/matmul/`
- [x] 24-05: Enforce 600-1000 LOC per file limit
- [x] 24-06: Verification: All kernel files under limit, clear module boundaries

**Phase 25: Architectural Decomposition**
- [x] 25-01: Magellan code mapping (symbol clusters, dependencies)
- [x] 25-02: Responsibility analysis (domain concern grouping)
- [x] 25-03: Module boundary proposal (decomposition map)
- [x] 25-04: Refactor Wave 1 - Loader (gguf.rs → 8 modules)
- [x] 25-05: Refactor Wave 2 - Execution/Mid-tier (5 of 8 targets complete)
- [x] 25-06: Refactor Wave 3 - Backend/Core (backend.rs → 10 modules)
- [x] 25-07: QA + Verification (partial completion verified)
- [x] 25-08: Gap Closure - engine.rs decomposition (Wave 2A)
- [x] 25-09: Gap Closure - scheduler/scheduler.rs decomposition (Wave 2A)
- [x] 25-10: Gap Closure - ops/attention_gpu.rs decomposition (Wave 2A)
- [x] 25-11: Gap Closure - kv_cache/kv_cache.rs further decomposition (Wave 2B)
- [x] 25-12: Gap Closure - ggml/hip_backend/execution.rs further decomposition (Wave 2B)
- [x] 25-13: Gap Closure - http/server.rs decomposition (Wave 4)
- [x] 25-14: Gap Closure - profiling/rocprof_integration.rs decomposition (Wave 5)
- [x] 25-15: Gap Closure - profiling/baseline.rs decomposition (Wave 5)
- [x] 25-16: Gap Closure - backend/cpu/simd_ops.rs decomposition (Wave 6)
- [x] 25-17: Gap Closure - backend/cpu/simd.rs decomposition (Wave 6)

**Phase 26: Compiler Warning Cleanup**
- [x] 26-01: Fix deprecated dequant functions and unused imports
- [x] 26-02: Migrate to_host_vec() to copy_from_device_safe()
- [x] 26-03: Suppress dead code warnings with justification
- [x] 26-04: Fix KernelCache visibility mismatch, verify zero warnings

**Status**: Phase 26 COMPLETE - Zero compiler warnings achieved (42 → 0)
**Full details:** `.planning/phases/26-warning-cleanup/26-RESEARCH.md`

**Rationale for Phase 26**: Phase 25's architectural decomposition (110+ new modules) introduced 42 compiler warnings across 5 categories. These warnings need systematic elimination to restore the zero warnings baseline established in Phase 20.

**Phase 25 Results:**
- ✓ loader/gguf.rs (2,284 LOC) → 8 modules
- ✓ execution_plan_src.rs (4,224 LOC) → 12 modules
- ✓ sampler/gpu.rs (1,858 LOC) → 4 modules
- ✓ backend/hip_backend/backend.rs (4,243 LOC) → 10 modules
- ✓ engine.rs (1,386 LOC) → 5 modules in engine/ directory
- ✓ scheduler/scheduler.rs (1,307 LOC) → 4 modules
- ✓ ops/attention_gpu.rs (1,232 LOC) → 4 modules in ops/attention/
- ✓ kv_cache/kv_cache.rs (1,304 LOC → 1,032 LOC, 3 new modules)
- ✓ ggml/hip_backend/execution.rs (1,207 LOC → 84 LOC, op_dispatch.rs created)
- ✓ http/server.rs (1,518 LOC) → 5 modules in http/ directory
- ✓ profiling/rocprof_integration.rs (1,396 LOC) → 3 modules
- ✓ profiling/baseline.rs (1,233 LOC) → 3 modules
- ✓ backend/cpu/simd_ops.rs (1,198 LOC) → 4 modules in simd_ops/
- ✓ backend/cpu/simd.rs (1,093 LOC) → 3 modules in simd/

**Decomposition Summary:**
- 110+ new modules created
- Original files >1,000 LOC: 15 → 2 (87% reduction)
- Largest file reduced: 4,243 → 1,232 LOC (-71%)
- All modules under 1,200 LOC with focused responsibilities
- Re-export chains preserve backward compatibility
- 701 lib tests passing (+26 from baseline)

**Remaining:** 2 original files >1,000 LOC (legacy, acceptable):
- kv_cache/kv_cache.rs (1,032 LOC) - main implementation
- ops/attention_gpu.rs (1,232 LOC) - legacy facade, can be removed

**Phase 26 Warning Cleanup Goals:**
- Fix 8 deprecated dequant function warnings (loader module)
- Fix 13 deprecated to_host_vec() warnings (model execution, attention)
- Fix 16 dead code warnings with #[allow(dead_code)] justification
- Fix 1 visibility mismatch warning (KernelCache)
- Fix 2 unused import warnings
- Target: Zero compiler warnings baseline (42 → 0)

</details>
