# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models.

## Milestones

- **v1.0 Production-Ready** — Phases 1-12 + 12.1A + 12.1B (shipped 2026-01-19)
- **v1.1 Bug Fix Release** — Phases 13-01, 13-02, 13-03 (shipped 2026-01-19)
- **v1.2 Technical Debt Cleanup + Performance** — Phases 14-18 (shipped 2026-01-19)
- **v1.3 Test Health & Performance Validation** — Phases 19-21 (shipped 2026-01-20)

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
| 25 | v1.4 | 4/17 | Partial* | 2026-01-20 |

**Total Progress:** 150/164 plans complete (91%)

**Note:** Phase 21-06 (Performance Validation) skipped by user request. All test health goals (TEST-01 through TEST-06) achieved.

---

## Future Milestones

**v1.4**: Memory Safety + Code Restructure — IN PROGRESS 2026-01-20

### v1.4 Overview

**Focus**: Fix GPU memory allocation issues, remove dead/duplicate code, decompose all monolithic files

**Phases:**
- Phase 22: Memory Pool Implementation (Complete - fixes GPU hang on desktop)
- Phase 23: Dead/Duplicate Code Removal (Complete)
- Phase 24: Kernel-Centric Restructure (Complete)
- Phase 25: Architectural Decomposition (Planning complete, Gap Closure plans created)

**Total:** 4 phases, 34 plans (17 complete, 14 original planned, 3 gap closure planned)

**Rationale:** Research on 2026-01-20 identified critical issues:
1. **GPU hang risk**: Model loading uses 200-300 individual `hipMalloc` calls instead of memory pooling (llama.cpp analysis)
2. **Duplicate MXFP code**: `MxfpBlock` and `E8M0` defined in both `gguf.rs` and `mxfp.rs`
3. **File size issues**: `gguf.rs` is 2850 lines, loader total ~5000 lines - needs modularization
4. **AMD-only focus**: Confirm pure AMD HIP + CPU architecture, no CUDA dependencies

<details>
<summary>v1.4 Memory Safety + Code Restructure (Phase 22-25) — IN PROGRESS 2026-01-20</summary>

- [x] Phase 22: Memory Pool Implementation (5/5 plans) — **COMPLETE** 2026-01-20
- [x] Phase 23: Dead/Duplicate Code Removal (5/5 plans) — **COMPLETE** 2026-01-20
- [x] Phase 24: Kernel-Centric Restructure (6/6 plans) — **COMPLETE** 2026-01-20
- [ ] Phase 25: Architectural Decomposition (4/17 plans) — **PARTIAL** 2026-01-20

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
- [-] 25-05: Refactor Wave 2 - Execution/Mid-tier (5 of 8 targets complete)
- [x] 25-06: Refactor Wave 3 - Backend/Core (backend.rs → 10 modules)
- [x] 25-07: QA + Verification (partial completion verified)
- [ ] 25-08: Gap Closure - engine.rs decomposition (Wave 2A)
- [ ] 25-09: Gap Closure - scheduler/scheduler.rs decomposition (Wave 2A)
- [ ] 25-10: Gap Closure - ops/attention_gpu.rs decomposition (Wave 2A)
- [ ] 25-11: Gap Closure - kv_cache/kv_cache.rs further decomposition (Wave 2B)
- [ ] 25-12: Gap Closure - ggml/hip_backend/execution.rs further decomposition (Wave 2B)
- [ ] 25-13: Gap Closure - http/server.rs decomposition (Wave 4)
- [ ] 25-14: Gap Closure - profiling/rocprof_integration.rs decomposition (Wave 5)
- [ ] 25-15: Gap Closure - profiling/baseline.rs decomposition (Wave 5)
- [ ] 25-16: Gap Closure - backend/cpu/simd_ops.rs decomposition (Wave 6)
- [ ] 25-17: Gap Closure - backend/cpu/simd.rs decomposition (Wave 6)

**Status**: Phase 25 PARTIAL COMPLETE - 4 of 17 plans executed, 10 gap closure plans created
**Full details:** `.planning/milestones/v1.4-ROADMAP.md`

**Rationale for Phase 25**: 15 files remain over 1,000 LOC (4,243 LOC largest). Phase 24 addressed kernel files only; Phase 25 completes the decomposition for all remaining monolithic files.

**Phase 25 Progress:**
- ✓ loader/gguf.rs (2,284 LOC) → 8 modules
- ✓ execution_plan_src.rs (4,224 LOC) → 12 modules
- ✓ sampler/gpu.rs (1,858 LOC) → 4 modules
- ✓ backend/hip_backend/backend.rs (4,243 LOC) → 10 modules
- ⚠ kv_cache/kv_cache.rs partially decomposed (still 1,304 LOC) - 25-11 planned
- ⚠ ggml/hip_backend/execution.rs partially decomposed (still 1,207 LOC) - 25-12 planned
- ○ engine.rs (1,386 LOC) - 25-08 planned
- ○ scheduler/scheduler.rs (1,307 LOC) - 25-09 planned
- ○ ops/attention_gpu.rs (1,232 LOC) - 25-10 planned
- ○ http/server.rs (1,518 LOC) - 25-13 planned
- ○ profiling/rocprof_integration.rs (1,396 LOC) - 25-14 planned
- ○ profiling/baseline.rs (1,233 LOC) - 25-15 planned
- ○ backend/cpu/simd_ops.rs (1,198 LOC) - 25-16 planned
- ○ backend/cpu/simd.rs (1,093 LOC) - 25-17 planned

**Gap Closure Plans (Wave Structure):**
- Wave 2A: 25-08, 25-09, 25-10 (engine, scheduler, ops/attention_gpu) - 3 plans, can run in parallel
- Wave 2B: 25-11, 25-12 (kv_cache, ggml/execution) - 2 plans, can run in parallel
- Wave 4: 25-13 (http/server) - 1 plan
- Wave 5: 25-14, 25-15 (profiling) - 2 plans, can run in parallel
- Wave 6: 25-16, 25-17 (cpu/simd) - 2 plans, can run in parallel

**Remaining:** 12 files > 1,000 LOC (down from 15)

</details>
