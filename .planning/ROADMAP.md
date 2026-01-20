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

**Total Progress:** 146/147 plans complete (99%)

**Note:** Phase 21-06 (Performance Validation) skipped by user request. All test health goals (TEST-01 through TEST-06) achieved.

---

## Future Milestones

**v1.4**: Memory Safety + Code Restructure — SHIPPED 2026-01-20

### v1.4 Overview

**Focus**: Fix GPU memory allocation issues, remove dead/duplicate code, restructure for maintainability

**Phases:**
- Phase 22: Memory Pool Implementation (Critical - fixes GPU hang on desktop)
- Phase 23: Dead/Duplicate Code Removal
- Phase 24: Kernel-Centric Restructure

**Total:** 3 phases, ~15-20 plans

**Rationale:** Research on 2026-01-20 identified critical issues:
1. **GPU hang risk**: Model loading uses 200-300 individual `hipMalloc` calls instead of memory pooling (llama.cpp analysis)
2. **Duplicate MXFP code**: `MxfpBlock` and `E8M0` defined in both `gguf.rs` and `mxfp.rs`
3. **File size issues**: `gguf.rs` is 2850 lines, loader total ~5000 lines - needs modularization
4. **AMD-only focus**: Confirm pure AMD HIP + CPU architecture, no CUDA dependencies

<details>
<summary>v1.4 Memory Safety + Code Restructure (Phase 22-24) — IN PROGRESS 2026-01-20</summary>

- [x] Phase 22: Memory Pool Implementation (5/5 plans) — **COMPLETE** 2026-01-20
- [x] Phase 23: Dead/Duplicate Code Removal (5/5 plans) — **COMPLETE** 2026-01-20
- [ ] Phase 24: Kernel-Centric Restructure (0/6 plans)

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
- 24-01: Create new `src/kernels/` directory layout
- 24-02: Split quantization into separate modules (q4_0.rs, q4_k.rs, q6_k.rs, fp16.rs)
- 24-03: Move attention kernels to `src/kernels/attention/`
- 24-04: Move matmul kernels to `src/kernels/matmul/`
- 24-05: Enforce 600-1000 LOC per file limit
- 24-06: Verification: All files under limit, clear module boundaries

**Status**: Planning complete, ready to execute
**Full details:** `.planning/milestones/v1.4-ROADMAP.md`

</details>
