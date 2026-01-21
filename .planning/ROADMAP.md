# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models.

## Milestones

- **v1.0 Production-Ready** — Phases 1-12 + 12.1A + 12.1B (shipped 2026-01-19)
- **v1.1 Bug Fix Release** — Phases 13-01, 13-02, 13-03 (shipped 2026-01-19)
- **v1.2 Technical Debt Cleanup + Performance** — Phases 14-18 (shipped 2026-01-19)
- **v1.3 Test Health & Performance Validation** — Phases 19-21 (shipped 2026-01-20)
- **v1.4 Memory Safety + Code Restructure** — Phases 22-24 (shipped 2026-01-20)
- **v1.5 Env Var & Transpose Fix** — Phases 25-29 (shipped 2026-01-21)
- **v1.6 FFI Device Props Fix** — Phases 30-32 (planned)

## Phases

<details>
<summary>v1.0-v1.5 (Phases 1-29) — SHIPPED 2026-01-21</summary>

**Full details archived in:** [.planning/milestones/](.planning/milestones/)

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
- [x] Phase 13-01: Qwen2 head_dim Fix (1/1 plans)
- [x] Phase 13-02: Memory Pooling Documentation (1/1 plans)
- [x] Phase 13-03: Dead Code Removal (4/4 plans)
- [x] Phase 14: Scheduler Clone Bug Fix (2/2 plans)
- [x] Phase 15: GPU Sampling Kernels (7/7 plans)
- [x] Phase 16: GPU RoPE Implementation (2/2 plans)
- [x] Phase 17: GPU Quantization (3/3 plans)
- [x] Phase 18: GPU Attention Completion (3/3 plans)
- [x] Phase 19: Wavefront-Native Quantized Matmul (4/4 plans)
- [x] Phase 20: Code Hygiene Completion (8/8 plans)
- [x] Phase 21: Test Health & Performance (5/6 plans)
- [x] Phase 22: Memory Pool Implementation (5/5 plans)
- [x] Phase 23: Dead/Duplicate Code Removal (5/5 plans)
- [x] Phase 24: Kernel-Centric Restructure (6/6 plans)
- [x] Phase 25: Env Var Fix (12/12 plans)
- [x] Phase 26: Transpose Kernel Fix (3/3 plans)
- [x] Phase 27: Device Property Infrastructure (4/4 plans)
- [x] Phase 28: Debug Hygiene (4/4 plans)
- [x] Phase 29: Validation & E2E (5/5 plans)

**Total:** 195 plans across 29 phases

</details>

---

## v1.6 FFI Device Props Fix (Planned)

**Milestone Goal:** Fix FFI device properties bug causing "block.y exceeds limit 0" errors during kernel launch.

**Issue:** The FFI device property sanity check only validates `max_threads_dim[0] > 0`, allowing garbage values like `[1024, 0, 0]` to pass. Later kernel launch validation fails because `block.y=1 exceeds limit 0`.

**Root Causes (from research/SUMMARY.md):**
- Incomplete sanity check: Only validates `dim[0]`, allowing `dim[1]` or `dim[2]` to be zero
- Duplicate DeviceLimits assignment: Lines 279-289 overwrite vetted values with undefined `max_grid`
- Hardcoded struct offsets: Manual offsets break across ROCm versions as struct grows

### Phase 30: Immediate Bugfix

**Goal**: Device properties sanity check validates ALL dimensions before use.

**Depends on**: Phase 29 complete

**Requirements**: FFI-01, FFI-02

**Success Criteria** (what must be TRUE):
1. Sanity check validates all three dimensions (X, Y, Z) of maxThreadsDim are greater than 0
2. Sanity check validates all three grid dimensions are greater than 0
3. Sanity check validates warp size is 32 or 64 (valid AMD GPU values)
4. Sanity check validates shared memory is in range (0 < shared <= 1MB)
5. Sanity check validates max threads per block is in range (0 < max_tb <= 4096)
6. When checks fail, warning is logged with actual values before using safe defaults
7. Only ONE DeviceLimits construction exists (duplicate deleted)

**Plans**: 1 plan (1 wave)
- [x] 30-01-PLAN.md — Comprehensive sanity check + delete duplicate DeviceLimits

### Phase 31: Bindgen Infrastructure

**Goal**: Compile-time HIP bindings generation for offset verification.

**Depends on**: Phase 30 complete

**Requirements**: FFI-03

**Success Criteria** (what must be TRUE):
1. bindgen 0.70 added to build-dependencies in Cargo.toml
2. build.rs generates hip_device_bindings.rs with hipDeviceProp_t struct only
3. Generated bindings are accessible via include!() in test code
4. Existing FFI declarations in ffi.rs remain unchanged (no 10,000+ line replacement)
5. cargo build successfully generates bindings without errors

**Plans**: 1 plan (1 wave)
- [x] 31-01-PLAN.md — Add bindgen infrastructure with HIP allowlist

### Phase 32: Offset Verification Test

**Goal**: Compile-time test asserts manual offsets match bindgen-generated offsets.

**Depends on**: Phase 31 complete

**Requirements**: FFI-04

**Success Criteria** (what must be TRUE):
1. Test module created in device.rs (test-only code with #[cfg(test)])
2. Test uses include!() to load bindgen-generated bindings
3. Test asserts TOTAL_GLOBAL_MEM_OFFSET matches bindgen offsetof
4. Test asserts MAX_THREADS_PER_BLOCK_OFFSET matches bindgen offsetof
5. Test asserts MAX_THREADS_DIM_OFFSET matches bindgen offsetof
6. Test runs with `cargo test --lib device::offset_verification`
7. Test fails with clear message if ROCm version changes struct layout

**Plans**: 1 plan (1 wave)
- [ ] 32-01-PLAN.md — Create offset verification test module with memoffset dependency

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
| 25 | v1.5 | 12/12 | Complete | 2026-01-21 |
| 26 | v1.5 | 3/3 | Complete | 2026-01-21 |
| 27 | v1.5 | 4/4 | Complete | 2026-01-21 |
| 28 | v1.5 | 4/4 | Complete | 2026-01-21 |
| 29 | v1.5 | 5/5 | Complete | 2026-01-21 |
| 30 | v1.6 | 1/1 | Complete | 2026-01-21 |
| 31 | v1.6 | 1/1 | Complete | 2026-01-21 |
| 32 | v1.6 | 0/1 | Planned | - |

**Total Progress:** 197/198 plans complete (v1.0-v1.5 + v1.6 phases 30-31)

**Note:** Phase 21-06 (Performance Validation) skipped by user request. All test health goals (TEST-01 through TEST-06) achieved.

---

## Session Continuity

Last session: 2026-01-21
Stopped at: Phase 32 planned (Offset Verification Test)

**Milestone v1.5 COMPLETE!** Ready to begin v1.6 planning.

**v1.6 - FFI Device Props Fix (2026-01-21):**
- Phase 30: Immediate Bugfix (1 plan) - COMPLETE
- Phase 31: Bindgen Infrastructure (1 plan) - COMPLETE
- Phase 32: Offset Verification Test (1 plan) - PLANNED

**Coverage:** 4/4 requirements mapped to phases (100%)
