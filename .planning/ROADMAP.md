# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models.

## Milestones

- **v1.0 Production-Ready** — Phases 1-12 + 12.1A + 12.1B (shipped 2026-01-19)
- **v1.1 Bug Fix Release** — Phases 13-01, 13-02, 13-03 (shipped 2026-01-19)
- **v1.2 Technical Debt Cleanup + Performance** — Phases 14-18 (shipped 2026-01-19)
- **v1.3 Test Health & Performance Validation** — Phases 19-21 (shipped 2026-01-20)
- **v1.4 Memory Safety + Code Restructure** — Phases 22-24 (shipped 2026-01-20)
- **v1.5 Env Var & Transpose Fix** — Phases 25-29 (planned)

## Phases

<details>
<summary>v1.0-v1.4 (Phases 1-24) — SHIPPED 2026-01-20</summary>

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

**Total:** 168 plans across 24 phases

</details>

---

## v1.5 Env Var & Transpose Fix (Planned)

**Milestone Goal:** Fix runtime kernel loading and large tensor transpose to enable actual GGUF model inference.

**Issue:** After Phase 24 kernel-centric restructuring, two critical bugs block actual model inference:
1. **Env var embedding mismatch**: `build.rs` sets compile-time env vars but runtime code uses `std::env::var()` - paths not embedded in binary
2. **Transpose kernel failure**: `block=(64,64,1)` = 4096 threads exceeds `maxThreadsPerBlock=1024` limit for large tensors

**Root Causes (from research/SUMMARY_v1.5.md):**
- Env var bug: `cargo:rustc-env=VAR=path` at compile time vs `std::env::var("VAR")` runtime lookup returns None
- Transpose bug: TILE_DIM=64 creates 64x64x1=4096 threads per block, exceeding AMD GPU limit of 1024

### Phase 25: Env Var Fix

**Goal**: HSACO kernel paths are embedded in binary at compile time, not read from runtime environment.

**Depends on**: Phase 24 complete

**Requirements**: ENV-01, ENV-02, ENV-03

**Success Criteria** (what must be TRUE):
1. User can run `./target/release/rocmforge --help` without manually setting environment variables
2. All 31 HSACO kernel paths use `option_env!()` compile-time macro instead of `std::env::var()` runtime lookup
3. Missing kernel file produces clear error message with full path to compiled-in HSACO location
4. `cargo build --release` recompiles build.rs when ROCM_PATH, HIPCC, or ROCm_ARCH environment variables change

**Plans**: 12 plans (1 wave, all parallel)
- [x] 25-01-PLAN.md — Fix attention kernels (12 kernels) to use `option_env!()`
- [x] 25-02-PLAN.md — Fix sampler kernels (7 kernels) to use `option_env!()`
- [x] 25-03-PLAN.md — Fix MLP kernels (2 kernels) to use `option_env!()`
- [x] 25-04-PLAN.md — Fix quantization kernels (2 kernels) to use `option_env!()`
- [x] 25-05-PLAN.md — Fix fused kernels (2 kernels) to use `option_env!()`
- [x] 25-06-PLAN.md — Fix transpose kernel (1 kernel) to use `option_env!()`
- [x] 25-07-PLAN.md — Add cargo rerun directives for ROCm env vars
- [x] 25-08-PLAN.md — Fix Q4_0_DEQUANT_HSACO (duplicate ggml/hip_backend impl)
- [x] 25-09-PLAN.md — Fix Q4_K_DEQUANT_HSACO (duplicate ggml/hip_backend impl)
- [x] 25-10-PLAN.md — Fix Q4_0_MATMUL_HSACO (quantized matmul)
- [x] 25-11-PLAN.md — Fix Q4_K_MATMUL_HSACO (quantized matmul)
- [x] 25-12-PLAN.md — Fix Q6_K_MATMUL_HSACO (quantized matmul)

### Phase 26: Transpose Kernel Fix

**Goal**: Transpose kernel launches with valid block dimensions that respect AMD GPU limits.

**Depends on**: Phase 25 complete (kernel loading must work first)

**Requirements**: TRN-01, TRN-02, TRN-03

**Success Criteria** (what must be TRUE):
1. Transpose of tensor shape [896, 151936] completes without `hipErrorInvalidValue`
2. Block dimensions changed from (64,64,1) to (32,32,1) with 1024 threads per block (within limit)
3. Grid dimensions validated as non-zero and within maxGridSize limits before kernel launch
4. Shared memory size (16400 bytes) verified to be under sharedMemPerBlock limit (65536 bytes)

**Plans**: 3 plans (2 waves)
- [x] 26-01-PLAN.md — Fix block dimension to (32, 32, 1) for 1024 threads (Wave 1)
- [x] 26-02-PLAN.md — Add validation assertions for grid/block/shared memory (Wave 2)
- [x] 26-03-PLAN.md — Add unit test for [896, 151936] tensor transpose (Wave 2)

### Phase 27: Device Property Infrastructure

**Goal**: GPU device properties queried once at backend init and used for launch validation.

**Depends on**: Phase 26 complete (validation needs working transpose)

**Requirements**: DEV-01, DEV-02, DEV-03

**Success Criteria** (what must be TRUE):
1. `hipGetDeviceProperties` called once during `HipBackend::new()` and cached in backend struct
2. Cached properties include maxThreadsPerBlock, maxGridSize[3], warpSize, sharedMemPerBlock
3. All kernel launches assert block dimensions within cached limits before calling `hipLaunchKernel`
4. Grid calculations use u64 arithmetic to prevent overflow for very large tensors

**Plans**: 4 plans (3 waves)
- [x] 27-01-PLAN.md — Extend HipDeviceProp with launch limit accessors (Wave 1)
- [x] 27-02-PLAN.md — Add DeviceLimits struct and cache in HipBackend (Wave 1)
- [x] 27-03-PLAN.md — Add validation methods and safe grid helpers (Wave 2)
- [x] 27-04-PLAN.md — Update kernel launch sites to use cached limits (Wave 3)

### Phase 28: Debug Hygiene

**Goal**: HIP error logging and debug instrumentation for faster kernel troubleshooting.

**Depends on**: Phase 27 complete (device props enable better error messages)

**Requirements**: DBG-01, DBG-02, DBG-03

**Success Criteria** (what must be TRUE):
1. Kernel launch failures log HIP error string from `hipGetLastError` with grid/block dimensions
2. Debug builds log computed grid/block dimensions and shared memory size before each kernel launch
3. Setting `HIP_LAUNCH_BLOCKING=1` enables synchronous kernel execution for debugging
4. Developer documentation updated with HIP debugging procedures

**Plans**: 4 plans (2 waves)
- [ ] 28-01-PLAN.md — Store kernel name in HipKernel for error messages (Wave 1)
- [ ] 28-02-PLAN.md — Add hipGetLastError async error checking after launch (Wave 1)
- [ ] 28-03-PLAN.md — Add debug logging and HIP_LAUNCH_BLOCKING support (Wave 1)
- [ ] 28-04-PLAN.md — Create HIP debugging documentation (Wave 2)

### Phase 29: Validation & E2E

**Goal**: End-to-end verification that qwen2.5-0.5b.gguf model loads and generates tokens successfully.

**Depends on**: Phase 25, 26, 27, 28 complete

**Requirements**: VAL-01, VAL-02, VAL-03

**Success Criteria** (what must be TRUE):
1. Unit test for transpose with shape [896, 151936] passes and matches CPU reference result
2. `rocmforge load --model qwen2.5-0.5b.gguf` loads model without KernelLoadFailed errors
3. `rocmforge generate --model qwen2.5-0.5b.gguf --prompt "The"` generates a single token without errors
4. Embedding weights transpose [896, 151936] completes during model loading

**Plans**: TBD

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
| 28 | v1.5 | 0/4 | Not started | - |
| 29 | v1.5 | 0/0 | Not started | - |

**Total Progress:** 187/194 plans complete (v1.0-v1.4 + Phases 25-27), v1.5 Phases 28-29 remaining

**Note:** Phase 21-06 (Performance Validation) skipped by user request. All test health goals (TEST-01 through TEST-06) achieved.

---

## Session Continuity

Last session: 2026-01-21
Stopped at: Phase 27 execution complete

**v1.5 - Env Var & Transpose Fix (2026-01-21):**
- Phase 25: Env Var Fix (12/12 complete) ✓
- Phase 26: Transpose Kernel Fix (3/3 complete) ✓
- Phase 27: Device Property Infrastructure (4/4 complete) ✓
- Phase 28: Debug Hygiene (Not started)
- Phase 29: Validation & E2E (Not started)

**Decisions:**
- (See STATE.md for historical decisions from v1.0-v1.4)
