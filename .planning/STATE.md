# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-21)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Milestone v1.5 - Env Var & Transpose Fix

## Current Position

Phase: 27 of 29 (Device Property Infrastructure)
Plan: —
Status: Not started
Last activity: 2026-01-21 — Phase 26 complete (Transpose Kernel Fix)

Progress: [█████████░░░░░░░░░░░] 90% (183/186 plans complete, Phases 25-26 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 168 (v1.0-v1.4)
- Average duration: ~42 min
- Total execution time: ~120 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-24 | 168 | 168 | ~42 min |
| 25-29 | 12 | TBD | ~5 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions from research (2026-01-21):

- **Env Var Compile-Time Embedding**: HSACO paths must be embedded at compile time using `option_env!()` macro, not read via `std::env::var()` at runtime. build.rs sets `cargo:rustc-env=VAR=path` which is only accessible via `option_env!()` (ENV-01)
- **Transpose Block Dimension Fix**: Change block dimensions from (64,64,1) to (32,32,1) to reduce thread count from 4096 to 1024, staying within maxThreadsPerBlock limit (TRN-01)
- **Transpose Pre-Launch Validation**: Add assertions before kernel launch to validate thread count, grid dimensions, block dimensions, and shared memory against AMD GPU limits from DEVICE_LIMITS.md (TRN-02)
- **Device Property Caching**: Query `hipGetDeviceProperties` once at backend init, not per-launch, to avoid overhead and enable consistent validation (DEV-01)
- **Cargo Rerun Directives**: Added explicit `cargo:rerun-if-env-changed` directives for ROCM_PATH, HIPCC, ROCm_ARCH to ensure automatic rebuild when ROCm configuration changes (BLD-01)

Historical decisions (see STATE.md archive for v1.0-v1.4 details):
- GPU Transpose for Embedding Weights, Memory Arena for GPU Weights, Zero Warnings Baseline, Unconditional GPU Compilation

### Pending Todos

None yet.

### Blockers/Concerns

**CRITICAL: Transpose Kernel Fails for Large Tensors (2026-01-21)**

**Issue 1: Env var embedding mismatch — RESOLVED ✓**
- build.rs sets `cargo:rustc-env=Q4_0_DEQUANT_HSACO=/path/to/kernel.hsaco` (compile-time)
- Code was using `std::env::var("Q4_0_DEQUANT_HSACO")` (runtime lookup)
- **Fixed in Phase 25:** All 29 HSACO env vars now use `option_env!()` macro
- Kernels load without manual environment variables

**Issue 2: Transpose kernel fails for large tensors — RESOLVED ✓**
- Tensor shape [896, 151936] (Qwen2.5 embedding weights)
- Kernel launch returns `hipErrorInvalidValue` (invalid argument)
- Root cause: block=(64,64,1) = 4096 threads exceeds maxThreadsPerBlock=1024
- **Fixed in Phase 26:** Block dimension changed to (32,32,1) = 1024 threads
- **Validated in Phase 26:** Pre-launch assertions added, unit test for [896, 151936] transpose

## Session Continuity

Last session: 2026-01-21
Stopped at: Phase 26 complete, transpose kernel fix and validation done
Resume file: None

**v1.5 - Env Var & Transpose Fix (2026-01-21):**
- Phase 25: Env Var Fix (12/12 complete ✓)
  - Original: 25-01 (attention), 25-02 (sampler), 25-03 (MLP), 25-04 (quant), 25-05 (fused), 25-06 (transpose), 25-07 (cargo rerun)
  - Gap closure: 25-08 (Q4_0_DEQUANT duplicate), 25-09 (Q4_K_DEQUANT duplicate), 25-10 (Q4_0_MATMUL), 25-11 (Q4_K_MATMUL), 25-12 (Q6_K_MATMUL)
- Phase 26: Transpose Kernel Fix (3/3 complete ✓)
  - 26-01: Block dimension changed to (32,32,1)
  - 26-02: Pre-launch validation assertions added
  - 26-03: Unit test for [896, 151936] tensor transpose
- Phase 27: Device Property Infrastructure (Not started)
- Phase 28: Debug Hygiene (Not started)
- Phase 29: Validation & E2E (Not started)

**Coverage:** 15/15 requirements mapped to phases (100%)
