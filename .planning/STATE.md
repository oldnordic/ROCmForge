# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-21)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Milestone v1.5 - Env Var & Transpose Fix

## Current Position

Phase: 25 of 29 (Env Var Fix)
Plan: 05 of 07
Status: In progress
Last activity: 2026-01-21 — Completed 25-05 (fused kernel compile-time env var fix)

Progress: [████████░░░░░░░░░░░░░] 87% (171/196 plans complete, 4/7 in phase 25)

## Performance Metrics

**Velocity:**
- Total plans completed: 168 (v1.0-v1.4)
- Average duration: ~42 min
- Total execution time: ~120 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-24 | 168 | 168 | ~42 min |
| 25-29 | 4 | TBD | ~3 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions from research (2026-01-21):

- **Env Var Compile-Time Embedding**: HSACO paths must be embedded at compile time using `option_env!()` macro, not read via `std::env::var()` at runtime. build.rs sets `cargo:rustc-env=VAR=path` which is only accessible via `option_env!()` (ENV-01)
- **Transpose Block Dimension Fix**: Change block dimensions from (64,64,1) to (32,32,1) to reduce thread count from 4096 to 1024, staying within maxThreadsPerBlock limit (TRN-01)
- **Device Property Caching**: Query `hipGetDeviceProperties` once at backend init, not per-launch, to avoid overhead and enable consistent validation (DEV-01)
- **Cargo Rerun Directives**: Added explicit `cargo:rerun-if-env-changed` directives for ROCM_PATH, HIPCC, ROCm_ARCH to ensure automatic rebuild when ROCm configuration changes (BLD-01)

Historical decisions (see STATE.md archive for v1.0-v1.4 details):
- GPU Transpose for Embedding Weights, Memory Arena for GPU Weights, Zero Warnings Baseline, Unconditional GPU Compilation

### Pending Todos

None yet.

### Blockers/Concerns

**CRITICAL: Runtime Kernel Loading Broken (2026-01-21)**

Two critical issues block actual GGUF model inference:

**Issue 1: Env var embedding mismatch**
- build.rs sets `cargo:rustc-env=Q4_0_DEQUANT_HSACO=/path/to/kernel.hsaco` (compile-time)
- Code uses `std::env::var("Q4_0_DEQUANT_HSACO")` (runtime lookup)
- Result: HSACO paths not embedded in binary, runtime lookup fails
- Manual `env VAR=value ./binary` works - proves runtime vs compile-time mismatch

**Issue 2: Transpose kernel fails for large tensors**
- Tensor shape [896, 151936] (Qwen2.5 embedding weights)
- Kernel launch returns `hipErrorInvalidValue` (invalid argument)
- Root cause: block=(64,64,1) = 4096 threads exceeds maxThreadsPerBlock=1024

## Session Continuity

Last session: 2026-01-21
Stopped at: Completed 25-05 (fused kernel compile-time env var fix)
Resume file: 25-05-SUMMARY.md

**v1.5 - Env Var & Transpose Fix (2026-01-21):**
- Phase 25: Env Var Fix (5/7 complete - 25-03, 25-04, 25-05, 25-06, 25-07 done; 25-01, 25-02 remain)
- Phase 26: Transpose Kernel Fix (Not started)
- Phase 27: Device Property Infrastructure (Not started)
- Phase 28: Debug Hygiene (Not started)
- Phase 29: Validation & E2E (Not started)

**Coverage:** 15/15 requirements mapped to phases (100%)
