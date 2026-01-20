---
phase: 28-rocm-compilation-fix
plan: 08
subsystem: gpu-compilation
tags: [rocm, hip, cfg-gates, unconditional-compilation, sampling, mlp, profiling]

# Dependency graph
requires:
  - phase: 28-05
    provides: cfg gate removal from src/kernels/
provides:
  - Unconditional compilation of GPU sampling kernels (top-k, top-p, fused)
  - Unconditional compilation of GPU MLP operations (SwiGLU, RMSNorm)
  - Unconditional compilation of profiling tools (kernel timer, TTFT)
  - Unconditional compilation of HIP backend memory operations
affects: [future-phases]

# Tech tracking
tech-stack:
  added: []
  patterns: [unconditional-gpu-compilation]

key-files:
  created: []
  modified:
    - src/sampler/gpu.rs - Removed cfg gates from kernel re-exports and tests
    - src/sampler/gpu/fused.rs - Removed cfg gates from GpuFusedSampler
    - src/sampler/gpu/kernels.rs - Removed cfg gates from kernel functions
    - src/sampler/gpu/top_k.rs - Removed cfg gates from GpuTopKSampler
    - src/sampler/gpu/top_p.rs - Removed cfg gates from GpuTopPSampler
    - src/mlp/mod.rs - Removed cfg gates from module declarations
    - src/mlp/kernels.rs - Removed cfg gates from swiglu_gpu_kernel, rms_norm_gpu_kernel
    - src/mlp/rms_norm_tests.rs - Removed cfg gates from test module
    - src/mlp/swiglu_tests.rs - Removed cfg gates from test module
    - src/mlp/gpu_path_regression_tests.rs - Removed cfg gates from test module
    - src/profiling/mod.rs - Removed cfg gates from documentation
    - src/profiling/ttft.rs - Removed cfg gate from record_kernel_from_timer
    - src/profiling/kernel_timer.rs - Removed cfg gates from TimerStart, TimerStop, start, stop
    - src/backend/hip_backend/backend.rs - Removed cfg gate from mlp_swiglu SwiGLU block
    - src/backend/hip_backend/memory.rs - Removed cfg gates from backtrace logging

key-decisions:
  - "Unconditional GPU Sampling Compilation: GPU sampling (top-k, top-p, fused) is core to ROCmForge - should always be available, not conditionally compiled"
  - "Unconditional GPU MLP Compilation: GPU MLP operations (SwiGLU, RMSNorm) are core - should always be available"
  - "Unconditional Profiling Compilation: Profiling tools (kernel timer, TTFT) are always available for performance measurement"
  - "Unconditional HIP Backend: HIP backend memory operations are always available"

patterns-established:
  - "Unconditional GPU Compilation: All GPU operations compile unconditionally - no feature gates"

# Metrics
duration: 7min
completed: 2026-01-20
---

# Phase 28: Plan 08 Summary

**Unconditional GPU compilation for sampling kernels, MLP operations, profiling tools, and HIP backend - removed 42 cfg gates from 15 files**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-20T21:39:20Z
- **Completed:** 2026-01-20T21:46:22Z
- **Tasks:** 1
- **Files modified:** 15

## Accomplishments

- Removed all `#[cfg(feature = "rocm")]` gates from src/sampler/ directory (5 files)
- Removed all `#[cfg(feature = "rocm")]` gates from src/mlp/ directory (5 files)
- Removed all `#[cfg(feature = "rocm")]` gates from src/profiling/ directory (3 files)
- Removed all `#[cfg(feature = "rocm")]` gates from src/backend/hip_backend/ (2 files)
- Verified zero cfg gates remain in target directories

## Task Commits

1. **Task 1: Remove cfg gates from sampler, mlp, profiling, backend files** - `63f8583` (refactor)

**Plan metadata:** N/A (will be in final metadata commit)

## Files Created/Modified

### Sampler (5 files)
- `src/sampler/gpu.rs` - Removed cfg gates from kernel re-exports and all test functions (16 gates removed)
- `src/sampler/gpu/fused.rs` - Removed cfg gates from imports, struct, and impl (3 gates removed)
- `src/sampler/gpu/kernels.rs` - Removed cfg gates from imports, const, struct, functions (13 gates removed)
- `src/sampler/gpu/top_k.rs` - Removed cfg gates from imports, struct, and impl (3 gates removed)
- `src/sampler/gpu/top_p.rs` - Removed cfg gates from imports, struct, and impl (3 gates removed)

### MLP (5 files)
- `src/mlp/mod.rs` - Removed cfg gates from module declarations (4 gates removed)
- `src/mlp/kernels.rs` - Removed cfg gates from swiglu_gpu_kernel, rms_norm_gpu_kernel (2 gates removed)
- `src/mlp/rms_norm_tests.rs` - Removed cfg gate from test module (1 gate removed)
- `src/mlp/swiglu_tests.rs` - Removed cfg gate from test module (1 gate removed)
- `src/mlp/gpu_path_regression_tests.rs` - Removed cfg gate from test module (1 gate removed)

### Profiling (3 files)
- `src/profiling/mod.rs` - Removed cfg gates from documentation example (documentation cleanup)
- `src/profiling/ttft.rs` - Removed cfg gate from record_kernel_from_timer (1 gate removed)
- `src/profiling/kernel_timer.rs` - Removed cfg gates from TimerStart, TimerStop, start, stop (6 gates removed)

### Backend (2 files)
- `src/backend/hip_backend/backend.rs` - Removed cfg gate from SwiGLU block in mlp_swiglu (1 gate removed)
- `src/backend/hip_backend/memory.rs` - Removed cfg gates from backtrace logging (2 gates removed)

## Decisions Made

- **Unconditional GPU Sampling**: GPU sampling kernels (top-k, top-p, fused) are core functionality and should always be compiled, not conditionally available via feature flags
- **Unconditional GPU MLP**: GPU MLP operations (SwiGLU activation, RMSNorm normalization) are core to transformer inference - always available
- **Unconditional Profiling**: Kernel timing and TTFT profiling are essential development tools - always available
- **Unconditional HIP Backend**: HIP backend memory operations are always available as the project's core abstraction

## Deviations from Plan

None - plan executed exactly as written. All 15 files were processed as specified in the plan.

## Issues Encountered

None - all cfg gate removals were straightforward and cargo check passed with only minor pre-existing warnings (11 warnings unrelated to this change).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- GPU sampling, MLP operations, profiling, and HIP backend are now always compiled
- Ready for any subsequent plans that depend on unconditional GPU compilation
- Zero cfg gates remain in src/sampler/, src/mlp/, src/profiling/, src/backend/hip_backend/

---
*Phase: 28-rocm-compilation-fix*
*Plan: 08*
*Completed: 2026-01-20*
