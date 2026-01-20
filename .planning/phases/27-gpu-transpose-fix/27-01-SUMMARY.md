---
phase: 27-gpu-transpose-fix
plan: 01
subsystem: gpu-kernels
tags: [hip, transpose, kernel, gpu-memory, rdna3]

# Dependency graph
requires:
  - phase: 24-kernel-centric-restructure
    provides: kernels directory structure with quant, attention, matmul, element modules
  - phase: 15-gpu-sampling
    provides: HipKernel management pattern from sampler/gpu.rs
provides:
  - TransposeKernel struct with lazy HSACO loading
  - TRANSPOSE_HSACO build.rs integration
  - Module skeleton for GPU transpose (kernel implementation in 27-02)
affects:
  - phase: 27-02 (HIP kernel source creation)
  - phase: 27-03 (CPU transpose replacement with GPU kernel)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Lazy kernel loading via initialize() pattern
    - Environment variable-based HSACO path discovery
    - HipResult error handling for missing kernels

key-files:
  created:
    - src/kernels/transpose/mod.rs
    - kernels/transpose.hip (external, for 27-02)
  modified:
    - src/kernels/mod.rs
    - build.rs

key-decisions:
  - "Used existing kernel cache pattern from sampler/gpu.rs for TransposeKernel structure"
  - "Kernel name 'transpose_kernel' follows existing naming convention (scale_kernel, softmax_kernel, etc.)"
  - "Module under 600 LOC (263 LOC) following Phase 24 convention"

patterns-established:
  - "Kernel module pattern: new() -> initialize() -> execute()"
  - "Error handling: HipError::KernelLoadFailed for missing HSACO with descriptive messages"
  - "Shape validation: reject non-2D tensors with DeviceError"

# Metrics
duration: 4min
completed: 2026-01-20
---

# Phase 27 Plan 01: GPU Transpose Kernel Infrastructure Summary

**TransposeKernel module with lazy HSACO loading, build.rs integration for TRANSPOSE_HSACO environment variable, and 2D tensor shape validation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-20T19:42:20Z
- **Completed:** 2026-01-20T19:46:44Z
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments

- Created TransposeKernel struct following existing kernel pattern (new, initialize, transpose methods)
- Integrated TRANSPOSE_HSACO compilation into build.rs with proper source path
- Added transpose module to kernels/mod.rs with documentation
- Implemented error handling for missing HSACO (env var check, file existence, debug logging)
- All 701 lib tests passing with no compiler warnings

## Task Commits

Each task was committed atomically:

1. **Task 1: Create transpose kernel module** - `6b80ec0` (feat)
   - Created src/kernels/transpose/mod.rs with TransposeKernel struct
   - new(), initialize(), and transpose() methods
   - Unit tests for creation, initialization, and shape validation

2. **Task 2: Add module to kernels/mod.rs** - `5112072` (feat)
   - Added pub mod transpose; to kernels/mod.rs
   - Updated module documentation
   - Fixed import paths (DeviceTensor, TensorShape from correct modules)
   - Fixed shape.dims().len() instead of non-existent rank() method
   - Added TRANSPOSE_HSACO to build.rs kernel compilation list

**Plan metadata:** (pending - will be in final commit)

## Files Created/Modified

- `src/kernels/transpose/mod.rs` (263 LOC) - TransposeKernel with lazy loading, shape validation, unit tests
- `src/kernels/mod.rs` - Added pub mod transpose; and module documentation
- `build.rs` - Added TRANSPOSE_HSACO kernel compilation entry
- `kernels/transpose.hip` - External file (will be used in 27-02)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed import paths for DeviceTensor and TensorShape**
- **Found during:** Task 1 (initial compilation)
- **Issue:** Used private path crate::backend::hip_backend::backend::DeviceTensor which is not accessible
- **Fix:** Changed to use public re-export from crate::backend::hip_backend and TensorShape from crate::loader::mmap_loader
- **Files modified:** src/kernels/transpose/mod.rs
- **Verification:** cargo check --lib passes
- **Committed in:** 5112072 (part of Task 2 commit)

**2. [Rule 1 - Bug] Fixed TensorShape construction method**
- **Found during:** Task 1 (initial compilation)
- **Issue:** Used non-existent TensorShape::new() method
- **Fix:** Changed to TensorShape::from_dims() which is the actual public API
- **Files modified:** src/kernels/transpose/mod.rs (3 locations)
- **Verification:** cargo check --lib passes
- **Committed in:** 5112072 (part of Task 2 commit)

**3. [Rule 1 - Bug] Fixed tensor rank check method**
- **Found during:** Task 1 (initial compilation)
- **Issue:** Used non-existent shape.rank() method
- **Fix:** Changed to shape.dims().len() which is the correct way to get dimension count
- **Files modified:** src/kernels/transpose/mod.rs
- **Verification:** cargo check --lib passes, test_transpose_rejects_non_2d_tensor works
- **Committed in:** 5112072 (part of Task 2 commit)

**4. [Rule 1 - Bug] Reverted external kernel name change**
- **Found during:** Task 2 (after external edit)
- **Issue:** External edit changed kernel name from "transpose_kernel" to "transposeLdsNoBankConflicts"
- **Fix:** Reverted to "transpose_kernel" to match plan specification and build.rs entry
- **Files modified:** src/kernels/transpose/mod.rs
- **Verification:** Kernel name matches build.rs TRANSPOSE_HSACO entry
- **Committed in:** 5112072 (part of Task 2 commit)

---

**Total deviations:** 4 auto-fixed (all Rule 1 - Bug fixes for compilation correctness)
**Impact on plan:** All fixes necessary for correct compilation. No scope creep. Plan executed as specified.

## Issues Encountered

None - all issues were auto-fixed via deviation rules.

## Acceptance Criteria Status

- [x] `src/kernels/transpose/mod.rs` compiles without errors
- [x] `src/kernels/transpose/mod.rs` under 600 LOC (263 LOC)
- [x] `TransposeKernel` struct has `new()`, `initialize()`, and `transpose()` methods
- [x] `src/kernels/mod.rs` includes `pub mod transpose;`
- [x] `build.rs` sets TRANSPOSE_HSACO environment variable
- [x] Module documentation updated in kernels/mod.rs
- [x] No compiler warnings
- [x] All lib tests pass (701/701)

## Next Phase Readiness

**Ready for Plan 27-02:**
- TransposeKernel struct in place with initialize() and transpose() methods
- build.rs configured to compile src/kernels/transpose/hip transpose.hip
- TRANSPOSE_HSACO environment variable will be set during build

**Blockers:** None

**Known TODOs:**
- transpose() method currently returns placeholder output (marked with TODO for plan 27-02)
- Kernel launch implementation pending HIP kernel source creation

---
*Phase: 27-gpu-transpose-fix*
*Completed: 2026-01-20*
