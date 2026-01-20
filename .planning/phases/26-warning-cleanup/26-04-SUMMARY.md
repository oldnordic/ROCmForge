---
phase: 26-warning-cleanup
plan: 04
subsystem: attention-kernels
tags: [visibility, kernel-cache, rocm, warnings]

# Dependency graph
requires:
  - phase: 25-architectural-decomposition
    provides: kernel cache module structure with kernels_basic and kernels_flash submodules
provides:
  - Fixed visibility mismatch between pub fn get_or_init_cache() and private struct KernelCache
  - Eliminated 1 compiler warning related to private type visibility
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - src/attention/kernels/kernels_cache/mod.rs
    - src/attention/flash_attention.rs
    - src/attention/multi_query.rs
    - src/tensor/matmul.rs

key-decisions:
  - "Made get_or_init_cache() pub(crate) instead of pub - function only used within kernels_cache module and its child submodules"
  - "Gated get_or_init_cache() with #[cfg(feature = \"rocm\")] to align with ROCm-specific code"
  - "Gated kernels_basic and kernels_flash submodules with #[cfg(feature = \"rocm\")] for consistency"

patterns-established:
  - "Visibility alignment: Function visibility should match the visibility of types it returns"

# Metrics
duration: 12min
completed: 2026-01-20
---

# Phase 26: Plan 04 - Fix KernelCache Visibility Mismatch Summary

**Made get_or_init_cache() function crate-private to match private KernelCache struct, eliminating visibility warning**

## Performance

- **Duration:** 12 min
- **Started:** 2026-01-20T18:31:51Z
- **Completed:** 2026-01-20T18:43:00Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments

- Fixed private type visibility mismatch: `pub fn get_or_init_cache()` returning private `KernelCache`
- Changed function to `pub(crate)` and gated with `#[cfg(feature = "rocm")]`
- Fixed test module imports broken by previous plan's cargo fix execution
- All 701 lib tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Determine KernelCache usage and fix visibility mismatch** - `b7016ad` (fix)
2. **Task 1-bonus: Fix test module imports broken by cargo fix in 26-03** - `12ee42c` (fix)

**Plan metadata:** N/A (no metadata commit for single-task plan)

## Files Created/Modified

- `src/attention/kernels/kernels_cache/mod.rs` - Changed `get_or_init_cache()` to `pub(crate)` and added `#[cfg(feature = "rocm")]` gate
- `src/attention/flash_attention.rs` - Fixed test module structure (malformed from cargo fix)
- `src/attention/multi_query.rs` - Fixed test module structure and added RopeConfig import
- `src/tensor/matmul.rs` - Fixed test module structure

## Decisions Made

- **Made get_or_init_cache() crate-private**: The function is only used within the kernels_cache module and its child submodules (kernels_basic, kernels_flash). Making it `pub(crate)` instead of `pub` aligns with the actual usage pattern.
- **Added #[cfg(feature = "rocm")] gate**: Since the function and its child submodules are ROCm-specific, gating them ensures consistent behavior across feature configurations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test module imports broken by cargo fix**
- **Found during:** Task 1 (verification after KernelCache fix)
- **Issue:** Plan 26-03 ran `cargo fix --lib` which incorrectly removed `use super::*;` from test modules and malformed the test module structure. Tests failed to compile with "use of undeclared type" errors.
- **Fix:** Restored proper test module structure and imports in:
  - flash_attention.rs: Fixed `#[cfg(test)] mod tests` structure
  - multi_query.rs: Fixed structure and added `use super::super::rope::RopeConfig;`
  - matmul.rs: Fixed structure
- **Files modified:** src/attention/flash_attention.rs, src/attention/multi_query.rs, src/tensor/matmul.rs
- **Verification:** All 701 lib tests pass
- **Committed in:** `12ee42c` (separate commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test fixes were required for verification to pass. The KernelCache warning fix itself followed the plan exactly.

## Issues Encountered

- **Test compilation failures after cargo fix**: Plan 26-03's `cargo fix --lib` execution incorrectly removed `use super::*;` imports from test modules and malformed the `#[cfg(test)] mod tests` structure. Fixed by restoring proper test module structure.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 26 warning cleanup continues with remaining warning categories (deprecated functions, unused imports)
- KernelCache visibility warning eliminated
- All tests passing, ready for next warning cleanup task

---
*Phase: 26-warning-cleanup*
*Completed: 2026-01-20*
