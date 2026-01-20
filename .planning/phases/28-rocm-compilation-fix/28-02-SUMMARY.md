---
phase: 28-rocm-compilation-fix
plan: 02
subsystem: compilation-fix
tags: [rocm, flash-attention, variable-naming, mut-keyword]

# Dependency graph
requires:
  - phase: 28-01
    provides: c_void and HipError imports in FFI kernel files
provides:
  - Properly named parameters in flash_attention.rs (no misleading underscore prefixes)
  - Verified simple_transformer.rs already has correct mut keyword
affects: [28-03]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - src/attention/flash_attention.rs

key-decisions:
  - "No underscore prefix suppression for conditionally compiled code: Parameters used inside #[cfg(feature = \"rocm\")] blocks should have their real names, even if this causes warnings when compiling without the feature"

patterns-established:
  - "Real names for conditionally compiled parameters: Don't use underscore prefix to suppress warnings for code that IS used when features are enabled"

# Metrics
duration: 3min
completed: 2026-01-20
---

# Phase 28: Plan 02 - Fix Underscore-Prefixed Parameters Summary

**Removed misleading underscore prefixes from flash_attention.rs parameters that ARE used in GPU kernel code paths**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-20T21:23:07Z
- **Completed:** 2026-01-20T21:26:07Z
- **Tasks:** 2 (1 with changes, 1 verified already complete)
- **Files modified:** 1

## Accomplishments

- Fixed `forward()` function parameters: `_mask`, `_q`, `_k`, `_v` -> `mask`, `q`, `k`, `v`
- Fixed `FlashAttentionBackend` struct field: `_handle` -> `handle`
- Verified `simple_transformer.rs` already has correct `let mut linear` declaration

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix underscore-prefixed parameters in flash_attention.rs** - `06ab856` (fix)

**Task 2: Add missing mut keyword in simple_transformer.rs** - Already complete in Phase 27-04, no commit needed

**Plan metadata:** (to be added after STATE.md update)

## Files Created/Modified

- `src/attention/flash_attention.rs` - Removed underscore prefixes from `mask`, `q`, `k`, `v` parameters and `handle` field

## Decisions Made

- **No underscore prefix suppression for conditionally compiled code**: Parameters used inside `#[cfg(feature = "rocm")]` blocks should have their real names. The underscore prefix was only needed to suppress warnings when rocm code wasn't being compiled, but now that we're fixing rocm compilation, parameters need meaningful names.
- **Compiler warnings without feature flag are expected**: When compiling without `--features rocm`, the parameters appear unused (because the code using them is gated). This is acceptable - the warnings only appear when rocm is disabled, and we're making rocm the default.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all changes were straightforward.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- flash_attention.rs now has properly named parameters for rocm compilation
- simple_transformer.rs already has correct `mut` binding
- Ready for Phase 28-03 (kernels/transpose/mod.rs type mismatch fix)

---
*Phase: 28-rocm-compilation-fix*
*Completed: 2026-01-20*
