---
phase: 25-env-var-fix
plan: 08
subsystem: quantization
tags: [q4_0, dequantization, option_env, compile-time, hsaco]

# Dependency graph
requires:
  - phase: 25-env-var-fix
    plan: 04
    provides: Q4_K and Q6_K quant kernel env var pattern
provides:
  - Q4_0_DEQUANT_HSACO uses compile-time option_env!() for path embedding
  - Error messages distinguish compile-time missing vs file-not-found
affects: [25-09, 25-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "option_env!() macro for compile-time environment variable embedding"
    - "build.rs sets cargo:rustc-env=VAR=value, code reads with option_env!(\"VAR\")"

key-files:
  created: []
  modified:
    - src/ggml/hip_backend/ops/q4_0_dequant.rs

key-decisions:
  - "Gap closure: Duplicate Q4_0 dequant in ggml/hip_backend/ops/ needed same option_env!() fix as kernels/quant/q4_0.rs"
  - "Error message pattern: 'not set at compile time. Rebuild the project.' distinguishes from file-not-found"

patterns-established:
  - "Compile-time env var pattern: option_env!(\"VAR_NAME\") instead of std::env::var(\"VAR_NAME\")"

# Metrics
duration: 2min
completed: 2026-01-21
---

# Phase 25 Plan 08: Q4_0 Dequant Env Var Fix Summary

**Q4_0_DEQUANT_HSACO kernel path uses compile-time option_env!() macro instead of runtime std::env::var()**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-21T01:17:38Z
- **Completed:** 2026-01-21T01:19:26Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Fixed duplicate Q4_0 dequantization implementation in ggml/hip_backend/ops/ to use compile-time env var
- Removed unused `std::env` import
- Updated error messages to match q6_k_dequant.rs pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace std::env::var with option_env! for Q4_0_DEQUANT_HSACO** - `85e7285` (fix)

## Files Created/Modified

- `src/ggml/hip_backend/ops/q4_0_dequant.rs` - Changed from runtime `env::var("Q4_0_DEQUANT_HSACO")` to compile-time `option_env!("Q4_0_DEQUANT_HSACO")`

## Decisions Made

None - followed plan as specified. This was a straightforward application of the established pattern from q6_k_dequant.rs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Gap closure complete for Q4_0_DEQUANT_HSACO
- Remaining gaps: Q4_K_DEQUANT (25-09), Q4_0_MATMUL (25-10), Q4_K_MATMUL (25-11), Q6_K_MATMUL (25-12)

---
*Phase: 25-env-var-fix*
*Completed: 2026-01-21*
