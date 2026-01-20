---
phase: 26-warning-cleanup
plan: 01
subsystem: loader
tags: [dequantization, kernel-migration, compiler-warnings, rust]

# Dependency graph
requires:
  - phase: 24-kernel-centric-restructure
    plan: 02
    provides: CPU quantization kernels migrated to kernels/quant module
provides:
  - Clean loader module with zero deprecation warnings from deprecated dequant functions
  - Proper imports from kernels::quant module for Q4_0, Q4_K, Q6_K, Q8_0 dequantization
  - Removed unused imports (Seek, rayon::prelude) from loader module
  - Documented dead code markers with #[allow(dead_code)] attributes
affects:
  - Phase 26-02 (can continue warning cleanup without dequant warnings)
  - Future dequant refactoring (deprecated wrappers can be removed after consumers migrate)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Type alias pattern for backward compatibility: pub use crate::kernels::quant::dequantize_q4_0_cpu as dequant_q4_0"
    - "Direct kernel module calls in dequantize dispatcher function"
    - "Dead code documentation: #[allow(dead_code)] with explanatory comments"

key-files:
  created: []
  modified:
    - src/loader/mod.rs
    - src/loader/dequant.rs
    - src/loader/gguf/metadata.rs
    - src/loader/gguf/types.rs

key-decisions:
  - "Use type aliases for backward compatibility instead of removing deprecated exports immediately"
  - "Call kernel module functions directly in dequantize dispatcher instead of through deprecated wrappers"
  - "Document reserved functions with #[allow(dead_code)] instead of removing them"

patterns-established:
  - "Pattern: Migration compatibility layer - use pub use with as alias to preserve old API names"
  - "Pattern: Direct kernel calls bypass deprecated wrappers - eliminates deprecation warnings"

# Metrics
duration: 12min
completed: 2026-01-20
---

# Phase 26: Plan 01 Summary

**Migrated loader module dequantization exports from deprecated wrappers to direct kernel::quant module imports, eliminating 8 deprecation warnings and 2 unused import warnings**

## Performance

- **Duration:** 12 min
- **Started:** 2026-01-20T18:32:04Z
- **Completed:** 2026-01-20T18:44:00Z
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments

- Migrated Q4_0, Q4_K, Q6_K, Q8_0 dequant exports to use kernel::quant module directly
- Updated dequantize dispatcher to call kernel functions instead of deprecated wrappers
- Removed unused imports (Seek from metadata.rs, rayon::prelude from dequant.rs)
- Added #[allow(dead_code)] marker with explanatory comment for transpose_f32_matrix

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix deprecated dequant imports in loader/mod.rs** - `e8d8c7f` (feat)
2. **Task 2: Fix deprecated dequant calls in loader/dequant.rs** - `5b0f7ae` (feat)
3. **Task 3: Remove unused imports** - `b4247ff` (fix)
4. **Task 4: Mark unused transpose_f32_matrix with #[allow(dead_code)]** - `5a37b90` (fix)

**Plan metadata:** Not yet committed (pending STATE.md update)

## Files Created/Modified

- `src/loader/mod.rs` - Replaced deprecated dequant exports with kernel::quant imports using type aliases
- `src/loader/dequant.rs` - Updated dequantize dispatcher to call kernel functions directly, removed unused rayon::prelude import
- `src/loader/gguf/metadata.rs` - Removed unused Seek import
- `src/loader/gguf/types.rs` - Added #[allow(dead_code)] attribute to transpose_f32_matrix with explanatory comment

## Decisions Made

### Type Aliases for Backward Compatibility

The plan originally suggested using `as` aliases on the import statement. I used this approach in `src/loader/mod.rs`:

```rust
pub use crate::kernels::quant::dequantize_q4_0_cpu as dequant_q4_0;
pub use crate::kernels::quant::dequantize_q4_k_cpu as dequant_q4_k;
pub use crate::kernels::quant::dequantize_q6_k_cpu as dequant_q6_k;
pub use crate::kernels::quant::dequantize_q8_0 as dequant_q8_0;
```

This preserves the local function names for external consumers while eliminating deprecation warnings.

### Direct Kernel Calls in Dispatcher

For the internal `dequantize` function, I used direct kernel module calls instead of the deprecated wrapper functions. This eliminates the 4 remaining deprecation warnings that were coming from the dispatcher function.

Rationale: The deprecated wrapper functions are kept for backward compatibility for external consumers, but internal code should use the kernel module directly.

### Dead Code Documentation for transpose_f32_matrix

Added `#[allow(dead_code)]` with an explanatory comment indicating the function is "Reserved for future tensor layout conversions". This follows the project standard from Phase 20 that all dead code markers must have explanatory comments.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed as expected.

### Note on Uncommitted Working Directory Files

The git working directory contained many uncommitted modified files unrelated to this plan (src/attention/, src/ggml/, src/http/, etc.). These are from previous phase work (25-16, 25-17) and do not affect this plan's execution. The lib compiles successfully and the specific warning cleanup goals were achieved.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Loader module dequantization warnings eliminated
- Remaining warning categories can be addressed independently:
  - Deprecated to_host_vec (13 warnings) - separate plan
  - Kernel cache dead code (12 warnings) - already addressed in 26-03
  - FFI constants (2 warnings) - already addressed in 26-03
  - Visibility mismatch (1 warning) - separate plan needed
  - Other dead code items - individual warnings with clear justifications

### Blockers/Concerns

- None identified

---
*Phase: 26-warning-cleanup*
*Completed: 2026-01-20*
