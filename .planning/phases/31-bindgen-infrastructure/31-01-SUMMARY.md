---
phase: 31-bindgen-infrastructure
plan: 01
subsystem: build
tags: [bindgen, FFI, HIP, ROCm, compile-time-verification]

# Dependency graph
requires:
  - phase: 30-immediate-bugfix
    provides: sanity-check fixes and DeviceLimits cleanup
provides:
  - bindgen-based FFI binding generation for hipDeviceProp_t
  - Compile-time accessible hip_device_bindings.rs in OUT_DIR
  - Foundation for Phase 32 offset verification tests
affects: [32-offset-verification]

# Tech tracking
tech-stack:
  added: [bindgen 0.70]
  patterns:
    - Allowlist-only bindgen generation (not full HIP API)
    - Conditional binding generation (skip if HIP headers missing)
    - Version-aware struct binding (hipDeviceProp_tR0600)

key-files:
  created: [$OUT_DIR/hip_device_bindings.rs (generated at compile time)]
  modified: [Cargo.toml, build.rs]

key-decisions:
  - "FFI-03: Use bindgen allowlist-only for hipDeviceProp_t, not full HIP API"
  - "Bindgen requires HIP platform define (-D__HIP_PLATFORM_AMD__) and include paths"
  - "ROCm versioning uses _R0600 suffix on types (hipDeviceProp_tR0600)"
  - "Generated bindings are for verification only - existing FFI declarations remain"

patterns-established:
  - "Pattern: bindgen::Builder with .allowlist_type() for minimal generation"
  - "Pattern: Graceful skip when headers missing (cargo:warning, not error)"
  - "Pattern: .clang_arg() for platform-specific defines and include paths"

# Metrics
duration: 12min
completed: 2026-01-21
---

# Phase 31 Plan 01: Add bindgen Infrastructure Summary

**Bindgen 0.70 integrated with allowlist-only generation of hipDeviceProp_tR0600 struct for compile-time offset verification**

## Performance

- **Duration:** 12 minutes
- **Started:** 2026-01-21T11:18Z
- **Completed:** 2026-01-21T11:30Z
- **Tasks:** 3
- **Files modified:** 2 (Cargo.toml, build.rs)

## Accomplishments

- Added bindgen 0.70 to build-dependencies in Cargo.toml
- Created `generate_hip_bindings()` function in build.rs that generates hipDeviceProp_tR0600 bindings
- Generated bindings file (~1000 lines, not 10,000+ full HIP API) is accessible via OUT_DIR
- Verified existing FFI declarations in ffi.rs remain unchanged (manual declarations preserved)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add bindgen 0.70 to build-dependencies** - `dab3802` (feat)
2. **Task 2: Add generate_hip_bindings() function to build.rs** - `7f4bfa9` (feat)
3. **Task 3: Verification only (no commit)** - bindings generation verified, ffi.rs unchanged

**Plan metadata:** (to be committed after SUMMARY.md creation)

## Files Created/Modified

- `Cargo.toml` - Added bindgen = "0.70" to [build-dependencies]
- `build.rs` - Added `generate_hip_bindings()` function with HIP platform define and include paths
- `$OUT_DIR/hip_device_bindings.rs` - Generated at compile time (not checked in)

## Decisions Made

- **FFI-03: Bindgen Allowlist-Only**: Use bindgen with `.allowlist_type()` to generate only hipDeviceProp_t and related types, not the full HIP API. This keeps generated code minimal (~1000 lines vs 10,000+).

- **Bindgen requires HIP platform define**: Added `-D__HIP_PLATFORM_AMD__` clang arg to satisfy HIP header guards that otherwise cause compilation errors.

- **Include paths required**: Added explicit `-I{rocm_root}/include` and `-I{rocm_root}/include/hip` clang args so bindgen can find hip_version.h and other headers.

- **Regex allowlist patterns**: Used `hipDeviceProp.*` and `hipUUID.*` regex patterns to catch versioned struct names (hipDeviceProp_tR0600).

- **Graceful degradation**: Function returns early with warning if HIP headers not found, rather than failing the build.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added clang args for HIP header parsing**
- **Found during:** Task 2 (bindgen generation failing with missing header error)
- **Issue:** Bindgen's clang couldn't find hip_version.h and failed due to missing __HIP_PLATFORM_AMD__ define
- **Fix:** Added `.clang_arg("-D__HIP_PLATFORM_AMD__")` and include path clang args to bindgen builder
- **Files modified:** build.rs
- **Committed in:** `7f4bfa9` (Task 2 commit)

**2. [Rule 3 - Blocking] Used regex patterns for allowlist**
- **Found during:** Task 2 (hipDeviceProp_t not being generated)
- **Issue:** Exact type match "hipDeviceProp_t" didn't catch the versioned typedef `hipDeviceProp_tR0600`
- **Fix:** Changed to regex pattern `hipDeviceProp.*` and `hipUUID.*` to match versioned types
- **Files modified:** build.rs
- **Committed in:** `7f4bfa9` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking issues required for bindgen to work)
**Impact on plan:** Both fixes were necessary for bindgen to successfully parse HIP headers and generate bindings. No scope creep - these are bindgen-specific configuration requirements.

## Issues Encountered

- **Bindgen header parsing errors**: Initial bindgen invocation failed with "hip_version.h file not found" and "Must define __HIP_PLATFORM_AMD__". Fixed by adding clang args.

- **Type allowlist not matching**: Exact `hipDeviceProp_t` allowlist didn't generate the struct because ROCm uses a macro `#define hipDeviceProp_t hipDeviceProp_tR0600`. Fixed by using regex patterns.

## User Setup Required

None - no external service configuration required. Bindgen runs automatically during `cargo build` if HIP headers are present at ROCM_PATH (default: /opt/rocm).

## Next Phase Readiness

- **Ready for Phase 32**: `hip_device_bindings.rs` is generated in OUT_DIR and accessible via `include!(concat!(env!("OUT_DIR"), "/hip_device_bindings.rs"))` for offset verification tests
- **Generated struct**: `hipDeviceProp_tR0600` contains full field definitions with offsets for comparison
- **No blocking issues**: All success criteria met

## Verification Results

All success criteria verified:

1. [x] bindgen 0.70 added to Cargo.toml [build-dependencies]
2. [x] build.rs generates hip_device_bindings.rs with hipDeviceProp_tR0600 struct
3. [x] Generated bindings are accessible via include!(concat!(env!("OUT_DIR"), "/hip_device_bindings.rs"))
4. [x] Existing FFI declarations in ffi.rs remain unchanged (30 manual hip functions)
5. [x] cargo build successfully generates bindings without errors
6. [x] Generated bindings are minimal (1014 lines, allowlist-only, not full HIP API)

---
*Phase: 31-bindgen-infrastructure*
*Plan: 01*
*Completed: 2026-01-21*
