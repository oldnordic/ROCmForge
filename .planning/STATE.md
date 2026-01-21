# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-21)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Milestone v1.6 - FFI Device Props Fix

## Current Position

Phase: 31 of 32 (Bindgen Infrastructure)
Plan: 01 of 1
Status: Plan complete
Last activity: 2026-01-21 — Completed 31-01: Add bindgen Infrastructure with HIP Allowlist

Progress: [██░░░░░░░░░░░░░░░░░░░░░░░░░░] 67% (v1.6)

## Performance Metrics

**Velocity:**
- Total plans completed: 197 (v1.0-v1.5 + v1.6 phases 30-31)
- Average duration: ~42 min
- Total execution time: ~139 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-24 | 168 | 168 | ~42 min |
| 25-29 | 27 | 27 | ~7 min |
| 30-31 | 2 | 3 | ~9 min |
| 32 | 0 | 1 | TBD |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions from research (2026-01-21):

- **Env Var Compile-Time Embedding**: HSACO paths must be embedded at compile time using `option_env!()` macro, not read via `std::env::var()` at runtime. build.rs sets `cargo:rustc-env=VAR=path` which is only accessible via `option_env!()` (ENV-01)
- **Transpose Block Dimension Fix**: Change block dimensions from (64,64,1) to (32,32,1) to reduce thread count from 4096 to 1024, staying within maxThreadsPerBlock limit (TRN-01)
- **Transpose Pre-Launch Validation**: Add assertions before kernel launch to validate thread count, grid dimensions, block dimensions, and shared memory against AMD GPU limits (TRN-02)
- **Device Property Caching**: Query `hipGetDeviceProperties` once at backend init, not per-launch, to avoid overhead and enable consistent validation (DEV-01)
- **Cargo Rerun Directives**: Added explicit `cargo:rerun-if-env-changed` directives for ROCM_PATH, HIPCC, ROCm_ARCH to ensure automatic rebuild when ROCm configuration changes (BLD-01)
- **Launch Validation Returns Result**: Validation returns `HipError::KernelLaunchFailed` with detailed message, not panic - allows graceful error handling (VAL-01)
- **Private Helper Functions**: `ceil_div_u64` and `safe_grid_dim` are private module-level functions, not public API (VAL-02)
- **Grid Overflow Assert Pattern**: `safe_grid_dim` uses `assert!` for overflow detection - overflow is programmer error (VAL-03)
- **Validated Launch Wrapper**: All kernel launches use `launch_kernel_with_module_shared_validated()` which automatically validates against cached device limits before execution (VAL-04)
- **Test Skip on Bad Driver Data**: Unit tests detect and skip when HIP driver reports invalid maxThreadsDim (e.g., 0 for axes Y/Z), which is a known driver bug (VAL-05)
- **Kernel Name Storage**: HipKernel struct stores kernel name as owned String for better error messages. Using owned String avoids lifetime complexity and overhead is negligible for short kernel names (DBG-01)
- **Enhanced Launch Error Messages**: Kernel launch failures now include kernel name, grid dimensions, and block dimensions in error messages for better debugging (DBG-02)
- **Async Error Detection**: Call `hipGetLastError()` after successful kernel launch to catch asynchronous HIP errors. Uses `tracing::warn` (not error) since async errors may be from previous operations (DBG-03)
- **Debug-Only Logging**: Use `#[cfg(debug_assertions)]` for debug logging to ensure zero runtime overhead in release builds. The `debug_assertions` cfg is active for `cargo build` and `cargo test` but NOT for `cargo build --release` (DBG-04)
- **HIP_LAUNCH_BLOCKING Synchronous Execution**: `HIP_LAUNCH_BLOCKING` environment variable enables synchronous kernel execution for easier debugging. When set to "1" or "true", the backend calls `hipDeviceSynchronize()` after each kernel launch (DBG-05)
- **HIP Debugging Documentation**: Created comprehensive developer guide in docs/HIP_DEBUGGING.md covering debug builds, HIP_LAUNCH_BLOCKING usage, error message interpretation, common issues, and debugging tools (DOC-01)
- **Adaptive Transpose Block Sizing**: Transpose kernel now adapts to device maxThreadsPerBlock limit, supporting GPUs with varied limits (320, 512, 1024, etc.) - discovered during validation testing (TRN-03)
- **Validation Test Organization**: E2E validation tests organized in tests/validation/ with descriptive names and graceful skip patterns (VAL-06)
- **All-Axes Sanity Check**: Validate max_threads_dim[0], [1], [2] all > 0, not just [0] (FFI-01) - COMPLETED 2026-01-21
- **Single DeviceLimits Construction**: Delete duplicate assignment that overwrites vetted values (FFI-02) - COMPLETED 2026-01-21
- **Iterator-Based Validation Pattern**: Use `.iter().all(|&d| d > 0 && d <= 4096)` for clarity and correctness when validating array elements (FFI-01)
- **Detailed Warning Messages**: Include actual incorrect values in warning log to aid debugging of FFI driver issues (FFI-01)

v1.6 decisions (upcoming):
- **Bindgen Allowlist**: Use bindgen for hipDeviceProp_t only, not full HIP API (FFI-03) - COMPLETED 2026-01-21
- **Compile-Time Offset Verification**: Test asserts manual offsets match bindgen at compile time (FFI-04)

### Pending Todos

None yet.

### Blockers/Concerns

**RESOLVED: FFI Device Properties Bug (2026-01-21)**

**Issue:** Sanity check only validates `max_threads_dim[0] > 0`, allowing garbage like `[1024, 0, 0]` to pass. Later kernel launch fails with "block.y 1 exceeds limit 0".

**Root cause:** Two bugs compound:
1. Incomplete sanity check (only dim[0] validated)
2. Duplicate DeviceLimits assignment overwrites vetted values

**Resolution (2026-01-21):** Strengthened sanity check to validate ALL 3 dimensions plus grid, warp, shared, threads/block. Deleted duplicate DeviceLimits block. (Plan 30-01)

## Session Continuity

Last session: 2026-01-21
Stopped at: Completed 31-01 (Add bindgen Infrastructure with HIP Allowlist)
Resume file: None

**Milestone v1.5 COMPLETE!** All phases (25-29) finished with validation tests passing.

**v1.5 - Env Var & Transpose Fix (2026-01-21):**
- Phase 25: Env Var Fix (12/12 complete)
- Phase 26: Transpose Kernel Fix (3/3 complete)
- Phase 27: Device Property Infrastructure (4/4 complete)
- Phase 28: Debug Hygiene (4/4 complete)
- Phase 29: Validation & E2E (5/5 complete)

**v1.6 - FFI Device Props Fix (In Progress):**
- Phase 30: Immediate Bugfix (FFI-01, FFI-02) - COMPLETE (1/1 plans)
- Phase 31: Bindgen Infrastructure (FFI-03) - COMPLETE (1/1 plans)
- Phase 32: Offset Verification Test (FFI-04) - compile-time offset assertions (0/1 plans)
