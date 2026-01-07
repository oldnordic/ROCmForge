# Bug Hunt Fix Checklist

**Date**: 2026-01-06
**Agent**: debugger

Use this checklist to track progress on fixing bugs found during the bug hunt.

---

## Phase 1: Critical Fixes (Compilation Blockers)

### BUG #1: Remove Missing File Reference `gguf_loader.rs`

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 5 minutes

**Steps**:
- [ ] Edit `/home/feanor/Projects/ROCmForge/src/loader/mod.rs`
  - [ ] Remove line 4: `pub mod gguf_loader;`
  - [ ] Remove line 9: `pub use gguf_loader::*;`
- [ ] Edit `/home/feanor/Projects/ROCmForge/src/engine.rs`
  - [ ] Line 129: Change `crate::loader::gguf_loader::GgufLoader` to `crate::loader::gguf::GgufLoader`
- [ ] Verify: `cargo build --lib` succeeds
- [ ] Verify: No "ambiguous glob re-exports" warnings

**Files Changed**:
- [ ] `src/loader/mod.rs`
- [ ] `src/engine.rs`

---

### BUG #2: Fix Broken Test File `test_direct_cpu.rs`

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 10 minutes

**Steps**:
- [ ] Edit `/home/feanor/Projects/ROCmForge/tests/test_direct_cpu.rs`
  - [ ] Remove lines 3-10 (broken mod re-export structure)
  - [ ] Add proper import: `use rocmforge::attention::cpu::CpuBackend;`
  - [ ] Fix line 28: Add type annotation for `output_data`
- [ ] Verify: Test file compiles with `cargo test --test test_direct_cpu`
- [ ] Verify: Test runs without errors

**Files Changed**:
- [ ] `tests/test_direct_cpu.rs`

---

### BUG #3: Fix Ambiguous Re-Exports

**Status**: ⬜ TODO (Combined with BUG #1)
**Assigned To**: _______
**Estimated Time**: Already done in BUG #1

**Note**: This is automatically fixed when removing `gguf_loader` from mod.rs

---

## Phase 2: High-Severity Fixes (Code Quality)

### BUG #4: Rename MXFP6 Enum Variants

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 1-2 hours

**Steps**:
1. [ ] Rename enum variants in `src/loader/gguf.rs:379-380`
   - [ ] `MXFP6_E2M3` → `Mxfp6E2m3`
   - [ ] `MXFP6_E3M2` → `Mxfp6E3m2`
   - [ ] Update string representations in `as_str()` method

2. [ ] Update references in `src/loader/gguf.rs`
   - [ ] Line 394: `21 => Ok(GgufTensorType::Mxfp6E2m3)`
   - [ ] Line 395: `22 => Ok(GgufTensorType::Mxfp6E3m2)`
   - [ ] Line 410: Update string `"MXFP6_E2M3"` → `"Mxfp6E2m3"`
   - [ ] Line 411: Update string `"MXFP6_E3M2"` → `"Mxfp6E3m2"`
   - [ ] Line 439: Update match arm
   - [ ] Line 536: Update match arm
   - [ ] Line 1138: Update match arm

3. [ ] Update `src/loader/mxfp_tests.rs`
   - [ ] Line 423: Update assertion
   - [ ] Line 424: Update assertion
   - [ ] Line 436: Update match arm
   - [ ] Line 440: Update match arm
   - [ ] Line 451: Update assertion
   - [ ] Line 452: Update assertion
   - [ ] Line 415: Update comment

4. [ ] Update `src/bin/test_gguf_load.rs`
   - [ ] Line 50: Update string
   - [ ] Line 51: Update string

5. [ ] Update test files:
   - [ ] `tests/gguf_loader_structural_tests.rs` (lines 82-83)
   - [ ] `tests/mxfp_unit_tests.rs` (lines 344-345, 356, 360, 369-370)

6. [ ] Update documentation (39 references):
   - [ ] `docs/codebase_audit.md`
   - [ ] `docs/AGENT_3_FIX_CHECKLIST.md`
   - [ ] `docs/AGENT_3_COMPREHENSIVE_BUG_REPORT.md`
   - [ ] `docs/AGENT_3_BUG_REPORT.md`
   - [ ] `docs/AGENT_3_FINAL_BUG_REPORT_2026-01-06.md`

7. [ ] Verify: `cargo build --lib` succeeds
8. [ ] Verify: `cargo test --workspace` succeeds

**Files Changed**:
- [ ] `src/loader/gguf.rs` (9 locations)
- [ ] `src/loader/mxfp_tests.rs` (8 locations)
- [ ] `src/bin/test_gguf_load.rs` (2 locations)
- [ ] `tests/gguf_loader_structural_tests.rs` (2 locations)
- [ ] `tests/mxfp_unit_tests.rs` (6 locations)
- [ ] Documentation files (39 references)

**Total Locations**: 66+ changes

---

### BUG #5: Rename f16 Struct

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 5 minutes

**Steps**:
- [ ] Edit `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
  - [ ] Line 1328: Rename `struct f16(u16)` to `struct F16(u16)`
- [ ] Search for all `f16` struct usages (not type annotations)
- [ ] Update all usages to `F16`
- [ ] Verify: `cargo build --lib` succeeds

**Files Changed**:
- [ ] `src/loader/gguf.rs`

---

### BUG #6: Rename HIP Constants

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 10 minutes

**Steps**:
- [ ] Edit `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
  - [ ] Line 48: `hipMemcpyHostToDevice` → `HIP_MEMCPY_HOST_TO_DEVICE`
  - [ ] Line 49: `hipMemcpyDeviceToHost` → `HIP_MEMCPY_DEVICE_TO_HOST`
  - [ ] Line 50: `hipMemcpyDeviceToDevice` → `HIP_MEMCPY_DEVICE_TO_DEVICE`
  - [ ] Line 51: `hipSuccess` → `HIP_SUCCESS`
- [ ] Search for all usages of these constants
- [ ] Update all references (if any outside definition)
- [ ] Verify: `cargo build --lib` succeeds

**Files Changed**:
- [ ] `src/backend/hip_backend.rs`

---

### BUG #7: Rename BLAS Parameters

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 5 minutes

**Steps**:
- [ ] Edit `/home/feanor/Projects/ROCmForge/src/backend/hip_blas.rs`
  - [ ] Line 127: Rename parameter `A` to `a` or `matrix_a`
  - [ ] Line 129: Rename parameter `B` to `b` or `matrix_b`
  - [ ] Line 132: Rename parameter `C` to `c` or `matrix_c`
- [ ] Update function body to use new parameter names
- [ ] Verify: `cargo build --lib` succeeds

**Files Changed**:
- [ ] `src/backend/hip_blas.rs`

---

### BUG #8: Clean Up Unused Code

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 30 minutes

**Steps**:
- [ ] Run automatic fixes:
  ```bash
  cargo fix --lib -p rocmforge --allow-dirty
  cargo fix --bin "rocmforge_cli" -p rocmforge --allow-dirty
  ```
- [ ] Review changes and commit
- [ ] Run `cargo clippy -- -W dead_code` to find dead code
- [ ] Decide for each dead code item:
  - [ ] Remove if truly unused
  - [ ] Mark with `#[allow(dead_code)]` if keeping for future use
  - [ ] Actually use the code if it should be used
- [ ] Verify: `cargo build --lib` succeeds
- [ ] Verify: Warning count reduced significantly

**Files Changed**:
- [ ] Multiple files (automatic fixes)

---

## Phase 3: Test File Updates

### ISSUE #3: Update Test Imports

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 20 minutes

**Steps**:
- [ ] For each test file, update imports from `gguf_loader` to `gguf`:
  - [ ] `tests/embedding_to_lmhead_tests.rs` (lines 3, 95, 100, 171, 247, 345, 354)
  - [ ] `tests/execution_plan_construction_tests.rs` (lines 17, 41, 113)
  - [ ] `tests/execution_plan_forward_pass_tests.rs` (lines 16, 61, 109, 144, 192, 214, 257)
  - [ ] `tests/gguf_loader_structural_tests.rs` (check all imports)
  - [ ] `tests/gguf_loader_tests.rs` (check all imports)
  - [ ] `tests/loader_tests.rs` (check all imports)

- [ ] Search for any remaining `gguf_loader` references:
  ```bash
  grep -rn "gguf_loader" tests/
  ```
- [ ] Verify: `cargo test --workspace` succeeds

**Pattern**:
```rust
// OLD:
use rocmforge::loader::gguf_loader::{GgufLoader, GgufModel, GgufTensor};
use rocmforge::loader::gguf_loader::GgufDataType;

// NEW:
use rocmforge::loader::{GgufLoader, GgufModel, GgufTensor, GgufDataType};
```

**Files Changed**:
- [ ] 6+ test files

---

## Phase 4: Verification

### Final Verification Checklist

**Status**: ⬜ TODO
**Assigned To**: _______
**Estimated Time**: 15 minutes

**Steps**:
- [ ] Clean build:
  ```bash
  cargo clean
  cargo build --lib
  ```
  Expected: 0 errors, minimal warnings

- [ ] Build all binaries:
  ```bash
  cargo build --bins
  ```
  Expected: 0 errors, minimal warnings

- [ ] Run all tests:
  ```bash
  cargo test --workspace
  ```
  Expected: All tests pass

- [ ] Check for remaining issues:
  ```bash
  grep -rn "gguf_loader" src/ tests/
  grep -rn "MXFP6_E2M3\|MXFP6_E3M2" src/ tests/
  ```
  Expected: Only in comments/docs (not code)

- [ ] Run clippy:
  ```bash
  cargo clippy -- -D warnings
  ```
  Expected: No critical warnings

- [ ] Count warnings:
  ```bash
  cargo build 2>&1 | grep -c "^warning:"
  ```
  Expected: Significantly reduced from 81

---

## Progress Summary

| Phase | Bugs | Completed | Remaining |
|-------|------|-----------|-----------|
| Phase 1: Critical | 3 | 0 | 3 |
| Phase 2: High-Severity | 5 | 0 | 5 |
| Phase 3: Test Updates | 1 | 0 | 1 |
| Phase 4: Verification | 1 | 0 | 1 |
| **Total** | **10** | **0** | **10** |

**Overall Progress**: 0% (0/10 completed)

---

## Notes

- Execute phases in order (Phase 1 → Phase 2 → Phase 3 → Phase 4)
- Each phase should be completed and verified before moving to the next
- Update this checklist as you complete each item
- If you find additional bugs not in this checklist, add them
- If a bug is already fixed, mark it as complete with a note

---

**Last Updated**: 2026-01-06
**Next Review**: After Phase 1 completion
