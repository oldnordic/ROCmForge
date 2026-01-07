# Bug Hunt: Complete Bug List

**Date**: 2026-01-06
**Agent**: debugger
**Hunt Scope**: Full ROCmForge codebase after code drift fixes

---

## Bug Summary

**Total Bugs Found**: 12
- Critical: 3 (compilation failures)
- High-Severity: 5 (code quality violations)
- Medium-Severity: 1 (cleanup needed)
- Low-Severity: 3 (documentation/style)

---

## Complete Bug List

### Critical Bugs

#### BUG #1: Missing File Reference - `gguf_loader.rs`
- **ID**: BUG-2026-01-06-001
- **Severity**: CRITICAL
- **Location**: `src/loader/mod.rs:4,9`
- **Impact**: Compilation succeeds but creates ambiguous imports, breaks test imports
- **Root Cause**: File was deleted during refactoring but module declaration not removed
- **Files Affected**: 15+ files importing from `gguf_loader`
- **Fix**: Remove 2 lines from `mod.rs`, update `engine.rs` import
- **Estimated Time**: 5 minutes
- **Status**: ⬜ TODO

#### BUG #2: Broken Test File - `test_direct_cpu.rs`
- **ID**: BUG-2026-01-06-002
- **Severity**: CRITICAL
- **Location**: `tests/test_direct_cpu.rs:5-7,28`
- **Impact**: Test compilation fails with 3 errors
- **Root Cause**: Circular imports in test module structure
- **Errors**:
  1. Cannot glob-import module into itself (line 5)
  2. Cannot glob-import cpu module into itself (line 7)
  3. Type annotations needed (line 28)
- **Fix**: Remove circular imports, add direct import, add type annotation
- **Estimated Time**: 10 minutes
- **Status**: ⬜ TODO

#### BUG #3: Ambiguous Glob Re-Exports
- **ID**: BUG-2026-01-06-003
- **Severity**: CRITICAL
- **Location**: `src/loader/mod.rs:8-9`
- **Impact**: Compiler warnings, broken imports
- **Root Cause**: Both `gguf::*` and non-existent `gguf_loader::*` export same types
- **Warning**: `ambiguous glob re-exports` for `GgufLoader` and `GgufTensor`
- **Fix**: Remove `pub use gguf_loader::*;`
- **Estimated Time**: 1 minute (done with BUG #1)
- **Status**: ⬜ TODO

---

### High-Severity Bugs

#### BUG #4: Non-CamelCase Enum Variants - MXFP6
- **ID**: BUG-2026-01-06-004
- **Severity**: HIGH
- **Location**: `src/loader/gguf.rs:379-380`
- **Impact**: Violates Rust naming conventions, affects 25+ code locations and 39 doc references
- **Issues**:
  - `MXFP6_E2M3` should be `Mxfp6E2m3`
  - `MXFP6_E3M2` should be `Mxfp6E3m2`
- **Files Affected**:
  - `src/loader/gguf.rs` (definition + 7 usages)
  - `src/loader/mxfp_tests.rs` (8 usages)
  - `src/bin/test_gguf_load.rs` (2 usages)
  - `tests/gguf_loader_structural_tests.rs` (2 usages)
  - `tests/mxfp_unit_tests.rs` (6 usages)
  - Documentation (39 references)
- **Fix**: Rename enums and update all 66+ references
- **Estimated Time**: 1-2 hours
- **Breaking Change**: Yes (API change)
- **Status**: ⬜ TODO

#### BUG #5: Non-CamelCase Struct Name - f16
- **ID**: BUG-2026-01-06-005
- **Severity**: HIGH
- **Location**: `src/loader/gguf.rs:1328`
- **Impact**: Violates Rust naming conventions
- **Issue**: `struct f16(u16)` should be `struct F16(u16)`
- **Fix**: Rename struct and update all usages
- **Estimated Time**: 5 minutes
- **Breaking Change**: Yes (if struct is public API)
- **Status**: ⬜ TODO

#### BUG #6: Non-UpperCase Constants - HIP
- **ID**: BUG-2026-01-06-006
- **Severity**: HIGH
- **Location**: `src/backend/hip_backend.rs:48-51`
- **Impact**: Violates Rust naming conventions for constants
- **Issues**:
  - `hipMemcpyHostToDevice` → `HIP_MEMCPY_HOST_TO_DEVICE`
  - `hipMemcpyDeviceToHost` → `HIP_MEMCPY_DEVICE_TO_HOST`
  - `hipMemcpyDeviceToDevice` → `HIP_MEMCPY_DEVICE_TO_DEVICE`
  - `hipSuccess` → `HIP_SUCCESS`
- **Fix**: Rename all 4 constants
- **Estimated Time**: 10 minutes
- **Breaking Change**: Yes (if constants are public API)
- **Status**: ⬜ TODO

#### BUG #7: Non-SnakeCase Parameters - BLAS
- **ID**: BUG-2026-01-06-007
- **Severity**: HIGH
- **Location**: `src/backend/hip_blas.rs:127-132`
- **Impact**: Violates Rust naming conventions
- **Issues**:
  - Parameter `A` should be `a` or `matrix_a`
  - Parameter `B` should be `b` or `matrix_b`
  - Parameter `C` should be `c` or `matrix_c`
- **Fix**: Rename parameters and update function body
- **Estimated Time**: 5 minutes
- **Breaking Change**: Yes (if function is public API)
- **Status**: ⬜ TODO

#### BUG #8: Unused Code - 81 Warnings
- **ID**: BUG-2026-01-06-008
- **Severity**: MEDIUM
- **Location**: Throughout codebase
- **Impact**: Code clutter, maintenance burden
- **Breakdown**:
  - Unused imports: 20+
  - Unused variables: 40+
  - Unused functions/structs: 15+
  - Unnecessary mut: 6+
- **Notable Unused Code**:
  - `src/attention/kernels.rs`: Entire `KernelCache` system
  - `src/backend/hip_backend.rs`: Multiple FFI bindings
  - `src/model/execution_plan.rs`: 6+ weight mapping functions
- **Fix**: Run `cargo fix` and manually review
- **Estimated Time**: 30 minutes
- **Status**: ⬜ TODO

---

### Low-Severity Issues

#### ISSUE #1: Outdated Documentation
- **ID**: ISSUE-2026-01-06-001
- **Severity**: LOW
- **Location**: Multiple documentation files
- **Impact**: Documentation doesn't match code
- **Issue**: 39 references to old enum names `MXFP6_E2M3` and `MXFP6_E3M2`
- **Files Affected**:
  - `docs/codebase_audit.md`
  - `docs/AGENT_3_FIX_CHECKLIST.md`
  - `docs/AGENT_3_COMPREHENSIVE_BUG_REPORT.md`
  - `docs/AGENT_3_BUG_REPORT.md`
  - `docs/AGENT_3_FINAL_BUG_REPORT_2026-01-06.md`
- **Fix**: Update documentation after BUG #4 is fixed
- **Estimated Time**: 30 minutes
- **Status**: ⬜ TODO

#### ISSUE #2: Unnecessary Parentheses
- **ID**: ISSUE-2026-01-06-002
- **Severity**: LOW
- **Location**: `src/loader/gguf.rs:352`
- **Impact**: Style issue
- **Issue**: Unnecessary parentheses around block return value
- **Fix**: Remove parentheses
- **Estimated Time**: 1 minute
- **Status**: ⬜ TODO

#### ISSUE #3: Test File Import Paths
- **ID**: ISSUE-2026-01-06-003
- **Severity**: LOW
- **Location**: 6+ test files
- **Impact**: Tests will fail after BUG #1 is fixed
- **Issue**: Tests import from `gguf_loader` which doesn't exist
- **Files Affected**:
  - `tests/embedding_to_lmhead_tests.rs`
  - `tests/execution_plan_construction_tests.rs`
  - `tests/execution_plan_forward_pass_tests.rs`
  - `tests/gguf_loader_structural_tests.rs`
  - `tests/gguf_loader_tests.rs`
  - `tests/loader_tests.rs`
- **Fix**: Update imports from `gguf_loader` to `gguf` or direct loader module
- **Estimated Time**: 20 minutes
- **Status**: ⬜ TODO

---

## Bug Statistics

| Severity | Count | Estimated Time |
|----------|-------|----------------|
| Critical | 3 | 15 minutes |
| High | 5 | 2-3 hours |
| Medium | 1 | 30 minutes |
| Low | 3 | 50 minutes |
| **Total** | **12** | **3.5-4.5 hours** |

---

## Fix Priority Matrix

| Bug ID | Severity | Impact | Effort | Priority |
|--------|----------|--------|--------|----------|
| BUG-2026-01-06-001 | Critical | High | Low | P0 |
| BUG-2026-01-06-002 | Critical | High | Low | P0 |
| BUG-2026-01-06-003 | Critical | Medium | Low | P0 |
| BUG-2026-01-06-004 | High | High | High | P1 |
| BUG-2026-01-06-005 | High | Medium | Low | P1 |
| BUG-2026-01-06-006 | High | Medium | Low | P1 |
| BUG-2026-01-06-007 | High | Medium | Low | P1 |
| BUG-2026-01-06-008 | Medium | Low | Medium | P2 |
| ISSUE-2026-01-06-001 | Low | Low | Low | P3 |
| ISSUE-2026-01-06-002 | Low | Low | Low | P3 |
| ISSUE-2026-01-06-003 | Low | Medium | Low | P2 |

---

## Dependencies

Some bugs have dependencies on others:

- **BUG-2026-01-06-003** (ambiguous re-exports) is automatically fixed by **BUG-2026-01-06-001** (remove gguf_loader)
- **ISSUE-2026-01-06-001** (outdated docs) should be fixed after **BUG-2026-01-06-004** (rename MXFP6 enums)
- **ISSUE-2026-01-06-003** (test imports) should be fixed after **BUG-2026-01-06-001** (remove gguf_loader)

---

## Risk Assessment

| Bug | Risk Level | Reason |
|-----|------------|--------|
| BUG-2026-01-06-001 | LOW | Simple deletion, well-understood fix |
| BUG-2026-01-06-002 | LOW | Isolated to one test file |
| BUG-2026-01-06-003 | LOW | Automatic with BUG #1 |
| BUG-2026-01-06-004 | HIGH | 66+ changes, breaking API change |
| BUG-2026-01-06-005 | MEDIUM | Simple rename but need to verify all usages |
| BUG-2026-01-06-006 | MEDIUM | Simple rename but affects FFI code |
| BUG-2026-01-06-007 | LOW | Simple parameter rename |
| BUG-2026-01-06-008 | MEDIUM | Automatic fixes + manual review required |
| ISSUE-2026-01-06-001 | LOW | Documentation only |
| ISSUE-2026-01-06-002 | LOW | Trivial fix |
| ISSUE-2026-01-06-003 | LOW | Straightforward find-replace |

---

## Testing Requirements

After fixing bugs, run:

```bash
# Clean build
cargo clean && cargo build --lib

# Build all binaries
cargo build --bins

# Run all tests
cargo test --workspace

# Check for remaining issues
cargo clippy -- -D warnings
```

---

## Files Requiring Changes

**Summary**: 25+ files require changes

### Critical Files (3)
1. `src/loader/mod.rs`
2. `src/engine.rs`
3. `tests/test_direct_cpu.rs`

### High-Priority Files (10+)
4. `src/loader/gguf.rs`
5. `src/loader/mxfp_tests.rs`
6. `src/bin/test_gguf_load.rs`
7. `src/backend/hip_backend.rs`
8. `src/backend/hip_blas.rs`
9. `tests/gguf_loader_structural_tests.rs`
10. `tests/mxfp_unit_tests.rs`
11. `tests/embedding_to_lmhead_tests.rs`
12. `tests/execution_plan_construction_tests.rs`
13. `tests/execution_plan_forward_pass_tests.rs`

### Documentation Files (7)
14. `docs/codebase_audit.md`
15. `docs/AGENT_3_FIX_CHECKLIST.md`
16. `docs/AGENT_3_COMPREHENSIVE_BUG_REPORT.md`
17. `docs/AGENT_3_BUG_REPORT.md`
18. `docs/AGENT_3_FINAL_BUG_REPORT_2026-01-06.md`
19. `docs/BUG_HUNT_REPORT_2026-01-06.md` (this file)
20. `docs/BUG_HUNT_QUICKSUMMARY.md`

---

**Report Generated**: 2026-01-06
**Total Bugs Documented**: 12
**Next Action**: Begin Phase 1 fixes (Critical bugs)
