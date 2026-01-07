# Bug Hunt Quick Summary

**Date**: 2026-01-06
**Agent**: debugger
**Scope**: Post code-drift-fix bug hunt

---

## Critical Bugs (Must Fix)

### 1. Missing File `gguf_loader.rs`
- **File**: `src/loader/mod.rs:4`
- **Fix**: Remove `pub mod gguf_loader;` and `pub use gguf_loader::*;`
- **Files to Update**: 2 lines in `mod.rs`, 1 line in `engine.rs:129`

### 2. Broken Test `test_direct_cpu.rs`
- **File**: `tests/test_direct_cpu.rs:5-7`
- **Fix**: Remove circular imports, use direct imports
- **Impact**: Prevents test compilation

### 3. Ambiguous Re-Exports
- **File**: `src/loader/mod.rs:8-9`
- **Fix**: Remove `pub use gguf_loader::*;` (file doesn't exist)
- **Impact**: Compiler warnings, broken imports

---

## High-Severity Issues

### 4. MXFP6 Enum Naming
- **Files**: 25+ locations
- **Issue**: `MXFP6_E2M3` and `MXFP6_E3M2` violate UpperCamelCase
- **Fix**: Rename to `Mxfp6E2m3` and `Mxfp6E3m2`
- **Impact**: Breaking API change, affects tests and docs

### 5. f16 Struct Naming
- **File**: `src/loader/gguf.rs:1328`
- **Issue**: `struct f16` violates UpperCamelCase
- **Fix**: Rename to `F16`

### 6. HIP Constants Naming
- **File**: `src/backend/hip_backend.rs:48-51`
- **Issue**: 4 constants use lowerCamelCase instead of SCREAMING_SNAKE_CASE
- **Fix**: `hipMemcpy*` → `HIP_MEMCPY_*`, `hipSuccess` → `HIP_SUCCESS`

### 7. BLAS Parameters Naming
- **File**: `src/backend/hip_blas.rs:127-132`
- **Issue**: Parameters `A`, `B`, `C` violate snake_case
- **Fix**: Rename to `a`, `b`, `c` or `matrix_a`, `matrix_b`, `matrix_c`

### 8. Unused Code (81 warnings)
- **Scope**: Entire codebase
- **Fix**: Run `cargo fix` and manually review
- **Impact**: Code clutter, maintenance burden

---

## Quick Fix Commands

```bash
# Phase 1: Critical fixes (do these first)
# 1. Remove gguf_loader references from mod.rs
# 2. Fix test_direct_cpu.rs imports
# 3. Fix engine.rs import path

# Phase 2: Apply automatic fixes
cargo fix --lib -p rocmforge --allow-dirty
cargo fix --bin "rocmforge_cli" -p rocmforge --allow-dirty

# Phase 3: Manual fixes
# - Rename MXFP6 enum variants (25+ locations)
# - Rename f16 → F16
# - Rename HIP constants
# - Rename BLAS parameters

# Phase 4: Verify
cargo build --lib 2>&1 | grep -E "^error"
cargo test --workspace 2>&1 | grep -E "^error"
```

---

## Files Requiring Changes

### Critical (3 files)
- `src/loader/mod.rs` (2 lines)
- `tests/test_direct_cpu.rs` (imports)
- `src/engine.rs:129` (import path)

### High-Priority (30+ files)
- `src/loader/gguf.rs` (enum definitions, 7 usages)
- `src/loader/mxfp_tests.rs` (8 usages)
- `src/bin/test_gguf_load.rs` (2 usages)
- `tests/gguf_loader_structural_tests.rs` (2 usages)
- `tests/mxfp_unit_tests.rs` (6 usages)
- Plus 20+ more files with enum references

### Documentation (7 files)
- 39 references to old enum names across docs/

---

## Priority Order

1. **Fix critical bugs** (15 minutes)
2. **Apply cargo fix** (5 minutes)
3. **Rename enums** (1-2 hours)
4. **Update tests** (30 minutes)
5. **Update docs** (30 minutes)

**Total estimated time**: 2-4 hours

---

## Verification

```bash
# After fixes, run:
cargo build --lib && cargo test --workspace
```

Expected result: 0 compilation errors, minimal warnings.

---

**Full Report**: See `docs/BUG_HUNT_REPORT_2026-01-06.md`
