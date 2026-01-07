# Code Drift Fix Verification Checklist

**Date**: 2026-01-06
**Verification Status**: INCOMPLETE ❌

---

## Quick Status Summary

| Item | Status | Details |
|------|--------|---------|
| 1. Enum Naming (MXFP6_E2M3 → Mxfp6E2m3) | ⚠️ PARTIAL | Definition fixed, 7 files still use old names |
| 2. Enum Naming (MXFP6_E3M2 → Mxfp6E3m2) | ⚠️ PARTIAL | Definition fixed, 7 files still use old names |
| 3. Enum Naming (MXFP4 → Mxfp4) | ⚠️ PARTIAL | Definition fixed, 7 files still use old names |
| 4. Duplicate GGUF Loaders | 🟡 CLARIFY | Not actually duplicates, but import issues exist |
| 5. Test Suite Consolidation | 🟡 CLARIFY | Two similar test files exist, both broken |
| 6. Kernel Naming Correction | ✅ VERIFIED | build.rs matches actual kernel names |

---

## Detailed Verification Results

### ✅ Items Verified Correct

#### 1. Kernel Naming Correction
**File**: `/home/feanor/Projects/ROCmForge/build.rs:54`
```rust
("kernels/mxfp_dequant.hip", "MXFP_DEQUANT_HSACO", "mxfp4_to_fp16_kernel"),
```
**Verification**: Matches kernel function name in `kernels/mxfp_dequant.hip:85`
```hip
extern "C" __global__ void mxfp4_to_fp16_kernel(...)
```

**Status**: ✅ NO ISSUES - Kernel name correctly references actual function

---

### ⚠️ Items Partially Fixed (Critical Issues Remain)

#### 2. Enum Naming Corrections

**What Was Fixed**:
- ✅ Enum definition in `src/loader/gguf.rs:378-380`
- ✅ `from_u32()` method in `src/loader/gguf.rs:393-395`
- ✅ `to_string()` method in `src/loader/gguf.rs:409-411`
- ✅ `element_size()` pattern match (line 439)

**What Was MISSED** (Critical):
- ❌ `src/loader/gguf.rs:531` - `MXFP4` should be `Mxfp4`
- ❌ `src/loader/mxfp_tests.rs` - Multiple lines (422-452)
- ❌ `tests/mxfp_unit_tests.rs` - Multiple lines (344-370)
- ❌ `src/bin/test_gguf_load.rs` - Lines 49-51
- ❌ `tests/gguf_loader_structural_tests.rs` - Lines 81-83

**Impact**: **COMPILATION FAILS** - Cannot build or test

**Files Affected**: 7 files with outdated enum references

---

### ❌ Items Not Fixed (Compilation Blockers)

#### 3. Import Errors in engine.rs

**Error #1**: `src/engine.rs:5`
```rust
// WRONG - GgufModel does not exist
use crate::loader::{GgufModel, OnnxLoader};
```
**Fix Required**:
```rust
use crate::loader::{GgufLoader, OnnxLoader};
```

**Error #2**: `src/engine.rs:129`
```rust
// WRONG - gguf_loader module doesn't exist
let gguf_loader = Arc::new(RwLock::new(crate::loader::gguf_loader::GgufLoader::new()));
```
**Fix Required**:
```rust
let gguf_loader = Arc::new(RwLock::new(crate::loader::GgufLoader::new()));
```

**Impact**: **COMPILATION FAILS** - 2 errors in engine.rs

---

### 🟡 Items Needing Clarification

#### 4. Duplicate GGUF Loaders

**Finding**: Two files exist but are NOT duplicates:
- `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs` (1,506 lines)
  - Contains: `GgufLoader` struct, `GgufTensorType` enum
  - Status: **Active implementation**

- `/home/feanor/Projects/ROCmForge/src/loader/gguf_loader.rs` (550 lines)
  - Status: **Exists but purpose unclear**
  - Needs: Content verification to determine if duplicate or distinct

**Real Issue**: Import references to non-existent module path

**Required Action**:
1. Verify if `gguf_loader.rs` contains unique functionality
2. If duplicate: Delete file
3. If distinct: Document its purpose
4. Fix `engine.rs:129` to use correct module path

#### 5. Test Suite Consolidation

**Finding**: Two similar test files exist:
- `src/loader/mxfp_tests.rs` (454 lines) - More detailed, better documented
- `tests/mxfp_unit_tests.rs` (372 lines) - Simpler structure

**Status**: Both files use old enum names and will fail to compile

**Required Clarification**:
- Are these duplicates serving different purposes?
- If duplicates: Which one should be kept?
- If distinct: What's the difference?

**Current Impact**: Both files broken due to enum naming issues

---

## Compilation Status

```bash
$ cargo build
```

**Result**: ❌ FAILED - 3 compilation errors

```
error[E0432]: unresolved import `crate::loader::GgufModel`
  --> src/engine.rs:5:21

error[E0433]: failed to resolve: could not find `gguf_loader` in `loader`
  --> src/engine.rs:129:63

error[E0599]: no variant or associated item named `MXFP4` found for enum `GgufTensorType`
  --> src/loader/gguf.rs:531:29
```

---

## Test Status

**Status**: ❌ NOT RUN - Compilation must succeed first

**Expected Failures** (once build succeeds):
- All tests in `src/loader/mxfp_tests.rs` - Will fail to compile
- All tests in `tests/mxfp_unit_tests.rs` - Will fail to compile
- `src/bin/test_gguf_load.rs` - Will fail to compile
- Tests in `tests/gguf_loader_structural_tests.rs` - Will fail to compile

---

## Required Fixes (Priority Order)

### 🔴 CRITICAL (Must Fix)

1. **Fix engine.rs imports** (2 changes)
   ```rust
   // Line 5
   - use crate::loader::{GgufModel, OnnxLoader};
   + use crate::loader::{GgufLoader, OnnxLoader};

   // Line 129
   - let gguf_loader = Arc::new(RwLock::new(crate::loader::gguf_loader::GgufLoader::new()));
   + let gguf_loader = Arc::new(RwLock::new(crate::loader::GgufLoader::new()));
   ```

2. **Fix gguf.rs:531**
   ```rust
   - GgufTensorType::MXFP4 => {
   + GgufTensorType::Mxfp4 => {
   ```

3. **Fix all test files** (4 files, ~50+ lines total)
   - Search and replace: `MXFP4` → `Mxfp4`
   - Search and replace: `MXFP6_E2M3` → `Mxfp6E2m3`
   - Search and replace: `MXFP6_E3M2` → `Mxfp6E3m2`

   Files to update:
   - `src/loader/mxfp_tests.rs`
   - `tests/mxfp_unit_tests.rs`
   - `src/bin/test_gguf_load.rs`
   - `tests/gguf_loader_structural_tests.rs`

### 🟡 HIGH PRIORITY (Should Fix)

4. **Verify and fix gguf_loader.rs**
   - Determine if file is duplicate or distinct
   - If duplicate: Delete and verify no references remain
   - If distinct: Document purpose clearly

5. **Clarify test file structure**
   - Decide: Keep both `mxfp_tests.rs` and `mxfp_unit_tests.rs` or consolidate?
   - Update test configuration if needed
   - Ensure tests are properly organized

### 🔵 VERIFICATION (Do After Fixes)

6. **Run full build**
   ```bash
   cargo build --release
   ```

7. **Run full test suite**
   ```bash
   cargo test --all
   ```

8. **Check for warnings**
   ```bash
   cargo build 2>&1 | grep -i warning
   ```

9. **Verify enum consistency**
   ```bash
   grep -rn "GgufTensorType::\(MXFP\|Mxfp\)" --include="*.rs"
   ```

---

## Quick Reference: File Changes Needed

| File | Lines to Change | Type |
|------|----------------|------|
| `src/engine.rs` | 5, 129 | Import fixes |
| `src/loader/gguf.rs` | 531 | Enum name |
| `src/loader/mxfp_tests.rs` | 422-452 | Enum names |
| `tests/mxfp_unit_tests.rs` | 344-370 | Enum names |
| `src/bin/test_gguf_load.rs` | 49-51 | Enum names |
| `tests/gguf_loader_structural_tests.rs` | 81-83 | Enum names |

**Total Files**: 6 files
**Estimated Lines**: ~60 lines to change

---

## Search Commands for Verification

### Find all enum references (old and new)
```bash
grep -rn "GgufTensorType::\(MXFP\|Mxfp\)" --include="*.rs" /home/feanor/Projects/ROCmForge/src
```

### Find all old enum names (should be empty after fix)
```bash
grep -rn "GgufTensorType::\(MXFP4\|MXFP6_E[23]M[23]\)" --include="*.rs"
```

### Find all new enum names (should match expected count)
```bash
grep -rn "GgufTensorType::\(Mxfp4\|Mxfp6E[23]m[23]\)" --include="*.rs"
```

---

## Summary

### What Works
- ✅ Enum definition correctly renamed
- ✅ Kernel names verified correct
- ✅ Most implementation references updated

### What's Broken
- ❌ 3 compilation errors blocking build
- ❌ All test files use old enum names
- ❌ Import errors in engine.rs
- ❌ Incomplete enum reference updates

### What's Unclear
- 🟡 Purpose of gguf_loader.rs
- 🟡 Whether to keep or consolidate test files

### Overall Assessment

**Status**: ❌ **INCOMPLETE - Cannot Recommend Merge**

**Blocking Issues**:
- Code does not compile
- Tests cannot run
- Multiple critical references not updated

**Recommendation**:
1. Fix all 3 compilation errors first
2. Update all test files with correct enum names
3. Clarify test file organization
4. Re-run full verification

**Verification Report**: `docs/CODE_DRIFT_FIX_VERIFICATION.md`
**Date**: 2026-01-06
**Reviewer**: code-reviewer
