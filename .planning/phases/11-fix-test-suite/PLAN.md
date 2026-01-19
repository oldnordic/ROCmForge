# Phase 11: Fix Test Suite & Verify E2E - Execution Plan

**Phase:** 11
**Mode:** gap_closure

**Created:** 2026-01-19

---

## Frontmatter

```yaml
wave: 1
depends_on: [10]
gap_closure: true
autonomous: true
files_modified:
  - tests/loader_tests.rs
  - tests/embedding_to_lmhead_tests.rs
  - tests/transformer_integration_tests.rs
  - tests/kv_cache_tests.rs
  - tests/scheduler_tests.rs
  - tests/edge_case_tests.rs
  - tests/typed_view_tests.rs
  - tests/q_dequant_tests.rs
  - tests/attention_gpu_tests.rs
```

---

## Phase Goal

Fix test compilation errors and enable end-to-end verification with real GGUF models.

**Gap Closure:** Closes critical test compilation gap from v1.0 audit (98+ compilation errors).

---

## Problem Analysis

### Root Causes

**1. Missing `anyhow::Context` imports (75 occurrences)**
- Tests use `.context()` without importing `anyhow::Context` trait
- Files: kv_cache_tests.rs (38), loader_tests.rs (10), typed_view_tests.rs (10), edge_case_tests.rs (12), transformer_integration_tests.rs (5)

**2. Removed `element_size()` method (10 occurrences)**
- `GgufTensorType::element_size()` was removed from src/loader/tensor_type.rs
- Tests in loader_tests.rs (lines 173-181, 355) still reference this method

**3. Incorrect `crate::common` import path (3 files)**
- Tests reference `use crate::common::{...}` but common module is at tests/common/

**4. Duplicate `serial` import (1 occurrence)**
- attention_gpu_tests.rs imports serial_test::serial twice

---

## Tasks

### 11-01: Fix Test Compilation Errors

**Goal:** Resolve all 98+ test compilation errors

**Acceptance Criteria:**
- [ ] All test files compile without errors
- [ ] `cargo check --tests` passes
- [ ] `cargo test --tests` compiles (tests may skip without GPU)
- [ ] No unused import warnings

**Files to modify:**
- tests/kv_cache_tests.rs (38 fixes)
- tests/loader_tests.rs (11 fixes)
- tests/typed_view_tests.rs (10 fixes)
- tests/edge_case_tests.rs (12 fixes)
- tests/transformer_integration_tests.rs (5 fixes)
- tests/embedding_to_lmhead_tests.rs (1 fix)
- tests/q_dequant_tests.rs (1 fix)
- tests/attention_gpu_tests.rs (1 fix)
- src/loader/tensor_type.rs (restore element_size method if needed)

**Implementation Steps:**

1. Add `use anyhow::Context;` to affected test files
2. Restore `element_size()` method to `GgufTensorType` or update tests to use `block_size()`
3. Fix `crate::common` import paths
4. Remove duplicate `serial_test::serial` import

---

### 11-02: Verify E2E Flows with Real GGUF Models

**Goal:** Enable end-to-end testing with real GGUF models

**Acceptance Criteria:**
- [ ] E2E tests compile and run (may skip without model)
- [ ] ROCFORGE_TEST_MODEL environment variable support works
- [ ] Test suite documents model requirements
- [ ] At least one E2E test validates inference pipeline

**Implementation Steps:**

1. Verify E2E test infrastructure compiles
2. Create/update `tests/README_E2E_TESTS.md` with model requirements
3. Run E2E validation with real GGUF file (if available)
4. Document known issues

**Files to modify:**
- tests/e2e_inference_tests.rs (if fixes needed)
- tests/README_E2E_TESTS.md (update)

---

## Dependencies

**Phase 10 (Production Hardening):** Must be complete
- Error handling modules stable
- Test infrastructure patterns established

---

## Must Haves (Verification)

### Test Compilation
- [ ] All test files compile without errors
- [ ] `cargo check --tests` passes
- [ ] No critical test compilation errors

### E2E Verification
- [ ] E2E tests can run with real GGUF model
- [ ] Model requirements documented
- [ ] Graceful skip when model unavailable

---

## Definition of Done

- [ ] All 98+ compilation errors resolved
- [ ] `cargo check --tests` passes with warnings only
- [ ] `cargo test --tests` compiles successfully
- [ ] E2E tests can run with real GGUF model
- [ ] Test count documented in STATE.md

---

**Total Estimated Time:** 1-2 hours
