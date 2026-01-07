# Test Issues Checklist - ROCmForge

**Generated**: 2026-01-06  
**Total Issues Identified**: 24

---

## Compilation Errors (2) - CRITICAL

- [ ] **Fix**: `/home/feanor/Projects/ROCmForge/tests/loader_tests.rs`
  - Line 4: Change `GgufDataType, GgufModel` → `GgufTensorType, GgufLoader`
  - Line 320-330: Add type annotations to prop_assert
  - **Blocking**: Yes - prevents all tests from running

- [ ] **Fix**: `/home/feanor/Projects/ROCmForge/tests/embedding_to_lmhead_tests.rs`
  - Line 3: Change `gguf_loader` → `gguf`
  - Throughout: Update type names and API calls
  - **Blocking**: Yes - uses obsolete module path

---

## Files to Delete (9)

These are NOT test files - they're binaries or debug scripts:

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/simple_test.rs`
  - Reason: Contains `fn main()`, not a test
  - Alternative: Move to `examples/` if needed

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/test_hip_minimal.rs`
  - Reason: Contains `fn main()`, not a test
  - Alternative: Move to `examples/` if needed

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/minimal_hip_test.rs`
  - Reason: Duplicate of `test_hip_minimal.rs`, not a test
  - Recommendation: Just delete

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/test_cpu_fallback.rs`
  - Reason: Contains `fn main()`, not a test
  - Alternative: Move to `examples/` if needed

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/test_direct_cpu.rs`
  - Reason: Contains `fn main()`, not a test
  - Alternative: Move to `examples/` if needed

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/test_attention_debug.rs`
  - Reason: Debug script, not a test
  - Recommendation: Just delete

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/debug_test.rs`
  - Reason: Temporary debugging file
  - Recommendation: Just delete

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/debug_hip_backend.rs`
  - Reason: Temporary debugging file
  - Alternative: Move to `scripts/debug/` if needed for development

- [ ] **Delete**: `/home/feanor/Projects/ROCmForge/tests/engine_crash_test.rs`
  - Reason: Crash reproduction test (temporary)
  - Alternative: If crash is fixed, delete; otherwise convert to proper regression test

---

## Duplicate Tests to Remove (4)

- [ ] **Remove duplicate** from: `/home/feanor/Projects/ROCmForge/tests/multilayer_pipeline_tests.rs:84`
  - Test: `test_model_runtime_creation`
  - Keep in: `/home/feanor/Projects/ROCmForge/tests/model_runtime_tests.rs:14`

- [ ] **Remove duplicate** from: `/home/feanor/Projects/ROCmForge/tests/glm_model_tests.rs:226`
  - Test: `test_model_runtime_creation`
  - Keep in: `/home/feanor/Projects/ROCmForge/tests/model_runtime_tests.rs:14`

- [ ] **Remove duplicate** from: `/home/feanor/Projects/ROCmForge/tests/execution_plan_and_decode_tests.rs:21`
  - Test: `test_execution_plan_construction`
  - Keep in: `/home/feanor/Projects/ROCmForge/tests/execution_plan_construction_tests.rs:14`

- [ ] **Remove duplicate** from: `/home/feanor/Projects/ROCmForge/tests/execution_plan_forward_pass_tests.rs:59`
  - Test: `test_embedding_lookup`
  - Keep in: `/home/feanor/Projects/ROCmForge/tests/embedding_to_lmhead_tests.rs:142`

---

## Test Files to Create (5)

- [ ] **Create**: `/home/feanor/Projects/ROCmForge/tests/http_server_tests.rs`
  - Purpose: Test HTTP endpoints, request handling, error responses
  - Priority: HIGH (production API untested)
  - Tests needed: 10+

- [ ] **Create**: `/home/feanor/Projects/ROCmForge/tests/sampler_integration_tests.rs`
  - Purpose: Test temperature, top-k, top-p, repetition penalty
  - Priority: HIGH
  - Tests needed: 8+

- [ ] **Create**: `/home/feanor/Projects/ROCmForge/tests/gpu_memory_tests.rs`
  - Purpose: Test memory exhaustion, buffer reuse, allocation patterns
  - Priority: MEDIUM
  - Tests needed: 5+

- [ ] **Create**: `/home/feanor/Projects/ROCmForge/tests/attention_gpu_integration_tests.rs`
  - Purpose: Test QKV computation, attention scores, causal masks
  - Priority: MEDIUM
  - Tests needed: 8+

- [ ] **Create**: `/home/feanor/Projects/ROCmForge/tests/tensor_operations_tests.rs`
  - Purpose: Test matrix operations, different sizes, transpose
  - Priority: LOW
  - Tests needed: 6+

---

## Edge Case Tests to Add (3)

- [ ] **Add edge cases** to existing test files:
  - Empty sequences
  - Maximum sequence lengths
  - Non-power-of-2 dimensions
  - Zero variance in RMSNorm
  - Overflow/underflow in SwiGLU

---

## Compiler Warnings (158 total)

- [ ] **Fix**: Run `cargo fix --lib --tests` (auto-fixes ~100 warnings)
- [ ] **Fix manually**: Unused variables (prefix with `_`)
- [ ] **Fix manually**: Dead code (remove or allow)
- [ ] **Fix manually**: Naming conventions (`f16` → `F16`, etc.)

---

## Verification Steps

After fixes:

1. [ ] Verify compilation: `cargo test --workspace --no-run`
2. [ ] Run all tests: `cargo test --workspace`
3. [ ] Count test results: Document pass/fail counts
4. [ ] Check for new warnings
5. [ ] Update test documentation

---

## Priority Order

### Phase 1: Critical (Do First)
1. Fix 2 compilation errors
2. Delete 9 non-test files
3. Verify tests compile

### Phase 2: High Priority (Do Second)
4. Remove 4 duplicate tests
5. Run full test suite
6. Document results

### Phase 3: Medium Priority (Do Next)
7. Create HTTP server tests
8. Create sampler integration tests
9. Create GPU memory tests

### Phase 4: Low Priority (Do Later)
10. Fix compiler warnings
11. Add edge case tests
12. Add remaining integration tests

---

## Time Estimates

- Phase 1: 3-4 hours
- Phase 2: 1-2 hours
- Phase 3: 8-10 hours
- Phase 4: 6-8 hours

**Total**: 18-24 hours to reach 90% test health

---

## Notes

- Current test count: 343 functions
- Test health score: 68%
- Target health score: 90%
- Main blocker: Compilation errors

All file paths are absolute for easy reference.
