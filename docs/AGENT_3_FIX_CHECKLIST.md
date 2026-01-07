# Agent 3 Bug Fix Checklist

**Date**: 2026-01-06
**Purpose**: Quick reference for bug fixes

---

## 🔴 CRITICAL - Must Fix Before Agent 4

### 1. Fix `kv_cache_and_scratch_tests.rs` Compilation (9 errors)

- [ ] **Line 7**: Remove duplicate import
  ```rust
  // DELETE THIS LINE:
  use rocmforge::backend::ScratchBufferManager;
  ```

- [ ] **Lines 39-82**: Fix borrow checker issues (5 occurrences)
  ```rust
  // OPTION A: Restructure test to use buffers sequentially
  let attention_scores = scratch.attention_scores();
  // Use attention_scores, then drop
  let softmax_temp = scratch.softmax_temp();
  // Use softmax_temp, then drop
  let mlp_intermediate = scratch.mlp_intermediate();
  // etc.

  // OPTION B: Modify ScratchBufferManager API to return immutable views
  // (Requires changing backend/scratch.rs implementation)
  ```

- [ ] **Line 226**: Fix head_dim type
  ```rust
  // CHANGE FROM:
  head_dim: Some(128),
  // TO:
  head_dim: 128,
  ```

- [ ] **Line 223**: Add missing ModelConfig fields
  ```rust
  let model_config = ModelConfig {
      // ... existing fields ...
      rms_norm_eps: 1e-5,           // ADD THIS
      use_rotary_embeddings: true,   // ADD THIS
  };
  ```

- [ ] **Line 231**: Fix model_type
  ```rust
  // CHANGE FROM:
  model_type: "llama".to_string(),
  // TO:
  model_type: ModelType::Llama,
  ```

---

### 2. Fix `test_direct_cpu.rs` Compilation (3 errors)

- [ ] **Unresolved imports**: Fix module paths
  ```rust
  // CHANGE FROM:
  use super::super::rocmforge::attention::*;
  use super::super::super::rocmforge::attention::cpu::*;

  // TO:
  use rocmforge::attention::*;
  use rocmforge::attention::cpu::*;
  ```

---

## 🟠 HIGH - Should Fix Soon

### 3. Fix Multi-Query Attention (2 tests)

- [ ] **File**: `src/attention/multi_query.rs`
- [ ] **Line**: 588
- [ ] **Error**: `ShapeMismatch("Query tensor size 16 doesn't match expected 32")`
- [ ] **Fix**: Investigate query tensor shape calculation logic
  ```rust
  // Check: Why is query tensor 16 instead of 32?
  // Possible causes:
  // 1. Incorrect head_dim calculation
  // 2. Wrong batch size in reshape
  // 3. Transpose logic error
  ```

---

### 4. Fix KV Cache Tests (3 tests)

- [ ] **Test**: `test_sequence_removal`
- [ ] **Test**: `test_sequence_retrieval`
- [ ] **Test**: `test_token_appending`
- [ ] **File**: Likely `src/kv_cache/kv_cache.rs` or `src/model/kv_cache.rs`
- [ ] **Action**: Run tests individually to see specific errors

---

### 5. Fix Ambiguous Glob Re-exports

- [ ] **File**: `src/loader/mod.rs`
- [ ] **Lines**: 8-9
  ```rust
  // CHANGE FROM:
  pub use gguf::*;        // Exports GgufLoader
  pub use gguf_loader::*; // Also exports GgufLoader

  // TO (explicit re-exports):
  pub use gguf::{
      GgufTensorType,
      GgufTensorDataType,
      // ... other gguf-specific types
  };
  pub use gguf_loader::GgufLoader;  // Explicit path
  ```

---

## 🟡 MEDIUM - Fix When Convenient

### 6. Run `cargo fix` for Warnings

```bash
# Auto-fix most warnings:
cargo fix --features rocm --all-targets

# Apply fixes:
cargo fix --features rocm --all-targets --allow-dirty
```

- [ ] Run `cargo fix`
- [ ] Review changes
- [ ] Commit fixes
- [ ] Verify warning count reduced significantly

---

### 7. Fix Engine/HTTP Tests (4 tests)

- [ ] **Test**: `test_process_single_request`
- [ ] **Test**: `test_generate_request`
- [ ] **Test**: `test_get_request_status`
- [ ] **Test**: `test_get_nonexistent_request_status`
- [ ] **Action**: Run individually, investigate failures

---

## 🟢 LOW - Polish

### 8. Fix Naming Convention Violations (4 warnings)

- [ ] **File**: `src/loader/gguf.rs`
  ```rust
  // CHANGE:
  MXFP6_E2M3 → Mxfp6E2m3
  MXFP6_E3M2 → Mxfp6E3m2
  struct f16(u8) → struct F16(u8)
  ```

- [ ] **File**: `src/hip_isolation_test.rs`
  ```rust
  // CHANGE:
  const hipSuccess → const HIP_SUCCESS
  ```

---

### 9. Document HIP Kernel Limits

- [ ] **File**: `src/attention/kernels.rs` or HIP backend docs
- [ ] **Add docs**:
  ```rust
  /// # Kernel Limits
  ///
  /// - Max `head_dim`: 128 (register tiling limit)
  /// - Max `seq_len`: 256 (shared memory limit)
  /// - Block size: 256 threads (8 waves of 32)
  ///
  /// Exceeding these limits will cause kernel launch failures.
  ```

- [ ] **Optionally**: Add compile-time asserts
  ```rust
  const _: () = assert!(MAX_HEAD_DIM <= 128, "head_dim exceeds limit");
  const _: () = assert!(MAX_SEQ_LEN <= 256, "seq_len exceeds limit");
  ```

---

## Verification Commands

### After fixes, run these commands:

```bash
# 1. Check compilation
cargo test --features rocm --no-run

# 2. Run library tests
cargo test --features rocm --lib

# 3. Run specific failing tests
cargo test --features rocm --lib test_multi_query_attention_basic
cargo test --features rocm --lib test_sequence_removal

# 4. Check warnings
cargo clippy --features rocm --all-targets

# 5. Verify MXFP tests still pass
cargo test --features rocm mxfp
```

---

## Success Criteria

- [ ] All test files compile (0 compilation errors)
- [ ] Library test pass rate ≥ 95%
- [ ] CRITICAL issues resolved (compilation, multi-query, KV cache)
- [ ] HIGH priority issues resolved (ambiguous exports)
- [ ] Warning count reduced to < 50

---

## Estimated Fix Times

| Priority | Task | Time |
|----------|------|------|
| 🔴 CRITICAL | Compilation errors | 1-2 hours |
| 🟠 HIGH | Multi-query attention | 2-4 hours |
| 🟠 HIGH | KV cache tests | 2-3 hours |
| 🟠 HIGH | Ambiguous exports | 30 min |
| 🟡 MEDIUM | `cargo fix` | 1-2 hours |
| 🟡 MEDIUM | Engine/HTTP tests | 2-3 hours |
| 🟢 LOW | Naming violations | 30 min |
| 🟢 LOW | Kernel docs | 1 hour |
| **TOTAL** | **All issues** | **10-16 hours** |

---

## Order of Fixes

1. **Compilation errors** (blocks all testing)
2. **Multi-query attention** (high visibility issue)
3. **KV cache tests** (core functionality)
4. **Ambiguous exports** (quick win, high impact)
5. **Cargo fix** (quick cleanup)
6. **Engine/HTTP tests** (polish)
7. **Naming & docs** (final polish)

---

**Ready for Agent 4 when**: CRITICAL + HIGH issues completed
