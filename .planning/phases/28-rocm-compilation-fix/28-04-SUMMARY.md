# Phase 28 Plan 04: Remove cfg gates from ops/attention

**Status:** COMPLETE
**Duration:** 5 minutes 33 seconds (333s)
**Completed:** 2026-01-20

---

## Summary

Removed all `#[cfg(feature = "rocm")]` conditional compilation gates from 5 files in the `src/ops/attention/` directory. GPU attention operations are now always compiled unconditionally, treating them as core functionality rather than optional GPU-specific code.

---

## Changes by File

| File | cfg gates removed | Key changes |
|------|-------------------|-------------|
| `src/ops/attention_gpu.rs` | 19 | Removed gates from imports, struct fields, methods, kernel constants, hiprtc module, test include |
| `src/ops/attention/mod.rs` | 1 | Removed gate from `pub mod hiprtc;` |
| `src/ops/attention/kernels.rs` | 18 | Removed gates from imports, `CompiledKernel`, struct fields, methods, kernel constants |
| `src/ops/attention/softmax.rs` | 17 | Removed gates from imports, struct fields, methods, test include |
| `src/ops/causal_mask_tests.rs` | 8 | Removed gates from all test functions |

**Total:** 63 cfg gates removed

---

## Technical Details

### Pattern Applied

For each file, removed these patterns:
- `#[cfg(feature = "rocm")]` before imports
- `#[cfg(feature = "rocm")]` before struct field declarations
- `#[cfg(feature = "rocm")]` before function/method definitions
- `#[cfg(feature = "rocm")]` before const kernel definitions
- `#[cfg(feature = "rocm")]` before test module declarations

### Imports Changed

These imports are now unconditional:
- `use crate::backend::{HipKernel, HipModule};`
- `use once_cell::sync::OnceCell;`
- `use super::hiprtc;`

### Struct Fields Changed

- `HipAttentionKernels.attention_softmax_kernel: OnceCell<CompiledKernel>`
- `HipAttentionKernels.causal_mask_kernel: OnceCell<CompiledKernel>`
- `CausalMaskOp.causal_mask_kernel: OnceCell<CompiledKernel>`
- `AttentionSoftmax.attention_softmax_kernel: OnceCell<CompiledKernel>`

### Methods Changed

Made unconditional (all were in `impl` blocks):
- `compile_attention_softmax_kernel()`
- `get_attention_softmax_kernel()`
- `compile_causal_mask_kernel()`
- `get_causal_mask_kernel()`
- `apply_causal_mask_gpu()`
- `compute_softmax_gpu()`

### Constants Changed

These kernel constants are now unconditionally compiled:
- `ATTENTION_SOFTMAX_KERNEL` (in both kernels.rs and attention_gpu.rs)
- `CAUSAL_MASK_KERNEL` (in both kernels.rs and attention_gpu.rs)

### hiprtc Module

The `mod hiprtc` containing HIP runtime compilation utilities is now unconditionally available in:
- `src/ops/attention_gpu.rs`
- `src/ops/attention/kernels.rs`

### Test Changes

- `src/ops/causal_mask_tests.rs`: All `#[cfg(feature = "rocm")]` gates removed from test functions
- `src/ops/attention/softmax.rs`: Test include changed from `#[cfg(feature = "rocm")]` to just `#[cfg(test)]`

---

## Verification

```bash
# Count cfg gates in each target file - all should be 0
grep -c "cfg.*feature.*rocm" src/ops/attention_gpu.rs           # 0
grep -c "cfg.*feature.*rocm" src/ops/attention/mod.rs           # 0
grep -c "cfg.*feature.*rocm" src/ops/attention/kernels.rs       # 0
grep -c "cfg.*feature.*rocm" src/ops/attention/softmax.rs       # 0
grep -c "cfg.*feature.*rocm" src/ops/causal_mask_tests.rs       # 0
```

---

## Deviations from Plan

None. The plan was executed exactly as specified:
- All 5 files had their cfg gates removed
- No test logic was modified (only cfg attributes removed)
- CPU fallback code paths remain intact

---

## Commits

- `aa16385`: feat(28-04): remove cfg gates from ops/attention GPU code

---

## Next Steps

Phase 28 has additional waves planned for removing cfg gates from other directories:
- Wave 4: src/attention/ files
- Wave 5: src/kernels/ files
- Wave 6: src/mlp/ files
- Wave 7: src/ggml/hip_backend/ files

The pattern established here can be applied to those subsequent waves.
