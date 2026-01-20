# Phase 24 Plan 02: Migrate Quantization Kernels Summary

**Phase:** 24 - Kernel-Centric Restructure
**Plan:** 02 of 4
**Status:** COMPLETE
**Completed:** 2026-01-20

## One-Liner

Unified CPU and GPU quantization dequantization kernels (Q4_0, Q4_K, Q6_K, Q8_0) into `src/kernels/quant/` with fallback functions and backward-compatible re-exports.

## Summary

Migrated quantization kernels from their original locations (`loader/dequant.rs` for CPU, `ggml/hip_backend/ops/` for GPU) into a unified `src/kernels/quant/` directory structure. Each format now has a single file containing both CPU and GPU implementations with automatic CPU fallback when GPU kernels are unavailable.

### Deliverables

| File | LOC | Description |
|------|-----|-------------|
| `src/kernels/quant/q4_0.rs` | 446 | Q4_0 dequantization (CPU + GPU + fallback) |
| `src/kernels/quant/q4_k.rs` | 372 | Q4_K dequantization (CPU + GPU + fallback) |
| `src/kernels/quant/q6_k.rs` | 403 | Q6_K dequantization (CPU + GPU + fallback) |
| `src/kernels/quant/q8_0.rs` | 174 | Q8_0 dequantization (CPU only, Rayon-parallelized) |
| `src/kernels/quant/mod.rs` | 62 | Re-exports with cfg(feature="rocm") gating |
| `src/loader/dequant.rs` | Modified | Re-exports from kernels with deprecation notices |
| `src/ggml/hip_backend/ops/mod.rs` | Modified | Re-exports from kernels for backward compatibility |
| `src/kernels/matmul/mod.rs` | Modified | Made `quantized` module public |

### Acceptance Criteria

- [x] All quantization kernels in `src/kernels/quant/`
- [x] Each file under 600 LOC (max: 446)
- [x] CPU and GPU implementations in same file
- [x] Fallback functions provided (try GPU, fall back to CPU)
- [x] Original import paths still work (re-exports)
- [x] All tests pass (46/46 quantization tests)

## Technical Details

### File Structure

```
src/kernels/quant/
├── mod.rs          # Module declarations and re-exports
├── common.rs       # Shared constants and helper functions (from 24-01)
├── q4_0.rs         # Q4_0 dequantization (CPU + GPU + fallback)
├── q4_k.rs         # Q4_K dequantization (CPU + GPU + fallback)
├── q6_k.rs         # Q6_K dequantization (CPU + GPU + fallback)
└── q8_0.rs         # Q8_0 dequantization (CPU only)
```

### API Design

Each quantization format provides:
1. **CPU function**: `dequantize_{format}_cpu(data: &[u8], n: usize) -> Vec<f32>`
2. **GPU function**: `dequantize_{format}_gpu_kernel(...)` (cfg feature="rocm")
3. **Fallback function**: `dequantize_{format}_with_fallback(...)` (cfg feature="rocm")

### Backward Compatibility

- `src/loader/dequant.rs` re-exports with `#[deprecated]` attribute
- `src/ggml/hip_backend/ops/mod.rs` re-exports from kernels
- All existing import paths continue to work

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Module visibility issue**

- **Found during:** Task 8 - Running tests
- **Issue:** `crate::kernels::matmul::quantized` module was private, causing compilation errors in code that referenced it directly
- **Fix:** Changed `mod quantized;` to `pub mod quantized;` in `src/kernels/matmul/mod.rs`
- **Files modified:** `src/kernels/matmul/mod.rs`
- **Commit:** 9669ede

**2. [Rule 3 - Blocking] Missing cfg(feature="rocm") gating**

- **Found during:** Task 4 - Creating q4_0.rs
- **Issue:** `dequantize_q4_0_cpu_upload` used `HipBackend` and `HipBuffer` without proper feature gating
- **Fix:** Added `#[cfg(feature = "rocm")]` to `dequantize_q4_0_cpu_upload` and imported `HipBuffer` with cfg gate
- **Files modified:** `src/kernels/quant/q4_0.rs`
- **Commit:** 9669ede

### None from original plan requirements

All other requirements executed as specified.

## Test Results

```
cargo test dequant
running 42 tests
test result: ok. 42 passed; 0 failed; 0 ignored

cargo test quant
running 46 tests
test result: ok. 46 passed; 0 failed; 0 ignored
```

### Note on Pre-existing Test Failure

One test failed in the full test suite: `kernels::attention::mask::tests::test_apply_mask_batched`
This is unrelated to quantization kernels and was a pre-existing issue in the attention mask module.

## Metrics

| Metric | Value |
|--------|-------|
| **Duration** | 7 minutes |
| **Files Created** | 4 kernel files |
| **Files Modified** | 4 |
| **Lines Added** | ~1,400 |
| **Lines Removed** | ~1,540 |
| **Net Change** | -140 lines |
| **Tests Passing** | 46/46 quantization tests |
| **Commit** | 9669ede |

## Next Phase Readiness

- [x] All quantization kernels unified
- [x] Backward compatibility maintained
- [x] Tests passing
- [ ] Ready for 24-03 (Migrate MatMul Kernels)

## Dependencies

- **Requires:** 24-01 (Kernel directory structure created)
- **Provides:** Unified quantization kernel interface for 24-03
