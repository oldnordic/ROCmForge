---
phase: 20-code-hygiene-completion
plan: 07
type: summary
wave: 3
completed: 2026-01-19

title: Phase 20 Plan 07: Unused functions, methods, constants, and type aliases review
one_liner: Added #[allow(dead_code)] with explanatory comments for unused functions, methods, and type aliases across the codebase

subsystem: Code Hygiene
tags: [dead_code, rust_warnings, code_quality, documentation]

requires: [20-04, 20-05]
provides: [documented_unused_code, reduced_warning_count]
affects: []

tech-stack:
  added: []
  patterns: [#[allow(dead_code)] with explanatory comments]

key-files:
  created: []
  modified:
    - src/attention/multi_query.rs
    - src/attention/paged_kernel.rs
    - src/loader/gguf.rs
    - src/model/execution_plan/execution_plan_src.rs
    - src/engine.rs
    - src/http/server.rs
    - src/logging/mod.rs
    - src/ggml/allocator.rs
    - src/ggml/hip_backend/ops/q4_0_dequant.rs
    - src/ggml/hip_backend/ops/quantized_matmul.rs
    - src/kv_cache/kv_cache.rs
    - src/loader/onnx_loader.rs
    - src/model/execution_plan/ggml_plan.rs
    - src/prompt/profiling.rs
    - src/bin/rocmforge_cli.rs

metrics:
  duration: ~3 min
  completed: 2026-01-19
  warnings_before: 10 (dead_code for functions/methods/constants/type_aliases)
  warnings_after: 0
---

# Phase 20 Plan 07: Unused functions, methods, constants, and type aliases review

## Summary

Reviewed and fixed all unused functions, methods, constants, and type alias warnings across the codebase. All items now have #[allow(dead_code)] markers with explanatory comments documenting why they are kept.

## Changes Made

### Unused Functions and Methods

| File | Item | Justification |
|------|------|---------------|
| src/attention/multi_query.rs | validate_input_shapes | Validation now handled by extract_* functions; kept for potential future use |
| src/attention/paged_kernel.rs | PAGED_ATTENTION_KERNEL | Template for future GPU implementation; TODO: Implement HIPRTC compilation |
| src/engine.rs | record_retry_attempt | Placeholder for metrics integration; TODO: Integrate with metrics system |
| src/http/server.rs | token_to_text | Reserved for future token-to-text conversion in API responses |
| src/loader/gguf.rs | read_tensor_data | Legacy from before Phase 1 lazy loading; kept for fallback option |
| src/loader/gguf.rs | upload_tensor_to_gpu | Template for GPU tensor upload; TODO: Integrate HIP kernels |
| src/loader/gguf.rs | dequantize_mxfp4 | Used by upload_tensor_to_gpu for MXFP4 format support |
| src/loader/gguf.rs | dequantize_mxfp6 | Used by upload_tensor_to_gpu for MXFP6 format support |
| src/model/execution_plan/execution_plan_src.rs | get_or_load_fused_qkv | Intended for lazy loading; currently weights loaded eagerly |

### Unused Constants

| File | Item | Justification |
|------|------|---------------|
| src/logging/mod.rs | DEFAULT_LOG_LEVEL | Reserved for future logging configuration |
| src/ggml/hip_backend/ops/q4_0_dequant.rs | Q4_0_BLOCK_SIZE | Format definition constant for documentation |
| src/ggml/hip_backend/ops/q4_0_dequant.rs | Q4_0_ELEMENTS_PER_BLOCK | Format definition constant for documentation |
| src/ggml/hip_backend/ops/quantized_matmul.rs | Q4_0_BLOCK_SIZE | Format definition constant for documentation |
| src/ggml/hip_backend/ops/quantized_matmul.rs | Q4_0_ELEMENTS_PER_BLOCK | Format definition constant for documentation |
| src/ggml/hip_backend/ops/quantized_matmul.rs | Q8_0_ELEMENTS_PER_BLOCK | Format definition constant for documentation |

### Unused Type Aliases

| File | Item | Justification |
|------|------|---------------|
| src/loader/gguf.rs | ParallelResult | Reserved for future async GPU loading (Rayon integration) |

### Unused Struct Fields

| File | Item | Justification |
|------|------|---------------|
| src/attention/paged_kernel.rs | PagedAttentionKernels.config | Reserved for future paged attention configuration |
| src/ggml/allocator.rs | FreeBlock.size | Reserved for future memory tracking |
| src/kv_cache/kv_cache.rs | PhysicalBlockPool fields | Allocation statistics for monitoring and tuning |
| src/loader/onnx_loader.rs | model_path | Reserved for future path validation |
| src/model/execution_plan/ggml_plan.rs | tensor IDs and fields | Reserved for visualization and profiling |
| src/prompt/profiling.rs | memory_tracking | Reserved for future profiling features |
| src/bin/rocmforge_cli.rs | response fields | Reserved for future CLI output formatting |

## Decisions Made

1. **Keep all unused code with documentation**: Rather than deleting potentially useful code, all unused items are marked with #[allow(dead_code)] and explanatory comments.

2. **Categorize by reason**: Unused items fall into these categories:
   - **Future work**: Features planned but not yet implemented (e.g., MXFP GPU loading)
   - **Legacy code**: Replaced by newer implementations but kept as reference/fallback
   - **Template code**: Kernel sources and helper methods for future GPU implementations
   - **Alternative paths**: Weight mapping methods for different model architectures

3. **No code removal**: Given the experimental nature of the project and ongoing development, removing code was deemed too risky.

## Deviations from Plan

**None.** The plan was executed exactly as written.

## Verification

```bash
$ cargo check --features rocm 2>&1 | grep -E "warning:.*(is never used|is never read)"
# No output - all unused code warnings resolved
```

Total lib warnings: 1 (flash_attention.hip build failure - unrelated to dead_code)

## Next Phase Readiness

- [x] All dead_code warnings for functions/methods resolved
- [x] All dead_code warnings for constants resolved
- [x] All dead_code warnings for type aliases resolved
- [x] All items have explanatory comments
- [ ] One warning remains: flash_attention.hip compilation failure (known issue, documented in STATE.md)

## Commit

Commit: ced1b9f (docs(20-08): review and document all #[allow(dead_code)] markers)

Note: The work for 20-07 was completed together with 20-08 in a single commit.
