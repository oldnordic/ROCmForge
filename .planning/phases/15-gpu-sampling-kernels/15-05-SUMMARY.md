---
phase: 15-gpu-sampling-kernels
plan: 05
subsystem: gpu
tags: [hip, rocm, sampling, kernel-cache, api-integration, multi-kernel]

# Dependency graph
requires:
  - phase: 15-01
    provides: sampling_utils.hip kernel compilation setup
  - phase: 15-02
    provides: topk_sampling.hip kernel implementation
  - phase: 15-03
    provides: topp_sampling.hip multi-kernel pipeline
  - phase: 15-04
    provides: fused top-k + top-p sampling kernel

provides:
  - SamplingKernelCache: Updated with 7 kernel fields
  - Kernel launch wrappers: temperature_scale, topp_prefix_sum, topp_threshold, topp_sample
  - GpuTopPSampler: Updated to use 3-kernel pipeline

affects: [sampling-integration, gpu-testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Multi-kernel pipeline integration in Rust
    - Kernel cache with 7 separate kernel types
    - Deprecation pattern for API changes

key-files:
  created: []
  modified:
    - src/sampler/gpu.rs

key-decisions:
  - "Deprecate single-kernel topp_sampling_kernel in favor of 3-kernel pipeline"
  - "Update SamplingKernelCache to load all 7 kernel types from build.rs env vars"
  - "Preserve CPU fallback when kernels are not available"

patterns-established:
  - Kernel wrapper functions follow existing pattern
  - Cache initialization loads from env vars with fallback paths
  - Multi-kernel launch in try_gpu_sample with intermediate GPU buffers

# Metrics
duration: 290 seconds
completed: 2026-01-19
---

# Phase 15: Plan 05 Summary

**Updated GPU sampler cache to load compiled HSACO files from build.rs and use multi-kernel pipeline for top-p sampling**

## Performance

- **Duration:** 290 seconds (4 min 50 sec)
- **Started:** 2026-01-19T15:22:52Z
- **Completed:** 2026-01-19T15:27:42Z
- **Tasks:** 2 (merged into single commit)
- **Files modified:** 1

## Accomplishments

- Updated SamplingKernelCache struct from 5 to 7 kernel fields
- Changed env var names from old (SOFTMAX_HSACO, etc.) to new (SAMPLING_UTILS_HSACO, etc.)
- Added 4 new kernel launch wrapper functions for multi-kernel pipeline
- Deprecated old topp_sampling_kernel() with #[deprecated] attribute
- Updated GpuTopPSampler::try_gpu_sample() to use 3-kernel pipeline
- Preserved CPU fallback behavior when kernels not loaded

## Task Commits

1. **Task 1-2: Update kernel cache and add wrappers** - `623af7a` (feat)

**Plan metadata:** (to be added after summary creation)

## Files Created/Modified

- `src/sampler/gpu.rs` - Major updates:
  - SamplingKernelCache struct: 7 kernel fields (softmax, temperature_scale, topk, topp_prefix_sum, topp_threshold, topp_sample, fused)
  - get_or_init_sampling_cache(): Loads from 7 HSACO env vars (SAMPLING_UTILS_HSACO, TEMPERATURE_SCALE_HSACO, TOPK_SAMPLING_HSACO, TOPP_PREFIX_SUM_HSACO, TOPP_THRESHOLD_HSACO, TOPP_SAMPLE_HSACO, FUSED_SAMPLING_HSACO)
  - New wrappers: temperature_scale_kernel(), topp_prefix_sum_kernel(), topp_threshold_kernel(), topp_sample_kernel()
  - Deprecated: topp_sampling_kernel() now returns error with deprecation message
  - GpuTopPSampler::try_gpu_sample(): Uses 3-kernel pipeline with prefix_sum, threshold, sample kernels

## Kernel Field Mapping

| Old Field | New Field(s) | Env Var | Kernel Name |
|-----------|--------------|---------|-------------|
| softmax_kernel | softmax_kernel | SAMPLING_UTILS_HSACO | softmax_kernel |
| (none) | temperature_scale_kernel | TEMPERATURE_SCALE_HSACO | temperature_scale_kernel |
| topk_kernel | topk_kernel | TOPK_SAMPLING_HSACO | topk_sampling_kernel |
| prefix_sum_kernel | topp_prefix_sum_kernel | TOPP_PREFIX_SUM_HSACO | topp_prefix_sum_kernel |
| (none) | topp_threshold_kernel | TOPP_THRESHOLD_HSACO | topp_threshold_kernel |
| (none) | topp_sample_kernel | TOPP_SAMPLE_HSACO | topp_sample_kernel |
| topp_kernel | (deprecated) | (old TOPP_HSACO) | (removed) |
| fused_kernel | fused_kernel | FUSED_SAMPLING_HSACO | topk_topp_sampling_kernel |

## Decisions Made

1. **Multi-kernel pipeline integration:** Updated GpuTopPSampler to use 3 separate kernel launches instead of single deprecated topp_sampling_kernel. This required allocating intermediate GPU buffers (prefix_sum, threshold) and launching kernels in sequence.

2. **API deprecation strategy:** Old topp_sampling_kernel() function marked as #[deprecated] but kept for compatibility. Returns error directing users to new 3-kernel pipeline.

3. **Type consistency:** Used i32 for threshold indices and output tokens to match HIP kernel signatures (int vs uint32_t).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed buffer type mismatch for output tokens**
- **Found during:** Implementation of try_gpu_sample
- **Issue:** HIP kernels return i32 but old code expected u32
- **Fix:** Added conversion from i32 to u32 after copying results from GPU
- **Files modified:** src/sampler/gpu.rs
- **Verification:** Type conversion added at line 892
- **Committed in:** 623af7a (task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor type fix required for kernel interface compatibility.

## Issues Encountered

- **Pre-existing compilation errors:** The codebase has existing compilation errors in other files (src/ops/attention_gpu.rs, src/model/simple_transformer.rs, etc.) that are unrelated to this plan's changes. The src/sampler/gpu.rs file itself has no compilation errors.

## Verification Results

```bash
# Verify all 7 env vars are loaded
$ grep "std::env::var.*_HSACO" src/sampler/gpu.rs
std::env::var("SAMPLING_UTILS_HSACO")
std::env::var("TEMPERATURE_SCALE_HSACO")
std::env::var("TOPK_SAMPLING_HSACO")
std::env::var("TOPP_PREFIX_SUM_HSACO")
std::env::var("TOPP_THRESHOLD_HSACO")
std::env::var("TOPP_SAMPLE_HSACO")
std::env::var("FUSED_SAMPLING_HSACO")

# Verify all 7 kernel fields in cache struct
$ grep -A 30 "struct SamplingKernelCache" src/sampler/gpu.rs | grep "_kernel:"
    softmax_kernel: Option<HipKernel>,
    temperature_scale_kernel: Option<HipKernel>,
    topk_kernel: Option<HipKernel>,
    topp_prefix_sum_kernel: Option<HipKernel>,
    topp_threshold_kernel: Option<HipKernel>,
    topp_sample_kernel: Option<HipKernel>,
    fused_kernel: Option<HipKernel>,

# Verify 4 new wrapper functions exist
$ grep "^pub unsafe fn.*_kernel" src/sampler/gpu.rs
pub unsafe fn temperature_scale_kernel(
pub unsafe fn topp_prefix_sum_kernel(
pub unsafe fn topp_threshold_kernel(
pub unsafe fn topp_sample_kernel(
pub unsafe fn topp_sampling_kernel(    // deprecated
pub unsafe fn topk_sampling_kernel(
pub unsafe fn fused_sampling_kernel(

# Verify no sampler/gpu.rs compilation errors (other files have pre-existing issues)
$ cargo check --features rocm 2>&1 | grep "src/sampler/gpu.rs"
# (no output - no errors specific to this file)
```

## Next Phase Readiness

- **GPU sampler cache:** Ready to load all 7 compiled kernels
- **Top-p sampler:** Updated to use 3-kernel pipeline
- **Remaining work:** GPU kernels need to be compiled (requires hipcc/ROCm installation)
- **Testing:** Integration tests needed once HSACO files are available
- **Known limitation:** GpuTopKSampler and GpuFusedSampler still use CPU fallback (GPU kernels exist but not yet wired up in try_gpu_sample)

---
*Phase: 15-gpu-sampling-kernels*
*Completed: 2026-01-19*
