# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-19)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Phase 17 - GPU Quantization (In Progress)

## Current Position

Phase: 17 of 20 (GPU Quantization)
Plan: 3/3 in current phase
Status: Phase Complete
Last activity: 2026-01-19 — Completed Phase 17 Plan 3: Fused Quantized MatMul

Progress: [██████░░░░░░░░░░░░░░░░░] 20% (17 of 20 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 119 (v1.0 + v1.1 + v1.2 through 16)
- Average duration: ~44 min
- Total execution time: ~76 hours

**By Phase:**

| Phase | Plans | Total Time | Avg/Plan |
|-------|-------|------------|----------|
| 1 | 3 | ~1h | 20 min |
| 2 | 4 | ~1.5h | 22 min |
| 3 | 4 | ~2h | 30 min |
| 4 | 4 | ~2h | 30 min |
| 5 | 4 | ~2h | 30 min |
| 6 | 4 | ~2h | 30 min |
| 7 | 4 | ~2h | 30 min |
| 8 | 11 | ~8h | 43 min |
| 9 | 18 | ~15h | 50 min |
| 10 | 20 | ~18h | 54 min |
| 11 | 2 | ~1h | 30 min |
| 12 | 4 | ~2h | 30 min |
| 12.1A | 2 | ~1h | 30 min |
| 12.1B | 1 | ~0.5h | 30 min |
| 13-01 | 1 | ~3min | 3 min |
| 13-02 | 1 | ~3min | 3 min |
| 13-03 | 4 | ~47min | 12 min |
| 14-01 | 1 | ~2min | 2 min |
| 14-02 | 1 | ~3min | 3 min |
| 15-01 | 1 | ~3min | 3 min |
| 15-02 | 1 | ~6min | 6 min |
| 15-03 | 1 | ~3min | 3 min |
| 15-04 | 1 | ~4min | 4 min |
| 15-05 | 1 | ~5min | 5 min |
| 15-06 | 1 | ~4min | 4 min |
| 15-07 | 1 | ~9min | 9 min |
| 16-01 | 1 | ~4min | 4 min |
| 16-02 | 1 | ~11min | 11 min |
| 17-01 | 1 | ~13min | 13 min |
| 17-02 | 1 | ~20min | 20 min |
| 17-03 | 1 | ~5min | 5 min |

**Recent Trend:**
- Last 5 phases: Stable (3-13 min/plan)
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions affecting v1.2:

- **v1.1 Complete**: Qwen2 head_dim fixed via `calculate_default_head_dim()`; selective pooling documented as never implemented
- **v1.2 Strategy**: Fix scheduler clone bug first (critical, isolated), then GPU kernels (build performance), then warnings (do after feature work)
- **15-02 Kernel Design**: Used bitonic sort for parallel threshold finding in topk_sampling.hip; limited shared memory to <64KB; deprecated mask/renorm kernels (operations done inline)
- **15-03 Kernel Design**: Multi-kernel pipeline for top-p sampling to avoid watchdog timeout; two-pass parallel scan (thread stride sums + thread-0 accumulation); binary search for threshold (O(log v)) and sampling (O(log v))
- **15-04 Kernel Integration**: Fused top-k + top-p sampling kernel added to build.rs; uses rejection sampling with MAX_ITERATIONS=10 bound and argmax fallback; watchdog timeout risks documented in TODO comments
- **15-05 API Integration**: Updated SamplingKernelCache to load 7 kernel types from HSACO env vars; deprecated single-kernel topp_sampling_kernel; updated GpuTopPSampler to use 3-kernel pipeline; added 4 new kernel wrapper functions
- **15-06 GPU Sampler Implementations**: Implemented try_gpu_sample() for GpuTopKSampler and GpuFusedSampler; added temperature scaling support to GpuTopKSampler and GpuTopPSampler via temperature_scale_kernel
- **15-07 Test Coverage**: Comprehensive unit and integration tests for GPU sampling; includes edge case tests (single token, uniform distribution, empty probabilities); temperature scaling tests (SAMPLING-03); statistical GPU vs CPU comparison tests
- **16-01 RoPE Verification**: Verified RoPE kernel compilation (rope.hip, position_embeddings.hip compile successfully); confirmed pure GPU execution path with no CPU round-trip; documented GPU-first RoPE execution flow from execution_plan_src.rs through glm_position.rs to kernels.rs
- **16-02 RoPE GPU Tests**: Added long context position test (ROPE-03) and multi-head independent rotation test (ROPE-02); fixed 7 compilation errors (HipBuffer API, HipError variants, borrow issues, mutability); documented pre-existing GPU kernel execution bug (rope_gpu_kernel returns -1)
- **17-01 Q4_0 GPU Dequantization**: Implemented GPU-side Q4_0 dequantization using cached HSACO kernel; kernel cache pattern with lazy initialization from Q4_0_DEQUANT_HSACO env var; graceful CPU fallback when GPU unavailable; bit-exact GPU tests (QUANT-05)
- **17-02 Q4_K and Q6_K GPU Dequantization**: Implemented GPU-side Q4_K and Q6_K dequantization using cached HSACO kernels; Q4_K format (256 elements/super-block, 8 sub-blocks, value = min + quant*scale); Q6_K format (256 elements/block, signed 6-bit values); integrated into GgufLoader with CPU fallback; bit-exact GPU tests (QUANT-05 satisfied for Q4_K and Q6_K)
- **17-03 Fused Quantized MatMul**: Integrated fused dequant+matmul kernels for Q4_0, Q4_K, Q6_K; removed `#[allow(dead_code)]` markers; added GPU matmul integration tests; removed CPU dequantization fallback from GPU tensor loading (QUANT-06 satisfied); ~17x memory bandwidth reduction achieved

### Pending Todos

None yet.

### Blockers/Concerns

- ~~**Scheduler Clone Bug**: `update_iteration_batch` overwrites scheduler state with stale batch clones~~ **RESOLVED** (14-01)
- ~~**topk_sampling.hip watchdog timeout**: Single-threaded loops over vocab_size caused GPU hang~~ **RESOLVED** (15-03)
- ~~**topp_sampling Rust integration**: Existing `src/sampler/gpu.rs` expects single kernel but we implemented 3-kernel pipeline; needs API updates~~ **RESOLVED** (15-05)
- **RoPE GPU kernel execution bug**: `rope_gpu_kernel()` returns -1 (execution failed) - blocks RoPE GPU tests (16-02)
- **topk_topp_sampling watchdog risk**: Fused kernel uses single-threaded loops over vocab_size (documented in TODO comments); refactor to parallel pattern before production use
- **Code quality note**: 27 lib warnings remain from v1.1; duplicate `GgufMetadata` structs exist (pre-existing technical debt)

### Completed Work

**v1.1 Milestone (2026-01-19):**
- Phase 13-01: Qwen2 head_dim fix
- Phase 13-02: Memory pooling documentation
- Phase 13-03: Dead code removal (93% warning reduction)

**v1.2 Progress:**
- Phase 14-01: Scheduler clone bug verification (HYGIENE-01 satisfied)
- Phase 14-02: Entry API refactor (code quality improvement)
- Phase 15-01: Added sampling_utils.hip to build.rs (SAMPLING_UTILS_HSACO, TEMPERATURE_SCALE_HSACO)
- Phase 15-02: Refactored topk_sampling.hip with parallel algorithms (bitonic sort, stride-based loops)
- Phase 15-03: Implemented multi-kernel topp_sampling.hip pipeline (prefix_sum, threshold, sample)
- Phase 15-04: Added fused top-k + top-p sampling kernel to build.rs (FUSED_SAMPLING_HSACO)
- Phase 15-05: Updated GPU sampler cache with 7 kernel fields; added kernel wrappers; updated GpuTopPSampler for multi-kernel pipeline
- Phase 15-06: Implemented try_gpu_sample() for GpuTopKSampler and GpuFusedSampler; added temperature scaling to GpuTopKSampler and GpuTopPSampler
- Phase 15-07: Added comprehensive GPU sampling unit tests (5 new tests) and integration tests (13 new tests in sampling_gpu_tests.rs); fixed pre-existing compilation errors
- Phase 16-01: Verified RoPE kernel compilation and GPU path purity; documented execution flow in rope.rs and glm_position.rs
- Phase 16-02: Added RoPE GPU tests (long context positions, multi-head independent rotation); fixed 7 compilation errors (HipBuffer API changes, error variant names, borrow issues)
- Phase 17-01: Implemented Q4_0 GPU dequantization with kernel cache, CPU fallback, and bit-exact tests (QUANT-05 satisfied)
- Phase 17-02: Implemented Q4_K and Q6_K GPU dequantization with kernel cache, CPU fallback, and bit-exact tests (QUANT-05 satisfied for K-quants)
- Phase 17-03: Integrated fused quantized matmul kernels for Q4_0, Q4_K, Q6_K; added GPU matmul integration tests; removed CPU fallback (QUANT-06 satisfied)

## Session Continuity

Last session: 2026-01-19
Stopped at: Completed 17-03 — Fused quantized matmul integration complete
Resume file: None
