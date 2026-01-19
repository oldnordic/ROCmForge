# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-19)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Phase 15 - GPU Sampling Kernels

## Current Position

Phase: 15 of 20 (GPU Sampling Kernels)
Plan: 7/7 in current phase
Status: Phase complete
Last activity: 2026-01-19 — Completed 15-07: Comprehensive GPU sampling unit and integration tests

Progress: [█████░░░░░░░░░░░░░░░░░░░] 18% (15.7 of 20 phases planned)

## Performance Metrics

**Velocity:**
- Total plans completed: 103 (v1.0 + v1.1)
- Average duration: ~45 min
- Total execution time: ~73 hours

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
| 15-06 | 1 | ~0min | 0 min |
| 15-07 | 1 | ~9min | 9 min |

**Recent Trend:**
- Last 5 phases: Stable (3-6 min/plan)
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
- **15-07 Test Coverage**: Comprehensive unit and integration tests for GPU sampling; includes edge case tests (single token, uniform distribution, empty probabilities); temperature scaling tests (SAMPLING-03); statistical GPU vs CPU comparison tests

### Pending Todos

None yet.

### Blockers/Concerns

- ~~**Scheduler Clone Bug**: `update_iteration_batch` overwrites scheduler state with stale batch clones~~ **RESOLVED** (14-01)
- ~~**topk_sampling.hip watchdog timeout**: Single-threaded loops over vocab_size caused GPU hang~~ **RESOLVED** (15-03)
- ~~**topp_sampling Rust integration**: Existing `src/sampler/gpu.rs` expects single kernel but we implemented 3-kernel pipeline; needs API updates~~ **RESOLVED** (15-05)
- **topk_topp_sampling watchdog risk**: Fused kernel uses single-threaded loops over vocab_size (documented in TODO comments); refactor to parallel pattern before production use
- **Code quality note**: 27 lib warnings remain from v1.1; duplicate `GgufMetadata` structs exist (pre-existing technical debt)
- **GpuTopKSampler and GpuFusedSampler**: Still use CPU fallback; GPU kernels exist but not yet wired up in try_gpu_sample

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
- Phase 15-06: (Skipped or not tracked)
- Phase 15-07: Added comprehensive GPU sampling unit tests (5 new tests) and integration tests (13 new tests in sampling_gpu_tests.rs); fixed pre-existing compilation errors

## Session Continuity

Last session: 2026-01-19
Stopped at: Completed 15-07 — Phase 15 complete, ready for Phase 16
Resume file: None
