# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Phase 26 - Compiler Warning Cleanup

## Current Position

Phase: 26 - Warning Cleanup (Plan 3 of N - In Progress)
Status: PLAN 26-02 COMPLETE - All to_host_vec() deprecation warnings eliminated
Last activity: Completed 26-02 to_host_vec() migration at 2026-01-20T18:40:33Z

Note: 26-01 (loader dequant migration) also complete at 18:44:00Z

Progress: [████████████████████░] 98% (Phase 22 COMPLETE, Phase 23 COMPLETE, Phase 24 COMPLETE, Phase 25 COMPLETE, Phase 26 in progress)

## Milestone v1.3 Summary

**Phases:** 19-21 (17 plans, 5/6 in Phase 21)
**Delivered:**
- Pure HIP-native quantized kernels (all CUDA intrinsics eliminated)
- Zero compiler warnings baseline
- Test health validation with graceful skip pattern
- CI/CD compatibility without GPU kernel compilation

**Deferred:**
- Performance validation (21-06) - skipped to avoid desktop GPU stress testing

## Performance Metrics

**Velocity:**
- Total plans completed: 158 (v1.0 + v1.1 + v1.2 + v1.3 + v1.4 Phase 22, Phase 23 COMPLETE, Phase 24 COMPLETE, Phase 25 gap closure 9/10)
- Plans remaining: 6 (1 gap closure + 5 future)
- Average duration: ~44 min
- Total execution time: ~114 hours

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions:

- **Operation Module Pattern for Cache Decomposition**: Use functional operation modules taking explicit RwLock references instead of trait extensions. Enables testability and clear lock management. Applied to kv_cache in 25-11 with cache_ops, sequence_ops, and block_ops modules (25-11)
- **Submodule Pattern for Private Field Access**: When sibling modules need access to private fields, reorganize as parent-child submodules (e.g., kernels_cache/ containing kernels_basic.rs and kernels_flash.rs) to preserve encapsulation (25-07)
- **Memory Arena for GPU Weights**: Use ModelWeightArena with single HipBuffer backing store to prevent RDNA3 GPU hangs from multiple small allocations. Best-fit allocation with 256-byte alignment (22-01)
- **HTTP Context Endpoint Deferred**: HNSW index in sqlitegraph is not Send+Sync due to internal `Rc<RefCell<>>`, making it incompatible with Axum's async State extractor. CLI commands provide full context management functionality as alternative (12.1B-01)
- **Performance Validation Deferred**: GPU benchmarking (21-06) skipped by user to avoid desktop GPU stress testing

Historical decisions affecting v1.3:

- **CUDA Intrinsics Elimination**: ROCmForge does NOT port CUDA code. All kernels must be rewritten from mathematical first principles for AMD HIP (ANTI_CUDA_PORTING_RATIONALE.md)
- **WARP_SIZE Correction**: Changed from 32 to 64 for RDNA3 wavefront alignment (19-02)
- **Zero Warnings Baseline**: Achieved through systematic elimination of all compiler warning categories (20-08)
- **Graceful Skip Pattern**: GPU tests without HSACO kernels skip gracefully with helpful messages, enabling CI/CD without GPU kernel compilation (21-05)

**Phase 12.1B (2026-01-20):**
- Phase 12.1B-01: SQLiteGraph context integration with CLI commands (HTTP endpoint deferred due to HNSW Send+Sync limitation)

### Pending Todos

- **Context HTTP API integration**: Requires upstream sqlitegraph changes to make HNSW Send+Sync, or alternative thread-safe vector index

### Blockers/Concerns

- **HNSW Send+Sync**: Context HTTP endpoint blocked by sqlitegraph's HNSW using `Rc<RefCell<>>` internally. CLI commands work as alternative.
- **Performance validation**: Deferred to future phase (requires GPU stress testing infrastructure)
- **FlashAttention generic kernel**: CUDA intrinsic `__shfl_down_f32` compilation issue (documented in blockers)
- **Quantized matmul tile size alignment**: TILE_SIZE_K/N=32 not wave64-aligned (deferred optimization)

### Completed Work

**v1.3 Milestone (2026-01-20):**
- Phase 19-01: Quantization format analysis (Q4_0, Q4_K, Q6_K bit-packing documented)
- Phase 19-02: HIP intrinsics replacement in Q4_0, Q4_K, Q6_K matmul kernels
- Phase 19-03: Fused RMSNorm CUDA intrinsics removal
- Phase 19-04: HIP kernel compilation and validation (gfx1100 HSACO files)
- Phase 20-01: Fix unreachable pattern and cfg warnings
- Phase 20-02: Fix type naming style warnings (Q4_K, Q6_K)
- Phase 20-03: Remove unused imports
- Phase 20-04: Replace deprecated methods (19 calls)
- Phase 20-05: Fix unused mut and unused assignment warnings
- Phase 20-06: Dead code categorization and backend field fixes
- Phase 20-07: Unused functions, methods, constants review
- Phase 20-08: Dead code marker review and zero warnings baseline
- Phase 21-01: cfg(feature) gates fix for GPU dequant exports
- Phase 21-02: KV cache capacity enforcement fix
- Phase 21-03: Decode step integration tests fix
- Phase 21-04: E2E test graceful skip verification
- Phase 21-05: Full test suite health validation

**Phase 21 COMPLETE** - All test health requirements satisfied; performance validation deferred

**Milestone v1.3 COMPLETE**

**v1.4 - Memory & Performance (2026-01-20):**
- Phase 22-01: ModelWeightArena structure with best-fit allocation and 256-byte alignment
- Phase 22-02: MemoryCalculator and check_memory_for_model() for pre-allocation memory verification
- Phase 22-03: Arena-based tensor loading with upload_to_buffer_offset() and from_arena_slice()
- Phase 22-04: Verified memory pool integration in load_to_gpu_async(), enhanced fragmentation logging
- Phase 22-05: Memory pool verification complete - 13/13 arena tests pass, 7/7 calculator tests pass, single hipMalloc confirmed

**Milestone v1.4 COMPLETE**

**Decision: Safety Margin on Calculated Need**
- Use 10% + 100MB minimum safety margin on CALCULATED memory need
- NOT a percentage of free memory (the flawed 70% approach could still crash desktop)
- Based on actual tensor requirements from GGUF model (22-02)

**Decision: Arena Slice Reference via HipBuffer::sub_buffer_view()**
- Use existing HipBuffer::sub_buffer_view() for arena slice references
- Arc-based ownership ensures arena outlives all DeviceTensors using it (22-03)
- Offset-based upload via AsyncLoader::upload_to_buffer_offset() (22-03)

**v1.5 - Code Quality & Cleanup (2026-01-20):**
- Phase 23-01: Remove duplicate MXFP code (E8M0, MxfpBlock) from gguf.rs - use mxfp.rs
- Phase 23-02: Remove unused quantization formats (Q4_1, Q5_0, Q5_1) from GgufTensorType
- Phase 23-03: Consolidate GgufMetadata - remove duplicate struct from gguf.rs
- Phase 23-04: Remove ParallelResult dead code and fix compilation issues
- Phase 23-05: Verification - all 598 tests passing, 1200+ lines removed

**Phase 23 COMPLETE** - 5 of 5 plans complete

**Decision: Remove Unused Quantization Formats**
- Q4_1, Q5_0, Q5_1 formats removed - no common GGUF models use these as of 2026-01-20
- from_u32() returns descriptive Err for unsupported types (3, 6, 7) instead of Ok
- Active formats: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, and IQ formats (23-02)

**Decision: Re-export GgufMetadata for Tests**
- Re-export GgufMetadata from gguf.rs to maintain test import compatibility
- Single source of truth remains in metadata.rs (23-05)

**Phase 23 Metrics:**
- Lines removed: ~1,200 (dead/duplicate code)
- Files modified: 15
- Tests passing: 598/598 (100%)

**v1.6 - Kernel-Centric Restructure (2026-01-20):**
- Phase 24-01: Created src/kernels/ directory structure with quant/, attention/, matmul/, element/ subdirectories
- Phase 24-02: Migrated quantization kernels (Q4_0, Q4_K, Q6_K, Q8_0) to kernels/quant/
- Phase 24-03: Migrated attention kernels (softmax, matmul, flash, mask, rope) to kernels/attention/ with CPU/GPU fallback
- Phase 24-04: Migrated matmul kernels (quantized Q4_0, Q4_K, Q6_K, Q8_0, FP16/FP32) to kernels/matmul/ with re-export compatibility
- Phase 24-05: Migrated element-wise kernels (rms_norm, swiglu, scale, add) to kernels/element/ with re-export compatibility
- Phase 24-06: Verification - all 27 kernel files under 1,000 LOC, 630/630 tests passing, no circular dependencies

**Phase 24 COMPLETE** - 6 of 6 plans complete

**Phase 24 Metrics:**
- Kernel files: 27 (4,068 total LOC)
- Largest file: 448 LOC (q4_0.rs dequant and matmul variants)
- Tests passing: 630/630 (up from 598 baseline)
- Circular dependencies: 0

**Decision: Keep Re-Export Chains During Migration**
- Maintain backward compatibility by re-exporting from legacy ggml::hip_backend::ops modules
- Allows gradual migration without breaking existing code
- Can remove legacy re-exports in future cleanup phase (24-07)

**v1.6 - Phase 25: Architectural Decomposition (2026-01-20):**
- Phase 25-01: Magellan code mapping - symbol clusters, dependencies (PLANNED)
- Phase 25-02: Responsibility analysis - domain concern grouping (PLANNED)
- Phase 25-03: Module boundary proposal - decomposition map (PLANNED)
- Phase 25-04: Refactor Wave 1 - Loader (gguf.rs → 8 modules) (COMPLETE)
- Phase 25-05: Refactor Wave 2 - Execution/Mid-tier (8 files → ~35 modules) (PARTIAL - 5/8 targets complete)
- Phase 25-06: Refactor Wave 3 - Backend/Core (backend.rs → 10 modules) (COMPLETE)
- Phase 25-07: QA + Verification - compilation fixes and verification (COMPLETE)
- Phase 25-08: Gap Closure - engine.rs decomposition (COMPLETE)
- Phase 25-09: Gap Closure - scheduler/scheduler.rs decomposition (PLANNED)
- Phase 25-10: Gap Closure - ops/attention_gpu.rs decomposition (PLANNED)
- Phase 25-11: Gap Closure - kv_cache/kv_cache.rs further decomposition (COMPLETE)
- Phase 25-12: Gap Closure - ggml/hip_backend/execution.rs further decomposition (COMPLETE)
- Phase 25-13: Gap Closure - http/server.rs decomposition (COMPLETE)
- Phase 25-14: Gap Closure - profiling/rocprof_integration.rs decomposition (COMPLETE)
- Phase 25-15: Gap Closure - profiling/baseline.rs decomposition (COMPLETE)
- Phase 25-16: Gap Closure - backend/cpu/simd_ops.rs decomposition (COMPLETE)
- Phase 25-17: Gap Closure - backend/cpu/simd.rs decomposition (PLANNED)

**Phase 25-16 Summary:**
- Decomposed backend/cpu/simd_ops.rs (1,198 LOC) into 4 focused modules
- mod.rs (617 LOC): SIMD config, helper functions, re-exports, tests
- rmsnorm.rs (252 LOC): RMSNorm SIMD/scalar operations with dispatch
- rope.rs (258 LOC): RoPE SIMD/scalar operations with dispatch
- activations.rs (715 LOC): SiLU, SwiGLU, GELU SIMD/scalar operations
- Pure structural refactor - zero functional changes
- Re-export chains preserve backward compatibility
- Files > 1,000 LOC reduced: 15 -> 4 (-11 files total)

**Phase 25 GAP CLOSURE PLANNED** - 10 gap closure plans created to decompose remaining 12 files >1,000 LOC

**Phase 25-04 Summary:**
- Decomposed loader/gguf.rs (2,284 LOC) into 8 focused modules
- All modules under 1,000 LOC with single responsibility
- Re-export chains preserve backward compatibility
- Tests: 667/667 passing (up from baseline)

**Module Decomposition Pattern Established:**
- Pure structural refactor (ZERO functional changes)
- Re-export chains for backward compatibility
- Generic I/O functions (Read trait) for testability
- Each module < 1,000 LOC with clear responsibility

**Phase 25-05 Summary (PARTIAL - 5/8 targets complete):**
- Target 1: execution_plan_src.rs (4,224 LOC) → 8 modules
- Target 2: kv_cache/kv_cache.rs (2,094 LOC) → 5 modules
- Target 3: sampler/gpu.rs (1,858 LOC) → 4 modules
- Target 4: ggml/hip_backend/mod.rs (1,509 LOC) → 4 modules
- Target 5: attention/kernels.rs (1,395 LOC) → 3 modules

**Decomposition totals so far:**
- 11,310 LOC decomposed into 33 focused modules
- All modules under 1,000 LOC with single responsibility
- Re-export chains preserve backward compatibility
- Tests: 691/691 passing (up from 667 baseline)

**Gap Closure Plans Created (2026-01-20):**
- Wave 2A: 25-08 (engine.rs → 4 modules), 25-09 (scheduler.rs → 3 modules), 25-10 (attention_gpu.rs → 2 modules)
- Wave 2B: 25-11 (kv_cache.rs further decomposition), 25-12 (execution.rs further decomposition)
- Wave 4: 25-13 (http/server.rs → 4 modules)
- Wave 5: 25-14 (rocprof_integration.rs → 2 modules), 25-15 (baseline.rs → 3 modules)
- Wave 6: 25-16 (simd_ops.rs → 3 modules), 25-17 (simd.rs → 2 modules)

**Phase 25 Goal:** Decompose 15 monolithic files (>1,000 LOC) into focused modules

**Files to Decompose:**
| File | LOC | Target |
|------|-----|--------|
| backend/hip_backend/backend.rs | 4,243 | 10 modules (COMPLETE) |
| model/execution_plan/execution_plan_src.rs | 4,224 | 8 modules (COMPLETE) |
| loader/gguf.rs | 2,284 | 7 modules (COMPLETE) |
| kv_cache/kv_cache.rs | 2,094 | 5 modules (COMPLETE - 25-11: 1,032 LOC with cache_ops, sequence_ops, block_ops) |
| sampler/gpu.rs | 1,858 | 5 modules (COMPLETE) |
| http/server.rs | 1,518 | 4 modules (COMPLETE - 25-13) |
| ggml/hip_backend/mod.rs | 1,509 | 4 modules (COMPLETE) |
| profiling/rocprof_integration.rs | 1,396 | 3 modules (COMPLETE - 25-14) |
| attention/kernels.rs | 1,395 | 3 modules (COMPLETE) |
| engine.rs | 1,386 | 4 modules (COMPLETE) |
| scheduler/scheduler.rs | 1,307 | 3 modules (25-09 - COMPLETE) |
| profiling/baseline.rs | 1,233 | 3 modules (COMPLETE - 25-15) |
| ops/attention_gpu.rs | 1,232 | 3 modules (25-10) |
| backend/hip_backend/backend.rs | 1,209 | Main facade - acceptable |
| ggml/hip_backend/execution.rs | 1,207 | 84 LOC + op_dispatch.rs (COMPLETE in 25-12) |
| backend/cpu/simd_ops.rs | 1,198 | 4 modules (25-16 - COMPLETE) |
| backend/cpu/simd.rs | 1,093 | 3 modules (25-17) |

**Total:** 15 files, 28,552 LOC → ~70 modules

**Decision: Pure Structural Refactor (ZERO Functional Changes)**
- Only move code between modules, extract sub-modules, reorganize imports
- NO logic changes, NO bug fixes, NO optimization, NO behavior changes
- Re-export chains preserve backward compatibility
- All 630+ tests must pass before and after

**Phase 25-05 Summary (IN PROGRESS - 1/8 targets complete):**
- Target 1 (execution_plan_src.rs): 4,224 LOC → 8 modules (COMPLETE)
  - types.rs (652 LOC): ExecutionPlan struct, LoadingStats, core accessors
  - embedding.rs (359 LOC): map_embedding_lazy, map_lm_head_lazy, embedding_lookup
  - layer_tensors.rs (138 LOC): create_layer_plan_lazy
  - rope.rs (50 LOC): rope_cache function
  - matmul.rs (298 LOC): matmul, transpose_2d_tensor, reshape helpers
  - execute.rs (807 LOC): forward_layer, self_attention, MLP, attention ops
  - position.rs (47 LOC): get_rope_cache accessor
  - execution_plan_src.rs (814 LOC): GGML integration only
- Tests: 667/667 passing (up from baseline)
- All modules under 1,000 LOC
- Public API preserved via method delegation

**Remaining Targets (2):**
- Target 7: scheduler/scheduler.rs (1,307 LOC) → 3 modules (25-09 - COMPLETE)
- Target 8: ops/attention_gpu.rs (1,232 LOC) → check for duplicate (25-10)

**Phase 25-06 Summary:**
- Decomposed backend/hip_backend/backend.rs (4,243 LOC) into 10 focused modules
- ffi.rs (82 LOC): extern "C" HIP bindings and constants
- error.rs (82 LOC): HipError, HipResult types
- device.rs (113 LOC): HipDeviceProp, HipDevice, device queries
- stream.rs (83 LOC): HipStream wrapper
- event.rs (170 LOC): HipEvent wrapper
- memory.rs (582 LOC): HipBuffer with Arc-based cloning
- module.rs (124 LOC): HipModule, HipKernel loading
- backend.rs (1209 LOC): HipBackend main implementation
- runtime.rs (485 LOC): ModelRuntime for device buffers
- async_ops.rs (220 LOC): AsyncLoader multi-stream uploads
- Tests: 675/675 passing
- All modules under 1,000 LOC except backend.rs (main facade)

**Wave 3 totals:**
- 4,243 LOC decomposed into 10 focused modules
- 3,184 LOC total (original had duplication)
- Re-export chains preserve backward compatibility
- Pure structural refactor - ZERO functional changes

**Phase 25-07 Summary (QA + Verification):**
- Fixed CacheConfig import path after kv_cache decomposition
- Fixed KernelCache private field access by moving kernel modules into kernels_cache/ subdirectory
- Verification results: 675 tests passing (baseline stable)
- 11 files remain over 1,000 LOC (Wave 2 incomplete, Waves 4-6 not started)
- Largest file reduced: 4,243 LOC → 1,518 LOC (-64%)
- New modules created: 27 (all documented with `//!` comments)

**Phase 25-08 Summary (COMPLETE):**
- Decomposed engine.rs (1,386 LOC) into 4 focused modules
- mod.rs (33 LOC): Module facade with re-exports
- types.rs (221 LOC): EngineError, EngineResult, RetryConfig
- config.rs (164 LOC): EngineConfig with builder methods
- stats.rs (117 LOC): EngineStats, HealthStatus
- inference.rs (1119 LOC): InferenceEngine main implementation
- Pure structural refactor - zero functional changes
- Re-export chains preserve backward compatibility
- Files > 1,000 LOC reduced: 11 → 9 (-2 files)

**Phase 25-09 Summary (COMPLETE):**
- Decomposed scheduler/scheduler.rs (1,307 LOC) into 4 focused modules
- types.rs (214 LOC): SchedulerError, RequestState, GenerationRequest, SchedulerResult
- batch.rs (477 LOC): Batch, IterationBatch, SchedulerConfig with paged attention integration
- queue.rs (73 LOC): QueueStats for queue management
- scheduler.rs (712 LOC): Core Scheduler implementation
- Pure structural refactor - zero functional changes
- Re-export chains preserve backward compatibility
- Files > 1,000 LOC reduced: 15 → 9 (-6 files total)

**Phase 25-12 Summary (COMPLETE):**
- Decomposed ggml/hip_backend/execution.rs (1,207 LOC) into 2 modules
- execution.rs (84 LOC): Main execute_op dispatcher
- op_dispatch.rs (1,138 LOC): Individual operation implementations
- Pure structural refactor - zero functional changes
- All operations remain private (pub(crate)) within hip_backend module
- Files > 1,000 LOC reduced: 15 → 8 (-7 files total)
- 93% LOC reduction in execution.rs (1,207 → 84)

**Phase 25-11 Summary (COMPLETE):**
- Further decomposed kv_cache/kv_cache.rs from 1,304 LOC to 1,032 LOC (658 LOC non-test)
- Created 3 operation modules: cache_ops (353 LOC), sequence_ops (238 LOC), block_ops (426 LOC)
- Operation module pattern: Functional operations taking RwLock references
- Delegation pattern: Main struct delegates to operation modules
- Pure structural refactor - zero functional changes
- Files > 1,000 LOC reduced: 15 → 7 (-8 files total)

**Phase 25-10 Summary (COMPLETE):**
- Decomposed src/ops/attention_gpu.rs (1,232 LOC) into ops/attention/ directory
- Created 3 focused modules under 1,000 LOC each:
  - kernels.rs (597 LOC): HipAttentionKernels struct with full attention pipeline
  - softmax.rs (595 LOC): QkMatmul, CausalMaskOp, AttentionSoftmax, WeightedMatmul
  - hiprtc.rs (106 LOC): HIP runtime compilation utilities
- Verified ops/attention/ is distinct from kernels/attention/ (no duplicate code)
- Backward compatibility via module alias: `pub use attention as attention_gpu`
- Files > 1,000 LOC reduced: 15 → 7 (-8 files total)

**Phase 25 Gap Closure Plans (2026-01-20):**
- 10 plans created (25-08 through 25-17)
- Wave 2A: 3 parallel plans (engine COMPLETE, scheduler COMPLETE, ops/attention_gpu COMPLETE)
- Wave 2B: 2 parallel plans (kv_cache COMPLETE, ggml/execution COMPLETE)
- Wave 4: 1 plan (http/server COMPLETE)
- Wave 5: 2 parallel plans (profiling)
- Wave 6: 2 parallel plans (cpu/simd)

**Phase 25 Final Status:**
- Status: GAP CLOSURE IN PROGRESS (9 of 10 complete)
- Files > 1,000 LOC reduced: 15 → 4 (-11 files so far)
- Gap closure targets: 1 remaining file (Wave 6: simd.rs at 1,093 LOC)
- Tests passing: 701 (baseline increased by +1 test from simd_ops decomposition)
- Largest file: 1,093 LOC (backend/cpu/simd.rs - final gap closure target)

**Phase 25-15 Summary:**
- Decomposed profiling/baseline.rs (1,233 LOC) into 3 focused modules
- baseline_types.rs (461 LOC): HardwareInfo, BaselineMetrics, ComparisonResult, RegressionReport, BaselineError, RegressionThreshold
- baseline_storage.rs (167 LOC): save/load functions for baseline persistence, file I/O
- baseline.rs (637 LOC): PerformanceBaseline and BaselineCollection implementations, comparison logic, BenchmarkBaseline helper
- 48% LOC reduction in baseline.rs (1,233 → 637)
- Pure structural refactor - zero functional changes
- Re-export chains preserve backward compatibility
- Files > 1,000 LOC reduced: 15 → 5 (-10 files total)

**Phase 25-14 Summary:**
- Decomposed profiling/rocprof_integration.rs (1,396 LOC) into 3 focused modules
- types.rs (142 LOC): ProfilingError, ProfilingResult, ProfilingTool, CounterCategory
- rocprof.rs (915 LOC): RocprofSession, ProfilingConfig, ProfilingResults, helpers
- omniperf.rs (485 LOC): OmniperfProfileBuilder, MemoryBandwidthAnalysis, MemoryAccessPattern
- rocprof_integration.rs (163 LOC): thin re-export module for backward compatibility
- Pure structural refactor - zero functional changes
- Re-export chains preserve backward compatibility

**Phase 25-13 Summary:**
- Decomposed src/http/server.rs (1,518 LOC) into 4 focused modules
- types.rs (520 LOC): HttpError, ServerError, GenerateRequest, GenerateResponse, TokenStream, GenerationState, ServerState, TracesQuery
- routes.rs (44 LOC): create_router() function with CORS and all endpoints
- handlers.rs (580 LOC): All request handlers (generate, stream, status, cancel, models, health, ready, metrics, traces)
- server.rs (538 LOC): InferenceServer core + run_server() - 65% LOC reduction
- Pure structural refactor - zero functional changes
- Re-export chains preserve backward compatibility

## Session Continuity

Last session: 2026-01-20
Stopped at: Completed 26-02 to_host_vec() migration at 2026-01-20T18:40:33Z
Resume file: Continue warning cleanup with plan 26-03 (remaining warnings)
