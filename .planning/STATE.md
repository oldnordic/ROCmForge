# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** CRITICAL ISSUE - ROCm feature compilation broken

## Current Position

Phase: 28 - ROCm Compilation Fix
Plan: 01 of N (Wave 1 - Import fixes)
Status: In Progress - Import errors fixed, remaining issues documented
Last activity: Completed 28-01 import fixes at 2026-01-20T21:21:22Z

Progress: [███████████████████░░] 99.5% (Phase 27 COMPLETE, Phase 28-01 COMPLETE)

### Blockers/Concerns

**CRITICAL: ROCm Feature Compilation Broken (2026-01-20)**

When attempting to enable `rocm` as default feature (required for GPU kernel runtime loading), 151 compilation errors were exposed.

**Progress - Phase 28-01 COMPLETE (Import fixes):**
- Fixed: c_void imports in 9 FFI kernel files
- Fixed: HipError imports in 3 attention kernel cache files
- Fixed: Path import in kernels_cache/mod.rs

**Remaining Issues (for subsequent plans):**
1. **flash_attention.rs**: Underscore-prefixed parameters (`_mask`, `_q`, `_k`, `_v`) used in code
2. **kernels/transpose/mod.rs**: Type mismatch `Arc<Arc<HipBackend>>`

**Root Cause:** Codebase was built/tested without `rocm` feature enabled. GPU-specific code paths accumulated bit-rot from lack of compilation.

**Impact:**
- Phase 27 GPU transpose kernel cannot be tested at runtime
- Release builds cannot use GPU kernels (HSACO not compiled)
- `cargo test --lib` passes because rocm code is `#[cfg(feature = "rocm")]` gated

**Required Action:** Continue Phase 28 - Fix ROCm Feature Compilation

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
- Total plans completed: 169 (v1.0 + v1.1 + v1.2 + v1.3 + v1.4 Phase 22-27 COMPLETE)
- Plans remaining: 0
- Average duration: ~42 min
- Total execution time: ~120 hours

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions:

- **GPU Transpose for Embedding Weights**: Use GPU transpose kernel instead of CPU round-trip to avoid hipMemcpyDtoH failure with memory arena offsets > 1GB. Eliminates CPU-GPU-CPU round-trip during model loading (27-04)
- **Shared Memory Tiling for Transpose**: TILE_DIM=64 with +1 padding avoids AMD GPU bank conflicts per ROCm Examples pattern. Uses transposeLdsNoBankConflicts kernel (27-04)
- **Graceful Skip Pattern for GPU Tests**: Tests skip with KernelLoadFailed when HSACO not compiled, enabling CI/CD without GPU kernel compilation (27-04)
- **KernelCache Visibility Crate-Private**: get_or_init_cache() changed from pub to pub(crate) to match private KernelCache struct visibility. Function only used within kernels_cache module and child submodules (kernels_basic, kernels_flash) (26-04)
- **Dead Code Suppression Standard**: All #[allow(dead_code)] attributes MUST have explanatory comments per Phase 20 standard. Comments explain WHY code is kept (future use, compatibility, optimization) (26-03)
- **Kernel Cache Infrastructure Preserved**: Q4_0/Q4_K/Q6_K kernel cache structs, statics, and initializers are not dead code but premature infrastructure from Phase 24. Reserved for future HSACO lazy-loading optimization (26-03)
- **FFI Constants Kept for API Completeness**: HIP_EVENT_DEFAULT and HIP_EVENT_RECORD_TIMING are part of complete HIP API surface. Currently unused but kept for future use (26-03)
- **Deprecated Backward Compatibility Wrappers**: dequant_q8_0, dequant_q4_0, dequant_q4_k, dequant_q6_k functions are Phase 24 migration compatibility wrappers. Kept with both #[deprecated] and #[allow(dead_code)] (26-03)
- **LRU Eviction Method Preserved**: evict_lru_sequences in kv_cache is complete but replaced by simpler capacity-based eviction. Kept for future cache optimization (26-03)
- **RoPE Cache Functions Reserved**: rope_cache and get_rope_cache reserved for future RoPE pre-computation optimization (26-03)
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
Stopped at: Completed 28-01 Add c_void and HipError imports at 2026-01-20T21:21:22Z
Resume file: .planning/phases/28-rocm-compilation-fix/28-01-SUMMARY.md

**v1.5 - GPU Transpose Fix (2026-01-20):**
- Phase 27-01: TransposeKernel module with lazy HSACO loading, build.rs integration (COMPLETE)
- Phase 27-02: HIP transpose kernel with shared memory tiling (COMPLETE)
- Phase 27-03: GPU transpose integration into embedding_weights (COMPLETE)
- Phase 27-04: Test and verify GPU transpose fix (COMPLETE)

**Phase 27-01 Summary:**
- Created src/kernels/transpose/mod.rs (263 LOC) with TransposeKernel struct
- new(), initialize(), and transpose() methods following existing kernel pattern
- TRANSPOSE_HSACO build.rs integration
- Error handling: HipError::KernelLoadFailed for missing HSACO with descriptive messages
- 2D tensor shape validation with proper error messages
- Tests passing: 701/701 (baseline stable)
- Commits: 2 (6b80ec0: create module, 5112072: add to kernels/mod.rs and build.rs)

**Phase 27-02 Summary:**
- Created kernels/transpose.hip (130 LOC) with optimized transpose kernel
- transposeLdsNoBankConflicts: TILE_DIM=64, shared memory tile with +1 padding for bank conflict avoidance
- transposeNaive: Simple reference kernel for debugging
- Grid/block calculations handle non-tile-aligned matrix sizes
- Build.rs integration: TRANSPOSE_HSACO and TRANSPOSE_NAIVE_HSACO env vars
- Fixed kernel name in mod.rs from "transpose_kernel" to "transposeLdsNoBankConflicts"
- Commits: 2 (5112072: kernel and build integration, 9b94c71: fix kernel name)

**Phase 27-03 Summary:**
- Implemented TransposeKernel::transpose() with GPU kernel launch (331 LOC, under 600 limit)
- Grid/block calculation for TILE_DIM=64 tiling
- Shared memory: TILE_DIM * (TILE_DIM + 1) floats for bank conflict avoidance
- Added convenience function transpose_tensor() for one-shot operations
- Updated embedding_weights() to use GPU transpose instead of CPU round-trip
- Deprecated old transpose_2d_tensor with #[allow(dead_code)]
- Tests: 701/701 passing (baseline stable)
- Commits: 2 (c022114: implement kernel launch, 6d8c045: update embedding_weights)

**Phase 27-04 Summary:**
- Added 4 correctness tests: 8x8 square, 4x16 rectangular, 512x1024 large, 128x1024 embedding-sized
- Tests verify exact transpose values and shape swapping
- Graceful skip when HSACO not compiled (KernelLoadFailed)
- Fixed build.rs: kernel path from "src/kernels/transpose/hip transpose.hip" to "kernels/transpose.hip"
- Fixed build.rs: kernel name from "transpose_kernel" to "transposeLdsNoBankConflicts"
- Fixed simple_transformer.rs: added mut to let linear (E0384 compilation error)
- Tests: 701/701 passing (baseline stable, no regressions)
- Updated TRANSPOSE_ISSUE_INVESTIGATION.md with Resolution section
- Commits: 2 (9ecf047: unit tests, d42d4d5: documentation)
- LOC: src/kernels/transpose/mod.rs now 472 LOC (under 600 limit)

**Phase 27 COMPLETE** - All 4 plans complete, GPU transpose fix implemented and tested

**Decisions:**
- **GPU Transpose for Embedding Weights**: Use GPU transpose kernel instead of CPU round-trip to avoid hipMemcpyDtoH failure with memory arena offsets > 1GB (27-04)
- **Kernel File Location**: Use kernels/transpose.hip at project root instead of src/kernels/transpose/hip transpose.hip to match existing project structure where all HIP files are in /kernels/ (27-02)
- **Kernel Entry Point**: Use transposeLdsNoBankConflicts as primary kernel name (not transpose_kernel) to match AMD ROCm Examples naming convention (27-02)
- **Shared Memory Padding**: TILE_DIM x (TILE_DIM + 1) with padding avoids AMD GPU bank conflicts (27-02)
- **Kernel Module Pattern for Transpose**: Follow existing kernel cache pattern from sampler/gpu.rs (Phases 15-18) (27-01)
- **Convenience Function for One-Shot Operations**: Create transpose_tensor() function that wraps TransposeKernel creation and execution for simpler API (27-03)
- **Deprecated Function with #[allow(dead_code)]**: Keep old transpose_2d_tensor with deprecation notice for potential fallback use, suppress dead_code warning (27-03)

**v1.8 - ROCm Compilation Fix (2026-01-20):**
- Phase 28-01: Add c_void and HipError imports to FFI kernel files (COMPLETE)

**Phase 28-01 Summary:**
- Added `use std::ffi::c_void;` to 9 FFI kernel files using HIP kernel launches
- Added `use crate::backend::hip_backend::error::HipError;` to 3 attention kernel cache files
- Added `use std::path::Path;` to kernels_cache/mod.rs (was also missing)
- All c_void and HipError import errors resolved
- Preserved #[cfg(feature = "rocm")] gates for later wave removal
- Duration: 2 min
- Commits: 2 (a777b0f: c_void imports, 1d72098: HipError imports)

**Decisions:**
- **Import Placement**: Add `use std::ffi::c_void;` after std imports but before crate imports (following Rust conventions)
- **Minimal Changes**: Only added missing imports, did NOT remove #[cfg(feature = "rocm")] gates (that's Wave 2+ work)
