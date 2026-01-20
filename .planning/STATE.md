# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Phase 22 - Memory Pool Implementation

## Current Position

Phase: 23 - Dead/Duplicate Code Removal (Plan 03 of 5)
Status: Phase 23 In Progress - GgufMetadata consolidated, MXFP/Q-format cleanup pending
Last activity: GgufMetadata consolidation complete - duplicate struct removed from gguf.rs

Progress: [████████████████░░░░░] 60% (Phase 22 COMPLETE, Phase 23 3/5 plans complete)

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
- Total plans completed: 143 (v1.0 + v1.1 + v1.2 + v1.3 + v1.4 Phase 22, Phase 23 23-01/23-03)
- Plans remaining: 1 (21-06, skipped) + 9 (Phases 23-24)
- Average duration: ~44 min
- Total execution time: ~105 hours

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions:

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
- Phase 23-03: Consolidate GgufMetadata - remove duplicate from gguf.rs, keep metadata.rs version

**Phase 23 In Progress** - 3 of 5 plans complete

**Decision: Single Source of Truth for GgufMetadata**
- Keep GgufMetadata in src/loader/metadata.rs (has both update_from_kv() and calculate_default_head_dim())
- Remove incomplete duplicate from src/loader/gguf.rs (only had calculate_default_head_dim())
- Use canonical import path: crate::loader::metadata::GgufMetadata (23-03)

## Session Continuity

Last session: 2026-01-20
Stopped at: Completed Phase 23-03 (GgufMetadata consolidation - duplicate removed from gguf.rs)
Resume file: None
