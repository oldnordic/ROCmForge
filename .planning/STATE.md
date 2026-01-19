# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-19)

**Core value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.
**Current focus:** Phase 20 - Code Hygiene Completion

## Current Position

Phase: 20 of 20 (Code Hygiene Completion)
Plan: 1 of 8 in current phase
Status: In progress
Last activity: 2026-01-19 — Completed Phase 20-01: Fix unreachable pattern and cfg warnings

Progress: [█████████░░░░░░░░░░░░] 29% (19.125 of 20 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 133 (v1.0 + v1.1 + v1.2 through 20-01)
- Average duration: ~44 min
- Total execution time: ~80 hours

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
| 18-01 | 1 | ~10min | 10 min |
| 18-02 | 1 | ~12min | 12 min |
| 18-03 | 1 | ~18min | 18 min |
| 19-01 | 1 | ~1min | 1 min |
| 19-02 | 1 | ~3min | 3 min |
| 19-03 | 1 | ~1min | 1 min |
| 19-04 | 1 | ~15min | 15 min |
| 20-01 | 1 | ~4min | 4 min |

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
- **18-01 FlashAttention GPU Verification**: Verified all 3 FlashAttention kernels registered in build.rs; causal and non-causal kernels compile and execute correctly (GPU matches CPU within tolerance); generic kernel has compilation issue (CUDA `__shfl_down_f32` intrinsic needs HIP `__shfl_down`); FlashAttentionBackend verified pure GPU path with no CPU round-trips
- **18-02 MQA/GQA KV Replication Verification**: Verified mqa_kv_replicate.hip kernel source, build.rs integration, and Rust wrapper mqa_kv_replicate_gpu_kernel() all present; confirmed forward_device() uses pure GPU path with no CPU round-trips; RoPE is pre-applied at model layer so TODO in multi_query.rs is not a blocker; 4 MQA/GQA tests ready for GPU testing
- **18-03 End-to-End GPU Attention Integration Tests**: Added 8 comprehensive integration tests covering FlashAttention, MQA, GQA with realistic model configurations; all ATTENTION-01 through ATTENTION-05 requirements now satisfied; GPU vs CPU consistency tests with 1e-3 tolerance; REQUIREMENTS.md updated to mark all ATTENTION requirements complete
- **19-01 Quantization Format Analysis**: Documented Q4_0 (32 elements/block, 20 bytes, scale + 4-bit signed values), Q4_K (256-element super-blocks, 8 sub-blocks with scale/min), Q6_K (256-element blocks, 16 f16 scales, 6-bit signed values); CPU references verified as ground truth for GPU kernel numerical validation
- **19-02 HIP Intrinsics Replacement**: Corrected WARP_SIZE from 32 to 64 for RDNA3 wavefront alignment; replaced CUDA `__shfl_down_f32` with HIP `__shfl_down` in Q4_0, Q4_K, Q6_K matmul kernels (6 occurrences); TILE_SIZE_K/N=32 documented as not wave64-aligned (deferred optimization)
- **19-03 Fused RMSNorm CUDA Intrinsics Removal**: Corrected WARP_SIZE from 32 to 64 in fused_dequant_rmsnorm.hip; replaced final CUDA `__shfl_down_f32` with HIP `__shfl_down`; all 4 quantized kernels (Q4_0, Q4_K, Q6_K, fused RMSNorm) are now CUDA-intrinsic-free
- **19-04 HIP Kernel Compilation and Validation**: Compiled all 4 quantized matmul kernels for gfx1100 (q4_0_matmul.hsaco, q4_k_matmul.hsaco, q6_k_matmul.hsaco, fused_q4_0_rmsnorm.hsaco); replaced invalid `__builtin_amdgcn_wave_reduce_fadd` with manual `__shfl_down` reduction; fixed Q4_K and Q6_K CPU test bugs (bit_pos calculation and 6-bit extraction); validated 12 CPU dequantization tests passing; verified WARP_SIZE=64 and zero CUDA intrinsics across all kernels

### Pending Todos

None yet.

### Blockers/Concerns

- ~~**Scheduler Clone Bug**: `update_iteration_batch` overwrites scheduler state with stale batch clones~~ **RESOLVED** (14-01)
- ~~**topk_sampling.hip watchdog timeout**: Single-threaded loops over vocab_size caused GPU hang~~ **RESOLVED** (15-03)
- ~~**topp_sampling Rust integration**: Existing `src/sampler/gpu.rs` expects single kernel but we implemented 3-kernel pipeline; needs API updates~~ **RESOLVED** (15-05)
- **RoPE GPU kernel execution bug**: `rope_gpu_kernel()` returns -1 (execution failed) - blocks RoPE GPU tests (16-02)
- **FlashAttention generic kernel compilation**: CUDA intrinsic `__shfl_down_f32` in flash_attention.hip needs to be replaced with HIP `__shfl_down` (18-01) - separate from quantized kernels which are now CUDA-intrinsic-free
- **Quantized matmul tile size alignment**: TILE_SIZE_K=32, TILE_SIZE_N=32 not wave64-aligned (deferred optimization, documented in 19-02-SUMMARY.md)
- **topk_topp_sampling watchdog risk**: Fused kernel uses single-threaded loops over vocab_size (documented in TODO comments); refactor to parallel pattern before production use
- **Code quality note**: 71 lib warnings remain; duplicate `GgufMetadata` structs exist (pre-existing technical debt)

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
- Phase 18-01: Verified FlashAttention kernels; causal and non-causal kernels work correctly; generic kernel has CUDA intrinsic compilation issue
- Phase 18-02: Verified MQA/GQA KV replication kernel; confirmed pure GPU execution path; RoPE pre-applied at model layer
- Phase 18-03: End-to-end GPU attention integration tests; 8 new tests for FlashAttention, MQA, GQA; all ATTENTION requirements satisfied
- Phase 19-01: Quantization format analysis; documented Q4_0, Q4_K, Q6_K bit-packing layouts; CPU references validated as ground truth
- Phase 19-02: HIP intrinsics replacement; corrected WARP_SIZE to 64 for wave64; replaced __shfl_down_f32 with __shfl_down; documented tile size alignment
- Phase 19-03: Fused RMSNorm CUDA intrinsics removal; corrected WARP_SIZE to 64; replaced __shfl_down_f32 with __shfl_down; all 4 quantized kernels are CUDA-intrinsic-free
- Phase 19-04: Compile and validate HIP quantized kernels; compiled 4 HSACO files for gfx1100; replaced invalid __builtin_amdgcn_wave_reduce_fadd; fixed Q4_K and Q6_K CPU test bugs; validated all 12 CPU dequantization tests
- Phase 20-01: Removed duplicate GgufMetadata struct from ggml.rs; consolidated to single definition in loader.rs
- Phase 20-02: Renamed Q4_K/Q6_K type aliases (Q4_KDequantResult to Q4KdequantResult, Q6_KDequantResult to Q6KdequantResult, etc.) to follow Rust naming conventions; made Q4_0DequantCache public; eliminated 5 compiler warnings
- Phase 20-03: Removed 8 unused imports (6 HipError, 2 KernelTimer) across hip_backend ops modules; eliminated all unused import warnings

## Session Continuity

Last session: 2026-01-19
Stopped at: Completed 20-03 — Removed unused imports, HYGIENE-04 satisfied
Resume file: None
