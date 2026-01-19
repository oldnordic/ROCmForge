# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models.

## Milestones

- **v1.0 Production-Ready** — Phases 1-12 + 12.1A + 12.1B (shipped 2026-01-19)
- **v1.1 Bug Fix Release** — Phases 13-01, 13-02, 13-03 (shipped 2026-01-19)
- **v1.2 Technical Debt Cleanup + Performance** — Phases 14-20 (in progress)

## Phases

<details>
<summary>v1.0 Production-Ready (Phases 1-12 + 12.1A + 12.1B) — SHIPPED 2026-01-19</summary>

**Full details archived in:** [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

- [x] Phase 1: Critical Bug Fixes (3/3 plans)
- [x] Phase 2: Test Infrastructure (4/4 plans)
- [x] Phase 3: Codebase Modularization (4/4 plans)
- [x] Phase 4: CPU SIMD Backend (4/4 plans)
- [x] Phase 5: Quantized Operations (4/4 plans)
- [x] Phase 6: Attention Optimization (4/4 plans)
- [x] Phase 7: Hybrid Execution Scheduler (4/4 plans)
- [x] Phase 8: GGUF Compatibility (11/11 plans)
- [x] Phase 9: Performance Optimization (18/18 plans)
- [x] Phase 10: Production Hardening (20/20 plans)
- [x] Phase 11: Fix Test Suite & Verify E2E (2/2 plans)
- [x] Phase 12: Complete CPU SIMD Attention (4/4 plans)
- [x] Phase 12.1A: CPU SIMD Completion (2/2 plans)
- [x] Phase 12.1B: Context Engine Integration (1/1 plan)

**Total:** 96 plans across 13 phases

</details>

<details>
<summary>v1.1 Bug Fix Release (Phases 13-01, 13-02, 13-03) — SHIPPED 2026-01-19</summary>

**Full details archived in:** [.planning/milestones/v1.1-ROADMAP.md](.planning/milestones/v1.1-ROADMAP.md)

- [x] Phase 13-01: Qwen2 head_dim Fix (1/1 plans)
- [x] Phase 13-02: Memory Pooling Documentation (1/1 plans)
- [x] Phase 13-03: Dead Code Removal (4/4 plans)

**Total:** 6 plans across 3 phases

**Summary:** Fixed Qwen2 model loading, corrected documentation, reduced compiler warnings 93%

</details>

---

## v1.2 Technical Debt Cleanup + Performance (In Progress)

**Milestone Goal:** Eliminate technical debt, fix all broken tests, and implement missing GPU kernels for full GPU acceleration.

### Phase 14: Scheduler Clone Bug Fix

**Goal**: Verify and document the scheduler clone bug fix where `update_iteration_batch` could overwrite scheduler state with stale batch clones.

**Depends on**: Nothing (first phase of v1.2)

**Requirements**: HYGIENE-01

**Success Criteria** (what must be TRUE):
1. `update_iteration_batch` no longer overwrites scheduler state with stale clones
2. Test `test_update_iteration_batch_cannot_clobber_new_tokens` passes
3. Multi-token generation produces correct output without token loss

**Plans:** 2 plans

- [x] 14-01-PLAN.md — Verify fix and add test alias matching requirements
- [x] 14-02-PLAN.md — Refactor to Entry API for cleaner implementation (optional)

---

### Phase 15: GPU Sampling Kernels

**Goal**: All token sampling operations (top-k, top-p, temperature) run on GPU with 10x+ speedup.

**Depends on**: Phase 14

**Requirements**: SAMPLING-01, SAMPLING-02, SAMPLING-03, SAMPLING-04, SAMPLING-05, SAMPLING-06

**Success Criteria** (what must be TRUE):
1. Top-k sampling runs on GPU (no CPU fallback)
2. Top-p (nucleus) sampling runs on GPU
3. Temperature sampling runs on GPU
4. Sampling kernels are added to build.rs compilation list
5. Sampling kernels have unit tests verifying correctness

**Plans:** 7 plans in 5 waves

- [x] 15-01-PLAN.md — Add sampling_utils.hip (softmax, temperature) to build.rs
- [x] 15-02-PLAN.md — Add topk_sampling.hip to build.rs and fix timeout
- [x] 15-03-PLAN.md — Replace stub topp_sampling.hip with multi-kernel pipeline
- [x] 15-04-PLAN.md — Add fused topk_topp_sampling.hip to build.rs
- [x] 15-05-PLAN.md — Update GPU sampler cache to load compiled HSACO files
- [x] 15-06-PLAN.md — Update GPU samplers to use GPU kernels instead of CPU
- [x] 15-07-PLAN.md — Add comprehensive unit and integration tests

**Summary:** GPU sampling kernels implemented with multi-kernel pipeline for top-p (prefix_sum, threshold, sample), parallel bitonic sort for top-k, temperature scaling support, and comprehensive unit/integration tests.

---

### Phase 16: GPU RoPE Implementation

**Goal**: Rotary position embeddings apply on GPU, eliminating CPU-GPU transfer overhead per layer.

**Depends on**: Phase 15

**Requirements**: ROPE-01, ROPE-02, ROPE-03, ROPE-04, ROPE-05, ROPE-06

**Success Criteria** (what must be TRUE):
1. RoPE application runs on GPU for GPU tensors
2. Multi-head rotation is handled correctly on GPU
3. Position IDs are handled for long context (>2048 tokens)
4. RoPE kernel is added to build.rs compilation list
5. CPU-GPU transfer overhead is eliminated (no round-trip)

**Plans**: 2 plans in 1 wave

- [x] 16-01-PLAN.md — Verify RoPE kernel compilation and GPU path purity (ROPE-01, ROPE-04, ROPE-06)
- [x] 16-02-PLAN.md — Add comprehensive RoPE GPU tests (ROPE-02, ROPE-03, ROPE-05)

**Summary:** RoPE GPU kernels already exist (rope.hip, position_embeddings.hip). Plan 01 verifies compilation and confirms no CPU round-trip. Plan 02 adds long context and multi-head tests, verifies existing tests pass.

---

### Phase 17: GPU Quantization

**Goal**: On-device quantization operations (Q4_0, Q4_K, Q6_K dequant + fused matmul) keep weights on GPU.

**Depends on**: Phase 16

**Requirements**: QUANT-01, QUANT-02, QUANT-03, QUANT-04, QUANT-05, QUANT-06

**Success Criteria** (what must be TRUE):
1. Q4_0 dequantization runs on GPU
2. Q4_K dequantization runs on GPU
3. Q6_K dequantization runs on GPU
4. Quantized matmul fusion runs on GPU (dequant + matmul in one kernel)
5. CPU dequantization fallback is removed for GPU tensors

**Plans:** 3 plans in 2 waves

- [x] 17-01-PLAN.md — Implement Q4_0 GPU dequantization kernel wrapper
- [x] 17-02-PLAN.md — Implement Q4_K and Q6_K GPU dequantization kernel wrappers
- [x] 17-03-PLAN.md — Integrate fused matmul kernels and remove CPU fallback

**Wave Structure:**
- Wave 1: 17-01 (Q4_0 dequant), 17-02 (Q4_K/Q6_K dequant) — parallel execution
- Wave 2: 17-03 (fused matmul, depends on dequant wrappers)

---

### Phase 18: GPU Attention Completion

**Goal**: All attention mechanisms (FlashAttention, MQA, GQA) run fully on GPU.

**Depends on**: Phase 17

**Requirements**: ATTENTION-01, ATTENTION-02, ATTENTION-03, ATTENTION-04, ATTENTION-05

**Success Criteria** (what must be TRUE):
1. FlashAttention variant is verified working on GPU
2. Multi-query attention (MQA) runs fully on GPU
3. Grouped-query attention (GQA) runs fully on GPU
4. Attention kernels are added to build.rs (if missing)
5. Attention kernels have correctness tests

**Plans:** 3 plans in 2 waves

Plans:
- [x] 18-01-PLAN.md — Verify FlashAttention GPU kernels (generic, causal, non-causal)
- [x] 18-02-PLAN.md — Verify MQA/GQA KV replication and GPU execution path
- [x] 18-03-PLAN.md — Create integration tests and verify all ATTENTION requirements

**Wave Structure:**
- Wave 1: 18-01 (FlashAttention), 18-02 (MQA/GQA) — parallel execution
- Wave 2: 18-03 (integration tests, depends on 01 and 02)

---

### Phase 19: Wavefront-Native Quantized Matmul Kernels

**Goal**: Quantized matmul kernels (Q4_0, Q4_K, Q6_K, fused_dequant_rmsnorm) rewritten as HIP-native code, removing all CUDA intrinsics.

**Depends on**: Phase 18

**Requirements**: QUANT-07, QUANT-08, QUANT-09

**Success Criteria** (what must be TRUE):
1. All `__shfl_down_f32` intrinsics replaced with HIP `__shfl_down`
2. Kernels compile for gfx1100 (RDNA3) without errors
3. Tile sizes are wave64-aligned (divisible by 64)
4. No warp32 assumptions remain in quantized kernels
5. Numerical correctness validated against llama.cpp reference

**Plans**: 4 plans in 4 waves

Plans:
- [x] 19-01-PLAN.md — Analyze and document Q4_0, Q4_K, Q6_K bit-packing formats from CPU reference implementations
- [x] 19-02-PLAN.md — Replace __shfl_down_f32 with __shfl_down in q4_0_matmul.hip, q4_k_matmul.hip, q6_k_matmul.hip
- [x] 19-03-PLAN.md — Replace __shfl_down_f32 with __shfl_down in fused_dequant_rmsnorm.hip and verify all kernels
- [x] 19-04-PLAN.md — Compile kernels for gfx1100 and validate numerical correctness

**Wave Structure:**
- Wave 1: 19-01 (format analysis)
- Wave 2: 19-02 (Q4_0, Q4_K, Q6_K intrinsics - depends on 01)
- Wave 3: 19-03 (fused RMSNorm intrinsics - depends on 02)
- Wave 4: 19-04 (compilation and validation - depends on 02, 03)

**Local References:**
- `/home/feanor/Projects/rocm-examples/HIP-Basic/warp_shuffle/main.hip` — HIP wave operations
- `/home/feanor/Projects/rocm-examples/HIP-Basic/shared_memory/` — LDS usage patterns

**AMD Documentation:**
- https://rocm.docs.amd.com/projects/HIP/en/latest/
- https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html

---

### Phase 20: Code Hygiene Completion

**Goal**: Zero compiler warnings baseline achieved after all feature work is complete.

**Depends on**: Phase 19

**Requirements**: HYGIENE-02, HYGIENE-03, HYGIENE-04, HYGIENE-05, HYGIENE-06, HYGIENE-07

**Success Criteria** (what must be TRUE):
1. All unreachable pattern warnings are eliminated (2 locations)
2. All `#[allow(dead_code)]` markers are reviewed and resolved
3. All unused import warnings are eliminated
4. All unused variable/assignment warnings are eliminated
5. All unexpected_cfg warnings are resolved (feature name fixes)
6. Zero compiler warnings baseline achieved (`cargo check` passes clean)

**Plans:** 8 plans in 4 waves

Plans:
- [ ] 20-01-PLAN.md — Fix unreachable patterns and unexpected_cfg warnings (HYGIENE-02, HYGIENE-06)
- [ ] 20-02-PLAN.md — Fix type naming style warnings (Q4_K, Q6_K naming)
- [ ] 20-03-PLAN.md — Remove unused imports (8 imports across 6 files)
- [ ] 20-04-PLAN.md — Replace deprecated methods (18 calls to to_host_vec/copy_to_host)
- [ ] 20-05-PLAN.md — Fix unused mut, unused assignment, and privacy warnings
- [ ] 20-06-PLAN.md — Categorize dead code warnings and fix unused fields
- [ ] 20-07-PLAN.md — Review and fix unused functions, methods, constants
- [ ] 20-08-PLAN.md — Review #[allow(dead_code)] markers and verify zero baseline

**Wave Structure:**
- Wave 1: 20-01 (unreachable/cfg), 20-02 (naming), 20-03 (imports) — parallel execution
- Wave 2: 20-04 (deprecated methods), 20-05 (simple hygiene) — parallel execution
- Wave 3: 20-06 (dead code categorization + unused fields)
- Wave 4: 20-07 (functions/constants), 20-08 (#[allow] review + verify baseline) — parallel execution

---

### Phase 21: Test Health & Performance Validation

**Goal**: All tests passing and performance validated against v1.1 baseline.

**Depends on**: Phase 20

**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, PERF-01, PERF-02, PERF-03, PERF-04

**Success Criteria** (what must be TRUE):
1. Memory allocation crash in `decode_step_integration_tests` is fixed
2. Pre-existing `test_kv_cache_eviction_at_capacity` failure is fixed
3. E2E tests are unignored and run with `ROCFORGE_TEST_MODEL` env var
4. E2E tests have graceful skip when model file not found
5. All 572 lib tests pass
6. All integration tests pass
7. GPU sampling is faster than CPU fallback (10x+ speedup)
8. End-to-end inference latency is improved vs v1.1 baseline

**Plans**: TBD

---

## Progress

**Execution Order:** Phases execute in numeric order

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Critical Bug Fixes | v1.0 | 3/3 | Complete | 2026-01-18 |
| 2. Test Infrastructure | v1.0 | 4/4 | Complete | 2026-01-18 |
| 3. Codebase Modularization | v1.0 | 4/4 | Complete | 2026-01-18 |
| 4. CPU SIMD Backend | v1.0 | 4/4 | Complete | 2026-01-18 |
| 5. Quantized Operations | v1.0 | 4/4 | Complete | 2026-01-18 |
| 6. Attention Optimization | v1.0 | 4/4 | Complete | 2026-01-18 |
| 7. Hybrid Execution Scheduler | v1.0 | 4/4 | Complete | 2026-01-18 |
| 8. GGUF Compatibility | v1.0 | 11/11 | Complete | 2026-01-18 |
| 9. Performance Optimization | v1.0 | 18/18 | Complete | 2026-01-18 |
| 10. Production Hardening | v1.0 | 20/20 | Complete | 2026-01-18 |
| 11. Fix Test Suite & Verify E2E | v1.0 | 2/2 | Complete | 2026-01-18 |
| 12. Complete CPU SIMD Attention | v1.0 | 4/4 | Complete | 2026-01-18 |
| 12.1A. CPU SIMD Completion | v1.0 | 2/2 | Complete | 2026-01-18 |
| 12.1B. Context Engine Integration | v1.0 | 1/1 | Complete | 2026-01-18 |
| 13-01. Qwen2 head_dim Fix | v1.1 | 1/1 | Complete | 2026-01-18 |
| 13-02. Memory Pooling Documentation | v1.1 | 1/1 | Complete | 2026-01-18 |
| 13-03. Dead Code Removal | v1.1 | 4/4 | Complete | 2026-01-18 |
| 14. Scheduler Clone Bug Fix | v1.2 | 2/2 | Complete | 2026-01-19 |
| 15. GPU Sampling Kernels | v1.2 | 7/7 | Complete | 2026-01-19 |
| 16. GPU RoPE Implementation | v1.2 | 2/2 | Complete | 2026-01-19 |
| 17. GPU Quantization | v1.2 | 3/3 | Complete | 2026-01-19 |
| 18. GPU Attention Completion | v1.2 | 3/3 | Complete | 2026-01-19 |
| 19. Wavefront-Native Quantized Matmul | v1.3 | 4/4 | Complete | 2026-01-19 |
| 20. Code Hygiene Completion | v1.3 | 0/8 | Not started | - |
| 21. Test Health & Performance | v1.3 | 0/? | Not started | - |

**Total Progress:** 129/139 v1.0+v1.1+v1.2+v1.3 plans complete (93%)

---

## v1.3 Critical: CUDA Intrinsics Elimination

**Milestone Goal:** Remove ALL CUDA-specific code from ROCmForge, establishing pure HIP-native quantized matmul kernels.

**IMPORTANT:** This is a critical milestone. ROCmForge does NOT port CUDA code. All kernels must be rewritten from mathematical first principles for AMD HIP.

**Reference:** `.planning/research/ANTI_CUDA_PORTING_RATIONALE.md`

---

## Future Milestones

**v1.4**: Multi-GPU support, advanced optimizations
