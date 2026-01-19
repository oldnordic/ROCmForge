# Requirements: ROCmForge v1.2 Technical Debt Cleanup + Performance

**Version:** 1.2
**Date:** 2026-01-19
**Status:** Draft
**Core Value:** Reliable, fast inference on AMD GPUs with transparent CPU fallback.

---

## v1 Requirements

### Code Hygiene (HYGIENE)

Requirements for eliminating technical debt and establishing a clean codebase baseline.

- [x] **HYGIENE-01**: Scheduler clone bug is fixed — `update_iteration_batch` no longer overwrites with stale clones
- [ ] **HYGIENE-02**: All unreachable pattern warnings are eliminated (2 locations)
- [ ] **HYGIENE-03**: All `#[allow(dead_code)]` markers are reviewed and resolved
- [ ] **HYGIENE-04**: All unused import warnings are eliminated
- [ ] **HYGIENE-05**: All unused variable/assignment warnings are eliminated
- [ ] **HYGIENE-06**: All unexpected_cfg warnings are resolved (feature name fixes)
- [ ] **HYGIENE-07**: Zero compiler warnings baseline achieved (`cargo check` passes clean)

### GPU Sampling Kernels (SAMPLING)

Requirements for GPU-accelerated token sampling.

- [ ] **SAMPLING-01**: Top-k sampling runs on GPU (no CPU fallback)
- [ ] **SAMPLING-02**: Top-p (nucleus) sampling runs on GPU
- [ ] **SAMPLING-03**: Temperature sampling runs on GPU
- [ ] **SAMPLING-04**: Sampling kernels are added to build.rs compilation list
- [ ] **SAMPLING-05**: Sampling kernels have unit tests verifying correctness
- [ ] **SAMPLING-06**: Sampling kernels have integration tests with real models

### GPU RoPE (ROPE)

Requirements for GPU-accelerated rotary position embeddings.

- [x] **ROPE-01**: RoPE application runs on GPU for GPU tensors
- [x] **ROPE-02**: Multi-head rotation is handled correctly on GPU
- [x] **ROPE-03**: Position IDs are handled for long context (>2048 tokens)
- [x] **ROPE-04**: RoPE kernel is added to build.rs compilation list
- [x] **ROPE-05**: RoPE kernel has unit tests verifying rotation correctness
- [x] **ROPE-06**: CPU-GPU transfer overhead is eliminated (no round-trip)

### GPU Quantization (QUANT)

Requirements for on-device quantization operations.

- [x] **QUANT-01**: Q4_0 dequantization runs on GPU
- [x] **QUANT-02**: Q4_K dequantization runs on GPU (if not already implemented)
- [x] **QUANT-03**: Q6_K dequantization runs on GPU (if not already implemented)
- [x] **QUANT-04**: Quantized matmul fusion runs on GPU (dequant + matmul in one kernel)
- [x] **QUANT-05**: Quantization kernels have unit tests verifying bit-exact outputs
- [x] **QUANT-06**: CPU dequantization fallback is removed for GPU tensors

### GPU Attention (ATTENTION)

Requirements for GPU-accelerated attention mechanisms.

- [x] **ATTENTION-01**: FlashAttention variant is verified working on GPU
- [x] **ATTENTION-02**: Multi-query attention (MQA) runs fully on GPU
- [x] **ATTENTION-03**: Grouped-query attention (GQA) runs fully on GPU
- [x] **ATTENTION-04**: Attention kernels are added to build.rs (if missing)
- [x] **ATTENTION-05**: Attention kernels have correctness tests

### Test Health (TEST)

Requirements for fixing broken tests and completing test coverage.

- [ ] **TEST-01**: Memory allocation crash in `decode_step_integration_tests` is fixed
- [ ] **TEST-02**: Pre-existing `test_kv_cache_eviction_at_capacity` failure is fixed
- [ ] **TEST-03**: E2E tests are unignored and run with `ROCFORGE_TEST_MODEL` env var
- [ ] **TEST-04**: E2E tests have graceful skip when model file not found
- [ ] **TEST-05**: All 572 lib tests pass
- [ ] **TEST-06**: All integration tests pass

### Performance Validation (PERF)

Requirements for validating performance improvements.

- [ ] **PERF-01**: GPU sampling is faster than CPU fallback (10x+ speedup)
- [ ] **PERF-02**: GPU RoPE eliminates CPU-GPU transfer overhead
- [ ] **PERF-03**: End-to-end inference latency is improved vs v1.1 baseline
- [ ] **PERF-04**: Memory usage is efficient (quantized weights stay on GPU)

---

## v2 Requirements (Deferred)

Deferred to future milestone (v1.3+).

- [ ] **LARGE-01**: Multi-GPU support — Single GPU focus for v1.x
- [ ] **LARGE-02**: Continuous batching optimization — Fixed batch size for v1.2
- [ ] **LARGE-03**: KV cache eviction/paging to host — Fixed cache for v1.2
- [ ] **LARGE-04**: Speculative decoding — Performance optimization only

---

## Out of Scope

Explicitly excluded from v1.2.

| Feature | Reason |
|---------|--------|
| Training features (LoRA, fine-tuning) | Focus is inference-only |
| Non-text modalities (vision, audio) | Text-only for v1 |
| Non-AMD GPU support | ROCm/HIP only |
| New quantization formats | 15 formats sufficient for v1.2 |
| HTTP API new features | Focus on backend performance |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| HYGIENE-01 | Phase 14 | Complete |
| HYGIENE-02 | Phase 19 | Pending |
| HYGIENE-03 | Phase 19 | Pending |
| HYGIENE-04 | Phase 19 | Pending |
| HYGIENE-05 | Phase 19 | Pending |
| HYGIENE-06 | Phase 19 | Pending |
| HYGIENE-07 | Phase 19 | Pending |
| SAMPLING-01 | Phase 15 | Validated |
| SAMPLING-02 | Phase 15 | Validated |
| SAMPLING-03 | Phase 15 | Validated |
| SAMPLING-04 | Phase 15 | Validated |
| SAMPLING-05 | Phase 15 | Validated |
| SAMPLING-06 | Phase 15 | Validated |
| ROPE-01 | Phase 16 | Validated |
| ROPE-02 | Phase 16 | Validated |
| ROPE-03 | Phase 16 | Validated |
| ROPE-04 | Phase 16 | Validated |
| ROPE-05 | Phase 16 | Validated |
| ROPE-06 | Phase 16 | Validated |
| QUANT-01 | Phase 17 | Validated |
| QUANT-02 | Phase 17 | Validated |
| QUANT-03 | Phase 17 | Validated |
| QUANT-04 | Phase 17 | Validated |
| QUANT-05 | Phase 17 | Validated |
| QUANT-06 | Phase 17 | Validated |
| ATTENTION-01 | Phase 18 | Validated |
| ATTENTION-02 | Phase 18 | Validated |
| ATTENTION-03 | Phase 18 | Validated |
| ATTENTION-04 | Phase 18 | Validated |
| ATTENTION-05 | Phase 18 | Validated |
| TEST-01 | Phase 20 | Pending |
| TEST-02 | Phase 20 | Pending |
| TEST-03 | Phase 20 | Pending |
| TEST-04 | Phase 20 | Pending |
| TEST-05 | Phase 20 | Pending |
| TEST-06 | Phase 20 | Pending |
| PERF-01 | Phase 20 | Pending |
| PERF-02 | Phase 20 | Pending |
| PERF-03 | Phase 20 | Pending |
| PERF-04 | Phase 20 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0 ✓

---

*Requirements defined: 2026-01-19*
*Last updated: 2026-01-19 after roadmap creation*
