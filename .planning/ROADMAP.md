# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models.

## Milestones

- **v1.0 Production-Ready** — Phases 1-12 + 12.1A + 12.1B (shipped 2026-01-19)
- **v1.1 Bug Fix Release** — Phases 13-01, 13-02, 13-03 (current)

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

---

## v1.1 Bug Fix Release

**Milestone Goal:** Fix critical bugs blocking inference on Qwen2 models, correct documentation regarding memory allocation, and clean up dead code.

### Phase 13-01: Qwen2 head_dim Fix

**Goal**: Qwen2 models load and run inference without `buffer_size=0` errors caused by zero-initialized `head_dim` values.

**Depends on**: Phase 12.1B (v1.0 baseline)

**Requirements**: QWEN-01, QWEN-02, QWEN-03, QWEN-04, QWEN-05, QWEN-06

**Success Criteria** (what must be TRUE):
1. User can load any Qwen2 or Qwen2.5 GGUF model without encountering `buffer_size=0` errors
2. User can run inference on Qwen2 models and receive valid completions
3. LLaMA, Mistral, and Gemma models continue to load and run without regression
4. Unit tests verify `head_dim` is calculated correctly for models with and without the `qwen2.rope.dimension_count` GGUF key

**Plans:** 1 plan (Wave 1)

- [x] 13-01-PLAN.md — Add calculate_default_head_dim() and fix rope.dimension_count parsing

### Phase 13-02: Memory Pooling Documentation & Validation

**Goal**: Document the actual memory allocation strategy (direct allocation, not selective pooling) and validate that the current approach works correctly.

**Depends on**: Phase 13-01

**Requirements**: ROCM-01, ROCM-02, ROCM-03, ROCM-04

**Success Criteria** (what must be TRUE):
1. Documentation accurately reflects current code state (no selective pooling implemented)
2. D2H operations use direct allocation only (no sub-buffer D2H calls occur)
3. Current approach is documented as the intentional workaround for ROCm 7.1 D2H limitation
4. Full test suite passes, validating current approach is stable

**Plans:** 1 plan (Wave 1)

- [x] 13-02-01-PLAN.md — Verify current implementation, update misleading docs, create MEMORY_ARCHITECTURE.md

**Note:** This phase was reconceived from "validation of selective pooling" to "documentation + reality check" after research confirmed that selective memory pooling was never actually implemented. The current direct-allocation-only approach avoids the ROCm 7.1 D2H bug by not creating sub-buffers at all.

### Phase 13-03: Dead Code Removal

**Goal**: Reduce compiler warnings from 404 to under 60 by removing unused code, resolving `#[allow(dead_code)]` markers, replacing deprecated methods, and cleaning up unused imports/variables.

**Depends on**: Phase 13-02

**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03

**Success Criteria** (what must be TRUE):
1. All `#[allow(dead_code)]` markers have been reviewed and either removed (code deleted) or kept with justification (FFI, TODO for future features)
2. Unused FFI declarations have been reviewed (all FFI in backend.rs is actively used)
3. Deprecated method calls replaced with current APIs (copy_to_host -> copy_from_device_safe, ExecutionPlan::new -> from_gguf)
4. GGUF naming convention warnings suppressed with #[allow(non_camel_case_types)]
5. Compiler warnings reduced from 404 to under 60
6. Full test suite passes (no functionality was broken by cleanup)

**Plans:** 3 plans (2 waves)

- [ ] 13-03-01-PLAN.md — Resolve dead_code markers and suppress naming warnings (Wave 1)
- [ ] 13-03-02-PLAN.md — Replace deprecated methods with current APIs (Wave 2)
- [ ] 13-03-03-PLAN.md — Bulk cleanup of unused imports/variables (Wave 3, depends on 01-02)

---

## Progress

**Execution Order:** Phases execute in numeric order: 13-01 -> 13-02 -> 13-03

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
| 10. Production Hardening | v1.0 | 20/20 | Complete | 2026-01-19 |
| 11. Fix Test Suite & Verify E2E | v1.0 | 2/2 | Complete | 2026-01-19 |
| 12. Complete CPU SIMD Attention | v1.0 | 4/4 | Complete | 2026-01-19 |
| 12.1A. CPU SIMD Completion | v1.0 | 2/2 | Complete | 2026-01-19 |
| 12.1B. Context Engine Integration | v1.0 | 1/1 | Complete | 2026-01-19 |
| 13-01. Qwen2 head_dim Fix | v1.1 | 1/1 | Complete | 2026-01-19 |
| 13-02. Memory Pooling Documentation | v1.1 | 1/1 | Complete | 2026-01-19 |
| 13-03. Dead Code Removal | v1.1 | 0/3 | Not started | - |

**Total Progress:** 98/101 plans complete (97%)
