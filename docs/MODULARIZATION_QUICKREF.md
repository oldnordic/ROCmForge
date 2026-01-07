# ROCmForge Modularization Quick Reference

**Status:** 13 files violate GLOBAL ENGINEERING CONSTRAINTS
**Critical:** 5 files >1000 LOC require immediate splitting
**High Priority:** 6 files >600 LOC require splitting
**Monitor:** 2 files 400-600 LOC (cohesive, acceptable exceedance)

---

## Critical Violations (>1000 LOC) - P0

### 1. src/model/execution_plan.rs - 2,305 LOC (384% over limit)
**Split into:** `execution_plan/` directory with 11 files
**Concerns:** Architecture detection, weight mapping (5 types), forward pass, QKV concat, transpose
**Effort:** 16 hours
**Risk:** HIGH (complex weight mapping logic)

### 2. src/backend/hip_backend.rs - 2,135 LOC (356% over limit)
**Split into:** `hip_backend/` directory with 13 files + extract `ModelRuntime`
**Concerns:** FFI bindings, memory, kernel loading, tensor ops, BLAS, MLP, LayerNorm, transformer execution
**Effort:** 20 hours
**Risk:** CRITICAL (FFI safety, global state, GPU memory)

### 3. src/loader/gguf.rs - 1,950 LOC (325% over limit)
**Split into:** `gguf/` directory with 12 files
**Concerns:** MXFP types, parsing, dequantization (7 formats), GPU upload, config conversion
**Effort:** 14 hours
**Risk:** MEDIUM (dequantization correctness)

### 4. src/ops/attention_gpu.rs - 1,222 LOC (204% over limit)
**Split into:** `attention/` directory with 7 files
**Concerns:** Kernel compilation, QK^T, masking, softmax, attention-weighted V, CPU fallbacks
**Effort:** 10 hours
**Risk:** MEDIUM (clear operation boundaries)

### 5. src/attention/kernels.rs - 955 LOC (159% over limit)
**Split into:** `kernels/` directory with 8 files
**Concerns:** Kernel cache, 10+ kernel wrappers
**Effort:** 12 hours
**Risk:** MEDIUM (cache initialization complexity)

---

## High Priority Violations (600-1000 LOC) - P1

### 6. src/engine.rs - 803 LOC (134% over limit)
**Split into:** `engine/` directory with 6 files
**Concerns:** Config, model loading, inference loop, request management
**Effort:** 8 hours

### 7. src/http/server.rs - 662 LOC (110% over limit)
**Split into:** `http/` directory with 4 files
**Concerns:** Server setup, route handlers, DTOs, WebSocket
**Effort:** 6 hours

### 8. src/scheduler/scheduler.rs - 640 LOC (107% over limit)
**Split into:** `scheduler/` directory with 5 files
**Concerns:** Config, queue management, batching, state
**Effort:** 6 hours

### 9. src/attention/multi_query.rs - 609 LOC (102% over limit)
**Status:** MONITOR - Cohesive single concern
**Action:** Document exception, no split needed

### 10. src/model/simple_transformer.rs - 606 LOC (101% over limit)
**Status:** MONITOR - Cohesive single concern
**Action:** Document exception, no split needed

### 11. src/model/glm_position.rs - 601 LOC (100% over limit)
**Split into:** `glm/` directory with 3 files
**Concerns:** GLM architecture, position embeddings
**Effort:** 4 hours

---

## Monitoring Required (400-600 LOC) - P2

### 12. src/models.rs - 484 LOC
### 13. src/sampler/sampler.rs - 474 LOC
### 14. src/backend/gpu_executor.rs - 456 LOC

**Status:** MONITOR - Modest exceedance, likely cohesive
**Action:** Review during next refactoring cycle

---

## Refactoring Timeline

### Phase 1: Critical (Weeks 1-4) - 60 hours
- Week 1-2: `hip_backend.rs` (20h) - HIGHEST RISK
- Week 2-3: `execution_plan.rs` (16h)
- Week 3-4: `gguf.rs` (14h)
- Week 4: `attention_gpu.rs` (10h)

### Phase 2: High Priority (Weeks 5-7) - 26 hours
- Week 5: `kernels.rs` (12h)
- Week 6: `engine.rs` (8h)
- Week 6: `server.rs` (6h)
- Week 7: `scheduler.rs` (6h)
- Week 7: `glm_position.rs` (4h)

### Phase 3: Monitoring (Week 8+) - 8 hours
- Review P2 files
- Update documentation
- CI/CD integration

**Total Effort:** 94 hours over 8 weeks

---

## Risk Summary

| File | Risk Level | Primary Concerns |
|------|-----------|------------------|
| `hip_backend.rs` | CRITICAL | FFI safety, global state, GPU memory |
| `execution_plan.rs` | HIGH | Weight mapping, architecture detection |
| `gguf.rs` | MEDIUM | Dequantization correctness |
| `attention_gpu.rs` | MEDIUM | Operation boundaries, fallbacks |
| `kernels.rs` | MEDIUM | Cache initialization |
| Others | LOW | Clear boundaries, straightforward splits |

---

## Key Modularization Patterns

### Pattern 1: Subdirectory Modules
```
src/module_name/
├── mod.rs              # Public API
├── core.rs             # Primary struct
├── sub_concern_1.rs    # Isolated concern
└── utils.rs            # Helpers
```

### Pattern 2: Operations Subdirectory
```
src/ops/operation_name/
├── mod.rs              # API
├── gpu.rs              # GPU implementation
└── fallback.rs         # CPU fallback
```

### Pattern 3: Configuration Isolation
```
src/module/
├── config.rs           # Config structs
├── impl.rs             # Implementation
└── types.rs            # Associated types
```

---

## Success Metrics

- [ ] Zero files >600 LOC (hard ceiling)
- [ ] <5 files >300 LOC (soft limit)
- [ ] <10% average LOC per file
- [ ] 0% performance regression
- [ ] 100% test pass rate

---

## Quick Actions

1. **Immediate:** Begin `hip_backend.rs` refactoring (highest risk)
2. **This Week:** Set up CI/CD LOC monitoring
3. **Next Sprint:** Create tracking issues for Phase 1
4. **Ongoing:** Document modularization patterns for team

---

**Last Updated:** 2026-01-07
**Full Analysis:** See `ARCHITECTURE_MODULARIZATION_ANALYSIS.md`
