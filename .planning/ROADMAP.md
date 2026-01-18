# Roadmap: ROCmForge

## Overview

Build a production-ready LLM inference engine for AMD GPUs that is reliable, fast, and universally compatible with GGUF models. Start by fixing critical bugs blocking inference, then establish solid testing foundations, modularize the codebase, implement CPU SIMD fallback, complete GPU kernels, optimize attention, enable hybrid execution, ensure broad GGUF compatibility, optimize for balanced performance, and harden for production use.

## Domain Expertise

None (no applicable domain expertise found)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Critical Bug Fixes** - Fix inference hangs and GPU synchronization bugs
- [ ] **Phase 2: Test Infrastructure** - Restore commented tests and improve coverage
- [ ] **Phase 3: Codebase Modularization** - Split large files into focused modules
- [ ] **Phase 4: CPU SIMD Backend** - Implement optimized CPU operations with SIMD
- [ ] **Phase 5: Quantized Operations** - Native HIP dequantization kernels
- [ ] **Phase 6: Attention Optimization** - Flash attention detection and GPU kernels
- [ ] **Phase 7: Hybrid Execution Scheduler** - Automatic CPU/GPU op selection
- [ ] **Phase 8: GGUF Compatibility** - Universal model support across architectures
- [ ] **Phase 9: Performance Optimization** - Balanced throughput, latency, memory efficiency
- [ ] **Phase 10: Production Hardening** - Error handling, logging, monitoring, documentation

## Phase Details

### Phase 1: Critical Bug Fixes
**Goal**: Fix inference hangs and GPU synchronization bugs blocking reliable execution
**Depends on**: Nothing (first phase)
**Research**: Unlikely (bugs identified in CONCERNS.md)
**Status**: ✅ Complete (2026-01-18)

Plans:
- [x] 01-01: Fix GPU stream synchronization (hipBLAS vs hipMemcpy mismatch)
- [x] 01-02: Fix inference loop spawn race condition
- [x] 01-03: Fix engine cleanup in CLI

### Phase 2: Test Infrastructure
**Goal**: Restore commented tests and improve test coverage
**Depends on**: Phase 1
**Research**: Unlikely (testing patterns exist in codebase)
**Status**: 📋 Planned (2026-01-18)

Plans:
- [ ] 02-01: Rewrite 20+ commented GGUF loader tests for new API
- [ ] 02-02: Restore embedding_to_lmhead tests
- [ ] 02-03: Add end-to-end inference tests
- [ ] 02-04: Replace unwrap() with proper error handling in tests

### Phase 3: Codebase Modularization
**Goal**: Split large files (>3000 LOC) into focused, maintainable modules
**Depends on**: Phase 2
**Research**: Unlikely (internal refactoring)
**Plans**: TBD

Plans:
- [ ] 03-01: Split execution_plan.rs (4410 lines) into focused modules
- [ ] 03-02: Split hip_backend.rs (3625 lines) into focused modules
- [ ] 03-03: Split gguf.rs (2832 lines) into focused modules
- [ ] 03-04: Consolidate duplicate test fixtures

### Phase 4: CPU SIMD Backend
**Goal**: Implement optimized CPU operations with SIMD for transparent fallback
**Depends on**: Phase 3
**Research**: Likely (SIMD crate selection, CPU architecture optimization)
**Research topics**: Rust SIMD ecosystem (std::simd, packed_simd, wide), CPU feature detection
**Plans**: TBD

Plans:
- [ ] 04-01: Research and select SIMD strategy for CPU ops
- [ ] 04-02: Implement CPU backend trait with SIMD matmul
- [ ] 04-03: Implement SIMD for attention operations
- [ ] 04-04: Add CPU feature detection and runtime selection

### Phase 5: Quantized Operations
**Goal**: Native HIP dequantization kernels for efficient quantized inference
**Depends on**: Phase 3
**Research**: Likely (HIP kernel development, quantization formats)
**Research topics**: Quantization formats (Q4_0, Q4_K, Q5_0, Q6_K, Q8_0), HIP kernel patterns
**Plans**: TBD

Plans:
- [ ] 05-01: Research quantization formats and dequantization algorithms
- [ ] 05-02: Implement HIP dequantization kernel for Q4_0
- [ ] 05-03: Implement HIP dequantization kernels for remaining formats
- [ ] 05-04: Integrate dequantization + matmul fused kernel

### Phase 6: Attention Optimization
**Goal**: Flash attention detection and GPU kernels for optimized inference
**Depends on**: Phase 4
**Research**: Likely (flash attention algorithms, ROCm attention patterns)
**Research topics**: Flash attention algorithms, ROCm attention libraries, kernel optimization
**Plans**: TBD

Plans:
- [ ] 06-01: Research flash attention implementation for ROCm
- [ ] 06-02: Implement flash attention detection in backend registry
- [ ] 06-03: Implement flash attention HIP kernel
- [ ] 06-04: Benchmark and optimize attention performance

### Phase 7: Hybrid Execution Scheduler
**Goal**: Automatic CPU/GPU operation selection for maximum compatibility
**Depends on**: Phase 4, Phase 5
**Research**: Likely (scheduler design, op cost modeling)
**Research topics**: Hybrid execution patterns, op cost modeling, fallback strategies
**Plans**: TBD

Plans:
- [ ] 07-01: Design hybrid execution scheduler architecture
- [ ] 07-02: Implement per-operation CPU/GPU availability tracking
- [ ] 07-03: Implement automatic op selection based on availability
- [ ] 07-04: Add telemetry for execution path debugging

### Phase 8: GGUF Compatibility
**Goal**: Universal GGUF support across all model architectures and quantizations
**Depends on**: Phase 5
**Research**: Likely (GGUF format variations, model architectures)
**Research topics**: GGUF format spec, LLaMA vs Qwen vs Mistral architectures, quantization compatibility
**Plans**: TBD

Plans:
- [ ] 08-01: Research GGUF format variations and model architectures
- [ ] 08-02: Add support for missing architectures (Mistral, Yi, etc.)
- [ ] 08-03: Ensure all quantization formats load correctly
- [ ] 08-04: Add model compatibility test matrix

### Phase 9: Performance Optimization
**Goal**: Balanced optimization of throughput, latency, and memory efficiency
**Depends on**: Phase 6, Phase 7
**Research**: Unlikely (profiling and optimization based on existing code)
**Plans**: TBD

Plans:
- [ ] 09-01: Profile and optimize throughput (tokens/second)
- [ ] 09-02: Profile and optimize latency (first token time)
- [ ] 09-03: Profile and optimize memory efficiency (KV cache, allocations)
- [ ] 09-04: Add performance benchmarks and regression tests

### Phase 10: Production Hardening
**Goal**: Error handling, logging, monitoring, and documentation for production use
**Depends on**: Phase 9
**Research**: Unlikely (production readiness practices)
**Plans**: TBD

Plans:
- [ ] 10-01: Replace unwrap() with proper error handling in production paths
- [ ] 10-02: Complete eprintln! to tracing migration
- [ ] 10-03: Add .env.example and configuration documentation
- [ ] 10-04: Add health checks and monitoring endpoints

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Critical Bug Fixes | 3/3 (3 planned) | Complete | 2026-01-18 |
| 2. Test Infrastructure | 0/4 (4 planned) | Planned | - |
| 3. Codebase Modularization | 0/4 | Not started | - |
| 4. CPU SIMD Backend | 0/4 | Not started | - |
| 5. Quantized Operations | 0/4 | Not started | - |
| 6. Attention Optimization | 0/4 | Not started | - |
| 7. Hybrid Execution Scheduler | 0/4 | Not started | - |
| 8. GGUF Compatibility | 0/4 | Not started | - |
| 9. Performance Optimization | 0/4 | Not started | - |
| 10. Production Hardening | 0/4 | Not started | - |
