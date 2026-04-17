# Kernel Improvement Strategy

**Date:** 2026-04-15
**Status:** Active strategy document

## Overview

We have extensive documentation from debugging sessions, profiling work, and reference implementation analysis. This document outlines how to systematically leverage that knowledge to improve ALL GPU kernels, not just fix issues reactively.

---

## Documentation Assets

### Critical Lessons Learned

**From `q6_k_debugging_lessons_learned.md`:**
1. **Performance Without Correctness is Meaningless**
   - We measured 131 tok/s for months while Q6_K produced garbage
   - Always add output correctness tests, not just performance/safety tests
   - All performance benchmarks are meaningless without correctness validation

2. **Comparison with Reference Implementation is Essential**
   - Should have compared with llama.cpp from the start
   - Before optimizing, verify your implementation matches the reference
   - Reference implementation revealed bugs immediately

3. **Systematic Debugging Works**
   - Reproduce → Rule out → Compare → Fix → Verify
   - Don't skip steps, don't guess

### Design Guidelines

**From `gpu_kernel_design_guidelines.md`:**

**Graph Capture Compatibility:**
1. Linear thread processing (fixed iteration counts)
2. No data-dependent branching
3. Predictable memory access (direct indexing)
4. Warp-sized thread blocks (32 threads)

**Safety Verification (before committing):**
1. ✅ Works with graph capture enabled (for compatible types)
2. ✅ Works with graph capture disabled (all types)
3. ✅ No GPU crashes (check dmesg)
4. ✅ Tested with real model (not just unit tests)
5. ✅ **Output correctness verified** (new requirement after Q6_K)

### Reference Implementation Mapping

**From `llama_cpp_hip_kernel_mapping.md`:**

**Dominant kernels in llama.cpp:**
- `mul_mat_vec_q<Q4_0, 1>`: 25.64% - Q4_0 mat-vec decode
- `rms_norm_f32<32>`: 13.66% - RMS normalization
- `quantize_q8_1`: 13.12% - Activation quantization
- `k_bin_bcast<op_add/op_mul>`: 16.35% - Elementwise operations
- `flash_attn_vec_ext_f32`: 14.24% (with `-fa`) - Attention

**Actionable guidance:**
1. Prioritize Q4_0 decode GEMV buckets (matches both llama.cpp and rocmforge hotspots)
2. Reduce launch count for elementwise add/mul/silu chains
3. Keep reducing decode-time standalone quantization launches

### Hotpath Architecture

**From `GPU_DECODE_HOTPATH.md`:**

**Per-layer decode sequence (24 layers):**
1. RMS_NORM (attn_norm)
2. Fused QKV projection (Q4_0)
3. RoPE (Q)
4. KV write + RoPE (K)
5. Flash attention decode
6. Output projection + residual (Q4_0)
7. RMS_NORM (ffn_norm)
8. Fused gate+up+SwiGLU (Q4_0)
9. FFN down projection + residual (Q4_0)

**Kernel dispatch table:**
| Weight | Type | Kernel |
|--------|------|--------|
| attn_q,k,v | Q4_0 | `gemv_qkv_q4_0_f32_on_stream` (fused) |
| attn_o | Q4_0 | `gemv_q4_0_f32_residual_on_stream` (fused) |
| ffn_gate,up | Q4_0 | `gemv_gate_up_swiglu_q4_0_f32_on_stream` (fused) |
| ffn_down | Q4_0 | `gemv_q4_0_f32_residual_on_stream` (fused) |

---

## Systematic Improvement Strategy

### Phase 1: Correctness Audit (ALL kernels)

**Apply Q6_K lesson to all quantization kernels:**

For each kernel type (Q4_0, Q4_K, Q5_K, Q8_0, Q6_K):

1. **Create correctness test**
   - Load real model
   - Generate output
   - Verify coherence (English text, not garbage)
   - Compare against llama.cpp reference

2. **Compare with llama.cpp implementation**
   - Read reference implementation completely
   - Check for algorithmic differences
   - Verify memory access patterns match

3. **Document status**
   - Mark as ✅ Correct or ❌ Broken
   - Note any deviations from reference
   - Add performance numbers (only if correct)

**Priority order:**
1. Q4_0 (most used, 25% of llama.cpp time)
2. Q8_0 (embeddings and lm_head)
3. Q4_K (common format)
4. Q5_K
5. Q6_K (already fixed, document as ✅)

### Phase 2: Graph Compatibility Audit

**Apply design guidelines to all kernels:**

For each kernel:

1. **Test graph capture**
   ```bash
   ROCMFORGE_DISABLE_DECODE_GRAPH=0 ./target/release/rocmforge --gpu --model <model>
   ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu --model <model>
   ```

2. **Check for violations**
   - Data-dependent branching?
   - Non-linear iteration?
   - Complex pointer arithmetic?

3. **Document compatibility**
   - ✅ Graph compatible
   - ⚠️ Partially compatible (note limitations)
   - ❌ Graph incompatible (explain why)

### Phase 3: Performance Optimization (CORRECT kernels only)

**For kernels that pass Phase 1:**

1. **Profile with rocprofv3**
   ```bash
   ./.rocprofv3/profile_decode.sh runtime
   ```

2. **Compare with llama.cpp**
   - Check relative performance
   - Identify major gaps (>2x slower)

3. **Optimize systematically**
   - Use register pressure analysis
   - Apply vectorization cautiously
   - Test after EVERY change (correctness regression test)

### Phase 4: Documentation Updates

**Keep documentation synchronized with reality:**

1. **Update `gpu_kernel_design_guidelines.md`**
   - Add new patterns as discovered
   - Note graph compatibility status
   - Document safety requirements

2. **Update `llama_cpp_hip_kernel_mapping.md`**
   - Add new profiling data
   - Track optimization progress
   - Note remaining gaps

3. **Create per-format documents**
   - Q4_0 implementation notes
   - Q4_K implementation notes
   - etc.

---

## Implementation Roadmap

### Immediate (this week)

**Task 1: Correctness tests for all quantization formats**
- [ ] Create `tests/q4_0_correctness_test.rs`
- [ ] Create `tests/q8_0_correctness_test.rs`
- [ ] Create `tests/q4_k_correctness_test.rs`
- [ ] Create `tests/q5_k_correctness_test.rs`
- [ ] Run all tests, document results

**Task 2: Compare all kernels with llama.cpp**
- [ ] Read Q4_0 reference implementation
- [ ] Read Q8_0 reference implementation
- [ ] Read Q4_K reference implementation
- [ ] Document any algorithmic differences

**Task 3: Update CHANGELOG with correctness audit results**
- [ ] Mark each format as ✅ or ❌
- [ ] Note any issues found

### Short-term (next 2 weeks)

**Task 4: Graph compatibility audit**
- [ ] Test each kernel with graph capture
- [ ] Document compatibility status
- [ ] Fix any graph-breaking issues (if feasible)

**Task 5: Performance profiling**
- [ ] Profile correct kernels with rocprofv3
- [ ] Compare with llama.cpp baselines
- [ ] Identify optimization targets

### Long-term (next month)

**Task 6: Systematic optimization**
- [ ] Optimize largest gaps first
- [ ] Apply lessons from Q6_K register pressure work
- [ ] Re-verify correctness after each optimization

**Task 7: Documentation maintenance**
- [ ] Keep guidelines up to date
- [ ] Add new lessons learned
- [ ] Maintain reference mapping

---

## Success Metrics

**Correctness:**
- ✅ All quantization formats produce coherent output
- ✅ Output matches llama.cpp reference (within tolerance)
- ✅ No garbage output like Q6_K had

**Performance:**
- Target: Within 2x of llama.cpp HIP performance
- Current: Q4_0 ~146 tok/s (need llama.cpp baseline)
- Current: Q6_K ~79 tok/s (need llama.cpp baseline)

**Safety:**
- ✅ All 4 GPU safety tests pass
- ✅ No GPU crashes (dmesg clean)
- ✅ Works with both graph and non-graph paths

**Documentation:**
- ✅ All formats documented with status
- ✅ Design guidelines reflect reality
- ✅ Reference mapping up to date

---

## Anti-Patterns to Avoid

Based on Q6_K experience:

1. **❌ Don't measure performance without verifying correctness**
   - 131 tok/s is meaningless if output is garbage

2. **❌ Don't optimize without comparing to reference**
   - Our "linear optimization" was actually a bug

3. **❌ Don't assume previous work was correct**
   - Always verify with tests

4. **❌ Don't skip graph compatibility testing**
   - May work on single-token but fail on multi-token

5. **❌ Don't add features without correctness tests**
   - All new quantization formats need correctness validation

---

## Next Steps

1. **Start correctness audit immediately** - this is blocking everything else
2. **Create test infrastructure** - make correctness testing easy
3. **Document findings** - keep knowledge accessible
4. **Only optimize correct kernels** - performance is secondary to correctness

---

**Philosophy:** "It works fast" is not enough. It must work correctly first, then we make it fast.

**Reference:** Q6_K debugging session proved this - we celebrated 131 tok/s for months while producing garbage. Never again.
