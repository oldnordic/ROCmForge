# Qwen3.5 Hardware Compatibility Analysis

**Date:** 2026-04-14
**Source:** Community research by Kaden Schutt (schuttdev)
**Purpose:** Document Qwen3.5 compatibility issues across AMD GPU architectures

---

## Executive Summary

**Qwen3.5 is not a standard transformer.** It uses a **Gated DeltaNet architecture** with recurrent layers that require a triangular solve operation (`SOLVE_TRI`) through rocBLAS. Standard models (Llama, Mistral, Gemma) never hit this code path, which is why they run fine on the same hardware where Qwen3.5 breaks.

**Key Finding:** Qwen3.5 failures are architecture-specific rocBLAS deficiencies, not user code issues.

---

## Architecture-Specific Issues

### RDNA 4 (gfx1201, RX 9070 XT / R9700)

**Symptom:** SOLVE_TRI crashes

**Affected Models:**
- Qwen3.5-MoE 35B
- Dense Qwen3.5 variants (all sizes)

**Root Cause:** rocBLAS TRSM kernel deficiency at gfx1201 level

**Additional Issue:** ollama packaging bug where hipBLASLt (default rocBLAS backend for RDNA4) was never being copied to the install directory

**Status:**
- ✅ Fix submitted: https://github.com/ollama/ollama/pull/14979 (still open)
- ✅ MES-related bug (GPU stuck at 100% after inference) fixed: https://github.com/ROCm/ROCm/issues/5706 (closed mid-March)

**Workaround:** None currently. Waiting for rocBLAS fix.

---

### RDNA 3.5 (gfx1150/1151, Strix Point / Strix Halo APUs)

**Symptom:** ROCm hangs during load_tensors

**Details:**
- Hangs even with zero GPU layers offloaded
- Points to HIP backend initialization itself, not the offload path

**Workaround:**
```bash
# Use ollama Docker image with override
docker run ollama/ollama:0.17.4-rocm
HSA_OVERRIDE_GFX_VERSION=11.5.1
```

**Performance with workaround:** ~10 tok/s on 27B model

**Status:** Reported upstream: https://github.com/ROCm/ROCm/issues/6027

---

### CDNA 1 (gfx906, MI50)

**Symptom:** SOLVE_TRI crash

**Root Cause:** rocBLAS doesn't ship strsm kernels for gfx906 anymore

**Workarounds:**
1. Use ROCm 6.3.4
2. Use Arch Linux container with pacman-built rocBLAS

**Status:** No official fix. Workarounds required.

---

### MI200 / MI300X (Datacenter GPUs)

**Status:** ✅ Works fine

**Details:** No SOLVE_TRI issues reported on datacenter hardware

---

## Comparison Table

| Architecture | GFX ID | Status | Issue | Workaround |
|--------------|--------|--------|-------|------------|
| **RDNA 4** | gfx1201 | ❌ Crashes | SOLVE_TRI crash (rocBLAS TRSM kernel deficiency) | None (fix pending) |
| **RDNA 3.5** | gfx1150/1151 | ❌ Hangs | ROCm hangs during load_tensors | Docker + HSA_OVERRIDE_GFX_VERSION=11.5.1 |
| **RDNA 3** | gfx1100 | ⚠️ Unknown | Not mentioned in investigation | Needs testing |
| **RDNA 2** | gfx1030 | ⚠️ Unknown | Not mentioned in investigation | Needs testing |
| **RDNA 1** | gfx1010 | ⚠️ Unknown | Not mentioned in investigation | Needs testing |
| **CDNA 1** | gfx906 | ❌ Crashes | No strsm kernels in rocBLAS | ROCm 6.3.4 or Arch container |
| **CDNA 2** | gfx90a | ⚠️ Unknown | Not mentioned in investigation | Needs testing |
| **CDNA 3** | gfx942 | ✅ Works | MI200/MI300X work fine | None needed |

---

## Technical Background

### What is DeltaNet?

**DeltaNet** is a gated linear attention architecture that differs from standard transformers:

**Standard Transformer (Llama, Mistral, Gemma):**
- Self-attention with KV cache
- Standard matrix operations
- No SOLVE_TRI required

**DeltaNet (Qwen3.5):**
- Gated linear attention with recurrent layers
- Requires triangular solve operation (`SOLVE_TRI`)
- Uses rocBLAS TRSM kernels for the solve

**Why Qwen3.5 Breaks:**
- rocBLAS TRSM kernels have varying support across GPU architectures
- Newer architectures (RDNA 4, RDNA 3.5) have incomplete or buggy TRSM implementations
- Older architectures (CDNA 1) have dropped support

---

## Impact on LLM Inference Engines

### llama.cpp / ollama

**Issue:** Rely on rocBLAS for SOLVE_TRI operation

**Consequence:** Qwen3.5 fails where other models work

**Status:** At the mercy of rocBLAS support for each architecture

### hipfire

**Approach:** Native tiled LDS kernel for DeltaNet

**Advantage:** Bypasses rocBLAS entirely for DeltaNet operations

**Performance:** 8.7x faster than llama.cpp on Qwen3.5-9B (45 tok/s vs 4.93 tok/s on 5700 XT)

**Why It Works:**
- No dependency on rocBLAS TRSM kernels
- Custom kernel implementation for DeltaNet
- Architecture-specific optimization

---

## Recommendations

### For Users with Affected Hardware

**RDNA 4 (RX 9070 XT / R9700):**
- Wait for rocBLAS fix (no workaround currently)
- Monitor: https://github.com/ollama/ollama/pull/14979
- Consider hipfire for Qwen3.5 (if available)

**RDNA 3.5 (Strix Point / Strix Halo):**
- Use Docker workaround: `ollama/ollama:0.17.4-rocm` with `HSA_OVERRIDE_GFX_VERSION=11.5.1`
- Accept ~10 tok/s on 27B models
- Monitor: https://github.com/ROCm/ROCm/issues/6027

**CDNA 1 (MI50):**
- Use ROCm 6.3.4, or
- Use Arch Linux container with pacman-built rocBLAS
- Plan hardware upgrade if possible

### For Engine Developers

**Lesson from hipfire:** Custom kernel implementations can bypass rocBLAS limitations

**Approaches:**
1. **Native DeltaNet Kernel** (hipfire approach)
   - Tiled LDS implementation
   - Warp shuffle FWHT
   - Architecture-specific optimization

2. **Alternative SOLVE_TRI Implementation**
   - Custom triangular solve
   - Avoid rocBLAS dependency
   - Portable across architectures

3. **Architecture Detection and Fallback**
   - Detect rocBLAS TRSM support at runtime
   - Fall back to custom kernel if unavailable
   - Graceful degradation

---

## Relevance to rocmforge Q6_K Work

### Why This Matters for Q6_K

**Q6_K is a quantization format, not an architecture issue.** However, the Qwen3.5 investigation reveals:

1. **rocBLAS Has Gaps**
   - TRSM kernels missing or buggy on some architectures
   - Newer architectures don't always have complete rocBLAS support
   - Community testing exposes these gaps

2. **Custom Kernels Enable Portability**
   - hipfire bypasses rocBLAS with custom DeltaNet kernel
   - 8.7x speedup by avoiding rocBLAS limitations
   - Architecture-specific optimization yields massive gains

3. **Testing Across Architectures is Critical**
   - Issues vary wildly across GPU generations
   - What works on MI300X may crash on RX 9070
   - Community testing is invaluable

### Applied to Q6_K Refactoring

**Device Function Pattern:**
- ✅ Reduces rocBLAS dependency (more self-contained)
- ✅ Enables architecture-specific optimization
- ✅ Bypasses potential rocBLAS limitations
- ✅ Aligns with hipfire's approach

**Register Pressure Optimization:**
- ✅ Critical for all architectures (hipfire proves this)
- ✅ Target < 20 VGPRs (hipfire achieves 18)
- ✅ Yields 2.16x more concurrent wavefronts

**Testing Strategy:**
- Test across multiple GPU architectures (RDNA 1/2/3, CDNA)
- Monitor for architecture-specific issues
- Consider custom kernels for critical paths

---

## Additional Resources

### Community Investigation

**Original Research:** Kaden Schutt (schuttdev)
- GitHub: https://github.com/Kaden-Schutt/hipfire
- Models: https://huggingface.co/schuttdev/models

### Upstream Issues

**ollama packaging fix:**
- https://github.com/ollama/ollama/pull/14979

**MES bug fix:**
- https://github.com/ROCm/ROCm/issues/5706

**RDNA 3.5 hang issue:**
- https://github.com/ROCm/ROCm/issues/6027

### Related Documentation

**hipfire Analysis:** `docs/amd_resources/hipfire/ANALYSIS.md`
**AMD Resources Summary:** `docs/amd_resources/SUMMARY.md`
**HIP Graph Documentation:** `docs/hip_graph_capture_analysis.md`

---

## Conclusion

**Qwen3.5 compatibility issues are rocBLAS deficiencies, not fundamental incompatibilities.**

**Key Insights:**
1. DeltaNet requires SOLVE_TRI operation through rocBLAS
2. rocBLAS TRSM kernel support varies widely across architectures
3. Custom kernel implementations (hipfire) can bypass these limitations
4. Community testing exposes gaps that AMD hasn't addressed

**For Q6_K Work:**
- Device function pattern reduces rocBLAS dependency
- Custom kernels enable portability across architectures
- Register pressure optimization is critical (hipfire proves this)
- Testing across multiple GPU architectures is essential

**The Path Forward:**
- ✅ Refactor Q6_K with device function pattern (Task #63)
- ✅ Target low register pressure (< 20 VGPRs)
- ✅ Enable architecture-specific optimization
- ✅ Test across multiple GPU generations
- ✅ Consider custom kernels for critical paths

---

**Status:** Qwen3.5 hardware compatibility documented ✅ | rocBLAS limitations identified ✅ | Custom kernel approach validated ✅

**Last Updated:** 2026-04-14
