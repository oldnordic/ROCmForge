# Changelog

## [Unreleased]

### [CPU Backend]

**feat(cpu): Add per-tensor weight type support**

- **Issue:** Mixed quantization models (e.g., Q4_0 weights with Q4_1 ffn_down) couldn't be handled because CpuLayerWeights only stored a single weight_type per layer
- **Root Cause:** `dispatch_gemv` and `dispatch_gemm` used the general layer `weight_type` for all tensors, causing Q4_1 tensors to be treated as Q4_0 (wrong block size: 18 vs 20 bytes)
- **Fix:**
  - Added individual type fields to CpuLayerWeights: `attn_q_type`, `attn_k_type`, `attn_v_type`, `attn_o_type`, `ffn_gate_type`, `ffn_up_type`, `ffn_down_type`
  - Load actual tensor type from GGUF for each tensor individually
  - Updated `forward.rs` and `prefill.rs` to use per-tensor types in `dispatch_gemv`/`dispatch_gemm`
- **Impact:** Enables loading mixed quantization models correctly
- **Files Changed:** `src/cpu/weights.rs`, `src/cpu/forward.rs`, `src/cpu/prefill.rs`

**feat(cpu): Add Q4_1 GEMM support for prefill**

- **Issue:** Prefill path failed with "unsupported weight type: Q4_1" on mixed quantization models
- **Root Cause:** `dispatch_gemm` only supported F32, Q4_0, and Q8_0, but not Q4_1
- **Fix:**
  - Added `gemm_q4_1` function with proper min offset handling
  - Added Q4_1 case to `dispatch_gemm`
- **Files Changed:** `src/cpu/ops.rs`

**fix(cpu): Q4_0 scalar GEMV copy-paste error**

- **Issue:** Q4_0 scalar function referenced non-existent `min_offset` and `q8_sum` variables
- **Root Cause:** Copy-paste from Q4_1 function left incorrect variables (Q4_0 has no min_offset)
- **Fix:** Removed min_offset references, return only scaled accumulation
- **Files Changed:** `src/cpu/ops.rs`

**fix(cpu): AVX2 Q4_1 horizontal sum overflow protection**

- **Issue:** AVX2 Q4_1 horizontal sum was computing using non-existent `as_m128i()` method
- **Root Cause:** Attempted to use non-existent method for converting `__m256i` to extract sum
- **Fix:** Use `_mm256_hadd_epi16` pairwise addition followed by `_mm256_extract_epi16` to extract final sum
- **Files Changed:** `src/cpu/ops.rs`

### [0.1.1] - 2026-03-25

#### Bug Fixes

**fix(cpu): Q4_1 × Q8_0 GEMV dot product min_offset handling**

- **Issue:** Q4_1 × Q8_0 dot product was incorrectly computing `min_offset * 32` per block instead of `min_offset * sum(q8)`, causing value explosion in FFN down projection
- **Root Cause:** The min_offset parameter was being multiplied by the constant 32 (Q8_0 block size) instead of the actual sum of Q8_0 quantized input values
- **Fix:**
  - Compute `q8_sum` accumulation per block in `dot_q4_1_q8_0_block_scalar`
  - Apply `min_offset * q8_sum` instead of `min_offset * 32`
  - Same fix applied to AVX2 version `dot_q4_1_q8_0_block_avx2`
- **Impact:** Fixes incorrect output values (was exploding to mean=-185, std=29; now normal mean≈0, std≈0.2)
- **Files Changed:** `src/cpu/ops.rs`
