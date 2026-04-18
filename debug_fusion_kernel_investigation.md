# Fusion Kernel Investigation - maeddesg/feat/fused-norm-qkv-rope

**Date:** 2026-04-17
**Branch:** maeddesg/feat/fused-norm-qkv-rope
**Working Baseline:** commit 6dfa163

## Executive Summary

Friend's fusion kernel branch produces garbage output (repetitive loops, Chinese characters). Even with fusion disabled, output remains broken. Root cause analysis reveals **critical type mismatch between KV cache storage and kernel expectations**.

## Key Findings

### 1. KV Cache Type Change: FP16 → FP32

**Location:** `src/gpu/cache.rs`

**Working version (commit 6dfa163):**
```rust
let layer_bytes = max_seq_len * kv_size * std::mem::size_of::<u16>();  // FP16
pub fn k_ptr(&self, layer: usize) -> GpuResult<*mut u16>
pub fn v_ptr(&self, layer: usize) -> GpuResult<*mut u16>
```

**Friend's branch (BROKEN):**
```rust
let layer_bytes = max_seq_len * kv_size * std::mem::size_of::<f32>();  // FP32
pub fn k_ptr(&self, layer: usize) -> GpuResult<*mut f32>
pub fn v_ptr(&self, layer: usize) -> GpuResult<*mut f32>
```

**Impact:** Doubles KV cache memory usage (2x VRAM), changes cache layout interpretation.

### 2. Attention Kernel Updates

**Location:** `hip_kernels/attention.hip`

**Updated signatures:**
```diff
-    const half* __restrict__ k_cache,   // [max_seq * kv_size] (FP16)
-    const half* __restrict__ v_cache,   // [max_seq * kv_size] (FP16)
+    const float* __restrict__ k_cache,  // [max_seq * kv_size]
+    const float* __restrict__ v_cache,  // [max_seq * kv_size]
```

**Updated reads:**
```diff
-            score += head_q[i] * __half2float(k_cache[cache_base + i]);
+            score += head_q[i] * k_cache[cache_base + i];
```

**Status:** Attention kernels correctly updated to read FP32.

### 3. KV Write Kernel Updates

**Location:** `hip_kernels/attention.hip`

**kv_write_state_kernel:**
```diff
-        k_cache[cache_idx] = __float2half(k[i]);
-        v_cache[cache_idx] = __float2half(v[i]);
+        k_cache[cache_idx] = k[i];
+        v_cache[cache_idx] = v[i];
```

**Status:** KV write kernels correctly updated to write FP32.

### 4. Fusion Kernel KV Cache Write

**Location:** `hip_kernels/quant/q4_0_fused_norm_qkv_rope.hip`

**Line 288-291 (K cache write):**
```cpp
const size_t cache_base = (size_t)pos * kv_size;
#pragma unroll
for (int c = 0; c < 4; ++c) {
    k_cache[cache_base + out_col_base + c] = sums[c];
}
```

**BUG IDENTIFIED:** Missing head offset for GQA!

For GQA with 2 KV heads × 64 dims:
- `out_col_base` ranges 0..127 (flat K matrix columns)
- `cache_base = pos * 128` (flat position offset)
- **BUT** attention expects: `cache_base = pos * 128 + head * 64`

**Root cause:** Cache write uses flat K offset, attention read uses head-offset.

### 5. Cache Layout Mismatch

**Attention kernel READ pattern** (`hip_kernels/attention.hip:111`):
```cpp
const size_t cache_base = (size_t)pos * kv_size + head_offset;
score += head_q[i] * k_cache[cache_base + i];
```

**Fusion kernel WRITE pattern** (`hip_kernels/quant/q4_0_fused_norm_qkv_rope.hip:288`):
```cpp
const size_t cache_base = (size_t)pos * kv_size;
k_cache[cache_base + out_col_base + c] = sums[c];
```

**Result:** Attention reads wrong cache locations → garbage scores → garbage output.

## Test Results

### Working Version (commit 6dfa163)
```
Prompt: "Hello, how are you today?"
Output: "I am here to help you with your questions. Please, tell me what you want to know."
Speed: 462.9 tok/s
Coherence: ✅ Perfect
```

### Friend's Branch (BROKEN)
```
Prompt: "Hello, how are you today?"
Output: "I am how is how do what is how what is how如何的怎样的"
Speed: 436.4 tok/s
Coherence: ❌ Repetitive loops + Chinese characters
```

## Files Modified in Friend's Branch

### Core Source Changes:
- `hip_kernels/attention.hip` - Updated for FP32 cache
- `hip_kernels/quant/q4_0_fused_norm_qkv_rope.hip` - Fusion kernel (BUGGY)
- `hip_kernels/quant/q4_0_gemv.hip` - Modified (needs investigation)
- `src/gpu/cache.rs` - Changed to FP32 cache
- `src/gpu/forward.rs` - Updated cache pointer types
- `src/gpu/ops.rs` - Added fusion dispatch
- `src/gpu/kernels/quant.rs` - Added fusion FFI bindings

## Next Steps

1. **Fix fusion kernel KV cache write** - Add head offset (already attempted, still broken)
2. **Investigate other fusion kernel bugs** - Q4_0 dequantization, RoPE calculation
3. **Verify RMS norm computation** - Check if normalization is correct
4. **Compare GEMV implementations** - Fusion vs working kernel pattern matching

## Hypothesis

Even after fixing the KV cache head offset bug, the fusion kernel still produces garbage. This suggests **additional bugs** in:
- Q4_0 block dequantization (uint32_t packing pattern)
- RMS norm computation (shared memory reduction)
- RoPE application (head indexing, angle calculation)
- Bias addition (correctness of pointer arithmetic)

The fusion kernel is too complex to debug without systematic isolation of each phase.
