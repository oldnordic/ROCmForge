# Phase 25: CLI Inference Hang Investigation - GGUF Shape Transpose

**Date**: 2026-01-12
**Status**: IN PROGRESS
**Issue**: CLI inference hangs in first layer forward pass

---

## Summary

Discovered and partially fixed a critical issue where GGUF files store embedding weights in **transposed format** (`[hidden_size, vocab_size]` instead of `[vocab_size, hidden_size]`). This caused shape validation to fail. After fixing the transpose, a **new hang** was discovered in the first layer's `forward_layer()` computation.

---

## Problem 1: Transposed Embedding Shape ✅ FIXED

### Root Cause

GGUF format stores `token_embd.weight` as `[hidden_size, vocab_size]`:
```
GGUF:     [896, 151936]    → [hidden_size, vocab_size]
Expected: [151936, 896]    → [vocab_size, hidden_size]
```

### Why This Matters

The `embedding_lookup` function expects contiguous embeddings:
```rust
// For token_id T, embedding is at offset T * hidden_size
let src_offset = token_index * hidden_size;
```

With transposed format, each token's embedding is **not contiguous** - it's strided across memory. The lookup code cannot handle this.

### Fix Applied

**File**: `src/loader/gguf.rs:901-921`

```rust
// Special handling for embedding weights: GGUF stores them as [hidden_size, vocab_size]
// but our code expects [vocab_size, hidden_size]. We need to transpose.
let (f32_data, shape) = if name == "token_embd.weight" || name == "lm_head.weight" || name == "output.weight" {
    let dims = shape.dims();
    if dims.len() == 2 {
        let (d0, d1) = (dims[0], dims[1]);
        // Transpose: [hidden_size, vocab_size] -> [vocab_size, hidden_size]
        let mut transposed = vec![0.0f32; f32_data.len()];
        for i in 0..d0 {
            for j in 0..d1 {
                transposed[j * d0 + i] = f32_data[i * d1 + j];
            }
        }
        (transposed, TensorShape::from_dims(&[d1, d0]))
    } else {
        (f32_data, shape)
    }
} else {
    (f32_data, shape)
};
```

### Result

```
>>> embedding_lookup: Got shape [151936, 896]  ✅ CORRECT
>>> embedding_lookup: SHAPE VALIDATION... PASSED
```

---

## Problem 2: First Layer Forward Hang 🔍 ROOT CAUSE IDENTIFIED

### Symptom

After transpose fix, the model loads successfully, embeddings are fetched, but **first layer hangs**:

```
>>> decode_step: Layer 1/24 starting...
>>> load_tensor_to_gpu: 'blk.0.attn_q.weight' ... loaded
>>> load_tensor_to_gpu: 'blk.0.attn_output.weight' ... loaded
>>> load_tensor_to_gpu: 'blk.0.ffn_gate.weight' ... loaded
>>> load_tensor_to_gpu: 'blk.0.ffn_up.weight' ... loaded
>>> load_tensor_to_gpu: 'blk.0.ffn_down.weight' ... loaded
>>> load_tensor_to_gpu: 'blk.0.attn_norm.weight' ... loaded
>>> load_tensor_to_gpu: 'blk.0.ffn_norm.weight' ... loaded
>>> [HANG - never see "Layer 1/24 complete"]
```

---

## ROOT CAUSE: QKV Weight Shape Corruption 🎯

### The Bug

**Diagnostic logging revealed**: QKV weight shape is corrupted somewhere between GGUF loading and matmul execution.

| Stage | Shape | Status |
|-------|-------|--------|
| GGUF file (tensor metadata) | `[2688, 896]` | ✅ Correct |
| `load_tensor_to_gpu()` log | `[2688, 896]` | ✅ Correct |
| `LazyTensor` stored shape | `[2688, 896]` | ✅ Correct |
| `matmul()` receives | `[896, 896]` | ❌ **CORRUPTED** |

### Evidence from Logs

```
>>> load_tensor_to_gpu: Tensor 'blk.0.attn_qkv.weight' shape=[2688, 896] ... loaded
>>>       matmul: input_shape=[1, 896], weight_shape=[896, 896], expecting output_dim=3*hidden=2688
>>> extract_qkv: Splitting into Q,K,V each of size 896
>>> copy_from_device_slice: Copying K tensor: offset=896, size=896
>>> [HANG in hipMemcpy - trying to read past buffer]
```

### Why This Causes Hang

1. **Matmul with wrong shape**: `[896, 896]` instead of `[2688, 896]` produces output with **896 elements** instead of **2688**
2. **QKV extraction expects 2688**: Code tries to extract Q (0-896), K (896-1792), V (1792-2688)
3. **Buffer overrun**: K extraction tries to read from offset 896, but buffer only has 896 elements total
4. **hipMemcpy blocks**: `copy_from_device_slice` calls `hipMemcpy` which blocks indefinitely waiting for GPU data that doesn't exist

### Expected vs Actual Behavior

**Expected** (shape = [2688, 896]):
```
Input: [1, 896] @ Weight: [2688, 896]
Output: [1, 2688] → split into Q:[1,896], K:[1,896], V:[1,896]
```

**Actual** (shape = [896, 896]):
```
Input: [1, 896] @ Weight: [896, 896]
Output: [1, 896] → CANNOT split into 3 tensors
K extraction: offset=896 reads past buffer → HANG
```

### Where Shape Gets Corrupted

**Not yet confirmed** - needs further investigation. Likely locations:
- `DeviceTensor::from_host_vec()` - shape parameter handling
- `load_tensor_to_gpu()` - GPU tensor creation
- Lazy tensor cache lookup - shape retrieval
- Tensor conversion between GPU and CPU representations

---

## Diagnostic Logging Added

### Files Modified

| File | Change |
|------|--------|
| `src/model/execution_plan.rs:746-853` | `forward_layer()` - timing logs for all 6 steps |
| `src/model/execution_plan.rs:878-1000` | `self_attention()` - QKV projection, RoPE, SDPA logs |
| `src/model/execution_plan.rs:1027-1103` | `matmul()` - synchronization and shape logs |
| `src/model/execution_plan.rs:1102-1159` | `extract_qkv_tensors()` - Q/K/V extraction logs |
| `src/model/execution_plan.rs:1149-1259` | `scaled_dot_product_attention()` - attention kernel logs |

### Log Pattern Used

```rust
eprintln!(">>> <function>: <message>");
let timer = std::time::Instant::now();
// ... operation ...
eprintln!(">>> <function>: ... ({:?} elapsed)", timer.elapsed());
```

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/loader/gguf.rs` | 901-921 | Embedding transpose for `token_embd.weight`, `lm_head.weight`, `output.weight` |
| `src/backend/hip_backend.rs` | 2510-2571 | Layer timing logs in `decode_step()` |
| `src/model/execution_plan.rs` | 664-723 | Enhanced `embedding_lookup` logging |

---

## Test Output

### Before Transpose Fix
```
>>> embedding_lookup: Got shape [896, 151936]
>>> embedding_lookup: SHAPE VALIDATION FAILED
>>> Timeout
```

### After Transpose Fix
```
>>> load_tensor_to_gpu: Transposing embedding weights from [896, 151936] to [151936, 896]
>>> embedding_lookup: Got shape [151936, 896]  ✅
>>> embedding_lookup: SHAPE VALIDATION... PASSED  ✅
>>> decode_step: Layer 1/24 starting...
>>> [HANG in forward_layer computation]
```

---

## Architecture Notes

### GGUF vs Conventional Format

| Component | Conventional | GGUF | Action |
|-----------|--------------|------|--------|
| Embeddings | `[vocab, hidden]` | `[hidden, vocab]` | ✅ Transpose applied |
| Linear weights | `[out, in]` | `[out, in]` | ❓ Need to verify |
| Attention Q/K/V | `[3 * heads * dim, hidden]` | ❓ Need to verify |

### llama.cpp Behavior

llama.cpp handles this by:
1. Storing weights transposed on disk (saves dequantization time)
2. Handling transposed access patterns in computation
3. Using strided memory access for embeddings

---

## Remaining Work

### Phase 25.1: Find Shape Corruption Source ⚠️ NEXT

1. **Trace QKV weight shape through pipeline**
   - Read `DeviceTensor::from_host_vec()` implementation
   - Read `load_tensor_to_gpu()` implementation
   - Check how LazyTensor shape is passed to matmul
   - Add shape logging at each transition point

2. **Hypothesis: Shape truncation bug**
   - Is only the first dimension being used?
   - Is there a `.0` index that should be `.shape()`?
   - Is row/col being confused with full shape?

3. **Verify fix location**
   - Once found, fix the shape handling
   - Add assertion to catch this earlier
   - Test with full model

### Not In Scope

- Other layer weights (MLP, output projection) - same issue likely affects them too
- Performance optimization - fix correctness first
- Alternative implementations - fix the bug, don't work around it

---

## Status Summary

| Problem | Status | Notes |
|---------|--------|-------|
| Embedding transpose | ✅ FIXED | Weights now correctly [vocab, hidden] |
| forward_layer hang | 🔍 CAUSE FOUND | QKV shape corruption: [2688,896] → [896,896] |
| Shape corruption fix | ⚠️ TODO | Need to find where shape gets corrupted |

---

## References

- **Phase 24**: Vocab size inference for missing metadata
- **Phase 23**: hipDeviceSynchronize desktop hang fix
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
