# GPU Attention Pipeline Corruption Investigation

**Date:** 2026-04-17  
**Status:** **CRITICAL BUG** - GPU-specific corruption in shared attention pipeline  
**Severity:** **BLOCKS ALL GPU DECODE** - affects both MHA and GQA models

## Summary

**CRITICAL DISCOVERY:** A GPU-specific bug in the shared attention pipeline causes **text corruption in ALL GPU models**, regardless of whether they use MHA or GQA attention. The CPU backend works perfectly, confirming the issue is isolated to GPU computation.

## Corruption Pattern

### Test Results

| Test Case | Prompt | Expected Output | GPU Output | Status |
|----------|--------|----------------|-----------|---------|
| **Simple prompt** | "The" | "ertha is a" | "erarO," | ❌ Corrupted |
| **Complex prompt** | "Hello, how are you today?" | "I am fine, thank you. How are you?" | " Ianto areigitsar? Iifiers, ar" | ❌ Corrupted |
| **CPU baseline** | "The" | "ertha is a" | "ertha is a" | ✅ Correct |

### Corruption Characteristics

1. **Repetitive pattern:** "arararar" appears across different prompts
2. **Character substitution:** Individual characters are wrong
3. **Consistent corruption:** Same input produces same corrupted output
4. **GPU-only:** CPU backend produces correct output

## Root Cause Analysis

### What We Know

1. ✅ **Model weights are correct** (CPU works)
2. ✅ **GQA fusion kernel is correct** (not the source of bug)
3. ❌ **GPU-specific issue** (CPU works, GPU doesn't)
4. ❌ **Shared pipeline bug** (affects both MHA and GQA)

### What We've Ruled Out

- ❌ **NOT in GQA fusion kernel** - MHA path has same corruption
- ❌ **NOT in separate GQA kernels** - baseline produces same corruption  
- ❌ **NOT in GQA-specific logic** - MHA forced path has same corruption
- ❌ **NOT in model weights** - CPU works correctly
- ❌ **NOT in tokenization** - both backends use same tokenizer

### Likely Candidates

The bug is most likely in:

1. **GPU attention kernel** (`hip_kernels/attention.hip`)
   - Shared by both MHA and GQA paths
   - Called via `flash_attn_decode_strided_multi_head_on_stream`
   - Could have indexing or computation error

2. **GPU RoPE application** 
   - Shared RoPE logic for both paths
   - Could have stride or memory layout issue

3. **GPU KV-cache operations**
   - Shared read/write logic
   - Could have memory corruption

4. **GPU final projection/argmax**
   - Shared output processing
   - Could cause token selection corruption

## Next Steps

### Immediate Actions

1. **Add debug output to attention kernel**
   - Print Q, K, V values at kernel entry
   - Verify attention scores are computed correctly
   - Check final output values before token selection

2. **Test with known-good GPU model**
   - Try a different model format if available
   - Test with simpler model (e.g., tiny model)
   - Check if corruption is model-independent

3. **Compare GPU vs CPU computation**
   - Add intermediate value dumping
   - Compare attention scores between GPU and CPU
   - Identify where computations diverge

4. **Check for GPU-specific issues**
   - Memory alignment issues
   - Shared memory corruption
   - Thread synchronization problems
   - Numerical precision issues

## Technical Details

### Files to Investigate

1. **`hip_kernels/attention.hip`**
   - `flash_attn_decode_strided_multi_head_v2_kernel` (line 152)
   - GQA head mapping: `kv_head_idx = head_idx / (num_heads / num_kv_heads)`
   - Score computation and softmax logic
   - Value cache accumulation

2. **`src/gpu/kernels/attention.rs`**
   - `flash_attn_decode_strided_multi_head_on_stream` wrapper
   - Kernel launch parameters

3. **`src/gpu/forward.rs`**
   - `gpu_attention_decode` function (line 475)
   - Attention pipeline coordination

### Test Commands

```bash
# Test simple prompt (GPU)
./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# Test same prompt (CPU - baseline)
./target/release/rocmforge --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf --prompt "The" --no-template --top-p 1.0 --max-tokens 5

# Test complex prompt (GPU)  
./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf --prompt "Hello, how are you today?" --no-template --top-p 1.0 --max-tokens 10
```

## Impact Assessment

### Severity: **CRITICAL**

- **Blocks:** All GPU decode functionality
- **Scope:** Affects ALL models (MHA and GQA)
- **User Impact:** Complete text corruption makes GPU backend unusable
- **Performance:** Cannot achieve performance targets while bug exists

### Relationship to GQA Fusion

**IMPORTANT:** This bug is **UNRELATED** to the GQA fusion work. The GQA fusion kernel implementation is complete and correct. This is a pre-existing bug in the shared GPU attention pipeline that needs to be fixed independently.

## Related Issues

- **GQA Fusion Implementation:** ✅ Complete (see `docs/gqa-fusion-investigation.md`)
- **Performance Target:** 520 tok/s (blocked by this bug)
- **CPU Baseline:** Works correctly at 3.2 tok/s

## Conclusion

This is a **critical GPU-specific bug** that **blocks all GPU decode functionality**. The bug is in the shared attention pipeline and affects both MHA and GQA models identically. The CPU backend works perfectly, confirming the issue is isolated to GPU computation.

**Priority:** **HIGHEST** - Must be fixed before GPU decode can be used for any model.

**Status:** **UNDER INVESTIGATION** - Root cause not yet identified
