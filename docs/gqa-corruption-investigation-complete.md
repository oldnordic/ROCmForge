# GQA Corruption Investigation - Complete Summary

**Date:** 2026-04-17
**Status:** ROOT CAUSE FOUND - GEMV kernel bug in GPU Q4_0 dequantization

## Investigation Timeline

1. **Initial symptom:** GPU produces "er管辖 lash lash lash" instead of "ertha is"
2. **Investigated:** KV-cache, seq_len, flash attention - all working correctly
3. **Added diagnostics:** Compared CPU vs GPU intermediate values
4. **FOUND:** Q values differ by 10-100x between CPU and GPU
5. **NARROWED:** GPU Q values wrong even BEFORE RoPE → GEMV kernel bug

## Root Cause

**GPU GEMV kernel for Q4_0 quantization produces incorrect values.**

### Evidence

| Stage | CPU Value | GPU Value | Ratio |
|-------|-----------|-----------|------|
| GEMV output (before RoPE) | ~12-33 magnitude | ~0.005-0.5 magnitude | **10-100x smaller** |
| After RoPE | -12.04, -16.35, -33.63 | -0.003, -0.477, 0.484 | **Still wrong** |

The GPU values are consistently 10-100x smaller than CPU values, indicating the GEMV kernel is not dequantizing correctly.

## What Works

✅ KV-cache read/write operations  
✅ Flash attention computation  
✅ RoPE application (applied correctly to wrong input)  
✅ seq_len handling  
✅ Kernel call sequence and parameters  

## What's Broken

❌ **Q4_0 GEMV kernel** - Produces values 10-100x too small  
❌ **Result:** Wrong Q values → wrong attention → wrong output  

## Next Steps to Fix

1. **Verify Q4_0 block structure** - Ensure scale factors are loaded correctly
2. **Check block indexing** - Verify weight blocks are accessed in correct order
3. **Test dequantization** - Compare dequantized weights on CPU vs GPU
4. **Inspect kernel assembly** - Check for compiler optimizations or bugs

## Files to Investigate

- `/home/feanor/Projects/rocmforge/hip_kernels/quant/q4_0_gemv.hip` - GEMV kernel implementation
- `/home/feanor/Projects/rocmforge/src/gpu/weights.rs` - GPU weight loading
- `/home/feanor/Projects/rocmforge/src/gpu/ops.rs` - GEMV dispatch logic

## Test Commands

```bash
# Show GPU bug
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1

# CPU baseline
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1
```

## Impact

Affects **ALL Q4_0 quantized models** on GPU when using GQA. This is a critical bug preventing correct inference.

## Related Documentation

- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-phase1-evidence.md` - Initial investigation
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-seqlen-fix.md` - seq_len investigation (not the cause)
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-root-cause.md` - Root cause discovery
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-bug.md` - GEMV bug details
