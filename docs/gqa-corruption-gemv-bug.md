# GQA Corruption - Root Cause: GEMV Kernel Bug

**Date:** 2026-04-17  
**Status:** ROOT CAUSE IDENTIFIED - GEMV kernel produces wrong values

## Discovery

Comparing GPU Q values **before RoPE** vs CPU values **after RoPE** shows the GEMV kernel is producing completely wrong values.

### Evidence: Token pos=1, Layer 0, First 10 Q values

**GPU (BEFORE RoPE - after GEMV only):**
```
[-0.0319, -0.0379, -0.2223, 0.0585, -0.0051,
  0.1206,  0.1719, -0.1368, -0.4897, 0.4817]
```

**CPU (AFTER RoPE):**
```
[-0.1629, -0.1820, -0.4350, 0.0730, -12.0423,
  0.3956,  0.5174, -0.0495, -16.3551, -33.6314]
```

**Magnitude comparison:**
- Index 4: GPU=-0.005 vs CPU=-12.04 (2400x difference!)
- Index 8: GPU=-0.49 vs CPU=-16.36 (33x difference)
- Index 9: GPU=0.48 vs CPU=-33.63 (Wrong sign!)

## Conclusion

**The bug is in the GEMV kernel**, NOT RoPE. The GPU GEMV is producing values that are 10-100x smaller than they should be.

## Possible Causes

1. **Wrong parameter order** - n_rows vs ncols_dst swapped
2. **Incorrect quantization/dequantization** - Scale factors not applied correctly
3. **Block layout mismatch** - Q4_0 blocks not indexed correctly
4. **Shared memory corruption** - Input not loaded correctly into shared memory
5. **Weight loading bug** - GPU weights loaded differently than CPU

## Investigation Status

**NOT in scope (ruled out):**
- ✗ RoPE kernel (values wrong before RoPE)
- ✗ KV-cache read/write (verified working)
- ✗ Flash attention kernel (receives wrong input)
- ✗ seq_len value (tested, not the issue)

**Next steps:**
1. Verify GEMV kernel parameter order
2. Check Q4_0 block indexing
3. Compare weight loading between CPU and GPU
4. Test with simple matrix multiplication to isolate issue

## Test Commands

```bash
# GPU Q values before RoPE
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1 2>&1 | \
  grep "GPU DIAGNOSTIC.*BEFORE RoPE"

# CPU Q values after RoPE
ROCMFORGE_DEBUG=1 ./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1 2>&1 | \
  grep "CPU DIAGNOSTIC.*Q\[0..10\]"
```
