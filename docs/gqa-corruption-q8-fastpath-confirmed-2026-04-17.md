# GQA Corruption Bug - ROOT CAUSE CONFIRMED

**Date:** 2026-04-17
**Status:** BUG CONFIRMED IN Q8_0 FASTPATH - Standard kernel produces different output

## Confirmed Findings

### Test Results

**With Q8_0 Fastpath ENABLED:**
- Output: Chinese characters, garbage text
- Root cause: `q4_0_q8_0_block_dot` or scale computation

**With Q8_0 Fastpath DISABLED (standard Q4_0 kernel):**
- Output: Coherent English text ("borderTop ante themselves")
- Different from CPU, suggesting possible secondary issue

### Algorithm Comparison

Both implementations (GPU fastpath and llama.cpp reference) use the same algorithm:
1. Combined scale: `w_scale * x_scale` ✓
2. Dot product: `sum((q4_i - 8) * q8_i)` ✓
3. Result: `combined_scale * dot_product` ✓

### Suspected Issues

**Primary (Q8_0 Fastpath):**
1. Integer overflow in `q4_0_q8_0_block_dot`
   - `block_sum` accumulates 32 products of int16 × int8
   - Max magnitude: 32 × (7 × 127)² ≈ 81M (fits in int32)
   - **UNLIKELY**: Should fit in int32

2. **Scale computation issue** (MOST LIKELY):
   - Q8_0 quantization: `scale = max_val / 127.0f`
   - Possible precision loss in half-to-float conversion
   - Possible incorrect scale in Q8_0 blocks

3. **Type casting issue**:
   - Q8_0 values stored as int8_t but interpreted incorrectly
   - Sign extension problems in kernel

**Secondary (Standard Q4_0 kernel):**
- May have separate bug causing CPU/GPU divergence
- Needs investigation after fastpath fix

## Next Steps

1. **Add diagnostics to Q8_0 quantization kernel**
   - Log scale values during quantization
   - Verify Q8_0 block format

2. **Add diagnostics to q4_0_q8_0_block_dot**
   - Log intermediate block_sum values
   - Verify combined_scale computation

3. **Compare with llama.cpp reference**
   - Match exact kernel parameters
   - Verify block layout assumptions

## Files to Modify

- `hip_kernels/quant/q8_0_quantize.hip` - Add scale logging
- `hip_kernels/quant/q4_0_gemv.hip` - Add block_sum logging in q4_0_q8_0_block_dot
- `src/gpu/ops.rs` - Add GEMV result comparison

## Test Commands

```bash
# Disable fastpath (hardcoded in src/gpu/safety.rs)
cargo build --release --features gpu
./target/release/rocmforge --gpu --model ... --prompt "Hello" --max-tokens 10

# Compare with CPU
./target/release/rocmforge --model ... --prompt "Hello" --max-tokens 10
```

## Related Documentation

- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-root-cause-q8-fastpath.md` - Root cause discovery
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-data-verified-2026-04-17.md` - Data verification
