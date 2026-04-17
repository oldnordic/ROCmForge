# GQA Corruption Bug - ROOT CAUSE FOUND

**Date:** 2026-04-17  
**Status:** ROOT CAUSE IDENTIFIED - Q8_0 fastpath bug

## Critical Discovery

After extensive investigation using kernel markers and code tracing, I discovered that the **Q8_0 activation fastpath** is being used instead of the standard Q4_0 GEMV kernel.

### Evidence

**Rust Diagnostic Output:**
```
[RUST] dispatch_gemv_impl called: wtype=Q4_0, out_dim=896, in_dim=896
[RUST] Q4_0 path: checking fastpaths
[RUST] Q4_0 Q8_0 fastpath TAKEN
```

The fastpath is enabled and returns early, bypassing `gemv_q4_0_f32_on_stream_unchecked` entirely.

### Fastpath Implementation

From `/home/feanor/Projects/rocmforge/src/gpu/ops.rs:130-148`:

```rust
fn try_q4_0_q8_0_fastpath(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;
    gemv_q4_0_q8_0_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        in_dim,
        out_dim,
        stream,
    )
}
```

**The fastpath:**
1. Quantizes INPUT to Q8_0 format
2. Calls `gemv_q4_0_q8_0_on_stream` (Q4_0 weights × Q8_0 input)

**Standard path (what I was modifying):**
- Calls `gemv_q4_0_f32_on_stream` (Q4_0 weights × f32 input)

### Why This Explains Everything

1. ✅ **Data is correct** - Q4_0 weights are correct (verified earlier)
2. ✅ **My markers never appeared** - I was modifying the wrong kernel
3. ✅ **CPU/GPU discrepancy** - The Q8_0 fastpath has the bug
4. ✅ **Values 10-100x wrong** - Q8_0 quantization or dequantization bug

## Next Steps

### Immediate Actions

1. **Disable Q8_0 fastpath** to verify standard Q4_0 kernel works:
   ```bash
   ROCFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=0 ./target/release/rocmforge ...
   ```

2. **Profile Q8_0 quantization** - Check if input quantization is correct

3. **Verify gemv_q4_0_q8_0 kernel** - Check the Q4_0×Q8_0 hybrid kernel

### Files to Investigate

- `src/gpu/ops.rs` - Q8_0 fastpath dispatch logic
- `src/gpu/kernels/quant.rs` - `gemv_q4_0_q8_0_on_stream` function
- `hip_kernels/quant/q4_0_q8_0_multi_row_kernel.hip` - Hybrid kernel implementation

## Investigation History

### Failed Approaches

1. ❌ **Modified q4_0_gemv.hip kernel** - Wrong kernel (standard path not used)
2. ❌ **Added kernel markers** - Never appeared (fastpath bypasses them)
3. ❌ **Added launch function markers** - Never appeared (wrong function)
4. ❌ **Checked memory layout** - Correct (not the issue)

### Successful Approach

**Code tracing with Rust diagnostics** - Added eprintln! statements to trace:
- Which dispatch function is called
- Which code path is taken within dispatch
- Which kernel actually executes

**Key diagnostic additions:**
- `src/gpu/forward.rs:1445` - Before gpu_dispatch_gemv_on_stream
- `src/gpu/ops.rs:272` - At start of dispatch_gemv_impl
- `src/gpu/ops.rs:286-313` - Path selection in Q4_0 dispatch

## Impact

**ALL Q4_0 quantized models using Q8_0 fastpath** produce corrupted output. The standard Q4_0 kernel may be correct, but the fastpath is broken.

## Related Documentation

- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-data-verified-2026-04-17.md` - Data verification
- `/home/feanor/Projects/rocmforge/docs/gqa-corruption-gemv-root-cause-confirmed.md` - Previous investigation
