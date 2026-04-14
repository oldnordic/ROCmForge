# GPU Kernel Design Guidelines

## Graph Capture Compatibility

**Critical Rule:** All HIP kernels must be compatible with HIP graph capture/replay.

### Requirements for Graph Compatibility

1. **Linear Thread Processing**
   - Use fixed iteration counts (e.g., `for (int l = 0; l < 8; ++l)`)
   - Avoid `break` or `continue` based on data-dependent conditions
   - Thread ID to element mapping must be deterministic

2. **No Data-Dependent Branching**
   - No `if/else` branches that depend on computed values
   - Use lookup tables instead of conditional logic
   - Precompute all indices before loops

3. **Predictable Memory Access**
   - Direct array indexing: `arr[offset + i]` not `arr[computed_offset]`
   - No pointer arithmetic based on runtime values
   - Coalesced loads when possible

4. **Thread Configuration**
   - Use warp-sized thread blocks (32 threads)
   - Warp shuffle reduction for sums
   - Avoid dynamic shared memory allocation

### Examples

**Graph-Compatible Pattern (Q4_K):**
```cpp
for (int l = 0; l < 8; ++l) {
    const int i = tid * 8 + l;  // Fixed linear calculation
    sum += vec[offset + i];      // Direct memory access
}
```

**Graph-Incompatible Pattern (some quantization formats):**
```cpp
for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; ++l) {
        if (computed_value < 32) { ... }  // Data-dependent branch - FAIL
    }
}
```

### Known Incompatibilities

**Q6_K Quantization:**
- **Status:** Partially compatible with HIP graph capture (works for single-token prompts, crashes with multi-token prompts)
- **Root Cause:** Complex interleaved data layout (256 elements in non-linear pattern: T, T+32, T+64, T+96 per thread)
- **Kernel Rewrite (2026-04-14):** Eliminated nested loops and pointer arithmetic, now uses linear processing with interleaved element distribution
- **Remaining Issue:** Graph capture works for single-token prompts (~82 tok/s) but crashes with memory access fault on multi-token prompts
- **Current Solution:** Automatic detection disables graph capture for Q6_K models until multi-token issue is resolved
- **Performance:** ~95 tok/s without graph capture (still efficient)

**Implementation:**
```rust
// Automatic Q6_K detection in decode_graph_disabled()
fn decode_graph_disabled(gpu_weights: &GpuModelWeights) -> bool {
    // ... other checks ...
    || gpu_weights.uses_q6_k_quantization()  // Auto-disable for Q6_K
}
```

### Testing Graph Compatibility

Always test with:
```bash
# Test with graph (should work for non-Q6_K)
ROCMFORGE_DISABLE_DECODE_GRAPH=0 ./target/release/rocmforge --gpu --model <model>

# Test without graph (should work for all types)
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu --model <model>
```

Should not see HIP error 901 or graph capture failures.

### Performance Impact

Graph capture provides ~30-50% performance improvement for compatible kernels:
- **Q4_K:** ~15 tok/s with graph, ~10 tok/s without
- **Q8_0:** ~12 tok/s with graph, ~8 tok/s without
- **Q6_K:** ~82 tok/s with graph (single-token only), ~95 tok/s without graph (graph disabled due to multi-token crash)

### Adding New Quantization Types

Before implementing a new quantization type:

1. **Analyze data layout:** Can it be processed linearly like Q4_K?
2. **Test prototype:** Implement simple kernel first, test with graph capture
3. **Detect incompatibility:** If graph capture fails, add automatic detection
4. **Document behavior:** Note graph compatibility in this file

**Example for Q6_K:**
```rust
// In GpuModelWeights
pub fn uses_q6_k_quantization(&self) -> bool {
    // Check all weight types for Q6_K
    // Return true if any weight uses Q6_K
}
```

### Safety Verification

Before committing any GPU kernel:

1. ✅ Works with graph capture enabled (for compatible types)
2. ✅ Works with graph capture disabled (all types)
3. ✅ No GPU crashes (check dmesg for reset messages)
4. ✅ Tested with real model (not just unit tests)
5. ✅ Performance is acceptable (≥5 tok/s for decode)

### References

- Q4_K implementation: `hip_kernels/quant/q4_k_gemv.hip` (graph-compatible reference)
- Q8_0 implementation: `hip_kernels/quant/q8_0_gemv.hip` (graph-compatible reference)
- Q6_K implementation: `hip_kernels/quant/q6_k_gemv.hip` (graph-incompatible example)
- HIP graph docs: `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html`
