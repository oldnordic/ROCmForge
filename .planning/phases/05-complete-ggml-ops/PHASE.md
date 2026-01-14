# Phase 5: Complete Missing ggml Ops

## Goal

Implement remaining ggml operations for full IR compatibility.

## Missing Ops

1. **Accumulate** - For KV cache writes without Copy + manual offset
2. **Tensor Pool/Allocator** - Efficient buffer reuse (llama.cpp's `ggml_allocr`)
3. **Graph Optimizer** - CSE, dead code elimination, layout optimization

## Solution

### Accumulate Op
Add `Accumulate` to `Op` enum for in-place tensor accumulation:

```rust
Accumulate { src: TensorId, dst: TensorId, offset: usize }
```

### Tensor Allocator
Create `src/ggml/allocator.rs`:
- Track allocated buffers in pool
- Reuse buffers for same-sized tensors
- Free all at graph completion

### Graph Optimizer
Create `src/ggml/optimizer.rs`:
- Common subexpression elimination
- Dead code elimination
- Layout optimization (RowMajor vs ColMajor)

## Files to Create

- `src/ggml/allocator.rs` - Tensor pool/allocator
- `src/ggml/optimizer.rs` - Graph optimization passes

## Files to Modify

- `src/ggml/op.rs` - Add `Accumulate` op
- `src/ggml/hip_backend/mod.rs` - Implement accumulate kernel
- `src/ggml/executor.rs` - Integrate allocator and optimizer

## Success Criteria

- [ ] Accumulate op implemented and tested
- [ ] Tensor allocator reduces allocations by 50%+
- [ ] Graph optimizer eliminates redundant ops
- [ ] All tests pass
