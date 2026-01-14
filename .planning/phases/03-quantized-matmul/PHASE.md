# Phase 3: Quantized MatMul Operations

## Goal

Add Q4_0 and Q8_0 matmul operations for efficient quantized model inference.

## Problem

Currently `src/ggml/op.rs` only has F32 `MatMul`:

```rust
pub enum Op {
    GetRows,
    MatMul,  // Only F32!
    Add,
    // Missing: Quantized matmul variants
}
```

Quantized models must dequantize to F32 before matmul, negating memory/compute savings.

## Solution

1. Add `MatMulQ4_0`, `MatMulQ8_0` to `Op` enum
2. Implement HIP dequantization kernels
3. Follow llama.cpp quantization patterns from `ggml/src/ggml-cpu/quants.h`

Q4_0 format: block_size=32, scale (f32) + quantized data (4-bit per element)
Q8_0 format: block_size=32, scale (f32) + quantized data (8-bit per element)

## Files to Modify

- `src/ggml/op.rs` - Add quantized op variants
- `src/ggml/hip_backend/mod.rs` - Implement kernels
- `src/ggml/hip_backend/kernels/` - Add dequantization kernels

## Reference

`/home/feanor/Projects/llama.cpp/ggml/src/ggml-cpu/quants.h`

## Success Criteria

- [ ] Q4_0 and Q8_0 matmul ops implemented
- [ ] Can run Q4_0 quantized models
- [ ] Memory usage reduced for quantized models
- [ ] Output matches reference dequantization
