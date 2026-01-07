# Phase 8: Model Support - Status Summary

> **Date**: 2026-01-06
> **Status**: 🔄 **IN PROGRESS** (NOT YET COMPLETE)
> **Goal**: Add support for Q4_1/Q5_0/Q5_1 quantization, GPU MQA pipeline, improve test infrastructure

---

## Executive Summary

Phase 8 is **currently in progress** but **NOT yet complete**. Based on code inspection, the following tasks remain:

- ❌ **Task 8.1**: Q4_1/Q5_0/Q5_1 Dequantization (NOT IMPLEMENTED)
- ❌ **Task 8.2**: GPU MQA Pipeline (NOT IMPLEMENTED)
- ⚠️ **Task 8.3**: MLP API Exposure (INCOMPLETE TEST)
- ❌ **Task 8.4**: Dimension Checking (NOT IMPLEMENTED)

**Estimated Time to Complete**: 5-7 days

---

## Task 8.1: Q4_1/Q5_0/Q5_1 Dequantization

### Status: ❌ NOT IMPLEMENTED

**Location**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:1129-1131`

**Current Code**:
```rust
GgufTensorType::Q4_1 | GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => {
    // TODO: Implement dequantization for these types
    return Err(anyhow!("Unsupported tensor type for GPU upload: {:?}", tensor.tensor_type));
}
```

**Impact**:
- Many GGUF models use Q4_1, Q5_0, or Q5_1 quantization
- These models cannot be loaded on GPU
- Falls back to CPU or fails entirely

**Required Implementation**:

#### Q4_1 Dequantization
- **Block structure**: 32-element block, 4-bit values + min value
- **Block layout**: [scale (f16, 2 bytes), min (f16, 2 bytes), 16x packed 4-bit values (16 bytes)]
- **Total**: 20 bytes per 32 values

```rust
fn dequantize_q4_1(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let mut output = vec![0.0f32; num_elements];

    for (block_idx, out_block) in output.chunks_mut(BLOCK_SIZE).enumerate() {
        let block_offset = block_idx * (2 + 16);  // 2 bytes scale+min, 16 bytes packed

        // Read scale and min (f16)
        let scale = f16::from_bits(u16::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
        ])).to_f32();

        let min = f16::from_bits(u16::from_le_bytes([
            data[block_offset + 2],
            data[block_offset + 3],
        ])).to_f32();

        // Unpack 4-bit values and dequantize
        for (i, out) in out_block.iter_mut().enumerate() {
            let packed = data[block_offset + 4 + i / 2];
            let q4 = if i % 2 == 0 {
                packed & 0x0F
            } else {
                (packed >> 4) & 0x0F
            };

            *out = scale * (q4 as f32) + min;
        }
    }

    Ok(output)
}
```

#### Q5_0 Dequantization
- **Block structure**: 32-element block, 5-bit values + scale
- **Block layout**: [scale (f16, 2 bytes), 16x packed 5-bit values (20 bytes)]
- **Total**: 22 bytes per 32 values

#### Q5_1 Dequantization
- **Block structure**: 32-element block, 5-bit values + min + scale
- **Block layout**: [scale (f16, 2 bytes), min (f16, 2 bytes), 16x packed 5-bit values (20 bytes)]
- **Total**: 24 bytes per 32 values

**Estimated Effort**: 2-3 days

**Files to Modify**:
- `/src/loader/gguf.rs:1129-1131` - Main dequantization logic
- `/tests/q_dequant_tests.rs` (NEW) - Accuracy tests

**Reference Specifications**:
- [GGUF Quantization Formats](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- llama.cpp implementation: `ggml.c` dequantization functions

---

## Task 8.2: GPU MQA Pipeline

### Status: ❌ NOT IMPLEMENTED

**Location**: `/home/feanor/Projects/ROCmForge/src/attention/multi_query.rs:180`

**Current Code**:
```rust
// TODO: Implement full GPU pipeline for MQA
// Current: CPU-only implementation
```

**Impact**:
- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) models fall back to CPU
- Performance penalty for models with fewer KV heads than query heads
- Affects modern LLMs that use MQA/GQA for efficiency

**Required Implementation**:

#### 1. Multi-Query QKV Projection
```rust
pub fn compute_mqa_qkv_gpu(
    backend: &HipBackend,
    input: &HipBuffer,  // [batch, seq_len, hidden_size]
    q_weight: &HipBuffer,  // [num_heads * head_dim, hidden_size]
    kv_weight: &HipBuffer,  // [num_kv_heads * head_dim, hidden_size]
    output_q: &mut HipBuffer,  // [batch, num_heads, seq_len, head_dim]
    output_k: &mut HipBuffer,  // [batch, num_kv_heads, seq_len, head_dim]
    output_v: &mut HipBuffer,  // [batch, num_kv_heads, seq_len, head_dim]
    // ... dimensions
) -> Result<(), String> {
    // Project Q for all query heads
    hipblas_sgemm(/* Q projection */)?;

    // Project K/V for key-value heads (fewer heads)
    hipblas_sgemm(/* K projection */)?;
    hipblas_sgemm(/* V projection */)?;

    Ok(())
}
```

#### 2. Grouped-Query Attention Computation
```rust
pub fn compute_gqa_attention_gpu(
    backend: &HipBackend,
    q: &HipBuffer,  // [batch, num_heads, seq_len, head_dim]
    k: &HipBuffer,  // [batch, num_kv_heads, seq_len, head_dim]
    v: &HipBuffer,  // [batch, num_kv_heads, seq_len, head_dim]
    output: &mut HipBuffer,  // [batch, num_heads, seq_len, head_dim]
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), String> {
    // Repeat K/V across query heads
    // For example, if num_heads=8 and num_kv_heads=2:
    //   Each KV head is shared by 4 query heads

    let queries_per_kv = num_heads / num_kv_heads;

    for kv_head_idx in 0..num_kv_heads {
        for q_idx in 0..queries_per_kv {
            let q_head_idx = kv_head_idx * queries_per_kv + q_idx;

            // Compute attention for this query head with shared KV head
            compute_attention_head_gpu(
                backend,
                &q[q_head_idx],  // Query head
                &k[kv_head_idx],  // Shared KV head
                &v[kv_head_idx],  // Shared KV head
                &mut output[q_head_idx],
                // ...
            )?;
        }
    }

    Ok(())
}
```

**Estimated Effort**: 3-4 days

**Files to Modify**:
- `/src/attention/multi_query.rs:180` - Main implementation
- `/src/ops/attention_gpu.rs` - GPU kernels
- `/tests/mqa_gpu_tests.rs` (NEW) - Tests

**Dependencies**:
- ✅ Phase 7 (GPU attention kernel) - COMPLETE

---

## Task 8.3: MLP API Exposure for Tests

### Status: ⚠️ INCOMPLETE TEST

**Location**: `/home/feanor/Projects/ROCmForge/src/mlp/gpu_path_regression_tests.rs:87`

**Current Code**:
```rust
#[test]
fn test_mlp_swiglu_forward_pass() {
    // Test setup but no actual call to mlp_swiglu
}

// TODO: Add actual mlp_swiglu call once the API is exposed
// let result = crate::mlp::mlp_swiglu(
//     &hidden_states,
//     &gate_weight,
//     &up_weight,
// ).unwrap();
```

**Impact**:
- Test does not verify actual MLP SwiGLU computation
- Cannot detect regressions in MLP GPU path
- Test coverage is incomplete

**Required Changes**:

#### 1. Expose MLP Function
**File**: `/src/mlp/mod.rs`

Change:
```rust
fn mlp_swiglu(...) -> Result<Tensor> {
    // Implementation
}
```

To:
```rust
pub(crate) fn mlp_swiglu(
    hidden_states: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
) -> Result<Tensor> {
    // Implementation unchanged
}
```

#### 2. Update Test
**File**: `/src/mlp/gpu_path_regression_tests.rs:87`

```rust
#[test]
fn test_mlp_swiglu_forward_pass() {
    let backend = HipBackend::new().expect("Failed to create HipBackend");

    // Setup test data
    let seq_len = 4;
    let intermediate_size = 8;
    let hidden_states = /* ... */;
    let gate_weight = /* ... */;
    let up_weight = /* ... */;

    // Call actual implementation
    let result = crate::mlp::mlp_swiglu(
        &hidden_states,
        &gate_weight,
        &up_weight,
    ).unwrap();

    // Assert correctness
    assert_eq!(result.shape(), hidden_states.shape());

    // Verify GPU-only path (no host roundtrip)
    // Check result is on device
}
```

**Estimated Effort**: 2-3 hours

**Files to Modify**:
- `/src/mlp/mod.rs` - Expose API
- `/src/mlp/gpu_path_regression_tests.rs:87` - Update test

---

## Task 8.4: Dimension Checking in MatMul Tests

### Status: ❌ NOT IMPLEMENTED

**Location**: `/home/feanor/Projects/ROCmForge/tests/hip_blas_matmul_tests.rs:190`

**Current Code**:
```rust
#[test]
fn test_hipblas_matmul() {
    // No validation of input/output dimensions
}
```

**Impact**:
- Tests do not verify matrix dimensions are correct
- Can miss dimension mismatch bugs
- Less robust test coverage

**Required Implementation**:

#### 1. Add Validation Helper
```rust
fn validate_matmul_dims(
    expected: (usize, usize, usize),  // (m, k, n)
    a_shape: &[usize],
    b_shape: &[usize],
    c_shape: &[usize],
) -> Result<(), String> {
    let (m, k, n) = expected;

    if a_shape != &[m, k] {
        return Err(format!(
            "A shape mismatch: expected [{}, {}], got {:?}",
            m, k, a_shape
        ));
    }

    if b_shape != &[k, n] {
        return Err(format!(
            "B shape mismatch: expected [{}, {}], got {:?}",
            k, n, b_shape
        ));
    }

    if c_shape != &[m, n] {
        return Err(format!(
            "C shape mismatch: expected [{}, {}], got {:?}",
            m, n, c_shape
        ));
    }

    Ok(())
}
```

#### 2. Update Tests
```rust
#[test]
fn test_hipblas_matmul() {
    // Setup
    let (m, k, n) = (128, 256, 512);
    let a = /* ... */;
    let b = /* ... */;
    let mut c = /* ... */;

    // Compute
    hipblas_sgemm(/* ... */).unwrap();

    // Validate dimensions
    validate_matmul_dims((m, k, n), a.shape(), b.shape(), c.shape()).unwrap();

    // Validate correctness
    // ...
}
```

#### 3. Add Negative Tests
```rust
#[test]
#[should_panic(expected = "A shape mismatch")]
fn test_matmul_invalid_a_dims() {
    // Test that invalid dimensions are caught
}

#[test]
#[should_panic(expected = "B shape mismatch")]
fn test_matmul_invalid_b_dims() {
    // Test that invalid dimensions are caught
}
```

**Estimated Effort**: 1 hour

**Files to Modify**:
- `/tests/hip_blas_matmul_tests.rs:190` - Add validation logic
- `/src/tensor/matmul.rs` - Add helper functions (optional)

---

## Next Steps

### To Complete Phase 8:

1. **Implement Q4_1 Dequantization** (Day 1-2)
   - Add `dequantize_q4_1()` function
   - Add accuracy tests
   - Verify against llama.cpp reference

2. **Implement Q5_0/Q5_1 Dequantization** (Day 2-3)
   - Add `dequantize_q5_0()` function
   - Add `dequantize_q5_1()` function
   - Add accuracy tests

3. **Implement GPU MQA Pipeline** (Day 4-6)
   - Implement `compute_mqa_qkv_gpu()`
   - Implement `compute_gqa_attention_gpu()`
   - Add MQA/GQA tests
   - Verify accuracy and performance

4. **Complete Test Infrastructure** (Day 7)
   - Expose MLP API
   - Update MLP test
   - Add dimension checking to matmul tests

### After Phase 8:

**Phase 9: Code Quality** (1 week)
- Fix 84 compiler warnings
- Remove dead code
- Add edge case tests
- Improve documentation

---

## Files to Modify

| File | Lines | Task |
|------|-------|------|
| `/src/loader/gguf.rs` | 1129-1131 | Task 8.1: Q4_1/Q5_0/Q5_1 dequantization |
| `/src/attention/multi_query.rs` | 180+ | Task 8.2: GPU MQA pipeline |
| `/src/mlp/mod.rs` | N/A | Task 8.3: Expose MLP API |
| `/src/mlp/gpu_path_regression_tests.rs` | 87 | Task 8.3: Update MLP test |
| `/tests/hip_blas_matmul_tests.rs` | 190+ | Task 8.4: Dimension checking |

**Total Estimated LOC**: 600-800 lines

---

## Known Issues

1. **Q4_1/Q5_0/Q5_1 models cannot load on GPU**
   - Falls back to CPU or fails
   - Affects many popular GGUF models

2. **MQA/GQA models use CPU fallback**
   - Performance penalty
   - Affects modern efficient architectures

3. **Incomplete test coverage**
   - MLP test does not call actual implementation
   - MatMul tests lack dimension validation

---

## References

### GGUF Quantization
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp dequantization implementation](https://github.com/ggerganov/llama.cpp/blob/master/ggml.c)

### Multi-Query Attention
- [GQA: Multi-Query Attention (Google Research)](https://arxiv.org/abs/2305.13245)
- [vLLM GQA Implementation](https://github.com/vllm-project/vllm)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-06
**Author**: Claude Code (Phase 8 Investigation)
