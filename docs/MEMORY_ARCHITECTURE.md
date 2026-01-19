# ROCmForge Memory Architecture

**Last Updated:** 2026-01-19
**Status:** Direct Allocation (no pooling)

## Overview

ROCmForge uses direct GPU memory allocation for all tensors. Each tensor receives its own
independent buffer via `hipMalloc()`, avoiding the ROCm 7.1 D2H sub-buffer limitation.

## Allocation Strategy

### Direct Allocation (Current Implementation)

All tensors are allocated using `DeviceTensor::from_host_vec()`:

```rust
// From src/loader/gguf.rs:861
let device_tensor = DeviceTensor::from_host_vec(backend, f32_data, shape)
    .map_err(|e| anyhow!("Failed to upload tensor '{}' to GPU: {}", name, e))?;
```

**Characteristics:**
- Each tensor gets its own `hipMalloc()` allocation
- No sub-buffer views are created
- D2H operations (if needed) are safe (offset=0)
- Memory overhead: Minimal, no pool management complexity

### Why This Works

The ROCm 7.1 D2H bug (`hipMemcpyDtoH` error code 1) only occurs when:
1. A sub-buffer view is created (offset > 0)
2. That sub-buffer is read back to host

Direct allocation avoids both conditions:
- No sub-buffer views exist
- All D2H operations use the base buffer (offset=0)

## D2H Operations

Tensors that require read-back:

| Tensor Type | Operation | D2H Safe? |
|-------------|-----------|-----------|
| token_embd.weight | Transpose during mapping | Yes (direct alloc) |
| lm_head.weight | Transpose during mapping | Yes (direct alloc) |
| attn_qkv.weight | Concatenate during mapping | Yes (direct alloc) |

These are transposed/concatenated on CPU via `transpose_2d_tensor()` and
`concatenate_qkv_tensors()` in `src/model/execution_plan/execution_plan_src.rs`.

## Allocation Code Paths

### Primary: Tensor Loading (GGUF)

**File:** `src/loader/gguf.rs:861`

```rust
pub fn load_tensor_to_gpu(&self, name: &str, backend: &HipBackend)
    -> Result<Arc<DeviceTensor>>
{
    // ... dequantize to f32_data ...

    // Upload to GPU
    let device_tensor = DeviceTensor::from_host_vec(backend, f32_data, shape)
        .map_err(|e| anyhow!("Failed to upload tensor '{}' to GPU: {}", name, e))?;

    Ok(Arc::new(device_tensor))
}
```

All tensors use this path - no conditional logic for pooling.

### D2H Operations

**File:** `src/backend/hip_backend/backend.rs:2227`

```rust
pub fn to_host_vec(&self) -> HipResult<Vec<f32>> {
    let mut host_data = vec![0.0f32; self.len()];
    unsafe {
        let ptr = host_data.as_mut_ptr() as *mut u8;
        let byte_size = self.len() * std::mem::size_of::<f32>();
        let byte_slice = std::slice::from_raw_parts_mut(ptr, byte_size);
        self.buffer.copy_to_host(byte_slice)?;  // Calls hipMemcpyDtoH
    }
    Ok(host_data)
}
```

Since all tensors are directly allocated, `buffer.offset == 0` always, making D2H safe.

## Unused Pooling Infrastructure

The `from_pool()` and `from_pool_with_backend()` methods exist in `DeviceTensor` but are
never called:

```rust
// src/backend/hip_backend/backend.rs:2368
pub fn from_pool(
    pool: &HipBuffer,
    offset: usize,
    host_data: Vec<f32>,
    shape: TensorShape,
) -> HipResult<Self> { ... }

pub fn from_pool_with_backend(
    pool: &HipBuffer,
    offset: usize,
    host_data: Vec<f32>,
    shape: TensorShape,
    backend: &HipBackend,
) -> HipResult<Self> { ... }
```

These methods create sub-buffer views that would fail D2H on ROCm 7.1. They remain in the
codebase but are intentionally unused.

## Future Considerations

Selective memory pooling (documented in `ROCM_D2H_ERROR_RESEARCH.md`) was designed but
not implemented. If memory optimization becomes a priority, consider:

1. **GPU-only pooling**: Pool only tensors that never require D2H
2. **GPU transpose**: Implement transpose on GPU to avoid D2H entirely
3. **Alternative architecture**: Use a framework with built-in memory management

## Related Documentation

- `ROCM_D2H_ERROR_RESEARCH.md` - Original D2H bug investigation and proposed solutions
- `src/backend/hip_backend/backend.rs` - `DeviceTensor` implementation
- `src/loader/gguf.rs` - Tensor loading implementation
- `src/model/execution_plan/execution_plan_src.rs` - D2H operations (transpose, concatenate)
