# hipBLAS Stream Synchronization Guide

## The Problem

ROCmForge uses HIP streams for asynchronous GPU operations. A critical synchronization issue arises when mixing hipBLAS operations with custom HIP kernels:

- **hipBLAS uses the default stream** unless `set_stream()` is called
- **Custom HIP kernels** use the backend's custom stream
- **Mixing streams** causes race conditions, data corruption, and GPU hangs

### Symptoms

- GPU hangs during `copy_to_host()` or `hipDeviceSynchronize()`
- Intermittent test failures
- Incomplete data in host buffers
- "Hanging" inference that never completes

### Root Cause

When you call `hipDeviceSynchronize()`, it waits for operations on the **custom stream**. However, if hipBLAS operations were queued on the **default stream** (because `set_stream()` wasn't called), those operations may still be pending. The subsequent `hipMemcpy` then reads incomplete data, causing a hang or corruption.

## The Pattern

The correct pattern for hipBLAS operations:

```rust
use crate::backend::{HipBackend, HipBlasHandle, HipBuffer};

pub fn safe_matmul_example(
    backend: &HipBackend,
    a: &HipBuffer,
    b: &HipBuffer,
    m: i32,
    n: i32,
    k: i32,
) -> Result<HipBuffer, HipError> {
    // Step 1: Create hipBLAS handle
    let handle = HipBlasHandle::new()
        .map_err(|e| HipError::GenericError(format!("Failed to create handle: {}", e)))?;

    // Step 2: CRITICAL - Associate handle with backend's stream
    handle.set_stream(backend.stream().as_ptr())
        .map_err(|e| HipError::GenericError(format!("Failed to set stream: {}", e)))?;

    // Step 3: Perform hipBLAS operations
    let result = HipBuffer::new((m * n) as usize * std::mem::size_of::<f32>())?;
    sgemm(
        &handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        m, n, k,
        1.0,
        a.as_ptr() as *const f32,
        m,
        b.as_ptr() as *const f32,
        k,
        0.0,
        result.as_ptr() as *mut f32,
        m,
    )?;

    // Step 4: Copy results using stream-aware copy
    let mut host_data = vec![0.0f32; (m * n) as usize];
    result.copy_to_host_with_stream(host_data.as_mut_slice(), backend.stream().as_ptr())?;

    // Step 5: CRITICAL - Synchronize before using host data
    backend.synchronize()?;

    // Now host_data is valid
    Ok(result)
}
```

## Common Pitfalls

### 1. Forgetting set_stream()

**WRONG:**
```rust
let handle = HipBlasHandle::new()?;
sgemm(&handle, ...)?;  // Uses default stream!
```

**CORRECT:**
```rust
let handle = HipBlasHandle::new()?;
handle.set_stream(backend.stream().as_ptr())?;
sgemm(&handle, ...)?;  // Uses backend's stream
```

### 2. Using copy_from_device_safe without sync

**WRONG:**
```rust
backend.copy_from_device_safe(&gpu_buffer, &mut host_data)?;
// host_data is NOT valid yet!
process(&host_data);  // BUG: reads incomplete data
```

**CORRECT:**
```rust
backend.copy_from_device_safe(&gpu_buffer, &mut host_data)?;
backend.synchronize()?;  // Wait for copy to complete
process(&host_data);  // Now data is valid
```

### 3. Using deprecated copy_to_host()

**WRONG:**
```rust
// Uses global hipDeviceSynchronize() - may cause issues
let data = buffer.copy_to_host_vec()?;
```

**CORRECT:**
```rust
// Stream-aware copy with explicit sync
let mut data = vec![0.0f32; buffer.size() / 4];
buffer.copy_to_host_with_stream(&mut data, backend.stream().as_ptr())?;
backend.synchronize()?;
```

### 4. Assuming sync happens automatically

**WRONG:**
```rust
sgemm(&handle, ...)?;
let mut data = vec![0.0f32; size];
buffer.copy_to_host_with_stream(&mut data, backend.stream().as_ptr())?;
// BUG: No sync - data incomplete
```

**CORRECT:**
```rust
sgemm(&handle, ...)?;
let mut data = vec![0.0f32; size];
buffer.copy_to_host_with_stream(&mut data, backend.stream().as_ptr())?;
backend.synchronize()?;  // CRITICAL
```

## API Reference

### HipBackend Copy Methods

#### `copy_from_device_safe<T>()` - Async, NO sync

```rust
pub fn copy_from_device_safe<T>(
    &self,
    gpu_buffer: &HipBuffer,
    host_data: &mut [T],
) -> HipResult<()>
```

- **Behavior**: Asynchronous copy using stream-aware API
- **Synchronization**: NONE - you MUST call `synchronize()` before using `host_data`
- **Use when**: You want to batch multiple operations before syncing
- **WARNING**: Name is misleading - it's "safe" because it's stream-aware, not because it syncs

**Example:**
```rust
let mut data1 = vec![0.0f32; size1];
let mut data2 = vec![0.0f32; size2];

// Queue both copies
backend.copy_from_device_safe(&buf1, &mut data1)?;
backend.copy_from_device_safe(&buf2, &mut data2)?;

// Single sync for both operations
backend.synchronize()?;

// Now both data1 and data2 are valid
process(&data1);
process(&data2);
```

#### `copy_from_device<T>()` - Async WITH sync

```rust
pub fn copy_from_device<T>(
    &self,
    buffer: &HipBuffer,
    data: &mut [T],
) -> HipResult<()>
```

- **Behavior**: Asynchronous copy followed by synchronization
- **Synchronization**: YES - waits for copy to complete before returning
- **Use when**: You need the data immediately after the call
- **Performance**: Slightly slower if doing multiple copies (one sync per copy)

**Example:**
```rust
let mut data = vec![0.0f32; size];
backend.copy_from_device(&buffer, &mut data)?;
// data is valid here - sync already happened
process(&data);
```

#### `copy_to_host_with_stream()` - Async, caller must sync

```rust
pub fn copy_to_host_with_stream<T>(
    &self,
    host_data: &mut [T],
    stream_ptr: *mut hipStream_t,
) -> HipResult<()>
```

- **Behavior**: Low-level stream-aware copy
- **Synchronization**: NONE - caller must synchronize
- **Use when**: You need fine-grained control over synchronization
- **Note**: This is the underlying method used by `copy_from_device_safe`

**Example:**
```rust
let mut data = vec![0.0f32; size];
buffer.copy_to_host_with_stream(&mut data, backend.stream().as_ptr())?;
backend.synchronize()?;
```

### HipBlasHandle Methods

#### `new()` - Create handle (uses default stream)

```rust
pub fn new() -> Result<Self, HipBlasError>
```

- **Behavior**: Creates a new hipBLAS handle
- **Stream**: Uses HIP default stream initially
- **CRITICAL**: Must call `set_stream()` before use

#### `set_stream()` - Associate with custom stream

```rust
pub fn set_stream(&mut self, stream: *mut hipStream_t) -> Result<(), HipBlasError>
```

- **Behavior**: Associates the handle with a specific HIP stream
- **Required**: MUST be called before any hipBLAS operations
- **Pattern**: `handle.set_stream(backend.stream().as_ptr())?;`

## Reference Implementations

### 1. src/kernels/matmul/fp16.rs:27-34

Correct pattern:
```rust
let handle = HipBlasHandle::new()
    .map_err(|e| HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e)))?;

handle
    .set_stream(backend.stream().as_ptr())
    .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;
```

### 2. src/ops/attention_gpu.rs:58-76

Excellent documentation:
```rust
let blas_handle = HipBlasHandle::new().map_err(|e| {
    HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e))
})?;

// CRITICAL: Associate hipBLAS handle with the backend's HIP stream
//
// Without this, hipBLAS uses the default stream while our custom HIP kernels
// (softmax, causal_mask) use the backend's custom stream. This causes
// synchronization issues and hangs when copy_to_host() calls hipDeviceSynchronize().
//
// hipDeviceSynchronize() waits for operations on the custom stream, but hipBLAS
// operations on the default stream are still pending, causing the D2H copy to
// read incomplete data and hang.
//
// See: https://github.com/ROCm/hip/issues/3370
// See: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html
blas_handle
    .set_stream(backend.stream().as_ptr())
    .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;
```

### 3. src/kernels/element/scale.rs:45-50

Another example:
```rust
let handle = HipBlasHandle::new()
    .map_err(|e| HipError::GenericError(format!("Failed to create hipBLAS handle: {}", e)))?;

handle.set_stream(backend.stream().as_ptr())
    .map_err(|e| HipError::GenericError(format!("Failed to set hipBLAS stream: {}", e)))?;
```

## Testing Guidance

### Testing Concurrent Operations

To verify that concurrent matmul operations don't race:

```rust
#[cfg(test)]
#[serial]  // Critical - prevents concurrent GPU test conflicts
mod concurrent_matmul_tests {
    use super::*;
    use crate::backend::hip_blas::HipBlasHandle;
    use crate::tensor::matmul::matmul_f32;

    #[test]
    #[serial]
    fn test_concurrent_matmul_handles() {
        let backend = HipBackend::new().unwrap();

        // Create multiple handles (simulating concurrent use)
        let handle1 = HipBlasHandle::new().unwrap();
        let handle2 = HipBlasHandle::new().unwrap();

        // Set streams - CRITICAL
        handle1.set_stream(backend.stream().as_ptr()).unwrap();
        handle2.set_stream(backend.stream().as_ptr()).unwrap();

        // Create test data
        let a1 = HipBuffer::new(4 * 4 * 4).unwrap();
        let b1 = HipBuffer::new(4 * 4 * 4).unwrap();

        let a2 = HipBuffer::new(4 * 4 * 4).unwrap();
        let b2 = HipBuffer::new(4 * 4 * 4).unwrap();

        // Run concurrent matmuls
        let result1 = matmul_f32(&backend, &handle1, &a1, &b1, 4, 4, 4);
        let result2 = matmul_f32(&backend, &handle2, &a2, &b2, 4, 4, 4);

        // Both should succeed without hanging
        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // Sync and verify results
        backend.synchronize().unwrap();
    }
}
```

### Signs of Synchronization Bugs

If you see these symptoms, check stream synchronization:

1. **Tests hang indefinitely** → Likely stream mismatch
2. **Intermittent failures** → Race condition between streams
3. **NaN or corrupted data** → Reading from incomplete buffers
4. **GPU resets** → Severe synchronization violation

### Debugging Tips

1. **Add logging before/after sync points:**
   ```rust
   eprintln!(">>> About to call sgemm");
   sgemm(&handle, ...)?;
   eprintln!(">>> sgemm complete");

   eprintln!(">>> About to synchronize");
   backend.synchronize()?;
   eprintln!(">>> Synchronize complete");
   ```

2. **Use `rocprof` to trace streams:**
   ```bash
   rocprof --hip-trace --rocm-trace ./your_binary
   ```

3. **Check for multiple stream handles:**
   ```bash
   grep -rn "HipBlasHandle::new" src/
   # Verify each has set_stream() nearby
   ```

## Migration Checklist

When adding new hipBLAS code:

- [ ] Created `HipBlasHandle` with `new()`
- [ ] Called `set_stream(backend.stream().as_ptr())` immediately
- [ ] Used stream-aware copy methods
- [ ] Called `synchronize()` before using host data
- [ ] Added test with `#[serial]` attribute
- [ ] Ran tests with `--test-threads=1`
- [ ] Added documentation referencing this guide

## References

- [HIP Asynchronous Operations](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html)
- [hipBLAS Documentation](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/)
- [ROCm Issue #3370](https://github.com/ROCm/hip/issues/3370) - Stream synchronization context

---

**Last Updated:** 2025-01-21
**Phase:** 35 - Stream Synchronization
