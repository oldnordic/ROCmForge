# HIP/hipBLAS Reference Pattern Comparison

**Date**: 2026-01-20
**Source**: AMD ROCm Examples at `/home/feanor/Projects/rocm-examples`
**Purpose**: Compare AMD reference patterns against ROCmForge implementation
**Status**: INVESTIGATION - No code changes

---

## Table of Contents

1. [Memory Allocation Patterns](#1-memory-allocation-patterns)
2. [Stream Management Patterns](#2-stream-management-patterns)
3. [Event Synchronization Patterns](#3-event-synchronization-patterns)
4. [hipBLAS Usage Patterns](#4-hipblas-usage-patterns)
5. [Error Handling Patterns](#5-error-handling-patterns)
6. [Issues Found](#6-issues-found)

---

## 1. Memory Allocation Patterns

### AMD Reference Pattern

**Source**: `HIP-Basic/matrix_multiplication/main.hip:177-182`

```cpp
// AMD's approach: Direct allocation with error checking
float* d_A{};
float* d_B{};
float* d_C{};
HIP_CHECK(hipMalloc(&d_A, a_bytes));
HIP_CHECK(hipMalloc(&d_B, b_bytes));
HIP_CHECK(hipMalloc(&d_C, c_bytes));

// ... use the buffers ...

HIP_CHECK(hipFree(d_A));
HIP_CHECK(hipFree(d_B));
HIP_CHECK(hipFree(d_C));
```

**Key Characteristics**:
- Simple, direct `hipMalloc` calls
- Each allocation checked with `HIP_CHECK` macro
- Clean `hipFree` in reverse order
- Fixed, known number of allocations

### Our Implementation

**Source**: `src/backend/hip_backend/backend.rs:534-560`

```rust
pub fn new(size: usize) -> HipResult<Self> {
    let mut ptr: *mut c_void = ptr::null_mut();

    // Use hipMalloc to allocate device memory
    tracing::trace!("HipBuffer::new: Calling hipMalloc for {} bytes", size);
    let result = unsafe { hipMalloc(&mut ptr, size) };
    // ... error checking ...
}
```

**AND THE PROBLEM** (in `src/loader/gguf.rs:1197-1200`):

```rust
// allocate GPU buffer
let total_elements: usize = shape.iter().product();
let buffer = HipBuffer::new(total_elements * std::mem::size_of::<f32>())
    .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
```

**Issue**: Loop calling `HipBuffer::new()` hundreds of times for model weights.

### Comparison

| Aspect | AMD Reference | Our Implementation | Status |
|--------|---------------|-------------------|--------|
| Allocation call | Direct `hipMalloc` | Wrapped in `HipBuffer::new()` | ✅ OK |
| Error checking | `HIP_CHECK` macro | Result type | ✅ OK |
| Allocation pattern | Fixed, small count | Loop over hundreds of tensors | ❌ PROBLEM |
| Free pattern | Explicit `hipFree` | `Drop` trait | ✅ OK |

---

## 2. Stream Management Patterns

### AMD Reference Pattern

**Source**: `HIP-Basic/streams/main.hip:80-146`

```cpp
// 1. Create streams
std::vector<hipStream_t> streams(num_streams);
for(int i = 0; i < num_streams; i++)
{
    HIP_CHECK(hipStreamCreate(&streams[i]));
}

// 2. Allocate and use with streams
HIP_CHECK(hipMemcpyAsync(d_in[i], h_in, size_in_bytes, hipMemcpyHostToDevice, streams[i]));
kernel<<<..., ..., ..., streams[i]>>>(...);
HIP_CHECK(hipMemcpyAsync(h_out[i], d_out[i], size_in_bytes, hipMemcpyDeviceToHost, streams[i]));

// 3. Synchronize ALL streams
HIP_CHECK(hipDeviceSynchronize());

// 4. Destroy streams
for(int i = 0; i < num_streams; i++)
{
    HIP_CHECK(hipStreamDestroy(streams[i]))
}
```

**Key Characteristics**:
- Explicit stream creation
- All async operations specify a stream
- Global `hipDeviceSynchronize()` to wait for all
- Explicit stream destruction

### Our Implementation

**Source**: `src/backend/hip_backend/backend.rs:260-320`

```rust
impl HipStream {
    pub fn new() -> HipResult<Self> {
        let mut stream: *mut c_void = ptr::null_mut();
        let result = unsafe { hipStreamCreate(&mut stream) };
        // ... error checking ...
        Ok(HipStream { stream: Arc::new(HipStreamInner { stream }) })
    }

    pub fn synchronize(&self) -> HipResult<()> {
        let result = unsafe { hipStreamSynchronize(self.stream) };
        // ...
    }
}

impl Drop for HipStreamInner {
    fn drop(&mut self) {
        unsafe { hipStreamDestroy(self.stream); }
    }
}
```

### Comparison

| Aspect | AMD Reference | Our Implementation | Status |
|--------|---------------|-------------------|--------|
| Stream creation | `hipStreamCreate` with `HIP_CHECK` | `HipStream::new()` with Result | ✅ OK |
| Stream destruction | Explicit `hipStreamDestroy` | `Drop` trait | ✅ OK (better actually) |
| Synchronization | `hipDeviceSynchronize()` for global wait | `hipStreamSynchronize()` for per-stream | ⚠️ Different - need to verify correctness |
| Multiple streams | Vector of streams | `Arc<HipStreamInner>` sharing | ✅ OK |

---

## 3. Event Synchronization Patterns

### AMD Reference Pattern

**Source**: `HIP-Basic/events/main.hip:89-147`

```cpp
// 1. Create events
hipEvent_t start, stop;
HIP_CHECK(hipEventCreate(&start));
HIP_CHECK(hipEventCreate(&stop));

// 2. Record start
HIP_CHECK(hipEventRecord(start, hipStreamDefault));

// 3. Do work
HIP_CHECK(hipMemcpy(d_matrix, h_matrix.data(), size_bytes, hipMemcpyHostToDevice));

// 4. Record stop and synchronize
HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
HIP_CHECK(hipEventSynchronize(stop));

// 5. Get elapsed time
float elapsed_ms{};
HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

// 6. Destroy events
HIP_CHECK(hipEventDestroy(start));
HIP_CHECK(hipEventDestroy(stop));
```

**Key Characteristics**:
- Explicit event creation/destruction
- Record before and after operations
- `hipEventSynchronize` waits for event to complete
- `hipEventElapsedTime` measures duration

### Our Implementation

**FFI declarations exist** (`src/backend/hip_backend/backend.rs:46-51`):
```rust
fn hipEventCreate(event: *mut *mut c_void) -> i32;
fn hipEventDestroy(event: *mut c_void) -> i32;
fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
fn hipEventSynchronize(event: *mut c_void) -> i32;
fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
```

**But usage is unclear** - no clean wrapper like `HipEvent` type.

### Comparison

| Aspect | AMD Reference | Our Implementation | Status |
|--------|---------------|-------------------|--------|
| Event creation | `hipEventCreate` with `HIP_CHECK` | FFI declared only | ❌ No wrapper |
| Event usage | Record → Synchronize → ElapsedTime | Not used in main path | ❌ Missing |
| Event destruction | Explicit `hipEventDestroy` | N/A | ❌ Missing |

---

## 4. hipBLAS Usage Patterns

### AMD Reference Pattern

**Source**: `Libraries/hipBLAS/gemm_strided_batched/main.cpp:185-217`

```cpp
// 1. Create hipBLAS handle
hipblasHandle_t handle;
HIPBLAS_CHECK(hipblasCreate(&handle));

// 2. Set pointer mode
HIPBLAS_CHECK(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

// 3. Perform GEMM
HIPBLAS_CHECK(hipblasSgemmStridedBatched(handle,
                                         trans_a, trans_b, m, n, k,
                                         &h_alpha,  // <- pointer to host memory
                                         d_a, lda, stride_a,
                                         d_b, ldb, stride_b,
                                         &h_beta,
                                         d_c, ldc, stride_c,
                                         batch_count));

// 4. Get results
HIP_CHECK(hipMemcpy(h_c.data(), d_c, sizeof(float) * size_c, hipMemcpyDeviceToHost));

// 5. Destroy handle
HIPBLAS_CHECK(hipblasDestroy(handle));
```

**Key Characteristics**:
- Create handle once at startup
- Set pointer mode to HOST for scalars
- Pass pointers to host memory for alpha/beta
- Destroy handle when done
- **No stream management shown** (uses default stream)

### Our Implementation

**Source**: `src/backend/hip_blas.rs:82-101`

```rust
impl HipBlasHandle {
    pub fn new() -> HipBlasResult<Self> {
        let mut handle: *mut c_void = ptr::null_mut();
        let result = unsafe { hipblasCreate(&mut handle) };
        // ... error checking ...
        Ok(HipBlasHandle { raw: handle })
    }
}
```

**Source**: `src/tensor/matmul.rs:175-219`

```rust
pub fn sgemm(
    handle: &HipBlasHandle,
    // ... parameters ...
) -> HipBlasResult<()> {
    let result = unsafe {
        hipblasSgemm(
            handle.as_ptr(),
            transa, transb, m, n, k,
            &alpha,  // <- pointer to stack variable
            A, lda,
            B, ldb,
            &beta,
            C, ldc,
        )
    };
    // ...
}
```

### Comparison

| Aspect | AMD Reference | Our Implementation | Status |
|--------|---------------|-------------------|--------|
| Handle creation | `hipblasCreate` with `HIPBLAS_CHECK` | `HipBlasHandle::new()` with Result | ✅ OK |
| Handle destruction | Explicit `hipblasDestroy` | `Drop` trait | ✅ OK |
| Pointer mode | `hipblasSetPointerMode(HIPBLAS_POINTER_MODE_HOST)` | **NOT SET** | ⚠️ May cause issues |
| Scalar passing | `&h_alpha` (pointer to host) | `&alpha` (pointer to stack) | ✅ OK |
| Stream management | Not shown (default stream) | `set_stream()` exists but unclear if used | ⚠️ Verify |
| GEMM function | `hipblasSgemmStridedBatched` | `hipblasSgemm` (standard) | ✅ OK |

---

## 5. Error Handling Patterns

### AMD Reference Pattern

```cpp
// From example_utils.hpp
#define HIP_CHECK(cmd)                                                                                 \
    do {                                                                                               \
        hipError_t error = cmd;                                                                       \
        if (error != hipSuccess) {                                                                     \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":"        \
                      << __LINE__ << std::endl;                                                      \
            return error_exit_code;                                                                   \
        }                                                                                              \
    } while (0)
```

**Key Characteristics**:
- Macro-based error checking
- Immediate termination on error
- Prints file/line number
- Returns error code

### Our Implementation

```rust
// Result type based error handling
let result = unsafe { hipMalloc(&mut ptr, size) };
if result != HIP_SUCCESS {
    return Err(HipError::MemoryAllocationFailed(format!(
        "hipMalloc failed with code {} for {} bytes",
        result, size
    )));
}
```

### Comparison

| Aspect | AMD Reference | Our Implementation | Status |
|--------|---------------|-------------------|--------|
| Error detection | Check against `hipSuccess` | Check against `HIP_SUCCESS` | ✅ OK |
| Error reporting | `hipGetErrorString` + file/line | `format!` with code | ⚠️ No error string |
| Error handling | `return error_exit_code` | `Result::Err` | ✅ Better (Rust idiomatic) |

---

## 6. Issues Found

### Critical Issues

#### Issue #1: Multiple Small Allocations (GPU Hang Risk)

**Location**: `src/loader/gguf.rs:1197-1200`

```rust
// Inside load_to_gpu_async(), in a loop:
for (name, data) in dequantized.iter() {
    let buffer = HipBuffer::new(total_elements * std::mem::size_of::<f32>())
        .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
    // ...
}
```

**Problem**:
- Calls `hipMalloc` once per tensor (~200-400 times)
- Triggers "multiple small allocations" pathology on RDNA3
- Can cause GPU hangs per `docs/ROCM_HANG_INVESTIGATION_2026-01-07.md`

**AMD Reference**:
```cpp
// AMD allocates fixed number of buffers explicitly
HIP_CHECK(hipMalloc(&d_A, a_bytes));
HIP_CHECK(hipMalloc(&d_B, b_bytes));
HIP_CHECK(hipMalloc(&d_C, c_bytes));
// No loops over hundreds of allocations
```

**Fix Needed**: Memory pooling - allocate one large buffer, subdivide internally.

---

#### Issue #2: hipBLAS Pointer Mode Not Set

**Location**: `src/backend/hip_blas.rs`

**Problem**: `hipblasSetPointerMode` is never called.

**AMD Reference**:
```cpp
HIPBLAS_CHECK(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
```

**Impact**: May cause incorrect scalar handling or performance issues.

---

#### Issue #3: Stream Usage Not Clearly Documented

**Location**: Throughout codebase

**Problem**:
- `hipblasSetStream` exists but unclear if/when it's called
- Mixing `hipMemcpy` (default stream) with `hipMemcpyAsync` (custom stream) can cause issues

**AMD Reference**:
```cpp
// AMD uses consistent stream pattern
HIP_CHECK(hipMemcpyAsync(d_in[i], h_in, size_in_bytes, hipMemcpyHostToDevice, streams[i]));
kernel<<<..., ..., streams[i]>>>(...);
HIP_CHECK(hipMemcpyAsync(h_out[i], d_out[i], size_in_bytes, hipMemcpyDeviceToHost, streams[i]));
HIP_CHECK(hipDeviceSynchronize()); // Wait for ALL streams
```

---

### Medium Priority Issues

#### Issue #4: No HipEvent Wrapper

**Problem**: FFI declarations exist but no safe wrapper type.

**AMD Reference**: Full event lifecycle with create/record/sync/destroy.

---

#### Issue #5: Error Messages Don't Use hipGetErrorString

**Problem**: Our errors show numeric codes, not descriptive strings.

**AMD Reference**:
```cpp
std::cerr << "HIP error: " << hipGetErrorString(error) << ...
```

---

### Low Priority Issues

#### Issue #6: Matmul Transpose Logic Complexity

**Location**: `src/tensor/matmul.rs:115-179`

**Problem**: Complex transpose/conversion logic for row-major ↔ column-major.

**AMD Reference**: Uses hipBLAS directly with proper parameters, no manual transpose.

---

## 7. Code Snippets for Reference

### AMD: Complete GEMM Example

```cpp
// 1. Allocate memory
float* d_A{}, *d_B{}, *d_C{};
HIP_CHECK(hipMalloc(&d_A, a_bytes));
HIP_CHECK(hipMalloc(&d_B, b_bytes));
HIP_CHECK(hipMalloc(&d_C, c_bytes));

// 2. Copy to device
HIP_CHECK(hipMemcpy(d_A, h_A.data(), a_bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_B, h_B.data(), b_bytes, hipMemcpyHostToDevice));

// 3. Create hipBLAS handle
hipblasHandle_t handle;
HIPBLAS_CHECK(hipblasCreate(&handle));
HIPBLAS_CHECK(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

// 4. Perform GEMM
HIPBLAS_CHECK(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                             m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

// 5. Copy result back
HIP_CHECK(hipMemcpy(h_C.data(), d_C, c_bytes, hipMemcpyDeviceToHost));

// 6. Cleanup
HIPBLAS_CHECK(hipblasDestroy(handle));
HIP_CHECK(hipFree(d_A));
HIP_CHECK(hipFree(d_B));
HIP_CHECK(hipFree(d_C));
```

---

## 8. Recommended Actions

### Immediate (To Prevent GPU Hangs)

1. **Implement memory pooling** for model weight loading
   - Single large allocation
   - Subdivide internally
   - Eliminate hundreds of `hipMalloc` calls

2. **Set hipBLAS pointer mode** in `HipBlasHandle::new()`
   ```rust
   pub fn new() -> HipBlasResult<Self> {
       let mut handle: *mut c_void = ptr::null_mut();
       let result = unsafe { hipblasCreate(&mut handle) };
       if result != HIPBLAS_SUCCESS { /* error */ }

       // ADD THIS:
       let set_stream_result = unsafe { hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST) };
       if set_stream_result != HIPBLAS_SUCCESS { /* warning but continue? */ }

       Ok(HipBlasHandle { raw: handle })
   }
   ```

### Short Term (Correctness)

3. **Add HipEvent wrapper** for timing
4. **Add hipGetErrorString** to error messages
5. **Document stream usage** throughout codebase

### Long Term (Architecture)

6. **Consider memory pool architecture** per ROCM_HANG_INVESTIGATION recommendations
7. **Audit all GPU allocation sites** for safety

---

**End of Report** - Investigation only, no code changes made.
