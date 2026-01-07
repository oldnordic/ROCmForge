# Phase 10 Memory Pooling - Bug Hunt Report

**Date**: 2026-01-07
**Agent**: debugger
**Scope**: Phase 10 memory pooling code
**Files Analyzed**:
- `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
- `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
- `/home/feanor/Projects/ROCmForge/src/model/kv_cache.rs`

---

## Executive Summary

Conducted systematic bug hunt for dangerous patterns in Phase 10 memory pooling code. Found **3 HIGH severity**, **6 MEDIUM severity**, and **4 LOW severity** issues across unsafe pointer arithmetic, potential race conditions, and memory management edge cases.

**Critical Finding**: No CRITICAL bugs (memory corruption, use-after-free, double-free) detected in the core memory pooling implementation. The code is generally well-structured with proper Arc usage for memory ownership.

---

## HIGH Severity Issues

### BUG-1: Unsafe Pointer Arithmetic Without Overflow Check

**File**: `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
**Lines**: 268, 409-410, 961

**Problematic Code**:
```rust
// Line 268 - HipBuffer::ptr()
if self.inner.offset > 0 {
    unsafe { self.inner.ptr.add(self.inner.offset) }  // ⚠️ No overflow check
} else {
    self.inner.ptr
}

// Lines 409-410 - copy_from_buffer_region()
let dst_ptr = unsafe { (self.ptr() as *mut u8).add(dst_offset_bytes) } as *mut c_void;
let src_ptr = unsafe { (src.ptr() as *mut u8).add(src_offset_bytes) } as *const c_void;

// Line 961 - add_row_bias()
unsafe {
    row_ptr = row_ptr.add(stride);  // ⚠️ No overflow check
}
```

**Why It's Dangerous**:
- `ptr::add()` causes **undefined behavior** if the offset causes pointer arithmetic overflow
- For very large GPU allocations (>4TB on 64-bit), offset could overflow
- While unlikely in practice, this violates Rust's safety guarantees

**Suggested Fix**:
```rust
// Add checked arithmetic before unsafe block
let new_offset = self.inner.offset.checked_add(byte_offset)
    .ok_or_else(|| HipError::MemoryCopyFailed(
        "Pointer arithmetic overflow".to_string()
    ))?;

// Ensure offset is within buffer bounds
if new_offset > self.inner.size {
    return Err(HipError::MemoryCopyFailed(...));
}

unsafe { self.inner.ptr.add(new_offset) }
```

**Risk Assessment**:
- **Likelihood**: Very Low (requires >4TB allocations)
- **Impact**: Undefined behavior / process crash
- **Detection**: Hard to reproduce in testing

---

### BUG-2: Race Condition in Global Backend Singleton

**File**: `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
**Lines**: 544-574

**Problematic Code**:
```rust
pub fn new() -> HipResult<Arc<Self>> {
    // Double-checked locking pattern (INCORRECTLY IMPLEMENTED)
    if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {  // ⚠️ Check 1
        return Ok(GLOBAL_BACKEND.lock().unwrap()
            .as_ref()
            .map(Arc::clone)
            .expect("Global backend initialized but not set"));
    }

    let mut guard = GLOBAL_BACKEND.lock().unwrap();  // ⚠️ Lock acquired
    if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {  // ⚠️ Check 2 - RACE WINDOW
        return Ok(guard.as_ref().map(Arc::clone)
            .expect("Global backend initialized but not set"));
    }
    // ... initialization code ...
    *guard = Some(backend.clone());
    GLOBAL_INIT_CALLED.store(true, Ordering::Release);  // ⚠️ Flag set AFTER lock release
}
```

**Why It's Dangerous**:
- Classic **double-checked locking anti-pattern** - incorrect implementation
- Between releasing the mutex and setting `GLOBAL_INIT_CALLED`, other threads can see inconsistent state
- `GLOBAL_INIT_CALLED` should be set **inside** the lock
- Could cause multiple backend initializations or use of uninitialized backend

**Suggested Fix**:
```rust
pub fn new() -> HipResult<Arc<Self>> {
    // Fast path: read-only check without lock
    if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {
        return Ok(GLOBAL_BACKEND.lock().unwrap()
            .as_ref()
            .map(Arc::clone)
            .expect("Global backend initialized but not set"));
    }

    // Slow path: full initialization under lock
    let mut guard = GLOBAL_BACKEND.lock().unwrap();

    // Double-check after acquiring lock
    if GLOBAL_INIT_CALLED.load(Ordering::Acquire) {
        return Ok(guard.as_ref().map(Arc::clone)
            .expect("Global backend initialized but not set"));
    }

    // Initialize backend
    Self::initialize_hip()?;
    let device = Self::detect_amd_gpu()?;
    let stream = Arc::new(HipStream::new()?);
    let backend = Arc::new(HipBackend { device, stream });

    // CRITICAL: Set flag BEFORE releasing lock
    *guard = Some(backend.clone());
    GLOBAL_INIT_CALLED.store(true, Ordering::Release);

    Ok(backend)
}
```

**Risk Assessment**:
- **Likelihood**: Medium (multi-threaded applications)
- **Impact**: Race condition / multiple initialization / memory leak
- **Detection**: Difficult - timing-dependent

---

### BUG-3: Memory Leak on Error Path in load_to_gpu()

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Lines**: 590-764

**Problematic Code**:
```rust
pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
    // Allocate memory pools (Lines 615-638)
    let mut pools: Vec<HipBuffer> = Vec::new();
    let mut current_pool_bytes = 0usize;

    for (_, tensor_bytes) in &tensor_list {
        // ... allocation logic ...
        if current_pool_bytes + aligned_tensor_bytes > actual_pool_size {
            pools.push(backend.allocate_buffer(actual_pool_size)  // ⚠️ If this fails...
                .map_err(|e| anyhow!("Failed to allocate memory pool: {}", e))?);
            // ... previous pools are leaked if later allocation fails
        }
    }

    // Upload tensors (Lines 652-753)
    for (name, tensor) in &self.tensors {
        // ... tensor upload logic ...
        let device_tensor = DeviceTensor::from_pool(...)  // ⚠️ If this fails...
            .map_err(|e| anyhow!("Failed to create tensor '{}' from pool #{}: {}", name, pool_idx, e))?;
        // ... pools are leaked (no Drop called)
    }
}
```

**Why It's Dangerous**:
- If `allocate_buffer()` or `from_pool()` fails mid-loop, previously allocated pools are **leaked**
- GPU memory is **not freed** until process exit
- In long-running server applications, repeated failures could exhaust GPU memory

**Suggested Fix**:
```rust
// Use RAII wrapper or ensure cleanup on error path
pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
    struct PoolGuard {
        pools: Vec<HipBuffer>,
        backend: HipBackend,
    }

    impl Drop for PoolGuard {
        fn drop(&mut self) {
            // Explicit cleanup to ensure GPU memory is freed
            self.pools.clear();
        }
    }

    let mut pool_guard = PoolGuard {
        pools: Vec::new(),
        backend: backend.clone(),
    };

    // ... allocation logic using pool_guard.pools ...

    // On success, transfer ownership
    let pools = pool_guard.pools;

    // ... rest of function ...
}
```

**Risk Assessment**:
- **Likelihood**: Low (requires allocation failure during model load)
- **Impact**: GPU memory leak / OOM on repeated failures
- **Detection**: Easy - monitor GPU memory usage

---

## MEDIUM Severity Issues

### BUG-4: Integer Overflow in offset Calculation

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Line**: 744

**Problematic Code**:
```rust
// Advance offset for next tensor (ALIGN TO 4KB BOUNDARY)
const ALIGNMENT: usize = 4096;
offset = (offset + tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);  // ⚠️ Potential overflow
```

**Why It's Dangerous**:
- `offset + tensor_bytes + ALIGNMENT - 1` can overflow for large tensors
- Overflow wraps around in release builds (undefined behavior in Rust)
- Could cause sub-buffer to overlap with previous buffer

**Suggested Fix**:
```rust
offset = offset.checked_add(tensor_bytes)
    .and_then(|o| o.checked_add(ALIGNMENT - 1))
    .ok_or_else(|| anyhow!("Offset calculation overflow"))?;

offset = offset & !(ALIGNMENT - 1);

// Verify offset is within pool bounds
if offset >= pools[pool_idx].size() {
    return Err(anyhow!("Offset {} exceeds pool size {}", offset, pools[pool_idx].size()));
}
```

**Risk Assessment**:
- **Likelihood**: Low (requires very large tensors)
- **Impact**: Memory corruption / buffer overlap
- **Detection**: Medium - bounds checking

---

### BUG-5: Unchecked Array Index in parse_tensor_info()

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Lines**: 700-704, 732-737

**Problematic Code**:
```rust
// Check if we need to move to next pool
if offset + tensor_bytes > pools[pool_idx].size() {  // ⚠️ pool_idx could be out of bounds
    pool_idx += 1;
    offset = 0;
}

// ... later ...

let device_tensor = DeviceTensor::from_pool(
    &pools[pool_idx],  // ⚠️ No bounds check on pool_idx
    offset,
    f32_data,
    tensor.shape.clone(),
).map_err(|e| anyhow!("Failed to create tensor '{}' from pool #{}: {}", name, pool_idx, e))?;
```

**Why It's Dangerous**:
- If tensor sizes are underestimated during pool allocation, `pool_idx` can exceed `pools.len()`
- Results in **out-of-bounds panic** (not memory corruption, but crash)
- Fails gracefully (panic), but still a bug

**Suggested Fix**:
```rust
// Check if we need to move to next pool
if pool_idx >= pools.len() {
    return Err(anyhow!("Pool allocation insufficient for tensor '{}' (needed pool {})", name, pool_idx));
}

if offset + tensor_bytes > pools[pool_idx].size() {
    pool_idx += 1;

    if pool_idx >= pools.len() {
        return Err(anyhow!("Pool allocation insufficient for tensor '{}' (needed pool {})", name, pool_idx));
    }

    offset = 0;
}
```

**Risk Assessment**:
- **Likelihood**: Low (allocation logic is conservative)
- **Impact**: Process crash / panic
- **Detection**: Easy - panic stack trace

---

### BUG-6: Missing Error Handling in HIP FFI Calls

**File**: `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
**Lines**: 342-343, 1131

**Problematic Code**:
```rust
// copy_to_host() - Lines 342-343
// For sub-allocated buffers, synchronize before reading
if self.inner.offset > 0 {
    unsafe { hipDeviceSynchronize() };  // ⚠️ Result ignored
}

// DeviceTensor::empty() - Line 1131
let result = unsafe { hipMemset(buffer.as_ptr(), 0, total_bytes) };  // ⚠️ Error checked later

if result != HIP_SUCCESS {
    // ... error handling ...
}
```

**Why It's Dangerous**:
- Line 342-343: `hipDeviceSynchronize()` return value is **ignored** - async operation failure is undetected
- Could read **stale or garbage data** from GPU memory
- Line 1131 is **correctly handled** (false positive in analysis)

**Suggested Fix**:
```rust
// For sub-allocated buffers, synchronize before reading
if self.inner.offset > 0 {
    let result = unsafe { hipDeviceSynchronize() };
    if result != HIP_SUCCESS {
        return Err(HipError::MemoryCopyFailed(format!(
            "Device synchronization failed before copy: {}",
            result
        )));
    }
}
```

**Risk Assessment**:
- **Likelihood**: Medium (GPU errors are rare but possible)
- **Impact**: Data corruption / silent failures
- **Detection**: Hard - requires GPU error injection

---

### BUG-7: Arc Cloning in Hot Path May Cause Performance Degradation

**File**: `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
**Lines**: 525-532, 1999

**Problematic Code**:
```rust
// HipBackend::clone() - Lines 525-532
impl Clone for HipBackend {
    fn clone(&self) -> Self {
        HipBackend {
            device: self.device.clone(),
            stream: Arc::clone(&self.stream),  // ⚠️ Atomic refcount operation in hot path
        }
    }
}

// load_model() - Line 1999
Ok(ModelRuntime {
    backend: self.backend.clone(),  // ⚠️ Arc clone on every model load
    // ...
})
```

**Why It's Dangerous**:
- `Arc::clone()` performs atomic RMW operation on reference count
- In multi-threaded context, causes **cache line bouncing** between CPU cores
- Could degrade performance in high-throughput inference scenarios

**Suggested Fix**:
```rust
// For ModelRuntime, consider using raw pointer or non-atomic reference
// (Only if lifetime can be statically verified)

pub struct ModelRuntime<'a> {
    backend: &'a HipBackend,  // Borrowed instead of owned Arc
    // ...
}

// Or use unsafe option with clear documentation (NOT RECOMMENDED without strong justification)
```

**Risk Assessment**:
- **Likelihood**: Low (performance issue, not correctness)
- **Impact**: Performance degradation in multi-threaded scenarios
- **Detection**: Medium - profiling required

---

### BUG-8: Potential Deadlock in Recursive Backend Creation

**File**: `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`
**Lines**: 544-574, 1089

**Problematic Code**:
```rust
// HipBackend::new() - Lines 544-574
pub fn new() -> HipResult<Arc<Self>> {
    // ...
    let mut guard = GLOBAL_BACKEND.lock().unwrap();  // ⚠️ Mutex held
    // ... initialization code ...
    // ... calls backend.clone() elsewhere ...
}

// DeviceTensor::hip_backend() - Line 1089
pub fn hip_backend() -> HipResult<Arc<HipBackend>> {
    HipBackend::new()  // ⚠️ Recursive call could deadlock if called from within initialization
}
```

**Why It's Dangerous**:
- If `HipBackend::new()` is called recursively during initialization (e.g., in a callback), it will **deadlock**
- The mutex is not reentrant
- Current code doesn't trigger this, but it's a **latent bug**

**Suggested Fix**:
```rust
// Use reentrant mutex or document that HipBackend::new() cannot be called recursively
// Option 1: Use parking_lot::ReentrantMutex (more performant anyway)
// Option 2: Add runtime check for recursive calls

thread_local! {
    static INITIALIZING: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
}

pub fn new() -> HipResult<Arc<Self>> {
    INITIALIZING.with_borrow(|init| {
        assert!(!*init, "Recursive HipBackend::new() call detected");
    });

    INITIALIZING.with_borrow_mut(|init| *init = true);
    let result = self::new_internal();
    INITIALIZING.with_borrow_mut(|init| *init = false);

    result
}
```

**Risk Assessment**:
- **Likelihood**: Very Low (requires code change to trigger)
- **Impact**: Deadlock / hang
- **Detection**: Easy - debugger shows deadlock

---

### BUG-9: Inefficient Memory Pooling Strategy

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Lines**: 619-638

**Problematic Code**:
```rust
// Create memory pools (account for 4KB alignment padding)
const ALIGNMENT: usize = 4096;
let mut pools: Vec<HipBuffer> = Vec::new();
let mut current_pool_bytes = 0usize;

for (_, tensor_bytes) in &tensor_list {
    // Account for alignment padding when calculating pool usage
    let aligned_tensor_bytes = (tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);
    if current_pool_bytes + aligned_tensor_bytes > actual_pool_size {
        // Start a new pool
        pools.push(backend.allocate_buffer(actual_pool_size)
            .map_err(|e| anyhow!("Failed to allocate memory pool: {}", e))?);
        current_pool_bytes = 0;
    }
    current_pool_bytes += aligned_tensor_bytes;
}
```

**Why It's Dangerous**:
- **Two-pass algorithm** - first pass counts, second pass allocates
- Doesn't account for **worst-case fragmentation** - tensors might not fit in calculated pools
- Alignment padding is **approximated** - actual padding depends on tensor order
- Could waste GPU memory or fail unexpectedly

**Suggested Fix**:
```rust
// Use greedy first-fit allocation with actual pool objects
let mut pools: Vec<HipBuffer> = Vec::new();

for (_, tensor_bytes) in &tensor_list {
    let aligned_tensor_bytes = (tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);

    // Find pool with enough space
    let pool_idx = pools.iter().position(|p| p.size() >= aligned_tensor_bytes);

    if let Some(idx) = pool_idx {
        // Reuse existing pool
    } else {
        // Allocate new pool
        pools.push(backend.allocate_buffer(actual_pool_size)?);
    }
}
```

**Risk Assessment**:
- **Likelihood**: Medium (edge cases with large models)
- **Impact**: Memory waste / allocation failure
- **Detection**: Medium - monitor GPU memory usage

---

## LOW Severity Issues

### BUG-10: Incorrect Alignment Mask Comment

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Line**: 744

**Problematic Code**:
```rust
// Advance offset for next tensor (ALIGN TO 4KB BOUNDARY)
const ALIGNMENT: usize = 4096;
offset = (offset + tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);
```

**Why It's Problematic**:
- The mask `!(ALIGNMENT - 1)` is **correct** (clears lower 12 bits for 4KB alignment)
- But the comment is **misleading** - it says "align to 4KB boundary" but doesn't explain the bit math
- Could confuse maintainers

**Suggested Fix**:
```rust
// Align offset to next 4KB boundary (clear lower 12 bits)
// Formula: (offset + size + 4095) & ~4095
const ALIGNMENT: usize = 4096;
offset = (offset + tensor_bytes + ALIGNMENT - 1) & !(ALIGNMENT - 1);
```

**Risk Assessment**:
- **Likelihood**: N/A (comment issue)
- **Impact**: Maintainer confusion
- **Detection**: Code review

---

### BUG-11: Inconsistent Error Messages

**File**: Multiple files

**Problematic Code**:
```rust
// hip_backend.rs - Line 279
"Sub-buffer out of bounds: offset={} + size={} > parent_size={}"

// gguf.rs - Line 737
"Failed to create tensor '{}' from pool #{}: {}"

// Both describe similar failures but use different terminology
```

**Why It's Problematic**:
- Inconsistent error terminology makes debugging harder
- "Sub-buffer" vs "pool" vs "buffer" - unclear which failed
- Aggregating logs from multiple sources is difficult

**Suggested Fix**:
```rust
// Standardize on GPU memory terminology
// Use: "GPU memory pool", "sub-allocation", "base allocation"

// Example:
"GPU memory pool #{} sub-allocation failed: offset={} size={} > pool_size={}"
```

**Risk Assessment**:
- **Likelihood**: N/A (documentation issue)
- **Impact**: Debugging difficulty
- **Detection**: Code review

---

### BUG-12: Magic Number for Pool Size

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Line**: 594

**Problematic Code**:
```rust
// Pool size: 1 GB per pool (large enough for biggest tensors, small enough for ROCm)
const POOL_SIZE: usize = 1024 * 1024 * 1024;
```

**Why It's Problematic**:
- Magic number without justification
- 1GB might be too small for future models (larger embedding layers)
- Should be computed based on available GPU memory

**Suggested Fix**:
```rust
// Compute pool size based on GPU memory and safety margin
// Default: 80% of free memory / 4 pools, clamped to [256MB, 2GB]
const DEFAULT_POOL_SIZE_MB: usize = 1024;  // Fallback if query fails

let (free_mem, _) = backend.get_memory_info()
    .unwrap_or((DEFAULT_POOL_SIZE_MB * 1024 * 1024, 0));

let pool_size = (free_mem * 8 / 10 / 4)  // 80% of free / 4 pools
    .clamp(256 * 1024 * 1024, 2 * 1024 * 1024 * 1024);  // [256MB, 2GB]
```

**Risk Assessment**:
- **Likelihood**: Low (works for current models)
- **Impact**: Allocation failure for larger models
- **Detection**: Easy - OOM error

---

### BUG-13: Missing Documentation for Memory Pooling Strategy

**File**: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
**Lines**: 587-649

**Problematic Code**:
```rust
/// Load tensors and upload to GPU using batched memory pooling.
/// This allocates multiple moderate-sized buffers instead of thousands of small allocations,
/// avoiding ROCm driver bugs while staying within reasonable allocation limits.
pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
    // ... implementation ...
}
```

**Why It's Problematic**:
- Documentation doesn't explain **why** ROCm has issues with small allocations
- Doesn't document the **4KB alignment requirement** for D2H copies
- Doesn't explain **selective pooling logic** (why some tensors skip pooling)

**Suggested Fix**:
```rust
/// Load tensors and upload to GPU using batched memory pooling.
///
/// # Memory Pooling Strategy
///
/// ROCm driver has known issues with:
/// 1. Thousands of small hipMalloc() calls (performance degradation)
/// 2. Device-to-host copies from unaligned sub-buffers (data corruption)
/// 3. Memory pool exhaustion (allocation failures)
///
/// To mitigate these issues:
/// - Allocate large pools (1GB) instead of per-tensor allocations
/// - Align all sub-buffers to 4KB boundaries
/// - Skip pooling for large tensors (>32MB) or tensors needing transpose
///
/// # Selective Pooling Criteria
///
/// Tensors are pooled only if they meet **all** criteria:
/// - Size <= 32MB
/// - No transpose required (not embedding/lm_head)
/// - Not QKV attention weights (need concatenation)
///
/// # Arguments
///
/// * `backend` - HIP backend for GPU operations
///
/// # Returns
///
/// * `HashMap<String, DeviceTensor>` - Map from tensor name to GPU tensor
pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
    // ... implementation ...
}
```

**Risk Assessment**:
- **Likelihood**: N/A (documentation issue)
- **Impact**: Maintainer confusion / future bugs
- **Detection**: Code review

---

## Summary by Category

### Memory Safety
- **HIGH**: 3 bugs (pointer overflow, race condition, memory leak)
- **MEDIUM**: 4 bugs (integer overflow, bounds checking, error handling, performance)
- **LOW**: 3 bugs (documentation, error messages, magic numbers)

### Thread Safety
- **HIGH**: 1 bug (singleton race condition)
- **MEDIUM**: 1 bug (Arc cloning performance)
- **LOW**: 0 bugs

### Resource Management
- **HIGH**: 1 bug (memory leak on error path)
- **MEDIUM**: 2 bugs (pool allocation efficiency, error handling)
- **LOW**: 1 bug (magic number)

### Code Quality
- **HIGH**: 0 bugs
- **MEDIUM**: 0 bugs
- **LOW**: 3 bugs (documentation, comments, error messages)

---

## Recommended Action Plan

### Immediate Actions (Before Next Release)
1. **Fix BUG-2** (race condition in singleton) - HIGH priority
2. **Fix BUG-6** (ignored HIP error) - MEDIUM priority
3. **Add bounds checking** for BUG-5 (out-of-bounds pool access) - MEDIUM priority

### Short-term Actions (Next Sprint)
4. Add overflow checks for BUG-1 and BUG-4
5. Fix memory leak in BUG-3 (RAII wrapper)
6. Improve error messages (BUG-11)

### Long-term Actions (Technical Debt)
7. Reconsider singleton pattern (BUG-2, BUG-8)
8. Add dynamic pool sizing (BUG-12)
9. Improve documentation (BUG-13)

---

## Testing Recommendations

### Regression Tests
```rust
#[test]
fn test_singleton_thread_safety() {
    // Spawn 100 threads, all calling HipBackend::new() simultaneously
    // Verify only one backend is created
}

#[test]
fn test_memory_pool_cleanup_on_error() {
    // Inject error during tensor upload
    // Verify GPU memory is freed (monitor memory usage)
}

#[test]
fn test_offset_overflow_detection() {
    // Create buffer near address space limit
    // Attempt to add offset that would overflow
    // Verify error is returned (not panic)
}

#[test]
fn test_pool_allocation_exhaustion() {
    // Create model with many small tensors
    // Verify pool_idx doesn't exceed pools.len()
}
```

### Stress Tests
- Load/unload model 1000 times, monitor GPU memory
- Multi-threaded model loading (100 concurrent requests)
- Very large tensor (>2GB) allocation

### Fuzzing
- Fuzz tensor sizes and offsets in `load_to_gpu()`
- Fuzz alignment values (power-of-two, non-power-of-two)

---

## Conclusion

The Phase 10 memory pooling code is **generally well-implemented** with no critical memory corruption bugs. The main concerns are:

1. **Thread safety** in singleton backend initialization (BUG-2)
2. **Error handling** gaps in FFI calls (BUG-6)
3. **Resource cleanup** on error paths (BUG-3)

All HIGH and MEDIUM bugs have clear remediation paths. The code would benefit from:
- More aggressive bounds checking
- Better error propagation from FFI calls
- RAII patterns for resource management

**Overall Risk Assessment**: **MEDIUM**
- No critical bugs that would cause data corruption in normal operation
- Thread safety issue requires attention for multi-threaded deployments
- Memory leaks could accumulate in long-running server applications

**Recommendation**: Fix HIGH bugs (BUG-1, BUG-2, BUG-3) before next production release.
