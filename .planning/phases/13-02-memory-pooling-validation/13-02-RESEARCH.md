# Phase 13-02: Memory Pooling Validation - Research

**Researched:** 2026-01-19
**Domain:** ROCm GPU memory management, validation testing
**Confidence:** HIGH

## Summary

This research investigated the Phase 10 "selective memory pooling" implementation to understand how to validate its correctness. The primary finding is **critical**: the selective memory pooling documented as "COMPLETE" in `docs/ROCM_D2H_ERROR_RESEARCH.md` was **never actually implemented**.

**Key findings:**
1. No `LARGE_TENSOR_THRESHOLD` constant exists in the codebase (searched all `*.rs` files)
2. No tensor classification logic (`needs_transpose`, `is_qkv`, `is_large`) exists
3. All tensors currently use direct allocation via `DeviceTensor::from_host_vec()`
4. The `DeviceTensor::from_pool()` method exists but is **never called**
5. D2H operations occur via `transpose_2d_tensor()` and `concatenate_qkv_tensors()` which call `to_host_vec()`

**Primary recommendation:** This phase should first verify that selective memory pooling was implemented. If not found (current state), the validation phase must either:
- Document that the feature was never implemented and skip validation
- Implement selective memory pooling first, then validate

The "validation" must also verify that the D2H workaround (avoiding sub-buffer D2H copies) is actually in place.

## Standard Stack

### Core Libraries for Validation

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cargo test` | built-in | Unit test framework | Rust standard for testing |
| `serial_test` | existing | Serialized GPU tests | Already in use for GPU tests |
| `anyhow` | existing | Error handling | Project standard |
| `tracing` | existing | Debug logging | Project standard |

### Test Data Creation

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Custom fixtures | existing | Mock tensor creation | `tests/common/fixtures.rs` |
| Real GGUF models | N/A | Integration testing | Download from HuggingFace |

**No external dependencies needed** - all testing infrastructure exists.

## Architecture Patterns

### Current Code Structure (What Actually Exists)

```
src/
├── backend/hip_backend/
│   └── backend.rs              # HipBuffer, DeviceTensor, AsyncLoader
│                               # Lines 588: sub_buffer_view() - creates offset views
│                               # Lines 2368: from_pool() - UNUSED
│                               # Lines 2342: from_host_vec() - all tensors use this
│                               # Lines 755: copy_to_host() - calls hipMemcpyDtoH
│                               # Lines 2227: to_host_vec() - D2H for transpose
│
├── loader/
│   └── gguf.rs                 # GGUF tensor loading
│                               # Lines 861: load_tensor_to_gpu() - uses from_host_vec()
│                               # Lines 922: load_to_gpu() - sequential loading
│                               # Lines 957: load_to_gpu_async() - parallel loading
│
├── model/execution_plan/
│   └── execution_plan_src.rs   # Model weight mapping
│                               # Lines 2697: map_embedding() - may transpose
│                               # Lines 2777: map_lm_head() - may transpose
│                               # Lines 2899: transpose_2d_tensor() - D2H operation
│                               # Lines 3260: concatenate_qkv_tensors() - D2H via to_host_vec()
│
└── ggml/hip_backend/
    └── allocator.rs            # TensorAllocator (different codebase path)

tests/
├── common/
│   └── fixtures.rs             # GGUF file creation helpers
│                               # Lines 18: create_test_gguf()
│                               # Lines 57: create_test_gguf_with_f32()
│                               # Lines 112: create_embedding_gguf()
│
└── gguf_loader_tests.rs        # Existing GGUF loader tests
```

### Pattern 1: Tensor Loading (Current Implementation)

**What:** All tensors use direct allocation via `from_host_vec()`

**File:** `src/loader/gguf.rs:861`
```rust
// Upload to GPU
let device_tensor = DeviceTensor::from_host_vec(backend, f32_data, shape)
    .map_err(|e| anyhow!("Failed to upload tensor '{}' to GPU: {}", name, e))?;
```

**Key characteristic:** Each tensor gets its own `hipMalloc` - no memory pooling.

### Pattern 2: D2H Operations (The Problem Zone)

**What:** Tensors that need transpose/concatenate use `to_host_vec()` then re-upload

**File:** `src/model/execution_plan/execution_plan_src.rs:2899-2928`
```rust
fn transpose_2d_tensor(backend: &HipBackend, tensor: &DeviceTensor) -> HipResult<DeviceTensor> {
    let host = tensor.to_host_vec()?;  // D2H copy via hipMemcpyDtoH
    let mut transposed = vec![0.0f32; host.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src_idx = r * cols + c;
            let dst_idx = c * rows + r;
            transposed[dst_idx] = host[src_idx];
        }
    }
    DeviceTensor::from_host_vec(backend, transposed, new_shape)
}
```

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

### Pattern 3: Sub-Buffer View (Unused but Available)

**What:** `HipBuffer::sub_buffer_view()` creates offset views into parent buffers

**File:** `src/backend/hip_backend/backend.rs:588`
```rust
pub fn sub_buffer_view(&self, offset: usize, size: usize) -> HipResult<Self> {
    if offset + size > self.size() {
        return Err(HipError::MemoryAllocationFailed(...));
    }
    Ok(HipBuffer {
        inner: Arc::new(HipBufferInner {
            ptr: self.inner.ptr,        // Share same base pointer
            size,
            offset: self.inner.offset + offset,  // Accumulate offset
        }),
    })
}
```

**Key characteristic:** Creates views that would fail D2H on ROCm 7.1.1 (per research doc).

### Pattern 4: Mock Tensor Creation for Testing

**What:** Test fixtures create synthetic GGUF files with controlled tensor properties

**File:** `tests/common/fixtures.rs:18-55`
```rust
pub fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write GGUF magic
    writer.write_all(b"GGUF")?;
    // ... write headers, tensor info, data
}
```

**File:** `tests/common/fixtures.rs:112-150`
```rust
pub fn create_embedding_gguf(
    path: &Path,
    vocab_size: usize,
    hidden_size: usize,
) -> anyhow::Result<()> {
    // Creates token_embd.weight [vocab_size, hidden_size]
    // Creates output.weight [vocab_size, hidden_size]
}
```

### Anti-Patterns to Avoid

- **Assuming implementation exists:** The documented selective pooling was never implemented
- **Testing non-existent code:** Cannot validate "pooling behavior" that doesn't exist
- **Missing D2H detection:** Must detect where `hipMemcpyDtoH` is called on sub-buffers

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GGUF file parsing | Manual binary parsing | `GgufLoader` | Already handles all tensor types |
| Mock tensor data | Manual tensor creation | `tests/common/fixtures.rs` | Proven patterns for synthetic GGUF |
| GPU test setup | Custom test init | `GPU_FIXTURE` from `tests/common` | Handles backend initialization |
| Test isolation | Manual cleanup | `#[serial]` from `serial_test` | Prevents concurrent GPU test conflicts |

## Common Pitfalls

### Pitfall 1: Validating Non-Existent Code

**What goes wrong:** Writing tests for "selective memory pooling" that was never implemented.

**Why it happens:** Documentation (`ROCM_D2H_ERROR_RESEARCH.md`) claims status is "COMPLETE" but code doesn't exist.

**How to avoid:**
1. First verify implementation exists by grepping for `LARGE_TENSOR_THRESHOLD`
2. If not found, document the gap before planning validation
3. Either skip validation or plan implementation first

**Warning signs:** Test code passes but never actually calls the supposed logic.

### Pitfall 2: Missing D2H Detection

**What goes wrong:** Not detecting when `hipMemcpyDtoH` is called on sub-buffer views.

**Why it happens:** D2H calls are wrapped in abstractions (`to_host_vec()`, `copy_to_host()`).

**How to avoid:**
1. Use static analysis to trace all D2H paths
2. Add logging to `HipBuffer::copy_to_host()` to detect sub-buffer usage
3. Check `buffer.inner.offset > 0` to identify sub-buffer D2H calls

**Warning signs:** Tests pass but actual D2H errors occur in production.

### Pitfall 3: Confusing Direct Allocation with Pooling

**What goes wrong:** Assuming `from_host_vec()` uses pooling under the hood.

**Why it happens:** Both allocate GPU memory; only difference is pooling reuses allocations.

**How to avoid:**
1. `from_host_vec()` calls `allocate_buffer()` which calls `hipMalloc()` directly
2. `from_pool()` would use `sub_buffer_view()` on pre-allocated pools
3. Count `hipMalloc` calls to verify pooling reduces allocations

**Warning signs:** Memory usage doesn't decrease despite "pooling".

### Pitfall 4: Testing Only Small Tensors

**What goes wrong:** Validation uses small tensors that wouldn't trigger pooling logic anyway.

**Why it happens:** Small tensors are faster for tests.

**How to avoid:**
1. Test with tensors < 32MB (should pool if implemented)
2. Test with tensors > 32MB (should use direct allocation)
3. Test exactly at 32MB boundary
4. Include the original failing case: 520MB `token_embd.weight`

**Warning signs:** All tensors use same allocation path regardless of size.

## Code Examples

### Example 1: Creating Mock Tensors for Testing

**Source:** `tests/common/fixtures.rs:18`

```rust
pub fn create_test_gguf(path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write GGUF magic
    writer.write_all(b"GGUF")?;
    writer.write_all(&3u32.to_le_bytes())?;  // Version 3
    writer.write_all(&0u64.to_le_bytes())?;  // Tensor count: 0

    // Write metadata
    write_kv_string(&mut writer, "general.architecture", "test")?;

    writer.flush()?;
    Ok(())
}
```

### Example 2: Creating Embedding Tensors (D2H Trigger)

**Source:** `tests/common/fixtures.rs:112`

```rust
pub fn create_embedding_gguf(
    path: &Path,
    vocab_size: usize,
    hidden_size: usize,
) -> anyhow::Result<()> {
    // GGUF magic + version
    file.write_all(b"GGUF")?;
    file.write_all(&3u32.to_le_bytes())?;

    // 2 tensors: token_embd.weight and output.weight
    file.write_all(&2u64.to_le_bytes())?;

    // ... write tensor info for [vocab_size, hidden_size] tensors
    // These will trigger transpose if [hidden, vocab] layout detected
}
```

### Example 3: Detecting D2H on Sub-Buffers

**Pattern to implement:**

```rust
// Add to HipBuffer::copy_to_host() for detection
pub fn copy_to_host<T>(&self, data: &mut [T]) -> HipResult<()> {
    // Detect if this is a sub-buffer D2H
    if self.inner.offset > 0 {
        tracing::warn!(
            "D2H from sub-buffer detected: base_ptr={:?}, offset={}, size={} MB",
            self.inner.ptr,
            self.inner.offset,
            self.size() / 1024 / 1024
        );
        // This would fail on ROCm 7.1.1 with HIP_ERROR_INVALID_VALUE
    }

    let result = unsafe {
        hipMemcpy(
            data.as_mut_ptr() as *mut c_void,
            self.ptr(),
            byte_size,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
    };
    // ...
}
```

### Example 4: Tensor Classification Logic (If Implementing)

**Source:** `docs/ROCM_D2H_ERROR_RESEARCH.md:218-244` (NOT IMPLEMENTED)

```rust
// This code does NOT exist in the codebase
const LARGE_TENSOR_THRESHOLD: usize = 32 * 1024 * 1024;  // 32 MB

for (name, tensor) in &self.tensors {
    let tensor_bytes = num_elements * std::mem::size_of::<f32>();

    // Check if tensor needs direct allocation
    let needs_transpose = tensor.shape.dims().len() == 2 &&
        ((tensor.shape.dims()[0] == vocab_size || tensor.shape.dims()[1] == vocab_size) ||
         name.contains("embd") || name.contains("output"));
    let is_qkv = name.contains("attn_") || name.contains("q_proj") ||
                 name.contains("k_proj") || name.contains("v_proj");
    let is_large = tensor_bytes > LARGE_TENSOR_THRESHOLD;

    if is_large || needs_transpose || is_qkv {
        // Direct allocation - bypass pooling
        let device_tensor = DeviceTensor::from_host_vec(backend, data, tensor.shape.clone())?;
        gpu_tensors.insert(name.clone(), device_tensor);
    } else {
        // Use memory pooling (NOT IMPLEMENTED)
        let device_tensor = DeviceTensor::from_pool(&pools[pool_idx], offset, f32_data, tensor.shape.clone())?;
        offset += aligned_bytes;
    }
}
```

## State of the Art

### Documentation vs Reality Gap

| Documented | Actual | Status |
|------------|--------|--------|
| Selective memory pooling implemented | No `LARGE_TENSOR_THRESHOLD` found | **NOT IMPLEMENTED** |
| Tensors classified for pooling | No classification logic | **NOT IMPLEMENTED** |
| 70% reduction in hipMalloc calls | All tensors use `from_host_vec()` | **NOT IMPLEMENTED** |
| ~200 tensors pooled | `from_pool()` never called | **NOT IMPLEMENTED** |

**Root cause:** Research document (`ROCM_D2H_ERROR_RESEARCH.md`) was written as a design/proposal, not as implementation documentation. The "Status: COMPLETE" marker is misleading.

### D2H Operations (Current Implementation)

| Location | Trigger | Uses Sub-Buffer? |
|----------|---------|------------------|
| `transpose_2d_tensor()` | Embedding/LM head transpose | **No** - uses `to_host_vec()` on direct allocation |
| `concatenate_qkv_tensors()` | Separate QKV fusion | **No** - uses `to_host_vec()` on direct allocation |
| MLP validation reads | Debug output | **No** - direct allocation |
| Test assertions | Verification | **No** - direct allocation |

**Current status:** No D2H from sub-buffers because no sub-buffers exist (no pooling).

## Open Questions

### Question 1: Should This Phase Implement or Validate?

**What we know:**
- Selective memory pooling was designed but not implemented
- Documentation claims it's "COMPLETE"
- Code shows no evidence of implementation

**What's unclear:**
- Was the implementation lost in a refactoring?
- Was it never implemented despite documentation?
- Should Phase 13-02 implement it, or just document that it doesn't exist?

**Recommendation:**
1. First add a verification step: check if `LARGE_TENSOR_THRESHOLD` exists
2. If not found, document the gap and recommend either:
   - Skip validation (feature doesn't exist)
   - Convert to implementation phase (implement first, then validate)

### Question 2: What Is the Actual Validation Target?

**What we know:**
- Phase 13-02 is titled "Memory Pooling Validation"
- Success criteria refer to "selective memory pooling"
- No selective pooling exists to validate

**What's unclear:**
- What should actually be validated?
- Is there some other memory optimization that was implemented?
- Should we validate that current code (direct allocation) works correctly?

**Recommendation:**
Clarify with stakeholder whether to:
- Validate current direct-allocation-only approach
- Implement selective pooling first
- Document that the documented feature was never implemented

### Question 3: How to Detect hipMemcpyDtoH Calls?

**What we know:**
- D2H happens via `copy_to_host()` method
- Wrapped by `to_host_vec()` for convenience
- Called from `transpose_2d_tensor()` and `concatenate_qkv_tensors()`

**What's unclear:**
- Should we add runtime detection/logging?
- Is static analysis sufficient?
- What constitutes "validation" that no sub-buffer D2H occurs?

**Recommendation:**
1. Add compile-time assertion that `from_pool()` is never called (currently true)
2. Add runtime logging to detect if `copy_to_host()` is called with `offset > 0`
3. Document that current implementation avoids sub-buffer D2H by not using sub-buffers

## Sources

### Primary (HIGH confidence)

| Source | Location | What Was Checked |
|--------|----------|------------------|
| `docs/ROCM_D2H_ERROR_RESEARCH.md` | Lines 218-244 | Documented selective pooling design |
| `src/backend/hip_backend/backend.rs` | Lines 588-605 | `sub_buffer_view()` implementation |
| `src/backend/hip_backend/backend.rs` | Lines 2368-2420 | `from_pool()` implementation (unused) |
| `src/backend/hip_backend/backend.rs` | Lines 2342-2353 | `from_host_vec()` implementation |
| `src/backend/hip_backend/backend.rs` | Lines 755-809 | `copy_to_host()` - D2H operation |
| `src/backend/hip_backend/backend.rs` | Lines 2227-2236 | `to_host_vec()` - wrapper |
| `src/loader/gguf.rs` | Lines 855-886 | `load_tensor_to_gpu()` - all use `from_host_vec()` |
| `src/model/execution_plan/execution_plan_src.rs` | Lines 2899-2929 | `transpose_2d_tensor()` - D2H trigger |
| `src/model/execution_plan/execution_plan_src.rs` | Lines 3260-3330 | `concatenate_qkv_tensors()` - D2H trigger |
| `tests/common/fixtures.rs` | Lines 18-150 | Mock GGUF creation patterns |
| `docs/QWEN2_HEAD_DIM_BUG_INVESTIGATION.md` | Lines 150-166 | Confirmation that selective pooling was not found |

### Secondary (MEDIUM confidence)

| Source | Location | What Was Checked |
|--------|----------|------------------|
| `docs/CODE_REVIEW_PHASE_10_MEMORY_POOLING_2026-01-07.md` | Lines 96-112 | Review notes on threshold and classification |
| `src/backend/hip_backend/backend.rs` | Lines 3497-3648 | `AsyncLoader` implementation for parallel uploads |

### Tertiary (LOW confidence)

| Source | Notes |
|--------|-------|
| `/home/feanor/Projects/rocm-examples/` | Referenced in context but not accessed during research |

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** - All test infrastructure exists in codebase
- Architecture patterns: **HIGH** - Code paths traced and verified
- Pitfalls: **HIGH** - Documentation vs code gap is clear
- Open questions: **HIGH** - Existence of implementation is verifiably absent

**Research date:** 2026-01-19
**Valid until:** 30 days (codebase is stable; no major refactoring expected)

## Critical Finding for Planning

**The selective memory pooling feature documented as "COMPLETE" does not exist in the codebase.**

Verification commands:
```bash
# Search for LARGE_TENSOR_THRESHOLD
grep -r "LARGE_TENSOR_THRESHOLD" src/
# Result: No matches found

# Search for from_pool usage
grep -r "from_pool" src/
# Result: Only definition in backend.rs, never called

# Search for tensor classification
grep -r "needs_transpose\|is_qkv\|is_large" src/
# Result: No matches found
```

**Implication for Phase 13-02:**
This cannot be a "validation-only" phase because there is nothing to validate. The phase must either:
1. Convert to an implementation phase (implement selective pooling, then validate)
2. Become a documentation phase (document that the feature was never implemented)
3. Validate the current state (direct allocation only, no pooling)

The planner should clarify the expected outcome before creating tasks.
