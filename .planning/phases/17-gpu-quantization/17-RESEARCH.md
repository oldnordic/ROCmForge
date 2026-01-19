# Phase 17: GPU Quantization - Research

**Researched:** 2026-01-19
**Domain:** GPU quantization dequantization and fused matmul for GGUF formats (Q4_0, Q4_K, Q6_K)
**Confidence:** HIGH

## Summary

This phase implements **on-device quantization operations** for GGUF quantized weights. The codebase already has comprehensive HIP kernels for Q4_0, Q4_K, and Q6_K dequantization and fused matmul operations. The kernels compile successfully and are integrated into the build system. However, the current implementation uses CPU dequantization followed by GPU upload of FP32 data, which wastes memory bandwidth.

**Key finding:** All required GPU kernels already exist. The work is primarily:
1. Integrate existing GPU dequantization kernels into the tensor loading path
2. Integrate existing fused matmul kernels for on-device computation
3. Remove CPU fallback paths for quantized tensors on GPU
4. Add tests to verify GPU execution path is used

**Primary recommendation:** Use existing HIP kernels (`q4_0_dequant.hip`, `q4_k_dequant.hip`, `q6_k_dequant.hip`) and fused matmul kernels (`q4_0_matmul.hip`, `q4_k_matmul.hip`, `q6_k_matmul.hip`). Wire them into the `GgufLoader::load_tensor_to_gpu()` path.

## Standard Stack

### Core (Already Exists)
| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| `q4_0_dequant.hip` | `kernels/` | Q4_0 GPU dequantization kernel | Compiled, not wired to loader |
| `q4_k_dequant.hip` | `kernels/` | Q4_K GPU dequantization kernel | Compiled, not wired to loader |
| `q6_k_dequant.hip` | `kernels/` | Q6_K GPU dequantization kernel | Compiled, not wired to loader |
| `q4_0_matmul.hip` | `kernels/` | Fused Q4_0 dequant+matmul | Compiled, partially integrated |
| `q4_k_matmul.hip` | `kernels/` | Fused Q4_K dequant+matmul | Compiled, partially integrated |
| `q6_k_matmul.hip` | `kernels/` | Fused Q6_K dequant+matmul | Compiled, partially integrated |

### Supporting Rust Modules
| Module | Location | Purpose |
|--------|----------|---------|
| `quantized_matmul.rs` | `src/ggml/hip_backend/ops/` | GPU matmul with quantized weights |
| `batch_quantized.rs` | `src/ggml/hip_backend/ops/` | Batch processing for quantized ops |
| `q4_0_dequant.rs` | `src/ggml/hip_backend/ops/` | Q4_0 dequantization (currently CPU fallback) |
| `dequant.rs` | `src/loader/` | CPU dequantization dispatcher |

### Build Integration
All kernels are already configured in `build.rs`:
- `Q4_0_DEQUANT_HSACO` - Path to compiled Q4_0 dequant kernel
- `Q4_K_DEQUANT_HSACO` - Path to compiled Q4_K dequant kernel
- `Q6_K_DEQUANT_HSACO` - Path to compiled Q6_K dequant kernel
- `Q4_0_MATMUL_HSACO` - Path to compiled Q4_0 matmul kernel
- `Q4_K_MATMUL_HSACO` - Path to compiled Q4_K matmul kernel
- `Q6_K_MATMUL_HSACO` - Path to compiled Q6_K matmul kernel

## Architecture Patterns

### Current State (CPU Dequantization)

```
GGUF File -> mmap read -> CPU dequantize -> FP32 upload -> GPU
      |              |            |               |            |
      v              v            v               v            v
   Quantized      Raw bytes   CPU work     FP32 buffer   GPU tensor
   data (on
    disk)
```

**Problems:**
- CPU dequantization is slow (blocks GPU loading)
- Uploading FP32 wastes 4-17x more memory bandwidth
- Weights stored as FP32 on GPU (wastes VRAM)

### Target State (GPU Dequantization)

```
GGUF File -> mmap read -> Direct GPU upload -> GPU dequantize -> GPU use
      |              |              |                |            |
      v              v              v                v            v
   Quantized      Raw bytes    Quantized        Dequant      Compute
   data (on       (minimal)    on GPU          on kernel    directly
    disk)
```

**Benefits:**
- ~17x reduction in memory bandwidth for weight loading
- ~4x reduction in GPU memory usage for weights
- Parallel GPU dequantization

### Pattern 1: GPU Dequantization Kernel Invocation

```rust
// Source: Existing kernel invocation pattern in quantized_matmul.rs

// Load kernel module from HSACO file
let kernel_path = env::var("Q4_0_DEQUANT_HSACO")?;
let module = backend.load_module(&kernel_path)?;
let kernel = backend.get_kernel_function(&module, "q4_0_to_fp32_batch_kernel")?;

// Calculate grid dimensions
let num_elements = total_elements;
let block_size = 256;
let grid_size = (num_elements + block_size - 1) / block_size;

// Launch kernel
let grid_dim = (grid_size as u32, 1, 1);
let block_dim = (block_size, 1, 1);

backend.launch_kernel_with_module_shared(
    &kernel,
    grid_dim,
    block_dim,
    &[
        &mut input_ptr as *mut _ as *mut c_void,
        &mut output_ptr as *mut _ as *mut c_void,
        &mut num_elements_arg as *mut _ as *mut c_void,
    ],
    0, // shared_mem_bytes
)?;
```

### Pattern 2: Fused Dequant+MatMul

```rust
// Source: Existing fused kernel in q4_0_matmul.hip
// Key insight: Never materialize FP32 weights, dequant on-the-fly during matmul

// CPU-side: Upload quantized weights directly to GPU
let weight_buffer = backend.allocate_buffer(quantized_weights.len())?;
weight_buffer.copy_from_host(quantized_weights)?;

// GPU kernel: Dequantizes weights during matmul computation
// Eliminates intermediate FP32 buffer
unsafe {
    matmul_q4_0_gpu(
        backend,
        activations_ptr,
        weights_q4_0_ptr,  // Still quantized!
        output_ptr,
        m, n, k,
    )?;
}
```

### Pattern 3: LazyTensor GPU Loading

The `LazyTensor` enum needs a new variant for quantized GPU tensors:

```rust
pub enum LazyTensor {
    // Existing variants...
    Unloaded { ... },
    Gpu { tensor: Arc<DeviceTensor> },  // FP32 tensor

    // NEW: Quantized tensor on GPU (not yet dequantized)
    GpuQuantized {
        name: String,
        buffer: HipBuffer,      // Raw quantized data on GPU
        shape: TensorShape,
        tensor_type: GgufTensorType,
    },
}
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Q4_0 dequantization | New kernel | `kernels/q4_0_dequant.hip` | Already handles 4-bit unpacking, scale multiplication |
| Q4_K dequantization | New kernel | `kernels/q4_k_dequant.hip` | Complex super-block structure with mins/scales |
| Q6_K dequantization | New kernel | `kernels/q6_k_dequant.hip` | 6-bit unpacking across byte boundaries |
| Fused matmul | Custom fusion | `kernels/q4_0_matmul.hip`, etc. | Memory bandwidth optimization is critical |
| FP16 conversion | Bit manipulation | Existing `f16_to_f32()` in kernels | Tested and optimized |

**Key insight:** The kernels are already written and tuned for AMD GPU architectures (RDNA3/CDNA2). The work is integration, not kernel development.

## Common Pitfalls

### Pitfall 1: Synchronous CPU Dequantization Before GPU Upload

**What goes wrong:** Current code dequantizes on CPU, then uploads FP32 to GPU. This defeats the purpose of GPU quantization.

**Why it happens:** The `GgufLoader::load_tensor_to_gpu()` method calls `dequantize_q4_0()`, `dequantize_q4_k()`, etc., which are CPU functions.

**How to avoid:**
- Upload quantized bytes directly to GPU
- Launch dequantization kernel on GPU
- Only read back to CPU when necessary (rare)

**Warning signs:**
- `Vec<f32>` allocation for dequantized data
- `copy_from_host` with FP32 data for quantized tensors
- CPU loops processing quantized bytes

### Pitfall 2: Materializing FP32 Weights Before MatMul

**What goes wrong:** Dequantizing weights to FP32 buffer, then doing matmul. Wastes memory bandwidth.

**Why it happens:** Easiest path is to dequantize first, then use existing matmul.

**How to avoid:**
- Use fused matmul kernels (`q4_0_matmul_kernel`, etc.)
- These kernels dequantize on-the-fly during dot product computation
- Eliminates intermediate FP32 buffer entirely

**Warning signs:**
- Allocating FP32 buffer for dequantized weights
- Separate dequant then matmul steps
- High memory usage for weight matrices

### Pitfall 3: Incorrect Grid/Block Dimensions

**What goes wrong:** Incorrect grid calculation leads to out-of-bounds access or incomplete processing.

**Why it happens:** GPU kernels need exact grid dimensions based on tensor size.

**How to avoid:**
- Always calculate: `grid_size = (num_elements + block_size - 1) / block_size`
- Add bounds checks in kernel
- Test with non-multiple-of-block-size tensors

**Warning signs:**
- Segfaults in GPU kernels
- Incorrect output values
- Crashes on specific tensor sizes

### Pitfall 4: Forgetting to Synchronize

**What goes wrong:** Launching kernel and immediately using output without synchronization.

**Why it happens:** GPU execution is asynchronous.

**How to avoid:**
- Call `backend.synchronize()` after kernel launch
- Or use events for async synchronization

**Warning signs:**
- Random output values
- Flaky tests
- Heisenbugs

## Code Examples

### Example 1: GPU Dequantization Invocation (Pattern)

```rust
// Source: Adapted from quantized_matmul.rs kernel loading pattern

use crate::backend::hip_backend::{HipBackend, HipError, HipKernel, HipModule};
use std::env;

pub fn dequantize_q4_0_gpu(
    backend: &HipBackend,
    quantized_data: &[u8],
    output: &HipBuffer,
    num_elements: usize,
) -> Result<(), HipError> {
    // Load kernel module
    let kernel_path = env::var("Q4_0_DEQUANT_HSACO")
        .map_err(|_| HipError::KernelLoadFailed("Q4_0_DEQUANT_HSACO not set".into()))?;

    let module = backend.load_module(&kernel_path)?;
    let kernel = backend.get_kernel_function(&module, "q4_0_to_fp32_batch_kernel")?;

    // Allocate GPU buffer for input (quantized data)
    let input_buffer = backend.allocate_buffer(quantized_data.len())?;
    input_buffer.copy_from_host(quantized_data)?;

    // Calculate grid dimensions
    let block_size = 256;
    let grid_size = (num_elements + block_size - 1) / block_size;

    // Prepare kernel arguments
    let mut input_arg = input_buffer.as_ptr() as *const u8;
    let mut output_arg = output.as_mut_ptr() as *mut f32;
    let mut count_arg = num_elements as i32;

    let args: &[*mut c_void] = &[
        &mut input_arg as *mut _ as *mut c_void,
        &mut output_arg as *mut _ as *mut c_void,
        &mut count_arg as *mut _ as *mut c_void,
    ];

    // Launch kernel
    backend.launch_kernel_with_module_shared(
        &kernel,
        (grid_size as u32, 1, 1),
        (block_size, 1, 1),
        args,
        0,
    )?;

    // Synchronize to ensure completion
    backend.synchronize()?;

    Ok(())
}
```

### Example 2: Fused Matmul Usage

```rust
// Source: Adapted from quantized_matmul.rs

pub fn matmul_with_q4_0_weights(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &HipBuffer,  // FP32 activations
    output: &HipBuffer, // FP32 output
    n_rows: usize,
    n_cols: usize,
) -> Result<(), String> {
    // Upload quantized weights directly (no dequantization!)
    let weight_buffer = backend.allocate_buffer(quantized_weights.len())?;
    weight_buffer.copy_from_host(quantized_weights)?;

    // Get pointers
    let input_ptr = input.as_ptr() as *const f32;
    let weight_ptr = weight_buffer.as_ptr() as *const u8;  // Still Q4_0!
    let output_ptr = output.as_mut_ptr() as *mut f32;

    // Launch fused kernel (dequantizes on-the-fly)
    unsafe {
        matmul_q4_0_gpu(
            backend,
            input_ptr,
            weight_ptr,
            output_ptr,
            1,  // m (batch size)
            n_rows,
            n_cols,
        )?;
    }

    backend.synchronize()?;
    Ok(())
}
```

### Example 3: Loader Integration Point

```rust
// Source: Based on GgufLoader::load_tensor_to_gpu() in src/loader/gguf.rs

impl GgufLoader {
    pub fn load_tensor_to_gpu(
        &self,
        name: &str,
        backend: &HipBackend,
    ) -> Result<Arc<DeviceTensor>> {
        // ... cache check and metadata lookup ...

        let (offset, size, shape, tensor_type) = match lazy {
            LazyTensor::Unloaded { offset, size, shape, tensor_type, .. } => {
                (*offset, *size, TensorShape::from_dims(shape), *tensor_type)
            }
            _ => return Err(...),
        };

        // Read quantized bytes from mmap
        let tensor_bytes = mmap.get_slice(offset, size)?;

        // NEW: Check if quantized type that supports GPU dequantization
        match tensor_type {
            GgufTensorType::Q4_0 => {
                // Upload quantized data directly to GPU
                let output = self.dequantize_q4_0_gpu(backend, tensor_bytes, &shape)?;
                // Cache and return
            }
            GgufTensorType::Q4_K => {
                let output = self.dequantize_q4_k_gpu(backend, tensor_bytes, &shape)?;
                // Cache and return
            }
            GgufTensorType::Q6_K => {
                let output = self.dequantize_q6_k_gpu(backend, tensor_bytes, &shape)?;
                // Cache and return
            }
            // ... other types use existing CPU dequantization ...
        }
    }
}
```

## Quantization Format Reference

### Q4_0 Format
- **Block size:** 32 elements
- **Bytes per block:** 20 (4-byte scale + 16 bytes packed 4-bit values)
- **Dequantization:** `value = scale * ((packed & 0x0F) - 8)`
- **Kernel:** `q4_0_to_fp32_kernel` (basic), `q4_0_to_fp32_batch_kernel` (optimized)

### Q4_K Format
- **Super-block size:** 256 elements (8 sub-blocks of 32)
- **Bytes per super-block:** 256
  - 16 bytes: 8 half-precision scales (2 bytes each)
  - 16 bytes: 8 int8 mins (1 byte each)
  - 224 bytes: quantized values
- **Dequantization:** `value = min + (quant * scale)`
- **Kernel:** `q4_k_to_fp32_kernel`, `q4_k_to_fp32_batch_kernel`

### Q6_K Format
- **Block size:** 256 elements
- **Bytes per block:** 256
  - 32 bytes: 16 half-precision scales (2 bytes each)
  - 192 bytes: 6-bit packed values
  - 32 bytes: padding
- **Dequantization:** `value = signed_6bit * scale`
- **Kernel:** `q6_k_to_fp32_kernel`, `q6_k_to_fp32_batch_kernel`

## State of the Art

### Existing Kernels (High Confidence)

| Kernel | Source | Status | Env Var |
|--------|--------|--------|---------|
| Q4_0 dequant | `kernels/q4_0_dequant.hip` | Compiles | `Q4_0_DEQUANT_HSACO` |
| Q4_K dequant | `kernels/q4_k_dequant.hip` | Compiles | `Q4_K_DEQUANT_HSACO` |
| Q6_K dequant | `kernels/q6_k_dequant.hip` | Compiles | `Q6_K_DEQUANT_HSACO` |
| Q4_0 matmul | `kernels/q4_0_matmul.hip` | Compiles | `Q4_0_MATMUL_HSACO` |
| Q4_K matmul | `kernels/q4_k_matmul.hip` | Compiles | `Q4_K_MATMUL_HSACO` |
| Q6_K matmul | `kernels/q6_k_matmul.hip` | Compiles | `Q6_K_MATMUL_HSACO` |

### CPU Fallback (Current Implementation)

The current implementation in `src/loader/gguf.rs` uses CPU dequantization:

```rust
// Current (lines 744-753 for Q4_0)
GgufTensorType::Q4_0 => {
    let temp_tensor = GgufTensor { ... };
    self.dequantize_q4_0(&temp_tensor)?  // CPU dequantization!
}
```

This needs to be replaced with GPU kernel invocation.

### Deprecated/Outdated

- **CPU-only dequantization for GPU tensors:** Should be removed once GPU path is verified
- **Separate dequant+matmul:** Use fused kernels instead

## Open Questions

### Question 1: How to handle mixed precision in GPU cache?

**What we know:** `LazyTensor::Gpu` stores `Arc<DeviceTensor>` which assumes FP32 data. Quantized tensors on GPU are not FP32.

**Options:**
1. Add new `LazyTensor::GpuQuantized` variant
2. Store quantized data in `HipBuffer` and dequant on-demand
3. Always dequant to FP32 immediately (simpler, loses some benefits)

**Recommendation:** Option 2 for Phase 17 - keep quantized data on GPU, dequant just-in-time for compute. This enables fused matmul optimization.

### Question 2: Should we pre-dequant or use fused kernels?

**What we know:** Fused kernels eliminate intermediate FP32 buffer (~17x bandwidth savings).

**Trade-offs:**
- Pre-dequant: Simpler, can cache FP32 results
- Fused: More complex, but saves memory bandwidth

**Recommendation:** Use fused kernels for matmul operations. Pre-dequant only for operations that don't have fused kernels.

### Question 3: Test coverage for GPU quantization?

**What we know:** Existing tests in `tests/q_dequant_tests.rs` are CPU-only.

**What's unclear:** What tests are needed to verify GPU execution path?

**Recommendation:** Add GPU tests that:
1. Verify GPU dequantization matches CPU reference
2. Verify fused matmul matches dequant+matmul reference
3. Benchmark performance improvement

## Sources

### Primary (HIGH confidence)
- `kernels/q4_0_dequant.hip` - Verified Q4_0 GPU dequantization kernel implementation
- `kernels/q4_k_dequant.hip` - Verified Q4_K GPU dequantization kernel implementation
- `kernels/q6_k_dequant.hip` - Verified Q6_K GPU dequantization kernel implementation
- `kernels/q4_0_matmul.hip` - Verified fused Q4_0 dequant+matmul kernel
- `kernels/q4_k_matmul.hip` - Verified fused Q4_K dequant+matmul kernel
- `kernels/q6_k_matmul.hip` - Verified fused Q6_K dequant+matmul kernel
- `build.rs` - Verified kernel compilation configuration and env var setup

### Secondary (MEDIUM confidence)
- `src/ggml/hip_backend/ops/quantized_matmul.rs` - Verified kernel loading and invocation patterns
- `src/ggml/hip_backend/ops/batch_quantized.rs` - Verified batch processing patterns
- `src/ggml/hip_backend/ops/q4_0_dequant.rs` - Verified CPU fallback implementation (to be replaced)
- `src/loader/gguf.rs` - Verified current CPU dequantization path
- `src/loader/lazy_tensor.rs` - Verified LazyTensor structure

### Tertiary (LOW confidence)
- `tests/q_dequant_tests.rs` - CPU-only dequantization tests (need GPU variants)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All kernels verified in source, compiled by build.rs
- Architecture: HIGH - Integration points identified, patterns documented
- Pitfalls: HIGH - Based on verified code analysis

**Research date:** 2026-01-19
**Valid until:** 30 days (kernels and build system are stable)
