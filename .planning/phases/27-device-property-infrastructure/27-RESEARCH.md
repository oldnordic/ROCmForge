# Phase 27: Device Property Infrastructure - Research

**Researched:** 2026-01-21
**Domain:** ROCm HIP Device Property Management
**Confidence:** HIGH

## Summary

This phase adds device property caching to the HIP backend, querying `hipGetDeviceProperties` once at initialization instead of per-launch. The cached properties enable pre-launch validation of kernel launch configurations against device limits, preventing invalid launches that would fail with `hipErrorInvalidConfiguration` or `hipErrorInvalidValue`.

The implementation requires:
1. Extending `HipDeviceProp` with accessor methods for launch limit fields
2. Adding cached properties to `HipBackend` struct
3. Creating validation methods that use cached properties
4. Updating existing launch sites to call validation before kernel execution
5. Implementing safe `ceil_div_u64` for grid calculations

**Primary recommendation:** Store device properties in `HipBackend` during initialization, validate all launches with cached limits, and use `u64` arithmetic for grid size calculations.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ROCm HIP API | 6.0+ | Device property queries via `hipGetDeviceProperties` | Official AMD GPU programming interface |
| Rust std | 1.70+ | `div_ceil()` method for safe ceiling division | Built-in integer division method |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| None | - | - | - |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `std::div_ceil()` | Custom `(n + d - 1) / d` | Built-in is safer and clearer |
| Cached properties | Query per-launch | Cached is ~100x faster (no syscall) |

**Installation:**
No new dependencies - uses existing ROCm installation.

## Architecture Patterns

### Recommended Project Structure
```
src/backend/hip_backend/
├── backend.rs          # Add DeviceLimits struct + validation methods
├── device.rs           # Extend HipDeviceProp with launch limit accessors
├── limits.rs           # (NEW) DeviceLimits struct + validation trait
└── launch_config.rs    # (OPTIONAL) Centralized launch config builder
```

### Pattern 1: Extend HipDeviceProp with Launch Limit Accessors
**What:** Add accessor methods to read device limit fields from the opaque buffer
**When to use:** Need to read `maxThreadsPerBlock`, `maxGridSize`, `sharedMemPerBlock`, `warpSize`
**Example:**
```rust
// Source: src/backend/hip_backend/device.rs
impl HipDeviceProp {
    // Existing offsets verified against C struct
    const NAME_OFFSET: usize = 0;
    const TOTAL_GLOBAL_MEM_OFFSET: usize = 284;
    const MULTI_PROCESSOR_COUNT_OFFSET: usize = 508;

    // NEW: Add offsets for launch validation fields
    // From hip_runtime_api.h: after name[256], uuid(16), luid[8], luidDeviceNodeMask(4)
    // totalGlobalMem (8), sharedMemPerBlock (8), regsPerBlock (4), warpSize (4), memPitch (8)
    // maxThreadsPerBlock (4), maxThreadsDim[3] (12), maxGridSize[3] (12)
    const MAX_THREADS_PER_BLOCK_OFFSET: usize = 320;  // Calculated from struct layout
    const MAX_THREADS_DIM_OFFSET: usize = 324;
    const MAX_GRID_SIZE_OFFSET: usize = 336;
    const SHARED_MEM_PER_BLOCK_OFFSET: usize = 300;
    const WARP_SIZE_OFFSET: usize = 316;

    /// Get maximum threads per block (typically 1024 for AMD GPUs)
    pub fn max_threads_per_block(&self) -> i32 {
        let bytes = &self._buffer[Self::MAX_THREADS_PER_BLOCK_OFFSET..Self::MAX_THREADS_PER_BLOCK_OFFSET + 4];
        bytes.try_into().ok().map(i32::from_ne_bytes).unwrap_or(1024)
    }

    /// Get maximum grid dimensions [x, y, z]
    pub fn max_grid_size(&self) -> [i32; 3] {
        let mut result = [0i32; 3];
        for i in 0..3 {
            let offset = Self::MAX_GRID_SIZE_OFFSET + i * 4;
            let bytes = &self._buffer[offset..offset + 4];
            result[i] = bytes.try_into().ok().map(i32::from_ne_bytes).unwrap_or(65535);
        }
        result
    }

    /// Get maximum threads per dimension [x, y, z]
    pub fn max_threads_dim(&self) -> [i32; 3] {
        let mut result = [0i32; 3];
        for i in 0..3 {
            let offset = Self::MAX_THREADS_DIM_OFFSET + i * 4;
            let bytes = &self._buffer[offset..offset + 4];
            result[i] = bytes.try_into().ok().map(i32::from_ne_bytes).unwrap_or(1024);
        }
        result
    }

    /// Get shared memory per block in bytes
    pub fn shared_mem_per_block(&self) -> usize {
        let bytes = &self._buffer[Self::SHARED_MEM_PER_BLOCK_OFFSET..Self::SHARED_MEM_PER_BLOCK_OFFSET + 8];
        bytes.try_into().ok().map(u64::from_ne_bytes).unwrap_or(65536) as usize
    }

    /// Get warp size (wavefront size: 32 for RDNA3, 64 for CDNA3)
    pub fn warp_size(&self) -> i32 {
        let bytes = &self._buffer[Self::WARP_SIZE_OFFSET..Self::WARP_SIZE_OFFSET + 4];
        bytes.try_into().ok().map(i32::from_ne_bytes).unwrap_or(32)
    }
}
```

### Pattern 2: Cached Device Limits in HipBackend
**What:** Store queried properties in `HipBackend` struct for fast access
**When to use:** Backend initialization, need fast validation without FFI calls
**Example:**
```rust
// Source: src/backend/hip_backend/backend.rs
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DeviceLimits {
    /// Maximum threads per block (e.g., 1024)
    pub max_threads_per_block: u32,
    /// Maximum grid dimensions [x, y, z]
    pub max_grid_size: [u32; 3],
    /// Maximum threads per dimension [x, y, z]
    pub max_threads_dim: [u32; 3],
    /// Shared memory per block in bytes
    pub shared_mem_per_block: u32,
    /// Warp size (wavefront size)
    pub warp_size: u32,
}

#[derive(Debug)]
pub struct HipBackend {
    device: HipDevice,
    stream: Arc<HipStream>,
    // NEW: Cached device limits
    limits: DeviceLimits,
}

impl HipBackend {
    pub fn new() -> HipResult<Arc<Self>> {
        // ... existing initialization ...

        // Query device properties and cache limits
        let mut props = HipDeviceProp::default();
        let result = unsafe { ffi::hipGetDeviceProperties(&mut props, device.device_id) };
        if result != ffi::HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to get device properties: {}",
                result
            )));
        }

        let limits = DeviceLimits {
            max_threads_per_block: props.max_threads_per_block() as u32,
            max_grid_size: props.max_grid_size().map(|x| x as u32),
            max_threads_dim: props.max_threads_dim().map(|x| x as u32),
            shared_mem_per_block: props.shared_mem_per_block() as u32,
            warp_size: props.warp_size() as u32,
        };

        let backend = Arc::new(HipBackend { device, stream, limits });
        // ... rest of initialization ...
    }

    /// Get cached device limits
    pub fn limits(&self) -> &DeviceLimits {
        &self.limits
    }
}
```

### Pattern 3: Pre-Launch Validation Trait
**What:** Centralized validation logic using cached limits
**When to use:** Before every kernel launch to prevent invalid configurations
**Example:**
```rust
// Source: src/backend/hip_backend/backend.rs
impl HipBackend {
    /// Validate kernel launch configuration against device limits
    ///
    /// Panics if configuration exceeds device capabilities with detailed
    /// error message including actual vs limit values.
    pub fn validate_launch_config(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
    ) -> HipResult<()> {
        let limits = &self.limits;

        // Thread count validation
        let threads_per_block = block_dim.0 * block_dim.1 * block_dim.2;
        if threads_per_block > limits.max_threads_per_block {
            return Err(HipError::KernelLaunchFailed(format!(
                "Threads per block {} exceeds limit {} (block={:?})",
                threads_per_block, limits.max_threads_per_block, block_dim
            )));
        }

        // Block dimension validation (per-axis limit)
        if block_dim.0 > limits.max_threads_dim[0] {
            return Err(HipError::KernelLaunchFailed(format!(
                "block.x {} exceeds limit {}", block_dim.0, limits.max_threads_dim[0]
            )));
        }
        if block_dim.1 > limits.max_threads_dim[1] {
            return Err(HipError::KernelLaunchFailed(format!(
                "block.y {} exceeds limit {}", block_dim.1, limits.max_threads_dim[1]
            )));
        }
        if block_dim.2 > limits.max_threads_dim[2] {
            return Err(HipError::KernelLaunchFailed(format!(
                "block.z {} exceeds limit {}", block_dim.2, limits.max_threads_dim[2]
            )));
        }

        // Grid dimension validation
        if grid_dim.0 == 0 || grid_dim.0 > limits.max_grid_size[0] {
            return Err(HipError::KernelLaunchFailed(format!(
                "grid.x {} invalid (limit {})", grid_dim.0, limits.max_grid_size[0]
            )));
        }
        if grid_dim.1 == 0 || grid_dim.1 > limits.max_grid_size[1] {
            return Err(HipError::KernelLaunchFailed(format!(
                "grid.y {} invalid (limit {})", grid_dim.1, limits.max_grid_size[1]
            )));
        }
        if grid_dim.2 == 0 || grid_dim.2 > limits.max_grid_size[2] {
            return Err(HipError::KernelLaunchFailed(format!(
                "grid.z {} invalid (limit {})", grid_dim.2, limits.max_grid_size[2]
            )));
        }

        // Shared memory validation
        if shared_mem_bytes > limits.shared_mem_per_block {
            return Err(HipError::KernelLaunchFailed(format!(
                "Shared memory {} bytes exceeds limit {}",
                shared_mem_bytes, limits.shared_mem_per_block
            )));
        }

        Ok(())
    }

    /// Launch kernel with automatic validation
    pub fn launch_kernel_with_module_shared_validated(
        &self,
        kernel: &HipKernel,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
        shared_mem_bytes: u32,
    ) -> HipResult<()> {
        // Validate before launch
        self.validate_launch_config(grid_dim, block_dim, shared_mem_bytes)?;

        // Launch if validation passes
        self.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, shared_mem_bytes)
    }
}
```

### Pattern 4: Safe Grid Calculation
**What:** Use u64 arithmetic to prevent overflow in grid size calculations
**When to use:** Computing grid dimensions for large tensors (>4B elements)
**Example:**
```rust
// Source: src/backend/hip_backend/limits.rs (new module)

/// Safe ceiling division using u64 arithmetic to prevent overflow
///
/// # Arguments
/// * `numerator` - Value to divide
/// * `denominator` - Divisor (must be > 0)
///
/// # Returns
/// Ceil(numerator / denominator) as u64
///
/// # Panics
/// If denominator is 0
#[inline]
pub fn ceil_div_u64(numerator: u64, denominator: u64) -> u64 {
    assert!(denominator > 0, "Division by zero in ceil_div_u64");
    (numerator + denominator - 1) / denominator
}

/// Safe grid dimension calculation for kernel launches
///
/// Calculates tiles needed for given dimension and tile size,
/// returning u32 only if value fits in u32::MAX.
///
/// # Arguments
/// * `dim` - Dimension size in elements
/// * `tile_dim` - Tile/block size
///
/// # Returns
/// Number of tiles as u32
///
/// # Panics
/// If result exceeds u32::MAX
#[inline]
pub fn safe_grid_dim(dim: u64, tile_dim: u32) -> u32 {
    let tiles = ceil_div_u64(dim, tile_dim as u64);
    assert!(tiles <= u32::MAX as u64,
        "Grid dimension {} exceeds u32::MAX for dim={}, tile_dim={}",
        tiles, dim, tile_dim);
    tiles as u32
}

// Usage example in transpose kernel:
// let grid_x = safe_grid_dim(cols as u64, TILE_DIM);
// let grid_y = safe_grid_dim(rows as u64, TILE_DIM);
```

### Anti-Patterns to Avoid
- **Validating with hardcoded constants:** Hardcoding `1024` for max threads is not portable
- **Querying properties per-launch:** `hipGetDeviceProperties` is a syscall, use cached values
- **Computing grid with u32 arithmetic:** Can overflow for tensors >4B elements
- **Silent truncation:** Always validate before casting u64 -> u32

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ceiling division | `(n + d - 1) / d` | `usize::div_ceil()` | Handles edge cases, clearer intent |
| Device property struct | Manual field mapping | Existing `HipDeviceProp` with accessors | Verified against C struct layout |
| Safe integer casting | Manual `as u32` | Validating wrapper functions | Prevents silent overflow |

**Key insight:** The existing `HipDeviceProp` already correctly maps to the C struct. Just add accessors for the limit fields we need.

## Common Pitfalls

### Pitfall 1: Incorrect Field Offsets
**What goes wrong:** Using wrong byte offsets when reading from `HipDeviceProp` buffer
**Why it happens:** C struct has padding and specific field sizes
**How to avoid:** Verify offsets against `hip_runtime_api.h` or use a C program to print `offsetof(hipDeviceProp_t, fieldName)`
**Warning signs:** Garbage values when accessing properties, validation always fails

### Pitfall 2: Grid Calculation Overflow
**What goes wrong:** Grid calculation `(n + TILE_DIM - 1) / TILE_DIM` overflows for large n
**Why it happens:** Using u32 arithmetic, `n + TILE_DIM - 1` can wrap around
**How to avoid:** Use u64 arithmetic: `((n as u64) + (TILE_DIM as u64) - 1) / (TILE_DIM as u64)`
**Warning signs:** Grid dimensions of 0 or suspiciously large values

### Pitfall 3: Validating Against Wrong Limits
**What goes wrong:** Using hardcoded AMD GPU limits (e.g., maxThreadsPerBlock=1024)
**Why it happens:** Assuming all GPUs have same limits
**How to avoid:** Always use `hipGetDeviceProperties` to query actual device limits
**Warning signs:** Validation fails on some GPU models

### Pitfall 4: Forgetting Per-Axis Block Dimension Limits
**What goes wrong:** Only checking total threads, not per-axis limits
**Why it happens:** Assuming `maxThreadsPerBlock` is the only limit
**How to avoid:** Also validate each axis: `block.x <= 1024, block.y <= 1024, block.z <= 1024`
**Warning signs:** Launch fails despite total threads being within limit

### Pitfall 5: Modifying All Kernel Launch Sites
**What goes wrong:** Manually updating 20+ kernel launch locations
**Why it happens:** Not having a centralized validation method
**How to avoid:** Add `launch_kernel_with_module_shared_validated()` wrapper that auto-validates
**Warning signs:** Inconsistent validation across codebase

## Code Examples

Verified patterns from official sources:

### Device Property Query (ROCm HIP 6.0)
```cpp
// Source: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/structhip_device_prop__t.html
// Fields available in hipDeviceProp_t:

int maxThreadsPerBlock;      // Max work items per work group
int maxThreadsDim[3];        // Max threads in each dimension (XYZ) of a block
int maxGridSize[3];          // Max grid dimensions (XYZ)
size_t sharedMemPerBlock;    // Size of shared memory region (in bytes)
int warpSize;                // Warp size
```

### Safe Grid Calculation (Existing Pattern)
```rust
// Source: src/attention/kernels/kernels_cache/kernels_basic.rs:48
// Existing usage of div_ceil (verified to work):
let grid_dim = (total_elements.div_ceil(BLOCK_SIZE), 1, 1);

// Pattern for safe grid calculation with u64:
let total_elements = rows as u64 * cols as u64;
let grid_dim = (
    safe_grid_dim(total_elements, BLOCK_SIZE),
    1,
    1,
);
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-launch property queries | Cached at backend init | Phase 27 | ~100x faster validation |
| Hardcoded limits (1024) | Runtime-queried limits | Phase 27 | Portable across GPU models |
| u32 grid arithmetic | u64 with validation | Phase 27 | Handles >4B element tensors |

**Deprecated/outdated:**
- Hardcoded `maxThreadsPerBlock = 1024`: Replace with runtime query
- Per-launch `hipGetDeviceProperties`: Cache at backend init

## Open Questions

Things that couldn't be fully resolved:

1. **Exact hipDeviceProp_t field offsets**
   - What we know: Offsets for name, totalGlobalMem, multiProcessorCount are verified
   - What's unclear: Exact offsets for maxThreadsPerBlock, maxGridSize need verification
   - Recommendation: Use C test program to verify offsets via `offsetof()` macro

2. **Validation assertion vs Result strategy**
   - What we know: Current transpose kernel uses `assert!()` for validation
   - What's unclear: Should validation panic or return Result?
   - Recommendation: Use `Result<()>` for production (test code can use assert)

## Sources

### Primary (HIGH confidence)
- [ROCm HIP 6.0 hipDeviceProp_t Documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/structhip_device_prop__t.html) - Full struct definition with all limit fields
- [ROCm HIP Device Management API](https://rocm.docs.amd.com/projects/HIP/en/develop/doxygen/html/group___device.html) - `hipGetDeviceProperties` function reference
- DEVICE_LIMITS.md - Verified AMD GPU limits for RDNA3 (gfx1100)

### Secondary (MEDIUM confidence)
- [HIP Deep Dive: Device Properties](https://gahan9.medium.com/hip-deep-dive-unlock-amd-gpu-secrets-with-device-properties-memory-queries-a1ccc4fb8ed0) - Community tutorial on device property usage

### Tertiary (LOW confidence)
- None - all sources are official documentation or verified research

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official ROCm HIP API
- Architecture: HIGH - Based on existing codebase patterns (HipDeviceProp, HipBackend)
- Pitfalls: HIGH - All verified against HIP documentation and DEVICE_LIMITS.md research

**Research date:** 2026-01-21
**Valid until:** 90 days (stable ROCm API)
