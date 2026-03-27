# GPU Quantization Kernels - Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GPU quantization kernels for AMD ROCm to match CPU functionality, enabling full GPU inference for quantized LLM models.

**Architecture:** Layered design with llama.cpp HIP kernels wrapped in ROCmForge safety patterns, CMake build system, per-format module organization.

**Tech Stack:** ROCm/HIP, CMake, Rust FFI, llama.cpp kernel sources

---

## Context

ROCmForge currently has CPU quantization support for 8+ GGUF formats (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q4_1, Q5_0, Q3_K) but no GPU quantized GEMM. GPU kernels exist for basic ops (attention, RoPE, norm) but the heavy computation is still CPU-bound.

### Current State
- **GPU:** Basic kernels only (attention, RoPE, norm, elementwise)
- **CPU:** Full quantization support with SIMD optimizations
- **Build:** build.rs with hipcc for simple kernels
- **Testing:** Basic GPU tests, no comprehensive safety framework

### Target State
- **GPU:** Pure quantized GEMM for all formats (Q4_K×Q8_K, Q5_K×Q8_K, etc.)
- **Build:** CMake-based, professional GPU toolchain
- **Safety:** Multi-tier testing with VRAM monitoring, no crashes
- **Performance:** GPU-optimized memory layouts, native HIP optimizations

---

## Architecture

### Build System Migration

**Current state:** build.rs with direct hipcc compilation
**Target state:** CMake for quantization kernels (parallel with build.rs for existing kernels)

**Migration path:**
1. Keep build.rs for existing kernels (norm, rope, attention, elementwise)
2. Add CMakeLists.txt in `hip_kernels/quant/` for quantization kernels only
3. build.rs invokes CMake for quantization directory
4. Eventually migrate all kernels to CMake (future work, out of scope)

### Layered Architecture

```
Inference Layer
    ↓ (uses quantized weights via dispatch)
Rust Wrapper Layer (per-format)
    ↓ (safe wrappers, bounds checking, RAII)
CMake Build System (quantization only)
    ↓ (compiles HIP to static libs)
HIP Kernel Layer (per-format)
    ↓ (llama.cpp sources + ROCmForge safety)
ROCm HIP Runtime
```

### Component Structure

**1. HIP Kernel Layer** (`hip_kernels/quant/`)
- `common.hip` - Shared utilities, CHECK_HIP macro, device helpers
- `gemm_q4k_q8.hip` - Q4_K × Q8_K matrix multiplication
- `gemm_q5k_q8.hip` - Q5_K × Q8_K matrix multiplication
- `gemm_q6_k_q8.hip` - Q6_K × Q8_K matrix multiplication
- `dequant_q4k.hip` - Q4_K dequantization kernels
- `dequant_q5k.hip` - Q5_K dequantization kernels
- `embed.hip` - Embedding extraction kernels

**2. Rust Wrapper Layer** (`src/gpu/kernels/quant/`)
- `mod.rs` - Module exports, dispatch functions
- `gemm_q4k_q8.rs` - Q4_K GEMM wrapper + tests
- `gemm_q5k_q8.rs` - Q5_K GEMM wrapper + tests
- `gemm_q6_k_q8.rs` - Q6_K GEMM wrapper + tests
- `dequant.rs` - Dequantization kernel wrappers
- `embed.rs` - Embedding extraction wrappers

**3. Detection** (`src/gpu/detect.rs` - extended)
```rust
/// AMD GPU architecture identifiers for ROCm-Forge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchitecture {
    /// RDNA3 (RX 7000 series, W7900, 7900XTX)
    Gfx1100,
    /// RDNA2 (RX 6000 series)
    Gfx1030,
    /// CDNA2 (MI210)
    Gfx90a,
    /// CDNA1 (MI100)
    Gfx908,
    /// Vega (Vega 64, 56, 20, MI25)
    Gfx900,
    /// Unknown or unsupported architecture
    Unknown(u32),
}

impl GpuArchitecture {
    /// Maximum threads per block for this architecture
    pub fn max_threads_per_block(&self) -> u32 {
        match self {
            Self::Gfx1100 | Self::Gfx1030 => 1024,
            Self::Gfx90a | Self::Gfx908 => 1024,
            Self::Gfx900 => 1024,
            Self::Unknown(_) => 256, // Conservative default
        }
    }

    /// Warp size (wavefront size) for this architecture
    pub fn warp_size(&self) -> u32 {
        match self {
            Self::Gfx1100 | Self::Gfx1030 => 32, // RDNA
            Self::Gfx90a | Self::Gfx908 | Self::Gfx900 => 64, // CDNA/Vega
            Self::Unknown(_) => 32,
        }
    }

    /// Shared memory per block (bytes)
    pub fn shared_mem_per_block(&self) -> usize {
        match self {
            Self::Gfx1100 | Self::Gfx1030 => 64 * 1024,
            Self::Gfx90a | Self::Gfx908 => 64 * 1024,
            Self::Gfx900 => 64 * 1024,
            Self::Unknown(_) => 32 * 1024,
        }
    }

    /// Parse from device name string (e.g., "gfx1100")
    pub fn from_name(name: &str) -> Option<Self> {
        let arch_id = u32::from_str_radix(name.trim_start_matches("gfx"), 16).ok()?;
        Some(match arch_id {
            0x1100 => Self::Gfx1100,
            0x1030 => Self::Gfx1030,
            0x90a => Self::Gfx90a,
            0x908 => Self::Gfx908,
            0x900 => Self::Gfx900,
            id => Self::Unknown(id),
        })
    }
}

/// GPU capabilities detected at runtime
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub architecture: GpuArchitecture,
    pub total_memory: usize,      // Total VRAM in bytes
    pub free_memory: usize,       // Current free VRAM
    pub max_alloc_size: usize,    // Maximum single allocation
    pub compute_units: u32,       // Number of CUs
    pub max_clock: u32,           // Max clock in MHz
}
```

**4. Weights** (`src/gpu/weights.rs` - extended)
```rust
/// GPU-optimized memory layout for Q4_K blocks
///
/// Reorders CPU block format for coalesced GPU access:
/// - Group by 32-element chunks (matches warp size)
/// - Interleave scales with quantized data
/// - Pad to 256-byte alignment
pub fn q4k_cpu_to_gpu_layout(cpu_blocks: &[BlockQ4K]) -> Vec<u8> {
    // Each block: 256 weights, 144 bytes
    // GPU layout: [scales[16]][ql[128]][qh[32]][qsum[16]] (reordered)
    let mut gpu_data = Vec::with_capacity(cpu_blocks.len() * BlockQ4K::SIZE);

    for block in cpu_blocks {
        // Scales first (for faster access during dequantization)
        gpu_data.extend_from_slice(&block.scales);

        // Quantized low bits
        gpu_data.extend_from_slice(&block.ql);

        // Quantized high bits
        gpu_data.extend_from_slice(&block.qh);

        // Sums (for partial dequantization optimization)
        gpu_data.extend_from_slice(&block.scales);
    }

    gpu_data
}

/// GPU-optimized memory layout for Q5_K blocks
pub fn q5k_cpu_to_gpu_layout(cpu_blocks: &[BlockQ5K]) -> Vec<u8> {
    let mut gpu_data = Vec::with_capacity(cpu_blocks.len() * BlockQ5K::SIZE);

    for block in cpu_blocks {
        // Super-block scales (fp16) first
        gpu_data.extend_from_slice(&block.d);
        gpu_data.extend_from_slice(&block.dmin);

        // Packed scales (6-bit, need unpacking on GPU)
        gpu_data.extend_from_slice(&block.scales);

        // High bit mask (for 5-bit reconstruction)
        gpu_data.extend_from_slice(&block.qh);

        // Low 4 bits (packed 2 per byte)
        gpu_data.extend_from_slice(&block.ql);
    }

    gpu_data
}
```

**4. Weights** (`src/gpu/weights.rs` - extended)
- GPU-optimized layout conversion
- Block reordering for coalesced access
- Per-format conversion functions

---

## Data Flow

### Weight Loading Path

1. **GGUF file** → CPU weights (raw quantized bytes)
2. **detect()** → GpuCapabilities (architecture, memory limits)
3. **Model size check** → Fail-fast if not enough VRAM
4. **Layout conversion** → CPU block format → GPU-optimized format
5. **hipMalloc** → GPU memory allocation (checked against limits)
6. **hipMemcpy H2D** → Copy to GPU VRAM

### Inference Execution Path

1. **Token input** → embed kernel (dequantize + lookup)
2. **GEMM dispatch** → Route by weight type
3. **Wrapper function** → Validate parameters, bounds check
4. **HIP kernel launch** → Execute on GPU
5. **hipStreamSynchronize** → Check errors immediately
6. **Results** → Next layer or output token

### Error Handling Flow

```
Kernel launch failure
    ↓
Capture: hipError_t + kernel name + parameters
    ↓
Return GpuError::KernelLaunchFailed with details
    ↓
Propagate to main with detailed error message
    ↓
Fail-fast (no silent fallbacks, no hidden errors)
```

---

## Safety Strategy

### Pre-execution Safety

1. **Architecture Detection**
   - Query GPU properties dynamically
   - No hardcoded assumptions (e.g., gfx1100)
   - Detect limits (max threads, block size, shared memory)

2. **Memory Validation**
   - rocm-smi check before allocation
   - Test allocations capped at 10GB
   - Page boundary alignment for all allocations

3. **Kernel Parameter Validation**
   - CPU-side bounds checking before HIP launch
   - Tensor dimensions validated against GPU limits
   - Scale calculations checked for overflow

### Runtime Safety

4. **RAII Resource Management**
   - `GpuBuffer` auto-frees on drop (even during panic)
   - `GpuDevice` stream cleanup
   - No raw pointers in Rust API

5. **Error Propagation**
   - All HIP calls wrapped in CHECK_HIP macro
   - `GpuError<T>` for all failures
   - Detailed messages with context

6. **Synchronization Points**
   - `hipStreamSynchronize` after each kernel
   - Immediate error checking (no batching)

### Test Safety

7. **Three-Tier Testing**
   - **Sanity:** Detection works? Allocation possible?
   - **Unit:** Dequant math matches CPU? Scale unpacking correct?
   - **Integration:** Full kernel with real data produces correct results?

8. **Sequential Test Execution**
   - No parallel GPU tests (avoid driver contention)
   - VRAM cleanup after each test
   - rocm-smi verification between tests

9. **Process Isolation** (future)
   - GPU tests in separate executable
   - Can crash without affecting main process

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal:** Infrastructure for GPU quantization development

**Tasks:**
- Set up CMakeLists.txt for HIP kernel compilation
- Extend `GpuCapabilities` with architecture detection
- Add GPU memory management helpers (with rocm-smi integration)
- Create `hip_kernels/quant/common.hip` with safety macros
- Set up three-tier test framework (santy/unit/integration)
- Update `build.rs` to invoke CMake for quantization kernels

**Deliverables:**
- CMake build compiles a test HIP kernel
- GPU architecture detection working
- Test framework with sequential execution and VRAM cleanup

**Success Criteria:**
- CMake builds HIP kernels to static libraries
- Cargo links and runs test kernel successfully
- Tests run sequentially with VRAM monitoring

---

### Phase 2: Q4_K Foundation (Week 3-4)
**Goal:** First quantized GEMM working end-to-end

**Tasks:**
- Implement Q4_K block loading from GGUF
- Implement GPU-optimized layout conversion (reorder for coalesced access)
- Port `dequantize_q4k` kernel from llama.cpp
- Port `gemm_q4k_q8` kernel from llama.cpp
- Add ROCmForge safety wrappers (bounds checking, error handling)
- Rust wrappers: `src/gpu/kernels/quant/gemm_q4k_q8.rs`
- Integration: CPU dispatch calls GPU when available
- Full three-tier testing

**Deliverables:**
- `hip_kernels/quant/dequant_q4k.hip`
- `hip_kernels/quant/gemm_q4k_q8.hip`
- `src/gpu/kernels/quant/dequant.rs` (Q4_K)
- `src/gpu/kernels/quant/gemm_q4k_q8.rs`
- Working Q4_K GPU inference

**Success Criteria:**
- Q4_K model runs on GPU with correct results
- All three test tiers pass
- VRAM usage monitored and cleaned up
- No GPU crashes during testing

---

### Phase 3: Q5_K Implementation (Week 5-6)
**Goal:** Port Q5_K from CPU to GPU

**Tasks:**
- Port CPU Q5_K block structure to GPU kernel
- Implement Q5_K GPU-optimized layout
- Port `dequantize_q5k` kernel
- Port `gemm_q5k_q8` kernel
- Rust wrappers
- Integration testing with Q5_K model

**Deliverables:**
- `hip_kernels/quant/dequant_q5k.hip`
- `hip_kernels/quant/gemm_q5k_q8.hip`
- Rust wrappers for Q5_K
- Working Q5_K GPU inference

**Success Criteria:**
- Q5_K model runs on GPU correctly
- Tests pass
- Performance measured vs CPU

---

### Phase 4: Remaining Formats (Weeks 7-10)
**Goal:** Complete all quantization format support

**Order:** Q6_K → Q8_0 → Q4_0 → Q4_1 → Q5_0 → Q3_K

For each format:
- Port dequantization kernel
- Port GEMM kernel (with Q8_0)
- Rust wrappers
- Testing
- Integration

**Deliverables:**
- All quantization formats working on GPU
- Complete test coverage

**Success Criteria:**
- All CPU formats have GPU equivalents
- Full GPU inference pipeline working

---

### Phase 5: Optimization & Polish (Weeks 11-12)
**Goal:** Native HIP optimizations beyond hipify'd CUDA

**Tasks:**
- Profile kernels to find bottlenecks
- Implement native HIP optimizations (wavefront programming, LDS optimization)
- Add architecture-specific tuning (gfx1100 vs gfx1030)
- Performance benchmarking vs CPU
- Documentation

**Deliverables:**
- Optimized kernels beating CPU by target margin
- Performance benchmarks
- Complete documentation

**Performance Benchmarking Methodology:**
```rust
/// Benchmark CPU vs GPU for quantized GEMM
fn benchmark_q4k_gemm(m: usize, n: usize, k: usize) -> BenchmarkResults {
    // Create identical test data
    let (cpu_a, cpu_b) = create_cpu_test_data(m, n, k);
    let (gpu_a, gpu_b) = create_gpu_test_data(m, n, k);

    // Warm-up runs
    for _ in 0..5 {
        cpu_gemm_q4k_q8(&cpu_a, &cpu_b);
    }
    for _ in 0..5 {
        gpu::gemm_q4k_q8(&gpu_a, &gpu_b);
    }

    // Measure CPU
    let cpu_start = std::time::Instant::now();
    for _ in 0..100 {
        cpu_gemm_q4k_q8(&cpu_a, &cpu_b);
    }
    let cpu_time = cpu_start.elapsed() / 100;

    // Measure GPU (including host→device transfer)
    let gpu_start = std::time::Instant::now();
    for _ in 0..100 {
        gpu::gemm_q4k_q8(&gpu_a, &gpu_b);
        hipStreamSynchronize(stream);
    }
    let gpu_time = gpu_start.elapsed() / 100;

    BenchmarkResults {
        cpu_time,
        gpu_time,
        speedup: cpu_time.as_secs_f64() / gpu_time.as_secs_f64(),
    }
}
```

**Success Criteria:**
- GPU inference 2-5x faster than CPU
- Documentation complete
- Ready for production use

---

## Kernel Source Strategy

**Vendor Patch Approach:**

1. **Base Source:** `/home/feanor/Projects/llama.cpp`
   - Use HIP kernels as reference implementation
   - Kernels were hipify'd from CUDA (not native HIP)

2. **ROCmForge Fork:**
   - Maintain our branch with ROCmForge-specific modifications
   - Add safety wrappers (CHECK_HIP, bounds checking)
   - Optimize for native HIP features (wavefront, LDS)
   - Track upstream changes selectively

3. **Source File Mapping:**

| ROCmForge File | llama.cpp Source | Modifications |
|----------------|------------------|---------------|
| `hip_kernels/quant/common.hip` | `ggml/src/ggml-common.h` | Add CHECK_HIP macro, ROCmForge safety |
| `hip_kernels/quant/gemm_q4k_q8.hip` | `ggml/src/ggml-cuda/mmvq.cuh` | Port to HIP, add error handling |
| `hip_kernels/quant/gemm_q5k_q8.hip` | `ggml/src/ggml-cuda/mmvq.cuh` | Port Q5_K variant |
| `hip_kernels/quant/dequant_q4k.hip` | `ggml/src/ggml-cuda/dequantize.cuh` | Port to HIP, RAII integration |
| `hip_kernels/quant/embed.hip` | `ggml/src/ggml-cuda/dequantize.cuh` | Embedding lookup kernels |

4. **Attribution:**
   - Keep llama.cpp license headers
   - Document modifications in each file
   - Reference original kernel sources

---

## Memory Layout Strategy

**GPU-Optimized Layout:**

CPU block format is packed for storage, GPU needs coalesced access patterns.

**Conversion happens during weight loading:**
1. Read CPU block from GGUF
2. Reorder elements for GPU memory coalescing
3. Pad to alignment boundaries
4. Upload to GPU

**Per-format optimization:**
- Each format has different optimal layout
- Detection queries GPU architecture
- Layout selection based on GPU capabilities

---

## Build System

### CMake Integration

**File:** `hip_kernels/quant/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)
project(rocmerge_quant_kernels CXX HIP)

# Find ROCm
find_package(hip REQUIRED)

# Common library
add_library(quant_common STATIC
    common.hip
)

# Per-format libraries
add_library(gemm_q4k_q8 STATIC
    gemm_q4k_q8.hip
)
target_link_libraries(gemm_q4k_q8 quant_common hiprtc)

# ... repeat for each format
```

**Cargo Integration:**
- `build.rs` invokes CMake for quantization kernels
- Static libraries linked as native dependencies
- Feature flag `gpu` enables the entire stack

---

## Dispatch Integration

### CPU → GPU Dispatch Pattern

**Current CPU Dispatch** (in `src/cpu/ops.rs`):
```rust
pub fn matmul_quantized(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match (a.dtype(), b.dtype()) {
        (DataType::Q4_K, DataType::Q8_K) => {
            gemm_q4k_q8::gemm(a, b)
        }
        // ... other formats
    }
}
```

**New GPU-Aware Dispatch**:
```rust
pub fn matmul_quantized(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match (a.dtype(), b.dtype()) {
        (DataType::Q4_K, DataType::Q8_K) => {
            // Try GPU first if weights are on GPU
            if a.is_on_gpu() && b.is_on_gpu() {
                gpu::kernels::quant::gemm_q4k_q8(a, b)
            } else {
                gemm_q4k_q8::gemm(a, b)
            }
        }
        // ... other formats
    }
}
```

### Weight Loading with GPU Detection

```rust
impl Tensor {
    pub fn from_gguf_quantized(
        data: &[u8],
        dtype: DataType,
    ) -> Result<Self> {
        // 1. Detect GPU capabilities
        let caps = detect::gpu_capabilities()?;

        // 2. Check if model fits in VRAM
        let required_bytes = data.len();
        if caps.free_memory < required_bytes {
            return Ok(Tensor::cpu_quantized(data, dtype));
        }

        // 3. Convert to GPU layout and upload
        let gpu_data = match dtype {
            DataType::Q4_K => q4k_cpu_to_gpu_layout(data),
            DataType::Q5_K => q5k_cpu_to_gpu_layout(data),
            // ...
        };

        // 4. Allocate and upload to GPU
        let gpu_buffer = GpuBuffer::alloc_and_upload(&gpu_data)?;

        Ok(Tensor::gpu_quantized(gpu_buffer, dtype))
    }
}
```

---

## Testing Strategy

### Quantization-Specific Error Handling

```rust
/// GPU quantization-specific error types
#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("Block size mismatch: expected {expected}, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },

    #[error("Invalid quantization format: {0}")]
    InvalidFormat(String),

    #[error("Scale unpacking failed at index {index}")]
    ScaleUnpackFailed { index: usize },

    #[error("Layout conversion failed: {0}")]
    LayoutConversionFailed(String),

    #[error("GPU memory insufficient: need {needed}GB, have {available}GB")]
    InsufficientMemory { needed: f64, available: f64 },

    #[error("Architecture {arch:?} not supported for this operation")]
    UnsupportedArchitecture { arch: GpuArchitecture },

    #[error("GEMM dimension mismatch: M={m}, N={n}, K={k}")]
    GemmDimensionMismatch { m: usize, n: usize, k: usize },
}
```

### Three-Tier Testing

**Sanity Tests** (`tests/sanity_gpu_quant.rs`)
- Can we detect GPU? What architecture?
- Can we allocate memory? Does it free correctly?
- Does CMake build successfully?

**Unit Tests** (`tests/unit_gpu_quant.rs`)
- Does dequantization match CPU exactly?
- Are scale calculations correct?
- Do bounds checks work?

**Integration Tests** (`tests/integration_gpu_quant.rs`)
- Full kernel with real Q4_K model data
- Compare GPU vs CPU outputs (must match)
- Test with various tensor sizes

### Test Execution

**Test Utility Functions:**
```rust
/// Force GPU memory cleanup between tests
fn cleanup_gpu_memory() {
    // Drop all GPU buffers
    // Synchronize all streams
    // Call hipDeviceReset() to force cleanup
}

/// Verify VRAM state using rocm-smi
fn rocm_smi_verify() -> Result<(), GpuError> {
    let output = std::process::Command::new("rocm-smi")
        .args(&["showmem", "vram"])
        .output()?;

    // Parse output to verify VRAM is in expected state
    // Fail test if VRAM leak detected
    Ok(())
}

/// Check available VRAM before test
fn check_vram_available(required_gb: f64) -> Result<(), GpuError> {
    let output = std::process::Command::new("rocm-smi")
        .args(&["showmem", "vram", "--csv"])
        .output()?;

    // Parse CSV to get free VRAM
    // Return error if insufficient
    Ok(())
}
```

**Sequential Test Runner:**
```rust
#[test]
fn test_quantization_sequential() {
    // Run tests one at a time
    test_q4k_dequant_sanity();
    cleanup_gpu_memory();  // Wait for VRAM free
    rocm_smi_verify().unwrap();     // Check memory state

    test_q4k_gemm_unit();
    cleanup_gpu_memory();
    rocm_smi_verify().unwrap();

    test_q4k_full_integration();
    cleanup_gpu_memory();
    rocm_smi_verify().unwrap();
}
```

**VRAM Safety:**
- No test allocates > 10GB
- rocm-smi check before test
- rocm-smi verification after test
- Cleanup forced if verification fails

---

## File Structure Changes

### New Files

**HIP Kernels:**
- `hip_kernels/quant/CMakeLists.txt`
- `hip_kernels/quant/common.hip`
- `hip_kernels/quant/gemm_q4k_q8.hip`
- `hip_kernels/quant/gemm_q5k_q8.hip`
- `hip_kernels/quant/gemm_q6_k_q8.hip`
- `hip_kernels/quant/dequant_q4k.hip`
- `hip_kernels/quant/dequant_q5k.hip`
- `hip_kernels/quant/embed.hip`

**Rust Wrappers:**
- `src/gpu/kernels/quant/mod.rs`
- `src/gpu/kernels/quant/gemm_q4k_q8.rs`
- `src/gpu/kernels/quant/gemm_q5k_q8.rs`
- `src/gpu/kernels/quant/gemm_q6_k_q8.rs`
- `src/gpu/kernels/quant/dequant.rs`
- `src/gpu/kernels/quant/embed.rs`

**Detection:**
- Extend `src/gpu/detect.rs` with `GpuArchitecture`

**Weights:**
- Extend `src/gpu/weights.rs` with layout conversion

### Modified Files

- `build.rs` - Add CMake invocation for quantization kernels
- `src/gpu/kernels/mod.rs` - Export quantization kernels
- `src/gpu/detect.rs` - Add architecture detection
- `src/gpu/weights.rs` - Add layout conversion
- `src/cpu/ops.rs` - Add GPU dispatch in existing functions

### Removed Files

- `build.rs` kernel list (moved to CMake)
- Any dead code from refactors

---

## Dependencies

### External

- **ROCm SDK** - HIP compiler, runtime libraries
- **llama.cpp** - `/home/feanor/Projects/llama.cpp` (vendor fork)
- **rocm-smi** - VRAM monitoring (system tool)

### Internal

- Existing GPU infrastructure (`device.rs`, `error.rs`, `ffi.rs`)
- CPU quantization code (for reference testing)
- GGUF loader (for weight loading)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HIP kernel crashes GPU | System reset | Three-tier testing, process isolation, sequential tests |
| Memory layout bugs | Wrong results | Unit tests match CPU, extensive validation |
| CMake integration fragile | Build failures | Start simple, extensive testing, fallback to build.rs |
| Performance doesn't meet expectations | Slower than CPU | Profiling, native HIP optimizations in Phase 5 |
| Upstream llama.cpp changes | Merge conflicts | Vendor fork strategy, selective cherry-picking |

---

## Success Criteria

- [ ] All CPU quantization formats have GPU equivalents
- [ ] GPU inference 2-5x faster than CPU
- [ ] Zero GPU crashes during testing (sequential, monitored)
- [ ] All tests pass (sanity → unit → integration)
- [ ] VRAM properly managed (allocation, cleanup, monitoring)
- [ ] Code follows ROCmForge safety patterns (RAII, error handling)
- [ ] Documentation complete

---

## Open Questions

### Previously Unresolved (Now Addressed)

1. **Specific performance targets?** → **Resolved:** Target 2-5x speedup vs CPU, validated by `benchmark_q4k_gemm()` methodology
2. **CI/CD integration?** → **Resolved:** Out of Phase 1 scope; can be added in Phase 5 with process isolation for CI runners
3. **Multi-GPU support?** → **Resolved:** Out of scope for initial implementation; architecture designed for single-GPU, extensible later
4. **Fallback strategy?** → **Resolved:** `Tensor::from_gguf_quantized()` returns CPU tensor if VRAM insufficient; explicit fail-fast for kernel errors

### Remaining Questions

1. **CI/CD GPU runners?** Do we have GPU-enabled CI infrastructure or should this be tested locally only?
2. **Process isolation?** Should we implement the process-isolated test executable (Phase 1, point 9) immediately or defer?

---

## References

- llama.cpp: `/home/feanor/Projects/llama.cpp`
  - `ggml/src/ggml-cpu/arch/x86/quants.c` - CPU quantization reference
  - `ggml/src/ggml-cuda/` - CUDA kernels (convert to HIP)
- ROCm Documentation: AMD HIP programming guide
- Existing ROCmForge GPU code: `src/gpu/`

---

*Last Updated: 2025-03-27*
*Status: Ready for Review*
