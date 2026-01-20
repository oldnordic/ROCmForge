# Fast GPU Matrix Multiplication on RDNA3 - Analysis

**Date**: 2026-01-20
**Source**: "Optimizing Matrix Multiplication on RDNA3: 50 TFlops and 60% Faster Than rocBLAS"
**URL**: https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
**Author**: Simon Boehm (seb-v)
**Hardware**: AMD Radeon 7900 XTX (RDNA3)
**Purpose**: Understanding RDNA3 optimization techniques for matrix operations

---

## Executive Summary

This article documents achieving **50 TFLOPS** on AMD Radeon 7900 XTX - **60% faster than rocBLAS** - through iterative HIP kernel optimization and direct ISA manipulation.

**Key Achievement**: 50× speedup from naive implementation (136 ms → 2.8 ms)

---

## Part 1: Hardware Specifications (7900 XTX)

### Theoretical Performance Limits

| Metric | Value |
|--------|-------|
| Frequency | 2500 MHz |
| WGPs | 48 |
| Compute Units (CUs) | 96 (48 × 2) |
| SIMDs | 192 (48 × 2 × 2) |
| FLOP/cycle/SIMD | 128 (32-wide × 2 VALU × 2 FMA) |
| **Theoretical FLOPS** | **61.44 TFLOP/s** |
| VRAM Bandwidth | 960 GB/s (384-bit GDDR6 @ 20 Gbps) |

### SIMD and Wave Architecture

```
WGP (WorkGroup Processor)
  └─ CU (Compute Unit) × 2
      └─ SIMD × 2
          └─ VALU × 2 (Vector Arithmetic Logic Units)
          └─ 32 threads per wave
```

- **SIMD**: Can manage up to 16 wavefronts in parallel
- **Wave**: 32 threads (RDNA3 equivalent of CUDA warp)
- **VALU**: Two 32-way units per SIMD for floating point

---

## Part 2: Eight Kernel Iterations

### Performance Summary

| Kernel | Description | Time (ms) | GFLOPS | vs rocBLAS |
|--------|-------------|-----------|---------|------------|
| 0 | rocBLAS reference | 4.50 | 30,547 | 100% |
| 1 | Naive implementation | 136.01 | 1,011 | 3.3% |
| 2 | LDS tiling | 34.21 | 4,018 | 13.1% |
| 3 | Register tiling | 6.03 | 22,777 | 74.6% |
| 4 | GMEM double buffering | 5.38 | 25,560 | 83.7% |
| 5 | LDS utilization optimization | 4.10 | 33,527 | **109.8%** |
| 6 | VALU optimization (ISA) | 3.64 | 37,791 | **123.7%** |
| 7 | Loop unrolling | 3.33 | 41,256 | **135.1%** |
| 8 | Batched GMEM loads | **2.80** | **49,047** | **160.6%** |

---

## Part 3: Kernel 1 - Naive Implementation

### Code

```cpp
__global__ void kernel1_naive(const float *A, const float *B, float *C,
                              int M, int K, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float acc_c = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            acc_c += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc_c + beta * C[row * N + col];
    }
}
```

### Issues

- Direct global memory access in inner loop (high latency)
- No data reuse
- No memory coalescing optimization
- **Result**: 136 ms, 1010 GFLOPS

---

## Part 4: Kernel 2 - LDS Tiling

### Key Concept

Load tiles of A and B into **Local Data Store (LDS)** - fast on-chip memory shared by workgroup.

### Memory Hierarchy

```
Global Memory (high latency, 200+ cycles)
    └─ LDS (low latency, ~90 cycles)
        └─ VGPR (register, 1 cycle)
```

### Tiling Strategy

```cpp
#define TILE_SIZE 32
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

for (int t = 0; t < N; t += TILE_SIZE)
{
    // Load tiles to LDS
    As[threadIdx.y][threadIdx.x] = A[N * row + t + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[N * (threadIdx.y + t) + col];
    __syncthreads();

    // Compute using LDS
    for (int k = 0; k < TILE_SIZE; k++)
    {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
}
```

### Result

- **34.2 ms** (4018 GFLOPS) - **4× faster** than naive
- VALU utilization only **15%** (LDS bottleneck)

---

## Part 5: Kernel 3 - Register Tiling

### Key Concept

Each thread computes a **4×4 tile** instead of single value. Increases arithmetic intensity.

### Tiling Hierarchy

```
Block Tile: 128×128
  └─ Wave Tile: 32×16 (per wave)
      └─ Thread Tile: 4×4 (per thread)
```

### Parameters

```cpp
constexpr int BN = 128;  // Block Tile N
constexpr int BM = 128;  // Block Tile M
constexpr int BK = 8;    // Batch size
constexpr int TN = 4;    // Thread Tile N
constexpr int TM = 4;    // Thread Tile M
```

### Result

- **6.03 ms** (22,777 GFLOPS) - **5× faster** than LDS tiling
- VALU utilization increased significantly

---

## Part 6: Kernel 4 - GMEM Double Buffering

### Problem

Waves wait for global memory, then LDS - no overlap.

### Solution

Use intermediate registers to prefetch global memory while computing on LDS.

### Issue Encountered

HIP compiler spilled to **scratch memory** due to high register usage.

### Fix

```cpp
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
```

### Result

- **5.38 ms** (25,560 GFLOPS) - 83.7% of rocBLAS

---

## Part 7: Kernel 5 - LDS Bank Conflicts

### Problem

Matrix A stored by columns in LDS causes **bank conflicts** - threads accessing same bank serializes.

### Solution

Add **padding of 4 elements** to LDS matrix:

```cpp
__shared__ float As[BK][BM + 4];  // +4 padding avoids bank conflicts
```

### Also: Enable CU Mode

```cpp
-mcumode  // Split LDS into upper/lower halves per SIMD pair
```

### Thread Tile Increased

- From 8×8 to **16×8** per thread
- Better computation-to-data-read ratio

### Result

- **4.10 ms** (33,527 GFLOPS) - **109.8% of rocBLAS** ✓
- **First kernel to exceed rocBLAS!**

---

## Part 8: Kernel 6 - ISA-Level Optimization

### Problem

HIP compiler doesn't generate optimal `v_dual_fmac` instructions.

### Solution: Direct ISA Manipulation

```bash
# Extract ISA from HIP
hipcc --genco --offload-arch=gfx1100 kernel5.cpp -mcumode --save-temps

# Assemble modified ISA
hipcc -target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcumode -c kernel_modified.s -o kernel.o
ld.lld -shared kernel.o -o kernel.hsaco
```

### VGPR Bank Optimization

Distribute registers across 4 banks to avoid conflicts:

```asm
v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185
v_dual_fmac_f32 v3, v186, v185 :: v_dual_fmac_f32 v4, v187, v184
```

**Key Principle**:
- SRCX0 and SRCY0 must use **different banks**
- Distribute A_col to banks 2-3, B_row to banks 0-1
- Sequential bank access pattern

### Result

- **3.64 ms** (37,791 GFLOPS) - **123.7% of rocBLAS**
- VALU utilization: **76.2%**

---

## Part 9: Kernel 7 - Loop Unrolling

### Approach

Duplicate inner loop 8×, remove branching.

### Result

- **3.33 ms** (41,256 GFLOPS) - **135.1% of rocBLAS**
- VALU utilization: **>80%**

---

## Part 10: Kernel 8 - Batched GMEM Loads

### Problem

128 lines of address calculation before global loads.
Each load depends on previous VALU operation.

### Solution: SGPR-Based Addressing

```asm
# Pre-compute addresses in SGPRs
s_add_u32 s24, s22, 0x0000
s_addc_u32 s25, s23, 0
s_add_u32 s26, s22, 0x4000
s_addc_u32 s27, s23, 0
# ... (16 addresses)

# Then simple loads
global_load_b32 v167, v203, s[24:25]
global_load_b32 v168, v203, s[26:27]
# ... (batch of 16 loads)
```

### Distribution Strategy

Split 16 loads across inner loop to avoid wave contention.

### Final Result

- **2.80 ms** (49,047 GFLOPS)
- **160.6% of rocBLAS** - **60% faster!**
- **50× faster than naive implementation**

---

## Part 11: Critical Optimizations Summary

### Memory Coalescing

**DO** - Contiguous access by wave:
```
Threads 0-31: Access bytes 0-127 (single transaction)
```

**DON'T** - Strided access:
```
Thread 0: byte 0
Thread 1: byte 4096
Thread 2: byte 8192
```

### LDS Bank Conflicts

```cpp
// BAD: Column-wise storage causes conflicts
for (int i = 0; i < 32; i++) {
    As[i][tid] = ...;  // All threads write to same bank
}

// GOOD: Row-wise with padding
As[tid][i] = ...;
__shared__ float As[32][32 + 4];  // +4 padding
```

### VGPR Bank Distribution

```asm
# Distribute A and B to different banks
# A: banks 2-3
# B: banks 0-1
ds_load_b64 v[186:187], v183  # A_col
ds_load_b64 v[184:185], v202  # B_row
```

### Dual-Issue Instructions

```asm
# Dual FMA: 2 operations per cycle
v_dual_fmac_f32 DSTX, SRCX0, SRCX1 :: v_dual_fmac_f32 DSTY, SRCY0, SRCY1
```

**Constraints**:
- Instructions must be independent
- SRCX0 and SRCY0 different VGPR banks
- One DST even, one DST odd
- SRC1 use different banks

---

## Part 12: Tools Used

### RGP (Radeon GPU Profiler)

**Key Views**:
- **Instruction Timing**: Shows latency per instruction
- **VALU Utilization**: Percentage of FLOP units used
- **Occupancy**: Wave vs SIMD capacity

### ISA Assembly

**Commands**:
```bash
hipcc --genco --offload-arch=gfx1100 kernel.cpp -mcumode --save-temps
```

**Output files**:
- `kernel-hip-amdgcn-amd-amdhsa-gfx1100.s` - ISA source
- `kernel-hip-amdgcn-amd-amdhsa-gfx1100.hsaco` - Binary

---

## Part 13: Key Takeaways for ROCmForge

### 1. Memory Coalescing Critical

ROCmForge must ensure contiguous memory access patterns in kernel launches.

### 2. LDS Padding Required

```rust
// Add padding to avoid bank conflicts
const LDS_TILE_SIZE: usize = TILE_SIZE + 4;
```

### 3. Register Tiling Important

Compute multiple outputs per thread to increase arithmetic intensity.

### 4. CU Mode for RDNA3

```rust
// Enable CU mode to split LDS
// -mcumode flag in compilation
```

### 5. Double Buffering Helps

Prefetch next tile while computing current tile.

### 6. ISA-Level Optimization May Be Needed

HIP compiler not always optimal - direct ISA manipulation sometimes required.

---

## Part 14: Performance Gains Breakdown

| Optimization | Speedup | Cumulative |
|---------------|---------|------------|
| Baseline (naive) | 1.0× | 1.0× |
| + LDS tiling | 4.0× | 4.0× |
| + Register tiling | 5.7× | 22.7× |
| + Double buffering | 1.1× | 25.3× |
| + LDS optimization | 1.3× | 33.6× |
| + ISA optimization | 1.1× | 37.4× |
| + Loop unrolling | 1.1× | 40.9× |
| + Batched GMEM | 1.2× | **48.6×** |

---

## Part 15: RDNA3-Specific Considerations

### Architecture Differences vs CDNA

| Feature | RDNA3 | CDNA |
|---------|-------|------|
| Market | Consumer/Radeon | Datacenter/Instinct |
| LDS Mode | WGP (shared) or CU (split) | CU mode preferred |
| Matrix Cores | None | MFMA for FP4/FP6 |
| Target | Gaming, graphics | AI/HPC |

### Environmental Variables

```bash
HIP_FORCE_DEV_KERNARG=1
HSA_NO_SCRATCH_RECLAIM=1
```

### Compilation Flags

```bash
--offload-arch=gfx1100  # RDNA3
-mcumode                 # CU mode
--amdgpu-target=gfx90a;gfx942  # For MI300
```

---

## Conclusion

Achieving 60% better performance than rocBLAS required:
1. Understanding RDNA3 architecture deeply
2. Iterative optimization through 8 kernels
3. Direct ISA manipulation when HIP compiler failed
4. Careful attention to memory access patterns
5. VGPR bank distribution optimization

**Final achievement**: **49 TFLOPS** out of **61 TFLOPS theoretical** = 80% efficiency

This demonstrates that with careful optimization, RDNA3 can exceed vendor libraries for specific workloads.

---

**End of Report** - Investigation only, no code changes.
