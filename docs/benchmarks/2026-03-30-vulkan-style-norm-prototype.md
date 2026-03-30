# Benchmark: Vulkan-Style RMS Norm Prototype

**Date:** 2026-03-30
**Hardware:** AMD Radeon RX 7900 XT (gfx1100)
**ROCm Version:** 7.2.0
**Hidden Size:** 4096

## Overview

This experiment prototypes a "Vulkan-Style" RMS Norm kernel using architectural patterns from `llama.cpp`. The primary optimization is replacing Shared Memory (LDS) parallel reductions with high-performance **Wavefront Shuffles** (`__shfl_down`) and using **Vectorized Loads** (`float4`).

## Prototype Features

1.  **Wavefront Shuffles:** Reduces the partial sums within a wavefront using hardware shuffles, avoiding LDS bank conflicts and synchronization barriers.
2.  **Minimal Synchronization:** Only one `__syncthreads()` is used for the final cross-wave reduction, compared to multiple barriers in the production version.
3.  **Vectorized bandwidth:** Reads input and weight tensors using `float4` (16-byte) loads.
4.  **Zero LDS for small blocks:** For blocks of 32 threads or fewer, the kernel uses zero LDS.

## Numerical Results

| Metric | Production RMS Norm | Vulkan-Style (Shuffles) | Delta / Speedup |
|--------|---------------------|--------------------------|-----------------|
| Latency (ms) | 0.0292 ms | 0.0211 ms | **1.39x Faster** |
| Max Abs Error | 0.000000 | 0.00000012 | Negligible |

## Analysis

The **1.39x speedup** is significant for an operation that is already extremely fast. The performance gain comes from:
- **Reduced Latency:** Shuffles are faster than LDS round-trips.
- **Improved Occupancy:** By reducing LDS usage from `block_size * 4` bytes to a static `32 * 4` bytes (and 0 for small blocks), more wavefronts can be scheduled on the same Compute Unit.
- **Instruction Efficiency:** Vectorized loads reduce the number of instructions executed.

## Implementation Recommendation

1.  **Replace Production RMS Norm:** The shuffle-based approach is strictly better for RDNA3 hardware.
2.  **Apply to Softmax:** The same wavefront shuffle pattern should be applied to the `soft_max` kernel to achieve similar gains.
3.  **Keep Scalar Fallback:** Maintain the scalar fallback for cases where the hidden size is not a multiple of 4.

## Conclusion

Avoiding LDS in favor of wavefront shuffles is a key performance pattern for RDNA3, as demonstrated by the `llama.cpp` Vulkan architecture.
