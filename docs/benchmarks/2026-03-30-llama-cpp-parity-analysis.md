# Benchmark: Achieving llama.cpp Parity on RDNA3

**Date:** 2026-03-30
**Hardware:** AMD Radeon RX 7900 XT (gfx1100)
**Model:** Qwen2.5-0.5B-Instruct (Q4_0)

## Overview

We identified that `rocmforge` was significantly slower than `llama.cpp` (~69 tok/s vs 150+ tok/s) despite having fast kernels. This document details the architectural changes that closed the gap.

## Critical Improvements

### 1. Eliminating HIP Graph Overheads
**The Issue:** `rocprofv3` revealed that `hipGraphLaunch` was taking **11ms per token**. For a small model where kernels take < 3ms, the graph launch was the dominant bottleneck.
**The Fix:** Switched to direct standalone stream launches for the decode path.
**Impact:** Immediate jump from 69 to 154 tok/s.

### 2. Multi-Row Kernel Reuse (NUM_ROWS > 1)
**The Issue:** GEMV kernels were reading the input vector from Shared Memory (LDS) for every single output row.
**The Fix:** Ported the `llama.cpp` Vulkan pattern where each wavefront processes **4 output rows** simultaneously, reading the input once and using it 4 times.
**Impact:** 3.3x speedup for `gate_up` fusion and 1.59x for `ffn_down`.

### 3. Pinned Memory & Async Transfers
**The Issue:** Every token embedding upload and logits download was a synchronous, blocking `hipMemcpy`.
**The Fix:** Implemented `GpuPinnedBuffer` using `hipHostMalloc` and switched to `hipMemcpyAsync`.
**Impact:** Eliminated "Token Bubbles" between GPU execution and CPU sampling.

### 4. Hardware Shuffle Reductions
**The Issue:** RMS Norm used slow Shared Memory parallel reductions with multiple `__syncthreads()`.
**The Fix:** Implemented Wavefront Shuffles (`__shfl_down`) to perform reductions directly in registers.
**Impact:** 1.39x speedup for Norm operations.

## Results

| State | Throughput (Q4_0) | Latency / Token |
|-------|-------------------|-----------------|
| Baseline | 69.1 tok/s | 14.4 ms |
| Final Optimized | **162.9 tok/s** | **6.1 ms** |
| **Improvement** | **2.36x Faster** | **58% Reduced** |

## Conclusion

By adopting the architectural patterns used in `llama.cpp` (high input reuse, zero-LDS shuffles, and direct stream management), `rocmforge` now matches the performance of the industry standard for RDNA3 hardware.
