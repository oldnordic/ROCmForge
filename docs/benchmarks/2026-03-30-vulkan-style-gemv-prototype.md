# Benchmark: Vulkan-Style Multi-Row GEMV Prototype

**Date:** 2026-03-30
**Hardware:** AMD Radeon RX 7900 XT (gfx1100)
**ROCm Version:** 7.2.0
**Target Model:** Qwen2.5-0.5B-Instruct (Q4_0 layer)

## Overview

This experiment prototypes the "Multi-Row GEMV" pattern (NUM_ROWS > 1) from `llama.cpp`'s Vulkan/CUDA backends for the FFN-Down stage in `rocmforge`. The goal was to measure the impact of reusing the input vector from Shared Memory (LDS) across multiple output elements within a single wavefront.

## Prototype Features

1.  **Multi-Output Reuse:** Each wavefront processes **4 output columns** of the weight matrix.
2.  **Input Vector Reuse:** The input vector chunk is read from LDS once and used for 4 separate dot-products.
3.  **Vectorized Loads:** Uses `float4` for input vector access and `uint32_t` for quantized weights.
4.  **Loop Unrolling:** Inner loops are fully unrolled.

## Numerical Results

| Metric | Production GEMV Kernel | Vulkan-Style Multi-Row | Delta / Speedup |
|--------|------------------------|------------------------|-----------------|
| Latency (ms) | 0.0639 ms | 0.0401 ms | **1.59x Faster** |
| Max Abs Error | 0.000000 | 0.000000 | Identical |

## Analysis

The **1.59x speedup** confirms that LDS bandwidth and instruction overhead are significant bottlenecks in the decode path for RDNA3. 

While the speedup is less than the 3.3x seen in the `gate_up` prototype (which also fused and interleaved two projections), a ~60% improvement for a fundamental operation like GEMV is substantial. This optimization is particularly effective because it improves the arithmetic intensity of the kernel without increasing the amount of global memory that needs to be read (it just reads the same weights in a different block distribution).

## Implementation Recommendation

1.  **Integrate into FFN-Down:** This multi-row pattern should be the default for `ffn_down` in all quantized formats.
2.  **Apply to Attention Projections:** `attn_q`, `attn_k`, `attn_v`, and `attn_o` are all candidates for this same optimization.
3.  **Heuristic Tuning:** We used `NUM_ROWS=4` for this prototype. For very large models, `NUM_ROWS=2` might be safer for occupancy, while for very small models, `NUM_ROWS=8` could be even faster.

## Conclusion

Reusing LDS data across multiple output rows is a highly effective pattern for ROCm/HIP on RDNA3, matching the findings of the `llama.cpp` developers.
