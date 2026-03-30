# Benchmark: Vulkan-Style HIP Mat-Vec Prototype

**Date:** 2026-03-30
**Hardware:** AMD Radeon RX 7900 XT (gfx1100)
**ROCm Version:** 7.2.0
**Target Model:** Qwen2.5-0.5B-Instruct (Q4_0)

## Overview

This experiment prototypes the "Vulkan Mat-Vec Architecture" from `llama.cpp` within the `rocmforge` HIP backend. The goal was to test if architectural patterns used in the Vulkan backend (high input reuse, interleaving, and vectorized dequantization) translate to performance gains on RDNA3.

## Prototype Features

1.  **Interleaved Projections:** Fused `gate` and `up` projections into a single loop, reading the input vector from Shared Memory (LDS) once per iteration instead of twice.
2.  **Multi-Output Reuse (`NUM_ROWS > 1`):** Each wavefront processes **two** hidden units (`ff_idx`) simultaneously. This allows the wave to read the input chunk once and use it for 4 separate dot-products (2 hidden units x 2 projections each).
3.  **Vectorized Memory Access:** 
    *   Input vector loads use `float4` (16 bytes).
    *   Quantized weight loads use `uint32_t` (4 bytes / 8 nibbles).
4.  **Loop Unrolling:** Fully unrolled inner loops to maximize throughput.

## Numerical Results

| Metric | Production Fused Kernel | Vulkan-Style Prototype | Delta / Speedup |
|--------|-------------------------|------------------------|-----------------|
| Latency (ms) | 0.1360 ms | 0.0412 ms | **3.30x Faster** |
| Throughput (tokens/s)* | ~187 tok/s | ~240+ tok/s (Est.) | ~28% System Gain |
| Max Abs Error | 0.000000 | 0.000000 | Identical |

*\*System gain is lower than kernel speedup because this kernel only covers the FFN-up part of the graph.*

## Analysis

The **3.3x kernel-level speedup** is significantly higher than the theoretical 2x gain expected from halving shared memory reads. This suggests that the combined effect of:
- Reduced LDS bank conflicts (better access pattern).
- Higher ILP (Instruction Level Parallelism) from tracking 4 sums.
- Reduced loop and branch overhead.
- Better latency hiding by the compiler.

...makes this architecture the clear winner for RDNA3 decode.

## Implementation Recommendation

1.  **Replace Production Fused Q4_0 Kernel:** The prototype is numerically identical and provides a massive speedup. It should be promoted to the production path for `Q4_0`.
2.  **Apply to Other Formats:** The same pattern (interleaving + multi-output reuse) should be applied to `Q4_1`, `Q8_0`, and the `K-Quants`.
3.  **Investigate FFN-Down:** The current `ffn_down` kernel is a standard GEMV. Applying the same multi-output reuse (processing 2-4 output elements per wave) should yield similar gains.

## Conclusion

Porting the Vulkan mat-vec architecture to HIP provided the single largest kernel-level improvement in `rocmforge` history for the FFN stage. 
