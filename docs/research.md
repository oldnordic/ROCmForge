# Research Notes

Date: 2026-03-30

This document captures the current research direction for GPU decode performance in `rocmforge`, combining:

- AMD ROCm/HIP 7.2 guidance
- local `rocprofv3` and benchmark findings
- local `llama.cpp` source inspection
- the isolated FFN experiment added for this repo

This complements [amd-rocm-7.2-findings.md](./amd-rocm-7.2-findings.md). That file is AMD-focused. This file records the practical conclusions for `rocmforge`.

## Current local conclusions

- The decode bottleneck on the 0.5B Q4_0 regression model is still mostly FFN, not attention-first.
- `gate_up` and `ffn_down` remain the most important decode hotspots.
- The most reliable gains so far came from:
  - **Metadata-driven LM-head specialization:** Improved `Qwen2.5-0.5B Q4_0` from `117 tok/s` to `187.5 tok/s`.
  - **4-wave fast-path for small hidden sizes:** Improved graph-backed decode from `168 tok/s` to `227 tok/s`.
  - **Explicit-stream graph cleanup:** Provided a cleaner base for HIP graph replay.
  - **Reducing bytes moved in decode:** Critical for RDNA3 performance.

## Quantization Accuracy Research

Recent implementation of quantization formats on GPU shows the following accuracy vs CPU references:

- **Q4_0 (Uniform):** < 1% relative error (18-byte blocks).
- **Q4_1 (Affine):** < 1% relative error (20-byte blocks).
- **Q5_K (Non-uniform sub-blocks):** 0.005% relative error (176-byte blocks for 256 elements).
- **Q4_K (2-phase packing):** 0.35% relative error (matches llama.cpp uniform format).

## What AMD guidance changes in practice

AMD’s ROCm/HIP 7.2 guidance points to a stricter workflow:

1. profile first with `rocprofv3` (Trace mode is stable; PMC mode is currently unstable on this machine)
2. classify the hotspot (compute-bound vs memory-bound)
3. reduce bytes moved and improve reuse
4. only then tune launch geometry

The key AMD lessons that matter here are:

- arithmetic intensity matters
- bytes moved per token matter
- coalesced Wave32 access matters on RDNA3
- LDS reuse matters when it actually removes global-memory traffic
- occupancy must be considered together with register pressure and LDS pressure
- HIP graphs are a good fit for repetitive decode workloads

## What `llama.cpp` is actually doing

The useful `llama.cpp` ideas are not in `qwen2.cpp` itself. The model-side code is generic.

- Qwen2 uses the generic `build_ffn()` path in the local `llama.cpp` checkout at `/home/feanor/Projects/llama.cpp/src/llama-graph.cpp`
- For parallel gate/up FFN, `llama.cpp` emits `ggml_swiglu_split()` in its graph builder.
- The Vulkan backend has dedicated SwiGLU pipelines and multiple matmul/mat-vec families.
- Decode-shaped work is dispatched by runtime shape, not by model name.

Important local `llama.cpp` observations:

- `build_ffn()` makes the FFN activation boundary explicit.
- The backend has dedicated `pipeline_swiglu`.
- Vulkan switches to a mat-vec path when the output shape is decode-like (`dst->ne[1] == 1`).

Portable lesson for `rocmforge`:

- Make FFN structure visible to the backend.
- Keep shape-driven dispatch.
- Do not hide FFN behind only generic GEMV calls if a decode-specialized path is needed.

## FFN experiment added to this repo

An isolated FFN experiment was added without touching the working inference path.

Files:
- `hip_kernels/quant/q4_1_ffn_experimental.hip`
- `src/gpu/kernels/quant.rs`
- `tests/gpu_ffn_experimental.rs`

The experiment:
- Uses real model weights from layer 0.
- Compares:
  - **Baseline:** Materialize `swiglu`, then run `gemv_q4_1`.
  - **Experimental:** Fuse `silu(gate) * up` directly into the Q4_1 down projection.

Result on the local machine:
- **Baseline:** `0.0234 ms`
- **Experimental:** `0.0366 ms`
- **Speedup:** `0.640x` (Slower)

Conclusion:
- Naive direct fusion alone is not enough and can be slower.
- Improvement must come from better data movement or occupancy, not just "fusing more".

## Recommended next steps

1. Keep using the isolated FFN experiment as a sandbox instead of touching production decode.
2. Add more decode-specialized FFN-down variants behind test-only wrappers:
   - Small-output shape family.
   - Multi-column small-batch family.
   - LDS-tiled variant that avoids recomputing activation too aggressively.
3. Compare each variant against:
   - The current staged GPU path.
   - CPU reference numerics.
   - `rocprofv3` trace timing.
4. Only promote a variant into production if it wins consistently on the real-model harness.

## Bottom line

`llama.cpp` does help, but the useful lesson is architectural, not model-specific:
- Explicit FFN graph structure.
- Shape-driven backend selection.
- Dedicated decode mat-vec style paths.
