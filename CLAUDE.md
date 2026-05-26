# ROCmForge - AMD GPU LLM Inference Engine

High-performance LLM inference on AMD GPUs using pure HIP/ROCm. Q4_0 decode optimization for Qwen2.5 on RDNA2/3.

Reference: llama.cpp (quantization), hipfire (HIP optimization techniques).

## Build & Test

```
cargo check --features gpu
cargo test --features gpu -- --test-threads=1
cargo bench --features gpu
cargo clippy --all-targets --all-features
```

## Project Structure

- `src/gpu/` — GPU inference path (HIP)
  - `kernels/quant/` — Modularized quant kernels (q4_0, q4_1, q4_k, q5_k, q6_k, q8_0, legacy)
  - `ops.rs` — High-level GPU operations
  - `forward.rs` — Layer forward pass
  - `features.rs` — GPU architecture detection (RDNA1/2/3)
- `src/cpu/` — CPU fallback implementation
- `hip_kernels/quant/` — HIP kernel sources (.hip files)
- `tests/`, `benches/` — Correctness tests and benchmarks

## Hardware

- Dev machine: AMD RX 7900 XT (gfx1100, RDNA3), ROCm 7.2.0, clang 18
- Validation models: `~/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`, `Qwen2.5-7B-Instruct-Q4_0-Pure.gguf`
- Measured: ~106 tok/s decode (Qwen2.5-7B Q4_0, graph + Q8 fastpath)

## Rules

- NO Python in hot path — GPU kernels must be pure HIP
- NO CUDA/cross-platform abstractions — AMD only
- NO stubs/placeholders — AGENTS.md ZERO TOLERANCE policy applies
- Measure before optimizing — use rocprofv3, Criterion
- Test on real hardware — all GPU code runs on actual AMD GPUs
- Use `--test-threads=1` for GPU tests (serialize to avoid GPU lock contention)
- Profile: `./.rocprofv3/profile_decode.sh runtime` or manual rocprofv3 invocation

## GPU Safety Protocol

Real-model prefill caused an `amdgpu` page fault and full GPU reset. New GPU code can reset the desktop until proven otherwise.

Before ANY new GPU code:
1. Acquire cross-process GPU lock (`ROCMFORGE_GPU_LOCK_TIMEOUT=30`)
2. Run staged preflight: driver present → HIP sees device → memory round-trip → trivial kernel
3. Use timeout-wrapped subprocess for CLI GPU execution
4. Deterministic quality checks before speed benchmarks
5. Real-model profiling LAST

Forbidden without safety harness:
- Direct `./target/release/rocmforge --gpu ...` runs
- New GPU benchmarks on untested codepaths

## Key Files

- `AGENTS.md` — Subagent quality standards (READ FIRST)
- `CHANGELOG.md` — Detailed commit history
- External refs: `/home/feanor/Projects/llama.cpp`, `/home/feanor/Projects/rocm-examples/`

See also: `~/Projects/CLAUDE.md` for shared agent workflow and coding standards.
