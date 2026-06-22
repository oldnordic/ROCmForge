rocmforge - LLM inference on AMD GPUs (HIP) with a CPU fallback path.

## Current status

- The project currently works for local Qwen GGUF runs, Gemma4 models, and custom `.rfm` model architectures.
- The GPU path is AMD-only and pure HIP. There is no CUDA or cross-vendor backend here.
- Recent optimization passes introduced features: **GPU Decode Graph**, **F16 Embedding Support**, **TurboQuant KV Cache**, and **Pareto Synthesis** (VideoMLA/AdaState/MPO + GPU SVD + MoE VRAM optimizations).
- Measured token generation speed for a 0.5B model (e.g., Qwen2.5-0.5B Q8_0) reaches **~340 tok/s** on an AMD Radeon RX 7900 XT via the pre-compiled GPU Decode Graph path.
- Gemma4 12B Q4_0 achieves **~24 tok/s** decode on AMD RX 7900 XT with hybrid attention support.
- Robust runtime safety rules dynamically select the fastest inference path based on model-profile routing.

### Key Features & Optimizations

*   **GPU Decode Graph**: Captures the entire token generation execution sequence (including QKV projection, split, norms, attention, activations, and projection kernels) into a single HIP graph to bypass runtime host-to-device kernel launch overhead.
*   **F16 Embedding Support**: High-performance half-precision (FP16) input embeddings.
*   **TurboQuant KV Cache**: Ultra-low latency Key-Value cache management with automatic bounds and deadlock resolution.
*   **Pareto Synthesis**: Hardware-aware combined optimization path featuring GPU SVD, MPO kernel compression, and mixture of experts (MoE) VRAM paging.
*   **Inference Path Router**: Dynamically matches model traits to targeted hardware-optimized HIP fastpaths.

## Technical Architecture

**GPU compute path**
- HIP / ROCm kernels, AMD-only — RDNA2 DP4A and RDNA3 WMMA matrix instructions
- GPU Decode Graph — the entire token-generation sequence (QKV projection, split, norms, attention, activations, output projection) captured into a single HIP graph to eliminate per-launch host→device overhead
- F16 (half-precision) input embeddings
- TurboQuant KV cache — low-latency paged KV management with automatic bounds and deadlock resolution

**Pareto Synthesis** — hardware-aware combined optimization path
- GPU SVD correction kernels for attention
- MPO (Matrix Product Operator) kernel compression
- Mixture-of-Experts (MoE) VRAM paging
- VideoMLA / AdaState routing variants

**Quantization kernels (GPU)** — Q4_0, Q4_K, Q6_K, Q8_0 (Q5_0 not implemented on GPU)

**Inference Path Router** — model-profile-driven selection across `BatchedPrefill`, `DecodeStyle`, `SvdOptimized`, and `CpuFallback`

**CPU fallback** — pure-Rust decode path when no GPU is available or the profile is unsafe

**Model formats** — GGUF (Qwen2.5, Gemma4) plus custom `.rfm` architectures; speculative decoding via `--draft-model`

**Keyword index:** AMD GPU inference · HIP · ROCm · LLM inference · GGUF · GPU decode graph · DP4A · WMMA · RDNA2 · RDNA3 · F16 embeddings · TurboQuant KV cache · Pareto Synthesis · GPU SVD · MPO compression · mixture of experts · MoE VRAM paging · Q4_0 · Q4_K · Q6_K · Q8_0 · inference path router · speculative decoding · CPU fallback · Rust

### Supported quantizations (GPU)

| Quantization | Status | Notes |
|--------------|--------|-------|
| Q4_0 | Supported | Optimized with DP4A (RDNA2) and WMMA (RDNA3) |
| Q4_K | Supported | Mixed quantization |
| Q6_K | Supported | Works with graph capture |
| Q8_0 | Supported | 8-bit quantization |

Note: Q5_0 is not currently implemented on GPU.

## Requirements

- Rust 1.81+
- ROCm/HIP toolkit (tested locally on ROCm 7.2)
- ROCm runtime libraries visible at execution time, for example via your ROCm environment setup so `libamdhip64.so.7` resolves
- A GGUF model file
- Current development and validation focus: Qwen2.5 GGUF models

## Build

```bash
cargo build --release
cargo build --release --features gpu
```

## Run

GPU:

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Hello" \
  --gpu
```

CPU fallback:

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Hello"
```

Supported CLI options (from `--help`):

| Option | Description |
|---|---|
| `--model <path>` | GGUF or RFM model path |
| `--prompt <text>` | Prompt text |
| `--gpu` | Use GPU backend |
| `--max-tokens N` | Max generated tokens (default: 256) |
| `--temperature F` | Sampling temperature (default: 1.0) |
| `--top-p F` | Nucleus sampling threshold (default: 0.9) |
| `--no-template` | Disable chat template |
| `--list-tensors` | Print tensors and exit |
| `--debug` | Print debug logits info |
| `--kv-dump <path>` | Dump post-prefill KV cache to binary file (research tool) |
| `--prefill-only-validate` | Run prefill only, exit 0 on valid logits, 1 on NaN/Inf |
| `--draft-model <path>` | Draft model path for speculative decoding |
| `--speculative-tokens N` | Number of speculative tokens per step (default: 4) |

Note: `--device` is not a valid flag in the current binary.

## Inference Path Router

ROCmForge uses a model-profile-driven router to select the optimal inference path:

```
[Router] Model profile: arch=qwen2, quant=Q4_0
[Router] Selected path: BatchedPrefill(max_seq=512)
```

The router inspects loaded model metadata (quantization type, SVD/MPO/sparse flags, architecture) and selects from:

| Path | When Selected | Description |
|------|--------------|-------------|
| `BatchedPrefill` | Q4_0 model, prompt 2-512 tokens | Fastest path, processes all prompt tokens in one kernel launch |
| `DecodeStyle` | Mixed quant, single token, or unsafe model | Token-by-token processing, universal fallback |
| `SvdOptimized` | SVD model + experimental flag enabled | Uses SVD correction kernels for attention |
| `CpuFallback` | Incompatible or unsafe configuration | Falls back to CPU inference |

Safety rules:
- Sparse/MPO models always route to `DecodeStyle` (experimental kernels are opt-in)
- MoE/SSM models route to `DecodeStyle` (no batched kernels yet)
- SVD models only use `SvdOptimized` when `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1`

Engineering note:
- Inference-path, converter, router, or kernel-dispatch changes should follow the repo checklist in `docs/inference-change-checklist.md` so GGUF and `.rfm` paths are audited together.

## Runtime safety controls

- `ROCMFORGE_GPU_SAFE_MODE=1`
  - Forces conservative mode for this process.
  - Disables decode graph and experimental fastpaths.

- `ROCMFORGE_ENABLE_DECODE_GRAPH=1`
  - Enables decode graph replay.

- `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1`
  - Enables the Q8 activation fastpath used in decode kernels.

- `ROCMFORGE_DESKTOP_VRAM_GB=<float>`
  - Configures VRAM reserved for desktop/compositor (default: 4.0).
  - Lower for single-monitor setups (2.0), higher for multi-monitor 4K (6.0+).
  - Prevents inference from stealing memory needed by the display.

- `ROCMFORGE_GPU_LOCK_TIMEOUT=<seconds>`
  - Configures how long GPU tests wait for the shared cross-process GPU lock.
  - Cargo-launched tests default to `30`, so separate `cargo test` runs queue instead of colliding.

- `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1`
  - Enables sparse CSR and MPO kernels (potentially unsafe on display-attached GPUs).
  - Only use when testing compressed `.rfm` models with sparse/MPO weights.

- `ROCMFORGE_Q4_0_Q8_DP4A=1`
  - Enables the DP4A int8 dot-product path for Q4_0 × Q8_0 kernels (GEMV, gate-up, SwiGLU) on supported GPUs.
  - Software fallback is used automatically on architectures without DP4A.

- `ROCMFORGE_Q4_0_Q8_SINGLE_ROW=1`
  - Forces the single-row high-occupancy launch configuration for Q4_0 × Q8_0 kernels.
  - Useful on RDNA3 wave32 hardware when occupancy matters more than column parallelism.

- `ROCMFORGE_OBSERVE_DECODE_GRAPH_HEALTH=1`
  - Enables atomic telemetry for HIP decode graph capture/replay/cache/fallback events.
  - Query the live snapshot via `GpuDecodeGraphHealthSnapshot` in `src/gpu/decode_graph_health.rs`.

Conservative run:

```bash
ROCMFORGE_GPU_SAFE_MODE=1 ./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu
```

Tuned run:

```bash
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu
```

7B tuned example used during current development:

```bash
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge \
  --gpu \
  --model /home/feanor/Projects/Memoria/models/Qwen2.5-7B-Instruct-Q4_0-Pure.gguf \
  --prompt Hello --no-template --top-p 1.0 --temperature 0.0 --max-tokens 64
```

## Known limitations

- **Gemma4 E2B GPU crash:** Gemma4 E2B (2.3B effective parameters, hybrid attention with mixed head dimensions) loads successfully with optimized PLE VRAM allocation (4.3GB, 35× improvement) but crashes during decode with GPU memory access faults. Root cause: flash attention kernel assumes uniform cache stride incompatible with hybrid attention's mixed head_dim (256/512 across layers). Use Gemma4 12B or CPU fallback for E2B models until specialized kernel support is implemented.

## Measured results (local)

Machine path references in this section are from local runs on April 10, 2026.

1) Qwen2.5-0.5B-Instruct Q4_0 (graph path, benchmark harness)

```bash
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
ROCMFORGE_BENCH_RUNS=10 ROCMFORGE_BENCH_WARMUP=1 ROCMFORGE_BENCH_TOKENS=128 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```

`cargo test` already inherits `RUST_TEST_THREADS=1`, `ROCMFORGE_GPU_LOCK_TIMEOUT=30`, and `ROCMFORGE_DESKTOP_VRAM_GB=4.0` from [`.cargo/config.toml`](/home/feanor/Projects/rocmforge/.cargo/config.toml). The explicit `--test-threads=1` in examples is retained for clarity, but sequential execution and conservative VRAM gating are the default.

- Prefill average: `408.7 tok/s`
- Decode average: `526.8 tok/s`

2) Qwen2.5-7B-Instruct Q4_0 (`Qwen2.5-7B-Instruct-Q4_0-Pure.gguf`, CLI, 3 runs)

```bash
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge \
  --gpu \
  --model /home/feanor/Projects/Memoria/models/Qwen2.5-7B-Instruct-Q4_0-Pure.gguf \
  --prompt Hello --no-template --top-p 1.0 --temperature 0.0 --max-tokens 64
```

- Prefill: `31.5 / 32.4 / 32.0 tok/s` (avg `32.0 tok/s`)
- Decode: `106.7 / 106.7 / 106.5 tok/s` (avg `106.6 tok/s`)

3) Qwen2.5-0.5B-Instruct Q4_0 with decode graph disabled (for comparison)

```bash
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
ROCMFORGE_DISABLE_DECODE_GRAPH=1 \
ROCMFORGE_BENCH_RUNS=5 ROCMFORGE_BENCH_WARMUP=1 ROCMFORGE_BENCH_TOKENS=128 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```

- Decode average: `486.0 tok/s`

4) Qwen2.5-0.5B-Instruct Q8_0 (`qwen2.5-0.5b-instruct-q8_0.rfm` / `.gguf`, CLI, greedy `--top-p 1.0`)

```bash
./target/release/rocmforge \
  --model /path/to/qwen2.5-0.5b-instruct-q8_0.rfm \
  --prompt "Hello" \
  --gpu \
  --top-p 1.0 \
  --max-tokens 128
```

- Prefill speed: `151.8 tok/s`
- Decode speed (baseline / graph capture disabled or invalidated): `87.7 tok/s`
- Decode speed (optimized / graph capture active): `340.4 tok/s`
- Reference Native `llama.cpp` speed: `239.2 tok/s` (ROCmForge is ~42% faster with decode graph replay active)

## GPU Architecture Optimizations

ROCmForge automatically detects your GPU architecture at runtime to enable hardware-specific optimizations:

### Architecture Detection

The `GpuFeatures` module (`src/gpu/features.rs`) detects:

- **GPU Architecture**: Maps device names to architecture strings (gfx1010/gfx1030/gfx1100)
- **DP4A Support**: `v_dot4_i32_i8` instruction for 4-way int8 multiply-accumulate (RDNA2+)
- **WMMA Support**: Wave matrix multiply for 16×16×16 matrix operations (RDNA3+)
- **dot2 Support**: `v_dot2_f32_f16` instruction for FP16 operations

### Per-Architecture Features

| Architecture | GPUs | DP4A | WMMA | dot2 |
|--------------|------|------|------|------|
| RDNA1 (gfx1010) | RX 5700 XT | ❌ | ❌ | ❌ |
| RDNA1 (gfx1013) | BC-250 APU | ❌ | ❌ | ❌ |
| RDNA2 (gfx1030) | RX 6900 XT, RX 6800 XT | ✅ | ❌ | ✅ |
| RDNA3 (gfx1100) | RX 7900 XT, RX 7800 XT | ✅ | ✅ | ✅ |

### Optimizations Implemented

1. **Packed 32-bit Loads** (`hip_kernels/quant/q4_0_gemv.hip`)
   - Load 16 bytes as 4×uint32_t instead of 16×uint8_t
   - 4× fewer load instructions, better memory coalescing
   - Applied to Q4_0 GEMV kernels

2. **DP4A Q4_0/Q8_0 Decode Kernels** (`hip_kernels/quant/common.hip`, `q4_0_gemv.hip`, `q4_0_fused_q8.hip`)
   - Uses `__ockl_sdot4` for 4-way signed int8 multiply-accumulate on gfx1030/gfx1100+
   - Integrated into GEMV, gate-up, and SwiGLU dispatch paths
   - Enabled via `ROCMFORGE_Q4_0_Q8_DP4A=1`
   - Software fallback on architectures without DP4A
   - Bit-level nibble extraction matches the scalar path exactly

3. **High-Occupancy Multi-Head Prefill Attention** (`hip_kernels/attention.hip`)
   - Replaced the single-head, thread-serial prefill kernel with a `dim3(seq_len, num_heads)` cooperative kernel
   - Each (position, head) block scans its causal key range and reduces over `head_dim`
   - Preserves KV-lora reconstruction and causal mask semantics

4. **Decode Graph Health Telemetry** (`src/gpu/decode_graph_health.rs`)
   - Non-intrusive atomic counters for graph captures, replays, cache hits/misses, updates, fallbacks, and runtime disables
   - Enable with `ROCMFORGE_OBSERVE_DECODE_GRAPH_HEALTH=1`

5. **Multi-row GEMV** (`hip_kernels/quant/q4_0_gemv.hip`)
   - Processes 4 output columns per wave for better occupancy
   - Uses packed loads for dequantization
   - Shared memory input tiling for large rows

### Current Status

**Implemented:**
- ✅ GPU architecture and feature detection
- ✅ Performance profiling infrastructure (`src/gpu/profile.rs`)
- ✅ Packed load optimization for Q4_0 GEMV
- ✅ DP4A Q4_0/Q8_0 kernels integrated into decode/prefill pipeline
- ✅ High-occupancy multi-head prefill attention kernel
- ✅ Decode graph health observability
- ✅ Kernel correctness tests (`tests/integration_gpu.rs`, `tests/gpu_q4_0_q8_dispatch.rs`)
- ✅ Performance benchmarks (`benches/kernel_performance.rs`)

**Not Yet Implemented:**
- ⏳ WMMA-optimized kernel variant for RDNA3+
- ⏳ Automatic kernel dispatch based on detected features

### Performance Expectations

Measured and projected improvements on Qwen2.5-0.5B Q4_0:

| GPU | Architecture | Baseline | Expected | Speedup |
|-----|-------------|----------|----------|---------|
| RX 5700 XT | RDNA1 (gfx1010) | ~150 tok/s | 180-200 | 1.2-1.3× |
| RX 6900 XT | RDNA2 (gfx1030) | ~150 tok/s | 250-300 | 1.7-2.0× |
| RX 7900 XT | RDNA3 (gfx1100) | ~150 tok/s | 250-350 | 1.7-2.3× |
| BC-250 APU | RDNA1 (gfx1013) | ~150 tok/s | 200-220 | 1.3-1.4× |

*Note: DP4A and multi-head prefill paths are now integrated and verified against CPU oracles. Absolute speedups depend on batch size and model width; run the real-model harness for your GPU.*

### Accuracy

DP4A kernel uses the same Q4_0/Q8_0 quantization layout as the scalar path:
- Low nibble holds element `i`, high nibble holds element `i+16`, zero-point bias `-8`
- Bit-level packing via unsigned nibble extraction avoids signed-shift overflow
- Verified numerically against CPU and scalar GPU references

### Testing

Run DP4A correctness tests:

```bash
ROCMFORGE_Q4_0_Q8_DP4A=1 \
cargo test --features gpu --release --test gpu_q4_0_q8_dispatch -- --test-threads=1
```

Benchmark DP4A vs scalar baseline decode (merge 7B split shards first, see MANUAL):

```bash
ROCMFORGE_RUN_GPU_BENCHES=1 \
ROCMFORGE_DISABLE_DECODE_GRAPH=1 \
ROCMFORGE_BENCH_MODEL=/home/feanor/Projects/models/qwen2.5-7b-instruct-q4_0.gguf \
ROCMFORGE_BENCH_TOKENS=64 \
cargo bench --features gpu --bench gpu_decode_dp4a_vs_baseline
```

Measured on RX 7900 XT with a merged Qwen2.5-7B-Instruct-Q4_0 model:
- Baseline scalar fastpath: ~50 tok/s decode
- DP4A Q4_0/Q8_0: ~79 tok/s decode
- Speedup: ~1.57×

Run multi-head prefill correctness:

```bash
cargo test --features gpu --release --test integration_gpu \
  test_flash_attn_prefill_strided_kernel_correctness \
  -- --test-threads=1
```

Run decode graph health unit tests:

```bash
cargo test --features gpu --release --lib decode_graph_health -- --test-threads=1
```

Run performance benchmarks:

```bash
cargo bench --bench kernel_performance
```

## Documentation

- Main manual: [MANUAL.md](MANUAL.md)
- Project changelog: [CHANGELOG.md](CHANGELOG.md)
- Developer instructions: [AGENTS.md](AGENTS.md)
- License terms: [LICENSE](LICENSE)

## Positioning

- The main value today is that it is a small pure-HIP codebase that AMD developers can inspect, build, profile, and compare against other runtimes.
- **VRAM safety first:** The inference engine now respects your GPU by checking available VRAM before allocating, reserving headroom for the desktop compositor, and gating experimental kernels behind explicit opt-in flags.
- **Model-aware routing:** The router automatically selects the best inference path based on model metadata — no manual tuning needed for standard models.
- Expect more work on decode throughput, launch tuning, and profiling workflow before calling it broadly production-ready.

## License

GPL-3.0. See [LICENSE](LICENSE).
