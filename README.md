rocmforge - LLM inference on AMD GPUs (HIP) with a CPU fallback path.

## Current status

- The project currently works for local Qwen GGUF runs on this machine.
- The GPU path is AMD-only and pure HIP. There is no CUDA or cross-vendor backend here.
- Performance work is moving slowly and in small steps.
- Recent changes improved 7B decode throughput, but this project is still behind llama.cpp on the same hardware.

### Supported quantizations (GPU)

| Quantization | Status | Notes |
|--------------|--------|-------|
| Q4_0 | Supported | Baseline quantization |
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

- `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1`
  - Enables sparse CSR and MPO kernels (potentially unsafe on display-attached GPUs).
  - Only use when testing compressed `.rfm` models with sparse/MPO weights.

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

2. **DP4A-Optimized Fusion Kernel** (`hip_kernels/quant/q4_0_fused_norm_qkv_rope_dp4a.hip`)
   - Uses `__builtin_amdgcn_sdot4` for 4-way int8 multiply-accumulate
   - Expected 1.5-2× speedup on RDNA2+ (gfx1030+) and RDNA3+ (gfx1100+)
   - Trade-off: ~0.4% noise from on-the-fly activation quantization
   - Kernel implemented but not yet integrated into decode pipeline

3. **Multi-row GEMV** (`hip_kernels/quant/q4_0_gemv.hip`)
   - Processes 4 output columns per wave for better occupancy
   - Uses packed loads for dequantization
   - Shared memory input tiling for large rows

### Current Status

**Implemented:**
- ✅ GPU architecture and feature detection
- ✅ Performance profiling infrastructure (`src/gpu/profile.rs`)
- ✅ Packed load optimization for Q4_0 GEMV
- ✅ DP4A-optimized fusion kernel (implemented, pending pipeline integration)
- ✅ Kernel correctness tests (`tests/kernel_correctness.rs`)
- ✅ Performance benchmarks (`benches/kernel_performance.rs`)

**Not Yet Implemented:**
- ⏳ Environment variable overrides (ROCMFORGE_USE_DP4A, etc.)
- ⏳ WMMA-optimized kernel variant for RDNA3+
- ⏳ Automatic kernel dispatch based on detected features
- ⏳ Integration of DP4A kernel into decode pipeline

### Performance Expectations

Based on hipfire analysis, expected improvements on Qwen2.5-0.5B Q4_0:

| GPU | Architecture | Baseline | Expected | Speedup |
|-----|-------------|----------|----------|---------|
| RX 5700 XT | RDNA1 (gfx1010) | ~150 tok/s | 180-200 | 1.2-1.3× |
| RX 6900 XT | RDNA2 (gfx1030) | ~150 tok/s | 250-300 | 1.7-2.0× |
| RX 7900 XT | RDNA3 (gfx1100) | ~150 tok/s | 250-350 | 1.7-2.3× |
| BC-250 APU | RDNA1 (gfx1013) | ~150 tok/s | 200-220 | 1.3-1.4× |

*Note: Performance numbers are projections based on hipfire implementation. Actual results pending integration testing.*

### Accuracy

DP4A kernel quantizes activations on-the-fly to use int8 SIMD:
- Introduces ~0.4% noise vs scalar kernel
- Coherence not affected (verified by correctness tests)
- Noise level similar to quantization format itself

### Testing

Run kernel correctness tests:

```bash
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 cargo test --features gpu --test kernel_correctness -- --nocapture --test-threads=1
```

Run performance benchmarks:

```bash
cargo bench --bench kernel_performance
```

## Documentation

- Main manual: [MANUAL.md](MANUAL.md)
- hipfire analysis: [hipfire_detailed_analysis.md](hipfire_detailed_analysis.md)
- hipfire quick start: [hipfire_quick_start.md](hipfire_quick_start.md)

## Positioning

- The main value today is that it is a small pure-HIP codebase that AMD developers can inspect, build, profile, and compare against other runtimes.
- **VRAM safety first:** The inference engine now respects your GPU by checking available VRAM before allocating, reserving headroom for the desktop compositor, and gating experimental kernels behind explicit opt-in flags.
- **Model-aware routing:** The router automatically selects the best inference path based on model metadata — no manual tuning needed for standard models.
- Expect more work on decode throughput, launch tuning, and profiling workflow before calling it broadly production-ready.

## License

GPL-3.0. See [LICENSE](LICENSE).
