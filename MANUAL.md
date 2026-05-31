# ROCmForge Manual

## 1. Scope

This manual describes the current command-line workflow for `rocmforge` in this repository state.

This project is usable, but progress is incremental. Throughput work is improving in small steps.

## 2. Prerequisites

- Rust 1.81+
- ROCm/HIP toolkit (local validation on ROCm 7.2)
- AMD GPU for HIP path (local validation on RX 7900 XT)
- ROCm runtime libraries available at execution time so `libamdhip64.so.7` resolves
- GGUF model file

## 3. Build

```bash
cargo build --release
cargo build --release --features gpu
```

## 4. Run Inference

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

Valid CLI options from current binary:

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
| `--debug` | Print top logits info |
| `--kv-dump <path>` | Dump post-prefill KV cache to binary file (research tool, CPU copy, slow) |
| `--prefill-only-validate` | Run prefill only; exits 0 on finite logits, 1 on NaN/Inf |
| `--draft-model <path>` | Draft model for speculative decoding |
| `--speculative-tokens N` | Speculative tokens per step (default: 4) |

`--device` is not supported by the current CLI.

## 5. Safety and Performance Flags

Conservative mode:

```bash
ROCMFORGE_GPU_SAFE_MODE=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu
```

Tuned mode:

```bash
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu
```

### VRAM Management (display-attached GPU safety)

For systems where the discrete GPU also powers the desktop compositor:

```bash
# Default: reserve 4 GB for desktop/compositor
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu

# Single-monitor setup: reduce reservation to 2 GB
ROCMFORGE_DESKTOP_VRAM_GB=2.0 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu

# Multi-monitor 4K setup: increase reservation to 6 GB
ROCMFORGE_DESKTOP_VRAM_GB=6.0 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu
```

The startup output shows a VRAM budget table:

```
VRAM status (device 0):
  Total              19.98 GB
  Used (other)        0.10 GB
  Free               19.89 GB
  Desktop reserved    4.00 GB  (ROCMFORGE_DESKTOP_VRAM_GB to change)
  For inference      15.89 GB
Estimated usage:
  Model weights         0.29 GB
  KV cache              0.02 GB
  Scratch buffers       0.00 GB
  Total required        0.31 GB  [OK]
```

### Experimental Kernels (opt-in, potentially unsafe)

Sparse CSR and MPO kernels are gated behind an experimental flag. Only enable if you are testing compressed model formats:

```bash
ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1 \
./target/release/rocmforge --model /path/to/model.rfm --prompt "Hello" --gpu
```

When enabled, the router may select `SvdOptimized` path for models with SVD-corrected weights.

**Warning:** Experimental kernels can fault on display-attached GPUs. Always ensure adequate VRAM headroom before enabling.

Current 7B development command:

```bash
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge \
  --gpu \
  --model /home/feanor/Projects/Memoria/models/Qwen2.5-7B-Instruct-Q4_0-Pure.gguf \
  --prompt Hello --no-template --top-p 1.0 --temperature 0.0 --max-tokens 64
```

## 6. Benchmarks Used In This Repo

### 6.1 Real-model decode harness (recommended for regressions)

```bash
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
ROCMFORGE_BENCH_RUNS=10 ROCMFORGE_BENCH_WARMUP=1 ROCMFORGE_BENCH_TOKENS=128 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```

### 6.2 Criterion benchmark

```bash
ROCMFORGE_RUN_GPU_BENCHES=1 cargo bench --bench gpu_decode --features gpu -- --noplot
```

### 6.3 rocprofv3 timeline/call profiling

```bash
./.rocprofv3/profile_decode.sh runtime
./.rocprofv3/profile_decode.sh runtime-graph
```

Use `runtime-graph` when you want graph-enabled profiling. `runtime` defaults to graph disabled.

## 7. Local Measured Results (April 10, 2026)

### 7.1 Qwen2.5-0.5B-Instruct Q4_0 (graph path, harness)

- Prefill average: `408.7 tok/s`
- Decode average: `526.8 tok/s`
- Command: section 6.1 above

### 7.2 Qwen2.5-7B-Instruct Q4_0 (`Qwen2.5-7B-Instruct-Q4_0-Pure.gguf`, CLI, 3 runs)

Command:

```bash
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge \
  --gpu \
  --model /home/feanor/Projects/Memoria/models/Qwen2.5-7B-Instruct-Q4_0-Pure.gguf \
  --prompt Hello --no-template --top-p 1.0 --temperature 0.0 --max-tokens 64
```

Observed:

- Prefill: `31.5 / 32.4 / 32.0 tok/s` (avg `32.0 tok/s`)
- Decode: `106.7 / 106.7 / 106.5 tok/s` (avg `106.6 tok/s`)

### 7.3 Qwen2.5-0.5B-Instruct Q4_0 with decode graph disabled

- Decode average: `486.0 tok/s`

## 8. What Works and What Still Needs Work

What works now:

- End-to-end local inference on AMD GPU with Qwen2.5 GGUF models
- Decode graph replay path
- Profiling and benchmark scripts in-repo

What still needs work:

- Further decode throughput improvements
- Better parity with llama.cpp on the same hardware
- Cleaner and lower-noise profiling workflow
- Broader model-family validation beyond the current Qwen-first scope

## 9. Troubleshooting

If performance is unexpectedly low:

1. Confirm `--release` build.
2. Confirm `--gpu` is used.
3. Check whether `ROCMFORGE_GPU_SAFE_MODE` is set.
4. Check whether decode graph is enabled when expected.
5. Confirm your ROCm runtime environment is loaded so the binary can resolve `libamdhip64.so.7`.
6. Re-run section 6.1 benchmark and compare against this manual.

If the process aborts with GPU fault / desktop crash:

1. Check VRAM budget in startup output — ensure `Total required` fits within `For inference`.
2. Increase desktop reservation: `ROCMFORGE_DESKTOP_VRAM_GB=6.0` (or higher).
3. Do NOT set `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1` unless testing compressed models.
4. Check if the model uses sparse/MPO weights — these use experimental kernels.
5. Run with `ROCMFORGE_GPU_SAFE_MODE=1` to disable all fastpaths and graphs.
6. Report the `[Router] Selected path: ...` line from the output — this tells us which code path faulted.

## 10. Model Selection & Quantization Guidelines (TurboQuant Invariants)

To get the most out of TurboQuant ultra-low-bit KV cache compression without degrading model intelligence, follow these core guidelines:

### 10.1 The Compounding Quantization Penalty
Avoid running TurboQuant KV cache compression (3-bit Lloyd-Max + 1-bit QJL) on top of standard lossy mixed weight quantizations (like `Q4_K_M`, `Q5_K_M`, etc.). 
* **The Penalty:** Mixed-precision weight quantizations generate chaotic, noisy, and high-variance activations that vary across layers. This noise degrades the performance of the Fast Walsh-Hadamard Transform (FWHT) pre-rotation and prevents the offline-computed Lloyd-Max centroids from fitting the distribution uniformly.
* **The Sweet Spot:** Use a **high-precision base model (FP32, FP16, or high-fidelity Q8_0)** GGUF as the input. Pristine base weights generate highly stable, low-variance activations. The base model's self-attention blocks are exceptionally resilient, absorbing the minor noise of the 4-bit KV Cache while achieving up to **4x dynamic VRAM savings** during decoding.

### 10.2 Transcoding & Preservation Safety
The `rocmforge-convert` tool guarantees that a high-precision input remains high-precision during the `.rfm` binary conversion:
* **Float32 (`F32`):** Copied directly as float32.
* **Q4_0 (`Q4Split`):** Transposed and rearranged for maximum GPU memory bus transaction efficiency (no precision loss relative to the original Q4_0).
* **Other Quantizations (e.g. `Q8_0`):** Packed directly into `GgufPassthrough` block-for-block, ensuring zero metadata or weight loss.

### 10.3 Dynamic Metadata-Driven Routing
ROCmForge automatically prevents hardcoded architectural mismatches or overlaps. When a model is loaded:
1. `ModelConfig` ingests metadata flags (e.g. `kv_quant_bits`, `kv_lora_dim`).
2. `gpu::router::select_path` inspects layer weight characteristics (checking for MoE, SSM, SVD) and automatically dispatches to the correct `InferencePath` (e.g. `BatchedPrefill`, `SvdOptimized`, or `DecodeStyle`).
3. Under GQA-only mode (unprojected weights), the GPU kernels bypass MLA projection blocks dynamically to guarantee bit-level attention score correctness.

