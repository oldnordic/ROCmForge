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
| `--threads N` / `-t N` | Number of CPU threads/cores to use (default: auto-detect) |
| `--ctx-size N` / `-c N` | Override maximum context window size (default: model default) |

`--device` is not supported by the current CLI.

## 5. Safety and Performance Flags

Conservative mode:

```bash
ROCMFORGE_GPU_SAFE_MODE=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu
```

Tuned mode:

The GPU Decode Graph is **enabled by default** for GPU execution when greedy decoding is active (e.g. `--top-p 1.0`). To explicitly force or toggle features, use:

```bash
# Explicitly enable graph capture (default behavior for greedy paths)
ROCMFORGE_ENABLE_DECODE_GRAPH=1 \
ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu --top-p 1.0

# Disable graph capture to fall back to the standard decode loop
ROCMFORGE_DISABLE_DECODE_GRAPH=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu --top-p 1.0

# Enable DP4A int8 dot-product for Q4_0/Q8_0 kernels (gfx1030/gfx1100+)
# DP4A is now AUTOMATIC for models >3B parameters (e.g., 7B-class Q4_0 models)
# This manual override is only needed to force DP4A on small models (not recommended)
# or to disable DP4A on large models (for correctness testing).
ROCMFORGE_Q4_0_Q8_DP4A=1 \
./target/release/rocmforge --model /path/to/model.gguf --prompt "Hello" --gpu

# Merge split GGUF shards (e.g. 7B Q4_0 downloaded in two parts)
llama-gguf-split --merge qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf \
    qwen2.5-7b-instruct-q4_0.gguf

# Observe decode graph health telemetry
ROCMFORGE_OBSERVE_DECODE_GRAPH_HEALTH=1 \
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

For `cargo test` GPU work, the repo default is already conservative:
- `RUST_TEST_THREADS=1`
- `ROCMFORGE_GPU_LOCK_TIMEOUT=30`
- `ROCMFORGE_DESKTOP_VRAM_GB=4.0`

These defaults come from [`.cargo/config.toml`](/home/feanor/Projects/rocmforge/.cargo/config.toml). The shared test helper now skips based on the guarded allocation budget after desktop reservation and safety margin, not raw free VRAM.

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

### 7.4 Qwen2.5-0.5B-Instruct Q8_0 (`qwen2.5-0.5b-instruct-q8_0.rfm` / `.gguf`, CLI, greedy `--top-p 1.0`)

- Prefill: `151.8 tok/s`
- Decode speed (baseline / graph capture disabled or invalidated): `87.7 tok/s`
- Decode speed (optimized / graph capture active): `340.4 tok/s`
- Reference Native `llama.cpp` speed: `239.2 tok/s` (ROCmForge is ~42% faster with decode graph replay active)

## 8. What Works and What Still Needs Work

What works now:

- End-to-end local inference on AMD GPU with Qwen2.5, Gemma4 GGUF and `.rfm` models
- Gemma4 12B Q4_0 with hybrid attention and per-layer embeddings (PLE)
- PLE VRAM optimization (35× reduction for Gemma4 models)
- Decode graph replay path with zero dynamic allocations in the generation hotpath
- DP4A int8 dot-product acceleration for Q4_0 × Q8_0 kernels on RDNA2/RDNA3
- High-occupancy multi-head prefill attention kernel
- Decode graph health telemetry for capture/replay/fallback observability
- Parity/outperformance vs native `llama.cpp` for supported quantizations (e.g. Q8_0, Q4_0)
- Profiling and benchmark scripts in-repo

What still needs work:

- Gemma4 E2B hybrid attention support (flash attention kernel incompatible with mixed head_dim)
- Further decode throughput improvements for other quantization styles (K-quants)
- Cleaner and lower-noise profiling workflow
- Broader model-family validation beyond the current Qwen/Gemma4 scope
- Automatic feature dispatch based on detected GPU architecture

## 9. Troubleshooting

If performance is unexpectedly low:

1. Confirm `--release` build.
2. Confirm `--gpu` is used.
3. Check whether `ROCMFORGE_GPU_SAFE_MODE` is set.
4. Check whether decode graph is enabled when expected.
5. DP4A is automatic for models >3B parameters on RDNA2/RDNA3 (no manual env var needed).
6. Try the single-row high-occupancy launch: `ROCMFORGE_Q4_0_Q8_SINGLE_ROW=1`.
7. Confirm your ROCm runtime environment is loaded so the binary can resolve `libamdhip64.so.7`.
8. Re-run section 6.1 benchmark and compare against this manual.

If the process aborts with GPU fault / desktop crash:

1. Check VRAM budget in startup output — ensure `Total required` fits within `For inference`.
2. Increase desktop reservation: `ROCMFORGE_DESKTOP_VRAM_GB=6.0` (or higher).
3. Do NOT set `ROCMFORGE_ENABLE_EXPERIMENTAL_GPU_KERNELS=1` unless testing compressed models.
4. Check if the model uses sparse/MPO weights — these use experimental kernels.
5. Run with `ROCMFORGE_GPU_SAFE_MODE=1` to disable all fastpaths and graphs.
6. Report the `[Router] Selected path: ...` line from the output — this tells us which code path faulted.

If decode graph health counters show repeated cache misses or fallbacks:

1. Look at the snapshot from `src/gpu/decode_graph_health.rs`.
2. A high `cache_misses` count means shapes/strides are changing per token; check the model config.
3. A high `fallbacks` count means capture failed or was disabled; review the VRAM budget and safety flags.
4. Enable observability with `ROCMFORGE_OBSERVE_DECODE_GRAPH_HEALTH=1` and capture output during the failing run.

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

### 10.4 Inference Change Discipline

When changing inference behavior, do not validate only one model source or one
runtime path. Use `docs/inference-change-checklist.md` and audit:
- `ModelFile`
- GGUF loader path
- RFM loader path
- converter assumptions when `.rfm` behavior can change
- router selection
- decode-graph eligibility
- GEMV/GEMM dispatch eligibility
- speculative loading if model-open logic changed

This prevents the common failure mode where a fix improves one path while
quietly regressing the other.
