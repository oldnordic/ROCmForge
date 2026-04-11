# ROCmForge Manual

## 1. Scope

This manual describes the current command-line workflow for `rocmforge` in this repository state.

This project is usable, but progress is incremental. Throughput work is improving in small steps.
The current validated scope is Qwen2.5 GGUF inference, especially on the AMD GPU HIP path.
This is best treated as a pure-HIP AMD prototype that colleagues can build, run, and profile locally.

## 2. Prerequisites

- Rust 1.81+
- ROCm/HIP toolkit (local validation on ROCm 7.2)
- AMD GPU for HIP path (local validation on RX 7900 XT)
- ROCm runtime libraries available at execution time so `libamdhip64.so.7` resolves
- GGUF model file
- Current development focus: Qwen2.5 GGUF models

## 3. Build

```bash
cargo build --release
cargo build --release --features gpu
```

## 4. Run Inference

Current user-facing testing is centered on Qwen2.5 GGUF models. Other GGUF models may partially work, but they are not the supported focus of the current prototype.

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

- `--model <path>`
- `--prompt <text>`
- `--gpu`
- `--max-tokens N`
- `--temperature F`
- `--top-p F`
- `--no-template`
- `--list-tensors`
- `--debug`

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
