rocmforge - LLM inference on AMD GPUs (HIP) with a CPU fallback path.

## Current status

- The project currently works for local Qwen GGUF runs on this machine.
- The GPU path is AMD-only and pure HIP. There is no CUDA or cross-vendor backend here.
- This is still a prototype, but it is usable enough for AMD colleagues to build, run, and profile locally.
- Performance work is moving slowly and in small steps.
- Recent changes improved 7B decode throughput, but this project is still behind llama.cpp on the same hardware.

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

Supported CLI options (from `--help`):

| Option | Description |
|---|---|
| `--model <path>` | GGUF model path |
| `--prompt <text>` | Prompt text |
| `--gpu` | Use GPU backend |
| `--max-tokens N` | Max generated tokens (default: 256) |
| `--temperature F` | Sampling temperature (default: 1.0) |
| `--top-p F` | Nucleus sampling threshold (default: 0.9) |
| `--no-template` | Disable chat template |
| `--list-tensors` | Print tensors and exit |
| `--debug` | Print debug logits info |

Note: `--device` is not a valid flag in the current binary.

## Runtime safety controls

- `ROCMFORGE_GPU_SAFE_MODE=1`
  - Forces conservative mode for this process.
  - Disables decode graph and experimental fastpaths.

- `ROCMFORGE_ENABLE_DECODE_GRAPH=1`
  - Enables decode graph replay.

- `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1`
  - Enables the Q8 activation fastpath used in decode kernels.

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

## Documentation

- Main manual: [MANUAL.md](MANUAL.md)

## Positioning

- `rocmforge` is currently best treated as a Qwen-focused AMD/HIP prototype.
- The main value today is that it is a small pure-HIP codebase that AMD developers can inspect, build, profile, and compare against other runtimes.
- Expect more work on decode throughput, launch tuning, and profiling workflow before calling it broadly production-ready.

## License

GPL-3.0. See [LICENSE](LICENSE).
