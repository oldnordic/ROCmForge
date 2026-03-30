# Benchmark Results

This directory contains automated benchmark reports for ROCmForge.

## Files

- `PERFORMANCE_REPORT_*.md` - Comprehensive performance comparison reports
- `real-model-*.md` - Real model inference benchmark results
- `data.csv` - Raw benchmark data in CSV format

## Running Benchmarks

### Kernel Benchmarks

```bash
cargo bench --bench kernels
```

View HTML report: `target/criterion/report/index.html`

### Criterion GPU Decode Benchmark

Use the Criterion bench below for a steadier end-to-end measurement of the graph-backed GPU decode path on the real 0.5B regression model:

```bash
cargo bench --bench gpu_decode --features gpu -- --noplot
```

The bench defaults to:

- model: `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`
- prompt: `Hello`
- decode tokens: `64`

Useful overrides:

```bash
ROCMFORGE_BENCH_MODEL=/path/to/model.gguf cargo bench --bench gpu_decode --features gpu -- --noplot
ROCMFORGE_BENCH_TOKENS=128 cargo bench --bench gpu_decode --features gpu -- --noplot
ROCMFORGE_CRITERION_SAMPLE_SIZE=10 ROCMFORGE_CRITERION_MEASUREMENT_SECS=20 cargo bench --bench gpu_decode --features gpu -- --noplot
```

If the model file is missing or no AMD GPU is detected, the Criterion target skips itself instead of failing the whole bench run.

The Criterion target loads the model once, then measures repeated prompt+decode iterations inside the same process. Use it for throughput comparisons between code changes.

### Real Model Benchmark

```bash
# Default (all models)
cargo run --release --example benchmark_real_model

# Specific models
cargo run --release --example benchmark_real_model -- --model "*q4_k*"
```

### ROCm/HIP 7.2 Decode Profiling

Use the ignored real-model test below as the blessed batch-1 decode workload. It runs the same raw `Hello` prompt and `64` greedy decode tokens that we have been using for CLI spot checks, and it keeps the full-token HIP graph path active.

```bash
cargo test --release --features gpu --test gpu_decode_real test_gpu_greedy_decode_profile_real_model -- --ignored --nocapture --test-threads=1
```

For a repeatable graph-backed throughput baseline, use the multi-run harness below. It performs one warmup run plus five measured runs by default and prints average, stddev, min, and max decode throughput. Override the defaults with `ROCMFORGE_BENCH_RUNS`, `ROCMFORGE_BENCH_WARMUP`, and `ROCMFORGE_BENCH_TOKENS`.

```bash
cargo test --release --features gpu --test gpu_decode_real test_gpu_greedy_decode_benchmark_real_model_multi_run -- --ignored --nocapture --test-threads=1
```

The test prints one summary line in this format:

```text
PROFILE gpu_greedy_decode_real_model prompt_tokens=... decode_tokens=... prefill_ms=... prefill_tok_s=... decode_ms=... decode_tok_s=...
```

If `rocprofv3` is installed, prefer the repo-local wrapper in [`.rocprofv3/profile_decode.sh`](/home/feanor/Projects/rocmforge/.rocprofv3/profile_decode.sh):

```bash
./.rocprofv3/profile_decode.sh runtime
./.rocprofv3/profile_decode.sh runtime-gate-up
./.rocprofv3/profile_decode.sh runtime-ffn-down
./.rocprofv3/profile_decode.sh system
```

The wrapper keeps `--output-config` on by default, so each run emits a resolved `*_config.json` next to the trace outputs.

Kernel filtering and PMC inputs now live under [`.rocprofv3/README.md`](/home/feanor/Projects/rocmforge/.rocprofv3/README.md). The local ROCm examples in `/home/feanor/Projects/rocm-examples/Tools/rocprofv3/` remain the reference templates for:

- runtime traces in `rocprofv3-basic.sh`
- kernel filtering in `kernel_filter.yml`
- PMC counters in `wavefront_stats.yml`

On this machine, `rocprofv3` is installed under `/opt/rocm/bin/rocprofv3` and is not guaranteed to be in `PATH`. The wrapper defaults to that location but allows `ROCPROF_BIN` override.

PMC collection is still experimental on this ROCm 7.2.0 setup. Even a single-counter `SQ_WAVES` run filtered to `gate_up` still aborts `rocmforge`, including with `ROCMFORGE_DISABLE_DECODE_GRAPH=1` and `--disable-signal-handlers`.

### perf Host-Side Decode Checks

Use `perf` for host-side counters only. It is useful for checking launcher/runtime overhead, page faults, and scheduling noise around the decode path, but it does not replace `rocprofv3` for GPU kernels.

```bash
cargo build --release --features gpu
./.perf/perf_decode.sh
```

The wrapper defaults to software counters because the usual hardware-event set is not available cleanly on this machine. See [`.perf/README.md`](/home/feanor/Projects/rocmforge/.perf/README.md) for overrides.

Unlike the Criterion bench, the `perf` wrapper measures the full CLI process, so it includes startup and model loading. Use it to catch host-side regressions, not to replace the graph-backed decode throughput baseline.

For each decode profiling pass, record:

- total time in `gpu_flash_attn_decode_strided_multi_head_state`
- total time in the decode projection kernels and GPU argmax tail
- launch gaps between fused QKV, `kv_write_rope_state`, decode attention, output projection, and argmax
- whether the captured graph path is active throughout the run

Only tune block size, LDS footprint, or generic decode GEMV after these traces show a stable hotspot.

### Generate Report

```bash
cargo run --release --example generate_report
```

## Interpreting Results

- **Speedup**: How much faster AVX2 is compared to scalar baseline
- **Tokens/sec**: Higher is better (more tokens generated per second)
- **Prefill/Decode ms**: Lower is better (faster inference)

## Regression Detection

To detect performance regressions:

```bash
# Save baseline
cargo bench --bench kernels -- --save-baseline main

# Compare against baseline
cargo bench --bench kernels -- --baseline main
```
