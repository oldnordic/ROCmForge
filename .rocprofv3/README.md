# rocprofv3 Workflows

This directory contains reproducible `rocprofv3` inputs for the main `rocmforge` batch-1 decode workload.

Primary workload:

- binary: `./target/release/rocmforge`
- model: `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`
- prompt: `Hello`
- decode tokens: `64`

Quick start:

```bash
./.rocprofv3/profile_decode.sh runtime
./.rocprofv3/profile_decode.sh runtime-graph
./.rocprofv3/profile_decode.sh runtime-gate-up
./.rocprofv3/profile_decode.sh runtime-ffn-down
./.rocprofv3/profile_decode.sh system
```

Modes:

- `runtime`: full runtime trace with summary and resolved config output (default wrapper flags keep decode graph off)
- `runtime-graph`: runtime trace with graph-backed decode forced on (`ROCMFORGE_ENABLE_DECODE_GRAPH=1`)
- `system`: broader system trace
- `runtime-gate-up`: runtime trace filtered to `gemv_gate_up_swiglu_q4_0_f32_kernel`
- `runtime-ffn-down`: runtime trace filtered to `gemv_q4_0_f32_residual_wave_parallel_kernel`
- `pmc-gate-up`: single-counter PMC run for the `gate_up` kernel
- `pmc-ffn-down`: single-counter PMC run for the `ffn_down` kernel

Important caveat on this ROCm 7.2.0 setup:

- trace mode works
- `--output-config` works and is worth keeping enabled
- PMC collection still aborts `rocmforge`, even with `ROCMFORGE_DISABLE_DECODE_GRAPH=1` and `--disable-signal-handlers`

Useful outputs:

- `*_config.json`: resolved profiler settings
- `*_kernel_trace.csv`: per-dispatch timings and kernel resource metadata
- `*_hip_api_trace.csv`: host-side HIP API activity
- `*_memory_copy_trace.csv`: memory copies
- `*_memory_allocation_trace.csv`: allocations
- `*_domain_stats.csv`: summarized domain totals

Environment overrides:

- `ROCPROF_BIN`: profiler binary, default `/opt/rocm/bin/rocprofv3`
- `ROCPROF_OUTDIR`: output directory, default `/tmp/rocprof-<mode>`
- `ROCPROF_ENABLE_DECODE_GRAPH_DEFAULT`: wrapper default when `ROCMFORGE_ENABLE_DECODE_GRAPH` is unset (`0`)
- `ROCPROF_ENABLE_Q8_ACTIVATION_FASTPATH_DEFAULT`: wrapper default when `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH` is unset (`1`)
- `ROCMFORGE_BIN`: profiled binary, default `./target/release/rocmforge`
- `ROCMFORGE_MODEL`: model path
- `ROCMFORGE_PROMPT`: prompt string
- `ROCMFORGE_MAX_TOKENS`: decode length
- `ROCMFORGE_EXTRA_ARGS`: extra CLI args appended to `rocmforge`
- `ROCMFORGE_ENABLE_DECODE_GRAPH`: explicit decode graph toggle for the profiled process
- `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH`: explicit decode q8-activation fastpath toggle

Use the ignored throughput harness to validate any code change against the same real model:

```bash
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```
