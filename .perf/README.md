# perf Decode Workflow

Use `perf` here for host-side overhead only. It can help catch regressions in CPU scheduling, page faults, and launcher/runtime cost around the GPU decode loop, but it does not replace `rocprofv3` for GPU kernels.

This wrapper measures the full CLI process, not just the steady-state decode loop. That means it includes startup, model open/load, and argument parsing in addition to the decode run itself.

## Default workflow

Build the release binary first:

```bash
cargo build --release --features gpu
```

Then run the repo-local wrapper:

```bash
./.perf/perf_decode.sh
```

That defaults to software counters because this machine does not expose the usual hardware counters cleanly for this workload.

## Defaults

- binary: `./target/release/rocmforge`
- model: `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`
- prompt: `Hello`
- decode tokens: `64`
- repeats: `3`
- events:
  - `task-clock`
  - `context-switches`
  - `cpu-migrations`
  - `page-faults`
  - `major-faults`
  - `minor-faults`

## Useful overrides

```bash
PERF_REPEATS=5 ./.perf/perf_decode.sh
PERF_EVENTS=cycles,instructions,branches,branch-misses ./.perf/perf_decode.sh
ROCMFORGE_BENCH_TOKENS=128 ./.perf/perf_decode.sh
ROCMFORGE_BENCH_MODEL=/path/to/model.gguf ./.perf/perf_decode.sh
```

On this machine, the hardware-event example above may fail with `No supported events found.` If that happens, fall back to the default software-counter set.

## Interpretation

- `task-clock` rising without throughput gains usually means more host overhead.
- `page-faults` spikes often indicate a cold run or extra host allocations.
- `context-switches` and `cpu-migrations` should stay low on a clean decode run.

Keep using `rocprofv3` for GPU hotspot ranking and use `perf` to check whether a change moved host-side overhead in the wrong direction.
