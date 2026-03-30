# AMD ROCm/HIP 7.2 Findings for ROCmForge

Date: 2026-03-30

This note captures the AMD-side findings that matter for `rocmforge` on the local ROCm 7.2 setup. It combines official AMD documentation with what was verified on the local RX 7900 XT (`gfx1100`) machine.

## Local context

- ROCm install: `7.2.0`
- Profiling tools:
  - `/opt/rocm/bin/rocprofv3`
  - `/opt/rocm/bin/rocprofv3-avail`
- Local ROCm examples: `/home/feanor/Projects/rocm-examples/`
- Primary regression/perf model:
  - `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`

## 1. What AMD says about rocprofv3

### Use trace mode first

AMD’s HIP performance guidance says to start by profiling and measuring a baseline with `rocprofv3`, then determine whether the workload is compute-bound or memory-bound by looking at kernel time, memory bandwidth, occupancy, and CU utilization.

Repo implication:

- do not tune kernel shapes first
- trace first
- only tune after a stable hotspot ranking exists

### Runtime trace is the right first pass

AMD documents `--runtime-trace` as the aggregate tracing mode for:

- HIP runtime API
- marker API
- memory operations
- kernel dispatches

That makes it the right first pass for `rocmforge`, because it exposes both launch overhead and kernel time without immediately depending on PMCs.

### `--output-config` is worth using every time

AMD documents `--output-config` as a way to emit the resolved profiler configuration for the current run. On this machine that works cleanly and writes a `*_config.json` next to the trace outputs.

Repo implication:

- check the resolved config before assuming a trace used the options you intended

### YAML and JSON input files are the intended way to make runs reproducible

AMD documents YAML/JSON `jobs:` input files for both tracing and counter collection. Text input files only support PMC rows. YAML/JSON support the broader profiler configuration surface.

Repo implication:

- keep kernel filters and PMC sets in files instead of hand-editing long commands

### PMC collection is stricter than tracing

AMD documents two important PMC rules:

- direct `--pmc` CLI collection must fit in a single pass
- multi-pass collection should be expressed with multiple `pmc` rows in an input file

Repo implication:

- large counter sets should not be shoved into one CLI invocation
- use `rocprofv3-avail` and small counter groups first

### `--disable-signal-handlers` is not a performance flag

AMD documents `--disable-signal-handlers` as a control over profiler-vs-application signal handling precedence. It is useful when the application or test framework already owns signal handling. It is not documented as a fix for profiling failures in general.

Repo implication:

- use it when signal ownership matters
- do not expect it to fix a broken PMC run by itself

## 2. What AMD says about HIP performance on Radeon / RDNA3

### Profile, classify, then optimize

AMD’s guidance is explicit:

1. profile baseline
2. analyze whether the hotspot is compute-bound or memory-bound
3. apply targeted changes
4. re-profile

This matches the current direction for `rocmforge`. Blind kernel rewrites were not paying off.

### Arithmetic intensity and bytes moved matter

AMD’s performance guidance points directly at arithmetic intensity, achieved bandwidth, and compute throughput as the key classification inputs. That is the GPU version of the same lesson from CPU inference: bytes moved per token matter.

Repo implication:

- every extra global-memory round-trip in decode hurts
- fusing kernels and reusing LDS are higher-value than random launch-shape changes

### Coalescing and LDS reuse are first-order concerns

AMD’s performance guidance says consecutive threads should access consecutive addresses to maximize coalescing, and that LDS should be used when data can be loaded once and reused many times.

Repo implication:

- decode kernels should stay Wave32-friendly on Radeon
- access patterns should target wide, aligned memory transactions
- LDS should be used to cut repeated VRAM reads, not just because it exists

### Occupancy matters, but only together with register and LDS pressure

AMD’s guidance emphasizes that occupancy is limited by registers and shared memory. It also recommends block sizes that are multiples of the hardware warp size, and notes that Radeon GPUs execute in warps of 32.

Repo implication:

- avoid treating occupancy as an isolated goal
- block shapes should respect Wave32 on `gfx1100`
- register pressure and LDS footprint need to be considered together

### RDNA3 memory layout matters

AMD’s hardware notes for RDNA/RDNA3 highlight:

- Wave32 execution
- 128-byte cache lines aligned with 32 lanes x 4 bytes
- LDS bank conflicts are serialized across cycles

Repo implication:

- 32-lane-friendly contiguous access is the default target
- strided or bank-conflicting patterns need a concrete reason to exist

## 3. What AMD says about HIP graphs in 7.2

### Graphs fit repetitive decode workloads

AMD’s HIP graph documentation says graphs help when the same workflow runs many times, because graph definition and instantiation are paid once, then replay uses a single `hipGraphLaunch()` call. AMD also documents stream capture as a way to turn an existing stream workflow into a graph.

Repo implication:

- autoregressive decode is a valid graph target
- one-shot or structurally changing paths are weaker graph candidates

### Update node params instead of rebuilding graphs when the structure is fixed

AMD’s graph tutorial explicitly calls out per-node update APIs such as `hipGraphExecKernelNodeSetParams()` for cases where only a subset of node parameters changes between executions.

Repo implication:

- fixed decode topology plus changing `pos`, `seq_len`, and pointers is a good fit

### ROCm 7.2 adds graph runtime optimizations

AMD’s ROCm 7.2.0 release notes call out:

- optimized graph node batching via a doorbell mechanism for certain graph topologies
- memset graph-node handling improvements
- reduced async-handler enqueue-path contention

Repo implication:

- graph replay is one of the few ROCm 7.2 runtime changes that maps directly onto `rocmforge`’s decode loop

## 4. What was confirmed locally on this machine

### Trace mode works

The following class of runs works:

```bash
/opt/rocm/bin/rocprofv3 \
  --runtime-trace \
  --stats \
  --summary \
  --summary-output-file stdout \
  --summary-units usec \
  --group-by-queue \
  --output-config \
  --output-directory /tmp/rocprof-decode \
  --output-format csv \
  -- ./target/release/rocmforge \
    --gpu \
    --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt Hello \
    --no-template \
    --top-p 1.0 \
    --max-tokens 64
```

`--output-config` also works. It emits a resolved config JSON and is worth keeping on by default.

### PMC collection is still unstable here

Even a very small PMC attempt still aborts `rocmforge` on this machine:

- single counter: `SQ_WAVES`
- filtered to one hotspot kernel
- `ROCMFORGE_DISABLE_DECODE_GRAPH=1`
- `--disable-signal-handlers`

The profiler still ends in a foreign-exception abort. That means:

- trace mode is usable now
- PMC mode should still be treated as experimental on this setup

### perf helps only on the host side here

`perf` is still useful on this machine, but only for host-side overhead:

- `perf --version` works
- default software counters work
- common hardware-event sets can fail with `No supported events found`

Repo implication:

- use `perf` to watch `task-clock`, page faults, context switches, and migrations
- keep using `rocprofv3` for GPU kernel timing and hotspot ranking
- do not treat `perf` as a substitute for HIP tracing

### The local hotspot ranking is clear enough to act on

The runtime trace and the built-in decode profiler both point to the same decode bottlenecks:

1. fused `gate_up` FFN kernel
2. `ffn_down` residual projection
3. QKV path
4. decode attention
5. Q8 LM head

Repo implication:

- decode is FFN-heavy on this model
- attention is not the first optimization target anymore
- the next safe work should target FFN bytes moved and FFN kernel efficiency

### Current kernel-resource snapshot from filtered traces

Filtered runtime traces on the accepted FFN fast path now show:

- `gemv_gate_up_swiglu_q4_0_f32_kernel<4>`
  - dispatches: `1536`
  - average duration under filtered runtime trace: about `80.2 us`
  - workgroup size: `128`
  - VGPR count: `56`
  - SGPR count: `128`
  - scratch size: `0`
- `gemv_q4_0_f32_residual_wave_parallel_kernel<8>`
  - dispatches: `2880`
  - average duration: about `25.8 us`
  - workgroup size: `256`
  - VGPR count: `56`
  - SGPR count: `128`
  - scratch size: `0`

The important implication is that the current FFN hotspots are already fairly register-heavy. A naive “merge more work into one kernel” rewrite can easily make occupancy worse instead of better. The accepted change was narrower: lower the `gate_up` fast-path workgroup from `8` waves to `4` waves for small hidden sizes, without touching model-specific logic.

### Current local baseline on this tree

The current reproducible baseline from `test_gpu_greedy_decode_benchmark_real_model_multi_run` is about:

- `227.0 tok/s` decode average
- `207.6 tok/s` prefill average

The direct decode-stage profiler without graph replay currently reports:

- `121.0 tok/s` decode
- `80.0 tok/s` prefill
- FFN stage totals:
  - `gate_up`: `152.539 ms`
  - `ffn_down`: `89.550 ms`

This split matters: the accepted 4-wave `gate_up` heuristic improves the graph-backed decode path materially even though the direct stage profiler still shows `gate_up` as the dominant raw kernel bucket. The observed win appears to come from the graph-backed execution path as a whole, not from a simple reduction in isolated `gate_up` kernel time.

The new Criterion real-model bench on the same tree reports:

- total time for `Hello` + `64` greedy decode tokens: about `286.9-287.9 ms`
- throughput: about `222.3-223.1 tok/s`

That lines up closely with the ignored multi-run harness and is the best current end-to-end throughput baseline inside one loaded process.

The repo-local `perf` wrapper currently reports software-counter stats over the full CLI process:

- decode output during the three repeated CLI runs: about `211.5-216.2 tok/s`
- `task-clock`: about `621.0 ms`
- `page-faults`: about `64.5k`
- elapsed wall time: about `0.67 s`

This is useful, but it measures a broader scope than Criterion because it includes startup and model loading. It should be treated as a host-overhead check, not as the primary decode-throughput number.

## 5. What this means for ROCmForge

### Strong conclusions

- There is no AMD-documented “magic ROCm 7.2 switch” that should suddenly close the gap to `llama.cpp`.
- The AMD guidance strongly favors profile-driven work, not speculative kernel rewrites.
- HIP graphs were the right generic runtime feature to adopt first.
- For decode, bytes moved per token still look more important than one more round of launch-shape guessing.
- Criterion and `perf` are useful complements, but they are not replacements for `rocprofv3`.

### Practical next steps

1. Keep `rocprofv3` runtime trace as the default profiling pass.
2. Save kernel filters and counter sets in versioned files instead of ad hoc shell history.
3. Use the real-model multi-run benchmark and Criterion bench to gate throughput claims.
4. Keep the accepted small-hidden-size `gate_up` 4-wave fast-path heuristic and use the graph-backed multi-run benchmark as the truth source for decode throughput.
5. Use the repo-local `perf` wrapper to catch host-side regressions before assuming a slowdown is in the HIP kernels.
6. Focus the next kernel work on FFN hotspots first, especially `ffn_down`, because `gate_up` already has a stable launch-shape improvement on this path.
7. Revisit PMC collection only after the abort path is understood.

## 6. AMD sources

- ROCprofiler-SDK `rocprofv3` usage:
  - https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- HIP performance guidelines:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- HIP hardware implementation:
  - https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/understand/hardware_implementation.html
- HIP graphs:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
- HIP Graph API tutorial:
  - https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/tutorial/graph_api.html
- HIP occupancy API:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/occupancy.html
- ROCm 7.2.0 release notes:
  - https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html
- ROCm 7.2.1 release notes:
  - https://rocm.docs.amd.com/en/latest/about/release-notes.html
