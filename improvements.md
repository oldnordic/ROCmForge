# Improvements

## Scope

This document records the current GPU performance investigation for batch-1 decode on the RX 7900 XT. It focuses on changes and findings that survived measurement, not every experiment attempted along the way.

## Environment

- Repository: `rocmforge`
- GPU: AMD Radeon RX 7900 XT (`gfx1100`)
- Runtime: ROCm/HIP 7.2.x
- Reference model: `Qwen2.5-0.5B-Instruct Q4_0`
- Larger-model validation: `Qwen2.5-7B-Instruct Q4_0`
- Main comparator: local `llama.cpp` Vulkan backend

## What Landed

### 1. Large-model shared-memory fallback

The original fast path assumed the hidden and intermediate activations could fit in the GPU's shared memory budget. That broke on 7B-class models because the default single-pass LDS staging exceeded the practical 32 KB limit seen on consumer AMD GPUs.

The current GEMV path now supports chunked LDS loading:

- Small tensors keep the single-pass shared-memory fast path.
- Larger tensors automatically fall back to chunked LDS loading.
- Selection is based on runtime tensor shape and launch constraints, not model names.

This is why `Qwen2.5-7B-Instruct Q4_0` now runs correctly instead of failing at launch.

### 2. Metadata-driven LM-head specialization

The biggest measured decode win so far came from fixing the Q8_0 LM-head path without hardcoding any model family.

What changed:

- `WeightMeta` now carries a semantic `TensorRole`.
- GPU weight loading marks explicit LM heads as `LmHead` and tied embeddings-as-LM-head as `TiedLmHead`.
- GEMV dispatch selects a dedicated Q8_0 LM-head kernel only when metadata says the tensor really is an LM head.
- The dedicated kernel keeps the generic Q8_0 implementation intact and only changes launch geometry.
- Launch width is selected from runtime shape (`n_rows` / quant blocks), not from model identity.

Why it helped:

- The generic Q8_0 GEMV used a fixed 256-thread block per output column.
- On the tested 0.5B model, the hidden size is short enough that most of those lanes were idle in the LM head.
- Reducing block width for short hidden sizes removed a large amount of wasted reduction and synchronization overhead.

Measured effect during investigation:

- Before the LM-head specialization: about `117 tok/s` decode on the 64-token CLI run.
- After the LM-head specialization: about `187.5 tok/s` decode on the same setup.

### 3. Decode-path stream cleanup

The decode path previously mixed launches on the default stream with launches on the device-owned HIP stream. That was not the main throughput bottleneck, but it made ordering harder to reason about and blocked clean graph-capture work.

What changed:

- Added stream-aware wrappers for decode-used kernels.
- Routed decode-side GEMV, fused QKV, fused gate/up, norm, RoPE, KV writes, and decode attention through `device.stream()`.
- Kept behavior unchanged for the non-stream wrappers so the safer fallback path remains available.

Measured effect:

- Stable decode moved from about `202.5 tok/s` to about `205 tok/s`.
- The real value of this change is architectural: the decode loop is now on one explicit stream and is ready for HIP graph capture.

### 4. Device metadata plumbing for future tuning

The GPU device wrapper now exposes the HIP-reported wavefront size. That does not deliver a measurable win by itself, but it removes another hidden assumption from future tuning work and gives the Rust side the data needed for Wave32-aware dispatch decisions on `gfx1100`.

## What Did Not Survive Measurement

Two ideas were implemented, measured, and then backed out because they made things worse:

- A generic shape-driven Q8 mat-vec path inspired by `llama.cpp` Vulkan reduced decode to about `168-170 tok/s`.
- A generic decode-attention rewrite reduced decode to about `141-159 tok/s`.

Those experiments were useful because they narrowed the search space, but they were not kept in the tree.

## Current Findings

### VRAM usage is not the main problem

External counters were used instead of app-side estimates:

- `rocm-smi --showpids`
- `rocm-smi --showmeminfo vram`
- `/sys/class/drm/card1/device/mem_info_vram_used`

Result:

- The live VRAM delta for `rocmforge` and local `llama.cpp` on the 0.5B model was effectively the same.
- The residual VRAM left after exit looked like driver/runtime caching rather than process-owned memory.

Conclusion:

- There was no evidence of an extra `~100 MB` problem unique to `rocmforge`.
- The performance gap is not explained by VRAM footprint alone.

### The remaining gap is decode efficiency

Current local comparison shows the problem clearly:

- `rocmforge`: about `205 tok/s` decode on the 64-token CLI run.
- Local `llama-bench` on `Vulkan0`: `619.35 tok/s` for `tg128`.

These are not identical workloads, but they are close enough to show that `rocmforge` is still far behind on batch-1 decode.

The main reasons are:

1. Too many kernel launches per token

- Each layer still performs a long sequence of small decode launches.
- At batch size 1, launch overhead matters much more than in prefill.

2. Decode attention is still too naive

- The current kernel still does more serial work than it should.
- It does not yet resemble the more optimized flash-attention style path used by `llama.cpp` when backend conditions allow it.

3. Too many bytes moved per token

This is the GPU version of the same lesson from CPU inference: throughput depends heavily on how much data must move for each unit of useful work.

For batch-1 decode, arithmetic intensity is low. Each token streams:

- quantized weights
- Q/K/V activations
- KV cache slices
- attention outputs
- FFN intermediates

Every extra write to VRAM and re-read from VRAM costs bandwidth and latency. That is why launch cleanup alone did not close the gap.

## ROCm/HIP 7.2 Findings That Matter

The ROCm 7.2 research points to a clear direction for the next round of work.

### HIP graphs are the best runtime-level opportunity

ROCm 7.2.0 added graph runtime improvements that are directly relevant to autoregressive decode, including graph-node batching improvements and lower async enqueue overhead.

Why it matters here:

- Decode repeats the same launch topology once per token.
- Only a small amount of state changes each step, such as `pos`, sequence length, and a few pointers.
- HIP graphs can amortize launch overhead across that repeated structure.

### Graph node parameter updates fit this workload

HIP graph execution supports updating kernel-node parameters without rebuilding the whole graph. That matches the current decode loop much better than rebuilding or recapturing every token.

### Occupancy and dynamic shared-memory APIs are useful

ROCm 7.2 also exposes occupancy helpers that can guide safer runtime block-size and LDS decisions. That is a better long-term path than baking more static launch assumptions into HIP code.

### Profiling must focus on bytes and launch gaps

The next profiling pass should use `rocprofv3` to measure:

- per-kernel time
- idle gaps between launches
- memory traffic
- operational intensity

Without that, it is too easy to optimize the wrong kernel or mistake launch cleanup for a bandwidth fix.

## Recommended Next Steps

1. Capture and replay the decode subgraph with HIP graphs on the existing explicit stream.
2. Use `rocprofv3` to measure where time and bytes actually go per token.
3. Reduce bytes moved per token by fusing more of decode:
   - QKV projection
   - bias application
   - RoPE
   - KV write
4. Rework decode attention around wave-level reductions and better LDS reuse.
5. Keep all specialization shape-driven or metadata-driven. Do not branch on model name.

## Validation Commands Used

The following commands were used during the investigation:

```bash
cargo build --release --features gpu
cargo test --features gpu ops::tests --lib
cargo test --features gpu --test weights_gpu
./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf --prompt Hello --no-template --top-p 1.0 --max-tokens 64
rocm-smi --showpids
rocm-smi --showmeminfo vram
cat /sys/class/drm/card1/device/mem_info_vram_used
MANGOHUD=0 llama-bench -m /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf -p 0 -n 128 -fa 1 -sm none -ngl 999 -dev Vulkan0 -r 1 -o md
```

## References

- HIP graphs: <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html>
- HIP graph tutorial: <https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/tutorial/graph_api.html>
- ROCm 7.2.0 release notes: <https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html>
- ROCm latest release notes: <https://rocm.docs.amd.com/en/latest/about/release-notes.html>
- HIP performance optimization: <https://rocm.docs.amd.com/projects/HIP/en/develop/understand/performance_optimization.html>
- HIP performance guidelines: <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html>
