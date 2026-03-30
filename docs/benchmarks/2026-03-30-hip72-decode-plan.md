# GPU HIP 7.2 Decode Plan

> **For agentic workers:** keep this generic. No model-name checks, no one-off Qwen shortcuts, and no widening into speculative overlap work before the data says it matters. Specialization must be selected from tensor metadata, runtime shape, and device capabilities only.

**Goal:** use the ROCm/HIP 7.2 features that actually matter for this codebase to close the remaining batch-1 decode gap on RX 7900 XT: graph replay for repeated decode work, occupancy-guided launch tuning, and fewer bytes moved per token.

**Current state after the first HIP 7.2 passes:**
- 7B-class models run because the LDS fast path has a chunked fallback.
- The Q8_0 LM head already has a metadata-driven specialization.
- The decode path now runs on `device.stream()`.
- Greedy decode now captures and replays the full-token HIP graph on the explicit stream.
- The decode path already fuses `Q4_0` QKV bias and the decode-side `rope(k) + kv_write`.
- Measured decode on `Qwen2.5-0.5B-Instruct Q4_0` is now about `225 tok/s`, while local `llama-bench` on `Vulkan0` remains roughly `619 tok/s`.

**Active execution order from this point:**
1. Replace the serial decode-attention reductions with wave-aware block reductions.
2. Add the `rocprofv3` profiling gate so the next kernel work is data-driven.
3. Revisit generic decode GEMV only if profiling still shows a projection bottleneck.

**ROCm/HIP 7.2 features this plan targets:**
- HIP stream capture and graph replay
- `hipGraphExecKernelNodeSetParams()` for per-token param updates
- `hipOccupancyMaxPotentialBlockSize()` and `hipOccupancyAvailableDynamicSMemPerBlock()` for launch/LDS tuning
- `rocprofv3` for launch-gap and bytes-per-token profiling

**Non-goals in this pass:**
- model-specific tuning
- rocWMMA rewrite
- async host-device overlap
- changing the CPU fallback contract
- widening generic Q8 mat-vec dispatch before profiling proves it is a win

---

## File Structure

**Files to modify in this plan:**
1. `docs/benchmarks/2026-03-30-hip72-decode-plan.md`
2. `src/gpu/ffi.rs`
3. `src/gpu/error.rs`
4. `src/gpu/device.rs`
5. `src/gpu/graph.rs` (new)
6. `src/gpu/cache.rs`
7. `src/gpu/forward.rs`
8. `src/gpu/ops.rs`
9. `src/gpu/kernels/attention.rs`
10. `src/gpu/kernels/quant.rs`
11. `src/gpu/kernels/mod.rs`
12. `src/gpu/mod.rs`
13. `hip_kernels/attention.hip`
14. `hip_kernels/quant/q4_0_fused.hip`
15. `tests/integration_gpu.rs`
16. `tests/kv_cache_gpu.rs`
17. `tests/gpu_decode_real.rs`
18. `tests/gpu_graph.rs` (new)
19. `docs/benchmarks/README.md`

**Files to read but not change unless implementation proves it necessary:**
- `src/gpu/weights.rs`
- `src/gpu/dynamic_loader.rs`
- `src/main.rs`
- `src/cpu/forward.rs`
- `/home/feanor/Projects/rocm-examples/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_capture/main.hip`
- `/home/feanor/Projects/rocm-examples/HIP-Basic/occupancy/main.hip`
- `/home/feanor/Projects/rocm-examples/Tools/rocprofv3/README.md`

---

## Task 1: Add Safe HIP 7.2 Graph and Occupancy Wrappers

**Files:**
- Modify: `src/gpu/ffi.rs`
- Modify: `src/gpu/error.rs`
- Modify: `src/gpu/device.rs`
- Add: `src/gpu/graph.rs`
- Modify: `src/gpu/mod.rs`

- [x] Add FFI bindings and safe wrappers for:
  - `hipStreamBeginCapture`
  - `hipStreamEndCapture`
  - `hipGraphInstantiate`
  - `hipGraphDestroy`
  - `hipGraphLaunch`
  - `hipGraphExecDestroy`
  - `hipGraphExecKernelNodeSetParams`
  - `hipOccupancyMaxPotentialBlockSize`
  - `hipOccupancyAvailableDynamicSMemPerBlock`

- [x] Keep raw HIP graph handles inside the `gpu` module.
  The rest of the code should use Rust RAII wrappers only.

- [x] Add a `CapturedDecodeGraph` type in `src/gpu/graph.rs` that owns:
  - the executable graph
  - the graph topology key
  - kernel-node handles or node lookup data for per-token updates

- [x] Extend `GpuDevice` with small graph-safe helpers instead of leaking more raw runtime API calls into `forward.rs`.

**Acceptance:**
- Graph lifecycle and occupancy queries are available from safe Rust wrappers.
- No raw HIP graph API leaks outside `src/gpu/`.

---

## Task 2: Make the Decode Path Capture-Ready

**Files:**
- Modify: `src/gpu/cache.rs`
- Modify: `src/gpu/forward.rs`
- Modify: `src/gpu/ops.rs`
- Modify: `src/gpu/kernels/mod.rs`
- Modify: `src/gpu/mod.rs`

- [x] Extract a GPU-only single-token decode body from `gpu_full_forward_hybrid(...)`.
  It must:
  - keep a fixed launch topology for a given model/config
  - avoid host copies inside the captured region
  - avoid per-token allocations
  - avoid token-dependent control flow that changes which kernels launch

- [x] Add a topology key to invalidate captured graphs when any of these change:
  - model dimensions
  - quantization types used by active tensors
  - logits mode
  - device id / wavefront size

- [x] Extend `GpuForwardScratch` or a sibling cache object with reusable graph state:
  - current executable graph
  - current topology key
  - reusable parameter blobs for node updates

- [x] Keep CPU fallback paths outside the captured region.
  First graph pass should cover only the native GPU path.

**Acceptance:**
- The decode body can be invoked either directly or through a future captured graph without changing semantics.

---

## Task 3: Capture and Replay the Greedy Decode Graph

**Files:**
- Modify: `src/gpu/graph.rs`
- Modify: `src/gpu/cache.rs`
- Modify: `src/gpu/forward.rs`
- Modify: `tests/gpu_graph.rs`
- Modify: `tests/gpu_decode_real.rs`

- [x] Capture one token of decode work on `device.stream()` for the `GreedyArgmax` path.

- [x] Instantiate the graph once and replay it for later tokens with updated params rather than rebuilding it every step.

- [x] Use `hipGraphExecKernelNodeSetParams()` to update per-token values such as:
  - `pos`
  - `seq_len`
  - scale-dependent kernel arguments if needed
  - pointer arguments if the graph model requires them

- [x] Add invalidation rules:
  - recapture on model change
  - recapture on logits-mode change
  - recapture if a fallback kernel path is selected instead of the native GPU path

- [x] Keep the uncaptured path as the conservative fallback if capture fails.

**Acceptance:**
- The greedy decode loop can run through HIP graph replay on the same explicit stream.
- Real-model greedy output still matches the uncaptured path.

**Note:** do not block this task on `DownloadToHost`. If the host-logits path complicates capture, leave it uncaptured in the first pass and capture only the fast greedy path.

---

## Task 4: Add a Profiling Gate with rocprofv3

**Files:**
- Modify: `tests/gpu_decode_real.rs`
- Modify: `docs/benchmarks/README.md`

- [x] Add a stable, repeatable real-model decode benchmark entry point in `tests/gpu_decode_real.rs` or reuse an existing one with fixed arguments.

- [x] Document the exact `rocprofv3` workflow in `docs/benchmarks/README.md`:
  - runtime trace
  - Perfetto trace
  - kernel-name filtering
  - wavefront / memory metrics

- [x] Define the measurements required before the next optimization pass:
  - per-kernel time
  - gaps between launches
  - top kernels by time
  - memory-heavy kernels by operational intensity

**Acceptance:**
- There is one blessed profiling path for batch-1 decode.
- Future kernel work is gated on measured hotspots, not guesswork.

---

## Task 5: Remove Three Decode Launches per Layer by Fusing QKV Bias

**Files:**
- Modify: `hip_kernels/quant/q4_0_fused.hip`
- Modify: `src/gpu/kernels/quant.rs`
- Modify: `src/gpu/ops.rs`
- Modify: `src/gpu/forward.rs`
- Modify: `tests/integration_gpu.rs`

- [x] Extend the existing fused decode `Q4_0` QKV kernel so it can optionally consume `q`, `k`, and `v` bias pointers and apply the bias in-register.

- [x] Keep bias pointers optional. If a model has no bias tensors, the same kernel path should still work.

- [x] Preserve the current unfused path for unsupported layouts or non-`Q4_0` tensors.

- [x] Remove the three separate bias-add launches from the decode fast path when the fused kernel is active.

**Acceptance:**
- The decode loop performs fewer launches per layer without adding model-specific behavior.
- Synthetic parity tests match the current unfused result.

---

## Task 6: Fuse RoPE and KV Write Where It Actually Saves Traffic

**Files:**
- Modify: `hip_kernels/attention.hip`
- Modify: `src/gpu/kernels/attention.rs`
- Modify: `src/gpu/cache.rs`
- Modify: `src/gpu/forward.rs`
- Modify: `tests/kv_cache_gpu.rs`
- Modify: `tests/integration_gpu.rs`

- [x] Add a decode-side kernel that combines the operations that currently cause extra global-memory traffic:
  - apply RoPE to `k`
  - write `k` to the KV cache
  - write `v` to the KV cache

- [x] If fusing `q` RoPE into the same kernel complicates correctness or hurts occupancy, do not force it in this pass.
  The main byte-saving target is the `k/v` path.

- [x] Use cache stride helpers from `GpuKvCache` rather than hardcoded layout math in kernels.

- [x] Keep the current `rope_heads_on_stream(...)` and `kv.write_on_stream(...)` path as fallback for unsupported cases.

**Acceptance:**
- The decode path removes at least one extra read/write cycle for `k/v`.
- KV cache correctness tests still pass.

---

## Task 7: Replace the Naive Decode Attention Reduction

**Files:**
- Modify: `hip_kernels/attention.hip`
- Modify: `src/gpu/kernels/attention.rs`
- Modify: `src/gpu/device.rs`
- Modify: `src/gpu/ffi.rs`
- Modify: `src/gpu/forward.rs`
- Modify: `tests/integration_gpu.rs`
- Modify: `tests/gpu_decode_real.rs`

- [ ] Replace the current decode attention hot path with a shape-driven implementation that:
  - uses wave reductions instead of serial `tid == 0` max/sum work
  - tiles K/V through LDS when sequence length justifies it
  - uses device occupancy info to cap block size and dynamic shared memory safely

**Current execution slice:** parallelize the softmax max/sum reductions first while keeping the existing launch contract graph-safe. LDS tiling and occupancy-driven launch retuning stay in this task, but they come after parity and benchmark confirmation on the simpler reduction rewrite.

- [x] Replace the serial `tid == 0` max/sum work in both decode kernels with wave-aware block reductions while keeping the existing fixed launch contract graph-safe.
- [ ] Retune block size and LDS footprint from occupancy and runtime shape only after the profiling gate is in place.

- [ ] Keep launch decisions generic:
  - driven by `seq_len`
  - driven by `head_dim`
  - driven by `num_heads / num_kv_heads`
  - driven by occupancy / LDS limits from HIP

- [ ] Keep a conservative fallback kernel for small or awkward shapes.

**Acceptance:**
- The decode attention kernel no longer serializes max/sum on one lane.
- The optimized path is selected from runtime shape and device properties only.

---

## Task 8: Revisit Generic Decode GEMV Only After Profiling

**Files:**
- Modify: `src/gpu/ops.rs`
- Modify: `src/gpu/device.rs`
- Modify: `src/gpu/kernels/quant.rs`
- Modify: `hip_kernels/quant/q4_0_fused.hip`
- Modify: `tests/integration_gpu.rs`

- [ ] Do not widen the earlier generic Q8 mat-vec experiment blindly. It regressed.

- [ ] After `rocprofv3` confirms a remaining projection bottleneck, selectively generalize shape-driven launch tuning for decode GEMV where all of these hold:
  - runtime shape matches the profiled good case
  - occupancy estimate is not worse than the baseline launch
  - the tensor role does not require a separate specialization contract

- [ ] Prefer changing launch geometry or reuse inside existing fused kernels before adding new one-off kernels.

**Acceptance:**
- Any generalized GEMV tuning is backed by profiling and occupancy data, not guesswork.

---

## Validation Commands

- `cargo build --release --features gpu`
- `cargo test --features gpu --test integration_gpu -- --nocapture --test-threads=1`
- `cargo test --features gpu --test kv_cache_gpu -- --nocapture --test-threads=1`
- `cargo test --features gpu --test gpu_decode_real -- --nocapture --test-threads=1`
- `./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf --prompt "Hello" --no-template --top-p 1.0 --max-tokens 64`
- `rocprofv3 --help`

---

## Success Criteria

This plan is successful if it delivers these in order:

1. Graph replay works for the greedy decode path without changing outputs.
2. `rocprofv3` shows a clear reduction in launch gaps after graph replay.
3. Fused decode removes measurable bytes-per-token traffic, not just API overhead.
4. Decode attention becomes the new optimized hot path without model-specific dispatch.
5. The repo still runs the 7B model safely and keeps conservative fallbacks for unsupported cases.

---

## References

- Companion findings: `improvements.md`
- HIP graph capture example:
  `/home/feanor/Projects/rocm-examples/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_capture/main.hip`
- HIP occupancy example:
  `/home/feanor/Projects/rocm-examples/HIP-Basic/occupancy/main.hip`
- rocprofv3 examples:
  `/home/feanor/Projects/rocm-examples/Tools/rocprofv3/README.md`
- Official HIP graphs:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html>
- Official HIP graph tutorial:
  <https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/tutorial/graph_api.html>
- Official HIP performance guidelines:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html>
