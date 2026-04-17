# GPU Decode Hotpath Architecture

Ground-truth mapping of the GPU inference pipeline as of commit `dacace7`.
No hipBLAS, rocBLAS, or MIOpen — all kernels are custom HIP compiled via `hipcc`.

---

## 1. Module Dependency Graph

```
main.rs (CLI)
  |
  +-- run_gpu_inference()
        |
        v
  gpu::forward.rs  <--- THE CENTRAL DISPATCHER
    |
    +-- gpu::ops.rs          (weight-type dispatch: Q4_0 / Q4_1 / Q4_K / Q5_K / Q8_0)
    |     |
    |     +-- gpu::kernels/*  (raw HIP kernel launchers)
    |
    +-- gpu::cache.rs        (GpuKvCache, GpuForwardScratch, GpuPrefillScratch)
    +-- gpu::graph.rs         (HIP Graph capture/replay: CapturedDecodeGraph)
    +-- gpu::device.rs        (GpuDevice: device_id, stream, sync)
    +-- gpu::weights.rs       (GpuModelWeights, GpuLayerWeights, GpuBuffer)
    +-- gpu::ffi.rs           (raw HIP FFI: hipMemcpy, hipStreamSynchronize, etc.)
```

---

## 2. Two Execution Paths

The CLI (`main.rs`) and the benchmark (`benches/gpu_decode.rs`) use **the same**
public API, but with different logits handling:

| Path            | Entry (per token)                                  | Logits mode          | Used by |
|-----------------|----------------------------------------------------|----------------------|---------|
| **Greedy GPU**  | `gpu_embed_token_hybrid` → `gpu_full_forward_hybrid` | `GreedyArgmax`       | CLI (default), bench |
| **Host sampling** | `gpu_embed_token_hybrid` → `gpu_full_forward_hybrid` | `DownloadToHost`   | CLI (`--debug` or top_p < 1.0) |

---

## 3. Decode Loop Control Flow (CLI, `src/main.rs:691-762`)

```
┌──────────────────────────────────────────────────────────────────┐
│ for each decode step:                                            │
│                                                                  │
│  1. gpu_embed_token_hybrid(token_id)                             │
│     - Q8_0: embed_q8_0_token (GPU lookup)                       │
│     - else: CPU embed → hip_memcpy_h2d_async                    │
│                                                                  │
│  2. gpu_full_forward_hybrid(pos, GreedyArgmax)                  │
│     ├── TRY: gpu_try_full_greedy_decode_graph()  ← HIP GRAPH   │
│     │   ├── key match? → upload_decode_state → graph.launch()   │
│     │   │              → gpu_read_greedy_argmax_result           │
│     │   │              → device.synchronize()                    │
│     │   └── no key?   → capture new graph OR fall through       │
│     └── FALLBACK: per-layer loop + tail (see below)             │
│                                                                  │
│  3. If decode_next_token is Some(token): use it directly         │
│     Else: device.synchronize() + read argmax result              │
│                                                                  │
│  4. pos += 1                                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Per-Layer Decode Forward (`gpu_layer_forward_hybrid`, `forward.rs:1272-1449`)

Each of the 24 layers executes this sequence:

```
                    INPUT: hidden[0..h-1]  (GPU GpuBuffer)

    ┌─────────────────────────────────────────────────────┐
    │  1. RMS_NORM (attn_norm)                            │
    │     gpu_dispatch_rms_norm → rms_norm_on_stream       │
    │     hidden ──→ normed                                │
    ├─────────────────────────────────────────────────────┤
    │  2. FUSED QKV PROJECTION                            │
    │     gpu_dispatch_fused_qkv_on_stream                 │
    │     Q4_0: gemv_qkv_q4_0_f32_on_stream (single call) │
    │     else: 3x gemv + bias adds                       │
    │     normed ──→ q[0..q_size], k[0..kv_size], v[0..kv_size] │
    ├─────────────────────────────────────────────────────┤
    │  3. ROPE (Q)                                        │
    │     rope_heads_on_stream(q, pos, num_heads, ...)    │
    ├─────────────────────────────────────────────────────┤
    │  4. KV WRITE + ROPE (K)                             │
    │     kv_write_rope_on_stream                         │
    │     Applies rope to K, writes K and V to KV cache   │
    │     k,v ──→ kv_cache[layer][pos]                    │
    ├─────────────────────────────────────────────────────┤
    │  5. FLASH ATTENTION DECODE                          │
    │     flash_attn_decode_strided_multi_head_on_stream  │
    │     q ──×── kv_cache[0..seq_len] ──→ attn_out       │
    ├─────────────────────────────────────────────────────┤
    │  6. OUTPUT PROJECTION + RESIDUAL                    │
    │     gpu_dispatch_gemv_residual_on_stream (Q4_0)     │
    │       attn_out + hidden ──→ hidden   (fused)        │
    │     else: gemv(attn_o) + add(residual)              │
    ├─────────────────────────────────────────────────────┤
    │  7. RMS_NORM (ffn_norm)                             │
    │     gpu_dispatch_rms_norm                           │
    │     hidden ──→ normed                               │
    ├─────────────────────────────────────────────────────┤
    │  8. FUSED GATE+UP+SwiGLU                            │
    │     gpu_dispatch_fused_gate_up_on_stream             │
    │     Q4_0: gemv_gate_up_swiglu_q4_0_f32_on_stream    │
    │     normed ──→ swiglu[0..ff_size]                   │
    ├─────────────────────────────────────────────────────┤
    │  9. FFN DOWN PROJECTION + RESIDUAL                  │
    │     gpu_dispatch_gemv_residual_on_stream (Q4_0)     │
    │       swiglu + hidden ──→ hidden   (fused)          │
    │     else: gemv(ffn_down) + add(residual)            │
    └─────────────────────────────────────────────────────┘

                    OUTPUT: hidden[0..h-1]  (updated in-place)
```

---

## 5. Logits Tail (`gpu_launch_greedy_logits_tail_on_stream`, `forward.rs:299-346`)

After all 24 layers:

```
    ┌─────────────────────────────────────────────────────┐
    │  10. RMS_NORM (output_norm)                         │
    │      hidden ──→ normed                               │
    ├─────────────────────────────────────────────────────┤
    │  11. LM_HEAD PROJECTION                             │
    │      gpu_dispatch_gemv_on_stream (Q8_0 lm_head)     │
    │      normed ──→ logits[0..vocab_size-1]             │
    ├─────────────────────────────────────────────────────┤
    │  12. ARGMAX                                          │
    │      argmax_f32_on_stream                            │
    │      logits ──→ argmax_result_index (single i32)    │
    └─────────────────────────────────────────────────────┘
```

---

## 6. HIP Graph Capture & Replay

Two graph scopes exist:

### 6a. FullGreedyDecode (preferred, `gpu_try_full_greedy_decode_graph`)

Captures steps 1–12 for ALL 24 layers in one graph.

```
First call (pos=0):
  1. Check decode_graph_disabled() ← env vars
  2. Compute key from (device, config, weight ptrs, kv ptrs)
  3. No cached graph? → capture:
     a. upload_decode_state(0, 1)
     b. device.synchronize()
     c. begin_capture()
     d. gpu_launch_full_greedy_decode_on_stream()
     e. end_capture() → HipGraph
     f. Try update existing, or instantiate new CapturedDecodeGraph
  4. Store in scratch.captured_decode

Subsequent calls:
  1. Compute key → matches cached graph?
  2. upload_decode_state(pos, pos+1)
  3. graph.launch(stream)
  4. gpu_read_greedy_argmax_result (async D2H)
  5. device.synchronize()
  6. Read token from pinned host buffer
```

### 6b. GreedyTail (fallback, `gpu_try_greedy_decode_graph`)

Captures only steps 10–12 (output norm + lm_head + argmax).
Used when the full-decode graph capture fails.

### 6c. Graph Invalidation

Graph is re-captured when:
- `DecodeGraphKey` changes (weight ptrs, kv ptrs, config)
- `graph.launch()` fails
- `ROCMFORGE_DISABLE_DECODE_GRAPH` or `ROCMFORGE_PROFILE_DECODE_STAGES` is set

---

## 7. GPU Buffer Layout

### GpuForwardScratch (decode, `cache.rs:286-321`)

```
┌──────────────────────────┬────────────────────┬─────────────────────────┐
│ Buffer                   │ Size (Qwen2.5-0.5B) │ Purpose                 │
├──────────────────────────┼────────────────────┼─────────────────────────┤
│ hidden                   │ h=896 f32          │ Current hidden state    │
│ normed                   │ h f32              │ Post-RMS-norm           │
│ q                        │ num_heads*head_dim  │ Query vector (14*64)    │
│ k                        │ num_kv_heads*head_dim│ Key vector (2*64)      │
│ v                        │ num_kv_heads*head_dim│ Value vector (2*64)    │
│ attn_out                 │ num_heads*head_dim  │ Attention output        │
│ layer_out                │ h f32              │ Layer residual buffer   │
│ gate                     │ ff_size f32        │ FFN gate activations    │
│ swiglu                   │ ff_size f32        │ SwiGLU output           │
│ logits                   │ vocab_size f32     │ Final logits            │
│ argmax_partial_values    │ ceil(v/1024) f32   │ Argmax reduction       │
│ argmax_partial_indices   │ ceil(v/1024) i32   │ Argmax reduction       │
│ argmax_result_device     │ 1 i32 (GPU)        │ Argmax result          │
│ argmax_result_index      │ 1 i32 (pinned)     │ Host-side argmax copy  │
│ input_hidden_pinned      │ h f32 (pinned)     │ CPU embed staging      │
│ decode_state             │ 2 i32              │ [pos, seq_len] for graph│
└──────────────────────────┴────────────────────┴─────────────────────────┘
```

### GpuKvCache

```
Layer 0:  k[0..max_seq*kv_size]   v[0..max_seq*kv_size]
Layer 1:  k[0..max_seq*kv_size]   v[0..max_seq*kv_size]
...
Layer 23: k[0..max_seq*kv_size]   v[0..max_seq*kv_size]

kv_size = num_kv_heads * head_dim = 2 * 64 = 128 f32 per position
```

---

## 8. Kernel Dispatch Table (`ops.rs`)

Qwen2.5-0.5B uses **Q4_0** for all layer weights and **Q8_0** for token embeddings + lm_head.

| Weight           | Type | Kernel Path                                        | Notes                          |
|------------------|------|----------------------------------------------------|--------------------------------|
| attn_q,k,v       | Q4_0 | `gemv_qkv_q4_0_f32_on_stream` (fused)             | 3 GEMVs in 1 kernel           |
| attn_o           | Q4_0 | `gemv_q4_0_f32_residual_on_stream` (fused)        | GEMV + residual add           |
| ffn_gate,up      | Q4_0 | `gemv_gate_up_swiglu_q4_0_f32_on_stream` (fused)  | GEMV + SiLU + mul             |
| ffn_down         | Q4_0 | `gemv_q4_0_f32_residual_on_stream` (fused)        | GEMV + residual add           |
| token_emb        | Q8_0 | `embed_q8_0_token`                                | Direct row lookup             |
| lm_head          | Q8_0 | `gemv_q8_0_f32_lm_head_on_stream`                 | Specialized vocab-parallel    |
| attn_norm,ffn_norm | F32 | `rms_norm_on_stream`                               |                               |
| output_norm      | F32 | `rms_norm_on_stream`                               |                               |

**Vulkan-style kernels** (`gemv_q4_0_f32_vulkan_style`, `gemv_q4_k_f32_vulkan_style`) exist but
are only tried for GEMV dispatch — and the gate/up Vulkan variant is **disabled** due to
numerical divergence (see `ops.rs:401-416`).

---

## 9. Synchronization Points

Every decode step has these sync points (graph OR non-graph path):

1. **Graph path**: `device.synchronize()` after `gpu_read_greedy_argmax_result`
   - Graph launches all kernels
   - Async D2H copy of argmax result queued
   - Sync waits for D2H to complete
   - Read pinned host buffer

2. **Non-graph path**: Same `device.synchronize()` in `gpu_greedy_logits_tail_token`
   - Individual kernel launches on stream
   - D2H of argmax
   - Sync
   - Read

3. **CLI fallback**: Additional `device.synchronize()` at `main.rs:743` when
   `decode_next_token` is `None` (non-greedy path or graph miss)

---

## 10. Profiling Infrastructure

| Tool | Command | Measures |
|------|---------|----------|
| Criterion bench | `cargo bench --bench gpu_decode --features gpu` | End-to-end prompt+decode loop |
| CLI timer | `target/release/rocmforge --gpu ...` | Same, with startup |
| rocprofv3 | `.rocprofv3/profile_decode.sh runtime` | GPU kernel timings |
| Stage profiler | `ROCMFORGE_PROFILE_DECODE_STAGES=1` | Per-stage host-side timing (disables graphs) |
| perf | `.perf/perf_decode.sh` | Host-side counters |

---

## 11. Qwen2.5-0.5B Model Dimensions

```
num_layers       = 24
hidden_size      = 896
num_heads        = 14
num_kv_heads     = 2   (GQA, 7:1 ratio)
head_dim         = 64
intermediate_size = 4864
vocab_size       = 151936
rms_norm_eps     = 1e-5
rope_theta       = 1000000.0
rope_neox        = true
```

---

## 12. Performance Regression Context

**Baseline**: ~223 tok/s (before safety hardening commits)
**Current**:  ~9.2 tok/s (after commits `de70c68..dacace7`)

Commits that changed the hotpath:
1. `de70c68` — Task 1: Extended hardware capability detection + test fixes
2. `f37916e` — Task 2: Hardened HIP kernels with null checks, alignment guards, LDS limits
3. `4c89b78` — Task 3: Safety guards in Rust wrappers + test updates
4. `dacace7` — Task 4: Safety Orchestrator in ops.rs + forward path updates

The regression is reproducible with a clean build (`cargo clean --release && cargo build --release --features gpu`).

Key suspects to investigate:
- HIP kernel changes (Task 2) — null checks / alignment guards add branches
- ops.rs changes (Task 4) — dispatch path may have added overhead
- forward.rs changes — graph capture/replay may be broken, forcing fallback path
