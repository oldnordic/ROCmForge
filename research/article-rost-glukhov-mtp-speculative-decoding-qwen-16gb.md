---
slug: "article-rost-glukhov-mtp-speculative-decoding-qwen-16gb-trap"
title: "MTP Speculative Decoding on Qwen 3.6 — Real Benchmarks (Rost Glukhov, 2026-05-21)"
date: 2026-06-03
tags: [rocmforge, speculative-decoding, mtp, qwen, llama.cpp, inference, vram, benchmarks]
category: research
---

# MTP Speculative Decoding on Qwen 3.6 — Real Benchmarks

**Source:** Rost Glukhov, Medium (2026-05-21)  
**Hardware:** RTX 4080, 16 GB VRAM  
**Models:** Qwen 3.6 27B dense, Qwen 3.6 35B MoE  
**Quantizations:** IQ3_XXS (27B), IQ3_S (35B)  
**Framework:** llama.cpp (`--spec-type draft-mtp`)  
**Relevance to ROCmForge:** MTP is a model-native speculative decoding variant (extra prediction heads in the model checkpoint). ROCmForge already has a `SpeculativeEngine` in `src/gpu/speculative.rs` that supports draft/target dual-model execution. This article validates the VRAM cost model and context-window trade-offs that any speculative engine must account for.

---

## Core Finding

MTP speculative decoding is **not a free lunch**. The MTP heads consume VRAM proportional to `--spec-draft-n-max`. On a 16 GB GPU, that VRAM comes directly out of the KV cache budget, which shrinks the usable context window. The speedup is real — **67% on 27B dense** — but the cost can break agentic workflows that require long context.

| Model | MTP Enabled? | --spec-draft-n-max | KV Cache | Gen Speed (t/s) | Avg Context | Speedup vs Standard | Context Cost |
|-------|-------------|-------------------|----------|-------------------|------------|------------------- |-------------|
| **Qwen 3.6 27B dense** | No | — | q8 | 45 | **80K** | baseline | — |
| Qwen 3.6 27B dense | **Yes** | 2 | q8 | **75** | **40K** | **+67%** | **-50%** |
| Qwen 3.6 27B dense | Yes | 1 | q5 | 57 | 70K | +39% | -13% |
| Qwen 3.6 27B dense | Yes | 3 | q5 | 67 | 60K | +63% | -54% |
| **Qwen 3.6 35B MoE** | No | — | q8 | 146 | **80K** | baseline | — |
| Qwen 3.6 35B MoE | **Yes** | 1 | q8 | **186** | **15K** | **+27%** | **-81%** |
| Qwen 3.6 35B MoE | Yes | 2 | q8 | 189 | <10K | +29% | unusable |
| Qwen 3.6 35B MoE | No | — | q5 | 122 | **120K** | baseline | — |
| Qwen 3.6 35B MoE | Yes | 1 | q5 | 151 | **10K** | +24% | **-92%** |

*Avg Context = practical working window at ~14.8 GB VRAM usage, leaving ~500 MB headroom for desktop apps.*  
*Gen Speed = generation (decode) tokens per second.*  
*Prompt speed also drops with MTP (~200 → ~150 t/s for 27B) due to device-to-host transfer during prefill.*

---

## Key Insight: The MTP VRAM Cost Model

The MTP heads add roughly **1–2 GB** of overhead. On 16 GB cards, that is not "extra" VRAM — it is **reallocated from KV cache**, which cascades into a smaller context window. The critical failure mode:

| Scenario | Context Required | Available with MTP | Result |
|----------|---------------|-------------------|--------|
| Agentic workflow (tool calling, multi-step) | 64K | 40K (27B q8 max 2) | **Fails at step 3** |
| Agentic workflow | 64K | 15K (35B q8 max 1) | **Fails immediately** |
| Agentic workflow | 64K | 70K (27B q5 max 1) | Works, but q5 degrades quality |

**The failure is hard, not gradual.** There is no partial credit for "almost enough context." The agent hits the wall and rejects the model.

---

## Practical Ranking for 16 GB VRAM

| Rank | Configuration | Gen Speed | Avg Context | Verdict |
|------|--------------|-----------|------------|---------|
| 1 | **27B + MTP q8 max 2** | 75 t/s | 40K | Best raw throughput. Acceptable context for most tasks. |
| 2 | **27B + MTP q5 max 1** | 57 t/s | 70K | Best balance. Preserves agentic workflows but q5 quality drop is real. |
| 3 | **35B standard q8** | 146 t/s | 80K | No MTP overhead. Full context. Still fast. |
| 4 | **35B + MTP q8 max 1** | 186 t/s | 15K | Speed is impressive but context unusable. **Not recommended at 16 GB.** |

---

## Surprises the Author Found

| Surprise | Detail | Implication for ROCmForge |
|----------|--------|----------------------------|
| MTP on 27B dense is very effective | +67% speedup is a major jump; most local-inference optimizations land at 10–20% | SVD outlier correction in `.rfm` is our equivalent structural advantage. Combined with speculative decode, there may be multiplicative gains. |
| 35B MoE fails hard on context | Sparse routing makes MTP head cheap, but VRAM math doesn't work on 16 GB | If rocmforge supports MoE, speculative engine must account for this: larger models need **explicit VRAM pre-flight** that includes draft buffers + KV cache + weights. |
| q5 KV cache quality drop is significant | Recovering context comes at a real quality cost, workload-dependent | Any KV cache quantization option in rocmforge must expose **per-token quality metrics** (perplexity delta) so users can make informed trade-offs. |

---

## Relevance to ROCmForge INF-15 (Speculative Decode Server Plumbing)

The `SpeculativeEngine` in `src/gpu/speculative.rs` already has:

- Dual-model co-loading (target + draft)
- Independent KV cache isolation (`verify_cache_isolation`)
- VRAM pre-flight (`VramSession::check_fits` with 85% safety margin)
- Draft/verify forward passes

**What this article adds:**

| New Requirement | Source Evidence | Action for INF-15 |
|---------------|-----------------|-------------------|
| `--spec-draft-n-max`-style parameter | "Higher values increase VRAM pressure without proportional speed gains" | Add `draft_n_max` to model load API. Default to 2. Clamp based on VRAM budget estimate. |
| MTP native heads vs external draft model | "MTP heads built directly into certain model checkpoints" vs llama.cpp's `--spec-type draft-mtp` | Support both modes: `SpeculativeEngine` for external draft model, future `MtpEngine` for native MTP heads. |
| KV cache quantization trade-off surface | q5 recovers context but degrades quality; q8 is author's choice despite cost | If KV cache quant is configurable, expose it in the load API with clear speed/context/quality labels. |
| Agentic context floor | "Hermes Agent requires 64K context by default and rejects smaller windows" | VRAM pre-flight must compute **maximum usable context** after loading weights + draft buffers, not just "does it fit?" |
| Prompt-speed regression | "MTP requires device-to-host transfers during prefill" | Prefill path with speculative decode should warn that prompt throughput drops ~25%. Not a bug — architectural. |

---

## Concrete Numbers Summary

| Metric | 27B + MTP (q8, max 2) | 27B Standard (q8) | 35B + MTP (q8, max 1) | 35B Standard (q8) |
|--------|----------------------|-------------------|----------------------|-------------------|
| Prompt speed (t/s) | ~150 | ~200 | ~277 | ~368 |
| Generation speed (t/s) | **75** | 45 | **186** | 146 |
| Avg context (tokens) | **40K** | **80K** | **15K** | **80K** |
| Context cost | **-50%** | — | **-81%** | — |
| Practical? | **Yes** | Yes | **No (16 GB)** | Yes |

---

## ROCmForge Takeaway

MTTP speculative decoding belongs in the inference stack, but it must be **VRAM-aware**.

The per-ModelEntry semaphore (INF-2) serializes requests to avoid GPU contention, which is correct. What's missing is the **VRAM accounting** — `ModelEntry::load` should estimate not just weight size but draft buffer size, KV cache size at target context, and report the maximum usable context to the caller. If the estimate falls below a configured floor (e.g., 64K for agentic workflows), the load should return a clear error: "Insufficient VRAM for model + speculative draft buffers at requested context size."

This is exactly what vLLM calls a "placement policy" and what llama.cpp does manually via `--ctx-size`. RocmForge can automate it because Rust's ownership model lets us track GPU buffer lifetimes precisely — a capability Python-based servers approximate with best-effort heuristics.

---

**Links to related wiki pages:**
- `[[rocmforge-production-inference-gaps]]` — INF-15 (speculative decode plumbing) and INF-11..INF-16
- `[[vllm-mlx-research-2026-06-03]]` — Comparative analysis of PagedAttention, continuous batching, prefix caching
- `[[rocmforge-changelog]]` — What has already shipped in GPU speculative engine
