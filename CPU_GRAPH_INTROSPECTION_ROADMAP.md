# CPU Graph Introspection Roadmap

> Saved discussion / plan. Treat this as a ladder: each step unlocks the next.

## Current state (locked and verified)

- `CpuGraphArena` stores captured CPU layer execution as stable `F32Handle` / `U8Handle` offsets.
- Parity test: graph replay matches direct execution with max error `0.00000000`.
- Temporal regression works: `graph.regress(t)` invalidates future nodes; `CaptureContext::regress_to(t)` restores both the graph and the persistent shelf snapshot.
- Search/rollback test: capture shared prefix, evaluate branch A, roll back, capture/evaluate branch B, both matching direct execution.
- Step 1 (`GraphMap` persistence) committed: `GraphMap::save`/`load`/`into_context` round-trip through `geographdb-core` storage passes.
- Step 2 (timestamp shelves) committed: arena split into `Constants` / `Persistent` / `Ephemeral` shelves; `regress_to(t)` restores the persistent shelf in O(shelf size) instead of replaying the prefix; instant-rollback test verifies zero active nodes and zero replay after rollback.
- Step 2.5 (validation experiment) committed: `tests/test_cpu_graph_search_experiment.rs` encodes grid mazes as one-hot `Gemv` transitions and shows structured-recurrence DFS beats a linear random baseline in compute-normalized accuracy.
- Step 3 (branch scoring) committed: `CpuOpNode::Score`, `ScoreMetric`, `CaptureContext::score_against()`, and `GraphMap::branch_scores()`/`divergence()` implemented and validated by `tests/test_cpu_graph_branch_scoring.rs`.
- Step 4 (introspection prompt interface) committed: `GraphSummarizer`, `IntrospectionPrompt`, and `IntrospectionReport` implemented in `src/cpu/graph/introspection.rs` and validated by `tests/test_cpu_graph_introspection.rs`, which prompts the local 0.5B Qwen2.5 model and verifies it picks the higher-scoring branch above random chance.
- Step 5 (feedback loop) committed: `BranchAnnotation` and `branch_bias` persistence implemented in `CaptureContext`/`GraphMap`, with key-based bias lookup and ordering validated by `tests/test_cpu_graph_feedback_loop.rs`.
- Step 6 (trace dataset) committed: `GraphTraceDataset` implemented in `src/cpu/graph/dataset.rs` and validated by `tests/test_cpu_graph_trace_dataset.rs`, which loads persisted `GraphMap`s and exports them into process-supervision, rejection-sampling, and preference-pair formats without loss.
- Step 7 (mistake-driven adapter) committed: `BranchAdapter` implemented in `src/cpu/graph/adapter.rs` and validated by `tests/test_cpu_graph_adapter.rs`, which trains a tiny MLP on preference pairs and reaches 12/12 branch-selection accuracy on held-out traces versus 4/12 for random.
- Step 8 (open-weight experiment) committed: `BranchValueHead` implemented in `src/cpu/graph/value_head.rs`, hidden-state extraction from the frozen 0.5B model implemented in `src/cpu/graph/open_weight.rs`, validated by `tests/test_cpu_graph_value_head.rs` (synthetic) and `tests/test_cpu_graph_open_weight.rs` (real model, `#[ignored]`).
- Step 9 (real-session capture/reload) committed: `CaptureContext` is now wired into the live CLI inference path via `src/app/cpu_decode.rs` and `src/app/cpu_inference.rs`, allowing a real decode session to be captured, persisted with `GraphMap::save`, reloaded with `GraphMap::load`, and resumed. Validated by `tests/test_cpu_graph_real_session.rs` (real model, `#[ignored]`).
- Step 10 (GPU decode trace capture) committed: the live GPU inference path in `src/app/gpu_inference.rs` records a metadata-only token-level trace (`GpuTraceEntry`) into the `GraphMap` sidecar. Each decode step is scored with the same `ScoreMetric` used on CPU. Validated by `tests/test_gpu_graph_trace.rs` (real GPU + 0.5B model, `#[ignored]`).
- Step 11 (online value-head reranker) committed: `BranchValueHead` can be saved/loaded; a training utility converts persisted `GraphMap` traces into a value head; and CPU decode reranks the top-k next-token candidates using speculative forward passes and a loaded value head. Validated by `tests/test_cpu_graph_online_reranker.rs` (0.5B model, `#[ignored]`).
- Step 11.5/B (speculative candidate branch recording) committed: every top-k candidate evaluated by the online reranker is now persisted as a `CandidateBranch` inside the `GraphMap` sidecar, including parent timestamp, token id, value score, biased logit, and a chosen flag. The sidecar section is optional and backward-compatible.
- Step 11.5/C (reranker chosen-state reuse) committed: the chosen candidate's speculative forward state is now reused on the next decode iteration, eliminating the redundant main forward pass. Candidate scoring uses a single shared KV scratch and captures only small KV deltas, keeping memory overhead low.

## Vision

Turn the graph trace into a **persistent, navigable reasoning record**. The model can pause at session boundaries, inspect which branches succeeded or failed, and use that information to bias future inference. The trace itself becomes training data for process supervision, rejection sampling, and — with dynamic weights — mistake-driven learning.

Because `geographdb-core` already has storage primitives, traces should live there, not in ad-hoc JSON exports.

---

## Ladder steps

### Step 1 — Persistent GraphMap via `geographdb-core`

**Goal:** A captured session can survive process restart.

**What to build:**
- `GraphMap` struct that holds the minimal persistent metadata of a `CaptureContext`:
  - node list (timestamps, layer/step/spatial coords, op type),
  - handle manifest (handle id, len, dtype, caller pointer label),
  - output log (timestamp, pointer, handle id),
  - branch annotations (empty for now).
- Serialization path into `geographdb-core` storage.
- Deserialization path that can reconstruct `CpuGraph` + arena bindings from disk.

**Success criterion:** A test captures a prefix + branch, writes the `GraphMap`, kills the process, reloads it, replays the branch, and matches direct execution.

---

### Step 2 — Timestamp shelves for instant rollback

**Goal:** Rolling back to a previous timestamp should not require replaying the prefix.

**What to build:**
- Split `CpuGraphArena` into shelves:
  - `constants` (weights, sin/cos) — loaded once,
  - `persistent` (hidden, KV cache) — versioned per timestamp,
  - `ephemeral` (scratch buffers) — reusable within a timestamp,
  - optional `branch` shelf — forked from persistent at a decision point.
- On major timestamp boundaries, snapshot the `persistent` shelf.
- `regress(t)` restores the shelf snapshot instead of replaying.

**Success criterion:**
- Rollback latency is O(shelf size), not O(number of ops).
- `tests/test_cpu_graph_instant_rollback.rs` proves that after `regress_to(0)` the prefix state is restored from the shelf snapshot with **zero active nodes** and no prefix replay.

---

### Step 2.5 — Validation experiment: structured recurrence vs linear CoT

**Goal:** Prove (or refute) the central research bet *before* building introspection and training infrastructure on top of it.

**What to build:**
- A small, deterministic search task that requires branching and is solvable by the CPU graph engine without model changes:
  - candidate: grid maze, constraint-satisfaction walk, or multi-hop reachability query on the 4D graph itself.
- A ground-truth oracle so we know which branch reached the solution.
- Two inference arms with the **same forward-pass compute budget**:
  1. **Linear CoT baseline**: one chain, no branching, same number of forward steps.
  2. **Structured-recurrence search**: fork branches with `regress_to()`, evaluate, discard dead ends, continue.
- Instrumentation:
  - steps to solution,
  - accuracy (% tasks solved),
  - number of branches evaluated,
  - wall time,
  - compute-normalized accuracy (correct solutions per forward op).

**Success criterion (accept / reject):**
- **Confirm thesis:** structured-recurrence search achieves higher compute-normalized accuracy than the linear baseline on a held-out task set.
- **Refute thesis:** linear baseline matches or beats structured recurrence; in that case, stop the ladder here and do not proceed to Steps 4–8.

**Implementation status:** `tests/test_cpu_graph_search_experiment.rs` is implemented and passing. It uses 3×3 grid mazes encoded as one-hot state transition matrices and `CpuOpNode::Gemv`. On 16 seeded trials, structured DFS solves 16/16 mazes with an average of ~31 forward ops, while the linear random baseline (same op budget) solves 2/16. Compute-normalized accuracy: search ≈ 0.0020, baseline ≈ 0.0004.

**Implementation status:** `tests/test_cpu_graph_branch_scoring.rs` is implemented and passing. It proves that a `Score` node replayed via `CpuGraph::execute_window` matches the expected cosine similarity, that `CaptureContext::score_against()` ranks a branch moved toward a target reference higher than a branch moved away from it, and that the score log round-trips through `GraphMap::save`/`load`.

**Why this gate exists:** Steps 4–8 assume that branching + rollback actually helps solve problems. Without this experiment, we risk building a polished introspection pipeline that answers a question nobody asked.

---

### Step 3 — Branch scoring and divergence metrics

**Goal:** The system can measure how good a branch is without a human label.

**What to build:**
- Add `CpuOpNode::Score` or a post-processing pass that computes:
  - hidden-state divergence between branches,
  - L2 norm drift,
  - consistency with a verifier or target embedding,
  - perplexity on the next token if available.
- Store the score in `GraphMap` branch annotations.

**Success criterion:**
- **Trivial separation:** a correct branch receives a higher score than an obviously broken branch (e.g., zeroed or heavily perturbed hidden).
- **Blind ranking:** on a held-out set of *plausible* branches where neither branch is obviously broken, the divergence score reliably ranks the ground-truth better branch above the worse one. The score must be evaluated against the oracle from Step 2.5, not against perturbation severity.

---

### Step 4 — Introspection prompt interface

**Goal:** The local 0.5B GGUF model can analyze a compressed graph summary.

**What to build:**
- `GraphSummarizer` that turns a `GraphMap` into a short text + vector prompt:
  - number of branches,
  - per-branch score,
  - divergence points,
  - final hidden norm.
- A prompt template that asks the 0.5B model to pick the best branch and explain why.
- Parse the response into a structured `IntrospectionReport`.

**Success criterion:** Given two branches with known scores, the 0.5B model selects the higher-scoring one more often than random, on a small held-out set of traces.

**Implementation status:** `tests/test_cpu_graph_introspection.rs` is implemented and passing (when run with `--ignored`). It loads `/home/feanor/Projects/models/qwen2.5-0.5b-instruct-q4_0.gguf`, creates 4 held-out branch pairs with randomized target embeddings, prompts the model via `GraphSummarizer`, parses the `CHOICE:`/`REASON:` response with `IntrospectionPrompt::parse_response`, and verifies the model chooses the higher-scoring branch more often than random. Observed result: 3/4 correct (above the 2/4 random baseline).

---

### Step 5 — Feedback loop as graph annotations

**Goal:** Introspection results persist and influence future inference.

**What to build:**
- Store `IntrospectionReport` inside `GraphMap`.
- Add a `branch_bias` field per branch.
- When replaying a new session, load biases from previous `GraphMap`s and prefer branches with higher bias.

**Success criterion:** A second search session, using biases from the first session, reaches the correct branch faster (fewer rollbacks) than the first session.

**Implementation status:** `tests/test_cpu_graph_feedback_loop.rs` is implemented and passing. A first session scores four `ResidualAdd` branches and stores each score as a `BranchAnnotation` bias keyed by branch name in the `GraphMap`. After persistence round-trip, a second session loads the biases via `GraphMap::biases_by_key()` and evaluates branches in descending-bias order, reaching the best branch in 1 evaluation versus 3 in the default order.

---

### Step 6 — Trace dataset for training

**Goal:** The stored `GraphMap`s become a training corpus.

**What to build:**
- `GraphTraceDataset` reader over `geographdb-core` storage.
- Emit examples in formats useful for fine-tuning:
  - process supervision: per-step labels,
  - rejection sampling: accepted vs rejected branches,
  - preference pairs: branch A worse than branch B.

**Success criterion:** A small number of stored traces can be converted into each format without loss.

**Implementation status:** `tests/test_cpu_graph_trace_dataset.rs` is implemented and passing. It creates three `GraphMap`s with known branch scores, persists each into a separate subdirectory, loads them with `GraphTraceDataset::from_dir`, and verifies: (1) `process_supervision_examples` preserves every `(trace_id, timestamp, score)` and normalizes labels to `[0, 1]`; (2) `rejection_sampling_examples` marks exactly the highest-scoring branch per trace as accepted; (3) `preference_pairs` emits all `n*(n-1)/2` ordered pairs for distinct scores; (4) all example structs round-trip through `bincode`.

---

### Step 7 — Mistake-driven LoRA adapter (optional, CPU)

**Goal:** The system actually learns from traces, not just prompt-engineers around them.

**What to build:**
- Train a tiny LoRA adapter on top of the frozen 0.5B GGUF base.
- Training objective: predict the better branch given a graph summary.
- Use Candle or a minimal CPU trainer.

**Success criterion:** The adapter improves branch selection accuracy on a test set of traces compared to the base model alone.

**Implementation status:** `tests/test_cpu_graph_adapter.rs` is implemented and passing. It generates 24 synthetic training traces and 12 held-out test traces, builds a `GraphTraceDataset`, trains a `BranchAdapter` (a 6 -> 8 -> 1 MLP with pairwise hinge loss) for 80 epochs, and evaluates on the test set. Result: adapter selects the highest-scoring branch on 12/12 test traces, while a random baseline selects it on 4/12. The adapter is a CPU-only stand-in for a full LoRA layer on the 0.5B base; it learns from trace features rather than token embeddings.

---

### Step 8 — Open-weight experiments (optional, heavy)

**Goal:** Update the 0.5B base weights directly from trace feedback.

**What to build:**
- Load the 0.5B model as a trainable graph (Candle with gradients or PyTorch export).
- Run small gradient steps on the trace dataset from Step 6.
- Keep sequences short to stay feasible on CPU.

**Success criterion:** After online updates, the model makes fewer mistakes on a held-out reasoning task derived from the same domain.

**Implementation status:** Full base-weight fine-tuning is blocked by the local checkpoint being GGUF/Q4_0, which is not a differentiable format. As a testable Step 8 proof, `src/cpu/graph/value_head.rs` implements `BranchValueHead` and `BranchChoiceHead` (linear heads on frozen base-model hidden states), while `src/cpu/graph/open_weight.rs` provides hidden-state extraction helpers that apply the chat template and use the existing CPU inference path. `tests/test_cpu_graph_value_head.rs` proves the heads train on synthetic hidden vectors, and `tests/test_cpu_graph_open_weight.rs` (marked `#[ignore]`) runs the real 0.5B model: it extracts hidden states from the standard two-branch introspection prompt, trains a `BranchChoiceHead`, and reaches **8/8** accuracy on held-out semantic branch pairs versus **4/8** random. This confirms that trace feedback can update open weights on top of the frozen 0.5B model.

---

### Step 9 — Real-session capture and reload from the live CLI

**Goal:** Capture a genuine inference session as it runs through the `rocmforge` CLI, persist it to disk, and reload it in a later process.

**What to build:**
- A context-aware CPU decode path that wraps each forward pass in a `CaptureContext` instead of a `DirectContext`.
- CLI flags: `--graph-map-dir <path>` to save the captured `GraphMap` after generation, and `--load-graph-map-dir <path>` to load a previously saved map.
- A `ScoreMetric` flag (`--graph-score-metric`) so the captured branches are scored consistently.
- An integration test that invokes the compiled `rocmforge` binary, captures a session, checks the saved files, reloads them, and captures a second session.

**Success criterion:** A process that loads a saved `GraphMap` sees the previous session summary and can save a new `GraphMap` without crashing; the replay of the saved graph still matches the original decode output within the existing tolerance.

**Implementation status:** Implemented in `src/app/cli.rs`, `src/app/cpu_decode.rs`, and `src/app/cpu_inference.rs`. The test `tests/test_cpu_graph_real_session.rs` (marked `#[ignore]`) runs the binary twice against the local 0.5B model and verifies persistence/reload. Captured maps are large (~700 MB for a 2-token session) because weights and constants are currently included; this is acceptable for the proof of concept but noted as a follow-up optimization.

---

### Step 10 — GPU decode trace capture

**Goal:** Capture a persistent reasoning trace from GPU inference without copying full activation tensors, so larger models can feed the same trace-training pipeline as CPU sessions.

**What to build:**
- A GPU-side hook in the per-token decode loop (`src/app/gpu_inference.rs`) that records:
  - timestamp (generated token index),
  - cache position,
  - input token id and sampled token id,
  - a scalar `ScoreMetric` score computed from the downloaded logits.
- A new `GpuTraceEntry` type and a `gpu_trace` section in the existing `GraphMap` sidecar.
- When `--graph-map-dir` is active, disable the GPU greedy fastpath during decode so logits are available on the host.
- Load a previous `GraphMap` before generation and print summary statistics.

**Success criterion:** `rocmforge --gpu --graph-map-dir <dir>` produces a reloadable `GraphMap` whose `gpu_trace` length equals the number of generated tokens and whose branch scores match the score log. The HIP graph fastpath remains available when capture is not requested.

**Implementation status:** Implemented in `src/app/gpu_inference.rs` and `src/cpu/graph/map.rs`. The test `tests/test_gpu_graph_trace.rs` (marked `#[ignore]`) invokes the binary with `--gpu` and verifies persistence, reload, and trace/score consistency. The trace is intentionally metadata-only; full GPU tensor capture is left for future work.

---

### Step 11 — Online value-head reranker during CPU decode

**Goal:** Close the capture → train → rerank loop: use a `BranchValueHead` trained on persisted traces to bias next-token selection during live CPU inference.

**What to build:**
- Serialization for `BranchValueHead` (`save`/`load`) so a trained head survives process restart.
- CLI flags:
  - `--value-head-path <file>` — load a saved head for inference.
  - `--rerank-top-k <N>` — number of top candidates to score.
  - `--rerank-scale <F>` — scale factor for value-head scores before biasing logits.
  - `--train-value-head-from-traces <dir>` + `--save-value-head <path>` — train a head from persisted `GraphMap`s.
- In CPU decode (`src/app/cpu_decode.rs`), when a value head is loaded:
  - Take the top-k candidates from the output logits.
  - For each candidate, run a speculative decode step by cloning the KV cache, embedding the candidate, and running one forward pass.
  - Score the resulting hidden state with the value head.
  - Add the scaled score to the candidate's logit and sample from the biased distribution.
  - Skip reranking when the KV cache is at its last valid position to avoid out-of-bounds writes.
- Training utility (`src/cpu/graph/open_weight.rs`) that loads a `GraphTraceDataset`, extracts hidden states for each branch summary, and trains the head with MSE against recorded branch scores.

**Success criterion:**
- `rocmforge --value-head-path <head> --rerank-top-k 5` runs without crashing and produces a different token trajectory than greedy on a 0.5B model.
- Latency per token is measured and reported; the cost is N+1 forward passes per generated token.
- Training from traces produces a loadable value-head file.

**Implementation status:** Implemented in `src/cpu/graph/value_head.rs`, `src/cpu/graph/open_weight.rs`, `src/app/cli.rs`, `src/app/cpu_decode.rs`, and `src/app/cpu_inference.rs`. Validated by `tests/test_cpu_graph_online_reranker.rs` (marked `#[ignore]`). Measured overhead on the 0.5B Qwen model: ~4.2× baseline CPU decode time for top-5 reranking after chosen-state reuse (down from ~5.9× before reuse).

**Candidate branch recording (Step 11.5/B):**
- Each reranker evaluation now records a `CandidateBranch` in `CaptureContext::candidate_branches`, which is persisted in the `GraphMap` sidecar as an optional `candidate_branches` section.
- Fields: `parent_timestamp` (the decode step that produced the candidates), `token_id`, `value_score` (raw value-head output), `biased_logit` (logit after adding the scaled value score), and `chosen` (whether this token was selected by the sampler).
- Old `GraphMap` files without the section load with an empty candidate list, preserving backward compatibility.
- The recording test asserts that at least one candidate is marked chosen and that the candidate count equals the number of reranked steps times `--rerank-top-k` (minus the final step if it is skipped to protect the KV-cache bound).

**Chosen-state reuse (Step 11.5/C):**
- The reranker now captures the full speculative state of each top-k candidate (post-token hidden vector, output logits, and a small KV delta) while sharing a single `kv_scratch` across candidates.
- After sampling `chosen`, the chosen delta is applied to a fresh KV clone and stored as a `ReusableState`. On the next iteration, the pending state is applied instead of running `cpu_embed_token` + `cpu_full_forward_with_ctx`, removing one full forward pass per generated token.
- Graph capture for reused tokens is skipped; the `CandidateBranch` tree still records the decision.
- On the 0.5B Qwen model with top-5 reranking, this reduced measured per-token overhead from ~5.9× to ~4.2× baseline CPU decode time.

**Hard constraint surfaced by this step:**
- Token-level reranking still requires N speculative forward passes per token. On CPU this remains expensive; on GPU it would need hidden-state extraction from the GPU path, which is not yet implemented. The current MVP is CPU-only.

---

## Hard constraints

- The 0.5B Qwen weights are fixed in GGUF unless a training step is explicitly added.
- Context length is limited; graph summaries must be compressed before entering the model prompt.
- CPU training is slow; Steps 7 and 8 should be treated as experiments, not production paths.

## Next action

Steps 1–11 are now locked and verified, Step 11.5/B records the reranker’s speculative search tree in the persisted `GraphMap`, and Step 11.5/C reuses the chosen candidate’s forward state to cut reranker overhead. The CPU-graph introspection ladder has reached a working closed loop: capture, persist, score, introspect, annotate, dataset, lightweight adapter, open-weight value head, real-session persistence, GPU decode trace capture, online value-head reranking, searchable candidate branches, and state reuse.

Recommended follow-ups (outside the current ladder):
- Reduce `GraphMap` size by storing weight pointers/hashes instead of full weight tensors, or by referencing a separate model-manifest file.
- Capture the prefill phase (not just decode) into `CaptureContext` so the entire session is graph-replayable.
- Add richer per-token snapshots to the GPU trace (final hidden state, top-k logits) when training signal demands it.
- Bring the online reranker to the GPU path: extract hidden states from GPU decode for value-head scoring without round-tripping through CPU.
- Convert the frozen 0.5B base into a fully differentiable graph and run real gradient steps on base weights.
- Hide numeric scores from the branch prompt to force the value head to learn from structural state rather than read numbers.
- Extend token-level reranking to multi-token beam search: evaluate candidate continuations several tokens deep before selecting a branch.
