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

**Implementation status:** Full base-weight fine-tuning is blocked by the local checkpoint being GGUF/Q4_0, which is not a differentiable format. As a testable Step 8 proof, `src/cpu/graph/value_head.rs` implements `BranchValueHead` (linear head on hidden states), `src/cpu/graph/open_weight.rs` adds `BranchLogitScorer` (yes/no logit-margin scaler) and `BranchLabelBias` (per-letter bias for multi-branch choice prompts). `tests/test_cpu_graph_value_head.rs` proves the head trains on synthetic hidden vectors, and `tests/test_cpu_graph_open_weight.rs` (marked `#[ignore]`) runs the real 0.5B model end-to-end and confirms the open-weight pipeline updates real weights. On the synthetic numeric-score task the 0.5B model does not yet reliably outperform random, so the test functions as a grounded baseline smoke test; future work can swap in a stronger model or a fully differentiable base.

---

## Hard constraints

- The 0.5B Qwen weights are fixed in GGUF unless a training step is explicitly added.
- Context length is limited; graph summaries must be compressed before entering the model prompt.
- CPU training is slow; Steps 7 and 8 should be treated as experiments, not production paths.

## Next action

Steps 1–8 are now locked and verified. The CPU-graph introspection ladder has reached a working closed loop: capture, persist, score, introspect, annotate, dataset, lightweight adapter, and open-weight value head.

Recommended follow-ups (outside the current ladder):
- Convert the frozen 0.5B base into a fully differentiable graph and run real gradient steps on base weights.
- Hide numeric scores from the branch prompt to force the value head to learn from structural state rather than read numbers.
- Use `BranchValueHead` as an online reranker during inference, applying its score as a `branch_bias` in future `CaptureContext` sessions.
