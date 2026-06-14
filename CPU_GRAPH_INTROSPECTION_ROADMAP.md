# CPU Graph Introspection Roadmap

> Saved discussion / plan. Treat this as a ladder: each step unlocks the next.

## Current state (locked and verified)

- `CpuGraphArena` stores captured CPU layer execution as stable `F32Handle` / `U8Handle` offsets.
- Parity test: graph replay matches direct execution with max error `0.00000000`.
- Temporal regression works: `graph.regress(t)` invalidates future nodes; `rebind_after_regress(t)` restores arena bindings so `read_back()` returns the rolled-back state.
- Search/rollback test: capture shared prefix, evaluate branch A, roll back, capture/evaluate branch B, both matching direct execution.

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

**Success criterion:** Rollback latency is O(shelf size), not O(number of ops). Test shows a 100-op prefix can be rolled back in constant time.

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

**Success criterion:** A test with a correct branch and an obviously broken branch (e.g., perturbed hidden) assigns a higher score to the correct one.

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

---

### Step 5 — Feedback loop as graph annotations

**Goal:** Introspection results persist and influence future inference.

**What to build:**
- Store `IntrospectionReport` inside `GraphMap`.
- Add a `branch_bias` field per branch.
- When replaying a new session, load biases from previous `GraphMap`s and prefer branches with higher bias.

**Success criterion:** A second search session, using biases from the first session, reaches the correct branch faster (fewer rollbacks) than the first session.

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

---

### Step 7 — Mistake-driven LoRA adapter (optional, CPU)

**Goal:** The system actually learns from traces, not just prompt-engineers around them.

**What to build:**
- Train a tiny LoRA adapter on top of the frozen 0.5B GGUF base.
- Training objective: predict the better branch given a graph summary.
- Use Candle or a minimal CPU trainer.

**Success criterion:** The adapter improves branch selection accuracy on a test set of traces compared to the base model alone.

---

### Step 8 — Open-weight experiments (optional, heavy)

**Goal:** Update the 0.5B base weights directly from trace feedback.

**What to build:**
- Load the 0.5B model as a trainable graph (Candle with gradients or PyTorch export).
- Run small gradient steps on the trace dataset from Step 6.
- Keep sequences short to stay feasible on CPU.

**Success criterion:** After online updates, the model makes fewer mistakes on a held-out reasoning task derived from the same domain.

---

## Hard constraints

- The 0.5B Qwen weights are fixed in GGUF unless a training step is explicitly added.
- Context length is limited; graph summaries must be compressed before entering the model prompt.
- CPU training is slow; Steps 7 and 8 should be treated as experiments, not production paths.

## Next action

The recommended first move is **Step 1**: persist a `GraphMap` through `geographdb-core`. It unlocks every later step and requires no model changes.
