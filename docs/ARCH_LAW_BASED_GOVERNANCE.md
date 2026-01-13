# Architecture Proposal: Law-Based Code Governance

**Date**: 2026-01-13
**Status**: Design Proposal
**Related**: Magellan, Splice, CodeMCP tools

---

## Executive Summary

A proposal for **evidence-based law governance** of codebase architecture, where "laws" (invariants, constraints, boundaries) are:

1. **Discovered** via automated analysis (not assumed)
2. **Justified** with concrete evidence (call sites, tests, docs)
3. **Provisionally adopted** (guide but don't hard-fail initially)
4. **Evolved** only with proof (compiler, tests, benchmarks)

This turns architectural governance from "human rules" into "observable fact + verification loop".

---

## The Core Philosophy

### What the LLM Is NOT

- ❌ A Rust language lawyer with 10 years of tribal knowledge
- ❌ A framework veteran who knows all the unwritten rules

### What the LLM IS

- ✅ The **system architect** defining how truth is established
- ✅ The one deciding **what counts as proof**
- ✅ The one defining **how laws come into existence and change**

**The correct model**: Laws are derived hypotheses, not rules.

---

## The Four-Stage Pipeline

### 1️⃣ Discovery (Law Candidates)

A sub-agent scans the codebase using:

| Tool | What It Provides |
|------|------------------|
| **Magellan** | Symbols, call graphs, fan-in/fan-out |
| **Semantic Index** | Purpose, intent, naming patterns |
| **Git History** | Stability vs churn |
| **Docs/ADRs** | Explicit intent |

Produces **law candidates**, each with:
- `scope` - symbol / module / subsystem
- `proposed_role` - boundary, glue, API, leaf
- `proposed_constraint` - must_reuse, no_alloc, etc.
- `evidence_list` - links to concrete proof

**No enforcement yet. Just hypotheses.**

### 2️⃣ Justification (Mandatory)

Every proposed law MUST answer:

| Question | Required Evidence |
|----------|-------------------|
| What breaks if this changes? | Call sites, failure modes |
| Where is this reused? | Fan-out count, locations |
| What invariants does it enforce? | Tests, comments, behavior |
| Is this intentional or accidental? | Docs, commit history, naming |

**If the agent can't produce evidence → no law.**

This filters out hallucination at the source.

### 3️⃣ Provisional Adoption (Soft Law)

Accepted laws start as **provisional**:
- Guide the LLM
- Warn on violation
- Don't hard-fail yet

Important because:
- Early laws will be imperfect
- The system must be allowed to learn
- False positives are expected initially

### 4️⃣ Law Evolution (Safe Refactoring)

When the LLM believes a law is wrong:

```
1. Detect conflict
   "I cannot implement this change without violating law X"

2. Build proof
   - Show alternative structure
   - Show reduced duplication
   - Show preserved invariants
   - Show tests passing
   - Show removal of previous failure modes

3. Verify
   - Compile ✓
   - Tests ✓
   - Benchmarks (optional) ✓

4. Propose law change
   - Modify / relax / remove
   - Attach proof artifacts
```

**Law changes become refactor ADRs, not guesses.**

---

## Why This Is Safe

The LLM is **never trusted directly**. It is constrained by:

| Constraint | Source |
|------------|--------|
| Observable structure | Magellan graph |
| Historical evidence | Git history |
| Test outcomes | Cargo test |
| Explicit proof requirements | System design |

The LLM becomes:
- A **theorem proposer**, not a judge
- Truth is decided by: codebase, compiler, tests, documented evidence

---

## Three-Layer Graph Architecture

### Layer 1: Authoritative Graph (SQLiteGraph)

```
Persistent
Auditable
Versioned
Slower
```

**This is truth.** All laws reference this as source of fact.

### Layer 2: In-RAM Working Graph

```
Loaded from SQLiteGraph
Symbol IDs only (no raw text)
Fan-in / fan-out edges
Law tags
Hot metadata
```

**This is thinking space.**

What it enables:
- Fast law discovery (repeated graph traversals)
- Proof-based refactors (simulate before editing)
- Drift detection before edits (check laws in O(ms))

What it should NOT contain:
- ❌ Raw source code
- ❌ Long docs
- ❌ Semantic embeddings
- ❌ Free-text comments

Keep it: symbolic, structural, compact, invalidatable.

### Layer 3: Ephemeral Reasoning Graphs

```
Sub-agent specific
Filtered views
Temporary hypotheses
Discarded after decision
```

**This is scratch paper.**

---

## In-RAM Graph Design

### Data Structure

Don't need a full DB. Just:

```rust
// Node storage
slotmap::SlotMap<SymbolId, SymbolNode>

// Edge storage (adjacency list)
Vec<Vec<EdgeId>>

// Symbol node
struct SymbolNode {
    kind: SymbolKind,
    file: FileId,
    module: ModuleId,
    law_tags: Vec<LawId>,
    fan_in: usize,
    fan_out: usize,
}

// Edge
struct Edge {
    from: SymbolId,
    to: SymbolId,
    edge_type: EdgeType,  // CALLS, USES, OWNS, etc.
}
```

### Load / Invalidation

- **Loaded at**: Project open, or Magellan init
- **Updated on**: File change, splice apply, index refresh
- **Invalidated via**: Ops/events, generation counters

---

## What This Enables

| Capability | How |
|------------|-----|
| **Fast law discovery** | Walk call graphs repeatedly, cluster "glue" symbols, detect implicit boundaries |
| **Proof-based refactors** | Simulate removal, recompute affected nodes, check invariants before touching code |
| **Drift detection** | Before splice: compute impact, check law violations, abort if no proof provided |
| **Tool-guided LLM** | LLM asks graph questions: "what's central?", "what's reused?", "what allocates?" |

---

## The Critical Rule

**No law exists without evidence.**
**No law changes without proof.**

That's it.

You don't need Rust mastery. You don't need to "know better than the LLM".

You designed a system where:
- Laws emerge from **reality**
- Drift is detected **structurally**
- Refactors are allowed, but must **justify themselves**

---

## Why This Beats Human-Defined Laws

| Human Laws | Your System |
|------------|-------------|
| Incomplete rules | Tied to evidence |
| Forgot why they exist | Evidence attached |
| Violated under pressure | Re-proof required for changes |
| Tribal knowledge | Observable structure |

---

## Example: Finding a Shape Corruption Bug

### Without Law-Based Governance

```
"Maybe it's in DeviceTensor..."
"Could be a cache issue..."
[Trial and error]
```

### With Law-Based Governance

```
1. Discovery: What laws govern weight shapes?
   → Law: "Weight shapes must be preserved from GGUF to matmul"

2. Evidence: Where is this law enforced?
   → Trace code path: load_tensor_to_gpu → LazyTensor → matmul
   → Read actual source at each transition

3. Proof: Find exact line of corruption
   → "I read src/model/execution_plan.rs:1027-1103..."
   → "Line X truncates shape from [2688,896] to [896,896]"

4. Fix with proof
   → Change preserves invariant
   → Tests pass
   → Law no longer violated
```

---

## Implementation Roadmap

### Phase 1: Evidence Gathering (Current State)
- Magellan provides structure
- Semantic index provides meaning
- Already have the foundation

### Phase 2: Law Discovery
- Sub-agent scans for law candidates
- Generates evidence lists
- No enforcement yet

### Phase 3: Provisional Laws
- Store laws in operations.db
- Tag symbols with law IDs
- Warn on violation

### Phase 4: In-RAM Working Graph
- Load hot subgraph into memory
- Fast traversal for law checking
- Cache invalidation on changes

### Phase 5: Proof-Based Evolution
- Law changes require proof artifacts
- Compiler/test verification
- ADR generation for law changes

---

## References

- **Magellan**: Code graph database (symbols, edges, references)
- **SQLiteGraph**: Durable truth storage
- **Semantic Index**: Purpose tags, intent discovery
- **Operations DB**: Law storage, evolution tracking

---

## Status

📋 **Design Phase** - This is a proposal for future enhancement to Magellan/Splice tooling.

The current bug investigation (QKV shape corruption) demonstrates the **philosophy** in action:
- Evidence-based debugging
- No guessing, only observable structure
- Proof required before fixes
