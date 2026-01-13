# ADR: ggml IR Ownership, Backend Layering, and Graph Lifetime

**Date**: 2026-01-13
**Status**: Proposed

## Context

ROCmForge needs a stable, vendor-agnostic op IR to avoid embedding transpose stalls and to isolate HIP specifics. Phase 24/25 confirmed engine/request separation and exposed embedding/transposition bottlenecks. Llama.cpp/ggml shows embeddings are stored as `[n_embd, n_vocab]` and consumed via `get_rows`, avoiding full transpose.

## Decisions

1) **IR module name + ownership**
- Create `src/ggml/` as the IR + semantic contract layer.
- `src/ggml/backend.rs` defines `trait GgmlBackend`.
- Rationale: the layer defining ops owns the backend trait; avoid dependency inversion and leakage.

2) **Backend layering**
- `src/backend/hip_backend.rs` remains low-level hardware plumbing only (streams, buffers, alloc).
- `src/ggml/hip_backend/` implements ggml ops atop hip_backend.

3) **GQA handling**
- GQA is addressed at graph construction time (tensor wiring, head expansion/aliasing).
- No "GQA op" is added to the IR.
- Rationale: topology is not math; keep op surface minimal and vendor-agnostic.

4) **Graph lifetime**
- ggml graphs are persistent per model/session.
- Requests bind inputs and KV cache views; no graph rebuild per request.
- Rationale: aligns with Phase 24 ownership boundaries, reduces CPU churn, preserves determinism.

5) **Embedding layout**
- `token_embd.weight` stored and consumed as `[n_embd, n_vocab]`.
- `GetRows` is mandatory for embeddings.
- Full transpose path is removed.

## Consequences

- Clear ownership boundaries: ggml owns semantics; hip_backend owns hardware details.
- Easier backend extension (CPU, ROCm) with the same op set.
- Improved load-time performance: no full embedding transpose.
- Graph build is slightly more complex, but only done once per model.

## Alternatives Considered

- Backend trait under `src/backend/` (rejected: ownership inversion, future leakage).
- GQA as an op (rejected: model policy in IR, op surface bloat).
- Graph rebuild per request (rejected: CPU churn, breaks artifact reuse).

## References

- `docs/implementation_principles.md`
- `docs/PHASE_24_STATUS_2026-01-12.md`
- `docs/PHASE_25_STATUS_2026-01-13.md`
- `docs/ARCHITECTURE_MODULARIZATION_ANALYSIS.md`
