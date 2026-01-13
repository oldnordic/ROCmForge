# Architectural Decision: GGUF Embedding Layout + GetRows (No Transpose)

**Date**: 2026-01-12  
**Status**: IMPLEMENTED (Updated)  
**Phase**: 25

---

## Decision

Keep GGUF embedding weights in their native layout and make **GetRows** the required path
for embedding lookup. **No transpose on load.**

GGUF stores embeddings as `[hidden_size, vocab_size]`. We preserve that layout and
teach GetRows to gather token rows efficiently from the stored matrix.

---

## Rationale

- **Aligns with llama.cpp**: get_rows is the canonical embedding path.
- **Avoids transpose cost**: no 519MB staging buffer or CPU/GPU transpose time.
- **Keeps IR clean**: layout is an execution concern, not a tensor rewrite.

---

## Implementation Summary

- **Loader**: remove embedding transpose in `src/loader/gguf.rs`.
- **ExecutionPlan**: record `embedding_layout` based on the raw GGUF shape.
- **ggml GetRows**: implement layout-aware gather:
  - `RowMajor` → contiguous D2D copy
  - `ColMajor` → strided D2D copy via `hipMemcpy2D`
- **Embedding lookup**: call ggml GetRows with recorded layout.

---

## Layout Contract

```
GGUF layout (native): [hidden, vocab] => Layout::ColMajor
Alternate layout:     [vocab, hidden] => Layout::RowMajor
```

GetRows uses:
- `n_embd` from the hidden dimension
- `n_vocab` from the vocab dimension
- layout to choose copy strategy

---

## Validation

- Embedding lookup succeeds without transpose.
- No extra one-time allocations for transpose.
- Layout-aware GetRows prevents shape mismatches and token bounds errors.

---

## References

- llama.cpp `ggml_get_rows` usage (logic only, not code)
- `docs/ARCH_DECISION_2026-01-12_OWNERSHIP_BOUNDARIES.md`
