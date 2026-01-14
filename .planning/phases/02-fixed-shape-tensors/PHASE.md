# Phase 2: Fixed-Shape Tensors with Offset Views

## Goal

Eliminate O(tokens) graph rebuilds by pre-allocating max-size tensors and using offset-based views.

## Problem

Currently in `src/model/execution_plan.rs:1578-1680`:

```rust
pub(crate) fn forward_layer_ggml_decode(...) {
    // Shape mutation EVERY token:
    let new_len = current_len + 1;
    graph.tensors[plan.kv_read_k_id.0].set_shape(vec![new_len, num_heads, plan.head_dim]);
    graph.tensors[plan.scores_id.0].set_shape(vec![1, new_len]);
    graph.tensors[plan.softmax_id.0].set_shape(vec![1, new_len]);
}
```

## Solution

Follow llama.cpp pattern:

```c
// llama.cpp pattern
struct ggml_tensor * K = ggml_new_tensor(..., max_seq_len, n_heads, head_dim);
// Use offset-based views:
struct ggml_tensor * Kcur = ggml_view_1d(K, ..., pos * n_heads * head_dim);
```

1. Pre-allocate max-size tensors at graph construction
2. Add position-aware `View` op variant
3. Update decode to use views with position offsets

## Files to Modify

- `src/model/execution_plan.rs` - `build_layer_ggml_plans()`, `forward_layer_ggml_decode()`
- `src/ggml/op.rs` - Add `View` with offset parameter
- `src/ggml/hip_backend/mod.rs` - Implement view op

## Success Criteria

- [ ] No graph rebuilds during token generation
- [ ] Token generation 10-15% faster
- [ ] All existing tests pass
