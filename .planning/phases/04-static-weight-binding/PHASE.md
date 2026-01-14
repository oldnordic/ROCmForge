# Phase 4: Static Weight Binding

## Goal

Bind weight tensors once at graph construction instead of every decode step.

## Problem

Weights are added to graphs and bound on every decode pass in `src/model/execution_plan.rs:1027-1576`:

```rust
fn build_layer_ggml_plans(&self, _backend: &HipBackend) -> HipResult<Vec<LayerGgmlPlan>> {
    for layer_plan in &self.layers {
        let qkv_weight = self.get_or_load_tensor(&layer_plan.qkv_weight)?;
        let o_proj = self.get_or_load_tensor(&layer_plan.o_proj)?;
        // ... all weights loaded and added to graph

        // But these bindings happen EVERY decode step through execute_graph()
    }
}
```

## Solution

Follow llama.cpp pattern - weights are part of static graph structure:

```c
// Weights bound once at graph construction
struct ggml_tensor * wq = ggml_new_tensor(..., GGML_TYPE_F16, ...);
ggml_set_tensor(wq, weight_data);  // Bind once, reuse forever
```

1. Separate static weight graphs from dynamic decode graphs
2. Bind weight tensors once at graph construction
3. Cache weight buffer bindings
4. Update executor to skip redundant binds

## Files to Modify

- `src/model/execution_plan.rs` - Separate weight/decode graphs
- `src/ggml/executor.rs` - Cache bindings
- `src/ggml/graph.rs` - Add static/dynamic graph distinction

## Success Criteria

- [ ] Weights bound once at initialization
- [ ] No per-token rebinding of static weights
- [ ] Token generation 5-10% faster
- [ ] All tests pass
