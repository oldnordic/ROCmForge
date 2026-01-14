# Phase 1: Single-Pass GGUF Loading

## Goal

Eliminate redundant GGUF parsing. Load metadata once, reuse for both config and weights.

## Problem

Currently in `src/engine.rs:163-197`:

```rust
pub async fn from_gguf<P: AsRef<std::path::Path>>(path: P) -> EngineResult<Self> {
    // GgufLoader #1 - just to extract config
    let loader = GgufLoader::new(&path_string)?;
    let config = loader.to_model_config()?;

    let mut engine = Self::new(engine_config)?;
    engine.load_gguf_model(path).await?;  // Creates GgufLoader #2 inside
}
```

The GGUF file is parsed twice:
1. First to extract model config
2. Second during actual weight loading

## Solution

1. Parse GGUF once, wrap in `Arc` for sharing
2. Add `load_gguf_model_with_loader(Arc<GgufLoader>)` method
3. Update `from_gguf()` to reuse the same loader

## Files to Modify

- `src/engine.rs` - `InferenceEngine::from_gguf()`
- `src/loader/gguf.rs` - Possibly add `Arc` wrapper support

## Success Criteria

- [ ] Single GGUF parse per model load
- [ ] Startup time reduced by 20-30%
- [ ] All existing tests pass
