# CLI to GPU Workflow Analysis

**Date**: 2026-01-12
**Phase**: 24 - Debugging Model Loading Timeout
**Status**: Analysis in Progress

---

## Problem Statement

Running `rocmforge_cli generate --gguf models/qwen2.5-0.5b.gguf --prompt "Hi"` results in timeout during model loading. The code allocates ~2.1GB of GPU memory then hangs.

---

## Complete Workflow Trace

### 1. CLI Entry Point (`src/bin/rocmforge_cli.rs:532`)

```rust
async fn create_engine(gguf: &str) -> anyhow::Result<Arc<InferenceEngine>> {
    let mut engine = InferenceEngine::new(EngineConfig::default())?;
    engine.load_gguf_model(gguf).await?;
    let engine = Arc::new(engine);
    engine.start().await?;
    ...
}
```

### 2. InferenceEngine::new() (`src/engine.rs:88`)

Creates with **EngineConfig::default()**:
```rust
EngineConfig {
    max_batch_size: 32,
    max_sequence_length: 4096,
    cache_page_size: 16,
    max_cache_pages: 1000,    // ← PROBLEM: Creates paged cache with wrong values!
    num_heads: 32,            // ← WRONG for qwen2.5-0.5b (has 14)
    head_dim: 128,            // ← WRONG for qwen2.5-0.5b (has 64)
    num_layers: 24,           // ← WRONG for qwen2.5-0.5b (has 24 - this one is correct by luck)
    ...
}
```

Creates **Paged KvCache** (`src/kv_cache/kv_cache.rs:361`):
- `max_pages = 1000`
- `num_heads = 32`
- `head_dim = 128`
- `page_size = 16`
- `PhysicalBlockPool::new(1000, ...)` allocates **2000 buffers** (1000 blocks × 2 K+V)

### 3. load_gguf_model() (`src/engine.rs:142`)

```rust
let runtime = tokio::task::spawn_blocking(move || {
    ModelRuntime::load_from_gguf(&path_string)
}).await??;
```

### 4. ModelRuntime::load_from_gguf() (`src/backend/hip_backend.rs:2346`)

```rust
pub fn load_from_gguf(path: &str) -> HipResult<Self> {
    let loader = GgufLoader::new(path)?;
    let config = loader.to_model_config()?;

    let backend = HipBackend::new()?;

    // Creates FIRST ScratchBufferManager (with wrong default values!)
    let scratch = ScratchBufferManager::new(
        &backend,
        config.num_attention_heads,  // 14 for qwen2.5-0.5b
        config.max_position_embeddings,  // 2048 or default
        config.head_dim,  // 64 for qwen2.5-0.5b
        config.hidden_size,  // 896 for qwen2.5-0.5b
    )?;

    // Creates FIRST simple KVCache (correct model values)
    let kv_cache = KVCache::new(
        &backend,
        config.num_hidden_layers,  // 24
        config.num_attention_heads,  // 14
        config.head_dim,  // 64
        config.max_position_embeddings,  // 2048
    )?;

    // Creates ExecutionPlan
    let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;
    ...
}
```

### 5. ExecutionPlan::from_gguf() (`src/model/execution_plan.rs:318`)

```rust
pub fn from_gguf(backend: &HipBackend, loader: &GgufLoader) -> HipResult<Self> {
    let config = loader.to_model_config()?;
    let lazy_tensors = &loader.lazy_tensors;
    let architecture = Self::detect_architecture(&tensor_names)?;

    // Map embedding - triggers some allocation
    let embedding_weights_lazy = Self::map_embedding_lazy(...)?;

    // Create layers - MAY trigger additional allocations
    for layer_idx in 0..config.num_hidden_layers {
        let layer_plan = Self::create_layer_plan_lazy(...)?;
        layers.push(layer_plan);
    }
    ...
}
```

---

## Allocation Summary (Observed)

| Order | Component | Config | Memory |
|-------|-----------|--------|--------|
| 1 | Paged PhysicalBlockPool | 1000 blocks, 32 heads, 128 dim, page_size=16 | ~2000 buffers (K+V) |
| 2 | ModelRuntime ScratchBuffer | 14 heads, 896 hidden, 896 seq_len | ~18MB |
| 3 | ModelRuntime KVCache (simple) | 24 layers, 14 heads, 64 dim, 2048 seq | ~168MB |
| 4 | ??? ScratchBuffer | 32 heads, 4096 seq_len | **544MB** |
| 5 | InferenceEngine KVCache (simple) | 32 layers, 32 heads, 128 dim, 2048 seq | ~1GB |
| 6 | ModelRuntime KVCache (simple) | 24 layers, 14 heads, 64 dim, 2048 seq | ~168MB |

**Total**: ~2.1GB allocated

---

## Root Causes Identified

### Issue #1: `ensure_request_state()` Creates Duplicate ModelRuntime

**Location**: `src/engine.rs:321-360`

**Problem**: When the first request is submitted, `ensure_request_state()` creates a NEW `ModelRuntime::new()` from scratch:
```rust
// From ensure_request_state():
let new_runtime = tokio::task::spawn_blocking(move || {
    let temp_base = crate::backend::hip_backend::ModelRuntime::new()  // ← Creates 32-layer KV cache!
        .map_err(...)?;
    temp_base.from_execution_plan(execution_plan)
        .map_err(...)
}).await??;
```

**Impact**:
- `ModelRuntime::new()` (line 2314) creates ANOTHER KV cache with default values (32 layers)
- This defeats the purpose of loading the model with `load_from_gguf()` which has correct config
- Wastes ~1GB of GPU memory per request!

### Issue #2: Wrong Default Configuration

**Problem**: `ModelRuntime::new()` uses hardcoded default values:
```rust
let scratch = ScratchBufferManager::new(
    &backend, 32,   // num_heads (wrong for qwen2.5-0.5b which has 14)
    2048,           // max_seq_len
    128,            // head_dim (wrong for qwen2.5-0.5b which has 64)
    4096,           // hidden_size
)?;

let kv_cache = KVCache::new(
    &backend, 32,   // num_layers (somewhat correct)
    32,             // num_heads (WRONG)
    128,            // head_dim (WRONG)
    2048,           // max_seq_len
)?;
```

**Impact**:
- ScratchBuffer creates 544MB attention_scores buffer
- KV cache allocates ~1GB with wrong dimensions
- These caches are NEVER USED - they're discarded immediately!

### Issue #3: Architecture Design Flaw

**Problem**: `ensure_request_state()` was designed for a multi-model scenario where each request might use a different model. But in the current CLI usage:
- Only ONE model is loaded
- ALL requests use the same model
- Creating per-request ModelRuntimes is wasteful

**Correct Design**: Reuse the already-loaded ModelRuntime from `model_runtime` field instead of creating new ones.

---

## Proposed Fixes

### Fix #1: Reuse Loaded ModelRuntime (CRITICAL)

**Location**: `src/engine.rs:321-360` (`ensure_request_state`)

**Current Code**:
```rust
let new_runtime = tokio::task::spawn_blocking(move || {
    let temp_base = crate::backend::hip_backend::ModelRuntime::new()  // ← Creates wasteful KV cache
        .map_err(...)?;
    temp_base.from_execution_plan(execution_plan)
        .map_err(...)
}).await??;
```

**Fixed Code**:
```rust
// Reuse the already-loaded ModelRuntime instead of creating a new one
let runtime_arc = self
    .model_runtime
    .as_ref()
    .ok_or_else(|| EngineError::InferenceFailed("No GGUF model loaded".to_string()))?
    .clone();

// Clone the existing ModelRuntime for this request
let new_runtime = tokio::task::spawn_blocking(move || {
    // Clone the ModelRuntime (shares backend, execution_plan, etc.)
    let base_runtime = runtime_arc.read().await;
    // Create request-specific KV cache from the execution plan
    base_runtime.clone_for_request()
        .map_err(|e| EngineError::InferenceFailed(e.to_string()))
}).await??;
```

**Impact**: Eliminates ~1GB of wasteful allocations per request

### Fix #2: Remove ModelRuntime::new() Default Allocator

**Location**: `src/backend/hip_backend.rs:2314`

**Current**: `ModelRuntime::new()` creates KV cache with hardcoded defaults

**Fix**: Make `ModelRuntime::new()` private or deprecated. Use `load_from_gguf()` for all model loading.

### Fix #3: Add InferenceEngine::from_gguf()

**Location**: `src/engine.rs`

**Add**:
```rust
impl InferenceEngine {
    /// Load GGUF model and create engine with correct config
    pub async fn from_gguf<P: AsRef<std::path::Path>>(path: P) -> EngineResult<Self> {
        // Load model first to get config
        let loader = GgufLoader::new(path.as_ref())
            .map_err(|e| EngineError::ModelLoadFailed(e.to_string()))?;
        let config = loader.to_model_config()
            .map_err(|e| EngineError::ModelLoadFailed(e.to_string()))?;

        // Create engine with model-specific config
        let engine_config = EngineConfig {
            max_batch_size: 32,
            max_sequence_length: config.max_position_embeddings,
            cache_page_size: 16,
            max_cache_pages: 100,  // Reduced for smaller models
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
            num_layers: config.num_hidden_layers,
            batch_timeout: Duration::from_millis(50),
        };

        let mut engine = Self::new(engine_config)?;
        engine.load_gguf_model(path).await?;
        Ok(engine)
    }
}
```

---

## Implementation Priority

1. **[CRITICAL]** Fix `ensure_request_state()` to reuse loaded ModelRuntime
2. **[HIGH]** Add `InferenceEngine::from_gguf()` for proper initialization
3. **[MEDIUM]** Remove or deprecate `ModelRuntime::new()` with default values
4. **[LOW]** Optimize KV cache allocation (use paged attention)

---

## Files Modified for Debugging

- `src/backend/hip_backend.rs:1227` - Added allocation counter
- `src/model/kv_cache.rs:60` - Added eprintln! for KVCache::new()
- `src/model/execution_plan.rs:319` - Added eprintln! for from_gguf()
- `src/backend/scratch.rs:39` - Added eprintln! for ScratchBufferManager::new()
- `src/bin/rocmforge_cli.rs:533` - Added tracing to create_engine()
- `src/engine.rs:88` - Added tracing to InferenceEngine::new()
- `src/backend/hip_backend.rs:2375` - Added tracing to load_from_gguf()
