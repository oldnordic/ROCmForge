# Phase 6 Quick Fix Guide

## Immediate Actions Required

### Fix 1: tests/loader_tests.rs (5 minutes)

**Problem**: Type renamed from `GgufDataType` to `GgufTensorType`, method `.size()` → `.element_size()`

**Fix**: Run these replacements:
```bash
sed -i 's/GgufDataType/GgufTensorType/g' tests/loader_tests.rs
sed -i 's/\.element_size()/.element_size()/g' tests/loader_tests.rs
```

Or manually update lines 16-40 and 52-87.

### Fix 2: tests/test_direct_cpu.rs (5 minutes)

**Problem**: File should have been deleted in Phase 6

**Fix**: Delete the file:
```bash
rm tests/test_direct_cpu.rs
```

This is a debug helper that's no longer needed.

### Fix 3: tests/embedding_to_lmhead_tests.rs (30 minutes)

**Problem**: Module `gguf_loader` consolidated into `gguf`, API changed completely

**Fix**: Update imports (line 3):
```rust
// OLD:
use rocmforge::loader::gguf_loader::{GgufLoader, GgufModel, GgufTensor};

// NEW:
use rocmforge::loader::{GgufLoader, GgufTensor, GgufTensorType};
```

Update function signature (line 10):
```rust
// OLD:
fn extract_model_config_from_gguf(model: &GgufModel) -> ModelConfig {

// NEW:
fn extract_model_config_from_gguf(loader: &GgufLoader) -> ModelConfig {
```

Update function body (lines 13-52):
```rust
// OLD:
let num_layers = model.metadata.get("llama.block_count")
    .or_else(|| model.metadata.get("block_count"))
    .and_then(|s| s.parse().ok())
    .unwrap_or(32);

// NEW (access GgufMetadata fields directly):
let num_layers = loader.metadata().num_layers;

// OR (keep HashMap access):
let num_layers = loader.metadata()
    .get("llama.block_count")
    .or_else(|| loader.metadata().get("block_count"))
    .and_then(|s| s.parse().ok())
    .unwrap_or(32);
```

Update helper function (line 80):
```rust
// OLD:
fn find_tensor_by_pattern<'a>(model: &'a GgufModel, patterns: &[&str]) -> Option<&'a GgufTensor> {
    model.tensors.iter()
        .find(|(name, _)| patterns.iter().any(|pattern| name.contains(pattern)))
        .map(|(_, tensor)| tensor)
}

// NEW:
fn find_tensor_by_pattern(loader: &GgufLoader, patterns: &[&str]) -> Option<&GgufTensor> {
    loader.load_tensors().unwrap()
        .iter()
        .find(|(name, _)| patterns.iter().any(|pattern| name.contains(pattern)))
        .map(|(_, tensor)| tensor)
}
```

Update type references (lines 95, 100, 103, 107, 110, 113, 116):
```rust
// OLD:
rocmforge::loader::gguf_loader::GgufDataType::F32
rocmforge::loader::gguf_loader::GgufDataType::F16

// NEW:
rocmforge::loader::GgufTensorType::F32
rocmforge::loader::GgufTensorType::F16
```

---

## Verification

After fixes:
```bash
cargo test --workspace --no-run
cargo test --workspace
```

Should see: "0 errors" and all tests compile successfully.
