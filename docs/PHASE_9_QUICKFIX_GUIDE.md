# Phase 9 Bug Hunt - Quick Fix Guide

**Date**: 2026-01-06

This guide provides immediate fixes for the 6 critical bugs identified in the bug hunt.

---

## Fix 1: KV Cache Token Capacity (CRITICAL)

**Bug**: `CachePage` creates tokens Vec with capacity 0, preventing any tokens from being stored.

**Location**: `/home/feanor/Projects/ROCmForge/src/kv_cache/kv_cache.rs:83`

**Current Code**:
```rust
// Line 80-87
pub fn new(
    page_id: u32,
    sequence_id: u32,
    backend: &HipBackend,
    config: &CacheConfig,
) -> KvCacheResult<Self> {
    let key_size = config.page_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>();
    let value_size = config.page_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>();

    let key_buffer = backend.allocate_buffer(key_size)?;
    let value_buffer = backend.allocate_buffer(value_size)?;

    Ok(CachePage {
        page_id,
        sequence_id,
        tokens: Vec::new(),  // ❌ BUG: Capacity is 0!
        key_buffer,
        value_buffer,
        is_free: false,
    })
}
```

**Fixed Code**:
```rust
Ok(CachePage {
    page_id,
    sequence_id,
    tokens: Vec::with_capacity(config.page_size),  // ✅ Pre-allocate capacity
    key_buffer,
    value_buffer,
    is_free: false,
})
```

**Impact**: Fixes 3 failing tests related to KV cache token appending.

---

## Fix 2: Multi-Query Attention Test Data (CRITICAL)

**Bug**: Test provides wrong tensor size for MQA input.

**Location**: `/home/feanor/Projects/ROCmForge/src/attention/multi_query.rs:560-588`

**Current Test**:
```rust
#[test]
fn test_multi_query_attention_basic() {
    let config = MultiQueryConfig::new(2, 4).unwrap(); // num_heads=2, head_dim=4
    let mqa = MultiQueryAttention::new(config).unwrap();

    let q = vec![
        // batch=0, seq=2, heads=2, dim=4  = 16 elements needed
        // But only provides 8 elements ❌
        1.0, 2.0, 3.0, 4.0,  // q[0,0,0,:]
        5.0, 6.0, 7.0, 8.0,  // q[0,0,1,:]
    ];

    let k = vec![
        // batch=0, seq=2, heads=1, dim=4
        0.1, 0.2, 0.3, 0.4,  // k[0,0,0,:]
        0.5, 0.6, 0.7, 0.8,  // k[0,1,0,:]
    ];

    let v = vec![
        // batch=0, seq=2, heads=1, dim=4
        1.0, 2.0, 3.0, 4.0,  // v[0,0,0,:]
        5.0, 6.0, 7.0, 8.0,  // v[0,1,0,:]
    ];

    let output = mqa.forward(&q, &k, &v, None, None).unwrap();  // ❌ Panics here
}
```

**Fixed Test**:
```rust
#[test]
fn test_multi_query_attention_basic() {
    let config = MultiQueryConfig::new(2, 4).unwrap();
    let mqa = MultiQueryAttention::new(config).unwrap();

    // Shape: [batch=1, seq_len=2, num_heads=2, head_dim=4] = 16 elements
    let q = vec![
        // seq=0, heads=2, dim=4
        1.0, 2.0, 3.0, 4.0,  // q[0,0,0,:]
        5.0, 6.0, 7.0, 8.0,  // q[0,0,1,:]
        // seq=1, heads=2, dim=4
        9.0, 10.0, 11.0, 12.0,  // q[0,1,0,:]
        13.0, 14.0, 15.0, 16.0,  // q[0,1,1,:]
    ];

    // Shape: [batch=1, seq_len=2, num_kv_heads=1, head_dim=4] = 8 elements
    let k = vec![
        0.1, 0.2, 0.3, 0.4,  // k[0,0,0,:]
        0.5, 0.6, 0.7, 0.8,  // k[0,1,0,:]
    ];

    let v = vec![
        1.0, 2.0, 3.0, 4.0,  // v[0,0,0,:]
        5.0, 6.0, 7.0, 8.0,  // v[0,1,0,:]
    ];

    let output = mqa.forward(&q, &k, &v, None, None).unwrap();  // ✅ Now works

    // Output shape: [batch=1, seq=2, heads=2, dim=4] = 16 elements
    assert_eq!(output.len(), 16);
}
```

---

## Fix 3: RoPE Test Expectations (CRITICAL)

**Bug**: Test expects RoPE to modify values at position 0, but RoPE is identity at position 0.

**Location**: `/home/feanor/Projects/ROCmForge/src/attention/rope.rs:360-373`

**Current Test**:
```rust
#[test]
fn test_rope_application() {
    let config = RopeConfig::new(4, 8);
    let rope = Rope::new(config);

    let mut x = vec![
        1.0, 2.0, 3.0, 4.0,  // position 0
        5.0, 6.0, 7.0, 8.0,  // position 1
    ];
    let position_ids = vec![0, 1];  // ❌ Position 0 is identity!

    rope.apply_q(&mut x, &position_ids, 1).unwrap();

    // ❌ FAILS: x[0] is still 1.0 at position 0
    assert_ne!(x[0], 1.0);
    assert_ne!(x[1], 2.0);
}
```

**Fixed Test**:
```rust
#[test]
fn test_rope_application() {
    let config = RopeConfig::new(4, 8);
    let rope = Rope::new(config);

    let mut x = vec![
        1.0, 2.0, 3.0, 4.0,  // position 0 (identity)
        5.0, 6.0, 7.0, 8.0,  // position 1 (rotated)
    ];
    let position_ids = vec![0, 1];

    rope.apply_q(&mut x, &position_ids, 1).unwrap();

    // ✅ Position 0 should be unchanged (identity)
    assert_eq!(x[0], 1.0);  // Position 0: cos(0)=1, sin(0)=0
    assert_eq!(x[1], 2.0);

    // ✅ Position 1 should be changed (rotated)
    assert_ne!(x[4], 5.0);  // Position 1 starts at index 4
    assert_ne!(x[5], 6.0);
}
```

**Explanation**: RoPE rotation formula:
```
x_new = x * cos(pos) - y * sin(pos)
y_new = x * sin(pos) + y * cos(pos)
```

At position 0: `cos(0) = 1.0`, `sin(0) = 0.0`, so:
```
x_new = x * 1.0 - y * 0.0 = x  (no change)
y_new = x * 0.0 + y * 1.0 = y  (no change)
```

---

## Fix 4: HTTP Server Test Setup (CRITICAL)

**Bug**: HTTP server tests don't load a model but expect generation to succeed.

**Location**: `/home/feanor/Projects/ROCmForge/src/http/server.rs:600-624`

**Option A: Test Error Path** (Quickest Fix)
```rust
#[tokio::test]
async fn test_generate_request() {
    let server = InferenceServer::new(None, TokenizerAdapter::default());

    let request = GenerateRequest {
        prompt: "Test".to_string(),
        max_tokens: Some(3),
        temperature: Some(0.8),
        top_k: Some(40),
        top_p: Some(0.9),
        stream: Some(false),
    };

    // ✅ Test that it properly returns error when no model loaded
    let result = server.generate(request).await;
    assert!(result.is_err());  // Expected to fail
    assert!(matches!(result.unwrap_err(), InferenceError::NoModelLoaded));
}
```

**Option B: Mock the Model** (Better Fix)
```rust
#[tokio::test]
async fn test_generate_request() {
    // Create a mock model loader
    let mock_model = Arc::new(MockModel::new());

    let mut server = InferenceServer::new(None, TokenizerAdapter::default());
    server.load_model(mock_model).await.unwrap();

    let request = GenerateRequest {
        prompt: "Test".to_string(),
        max_tokens: Some(3),
        temperature: Some(0.8),
        top_k: Some(40),
        top_p: Some(0.9),
        stream: Some(false),
    };

    let response = server.generate(request).await.unwrap();
    assert_eq!(response.request_id, 0);
    assert!(!response.text.is_empty());
}
```

---

## Fix 5: Cleanup Unused Code (Automated)

**Step 1: Run cargo fix**
```bash
# Auto-fix unused imports, variables, and style issues
cargo fix --lib --allow-dirty
cargo fix --lib --tests --allow-dirty

# Verify fixes
cargo build
cargo test --lib
```

**Step 2: Manual cleanup**

Remove dead kernel cache code in `/home/feanor/Projects/ROCmForge/src/attention/kernels.rs`:
```rust
// DELETE lines 18-46 (KernelCache struct and GLOBAL_CACHE)
// DELETE function get_or_init_cache()
```

Remove unused weight mapping functions in `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`:
```rust
// DELETE or document why they're kept:
// - try_map_qwen2_attention_weights (line 1097)
// - map_llama_attention_weights (line 1157)
// - try_map_qwen2_mlp_weights (line 1496)
// - map_llama_mlp_weights (line 1555)
// - try_map_qwen2_layer_norm_weights (line 1806)
// - map_llama_layer_norm_weights (line 1895)
```

---

## Fix 6: Naming Conventions

**Fix f16 struct name** in `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:1535`:
```rust
// Before:
struct f16(u16);

// After:
struct F16(u16);  // UpperCamelCase for types
```

**Fix constant names** (multiple files):
```rust
// Before:
const hipSuccess: i32 = 0;
const hipMemcpyHostToDevice: u32 = 1;

// After:
const HIP_SUCCESS: i32 = 0;
const HIP_MEMCPY_HOST_TO_DEVICE: u32 = 1;  // SCREAMING_SNAKE_CASE
```

---

## Testing After Fixes

**Step 1: Apply fixes**
```bash
# Edit files with fixes above
# Then run:
cargo build
```

**Step 2: Run tests**
```bash
# Run all tests
cargo test --lib

# Expected: All 11 failing tests should now pass
# Pass rate should increase from 90.5% to 100%
```

**Step 3: Verify warnings reduced**
```bash
cargo clippy --all-targets --all-features

# Expected: Warnings reduced from 76 to ~20
# (Remaining warnings are intentional dead code to be removed)
```

---

## Priority Order

1. **Fix 1 (KV Cache)** - Do this first! Breaks core functionality.
2. **Fix 2 (MQA Test)** - Required for multi-query attention.
3. **Fix 3 (RoPE Test)** - Required for attention tests.
4. **Fix 4 (HTTP Tests)** - Required for HTTP API tests.
5. **Fix 5 (Cleanup)** - Run after all fixes applied.
6. **Fix 6 (Naming)** - Optional, improves code quality.

**Estimated time**: 30 minutes for all fixes.

---

## Verification Checklist

After applying fixes:

- [ ] All 11 tests now pass
- [ ] `cargo build` succeeds
- [ ] `cargo clippy` shows < 20 warnings
- [ ] KV cache can append tokens
- [ ] MQA processes tensors correctly
- [ ] RoPE tests pass with corrected expectations
- [ ] HTTP server tests properly handle no-model case
- [ ] No functional regressions in working code

---

**End of Quick Fix Guide**
