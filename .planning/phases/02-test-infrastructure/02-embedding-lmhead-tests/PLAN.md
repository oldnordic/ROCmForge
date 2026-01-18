# Plan 02-02: Restore embedding_to_lmhead Tests

**Phase**: 02 - Test Infrastructure
**Status**: Pending
**Complexity**: High
**Estimated Time**: 4-5 hours

---

## Problem Statement

The file `tests/embedding_to_lmhead_tests.rs` is entirely commented out (436 lines of original tests). It tests the critical embedding lookup and LM head computation pipeline, which is fundamental to LLM inference.

**Current State**:
- File exists but contains only TODO comments
- Original tests used obsolete `GgufModel` API
- No test coverage for embedding → LM head pipeline

---

## What Needs Testing

The embedding_to_lmhead pipeline consists of:

1. **Token Embedding Lookup**
   - Map token IDs to embedding vectors
   - Shape: `[vocab_size, hidden_size]`
   - Usually FP32 (not quantized)

2. **LM Head Weight Loading**
   - Output projection weights
   - Same shape as token embeddings (tied embeddings)
   - Maps hidden states back to logits

3. **CPU vs GPU Correctness**
   - CPU implementation using raw arrays
   - GPU implementation using DeviceTensor
   - Results should match

4. **Shape Validation**
   - Vocab size consistency
   - Hidden size consistency
   - Batch dimension handling

5. **Edge Cases**
   - Empty token sequences
   - OOM conditions
   - Invalid token IDs
   - Multiple vocab sizes (32k, 128k, etc.)

---

## Implementation Plan

### Task 1: Understand the Current Architecture

First, verify how embeddings and LM head work in the current codebase:

**Files to Read**:
- `src/model/transformer.rs` - Transformer layer implementation
- `src/loader/gguf.rs` - How token_embd.weight is loaded
- `src/tensor/matmul.rs` - Matmul operations for embeddings

**Key Questions**:
- Where is token embedding lookup performed?
- Is LM head tied to token embeddings (same weights)?
- How are embeddings uploaded to GPU?

### Task 2: Create Test Infrastructure

Add helper functions to `tests/embedding_to_lmhead_tests.rs`:

```rust
/// Create a minimal GGUF with embedding weights
fn create_embedding_gguf(path: &Path, vocab_size: usize, hidden_size: usize)
    -> anyhow::Result<()>
{
    // Write GGUF with:
    // - token_embd.weight: [vocab_size, hidden_size] (FP32)
    // - output.weight: [vocab_size, hidden_size] (tied to embd)
    // - Minimal metadata (n_embd, vocab_size)
}

/// Verify embedding lookup produces correct vectors
fn verify_embedding_lookup(
    embeddings: &HashMap<String, GgufTensor>,
    token_id: usize,
    expected_dim: usize
) -> bool
{
    // Extract embedding for token_id
    // Verify dimensionality
    // Return true if valid
}
```

### Task 3: Write Token Embedding Lookup Tests

```rust
#[test]
fn test_token_embedding_lookup_f32() -> anyhow::Result<()> {
    // Create GGUF with FP32 embeddings
    // Load embeddings via GgufLoader
    // Verify shape: [vocab_size, hidden_size]
    // Verify data type: F32
    // Verify specific values are accessible
}

#[test]
fn test_token_embedding_shape_validation() -> anyhow::Result<()> {
    // Test with various vocab sizes: 1000, 32000, 128000
    // Verify shape is correctly parsed
    // Verify total_elements calculation
}

#[test]
fn test_token_embedding_gpu_upload() -> anyhow::Result<()> {
    // Create GGUF with embeddings
    // Load to GPU via load_to_gpu()
    // Verify GPU tensor is valid
    // Verify memory is allocated
    // (Use #[serial] for GPU test)
}
```

### Task 4: Write LM Head Tests

```rust
#[test]
fn test_lm_head_weights_match_embeddings() -> anyhow::Result<()> {
    // Create GGUF with tied embeddings
    // Verify token_embd.weight == output.weight (if tied)
    // Or verify they have same shape if separate
}

#[test]
fn test_lm_head_matmul_correctness() -> anyhow::Result<()> {
    // Create test hidden state: [batch, seq_len, hidden_size]
    // Create test LM head weights: [vocab_size, hidden_size]
    // Compute logits via CPU matmul
    // Verify output shape: [batch, seq_len, vocab_size]
}

#[test]
fn test_lm_head_gpu_cpu_parity() -> anyhow::Result<()> {
    // Create same input for CPU and GPU
    // Compute LM head on both
    // Verify results match within tolerance
}
```

### Task 5: Write End-to-End Pipeline Tests

```rust
#[test]
fn test_embedding_to_lmhead_pipeline() -> anyhow::Result<()> {
    // Full pipeline test:
    // 1. Load token embeddings
    // 2. Look up tokens [1, 2, 3]
    // 3. Pass through dummy model
    // 4. Compute LM head logits
    // 5. Verify argmax matches expected tokens
}

#[test]
fn test_batch_embedding_lookup() -> anyhow::Result<()> {
    // Test embedding lookup with batch dimension
    // Input: [[1, 2, 3], [4, 5, 6]]
    // Verify output shape: [2, 3, hidden_size]
}
```

### Task 6: Write Edge Case Tests

```rust
#[test]
fn test_empty_token_sequence() {
    // Empty token list should return empty embeddings
    // Should not panic
}

#[test]
fn test_invalid_token_id() {
    // Token ID >= vocab_size should error
    // Or return zeros/padding (verify behavior)
}

#[test]
fn test_large_vocabulary() -> anyhow::Result<()> {
    // Test with 128k vocab size (common in modern LLMs)
    // Verify no overflow or allocation issues
}
```

---

## Dependencies

**Soft Dependency on Plan 02-01**: The `create_test_gguf()` helper from 02-01 can be reused here.

**Can Run in Parallel**: Yes, with 02-01, 02-03, 02-04

---

## Definition of Done

- [ ] All 7+ test categories implemented
- [ ] Tests cover: embedding lookup, LM head, GPU/CPU parity, edge cases
- [ ] All tests pass: `cargo test --test embedding_to_lmhead_tests`
- [ ] GPU tests use `#[serial]` attribute
- [ ] Proper error handling with `anyhow::Result`
- [ ] Test documentation comments explain what's being tested
- [ ] No unwrap() in test code (use ? operator instead)

---

## Notes

- This is the most complex test restoration in Phase 2
- Original file was 436 lines - expect new tests to be 300-500 lines
- Focus on critical paths: embedding lookup and LM head computation
- GPU tests require actual AMD GPU hardware
- Use `assert!`, `assert_eq!`, `assert_matches!` for assertions

---

*Plan: 02-02*
*Created: 2026-01-18*
