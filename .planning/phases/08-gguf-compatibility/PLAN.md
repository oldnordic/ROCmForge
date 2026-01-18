# Phase 08: GGUF Compatibility - Implementation Plan

**Phase:** 08
**Title:** GGUF Compatibility - Universal Model Support
**Status:** Ready for Execution
**Created:** 2026-01-18

---

## Frontmatter

```yaml
wave: 4
depends_on: [05]
files_modified:
  - src/loader/dequant.rs
  - src/loader/metadata.rs
  - src/model/config.rs
  - src/model/execution_plan/architecture.rs
  - src/ggml/hip_backend/ops/mod.rs
  - build.rs
  - tests/
autonomous: true
```

---

## Phase Goal

Achieve universal GGUF support across all model architectures and quantization formats, enabling ROCmForge to load and run any GGUF model regardless of architecture (Mistral, Yi, etc.) or quantization (Q2_K, Q3_K, Q5_K).

**must_haves for Goal Verification:**
1. All 13 GGUF quantization formats supported (Q2_K, Q3_K, Q5_K added)
2. Mistral, Yi, and Mixtral architectures have metadata key mappings
3. ModelType enum includes all detected architectures
4. GPU kernels exist for Q2_K, Q3_K, Q5_K dequantization
5. Test matrix documents compatibility across architectures x quantizations

---

## Current State (from Research)

**Supported (9/13 formats):**
- F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K, MXFP4/6

**Missing (3/13 formats):**
- Q2_K, Q3_K, Q5_K - return "not yet implemented" errors

**Architecture Detection:**
- Qwen2 (pattern: `blk.N.`)
- LLaMA (pattern: `transformer.layers.N.`)
- Mistral (pattern: `model.layers.N.`) - detected but no metadata keys

**Metadata Support:**
- GLM, Gemma3, Qwen2, LLaMA - complete
- Mistral - missing key mappings
- Yi, Mixtral - not implemented

**ModelType Enum:**
- Llama, Qwen only - Mistral missing

---

## Wave Structure

### Wave 1: Architecture Metadata (Parallel)
- 08-01: Add Mistral metadata keys
- 08-02: Add Yi architecture support
- 08-03: Add Mixtral (MoE) architecture detection

### Wave 2: ModelType Enum Fix
- 08-04: Add Mistral to ModelType enum

### Wave 3: Missing K-Quant CPU Dequantization (Sequential)
- 08-05: Implement Q5_K CPU dequantization
- 08-06: Implement Q3_K CPU dequantization
- 08-07: Implement Q2_K CPU dequantization

### Wave 4: GPU Kernels for Missing Formats (Parallel)
- 08-08: Create Q5_K GPU dequantization kernel
- 08-09: Create Q3_K GPU dequantization kernel
- 08-10: Create Q2_K GPU dequantization kernel

### Wave 5: Integration and Testing
- 08-11: Build compatibility test matrix

---

## Tasks

### Wave 1: Architecture Metadata (Parallel)

#### Task 08-01: Add Mistral Metadata Keys

**Description:** Add `mistral.*` metadata key mappings to `GgufMetadata::update_from_kv()`.

**Files:**
- `src/loader/metadata.rs`

**Actions:**
1. Add Mistral key patterns in `update_from_kv()`:
   - `mistral.n_layers` or `mistral.block_count` -> num_layers
   - `mistral.attention.head_count` -> num_heads
   - `mistral.attention.head_count_kv` -> num_kv_heads
   - `mistral.embedding_length` or `mistral.hidden_size` -> hidden_size
   - `mistral.feed_forward_length` or `mistral.intermediate_size` -> intermediate_size
   - `mistral.attention.key_length` -> head_dim
   - `mistral.max_position_embeddings` -> context length
   - `mistral.vocab_size` -> vocab size
   - `mistral.attention.layer_norm_rms_epsilon` -> rms_norm_eps

2. Add unit test `test_mistral_metadata_parsing()` in `metadata.rs`

**Acceptance Criteria:**
- All Mistral keys parse correctly
- Test passes with synthetic Mistral metadata
- No breaking changes to existing architectures

**Dependencies:** None

**Estimated Time:** 1-2 hours

---

#### Task 08-02: Add Yi Architecture Support

**Description:** Add Yi architecture detection and metadata key mappings.

**Files:**
- `src/model/execution_plan/architecture.rs`
- `src/loader/metadata.rs`

**Actions:**
1. Add `Yi` variant to `Architecture` enum
2. Add detection pattern: `model.layers.N.*` (same as Mistral, but differentiate via metadata)
3. Add Yi metadata keys in `metadata.rs`:
   - `yi.n_layers` or `yi.block_count` -> num_layers
   - `yi.n_heads` or `yi.attention.head_count` -> num_heads
   - `yi.n_embd` or `yi.hidden_size` -> hidden_size
   - `yi.intermediate_size` -> intermediate_size
   - `yi.head_dim` -> head_dim
   - `yi.max_position_embeddings` -> context length
   - `yi.vocab_size` -> vocab size
   - `yi.rms_norm_eps` -> rms_norm_eps

4. Add unit test `test_yi_detection()` in `architecture.rs`

**Acceptance Criteria:**
- Yi architecture detected correctly
- Metadata keys parse correctly
- Test passes

**Dependencies:** None

**Estimated Time:** 2-3 hours

---

#### Task 08-03: Add Mixtral (MoE) Architecture Detection

**Description:** Add Mixtral architecture detection for MoE models.

**Files:**
- `src/model/execution_plan/architecture.rs`
- `src/loader/metadata.rs`

**Actions:**
1. Add `Mixtral` variant to `Architecture` enum
2. Add detection pattern: `model.layers.N.*` with `general.architecture = mixtral`
3. Add Mixtral metadata keys in `metadata.rs`:
   - `mixtral.n_layers` or `mixtral.block_count` -> num_layers
   - `mixtral.n_heads` or `mixtral.attention.head_count` -> num_heads
   - `mixtral.n_embd` or `mixtral.hidden_size` -> hidden_size
   - `mixtral.ffn_dim` or `mixtral.intermediate_size` -> intermediate_size
   - `mixtral.head_dim` -> head_dim
   - `mixtral.max_position_embeddings` -> context length
   - `mixtral.vocab_size` -> vocab size
   - `mixtral.attention.layer_norm_rms_epsilon` -> rms_norm_eps
   - `mixtral.n_experts` or `mixtral.n_expert` -> num_experts (MoE-specific)
   - `mixtral.n_experts_per_tok` -> experts_per_token (MoE-specific)

4. Add unit test `test_mixtral_detection()` in `architecture.rs`

**Acceptance Criteria:**
- Mixtral architecture detected correctly
- Metadata keys parse correctly including MoE-specific keys
- Test passes

**Dependencies:** None

**Estimated Time:** 3-4 hours

---

### Wave 2: ModelType Enum Fix

#### Task 08-04: Add Missing ModelType Variants

**Description:** Add Mistral, Yi, and Mixtral to `ModelType` enum.

**Files:**
- `src/model/config.rs`

**Actions:**
1. Add `Mistral`, `Yi`, `Mixtral` variants to `ModelType` enum
2. Add `default_mistral()`, `default_yi()`, `default_mixtral()` factory functions
3. Update any match statements that exhaustively check ModelType
4. Add unit test for each new variant

**Acceptance Criteria:**
- All detected architectures have corresponding ModelType variants
- Factory functions create valid configurations
- Tests pass

**Dependencies:** None

**Estimated Time:** 30 minutes - 1 hour

---

### Wave 3: Missing K-Quant CPU Dequantization (Sequential)

#### Task 08-05: Implement Q5_K CPU Dequantization

**Description:** Implement CPU dequantization for Q5_K format.

**Files:**
- `src/loader/dequant.rs`

**Reference:** Q5_K format specification (from 08-RESEARCH.md)
- Block size: 256 bytes
- Super-block structure: 256 elements per block
- 16 half-precision scales (32 bytes)
- 16 int8 mins (16 bytes)
- 5-bit quantized values (160 bytes packed as 256*5/8)
- Additional data (48 bytes)

**Actions:**
1. Implement `dequant_q5_k()` function following the pattern of `dequant_q4_k()`
2. Use Rayon parallelization for blocks
3. Add to `dequantize()` dispatcher (remove from "not yet implemented" branch)
4. Add unit tests:
   - `test_dequant_q5_k_zeros()`
   - `test_dequant_q5_k_positive()`
   - `test_dequant_q5_k_partial_block()`

**Acceptance Criteria:**
- `dequant_q5_k()` returns correct FP32 values
- Unit tests pass
- No more "not yet implemented" error for Q5_K

**Dependencies:** None

**Estimated Time:** 3-4 hours

---

#### Task 08-06: Implement Q3_K CPU Dequantization

**Description:** Implement CPU dequantization for Q3_K format.

**Files:**
- `src/loader/dequant.rs`

**Reference:** Q3_K format specification (from 08-RESEARCH.md)
- Block size: 256 bytes
- Super-block structure with sub-blocks
- 3-bit quantized values

**Actions:**
1. Implement `dequant_q3_k()` function
2. Use Rayon parallelization
3. Add to `dequantize()` dispatcher
4. Add unit tests:
   - `test_dequant_q3_k_zeros()`
   - `test_dequant_q3_k_positive()`
   - `test_dequant_q3_k_partial_block()`

**Acceptance Criteria:**
- `dequant_q3_k()` returns correct FP32 values
- Unit tests pass
- No more "not yet implemented" error for Q3_K

**Dependencies:** 08-05 (pattern established)

**Estimated Time:** 3-4 hours

---

#### Task 08-07: Implement Q2_K CPU Dequantization

**Description:** Implement CPU dequantization for Q2_K format (most complex).

**Files:**
- `src/loader/dequant.rs`

**Reference:** Q2_K format specification (from 08-RESEARCH.md)
- Block size: 256 bytes
- Most complex super-block structure
- 2-bit quantized values

**Actions:**
1. Implement `dequant_q2_k()` function
2. Use Rayon parallelization
3. Add to `dequantize()` dispatcher
4. Add unit tests:
   - `test_dequant_q2_k_zeros()`
   - `test_dequant_q2_k_positive()`
   - `test_dequant_q2_k_partial_block()`

**Acceptance Criteria:**
- `dequant_q2_k()` returns correct FP32 values
- Unit tests pass
- No more "not yet implemented" error for Q2_K
- All 13 quantization formats now supported

**Dependencies:** 08-06 (pattern established)

**Estimated Time:** 4-5 hours

---

### Wave 4: GPU Kernels for Missing Formats (Parallel)

#### Task 08-08: Create Q5_K GPU Dequantization Kernel

**Description:** Create HIP kernel for Q5_K GPU dequantization.

**Files:**
- `kernels/q5_k_dequant.hip` (new)
- `src/ggml/hip_backend/ops/q5_k_dequant.rs` (new)
- `src/ggml/hip_backend/ops/mod.rs` (modify)
- `build.rs` (modify)

**Actions:**
1. Create `kernels/q5_k_dequant.hip` following `q4_k_dequant.hip` pattern:
   - Block size: 256 threads
   - Super-block structure handling
   - `q5_k_to_fp32_kernel()` and `q5_k_to_fp32_batch_kernel()`

2. Create `src/ggml/hip_backend/ops/q5_k_dequant.rs`:
   - `dequantize_q5_k_gpu()` function
   - `dequantize_q5_k_cpu()` reference implementation
   - 5+ unit tests

3. Add to build.rs kernels array

4. Update `mod.rs` to export q5_k_dequant

**Acceptance Criteria:**
- Kernel compiles with hipcc
- Unit tests pass (CPU reference)
- GPU function defined (test with #[ignore])

**Dependencies:** 08-05 (CPU implementation reference)

**Estimated Time:** 2-3 hours

---

#### Task 08-09: Create Q3_K GPU Dequantization Kernel

**Description:** Create HIP kernel for Q3_K GPU dequantization.

**Files:**
- `kernels/q3_k_dequant.hip` (new)
- `src/ggml/hip_backend/ops/q3_k_dequant.rs` (new)
- `src/ggml/hip_backend/ops/mod.rs` (modify)
- `build.rs` (modify)

**Actions:**
1. Create `kernels/q3_k_dequant.hip`
2. Create `src/ggml/hip_backend/ops/q3_k_dequant.rs`
3. Add to build.rs
4. Update mod.rs

**Acceptance Criteria:**
- Kernel compiles
- Unit tests pass
- GPU function defined

**Dependencies:** 08-06 (CPU implementation reference)

**Estimated Time:** 2-3 hours

---

#### Task 08-10: Create Q2_K GPU Dequantization Kernel

**Description:** Create HIP kernel for Q2_K GPU dequantization.

**Files:**
- `kernels/q2_k_dequant.hip` (new)
- `src/ggml/hip_backend/ops/q2_k_dequant.rs` (new)
- `src/ggml/hip_backend/ops/mod.rs` (modify)
- `build.rs` (modify)

**Actions:**
1. Create `kernels/q2_k_dequant.hip`
2. Create `src/ggml/hip_backend/ops/q2_k_dequant.rs`
3. Add to build.rs
4. Update mod.rs

**Acceptance Criteria:**
- Kernel compiles
- Unit tests pass
- GPU function defined

**Dependencies:** 08-07 (CPU implementation reference)

**Estimated Time:** 2-3 hours

---

### Wave 5: Integration and Testing

#### Task 08-11: Build Compatibility Test Matrix

**Description:** Create comprehensive test matrix documenting GGUF compatibility.

**Files:**
- `tests/gguf_compatibility_matrix.rs` (new)
- `.planning/phases/08-gguf-compatibility/COMPATIBILITY.md` (new)

**Actions:**
1. Create `tests/gguf_compatibility_matrix.rs`:
   - Table of architectures x quantization formats
   - Helper function to check if combination is supported
   - Tests for all supported combinations
   - Documentation of known limitations

2. Create `COMPATIBILITY.md`:
   - Support matrix table
   - Known issues per architecture
   - Per-format limitations
   - Future work items

**Acceptance Criteria:**
- Test matrix covers all architectures and formats
- Document clearly states what works and what doesn't
- Tests verify claimed support

**Dependencies:** All previous tasks (08-01 through 08-10)

**Estimated Time:** 1-2 hours

---

## Verification Criteria

Phase complete when:

1. **Quantization Support:** All 13 GGUF quantization formats supported
   - F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K, MXFP4/6 (existing)
   - Q2_K, Q3_K, Q5_K (new)

2. **Architecture Support:**
   - Metadata key mappings for Mistral, Yi, Mixtral
   - ModelType enum includes all detected architectures

3. **GPU Coverage:**
   - Q2_K, Q3_K, Q5_K have HIP kernels

4. **Test Coverage:**
   - Unit tests for all new dequantization functions
   - Architecture detection tests
   - Compatibility matrix documents all combinations

5. **No Regressions:**
   - All existing tests pass
   - No breaking changes to public API

---

## Rollback Plan

Each task can be independently rolled back:
- Revert the specific commit
- No inter-task dependencies except CPU -> GPU kernels

---

## References

- Research: `.planning/phases/08-gguf-compatibility/08-RESEARCH.md`
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Phase 5 Patterns: `.planning/phases/05-quantized-operations/`
