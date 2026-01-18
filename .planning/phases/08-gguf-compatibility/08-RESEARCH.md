# Phase 08: GGUF Compatibility - Research

**Date:** 2026-01-18
**Status:** Complete
**Objective:** Answer "What do I need to know to PLAN this phase well?"

---

## Executive Summary

Phase 08 aims to achieve universal GGUF support across all model architectures and quantization formats. This research documents the current state of GGUF compatibility, identifies gaps, and provides the foundation for planning implementation work.

**Key Findings:**
- **Quantization Support:** 9/13 formats implemented (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K, MXFP4/6)
- **Missing Quantization:** Q2_K, Q3_K, Q5_K (return "not yet implemented" errors)
- **Architecture Support:** 3 detected (Qwen2, LLaMA, Mistral) - but only Qwen2 and LLaMA have full metadata key mappings
- **ModelType Enum:** Only 2 variants (Llama, Qwen) - Mistral and others not represented

---

## 1. Current GGUF Loader Implementation

### 1.1 Core Files (from Phase 3 modularization)

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `src/loader/gguf.rs` | Main loader, dequantization orchestration | ~2800 LOC |
| `src/loader/tensor_type.rs` | GgufTensorType enum with 13 types | 133 LOC |
| `src/loader/metadata.rs` | GgufMetadata struct and KV parsing | 161 LOC |
| `src/loader/gguf_tensor.rs` | GgufTensor descriptor | 136 LOC |
| `src/loader/dequant.rs` | Dequantization implementations | 570 LOC |
| `src/loader/mxfp.rs` | MXFP4/MXFP6 block support | 359 LOC |

### 1.2 Architecture Detection

Located in: `src/model/execution_plan/architecture.rs`

```rust
pub enum Architecture {
    Qwen2,    // Pattern: "blk.N."
    LLaMA,    // Pattern: "transformer.layers.N."
    Mistral,  // Pattern: "model.layers.N."
}
```

**Detection Logic:**
- Scans tensor names for architecture-specific patterns
- Returns error if no known pattern matches
- Provides sample tensors in error message for debugging

### 1.3 Model Config

Located in: `src/model/config.rs`

```rust
pub enum ModelType {
    Llama,
    Qwen,
}
```

**Gap:** Mistral detected by architecture.rs but not represented in ModelType enum.

---

## 2. Quantization Format Support

### 2.1 Supported Formats (CPU Dequantization)

| Format | Status | Implementation | Notes |
|--------|--------|----------------|-------|
| F32 | ✅ Complete | `gguf.rs:817-821` | Direct chunk conversion |
| F16 | ✅ Complete | `gguf.rs:822-828` | Uses `half::f16` crate |
| Q4_0 | ✅ Complete | `dequant.rs:69-124` | Parallelized with Rayon |
| Q4_1 | ✅ Complete | `dequant.rs:128-175` | Scale + min format |
| Q5_0 | ✅ Complete | `dequant.rs:180-230` | High bit in qh field |
| Q5_1 | ✅ Complete | `dequant.rs:234-288` | Scale + min + qh format |
| Q8_0 | ✅ Complete | `dequant.rs:14-63` | Parallelized with Rayon |
| Q4_K | ✅ Complete | `dequant.rs:387-463` | Super-block structure |
| Q6_K | ✅ Complete | `dequant.rs:468-532` | 6-bit packed format |
| MXFP4 | ✅ Complete | `dequant.rs:291-332` | OCP MX Spec v1.0 |
| MXFP6 | ✅ Complete | `dequant.rs:335-382` | E2M3 variant |

### 2.2 Missing Quantization Formats

| Format | Status | Gap Location | What's Needed |
|--------|--------|--------------|---------------|
| Q2_K | ❌ Not Implemented | `gguf.rs:907-912` | CPU dequant, HIP kernel |
| Q3_K | ❌ Not Implemented | `gguf.rs:907-912` | CPU dequant, HIP kernel |
| Q5_K | ❌ Not Implemented | `gguf.rs:907-912` | CPU dequant, HIP kernel |

**Current Error Path:**
```rust
// From src/loader/gguf.rs:907-912
GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q5_K => {
    return Err(anyhow!(
        "K-quant type {:?} not yet implemented for tensor '{}'",
        tensor_type, name
    ));
}
```

### 2.3 GPU Dequantization Kernels (from Phase 5)

| Format | Kernel | File | Status |
|--------|--------|------|--------|
| Q4_0 | q4_0_dequant.hip | kernels/ | ✅ Complete |
| Q8_0 | q8_0_dequant.hip | kernels/ | ✅ Complete |
| Q4_K | q4_k_dequant.hip | kernels/ | ✅ Complete |
| Q6_K | q6_k_dequant.hip | kernels/ | ✅ Complete |
| Q4_0 matmul | q4_0_matmul.hip | kernels/ | ✅ Complete (fused) |
| Q2_K | - | - | ❌ Missing |
| Q3_K | - | - | ❌ Missing |
| Q5_K | - | - | ❌ Missing |

---

## 3. Model Architecture Support

### 3.1 Metadata Key Mappings

Located in: `src/loader/metadata.rs:44-126`

| Architecture | Key Pattern | Coverage |
|-------------|-------------|----------|
| **GLM** | `glm.*` | ✅ Complete (7 keys) |
| **Gemma 3** | `gemma3.*` | ✅ Complete (8 keys) |
| **Qwen2** | `qwen2.*` | ✅ Complete (7 keys) |
| **LLaMA** | `llama.*` | ✅ Complete (8 keys) |

**Metadata Keys Parsed:**
- `general.architecture` → String
- `general.file_type` → u32
- `*.block_count` / `*.n_layers` → num_layers
- `*.attention.head_count` / `*.n_heads` → num_heads
- `*.attention.head_count_kv` → num_kv_heads (MQA/GQA)
- `*.embedding_length` / `*.n_embd` → hidden_size
- `*.feed_forward_length` / `*.intermediate_size` → intermediate_size
- `*.rope.dimension_count` / `*.head_dim` → head_dim
- `*.max_position_embeddings` → context length
- `*.vocab_size` → vocab size
- `*.attention.layer_norm_rms_epsilon` → rms_norm_eps

### 3.2 Missing Architectures

Common GGUF architectures without dedicated metadata support:

| Architecture | GGUF Key Pattern | Tensor Pattern | Priority |
|--------------|------------------|----------------|----------|
| **Mistral** | `mistral.*` | `model.layers.N.*` | High |
| **Mixtral** (MoE) | `mixtral.*` | `model.layers.N.*` | Medium |
| **Yi** | `yi.*` | `model.layers.N.*` | Medium |
| **Phi-2/3** | `phi.*` | varies | Low |
| **Falcon** | `falcon.*` | `transformer.h.N.*` | Low |
| **Bloom** | `bloom.*` | `transformer.h.N.*` | Low |
| **MPT** | `mpt.*` | `transformer.blocks.N.*` | Low |

**Note:** Mistral architecture is detected in `architecture.rs` but has no metadata key mappings. It likely works with LLaMA-style keys (similar architecture), but this is untested.

### 3.3 Architecture-Specific Features

| Feature | LLaMA | Qwen | Mistral | Gemma | GLM |
|---------|-------|------|---------|-------|-----|
| MQA/GQA | ✅ | ✅ | ✅ | ✅ | ❓ |
| RoPE | ✅ | ✅ | ✅ | ✅ | ✅ |
| RMS Norm | ✅ | ✅ | ✅ | ✅ | ✅ |
| SwiGLU | ✅ | ✅ | ✅ | ✅ | ❓ |

---

## 4. GGUF Format Specification

### 4.1 File Structure

```
GGUF File:
├── Header (magic + version)
├── Tensor Count (u64)
├── KV Count (u64)
├── KV Pairs (metadata)
├── Tensor Info Array (name, shape, type, offset)
└── Tensor Data (binary blob)
```

### 4.2 Supported GGML Types

| Type ID | Name | Description | Block Size |
|---------|------|-------------|------------|
| 0 | F32 | 32-bit float | 1 |
| 1 | F16 | 16-bit float | 1 |
| 2 | Q4_0 | 4-bit quantized | 32 |
| 3 | Q4_1 | 4-bit quantized + min | 32 |
| 6 | Q5_0 | 5-bit quantized | 32 |
| 7 | Q5_1 | 5-bit quantized + min | 32 |
| 8 | Q8_0 | 8-bit quantized | 32 |
| 10 | Q2_K | K-quant 2-bit | 256 |
| 11 | Q3_K | K-quant 3-bit | 256 |
| 12 | Q4_K | K-quant 4-bit | 256 |
| 13 | Q5_K | K-quant 5-bit | 256 |
| 14 | Q6_K | K-quant 6-bit | 256 |
| 20-22 | MXFP4/6 | OCP MX block floating-point | 32 |

### 4.3 Tensor Naming Conventions

| Architecture | Weight Prefix | Layer N | Attention | FFN |
|--------------|---------------|---------|-----------|-----|
| LLaMA | `transformer.layers` | `.N.` | `.attention_q/k/v/o.weight` | `.ffn_*` |
| Qwen2 | `blk` | `.N.` | `.attn_q/k/v/o.weight` | `.ffn_*` |
| Mistral | `model.layers` | `.N.` | `.self_attn.q/k/v/o_proj` | `.mlp.*` |
| Gemma3 | `model.layers` | `.N.` | `.self_attn.q/k/v/o_proj` | `.mlp.*` |
| GLM | `transformer.encoder.layers` | `.N.` | varies | varies |

---

## 5. Current Gaps Analysis

### 5.1 Critical Gaps (Blockers for Universal Support)

1. **Q2_K, Q3_K, Q5_K Quantization**
   - Impact: Cannot load models using these formats
   - Files: `src/loader/dequant.rs`, new kernel .hip files
   - Estimate: 3-4 hours per format (CPU + GPU kernels)

2. **Mistral Metadata Keys**
   - Impact: Mistral models may fail to load correctly
   - Files: `src/loader/metadata.rs`
   - Estimate: 1-2 hours

3. **ModelType Enum Incomplete**
   - Impact: Runtime may not handle all architectures
   - Files: `src/model/config.rs`
   - Estimate: 30 minutes

### 5.2 Important Gaps (Wide Compatibility)

4. **Mixtral (MoE) Support**
   - Impact: Cannot load MoE models
   - Requires: MoE routing logic, expert loading
   - Estimate: 8-12 hours

5. **Yi Architecture**
   - Impact: Popular Chinese LLM family
   - Files: `src/loader/metadata.rs`, `src/model/execution_plan/architecture.rs`
   - Estimate: 2-3 hours

### 5.3 Nice-to-Have Gaps

6. **Additional Architectures** (Phi, Falcon, Bloom, MPT)
   - Impact: Less common model families
   - Estimate: 2-3 hours each

7. **MXFP GPU Dequantization**
   - Current: CPU only, errors on GPU path
   - Files: `src/loader/gguf.rs:914-919`
   - Estimate: 4-6 hours

---

## 6. Prior Architectural Decisions (Context)

### 6.1 Relevant Decisions

- **Use #[ignore] for E2E tests** (Phase 02-03)
  - E2E tests require real GGUF models
  - Use `ROCFORGE_TEST_MODEL` env var for model path

- **Use std::simd for CPU SIMD** (Phase 04-01)
  - MSRV: Rust 1.82+
  - x86_64: f32x8 (AVX2), aarch64: f32x4 (NEON)

- **CapabilityProvider trait decoupled from GgmlBackend** (Phase 07-01)
  - Enables dynamic dispatch for hybrid execution

- **2x threshold for CPU vs GPU selection** (Phase 07-03)
  - Prevents backend oscillation

### 6.2 Dequantization Patterns (from Phase 05)

**CPU Pattern:**
```rust
// Parallel processing with Rayon
(0..blocks).into_par_iter().for_each(|block_idx| {
    // Read scale, quants
    // Dequantize
    // Write to shared result via Arc<RwLock<Vec<f32>>>
});
```

**GPU Pattern:**
```rust
// One GPU block per quantized block
// One thread per element
// 256 threads per block (RDNA3 tuning)
```

---

## 7. Implementation Considerations

### 7.1 Quantization Implementation Order

Based on commonality and complexity:

1. **Q5_K** - Most common missing K-quant
2. **Q3_K** - Medium complexity, moderate usage
3. **Q2_K** - Most complex (super-block structure), least common

### 7.2 Architecture Implementation Order

Based on model popularity:

1. **Mistral** - High priority, already detected
2. **Yi** - Growing popularity
3. **Mixtral** - MoE complexity requires significant work
4. **Others** - As needed

### 7.3 Testing Strategy

Per Phase 02-03 decision:
- Use `#[ignore]` for E2E tests
- Use `ROCFORGE_TEST_MODEL` environment variable
- Create unit tests for dequantization with synthetic data
- Use `serial_test` for GPU tests (one at a time)

---

## 8. External References

### 8.1 GGUF Specification
- GitHub: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Key values: https://github.com/ggerganov/ggml/blob/master/include/ggml.h

### 8.2 Quantization Formats
- Q4_0, Q8_0: Basic block quantization
- K-quants (Q2_K-Q6_K): I-Matrix quantization
- MXFP: OCP MX Specification v1.0

### 8.3 Model Architectures
- LLaMA: https://arxiv.org/abs/2302.13971
- Qwen2: https://arxiv.org/abs/2307.17192
- Mistral: https://arxiv.org/abs/2310.06825
- Mixtral: https://arxiv.org/abs/2401.04088

---

## 9. Summary for Planning

### 9.1 What Works Now

- ✅ 9/13 quantization formats (all common ones)
- ✅ 3 detected architectures (Qwen2, LLaMA, Mistral)
- ✅ Metadata parsing for GLM, Gemma3, Qwen2, LLaMA
- ✅ CPU dequantization with Rayon parallelization
- ✅ GPU kernels for Q4_0, Q8_0, Q4_K, Q6_K
- ✅ Fused dequant+matmul for Q4_0

### 9.2 What Needs Implementation

1. **Q2_K, Q3_K, Q5_K** - CPU dequant + GPU kernels
2. **Mistral metadata** - Key mappings in metadata.rs
3. **ModelType enum** - Add Mistral variant
4. **MXFP GPU** - GPU dequantization kernels
5. **Additional architectures** - Yi, Mixtral, others as needed

### 9.3 Estimated Effort

| Task | Estimate | Dependencies |
|------|----------|--------------|
| Q5_K implementation | 3-4 hours | Phase 05 patterns |
| Q3_K implementation | 3-4 hours | Phase 05 patterns |
| Q2_K implementation | 4-5 hours | Phase 05 patterns |
| Mistral metadata | 1-2 hours | None |
| ModelType fix | 30 min | None |
| MXFP GPU kernels | 4-6 hours | Phase 05 patterns |
| **Total** | **16-22 hours** | |

---

## Appendix A: File Locations

```
/home/feanor/Projects/ROCmForge/
├── src/
│   ├── loader/
│   │   ├── gguf.rs (main loader, 2800 LOC)
│   │   ├── dequant.rs (all dequant functions, 570 LOC)
│   │   ├── metadata.rs (KV parsing, 161 LOC)
│   │   ├── tensor_type.rs (GgufTensorType enum, 133 LOC)
│   │   ├── gguf_tensor.rs (GgufTensor struct, 136 LOC)
│   │   └── mxfp.rs (MXFP support, 359 LOC)
│   ├── model/
│   │   ├── config.rs (ModelType enum, 193 LOC)
│   │   └── execution_plan/
│   │       └── architecture.rs (Architecture enum, 102 LOC)
│   └── ggml/hip_backend/ops/
│       ├── q4_0_dequant.rs (Q4_0 GPU wrapper)
│       └── quantized_matmul.rs (fused matmul)
├── kernels/
│   ├── q4_0_dequant.hip
│   ├── q8_0_dequant.hip
│   ├── q4_k_dequant.hip
│   ├── q6_k_dequant.hip
│   └── q4_0_matmul.hip
└── .planning/
    └── phases/08-gguf-compatibility/
        └── 08-RESEARCH.md (this file)
```

---

**End of Research Document**
