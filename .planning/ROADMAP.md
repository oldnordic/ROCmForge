# Roadmap: rocmforge

**Created:** 2026-03-24
**Granularity:** Coarse (4 phases)
**Model Target:** Qwen2.5 GGUF

---

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Foundation | Load model and tokenize input | LOAD-01..05, CONF-01..05, TOKN-01..04 | 4 |
| 2 | CPU Backend | Run inference on CPU | CPUF-01..09 | 4 |
| 3 | GPU Backend | Run inference on AMD GPU | GPUI-01..11 | 4 |
| 4 | CLI Integration | User-facing inference tool | CLII-01..07, ERRO-01..05 | 3 |

**Total phases:** 4
**Total requirements:** 32
**Coverage:** 100% ✓

---

## Phase 1: Foundation

**Goal:** Load Qwen2.5 GGUF model and tokenize input text

**Requirements:**
- LOAD-01: Memory-map GGUF file
- LOAD-02: Parse GGUF header and metadata
- LOAD-03: Extract architecture string
- LOAD-04: Extract tensor metadata
- LOAD-05: Validate GGUF magic/version
- CONF-01: Derive ModelConfig from metadata
- CONF-02: ModelTraits registry for RoPE style
- CONF-03: Determine attention layout
- CONF-04: Infer intermediate_size from tensors
- CONF-05: Vocab size from tokenizer, not GGUF key
- TOKN-01: Parse tokenizer.json
- TOKN-02: Encode text to token IDs
- TOKN-03: Decode token IDs to text
- TOKN-04: Apply ChatML template

**Success Criteria:**
1. Can load Qwen2.5 GGUF file and print model dimensions
2. Can encode "Hello, world!" and decode back to same text
3. ModelConfig correctly infers all values from GGUF
4. All loader tests pass with fixtures

**Dependencies:** None (first phase)

**Source Reference:** Memoria `src/loader/`, `src/config.rs`, `src/tokenizer/`

---

## Phase 2: CPU Backend

**Goal:** Run complete inference on CPU with single-request prefill and decode

**Requirements:**
- CPUF-01: CpuModelWeights with f32 tensors
- CPUF-02: RMS normalization
- CPUF-03: RoPE embedding (NeoX style)
- CPUF-04: Attention with KV cache
- CPUF-05: FFN (SwiGLU)
- CPUF-06: Greedy sampling
- CPUF-07: Top-p sampling
- CPUF-08: Prefill (single request)
- CPUF-09: Decode loop

**Success Criteria:**
1. Can run prefill on 10-token prompt, producing logits
2. Can sample tokens and generate coherent continuation
3. Greedy sampling produces deterministic output for fixed seed
4. Top-p sampling produces varied output for temperature > 0
5. KV cache correctly accumulates across decode steps

**Dependencies:** Phase 1 (loader, config, tokenizer)

**Source Reference:** Memoria `src/cpu/`

---

## Phase 3: GPU Backend

**Goal:** Run complete inference on AMD GPU via HIP

**Requirements:**
- GPUI-01: HIP context initialization
- GPUI-02: DeviceBuffer memory management
- GPUI-03: Weight copy with dequantization
- GPUI-04: GPU RMS norm kernel
- GPUI-05: GPU RoPE kernel
- GPUI-06: GPU attention kernel with KV cache
- GPUI-07: GPU FFN kernel
- GPUI-08: GPU greedy sampling
- GPUI-09: GPU top-p sampling
- GPUI-10: GPU prefill
- GPUI-11: GPU decode loop

**Success Criteria:**
1. Can initialize HIP context and allocate device memory
2. GPU prefill produces same logits as CPU (within float tolerance)
3. GPU decode produces same output as CPU for same seed
4. All GPU integration tests pass with serialization
5. Can generate 100 tokens without GPU reset

**Dependencies:** Phase 1 (loader, config), Phase 2 (reference implementation)

**Source Reference:** Memoria `src/gpu/`, `gpu/libgpu.hip`

---

## Phase 4: CLI Integration

**Goal:** User-facing CLI tool for inference with explicit device selection

**Requirements:**
- CLII-01: --model flag for GGUF path
- CLII-02: --device cpu|gpu (required)
- CLII-03: --max-tokens option
- CLII-04: --temperature option
- CLII-05: Prompt via --prompt or stdin
- CLII-06: Output to stdout
- CLII-07: Error on GPU init failure (no fallback)
- ERRO-01: Missing file error
- ERRO-02: Invalid GGUF error
- ERRO-03: Missing metadata error
- ERRO-04: GPU init error
- ERRO-05: VRAM exhaustion error

**Success Criteria:**
1. `rocmforge --model model.gguf --device cpu --prompt "Hi"` produces output
2. `rocmforge --model model.gguf --device gpu --prompt "Hi"` uses GPU
3. Invalid model path produces clear error message
4. GPU init failure produces clear error (no silent CPU fallback)
5. Help text documents all options

**Dependencies:** Phase 2 (CPU), Phase 3 (GPU)

**Source Reference:** Memoria `src/main.rs`

---

## Traceability Matrix

| Requirement | Phase | Status |
|-------------|-------|--------|
| LOAD-01 | Phase 1 | Pending |
| LOAD-02 | Phase 1 | Pending |
| LOAD-03 | Phase 1 | Pending |
| LOAD-04 | Phase 1 | Pending |
| LOAD-05 | Phase 1 | Pending |
| CONF-01 | Phase 1 | Pending |
| CONF-02 | Phase 1 | Pending |
| CONF-03 | Phase 1 | Pending |
| CONF-04 | Phase 1 | Pending |
| CONF-05 | Phase 1 | Pending |
| TOKN-01 | Phase 1 | Pending |
| TOKN-02 | Phase 1 | Pending |
| TOKN-03 | Phase 1 | Pending |
| TOKN-04 | Phase 1 | Pending |
| CPUF-01 | Phase 2 | Pending |
| CPUF-02 | Phase 2 | Pending |
| CPUF-03 | Phase 2 | Pending |
| CPUF-04 | Phase 2 | Pending |
| CPUF-05 | Phase 2 | Pending |
| CPUF-06 | Phase 2 | Pending |
| CPUF-07 | Phase 2 | Pending |
| CPUF-08 | Phase 2 | Pending |
| CPUF-09 | Phase 2 | Pending |
| GPUI-01 | Phase 3 | Pending |
| GPUI-02 | Phase 3 | Pending |
| GPUI-03 | Phase 3 | Pending |
| GPUI-04 | Phase 3 | Pending |
| GPUI-05 | Phase 3 | Pending |
| GPUI-06 | Phase 3 | Pending |
| GPUI-07 | Phase 3 | Pending |
| GPUI-08 | Phase 3 | Pending |
| GPUI-09 | Phase 3 | Pending |
| GPUI-10 | Phase 3 | Pending |
| GPUI-11 | Phase 3 | Pending |
| CLII-01 | Phase 4 | Pending |
| CLII-02 | Phase 4 | Pending |
| CLII-03 | Phase 4 | Pending |
| CLII-04 | Phase 4 | Pending |
| CLII-05 | Phase 4 | Pending |
| CLII-06 | Phase 4 | Pending |
| CLII-07 | Phase 4 | Pending |
| ERRO-01 | Phase 4 | Pending |
| ERRO-02 | Phase 4 | Pending |
| ERRO-03 | Phase 4 | Pending |
| ERRO-04 | Phase 4 | Pending |
| ERRO-05 | Phase 4 | Pending |

---
*Roadmap created: 2026-03-24*
