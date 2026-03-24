# Requirements: rocmforge

**Defined:** 2026-03-24
**Core Value:** Correct, fast inference on AMD hardware with zero hardcoded assumptions

## v1 Requirements

### Loader

- [x] **LOAD-01**: System can memory-map GGUF file for efficient weight access
- [x] **LOAD-02**: System parses GGUF header to extract metadata keys and values
- [x] **LOAD-03**: System extracts model architecture string (e.g., "qwen2") from metadata
- [x] **LOAD-04**: System extracts tensor metadata: name, dimensions, GGML type, offset
- [x] **LOAD-05**: System validates GGUF magic bytes and version compatibility

### Config

- [x] **CONF-01**: System derives ModelConfig from GGUF metadata (layers, hidden, heads, etc.)
- [x] **CONF-02**: System uses ModelTraits registry to map architecture string to RoPE style
- [x] **CONF-03**: System determines attention layout (SplitQkv vs FusedQkv) from traits
- [x] **CONF-04**: System infers intermediate_size from tensor shape when metadata missing
- [x] **CONF-05**: System determines vocab_size from tokenizer.tokens.len(), NOT GGUF key

### Tokenizer

- [ ] **TOKN-01**: System parses tokenizer.json from GGUF metadata
- [ ] **TOKN-02**: System encodes text to token IDs using BPE merge rules
- [ ] **TOKN-03**: System decodes token IDs back to text
- [x] **TOKN-04**: System applies ChatML template for Qwen2.5 chat formatting

### CPU Backend

- [ ] **CPUF-01**: System allocates CpuModelWeights with dequantized f32 tensors
- [ ] **CPUF-02**: System implements RMS normalization for layer norm
- [ ] **CPUF-03**: System implements RoPE embedding (NeoX style for Qwen)
- [ ] **CPUF-04**: System implements attention with KV cache (single request)
- [ ] **CPUF-05**: System implements FFN (SwiGLU for Qwen2.5)
- [ ] **CPUF-06**: System implements greedy sampling
- [ ] **CPUF-07**: System implements top-p (nucleus) sampling
- [ ] **CPUF-08**: System runs prefill for single prompt (process all tokens at once)
- [ ] **CPUF-09**: System runs decode loop (one token at a time with KV cache)

### GPU Backend

- [ ] **GPUI-01**: System initializes HIP context and selects GPU device
- [ ] **GPUI-02**: System allocates DeviceBuffer for GPU memory management
- [ ] **GPUI-03**: System copies weights to VRAM with dequantization kernel
- [ ] **GPUI-04**: System implements GPU-side RMS normalization kernel
- [ ] **GPUI-05**: System implements GPU-side RoPE kernel (NeoX style)
- [ ] **GPUI-06**: System implements GPU-side attention kernel with KV cache
- [ ] **GPUI-07**: System implements GPU-side FFN kernel
- [ ] **GPUI-08**: System implements GPU-side greedy sampling
- [ ] **GPUI-09**: System implements GPU-side top-p sampling
- [ ] **GPUI-10**: System runs prefill on GPU (single request)
- [ ] **GPUI-11**: System runs decode loop on GPU (one token at a time)

### CLI

- [ ] **CLII-01**: User can specify model path via --model flag
- [ ] **CLII-02**: User can select device via --device cpu|gpu (required, no default)
- [ ] **CLII-03**: User can set max_tokens for generation
- [ ] **CLII-04**: User can set temperature for sampling
- [ ] **CLII-05**: User can provide prompt via --prompt or stdin
- [ ] **CLII-06**: System outputs generated text to stdout
- [ ] **CLII-07**: System reports error if GPU init fails (no CPU fallback)

### Error Handling

- [ ] **ERRO-01**: System reports clear error on missing GGUF file
- [ ] **ERRO-02**: System reports clear error on invalid GGUF format
- [ ] **ERRO-03**: System reports clear error on missing required metadata
- [ ] **ERRO-04**: System reports clear error on GPU init failure
- [ ] **ERRO-05**: System reports clear error on VRAM exhaustion

## v2 Requirements

### HTTP Server

- **SERV-01**: System exposes OpenAI-compatible /v1/chat/completions endpoint
- **SERV-02**: System exposes Claude-compatible /v1/messages endpoint
- **SERV-03**: System handles concurrent requests with request queuing
- **SERV-04**: System supports streaming responses (SSE)

### Batched Inference

- **BATCH-01**: System supports batched prefill for multiple prompts
- **BATCH-02**: System supports continuous batching during decode

## Out of Scope

| Feature | Reason |
|---------|--------|
| CUDA support | AMD-only project, ROCm focus |
| Multi-GPU | v1 targets single GPU |
| Speculative decoding | Complexity not justified for v1 |
| Custom quantization | Support only what GGUF provides |
| HTTP server | CLI-first, server in v2 |
| Batched inference | Single request simpler for correctness |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| LOAD-01 | Phase 1 | Complete |
| LOAD-02 | Phase 1 | Complete |
| LOAD-03 | Phase 1 | Complete |
| LOAD-04 | Phase 1 | Complete |
| LOAD-05 | Phase 1 | Complete |
| CONF-01 | Phase 1 | Complete |
| CONF-02 | Phase 1 | Complete |
| CONF-03 | Phase 1 | Complete |
| CONF-04 | Phase 1 | Complete |
| CONF-05 | Phase 1 | Complete |
| TOKN-01 | Phase 1 | Pending |
| TOKN-02 | Phase 1 | Pending |
| TOKN-03 | Phase 1 | Pending |
| TOKN-04 | Phase 1 | Complete |
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

**Coverage:**
- v1 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-24*
*Last updated: 2026-03-24 after initialization*
