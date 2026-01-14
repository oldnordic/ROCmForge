# Phase 6: End-to-End Integration Tests

## Goal

Validate full pipeline with real GGUF models and compare against llama.cpp reference.

## Current State

Individual components have tests, but no end-to-end validation with real models.

## Solution

Create comprehensive integration tests:

### Test Models
- Qwen2 0.5B (small, fast for testing)
- LLaMA 7B (validate larger model support)
- Q4_0 quantized variant

### Test Cases
1. **Loading** - Single-pass GGUF load, metadata extraction
2. **Forward Pass** - Single token generation, verify output
3. **Decode Loop** - Multi-token generation, verify KV cache
4. **Quantization** - Q4_0 model inference correctness
5. **Performance** - Tokens/sec benchmark vs llama.cpp
6. **Memory** - Peak memory usage tracking

### Validation Strategy
- Compare outputs against llama.cpp for identical inputs
- Use same GGUF model files for both
- Verify token-by-token output matches

## Files to Create

- `tests/integration/mod.rs`
- `tests/integration/qwen2_0_5b_test.rs`
- `tests/integration/llama_7b_test.rs`
- `tests/integration/quantized_test.rs`
- `tests/integration/benchmark.rs`

## Success Criteria

- [ ] All integration tests pass
- [ ] Output matches llama.cpp reference
- [ ] Performance within 20% of llama.cpp
- [ ] No memory leaks (ROCm stable after 1000 tokens)
