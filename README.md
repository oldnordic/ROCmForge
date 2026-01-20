# ROCmForge

**AMD GPU Inference Engine for Large Language Models - Development Stage**

An LLM inference engine for AMD GPUs using ROCm and HIP. Loads GGUF-format models and provides an HTTP API.

**⚠️ Development Status: Not production-ready. Use for testing and development only.**

## Current Status

**Version:** 0.1.0-development

| Component | Status | Notes |
|-----------|--------|-------|
| Build | ✅ Working | Release builds in ~55 seconds |
| Lib Tests | ✅ 578 passing | Unit tests for core components |
| GGUF Loading | ⚠️ Partial | Loads metadata, some tensor types have issues |
| GPU Backend | ⚠️ Development | HIP kernels implemented, not all tested |
| HTTP Server | ⚠️ Untested | Server exists, not validated in production |
| CLI | ⚠️ Partial | `models` command works, others untested |
| Integration Tests | ⚠️ Mixed | Some pass, 3 failing in embedding_to_lmhead_tests |

### Test Results (Actual)

```
Lib tests:         578 passed
Integration tests: 35+ passed across multiple test files
Known failures:    3 tests in embedding_to_lmhead_tests (metadata parsing bug)
Known issues:      q_dequant_tests has compilation errors
                    attention_gpu_tests has compilation errors
```

### What Actually Works

Based on actual testing:

- **Build System**: `cargo build --release` completes successfully
- **Core Libraries**: Tensor operations, KV cache, scheduler components tested
- **GGUF Metadata Parsing**: Reads architecture, layers, some tensor info
- **GPU Detection**: Successfully detects AMD GPUs (tested on RX 7900 XT)
- **Test Infrastructure**: All GPU tests have `#[serial]` protection (no GPU resets)
- **Model Discovery**: `rocmforge_cli models` finds cached GGUF files

### Known Issues (Honest Assessment)

1. **GGUF Metadata Bug**: `metadata.vocab_size` returns 0 instead of actual value
   - Affects: `test_token_embedding_lookup_f32`, `test_batch_embedding_lookup`, `test_lm_head_matmul_correctness`
   - Root cause: Metadata parser not correctly reading vocab_size from GGUF

2. **Compilation Errors**:
   - `tests/q_dequant_tests.rs`: Type mismatches (usize vs u8), unresolved imports
   - `tests/attention_gpu_tests.rs`: Borrow checker issues with moved values

3. **Untested Features**:
   - HTTP server `/v1/completions` endpoint not validated
   - CLI `generate` and `serve` commands not tested
   - End-to-end inference not validated with real models

4. **Approximately 27 compiler warnings** (unused imports, deprecated methods)

## Requirements

- **Rust**: 1.82+
- **ROCm**: 5.0+ (Linux only)
- **GPU**: AMD GPU with ROCm support (tested on RX 7900 XT)
- **Memory**: 8GB+ VRAM recommended

## Build

```bash
# Build release binary
cargo build --release

# Run lib tests
cargo test --lib

# Run integration tests (excluding q_dequant_tests which has compilation errors)
cargo test --test '*'
```

## Usage

### Model Discovery

```bash
# List available GGUF models
cargo run --release --bin rocmforge_cli -- models
```

Output example:
```
- qwen2-0_5b-instruct-q5_k_m
  gguf: models/Qwen2-0.5B-Instruct-GGUF/qwen2.5-0.5b.Q4_K_M.gguf
  arch: qwen2 | layers: 24 | heads: 14 | hidden: 896 | ctx: 2048
```

### Running Tests

```bash
# Lib unit tests (578 tests, ~0.5s)
cargo test --lib

# Integration tests with GPU
cargo test --test multilayer_pipeline_tests

# Run all tests sequentially (prevents GPU conflicts)
cargo test -- --test-threads=1
```

## Architecture

```
src/
├── attention/       # Multi-head attention implementations
├── backend/         # CPU and HIP/ROCm backends
├── engine.rs        # Inference engine
├── http/            # HTTP API server (untested)
├── kv_cache/        # Paged key-value cache
├── loader/          # GGUF model loader
├── model/           # Model configuration and execution
├── sampler/         # Token sampling
├── scheduler/       # Request batching
└── tensor/          # Tensor data structures
```

## Known Limitations

| Issue | Impact |
|-------|--------|
| GGUF vocab_size metadata bug | 3 integration tests fail |
| q_dequant_tests compilation errors | Those tests cannot run |
| attention_gpu_tests compilation errors | GPU attention tests cannot run |
| HTTP server untested | Unknown if `/v1/completions` works |
| CLI generate/serve untested | Unknown if inference actually works |
| No end-to-end validation | Haven't confirmed full inference pipeline |

## Development Focus

Current priorities (based on actual test failures):

1. Fix GGUF metadata parser (`vocab_size` returns 0)
2. Fix compilation errors in q_dequant_tests and attention_gpu_tests
3. Validate HTTP server `/v1/completions` endpoint
4. Test end-to-end inference with real GGUF models
5. Add integration tests for CLI commands

## Contributing

This is a development project. Areas needing work:
- GGUF metadata parser fixes
- GPU kernel validation
- HTTP server testing
- End-to-end inference testing
- Error handling improvements

## License

GPL-3.0

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format
- [vLLM](https://github.com/vllm-project/vllm) - Paged attention
- [candle](https://github.com/huggingface/candle) - Rust ML design
