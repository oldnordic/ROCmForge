# ROCmForge Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-dev] - 2026-01-20

### Added
- GGUF model loading with multiple quantization format support
- CPU SIMD backend with AVX2, NEON support
- GPU backend with HIP kernels for dequantization, matmul
- Flash attention implementation with GPU backend
- Paged KV cache for efficient inference
- HTTP server skeleton with OpenAI-compatible API structure
- Token sampling: top-k, top-p, temperature
- CLI tools: `models`, `serve`, `generate` commands
- Test infrastructure with 578 lib unit tests passing
- GPU test protection with `#[serial]` attribute (prevents GPU resets)

### Fixed
- GPU stream synchronization (hipBLAS vs hipMemcpy mismatch)
- Race condition in inference loop spawn
- Engine cleanup issues in CLI
- Test GPU resets: All GPU tests now have `#[serial]` protection

### Test Status
- **Lib tests**: 578 passing
- **Integration tests**: 35+ passing
- **Known failures**: 3 tests in embedding_to_lmhead_tests (metadata bug)
- **Known compilation errors**: q_dequant_tests, attention_gpu_tests

### Known Issues
- **GGUF Metadata Bug**: `metadata.vocab_size` returns 0 instead of actual value
  - Affects: `test_token_embedding_lookup_f32`, `test_batch_embedding_lookup`, `test_lm_head_matmul_correctness`
- **Compilation Errors**:
  - `tests/q_dequant_tests.rs`: Type mismatches (usize vs u8), unresolved imports
  - `tests/attention_gpu_tests.rs`: Borrow checker issues with moved values
- **Untested**: HTTP server `/v1/completions`, CLI `generate`/`serve` commands
- **No end-to-end validation**: Full inference pipeline not tested with real models
- **Approximately 27 compiler warnings** (unused imports, deprecated methods)

## [0.0.1] - Earlier Development

### Added
- Initial project structure
- Basic HIP kernel stubs
- GGUF loader (initial implementation)
- HTTP server skeleton
- Test infrastructure

---

**Hardware Tested**: AMD Radeon RX 7900 XT (gfx1100, RDNA3)
**ROCm Version**: 5.0+
**Rust Version**: 1.82+
