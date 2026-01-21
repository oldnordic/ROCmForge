# ROCmForge Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3-dev] - 2026-01-21

### Fixed
- **GPU Test Safety**: Added `#[serial]` attribute to 27 GPU-related tests across 6 files
  - Prevents GPU crashes when running tests with multiple threads
  - All tests using GPU_FIXTURE, HipBackend, HipBuffer, or load_to_gpu now protected
  - Files updated: embedding_to_lmhead_tests.rs, gguf_loader_tests.rs, simple_model_tests.rs, phase6_integration_model_loading.rs, attention_tests.rs, mlp_validation_tests.rs
- **Duplicate function definitions**: Removed fallback CPU-only functions that conflicted with GPU test functions

### Technical Details
- 37 test files now use `serial_test` crate
- 233 `#[serial]` attributes across all GPU test files
- Pattern established: All GPU tests must have `#[serial]` to prevent parallel execution crashes

## [0.1.2-dev] - 2026-01-21

### Fixed
- **multiProcessorCount Offset**: Fixed from 508 to 388 (caught by offset verification test)
- **Reduced compiler warnings**: From 483 down to 11 (unused imports, deprecated methods, dead code)

### Added
- **Offset Verification Tests**: Compile-time tests assert manual FFI offsets match bindgen-generated values
- **memoffset 0.9**: Added to dev-dependencies for `offset_of!` macro

### Technical Details
- All 7 device property offsets verified at compile time
- Internal `.planning/` documentation removed from repository (kept locally for development)

## [0.1.1-dev] - 2026-01-21

### Fixed
- **FFI Device Properties Bug**: Sanity check now validates ALL three dimensions (X, Y, Z) of maxThreadsDim
  - Previously only validated dim[0], allowing garbage values like [1024, 0, 0]
  - Fixes "block.y exceeds limit 0" errors during kernel launch
- **Duplicate DeviceLimits Assignment**: Removed duplicate assignment that overwrote vetted device limits
- **Incorrect multiProcessorCount Offset**: Fixed offset from 508 to 388 (caught by offset verification test)

### Added
- **Bindgen Infrastructure**: Generate HIP device bindings at compile time for offset verification
- **Offset Verification Tests**: Compile-time tests assert manual FFI offsets match bindgen-generated values
  - Catches struct layout changes across ROCm versions

### Technical Details
- bindgen 0.70 added to build-dependencies
- memoffset 0.9 added to dev-dependencies for `offset_of!` macro
- build.rs generates `hip_device_bindings.rs` with `hipDeviceProp_t` struct only
- All 7 device property offsets verified at compile time

### Test Status
- **Lib tests**: 820 passing (up from 578)
- **Offset verification**: All offsets verified
- **Known warnings**: 483 compiler warnings (mostly unused, no critical issues)

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
