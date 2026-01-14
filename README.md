# ROCmForge

**AMD GPU Inference Engine for Large Language Models**

A high-performance inference engine specifically designed for AMD GPUs using ROCm and HIP. ROCmForge provides efficient LLM inference capabilities on AMD hardware with fully-tested GPU kernels.

## Project Status

**Alpha Software - Phase 26 (GQA Scaffolding)**

This is **alpha software** under active development. Some tests have compilation errors; end-to-end integration is incomplete.

| Component | Status | Notes |
|-----------|--------|-------|
| GPU Kernels | ✅ Complete | Phases 1-4: scale, mask, softmax, RoPE, FlashAttention, SwiGLU, RMSNorm |
| GPU Attention Path | ✅ Complete | Phase 7: GPU attention pipeline |
| GGUF Loader | ✅ Complete | F32, F16, Q8_0, Q4_0, MXFP4/MXFP6 support |
| MXFP Quantization | ✅ Complete | Phase 5: MXFP4/MXFP6 (OCP MX Spec v1.0) |
| KV Cache | ✅ Complete | Paged attention cache |
| HTTP Server | ✅ Complete | OpenAI-compatible API |
| Test Suite | ⚠️ Partial | 6 test files have compilation errors (Jan 2026) |
| CLI | ⚠️ Experimental | Untested end-to-end |
| End-to-End | ❌ Not Tested | Integration incomplete |

**Known Compilation Errors** (6 test files):
- `attention_gpu_tests` - 1 error
- `execution_plan_and_decode_tests` - 1 error
- `glm_model_tests` - 6 errors
- `gguf_loader_tests` - 1 error
- `hip_backend_smoke_tests` - 6 errors
- `multilayer_pipeline_tests` - 6 errors


## What Works

### GPU Kernels

Core transformer layer operations have GPU implementations:

- **Phase 1**: Basic kernels (scale, mask, softmax)
- **Phase 2**: RoPE (Rotary Position Embedding)
- **Phase 3a**: Non-Causal FlashAttention
- **Phase 3b**: Causal Masking for autoregressive decoding
- **Phase 4**: MLP Ops (SwiGLU, RMSNorm)
- **Phase 7**: GPU Attention Path
- **Phase 26**: GQA Support scaffolding (has compilation errors)

### Async GPU Loading

Phase 17: Multi-stream concurrent GPU uploads

- HIP Events for synchronization
- AsyncLoader for concurrent uploads
- See: `docs/ASYNC_LOADING_E2E_TEST_REPORT.md`

### MXFP Quantization

Phase 5: OCP MX Specification v1.0 compliant MXFP4/MXFP6

- E8M0 Scale Format (8-bit exponent-only)
- MXFP4: 4-bit E2M1 format
- MXFP6: 6-bit E3M2 format
- Block Scaling: 32 elements per block

### HIP/ROCm Integration

- AMD Radeon RX 7900 XT (gfx1100, RDNA3) tested
- Wave32 optimization (256 thread blocks)
- GPU-only execution path (no CPU round-trips in transformer layers)

### GGUF Model Loading

- Fixed spec compliance (array encoding, value types, tensor types)
- Vocab size inference from tensor shapes
- Architecture detection (Qwen2, LLaMA, Mistral, GLM)
- Supports: F32, F16, Q8_0, Q4_0, MXFP4, MXFP6

### Infrastructure

- HTTP Server: Axum-based REST API with OpenAI compatibility
- Scheduler: Request batching and queue management
- KV Cache: Paged attention cache for efficient inference
- Tokenizer: HuggingFace tokenizers with fallback
- Sampler: Top-k, top-p, temperature, repetition penalty

### Code Quality Improvements (Phase 15)

- **Logging**: Replaced 101 eprintln! statements with structured tracing macros
- **Naming**: Resolved AttentionBackend trait/enum naming conflict
- **Error Handling**: Audited 28 expect() calls (all acceptable)
- **Result Types**: Verified consistent naming conventions

## Known Issues

### High Priority

1. **CLI Stability**: Phase 21 fixes applied but not fully tested
   - Status: Fixed P0 GPU resource leak, P2 missing input validation
   - Remaining: Not tested end-to-end with real models
   - Workaround: Use HTTP server API which is more stable
   - Impact: CLI may still crash in edge cases
   - See: `docs/CLI_BUG_FIXES_2026-01-11.md`

2. **End-to-End Inference**: Not fully tested with real models
   - Status: Individual components tested, integration incomplete
   - Impact: Cannot guarantee reliable model execution
   - Plan: Add integration tests with real models

### Medium Priority (Non-Blockers)

3. **Compiler Warnings**: ~50 warnings remaining
   - Types: Dead code, unused imports, unused variables
   - Target: <10 warnings (only FFI `#[allow(...)]`)
   - Impact: Code quality, not functionality

4. **MQA/GQA CPU Fallback**: Multi-query attention uses CPU instead of GPU
   - Impact: Performance penalty for MQA/GQA models only
   - Workaround: CPU path is correct and tested
   - Plan: Add GPU kernels for MQA/GQA

5. **Missing Test Coverage**:
   - HTTP server integration tests (unit tests exist)
   - Sampler integration tests (inline tests only)
   - GPU memory exhaustion tests

### Low Priority

6. **RwLock Poisoning**: 6 expect() calls in kv_cache stats methods
   - Status: Documented as acceptable
   - Fix: Would require API breaking change
   - Impact: Low (stats methods only)

## What's In Progress

### Future Enhancements (Planned)

1. **CLI End-to-End Testing**: Test CLI with real models after Phase 21 fixes
2. **Integration Testing**: End-to-end tests with real models
3. **Warning Cleanup**: Reduce compiler warnings to <10
4. **Benchmark Suite**: Performance regression tracking
5. **MQA/GQA GPU Pipeline**: GPU acceleration for multi-query attention

## Architecture

```
src/
├── attention/      # Multi-head attention (GPU/CPU backends)
├── backend/        # HIP/ROCm backend abstraction
├── engine.rs       # Main inference engine
├── http/           # HTTP API server
├── kv_cache/       # Key-value cache (paged)
├── loader/         # GGUF model loader
├── mlp/            # MLP operations (SwiGLU, RMSNorm)
├── model/          # Configuration and execution plans
├── ops/            # High-level GPU operations
├── sampler/        # Token sampling (CPU)
├── scheduler/      # Request batching
├── tensor/         # Tensor data structures
└── tokenizer.rs    # Tokenization utilities
```

## Requirements

- **Rust**: 1.70+ (2021 edition)
- **GPU**: AMD GPU with ROCm 5.x+
- **OS**: Linux (ROCm requirement)
- **Memory**: 16GB+ recommended for 7B models

## Build

```bash
# Clone repository
git clone https://github.com/your-repo/ROCmForge.git
cd ROCmForge

# Build release binary
cargo build --release

# Run tests (requires AMD GPU)
cargo test --features rocm

# Run specific test
cargo test --features rocm --lib test_swiglu_matches_cpu_small
```

## Usage

### Testing

```bash
# Run all GPU kernel tests
cargo test --features rocm

# Test specific module
cargo test --features rocm --lib attention
cargo test --features rocm --lib mlp

# Monitor GPU during tests
watch -n 1 rocm-smi
```

### CLI (Experimental - Fixes Applied, Not Fully Tested)

```bash
# Inspect GGUF model metadata
./target/release/rocmforge_cli inspect --model /path/to/model.gguf

# Generate text (experimental - fixes applied but not fully tested)
./target/release/rocmforge_cli generate \
  --gguf ~/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "The future of AI is" \
  --max-tokens 20 \
  --temperature 0.7
```

### HTTP Server (Recommended for Testing)

```bash
# Start server
./target/release/rocmforge_cli serve --port 8080

# Health check
curl http://localhost:8080/health

# Completion request
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

## Roadmap

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 1 | Basic kernels (scale, mask, softmax) | ✅ Complete | - |
| Phase 2 | RoPE + KV Append | ✅ Complete | - |
| Phase 3 | FlashAttention (causal + non-causal) | ✅ Complete | - |
| Phase 4 | MLP Ops (SwiGLU, RMSNorm) | ✅ Complete | - |
| Phase 4.5 | GGUF Loader Fixes | ✅ Complete | - |
| Phase 4.6 | Qwen2 Tensor Mapping | ✅ Complete | - |
| Phase 5 | MXFP Quantization (MXFP4/MXFP6) | ✅ Complete | OCP MX Spec v1.0 |
| Phase 7 | GPU Attention Path | ✅ Complete | - |
| Phase 10 | Memory Pooling (ROCm workaround) | ✅ Complete | - |
| Phase 11 | P0/P1 Bug Fixes | ✅ Complete | - |
| Phase 15 | P1/P2 Code Quality Fixes | ✅ Complete | Logging, naming cleanup |
| Phase 17 | Async GPU Loading | ✅ Complete | Multi-stream uploads |
| Phase 20 | GPU Testing Safety | ✅ Complete | - |
| Phase 21 | CLI Stability Fixes | ✅ Complete | Untested end-to-end |
| Phase 26 | GQA Support Scaffolding | ⚠️ Incomplete | Has compilation errors |
| Phase 6 | GPU Sampler (top-k/top-p on device) | ❌ Pending | - |
| Phase 8 | Q4_1/Q5_0/Q5_1 Support | ❌ Not Verified | Claims unverified |
| Phase 9 | Code Quality (100% test health) | ❌ False Claim | 6 test files have errors |
| Phase 12-14 | - | ⚪ Skipped | Not documented |
| Phase 16 | - | ⚪ Skipped | Not documented |
| Phase 18 | Lazy ExecutionPlan | ⚠️ Claimed | "12x speedup" unverified |
| Phase 19 | - | ⚪ Skipped | Not documented |
| Phase 22-25 | - | ⚪ Skipped | Not documented |
| Future | End-to-End Integration Tests | ❌ Planned | Critical gap |
| Future | FP16 Compute Support | ❌ Planned | - |

### Future Work

**High Priority:**
- [ ] Fix 6 test files with compilation errors
- [ ] End-to-end integration tests with real models
- [ ] Remove unverified performance claims ("12x speedup", etc.)

**Medium Priority:**
- [ ] Test CLI end-to-end with real models
- [ ] GPU-based MXFP dequantization kernels
- [ ] Verify or remove Phase 18 (Lazy ExecutionPlan) claims

**Low Priority:**
- [ ] Multi-GPU tensor parallelism
- [ ] Performance benchmarks vs llama.cpp, vLLM
- [ ] Deployment guide

## Development

```bash
# Format code
cargo fmt

# Linter
cargo clippy -- -D warnings

# Run benchmarks
cargo bench

# Full test suite
cargo test --features rocm --workspace
```

## Dependencies

Key libraries:
- **axum**: HTTP server framework
- **tokio**: Async runtime
- **tokenizers**: HuggingFace tokenizers
- **half**: FP16 support
- **memmap2**: Memory-mapped I/O
- **serde/serde_json**: Serialization
- **tracing**: Structured logging

## Contributing

See [docs/TODO.md](docs/TODO.md) for detailed task tracking and [docs/PLAN.md](docs/PLAN.md) for implementation roadmap.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference optimization
- [vLLM](https://github.com/vllm-project/vllM) - Efficient batching
- [candle](https://github.com/huggingface/candle) - Rust ML design patterns

## Disclaimer

**This is alpha software.** It is not suitable for any critical use. APIs will change. Bugs exist.

**What this means:**
- ✅ Some GPU kernels are implemented
- ✅ Individual components have tests
- ⚠️ 6 test files have compilation errors (Jan 2026)
- ⚠️ End-to-end inference not fully tested
- ❌ Not ready for any real deployment
- ❌ Use at your own risk

---

**Status**: Alpha | **Tests**: Partial (6 files with compilation errors) | **Hardware**: AMD Radeon RX 7900 XT (gfx1100) | **Last Updated**: January 2026 | **Phase**: 26 (GQA Scaffolding)
