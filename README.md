# ROCmForge

**AMD GPU Inference Engine for Large Language Models**

A high-performance inference engine specifically designed for AMD GPUs using ROCm and HIP. ROCmForge provides efficient LLM inference capabilities on AMD hardware with fully-tested GPU kernels.

## Project Status

**Core GPU Acceleration Complete | End-to-End Integration in Progress**

| Component | Status | Notes |
|-----------|--------|-------|
| GPU Kernels | ✅ Complete | 41/41 tests passing (Phases 1-4) |
| HIP/ROCm Backend | ✅ Complete | AMD RX 7900 XT tested |
| GGUF Loader | ✅ Complete | Fixed spec compliance, vocab inference |
| Qwen2 Support | ✅ Complete | Phase 4.6 tensor mapping done |
| KV Cache | ✅ Complete | Paged attention cache |
| Sampler | ✅ Complete | CPU-based (GPU planned Phase 5.1) |
| HTTP Server | ✅ Complete | OpenAI-compatible API |
| CLI | ⚠️ Debugging | End-to-end generation crashes |
| Quantization | ❌ Missing | Q4_1, Q5_0, Q5_1 not supported |
| FP16 Compute | ❌ Missing | FP32 only |

## What Works

### GPU Kernels (100% Complete)

All transformer layer operations are GPU-accelerated with comprehensive testing:

- **Phase 1**: Basic kernels (scale, mask, softmax) - 3/3 tests
- **Phase 2**: RoPE (Rotary Position Embedding) - 5/5 tests
- **Phase 3a**: Non-Causal FlashAttention - 17/17 tests
- **Phase 3b**: Causal Masking for autoregressive decoding - 8/8 tests
- **Phase 4**: MLP Ops (SwiGLU, RMSNorm) - 8/8 tests

**Total: 41/41 tests passing with 1e-5 tolerance**

### HIP/ROCm Integration

- AMD Radeon RX 7900 XT (gfx1100, RDNA3) tested
- Wave32 optimization (256 thread blocks)
- GPU-only execution path (no CPU round-trips in transformer layers)

### GGUF Model Loading

- Fixed spec compliance (array encoding, value types, tensor types)
- Vocab size inference from tensor shapes
- Architecture detection (Qwen2, LLaMA, Mistral)
- Supports: F32, F16, Q8_0, Q4_0 tensor types

### Infrastructure

- HTTP Server: Axum-based REST API with OpenAI compatibility
- Scheduler: Request batching and queue management
- KV Cache: Paged attention cache for efficient inference
- Tokenizer: HuggingFace tokenizers with fallback
- Sampler: Top-k, top-p, temperature, repetition penalty

## What's In Progress

### CLI End-to-End Generation

The CLI crashes during inference with core dump. Individual components work, but the full pipeline has integration issues:

```bash
# This crashes with SIGSEGV
./target/release/rocmforge_cli generate \
  --gguf /path/to/model.gguf \
  --prompt "Hello" \
  --max-tokens 10
```

**Diagnosis**: Possible memory management or lifecycle issues between engine components.

## Known Issues

### Critical (Blockers)

1. **CLI Crashes**: `generate` command dumps core during inference
2. **Missing Quantization**: Q4_1, Q5_0, Q5_1 models cannot load
3. **Qwen2 Separate QKV**: Separate Q/K/V matrices need concatenation

### High Priority

4. **GPU Memory Leak** (kv_cache.rs:184): Leaks on page allocation failure
5. **Double-Free Risk** (hip_backend.rs:218): Auto-derived Clone causes corruption
6. **Race Condition** (hip_backend.rs:478): Flawed singleton initialization
7. **No End-to-End Tests**: Missing integration tests with real models

### Medium Priority

8. **Debug Output**: 50+ `eprintln!` statements in production code
9. **Code Duplication**: 3 separate KV cache implementations
10. **Inconsistent Error Types**: Mix of `i32`, `Result<(), String>`, `HipResult<T>`

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

### CLI (Experimental)

```bash
# Inspect GGUF model metadata
./target/release/rocmforge_cli inspect --model /path/to/model.gguf

# Generate text (may crash - known issue)
./target/release/rocmforge_cli generate \
  --gguf ~/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "The future of AI is" \
  --max-tokens 20 \
  --temperature 0.7
```

### HTTP Server

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

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Basic kernels (scale, mask, softmax) | ✅ Complete |
| Phase 2 | RoPE + KV Append | ✅ Complete |
| Phase 3 | FlashAttention (causal + non-causal) | ✅ Complete |
| Phase 4 | MLP Ops (SwiGLU, RMSNorm) | ✅ Complete |
| Phase 4.5 | GGUF Loader Fixes | ✅ Complete |
| Phase 4.6 | Qwen2 Tensor Mapping | ✅ Complete |
| Phase 5.1 | GPU Sampler (top-k/top-p on device) | ❌ Pending |
| Phase 5.2 | Custom GEMM (if needed) | ❌ Pending |
| Phase 5.3 | FP16 Support | ❌ Pending |
| Phase 5.4 | Wave64 Tuning (CDNA3) | ❌ Pending |

### Future Work

- [ ] Fix CLI crashes and enable end-to-end inference
- [ ] Quantization support (Q4_1, Q5_0, Q5_1)
- [ ] End-to-end integration tests with real models
- [ ] Multi-GPU tensor parallelism
- [ ] Performance benchmarks vs llama.cpp, vLLM
- [ ] Production deployment guide

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

## Contributing

See [docs/TODO.md](docs/TODO.md) for detailed task tracking and [docs/PLAN.md](docs/PLAN.md) for implementation roadmap.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference optimization
- [vLLM](https://github.com/vllm-project/vllm) - Efficient batching
- [candle](https://github.com/huggingface/candle) - Rust ML design patterns

## Disclaimer

This project is under active development. Core GPU kernels are complete and tested (41/41 tests passing), but end-to-end model execution has known issues. APIs may change.

---

**Status**: Kernels Complete | **Tests**: 41/41 Passing | **Hardware**: AMD Radeon RX 7900 XT (gfx1100) | **Last Updated**: January 2026
