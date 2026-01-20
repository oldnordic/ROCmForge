# Technology Stack

**Analysis Date:** 2025-01-20

## Languages

**Primary:**
- Rust 2021 Edition - Core inference engine and all libraries
  - Minimum rust-version: 1.82 (required for stable `std::simd`)
  - Located in: `/home/feanor/Projects/ROCmForge/src/`

**Secondary:**
- HIP (Heterogeneous Interface for Portability) - GPU kernel language
  - ROCm GPU kernels in: `/home/feanor/Projects/ROCmForge/kernels/`
  - Compiled to HSACO format via hipcc

## Runtime

**Environment:**
- Linux only (ROCm restriction)
- ROCm 5.0+ required for AMD GPU support
- Target GPU architectures: gfx1100 (RDNA3), gfx1030 (RDNA2), gfx90a (CDNA2)

**Package Manager:**
- Cargo
- Lockfile: `Cargo.lock` present (92KB)

## Frameworks

**Core:**
- axum 0.7 - HTTP server framework for `/v1/completions` API
- tokio 1.0 - Async runtime (features: "full")
- tower 0.4 / tower-http 0.5 - HTTP middleware (CORS, tracing)

**Testing:**
- No explicit test framework - uses Rust's built-in `#[test]`
- serial_test 3.0 - Sequential GPU test execution
- mockall 0.12 - Mocking support
- proptest 1.4 - Property-based testing
- criterion 0.5 - Benchmarking with HTML reports

**Build/Dev:**
- cc 1.0 - C compilation support for HIP kernels
- clap 4.5 - CLI argument parsing for `rocmforge_cli`

## Key Dependencies

**Critical:**

| Package | Version | Purpose |
|---------|---------|---------|
| `tokenizers` | 0.15 | Hugging Face tokenizer support (WordLevel, BPE, etc.) |
| `memmap2` | 0.9 | Memory-mapped file loading for GGUF models |
| `half` | 2.4 | FP16/BF16 arithmetic support |
| `rayon` | 1.10 | Parallel CPU operations |
| `bytemuck` | 1.15 | Safe memory casting for tensor operations |
| `prometheus-client` | 0.22 | Metrics export at `/metrics` endpoint |

**Infrastructure:**

| Package | Version | Purpose |
|---------|---------|---------|
| `serde` / `serde_json` | 1.0 | Serialization for config, API responses |
| `tracing` / `tracing-subscriber` | 0.1 / 0.3 | Structured logging |
| `anyhow` / `thiserror` | 1.0 / 1.0 | Error handling |
| `reqwest` | 0.11 | HTTP client for CLI (json, stream features) |
| `reqwest-eventsource` | 0.4 | Server-Sent Events for streaming responses |
| `byteorder` | 1.5 | Binary GGUF format parsing |
| `flate2` | 1.0 | GZIP decompression for GGUF metadata |
| `once_cell` | 1.18 | Global singleton initialization |
| `raw-cpuid` | 11.0 | Runtime CPU feature detection for SIMD dispatch |
| `hex` | 0.4 | Hex encoding/decoding |

**Feature-Gated:**

| Package | Version | Feature | Purpose |
|---------|---------|---------|---------|
| `sqlitegraph` | 1.0 | `context` | Graph database for context engine (HNSW vector search) |
| (placeholder) | - | `rocm` | HIP/ROCm bindings (not yet implemented) |
| (placeholder) | - | `simd` | CPU SIMD backend (requires nightly) |

## Configuration

**Environment:**
- Configuration via environment variables
- Example config: `/.env.example`
- Key variables:
  - `ROCMFORGE_GGUF` - Path to GGUF model file
  - `ROCMFORGE_TOKENIZER` - Path to tokenizer.json
  - `ROCMFORGE_MODELS` - Model discovery directory (default: `./models`)
  - `RUST_LOG` - Standard tracing filter
  - `ROCFORGE_LOG_LEVEL` - Simple log level override
  - `ROCFORGE_LOG_FORMAT` - "human" or "json"
  - `ROCFORGE_GPU_DEVICE` - GPU device number (default: 0)
  - `ROCM_PATH` - ROCm installation path (default: `/opt/rocm`)
  - `ROCm_ARCH` - Target GPU architecture (default: `gfx1100`)

**Build:**
- Build configuration: `/build.rs`
- Compile HIP kernels when `rocm` feature is enabled
- 32 HIP kernel sources in `/kernels/`
- Kernels compiled via `hipcc` to HSACO format

## Platform Requirements

**Development:**
- Rust 1.82+
- ROCm 5.0+ with hipcc compiler
- AMD GPU with ROCm support (tested on RX 7900 XT)
- 8GB+ VRAM recommended

**Production:**
- Linux x86_64
- AMD GPU (RDNA2/3 or CDNA)
- No Windows support (ROCm limitation)

**Build Commands:**
```bash
cargo build --release
cargo test --lib -- --test-threads=1  # GPU tests require serial execution
```

---

*Stack analysis: 2025-01-20*
