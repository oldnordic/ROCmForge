# Technology Stack

**Analysis Date:** 2025-01-20

## Languages

**Primary:**
- Rust 2021 Edition - Core inference engine, all modules

**Minimum Rust Version:**
- 1.82+ (required for stable `std::simd`)

**Secondary:**
- HIP (Heterogeneous Interface for Portability) - GPU kernels in `kernels/` directory
- C - FFI bindings for ROCm/HIP libraries (via `extern "C"` blocks)

## Runtime

**Environment:**
- Linux (ROCm is Linux-only)
- ROCm 5.0+ / HIP 7.1+

**Package Manager:**
- Cargo - Rust package manager
- Lockfile: `Cargo.lock` (present, 92653 bytes)

**Build System:**
- `cargo build` with custom `build.rs` for HIP kernel compilation
- `hipcc` compiler invoked via build script to compile `.hip` kernel sources to `.hsaco` binaries

## Frameworks

**Core:**
- std::simd (Rust 1.82+) - CPU SIMD operations for fallback backend (feature-gated: `simd`)
- Custom HIP FFI layer - Direct ROCm bindings in `src/backend/hip_backend/backend.rs`
- Custom GGUF loader - GGUF format model parsing in `src/loader/`

**HTTP Server:**
- axum 0.7 - HTTP web framework (features: json)
- tokio 1.0 - Async runtime (features: full)
- tower 0.4 - Middleware utilities
- tower-http 0.5 - HTTP middleware (features: cors, trace)

**Testing:**
- Built-in Rust test harness - Unit and integration tests
- serial_test 3.0 - Sequential test execution for GPU isolation
- proptest 1.4 - Property-based testing
- tempfile 3.8 - Temporary file/directory creation
- mockall 0.12 - Mocking framework

**Benchmarking:**
- criterion 0.5 - Statistical benchmarks (features: html_reports)

**CLI:**
- clap 4.5 - Command-line argument parsing (features: derive)

## Key Dependencies

**GPU/Compute:**
- ROCm HIP (amdhip64, hipblas, hiprtc) - Linked via `build.rs` for GPU operations
- None (placeholder crates commented out in `Cargo.toml`: `hip`, `hip-sys`)

**HTTP/Networking:**
- reqwest 0.11 - HTTP client (features: json, stream)
- reqwest-eventsource 0.4 - Server-Sent Events support

**Tokenization:**
- tokenizers 0.15 - Hugging Face tokenizer library (features: http)

**Metrics:**
- prometheus-client 0.22 - Prometheus metrics export

**Serialization:**
- serde 1.0 - Serialization framework (features: derive)
- serde_json 1.0 - JSON serialization

**Math:**
- half 2.4 - FP16/BF16 types (features: serde)
- num-traits 0.2 - Numeric traits
- rand 0.8 - Random number generation
- rand_chacha 0.3 - ChaCha RNG

**Async:**
- futures 0.3 - Async utilities
- async-stream 0.3 - Async stream support

**File I/O:**
- memmap2 0.9 - Memory-mapped files
- byteorder 1.5 - Byte order handling
- flate2 1.0 - Compression (gzip)
- rayon 1.10 - Parallel processing

**System:**
- bytemuck 1.15 - Safe byte casting (features: derive)
- raw-cpuid 11 - CPU feature detection
- once_cell 1.18 - One-time initialization
- hex 0.4 - Hex encoding/decoding

**Error Handling:**
- anyhow 1.0 - Error context
- thiserror 1.0 - Error derive macros

**Logging/Tracing:**
- tracing 0.1 - Instrumentation
- tracing-subscriber 0.3 - Log routing (features: env-filter, json)

**Context Engine (feature-gated):**
- sqlitegraph 1.0 - Graph database with HNSW vector indexing (feature: `context`)

**Build:**
- cc 1.0 - C compilation support

## Configuration

**Environment:**
- Environment variable based (see `.env.example` for complete list)
- No runtime config file - all config via env vars

**Key configs:**
- `ROCM_PATH` - ROCm installation directory (default: `/opt/rocm`)
- `HIPCC` - HIP compiler path (default: `$ROCM_PATH/bin/hipcc`)
- `ROCm_ARCH` - Target GPU architecture (default: `gfx1100`)
- `ROCMFORGE_GGUF` - Path to GGUF model file
- `ROCMFORGE_TOKENIZER` - Path to tokenizer.json
- `RUST_LOG` - Tracing filter (default: `info`)
- `ROCFORGE_LOG_LEVEL` - Simple log level override
- `ROCFORGE_LOG_FORMAT` - `human` or `json`
- `ROCMFORGE_GPU_DEVICE` - GPU device number (default: 0)
- `ROCMORGE_TRACE_SAMPLE_RATE` - OTEL trace sampling (default: 0.1)
- `ROCMFORGE_MAX_TRACES` - Max traces in memory (default: 1000)

**Build:**
- `build.rs` - Compiles HIP kernels from `kernels/*.hip` to `.hsaco` binaries
- Kernel binaries embedded in build via `cargo:rustc-env`
- Custom kernel tuning via env vars: `ROCFORGE_BLOCK_SIZE`, `ROCFORGE_WARP_SIZE`, `ROCFORGE_USE_LDS`, `ROCFORGE_LDS_SIZE`, `ROCFORGE_TILE_K`, `ROCFORGE_TILE_N`

## Platform Requirements

**Development:**
- Linux x86_64
- Rust 1.82+
- ROCm 5.0+ with HIP compiler
- AMD GPU with ROCm support (tested on RX 7900 XT - gfx1100)
- 8GB+ VRAM recommended

**Production Target:**
- Linux x86_64 only (ROCm limitation)
- AMD RDNA2/CDNA GPUs or newer
- ROCm 5.0+, 6.0+, or 7.0+ compatible drivers

**Feature Flags:**
- `default` - No features enabled
- `rocm` - Enable ROCm/HIP GPU backend (requires ROCm installation)
- `simd` - Enable CPU SIMD backend (requires nightly Rust)
- `avx512` - Enable AVX-512 code paths (implies `simd`, opt-in due to CPU throttling)
- `context` - Enable SQLiteGraph context engine
- `cuda` - Placeholder for future CUDA support (not implemented)

**Release Profile:**
- LTO enabled
- Single codegen unit
- Panic abort (no unwinding)

---

*Stack analysis: 2025-01-20*
