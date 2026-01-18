# Technology Stack

**Analysis Date:** 2026-01-18

## Languages

**Primary:**
- Rust 2021 Edition - All application code (`Cargo.toml:4`)

**Secondary:**
- C/C++ - GPU kernels (via HIPRTC compilation)
- Shell - Build scripts (Makefile)

## Runtime

**Environment:**
- ROCm/HIP runtime - AMD GPU platform (`build.rs:9-13` links amdhip64, hipblas, hiprtc)
- Tokio async runtime - Full featured async runtime (`Cargo.toml:16`)

**Package Manager:**
- Cargo - Rust package manager
- Lockfile: `Cargo.lock` present

## Frameworks

**Core:**
- Axum - HTTP web server with JSON support (`Cargo.toml:15`)
- Tokio - Async runtime with full features (`Cargo.toml:16`)

**Testing:**
- Rust built-in `#[test]` framework - Unit tests
- Criterion 0.5 - Benchmarking (`Cargo.toml:66`)
- Proptest 1.4 - Property-based testing (`Cargo.toml:67`)

**Build/Dev:**
- rustfmt - Code formatting (default settings)
- cargo - Build and test runner

## Key Dependencies

**Critical:**
- `tokenizers` (HuggingFace) - Text tokenization (`Cargo.toml:23`)
- `serde` / `serde_json` - Serialization (`Cargo.toml:26-27`)
- `half` (2.4) - Half-precision floating point (`Cargo.toml:30`)
- `anyhow` / `thiserror` - Error handling (`Cargo.toml:37-38`)

**Infrastructure:**
- `tracing` / `tracing-subscriber` - Structured logging (`Cargo.toml:41-42`)
- `memmap2` - Memory-mapped file I/O (`Cargo.toml:52`)
- `rayon` - Parallel processing (`Cargo.toml:57`)
- `bytemuck` - Safe byte casting (`Cargo.toml:54`)

## Configuration

**Environment:**
- Environment variables for model paths - `src/http/server.rs:504-510`
  - `ROCMFORGE_GGUF` - Model file path
  - `ROCMFORGE_TOKENIZER` - Tokenizer path
  - `ROCMFORGE_MODELS` - Models directory

**Build:**
- `Cargo.toml` - Rust package configuration
- `build.rs` - Build script (HIP kernel compilation)
- `Makefile` - Build automation

## Platform Requirements

**Development:**
- Linux with ROCm support (tested on AMD RX 7900 XT - gfx1100)
- HIP/ROCm SDK for GPU compilation
- Cargo 1.70+ (Rust 2021 edition)

**Production:**
- AMD GPU with ROCm support (gfx1100 default target)
- No cloud/container deployment configured

---

*Stack analysis: 2026-01-18*
*Update after major dependency changes*
