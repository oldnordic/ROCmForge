# Codebase Structure

**Analysis Date:** 2026-01-18

## Directory Layout

```
ROCmForge/
├── src/                    # Main source code
│   ├── api/               # HTTP API layer
│   ├── backend/           # HIP/ROCm backend abstraction
│   ├── bin/               # CLI binaries and utilities
│   ├── ggml/              # GGML-compatible operations
│   │   └── hip_backend/   # HIP backend implementation
│   │       └── ops/       # Individual GPU kernels
│   ├── http/              # HTTP server (Axum)
│   ├── kv_cache/          # Key-value cache (paged)
│   ├── loader/            # GGUF model loader
│   ├── model/             # Model configs and execution
│   ├── ops/               # High-level operations
│   ├── scheduler/         # Request batching
│   ├── tensor/            # Tensor operations
│   ├── tokenizer.rs       # Tokenization
│   ├── engine.rs          # Core inference engine
│   └── lib.rs             # Library root
├── tests/                 # Integration tests
├── docs/                  # Documentation
├── benches/               # Criterion benchmarks
├── build.rs               # Build script (kernel compilation)
├── Cargo.toml             # Package manifest
└── Makefile               # Build automation
```

## Directory Purposes

**src/api/**
- Purpose: HTTP API interface (OpenAI-compatible)
- Contains: API types, handlers
- Key files: `types.rs`, `mod.rs`
- Subdirectories: None

**src/backend/**
- Purpose: HIP/ROCm GPU backend abstraction
- Contains: Backend implementation, GPU management
- Key files: `hip_backend.rs`, `mod.rs`
- Subdirectories: None

**src/bin/**
- Purpose: Executable binaries
- Contains: CLI tools and test utilities
- Key files: `rocmforge_cli.rs`, `inspect_gguf.rs`, `test_gguf_load.rs`, `run_simple_model.rs`
- Subdirectories: None

**src/ggml/**
- Purpose: GGML-compatible operations
- Contains: GGML operation implementations
- Key files: `mod.rs`, tensor operations
- Subdirectories: `hip_backend/`

**src/ggml/hip_backend/**
- Purpose: HIP-specific GGML backend
- Contains: Backend implementation, kernel management
- Key files: `mod.rs`
- Subdirectories: `ops/`

**src/ggml/hip_backend/ops/**
- Purpose: Individual GPU kernel implementations
- Contains: matmul, softmax, rope, swiglu, etc.
- Key files: `matmul.rs`, `softmax.rs`, `rope.rs`, `swiglu.rs`
- Subdirectories: None

**src/http/**
- Purpose: HTTP server implementation (Axum)
- Contains: Server, routes, SSE streaming
- Key files: `server.rs`
- Subdirectories: None

**src/kv_cache/**
- Purpose: Paged attention KV cache
- Contains: Cache implementation, memory management
- Key files: `kv_cache.rs`, `mod.rs`
- Subdirectories: None

**src/loader/**
- Purpose: GGUF model loader
- Contains: Parser, weight loading
- Key files: `gguf.rs`
- Subdirectories: None

**src/model/**
- Purpose: Model configurations and execution
- Contains: Config structs, execution plan
- Key files: `config.rs`, `execution_plan.rs`
- Subdirectories: None

**src/scheduler/**
- Purpose: Request batching and scheduling
- Contains: Queue, batch management
- Key files: `mod.rs`
- Subdirectories: None

**src/sampler/**
- Purpose: Token sampling algorithms
- Contains: Top-k, top-p, temperature sampling
- Key files: `mod.rs`
- Subdirectories: None

**src/tensor/**
- Purpose: Tensor operations and math
- Contains: Matmul, element-wise ops
- Key files: `mod.rs`, `matmul.rs`
- Subdirectories: None

**tests/**
- Purpose: Integration tests
- Contains: Backend tests, loader tests, E2E tests
- Key files: `hip_blas_matmul_tests.rs`, `loader_tests.rs`
- Subdirectories: `common/`

## Key File Locations

**Entry Points:**
- `src/bin/rocmforge_cli.rs` - Main CLI entry point
- `src/lib.rs` - Library root

**Configuration:**
- `Cargo.toml` - Package manifest and dependencies
- `build.rs` - Build script (HIP kernel compilation)
- `Makefile` - Build automation

**Core Logic:**
- `src/engine.rs` - Central inference orchestrator
- `src/loader/gguf.rs` - GGUF model loading
- `src/backend/hip_backend.rs` - GPU backend
- `src/http/server.rs` - HTTP server

**Testing:**
- `tests/` - Integration tests
- `src/**/*test*.rs` - Unit tests (inline)
- `benches/` - Criterion benchmarks

**Documentation:**
- `docs/` - Project documentation
- `CLAUDE.md` - Development rules and instructions

## Naming Conventions

**Files:**
- `snake_case.rs` for modules (kebab-case for directories)
- `*_test.rs` or `tests/*` for test files
- `lib.rs` for library root
- `main.rs` or binary name for executables

**Directories:**
- `snake_case` for all directories
- Singular names (`kv_cache/`, not `kv_caches/`)
- Plural for collections (`tests/`, `benches/`)

**Special Patterns:**
- `mod.rs` for directory exports
- `bin/` for executable binaries
- `ops/` for operation collections

## Where to Add New Code

**New Feature (GPU operation):**
- Primary code: `src/ggml/hip_backend/ops/{operation}.rs`
- Tests: `src/ggml/hip_backend/ops/{operation}_test.rs` or `tests/`
- Documentation: Inline rustdoc comments

**New Feature (API endpoint):**
- Primary code: `src/http/server.rs` or new `src/api/{feature}.rs`
- Tests: `tests/{feature}_test.rs`

**New Feature (Model architecture):**
- Primary code: `src/model/{feature}.rs`
- Config updates: `src/model/config.rs`
- Tests: `tests/{feature}_test.rs`

**New Component/Module:**
- Implementation: `src/{module}/mod.rs`
- Types: `src/{module}/types.rs` if needed
- Tests: `tests/{module}_test.rs`

**Utilities:**
- Shared helpers: Create in relevant module or `src/util/`
- Type definitions: Inline in module or `src/types.rs`

## Special Directories

**target/**
- Purpose: Cargo build output
- Source: Auto-generated by Cargo
- Committed: No (in .gitignore)

**src/bin/**
- Purpose: Executable binaries (multiple binaries supported)
- Source: Source code for each binary
- Committed: Yes

**.codemcp/**
- Purpose: CodeMCP AI assistant configuration
- Source: Generated/configured
- Committed: Yes (config.toml)

---

*Structure analysis: 2026-01-18*
*Update when directory structure changes*
