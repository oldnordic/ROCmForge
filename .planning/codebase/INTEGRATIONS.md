# External Integrations

**Analysis Date:** 2025-01-20

## APIs & External Services

**None actively used.**

The project does not call external APIs in production code.

**Note on future integrations:**
- `reqwest` and `reqwest-eventsource` are dependencies but only used by the CLI client
- CLI connects to local ROCmForge server, not external services
- Context engine's `EmbeddingModel` trait (`/src/context/graph_context.rs`) is designed to support external embedding APIs (OpenAI, etc.) but only `DummyEmbedding` is implemented

## Data Storage

**Databases:**
- SQLiteGraph (feature-gated via `context` feature)
  - Package: `sqlitegraph` 1.0
  - Purpose: Graph-based context storage with HNSW vector indexing
  - Location: `/src/context/graph_context.rs`
  - Connection: File-based or in-memory
  - Usage: Optional context engine for LLM memory augmentation

**File Storage:**
- Local filesystem only
- GGUF models: User-specified paths, auto-discovered from `ROCMFORGE_MODELS` directory
- Tokenizer files: JSON format, auto-detected from model directory or embedded in GGUF
- HSACO kernels: Compiled to `target/` during build, loaded at runtime

**Caching:**
- Tokenizer cache: `/src/models.rs` - `CachedTokenizer` struct
- Embedded tokenizer JSON extracted from GGUF and cached
- Cache hit/miss metrics tracked in `/src/tokenizer.rs`

## Authentication & Identity

**Auth Provider:**
- None
- HTTP server (`/src/http/server.rs`) has no authentication
- Designed for local testing only

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Honeycomb, etc.)

**Logs:**
- Framework: `tracing` + `tracing-subscriber`
- Configuration: `/src/logging/mod.rs`
- Environment variables:
  - `RUST_LOG` - Standard tracing filter
  - `ROCFORGE_LOG_LEVEL` - Simple level (error/warn/info/debug/trace)
  - `ROCFORGE_LOG_FORMAT` - "human" (colored console) or "json"
  - `ROCFORGE_LOG_FILE` - Optional file output (JSON format)
- Output: stderr/stdout or file

**Metrics:**
- Framework: `prometheus-client` 0.22
- Export: HTTP GET `/metrics` endpoint (Prometheus text format)
- Metrics defined in: `/src/metrics.rs`
  - Request counters (started, completed, failed, cancelled)
  - Token generation totals
  - Duration histograms (prefill, decode, total, TTFT)
  - Gauges (queue length, active requests, tokens/sec)
  - GPU retry metrics

**Tracing:**
- OpenTelemetry-style tracing: `/src/otel_traces.rs`
- In-memory `TraceStore` for span collection
- Export via `export_traces()` function
- Sampling rate configurable: `ROCMORGE_TRACE_SAMPLE_RATE` (default: 0.1)

## CI/CD & Deployment

**Hosting:**
- None configured (development project)

**CI Pipeline:**
- Makefile: `/Makefile`
  - `make build` - Release build with `rocm` feature
  - `make test` - Run tests with serial execution
  - `make check` - Cargo check
  - `make clippy` - Lint with warnings as errors
- No GitHub Actions, GitLab CI, or similar configured

## Environment Configuration

**Required env vars:**

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ROCMFORGE_GGUF` | Yes (for server) | None | Path to GGUF model |
| `ROCMFORGE_TOKENIZER` | No | Auto-detected | Path to tokenizer.json |
| `ROCMFORGE_MODELS` | No | `./models` | Model discovery directory |

**Build-time env vars:**

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ROCM_PATH` | No | `/opt/rocm` | ROCm installation directory |
| `HIPCC` | No | `$ROCM_PATH/bin/hipcc` | HIP compiler path |
| `ROCm_ARCH` | No | `gfx1100` | Target GPU architecture |

**Optional env vars:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROCFORGE_GPU_DEVICE` | `0` | GPU device number |
| `ROCFORGE_LOG_LEVEL` | `info` | Simple log level |
| `ROCFORGE_LOG_FORMAT` | `human` | Log output format |
| `ROCFORGE_LOG_FILE` | None | Log file path |
| `ROCMORGE_TRACE_SAMPLE_RATE` | `0.1` | OTEL trace sampling |
| `ROCFORGE_MAX_TRACES` | `1000` | Max in-memory traces |

**Secrets location:**
- No secrets management
- All configuration via environment variables

## Webhooks & Callbacks

**Incoming:**
- HTTP server endpoints (`/src/http/server.rs`):
  - `GET /` - Root/health check
  - `GET /health` - Health status
  - `POST /v1/completions` - OpenAI-compatible completions API
  - `GET /metrics` - Prometheus metrics export

**Outgoing:**
- None (no webhooks or callbacks initiated by the server)

## Native Library Linking

**HIP/ROCm Libraries (linked via build.rs):**

| Library | Purpose | Source |
|---------|---------|--------|
| `amdhip64` | Core HIP API | `/opt/rocm/lib/libamdhip64.so` |
| `hipblas` | BLAS operations | `/opt/rocm/lib/libhipblas.so` |
| `hiprtc` | HIP runtime compilation | `/opt/rocm/lib/libhiprtc.so` |

**FFI Declarations:**
- HIP FFI: `/src/backend/hip_backend/backend.rs` (lines 14-74)
  - ~30 functions: `hipInit`, `hipMalloc`, `hipMemcpy`, `hipModuleLoad`, `hipModuleLaunchKernel`, etc.
- hipBLAS FFI: `/src/backend/hip_blas.rs`
  - Functions: `hipblasCreate`, `hipblasSgemm`, `hipblasSaxpy`, etc.

## GGUF Format Integration

**Purpose:**
- Load models in GGUF (GPT-Generated Unified Format) from llama.cpp ecosystem

**Implementation:**
- Parser: `/src/loader/gguf.rs` (113KB, primary loader)
- Metadata: `/src/loader/metadata.rs` (20KB)
- Tensor types: `/src/loader/tensor_type.rs`
- Dequantization: `/src/loader/dequant.rs`
- Memory mapping: `/src/loader/mmap_loader.rs`

**Supported tensor types:**
- F32, F16
- Q4_0, Q4_K, Q5_K, Q6_K, Q8_0
- MXFP (micro-scaling formats)

**References:**
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/include/gguf.h
- GGML tensor spec: https://github.com/ggml-org/ggml/blob/master/include/ggml.h

---

*Integration audit: 2025-01-20*
