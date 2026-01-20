# External Integrations

**Analysis Date:** 2025-01-20

## APIs & External Services

**HTTP Client (for CLI/server communication):**
- reqwest - HTTP client library
  - Used by: `src/bin/rocmforge_cli.rs`
  - Purpose: CLI communicates with ROCmForge HTTP server
  - No external API calls to third-party services

**Server-Sent Events:**
- reqwest-eventsource - SSE client library
  - Used by: `src/bin/rocmforge_cli.rs`
  - Purpose: Stream tokens from inference server
  - No external service dependencies

**Hugging Face Tokenizers:**
- tokenizers crate - Hugging Face tokenizer format
  - Used by: `src/tokenizer.rs`
  - Purpose: Load and process tokenizer.json files
  - No API calls - purely local file parsing
  - Optional HTTP feature for remote tokenizer loading (not actively used)

## Data Storage

**Databases:**
- SQLite (via sqlitegraph crate)
  - Feature-gated: `context` feature only
  - Used by: `src/context/graph_context.rs`
  - Connection: Path to SQLite database file or in-memory
  - Purpose: Graph-based context storage with HNSW vector indexing
  - ORM/client: sqlitegraph provides graph entity and edge storage

**Model Storage:**
- GGUF format files - Local filesystem
  - Used by: `src/loader/gguf.rs`
  - Location: Configured via `ROCMFORGE_GGUF` env var
  - Contains: Model weights, metadata, optional embedded tokenizer

**Tokenizer Files:**
- tokenizer.json - Hugging Face format
  - Used by: `src/tokenizer.rs`
  - Location: Same directory as GGUF or custom path via `ROCMFORGE_TOKENIZER`
  - Fallback: Embedded tokenizer JSON from GGUF file

**File Storage:**
- Local filesystem only
- No S3, cloud storage, or external file services

**Caching:**
- In-memory only
- Tokenizer cache via `src/models.rs`: `cached_embedded_tokenizer()`
- No external caching service (Redis, Memcached)

## Authentication & Identity

**Auth Provider:**
- None - Custom/local only

**Implementation:**
- No authentication in HTTP server
- No API keys, OAuth, or external auth providers
- Server binds to localhost by default (`127.0.0.1:8080`)
- CORS support via tower-http but no auth middleware

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking service

**Logs:**
- tracing/tracing-subscriber - Structured logging
  - Formats: Human-readable console or JSON
  - Output: stdout/stderr or file via `ROCFORGE_LOG_FILE`
  - Levels: error, warn, info, debug, trace
  - No external log aggregation

**Metrics:**
- prometheus-client - Prometheus metrics
  - Export endpoint: `/metrics` on HTTP server
  - Format: Prometheus text format
  - Metrics collected: request counts, durations, token throughput, queue length, TTFT
  - No Prometheus server included - export only

**Tracing:**
- OpenTelemetry-compatible (custom implementation)
  - Module: `src/otel_traces.rs`
  - Export endpoint: `/traces` on HTTP server
  - Format: OTLP JSON
  - Sampling: Configurable via `ROCMORGE_TRACE_SAMPLE_RATE` (default: 0.1)
  - No external OTEL collector - local storage only

**Profiling:**
- rocprof (ROCm Profiler) integration
  - Module: `src/profiling/rocprof_integration.rs`
  - Feature: Development profiling support

## CI/CD & Deployment

**Hosting:**
- None - Self-hosted only

**CI Pipeline:**
- None detected - No GitHub Actions, GitLab CI, or other CI config files

**Deployment:**
- Manual deployment via `cargo build --release`
- No containerization (Dockerfile) detected
- No deployment automation

## Environment Configuration

**Required env vars:**

**For builds with ROCm:**
- `ROCM_PATH` - ROCm installation (default: `/opt/rocm`)
- `HIPCC` - HIP compiler (default: `$ROCM_PATH/bin/hipcc`)
- `ROCm_ARCH` - GPU architecture (default: `gfx1100`)

**For runtime:**
- `ROCMFORGE_GGUF` - Path to model file (required for inference)
- `ROCMFORGE_TOKENIZER` - Path to tokenizer.json (optional, auto-detected)
- `ROCMFORGE_GPU_DEVICE` - GPU number (default: 0)

**Optional env vars:**
- `RUST_LOG` - Tracing filter (default: `info`)
- `ROCFORGE_LOG_FORMAT` - `human` or `json`
- `ROCFORGE_LOG_FILE` - Log file path
- `ROCMFORGE_MODELS` - Model discovery directory
- `ROCMORGE_TRACE_SAMPLE_RATE` - Trace sampling (default: 0.1)
- `ROCMFORGE_MAX_TRACES` - Max trace count (default: 1000)
- GPU tuning: `ROCFORGE_BLOCK_SIZE`, `ROCFORGE_WARP_SIZE`, `ROCFORGE_USE_LDS`, `ROCFORGE_LDS_SIZE`, `ROCFORGE_TILE_K`, `ROCFORGE_TILE_N`
- Testing: `ROCFORGE_TEST_MODEL`

**Secrets location:**
- No secrets storage - project has no external API keys or secrets

## Webhooks & Callbacks

**Incoming:**
- HTTP API endpoints (server, not webhooks):
  - `POST /v1/completions` - Text generation (OpenAI-compatible)
  - `GET /metrics` - Prometheus metrics
  - `GET /traces` - OpenTelemetry traces
  - `GET /health` - Health check

**Outgoing:**
- None - No external webhooks or callbacks

## GPU Integration

**ROCm/HIP FFI Bindings:**
- Direct `extern "C"` linkage in `src/backend/hip_backend/backend.rs`:
  - `#[link(name = "amdhip64")]` - HIP runtime
  - `#[link(name = "hipblas")]` - HIP BLAS library
  - Functions: `hipInit`, `hipMalloc`, `hipMemcpy`, `hipModuleLoad`, `hipLaunchKernel`, etc.
- Compiled via `build.rs` linking:
  - `rustc-link-lib=dylib=amdhip64`
  - `rustc-link-lib=dylib=hipblas`
  - `rustc-link-lib=dylib=hiprtc`

**HIP Kernels:**
- Source location: `kernels/*.hip` (31 kernel files)
- Compiled to: `.hsaco` (HSACO = AMD GPU code object)
- Kernels loaded at runtime via `hipModuleLoad`

---

*Integration audit: 2025-01-20*
