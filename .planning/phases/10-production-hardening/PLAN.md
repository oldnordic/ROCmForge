# Phase 10: Production Hardening - Execution Plan

**Phase:** 10
**Mode:** standard
**Created:** 2026-01-18

---

## Frontmatter

```yaml
wave: 4
depends_on: [9]
files_modified:
  - src/engine.rs
  - src/scheduler/scheduler.rs
  - src/kv_cache/kv_cache.rs
  - src/ggml/mod.rs
  - src/backend/hip_backend/backend.rs
  - src/loader/mod.rs
  - src/http/server.rs
  - src/bin/rocmforge_cli.rs
  - src/lib.rs
  - .env.example
  - docs/USER_GUIDE.md
  - docs/CLI_REFERENCE.md
  - docs/API_DOCUMENTATION.md
  - docs/DEPLOYMENT.md
autonomous: true
```

---

## Phase Goal

Error handling, logging, monitoring, and documentation for production use.

**Source:** ROADMAP.md Phase 10

> Error handling with recovery strategies, logging infrastructure with tracing, monitoring endpoints and metrics collection, user and API documentation.

---

## Current State Analysis

### Existing Error Types (Discovered)

| Module | Error Type | Status |
|--------|-----------|--------|
| `src/engine.rs` | `EngineError` (5 variants) | Complete |
| `src/scheduler/scheduler.rs` | `SchedulerError` (4 variants) | Complete |
| `src/kv_cache/kv_cache.rs` | `KvCacheError` (6 variants) | Complete |
| `src/ggml/mod.rs` | `GgmlError` (4 variants) | Complete |
| `src/http/server.rs` | `ServerError` (4 variants) | Complete |
| `src/backend/hip_backend/backend.rs` | `HipError` (discovered via context) | Needs verification |

### unwrap() Usage (Grep Results)

- **Total unwrap() calls**: 624 across 83 files
- **eprintln! calls**: 1365 across 146 files
- **Test unwrap() calls**: 463 total, reduced to 192 (58.5% reduction) in Phase 2

### Existing Logging

- **tracing** dependency present in Cargo.toml
- **tracing-subscriber** with env-filter feature enabled
- Mixed usage: `tracing::info!/error!/warn!` and `eprintln!`
- No centralized subscriber initialization

### Monitoring Endpoints

- `/health` endpoint exists (basic: status, service, version)
- No `/ready` (readiness probe)
- No `/metrics` (Prometheus format)
- No tracing/metrics infrastructure

### Documentation

- README.md exists (comprehensive but outdated status references)
- No user guide
- No CLI reference
- No API documentation beyond rustdoc
- No deployment guide

---

## Wave Structure

### Wave 1: Error Handling Foundation
**Focus:** Replace unwrap() in production paths, add missing error types

**Estimated Time:** 3-4 hours

### Wave 2: Logging Infrastructure
**Focus:** Complete eprintln! to tracing migration, add subscriber initialization

**Estimated Time:** 2-3 hours

### Wave 3: Monitoring Endpoints
**Focus:** Health checks, metrics, readiness probes

**Estimated Time:** 2-3 hours

### Wave 4: Documentation
**Focus:** User guide, CLI reference, API docs, deployment guide

**Estimated Time:** 4-5 hours

---

## Tasks

### Wave 1: Error Handling Foundation

#### 10-01: Create unified error module

**Description:** Create `src/error.rs` with consolidated error types and a top-level `RocmForgeError` that wraps all domain-specific errors.

**Files:**
- `src/error.rs` (new)
- `src/lib.rs`

**Actions:**
1. Create `RocmForgeError` enum with variants for each domain
2. Implement `From` conversions for all existing error types
3. Add error categories: Fatal, Recoverable, User, Internal
4. Add `is_recoverable()`, `is_user_error()`, `category()` methods
5. Export in lib.rs

**Acceptance Criteria:**
- [ ] `src/error.rs` compiles with all error variants
- [ ] All domain errors convert to `RocmForgeError`
- [ ] Category methods return correct classifications
- [ ] Unit tests for error categorization

**Dependencies:** None
**Estimated Time:** 60 minutes

---

#### 10-02: Replace unwrap() in engine.rs

**Description:** Identify and replace unwrap() calls in production code paths (not tests) with proper error handling.

**Files:**
- `src/engine.rs`

**Actions:**
1. Search for unwrap() calls in engine.rs
2. Categorize: test code vs production paths
3. Replace production unwrap() with `?` or `.map_err()`
4. Add context using `.context()` where helpful
5. Verify compilation

**Acceptance Criteria:**
- [ ] All unwrap() in production paths replaced
- [ ] Compilation succeeds
- [ ] Tests pass
- [ ] No functional changes (only error handling)

**Dependencies:** 10-01 (for unified error type)
**Estimated Time:** 45 minutes

---

#### 10-03: Replace unwrap() in scheduler and kv_cache

**Description:** Replace unwrap() calls in scheduler and kv_cache modules.

**Files:**
- `src/scheduler/scheduler.rs`
- `src/kv_cache/kv_cache.rs`
- `src/kv_cache/block_allocator.rs`
- `src/kv_cache/page_table.rs`

**Actions:**
1. Identify unwrap() in production paths
2. Replace with proper error handling
3. Handle lock poisoning gracefully
4. Add tests for error paths

**Acceptance Criteria:**
- [ ] Production unwrap() replaced
- [ ] Lock poisoning handled with `RocmForgeError::Internal`
- [ ] Tests pass
- [ ] No unwrap() in hot paths

**Dependencies:** 10-01
**Estimated Time:** 45 minutes

---

#### 10-04: Replace unwrap() in loader and backend modules

**Description:** Replace unwrap() calls in GGUF loader and HIP backend modules.

**Files:**
- `src/loader/gguf.rs`
- `src/loader/mmap.rs`
- `src/backend/hip_backend/backend.rs`
- `src/backend/gpu_executor.rs`

**Actions:**
1. Identify unwrap() in FFI and I/O paths
2. Replace with `HipError` or `GgufError`
3. Add context for FFI failures
4. Document unrecoverable errors

**Acceptance Criteria:**
- [ ] FFI unwrap() replaced with HipError
- [ ] I/O unwrap() replaced with proper errors
- [ ] HIP error codes translated to error types
- [ ] Tests pass

**Dependencies:** 10-01
**Estimated Time:** 60 minutes

---

### Wave 2: Logging Infrastructure

#### 10-05: Create tracing subscriber initialization

**Description:** Create a centralized logging setup module with configurable levels and formats.

**Files:**
- `src/logging.rs` (new)
- `src/lib.rs`

**Actions:**
1. Create `init_tracing()` function
2. Support `ROCFORGE_LOG_LEVEL` environment variable
3. Implement both JSON and human-readable formats
4. Add `ROCFORGE_LOG_FORMAT` (json/human) option
5. Filter by crate module
6. Initialize in lib.rs lazy static or main

**Acceptance Criteria:**
- [ ] `init_tracing()` configures subscriber
- [ ] `ROCFORGE_LOG_LEVEL` sets default level (warn+error default)
- [ ] JSON format for machine parsing
- [ ] Human-readable with colors for console
- [ ] Module-level filtering works

**Dependencies:** None
**Estimated Time:** 45 minutes

---

#### 10-06: Replace eprintln! with tracing in engine

**Description:** Replace debug eprintln! statements with structured tracing macros.

**Files:**
- `src/engine.rs`

**Actions:**
1. Identify all eprintln! calls (estimated ~14)
2. Replace with tracing::info!/debug!/warn!/error!
3. Add structured fields (request_id, phase, duration)
4. Use span! for request lifecycle
5. Remove unnecessary debug output

**Acceptance Criteria:**
- [ ] All eprintln! replaced with tracing macros
- [ ] Structured fields added for key operations
- [ ] Request spans cover inference lifecycle
- [ ] Compilation succeeds
- [ ] Tests pass

**Dependencies:** 10-05
**Estimated Time:** 30 minutes

---

#### 10-07: Replace eprintln! in remaining modules

**Description:** Replace eprintln! in http, loader, and CLI modules.

**Files:**
- `src/http/server.rs`
- `src/loader/gguf.rs`
- `src/bin/rocmforge_cli.rs`
- All other src/ modules

**Actions:**
1. Audit eprintln! across all modules
2. Replace with appropriate tracing level
3. Add instrumentation for HTTP requests
4. Add instrumentation for model loading
5. Add CLI-specific logging (user-friendly)

**Acceptance Criteria:**
- [ ] eprintln! reduced to <5% (only forced output for users)
- [ ] HTTP requests logged with method, path, status
- [ ] Model loading logged with progress
- [ ] CLI uses tracing for errors, print for output

**Dependencies:** 10-05
**Estimated Time:** 60 minutes

---

### Wave 3: Monitoring Endpoints

#### 10-08: Add readiness probe endpoint

**Description:** Implement `/ready` endpoint that checks if the engine can accept requests.

**Files:**
- `src/http/server.rs`

**Actions:**
1. Add `ready_handler()` function
2. Check: engine initialized, model loaded, below capacity
3. Return 503 if not ready with reason
4. Add to router

**Acceptance Criteria:**
- [ ] `/ready` returns 200 when ready
- [ ] `/ready` returns 503 when not ready
- [ ] Response includes: ready status, reason if not ready
- [ ] Test coverage

**Dependencies:** None
**Estimated Time:** 30 minutes

---

#### 10-09: Create metrics collection module

**Description:** Implement Prometheus-compatible metrics collection.

**Files:**
- `src/metrics.rs` (new)
- `Cargo.toml`

**Actions:**
1. Add metrics types: Counter, Gauge, Histogram
2. Define inference metrics:
   - `rocmforge_requests_total` (labels: status)
   - `rocmforge_tokens_generated_total`
   - `rocmforge_inference_duration_seconds` (labels: phase)
   - `rocmforge_queue_length`
   - `rocmforge_active_requests`
3. Thread-safe metric storage
4. Export helpers for instrumentation

**Acceptance Criteria:**
- [ ] Metric types implemented
- [ ] Thread-safe updates
- [ ] Prometheus text format export
- [ ] Unit tests for metric updates

**Dependencies:** None
**Estimated Time:** 60 minutes

---

#### 10-10: Add /metrics endpoint

**Description:** Expose metrics in Prometheus format at `/metrics` endpoint.

**Files:**
- `src/http/server.rs`
- `src/metrics.rs`

**Actions:**
1. Create `metrics_handler()` function
2. Export all metrics in Prometheus text format
3. Add to router
4. Instrument engine and scheduler with metrics
5. Add tests

**Acceptance Criteria:**
- [ ] `/metrics` returns Prometheus text format
- [ ] All defined metrics exported
- [ ] Engine metrics updated on inference
- [ ] Scheduler metrics updated on queue changes
- [ ] Tests verify format

**Dependencies:** 10-09
**Estimated Time:** 45 minutes

---

#### 10-11: Enhance /health endpoint

**Description:** Expand `/health` endpoint with detailed health information.

**Files:**
- `src/http/server.rs`

**Actions:**
1. Add checks: GPU availability, memory usage, model status
2. Return detailed health object:
   ```json
   {
     "status": "healthy",
     "service": "rocmforge",
     "version": "0.1.0",
     "checks": {
       "gpu": "ok",
       "model": "loaded",
       "memory": {"used": X, "total": Y}
     }
   }
   ```
3. Add tests

**Acceptance Criteria:**
- [ ] `/health` returns all check results
- [ ] GPU status included
- [ ] Memory stats included
- [ ] Test coverage

**Dependencies:** None
**Estimated Time:** 30 minutes

---

### Wave 4: Documentation

#### 10-12: Create .env.example

**Description:** Add example environment configuration file with all supported variables.

**Files:**
- `.env.example` (new)

**Actions:**
1. Document all environment variables:
   - ROCMFORGE_GGUF
   - ROCMFORGE_TOKENIZER
   - ROCMFORGE_MODELS
   - ROCMFORGE_LOG_LEVEL
   - ROCMFORGE_LOG_FORMAT
   - ROCMFORGE_GPU_DEVICE
2. Add descriptions and defaults
3. Include example values

**Acceptance Criteria:**
- [ ] All env vars documented
- [ ] Default values specified
- [ ] Helpful descriptions
- [ ] Valid .env format

**Dependencies:** None
**Estimated Time:** 20 minutes

---

#### 10-13: Write user guide

**Description:** Create comprehensive user guide for installation and usage.

**Files:**
- `docs/USER_GUIDE.md` (new)

**Actions:**
1. Document prerequisites (ROCm, Rust version)
2. Installation instructions (from source)
3. Quick start with example model
4. Common use cases (generate, serve, inspect)
5. Troubleshooting section
6. Include example commands and outputs

**Acceptance Criteria:**
- [ ] Prerequisites clearly listed
- [ ] Step-by-step installation
- [ ] Working examples for all commands
- [ ] Troubleshooting covers common issues
- [ ] Tested by following instructions

**Dependencies:** 10-12
**Estimated Time:** 90 minutes

---

#### 10-14: Write CLI reference

**Description:** Document all CLI commands, flags, and options.

**Files:**
- `docs/CLI_REFERENCE.md` (new)

**Actions:**
1. Document all subcommands (serve, generate, status, cancel, models)
2. List all flags and options
3. Include examples for each command
4. Document exit codes
5. Add troubleshooting section

**Acceptance Criteria:**
- [ ] All commands documented
- [ ] All flags listed with descriptions
- [ ] Examples for each command
- [ ] Exit codes documented
- [ ] Matches actual CLI behavior

**Dependencies:** 10-13
**Estimated Time:** 60 minutes

---

#### 10-15: Write API documentation

**Description:** Document HTTP API endpoints and request/response formats.

**Files:**
- `docs/API_DOCUMENTATION.md` (new)

**Actions:**
1. Document all HTTP endpoints:
   - POST /generate
   - POST /generate/stream
   - GET /status/:request_id
   - POST /cancel/:request_id
   - GET /models
   - GET /health
   - GET /ready
   - GET /metrics
2. Include request/response examples
3. Document error responses
4. Add cURL examples

**Acceptance Criteria:**
- [ ] All endpoints documented
- [ ] Request/response schemas provided
- [ ] Error codes documented
- [ ] cURL examples for each endpoint
- [ ] SSE streaming documented

**Dependencies:** 10-11
**Estimated Time:** 60 minutes

---

#### 10-16: Write deployment guide

**Description:** Create guide for deploying ROCmForge in production-like environments.

**Files:**
- `docs/DEPLOYMENT.md` (new)

**Actions:**
1. Document deployment options (binary, Docker)
2. Configuration management
3. Service setup (systemd)
4. Reverse proxy configuration
5. Monitoring and observability setup
6. Security considerations
7. Performance tuning

**Acceptance Criteria:**
- [ ] Binary deployment documented
- [ ] systemd service example included
- [ ] Reverse proxy config provided
- [ ] Monitoring endpoints documented
- [ ] Security best practices listed

**Dependencies:** 10-15, 10-13
**Estimated Time:** 90 minutes

---

## Must Haves (Verification)

### Error Handling
- [ ] Production paths have <10 unwrap() calls (excluding tests)
- [ ] Unified error type with categorization
- [ ] Graceful degradation for recoverable errors
- [ ] User-friendly error messages for user errors

### Logging
- [ ] eprintln! replaced with tracing in production code
- [ ] Structured logging with request context
- [ ] Configurable log level via environment variable
- [ ] JSON format option for log aggregation

### Monitoring
- [ ] `/health` endpoint with detailed checks
- [ ] `/ready` endpoint for readiness probes
- [ ] `/metrics` endpoint with Prometheus format
- [ ] Core metrics: requests, tokens, duration, queue

### Documentation
- [ ] User guide with working examples
- [ ] CLI reference covering all commands
- [ ] API documentation for all endpoints
- [ ] Deployment guide with systemd example
- [ ] .env.example with all variables

---

## Phase Completion Checklist

- [ ] All tasks completed
- [ ] All tests passing
- [ ] Documentation reviewed
- [ ] unwrap() count in production paths verified <10
- [ ] Monitoring endpoints tested manually
- [ ] Log output verified in both formats
- [ ] SUMMARY.md created

---

**Total Estimated Time:** 13-16 hours

**Next Phase:** None (Phase 10 is final planned phase)
