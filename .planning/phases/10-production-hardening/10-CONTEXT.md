# Phase 10: Production Hardening - Context

**Gathered:** 2026-01-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Production hardening for the ROCmForge LLM inference engine. This phase focuses on error handling, logging, monitoring, and documentation required for production deployment.

**Scope:**
- Error handling with recovery strategies
- Logging infrastructure with tracing
- Monitoring endpoints and metrics collection
- User and API documentation

**Out of scope:**
- New features or capabilities (belong in other phases)
- Performance optimization (Phase 9)
- New model architecture support (Phase 8)

</domain>

<decisions>
## Implementation Decisions

### Error Handling Strategy
- **Approach:** Best practices mix - domain-specific error types + anyhow::Context + thiserror! for user-facing errors
- **Recovery strategy:** Graceful degradation (CPU fallback if GPU fails, retry transient errors)
- **Fatal errors:** You decide based on severity (data corruption, security issues = fatal)
- **User error presentation:** You decide based on error type (technical vs non-technical users)

### Logging Approach
- **Framework:** tracing crate with subscriber modules
- **Default level:** warn + error (runtime configurable via env var)
- **Output format:** Both JSON and human-readable formats
  - JSON for machine parsing (logs sent to file)
  - Human-readable with colors for console output
- **Structured logging:** Use tracing::info!/error!/warn! macros

### Monitoring Surface
- **Endpoints:** Full observability (health, metrics, traces)
- **Metrics focus:** Inference metrics (tokens/sec, queue length, TTFT)
- **Health checks:** /health (liveness), /ready (readiness probe), /metrics (Prometheus format)
- **Traces:** OpenTelemetry integration for distributed tracing (future)

### Claude's Discretion
- Whether to add HTTP server for monitoring endpoints vs CLI-only
- Specific metric names and labels (follow Prometheus naming conventions)
- Log rotation strategy (size-based or time-based)
- Tracing exporter configuration (OTLP, Jaeger, or custom)

### Documentation Scope
- **Priority order:** User guide > CLI reference > API documentation > Deployment guide
- **User guide:** Installation, basic usage, examples, troubleshooting
- **CLI reference:** All commands, flags, options, exit codes
- **API documentation:** Generated docs from rustdoc for library users
- **Deployment guide:** Setup, configuration (including .env.example), operations

</decisions>

<specifics>
## Specific Ideas

- Error types should mirror existing crate structure (backend, loader, model domains)
- Tracing should integrate with existing tokio runtime (async context propagation)
- Health check endpoints should be simple (no external dependencies for liveness)
- Metrics format should be Prometheus-compatible for common monitoring stacks
- Documentation should include realistic examples based on actual ROCmForge usage

</specifics>

<deferred>
## Deferred Ideas

- **sqlitegraph for KV-cache augmentation** (context engine, relevance graph, semantic index) - Noted from user discussion, potential future enhancement beyond Phase 10

</deferred>

---

*Phase: 10-production-hardening*
*Context gathered: 2026-01-18*
