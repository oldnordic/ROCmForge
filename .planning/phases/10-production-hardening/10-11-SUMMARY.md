# Task 10-11: Create Tracing Endpoint - Summary

**Task:** Create a /traces endpoint for OpenTelemetry trace export
**Date:** 2026-01-19
**Status:** Complete

## Accomplishments

### 1. Created OpenTelemetry-Compatible Tracing Module

**File:** `/home/feanor/Projects/ROCmForge/src/otel_traces.rs`

Implemented a comprehensive tracing module with OpenTelemetry OTLP JSON format support:

- **Trace Storage**: `TraceStore` with FIFO eviction when max traces exceeded
- **Span Types**: `Span`, `SpanEvent`, `SpanKind`, `SpanStatus`
- **OTLP Structures**: `Resource`, `ResourceSpans`, `Scope`, `ScopeSpans`, `TraceExport`
- **Attribute System**: `Attribute` and `AttributeValue` supporting String, Bool, Int, Double, and arrays
- **Trace Sampling**: Configurable sample rate (default 10%)
- **ID Generation**: Random trace ID (16 bytes) and span ID (8 bytes) generation

### 2. HTTP Endpoint

**File:** `/home/feanor/Projects/ROCmForge/src/http/server.rs`

Added `/traces` GET endpoint with:

- **Query Parameters**:
  - `limit`: Maximum number of traces to return
  - `clear`: Clear traces after returning them
- **Response Format**: OpenTelemetry OTLP JSON
- **Route Integration**: Added to router alongside `/health`, `/ready`, `/models`

### 3. Configuration

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCMORGE_TRACE_SAMPLE_RATE` | Trace sampling rate (0.0-1.0) | 0.1 (10%) |
| `ROCMFORGE_MAX_TRACES` | Maximum traces to store in memory | 1000 |

### 4. API Functions

Public API exported in `lib.rs`:

```rust
// Store management
pub fn init_trace_store(config: TraceConfig)
pub fn get_trace_store() -> Arc<Mutex<TraceStore>>
pub fn record_span(span: Span) -> bool
pub fn export_traces() -> TraceExport
pub fn clear_traces()
pub fn trace_count() -> usize

// Builder for inference spans
pub struct InferenceSpanBuilder
```

### 5. Tests

**23/23 tests passing:**

#### Tracing Module Tests (17 tests)
- `test_span_creation` - Span basic creation
- `test_span_builder` - Span builder methods
- `test_span_duration` - Duration calculation
- `test_span_status_from_bool` - Status conversion
- `test_attribute_values` - All attribute types
- `test_trace_store_empty` - Empty store behavior
- `test_trace_store_add` - Adding spans
- `test_trace_store_sampling` - Sample rate enforcement
- `test_trace_store_max_traces` - FIFO eviction
- `test_trace_export` - OTLP format export
- `test_inference_span_builder` - Inference span creation
- `test_global_trace_store` - Global store access
- `test_empty_export` - Export with no traces
- `test_resource_creation` - Resource attributes
- `test_scope_spans_creation` - Scope spans creation
- `test_trace_id_generation` - Trace ID format
- `test_span_id_generation` - Span ID format

#### HTTP Endpoint Tests (6 tests)
- `test_traces_handler_empty` - Empty traces response
- `test_traces_handler_with_sample_data` - Sample data export
- `test_traces_handler_with_limit` - Limit parameter
- `test_traces_handler_with_clear` - Clear parameter
- `test_traces_query_params` - Query parameter parsing
- `test_traces_query_default` - Default query parameters

### 6. Example OTLP JSON Format

```json
{
  "resource_spans": [{
    "resource": {
      "attributes": {
        "service.name": "rocmforge",
        "service.version": "0.1.0"
      }
    },
    "scope_spans": [{
      "scope": {
        "name": "rocmforge",
        "version": "0.1.0"
      },
      "spans": [{
        "trace_id": "0123456789abcdef0123456789abcdef",
        "span_id": "0123456789abcdef",
        "name": "inference",
        "kind": "Server",
        "start_time_unix_nano": 1234567890000000,
        "end_time_unix_nano": 1234567900000000,
        "attributes": [
          {"key": "request.id", "Int": 123},
          {"key": "prompt.length", "Int": 50}
        ],
        "status": "Ok"
      }]
    }]
  }]
}
```

## Files Modified

- `/home/feanor/Projects/ROCmForge/src/otel_traces.rs` - New tracing module
- `/home/feanor/Projects/ROCmForge/src/http/server.rs` - Added /traces endpoint
- `/home/feanor/Projects/ROCmForge/src/lib.rs` - Added otel_traces module and exports
- `/home/feanor/Projects/ROCmForge/Cargo.toml` - Added `hex` dependency

## Technical Decisions

1. **Module Naming**: Used `otel_traces` instead of `tracing` to avoid conflict with the `tracing` logging crate.

2. **Default Sampling Rate**: Set to 10% to balance observability with memory usage. Can be increased via environment variable.

3. **In-Memory Storage**: Traces stored in memory with FIFO eviction. For production, consider integrating with OTLP exporters (Jaeger, Tempo, etc.).

4. **ID Generation**: Random 128-bit trace IDs and 64-bit span IDs following OpenTelemetry specification.

5. **Thread Safety**: Uses `Arc<Mutex<>>` for global trace store access.

## Acceptance Criteria

- [x] `/traces` endpoint created
- [x] OpenTelemetry format verified (OTLP JSON)
- [x] Sample traces exported
- [x] Trace sampling configured (default 10%, configurable)
- [x] Tests passing (23/23 tests)
- [x] Compiles without errors (507 total library tests passing)

## Usage Examples

### Get all traces:
```bash
curl http://localhost:8080/traces
```

### Get limited traces:
```bash
curl http://localhost:8080/traces?limit=10
```

### Get and clear traces:
```bash
curl http://localhost:8080/traces?clear=true
```

### Recording a span programmatically:
```rust
use rocmforge::{Span, InferenceSpanBuilder, AttributeValue};

// Using builder
let span = InferenceSpanBuilder::new("inference")
    .with_request_id(123)
    .with_prompt_length(50)
    .with_max_tokens(100)
    .build_and_record();

// Using Span directly
let span = Span::new("custom_operation")
    .with_attribute("operation", AttributeValue::String("compute".to_string()))
    .with_status(SpanStatus::Ok);
rocmforge::record_span(span);
```

## Next Steps

The tracing endpoint is complete and functional. Future enhancements could include:

1. **OTLP Export**: Integration with OpenTelemetry collectors (Jaeger, Tempo, etc.)
2. **Automatic Instrumentation**: Auto-generate spans for inference requests
3. **Distributed Tracing**: Propagate trace context across service boundaries
4. **Span Export to File**: Persist traces to disk for analysis

## Dependencies

- **hex**: 0.4 - For encoding trace/span IDs as hex strings
- **rand**: 0.8 - Already present, used for ID generation
- **serde/serde_json**: Already present, used for JSON serialization
