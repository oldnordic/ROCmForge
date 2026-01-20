# Codebase Concerns

**Analysis Date:** 2026-01-20

## Tech Debt

**Deprecated GPU Copy Methods:**
- Issue: Multiple deprecated methods (`copy_to_host`, `to_host_vec`) still in use across codebase
- Files: `src/backend/hip_backend/backend.rs:727`, `src/model/execution_plan/execution_plan_src.rs:5`, `src/attention/paged_kernel.rs:6`
- Impact: Deprecated methods lack clear intent and may be removed in future. Module-level `#![allow(deprecated)]` suppresses migration warnings.
- Fix approach: Replace all usages with `copy_from_device_safe()` which has clearer semantics. Remove module-level `#![allow(deprecated)]` to force migration.

**Large Source Files:**
- Issue: Several files exceed 1000 lines, indicating modularity debt
- Files:
  - `src/model/execution_plan/execution_plan_src.rs` (4,213 lines)
  - `src/backend/hip_backend/backend.rs` (4,039 lines)
  - `src/loader/gguf.rs` (2,846 lines)
  - `src/kv_cache/kv_cache.rs` (2,094 lines)
- Impact: Difficult to navigate, test, and maintain. High cognitive load for modifications.
- Fix approach: Gradual extraction into smaller modules. Execution plan already started (`execution_plan_src.rs` indicates modularization in progress).

**Dead Code Accumulation:**
- Issue: Extensive use of `#[allow(dead_code)]` to suppress warnings (~50+ occurrences)
- Files: Throughout codebase - `src/sampler/gpu.rs:6`, `src/attention/kernels.rs:15-90`, `src/loader/gguf.rs:34,965,1844`
- Impact: Code bloat, unclear what's actually used, hinders refactoring.
- Fix approach: Audit dead_code allowances, remove truly dead code, document reserved code with issue trackers.

**HSACO Kernel Loading Without Validation:**
- Issue: GPU kernels loaded from environment variables (`SAMPLING_UTILS_HSACO`, `TEMPERATURE_SCALE_HSACO`, etc.) without path validation
- Files: `src/sampler/gpu.rs:99-160`
- Impact: Runtime failures if HSACO files missing. Fallback to CPU may hide kernel bugs.
- Fix approach: Validate HSACO paths at startup, fail fast if kernels not found.

## Known Bugs

**GGUF Metadata vocab_size Returns Zero:**
- Symptoms: `metadata.vocab_size` returns 0 instead of actual vocabulary size
- Files: `src/loader/metadata.rs:17,36`, `src/loader/gguf.rs:1280-1297`
- Trigger: Loading GGUF models where vocab_size must be inferred from tensor shapes
- Workaround: Code has fallback `infer_vocab_size_from_tensors()` but inference may fail if `hidden_size` is also 0
- Root cause: `update_from_kv()` in metadata.rs doesn't handle all model architectures (only `glm`, `qwen2`, `llama`, `mistral`, `yi`, `mixtral`). Architectures using different key names return 0.
- Impact: 3 integration tests fail - `test_token_embedding_lookup_f32`, `test_batch_embedding_lookup`, `test_lm_head_matmul_correctness`

**Test Compilation Errors in q_dequant_tests:**
- Symptoms: Type mismatch errors (`usize` vs `u8`), unresolved imports
- Files: `tests/q_dequant_tests.rs:636,663,878,952`
- Trigger: `cargo test --test q_dequant_tests`
- Specific errors:
  - Line 636: `dequantize_q4_0_kernel_cached` not found (function is `#[cfg(feature = "rocm")]` but test module not gated)
  - Lines 663, 878, 952: Array index operations produce `usize` but array expects `u8`
- Impact: These tests cannot run, blocking validation of Q4_1, Q5_0, Q5_1 dequantization

**Parallel Result Lock Poisoning Silently Ignored:**
- Symptoms: RwLock poisoning errors caught and discarded
- Files: `src/loader/gguf.rs:1069`, `src/loader/dequant.rs:60,121`
- Trigger: Thread panic during parallel dequantization
- Workaround: Function returns early with no data
- Impact: Silent failures - no indication that dequantization failed

## Security Considerations

**Unsafe FFI Blocks Without Bounds Checking:**
- Risk: Raw pointer access in HIP FFI could cause memory corruption
- Files: `src/backend/hip_backend/backend.rs:16-74` (FFI declarations), `src/backend/hip_blas.rs:85,121,136,152`
- Current mitigation: HipBuffer and HipBackend wrappers provide safe interfaces
- Recommendations:
  - Audit all unsafe FFI call sites
  - Consider using `bindgen` to generate safer bindings
  - Add fuzzing for FFI boundary

**Global Mutable Backend State:**
- Risk: `GLOBAL_BACKEND: Mutex<Option<Arc<HipBackend>>>` is mutable singleton
- Files: `src/backend/hip_backend/backend.rs:1212`
- Current mitigation: Mutex protects access, but design allows runtime replacement
- Recommendations: Consider immutable backend passed through context, or Arc cloning pattern

**Unvalidated Model File Loading:**
- Risk: GGUF files loaded without size limits or validation
- Files: `src/loader/gguf.rs`, `src/loader/mmap.rs`
- Current mitigation: Memory-mapped files with bounds checking on reads
- Recommendations:
  - Add maximum model size limits
  - Validate tensor counts before allocation
  - Add timeout for maliciously-crafted GGUF files

**Lock Poison Handling Converts to String:**
- Risk: Poison error information lost in string conversion
- Files: `src/backend/hip_backend/backend.rs:194`, `src/kv_cache/kv_cache.rs:28`
- Current mitigation: Converted to generic error type
- Recommendations: Preserve poison error type for proper handling

## Performance Bottlenecks

**Synchronous GPU Operations in Async Context:**
- Problem: `hipDeviceSynchronize()` blocks async tasks
- Files: `src/backend/hip_backend/backend.rs:758`, `src/profiling/kernel_timer.rs:404-491`
- Cause: GPU operations require synchronization before reading results
- Improvement path: Use async-aware GPU task scheduling, batch operations to minimize sync points

**Lock Contention on GPU Cache:**
- Problem: `Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>` can become bottleneck
- Files: `src/loader/gguf.rs:546,601,622,676,782,953,1249`
- Cause: Multiple threads accessing same cache during model loading
- Improvement path: Consider sharded cache, lock-free structures, or per-thread caches with deduplication

**Thread Sleep in Hot Path:**
- Problem: `std::thread::sleep()` used in retry logic and profiling
- Files: `src/backend/hip_backend/backend.rs:1651`, `src/profiling/kernel_timer.rs:404-491`
- Cause: Waiting for GPU operations or retry delays
- Improvement path: Use condition variables, event-based synchronization, proper async/await

**String-Allocation Heavy Metadata Parsing:**
- Problem: `update_from_kv()` allocates strings for each metadata key
- Files: `src/loader/metadata.rs:49-240`
- Cause: String comparisons for metadata key matching
- Improvement path: Use string enums or compile-time hash-based dispatch

## Fragile Areas

**HIP Kernel Cache Initialization:**
- Files: `src/attention/kernels.rs:52-100`, `src/sampler/gpu.rs:62-90`
- Why fragile: Global `Mutex<Option<KernelCache>>` with double-checked locking. Race conditions possible if multiple threads initialize simultaneously.
- Safe modification: Use `OnceCell` or `LazyLock` (Rust 1.80+)
- Test coverage: Kernel cache tests exist but don't stress concurrent initialization

**GGUF Tensor Shape Inference:**
- Files: `src/loader/gguf.rs:1869-1920` (`infer_vocab_size_from_tensors`)
- Why fragile: Heuristic-based inference depends on `hidden_size` being correctly parsed. If metadata parsing fails, inference also fails.
- Safe modification: Parse tensor shapes first, then use them to validate/derive metadata
- Test coverage: Unit tests exist (`src/loader/metadata_tests.rs`) but may not cover all GGUF variants

**Paged KV Cache Block Management:**
- Files: `src/kv_cache/kv_cache.rs:698-707` (multiple RwLocks for internal state)
- Why fragile: Seven separate RwLocks for cache state - deadlocks possible if lock acquisition order not consistent
- Safe modification: Consolidate into single RwLock protecting all state, or use message-passing pattern
- Test coverage: Tests exist in `tests/kv_cache_tests.rs` but concurrent access may not be well tested

**Scheduler State Machine:**
- Files: `src/scheduler/scheduler.rs:22-29,83-100`
- Why fragile: Manual state transitions with `InvalidStateTransition` error - easy to add new states without updating all transitions
- Safe modification: Use type-level state machines (state pattern with Rust types)
- Test coverage: State transition tests exist but may not cover all edge cases

**Error Type Fragmentation:**
- Files: 18+ distinct error types across modules (`HipError`, `KvCacheError`, `SchedulerError`, etc.)
- Why fragile: Error conversion between types loses information. Inconsistent `?` propagation.
- Safe modification: Consolidate on `RocmForgeError` from `src/error.rs`
- Test coverage: Error path testing minimal

## Scaling Limits

**Single GPU Assumption:**
- Current capacity: One GPU device per process
- Limit: Cannot scale across multiple GPUs
- Scaling path: Multi-GPU support requires model sharding or data parallelism

**Global Backend Singleton:**
- Current capacity: Single `GLOBAL_BACKEND` instance
- Limit: Cannot run multiple models on different GPUs in same process
- Scaling path: Per-model backend instances, remove global singleton

**KV Cache Memory:**
- Current capacity: ~4-40GB VRAM per cache preset
- Limit: In-context length bounded by available VRAM
- Scaling path: Already implementing paged attention, but multi-GPU KV cache needed for longer contexts

**HTTP Server Concurrency:**
- Current capacity: Tokio runtime default (threads = CPU cores)
- Limit: Request throughput limited by single-threaded inference
- Scaling path: Batching scheduler already implemented, needs load balancing across instances

## Dependencies at Risk

**ROCm/HIP System Libraries:**
- Risk: External library dependency (`amdhip64.so`) must be present at runtime
- Impact: Application fails to start if ROCm not installed
- Migration plan: No alternative - AMD GPU only. Could add CPU fallback for testing.

**HSACO Kernel Files:**
- Risk: Pre-compiled kernel files must exist at paths specified by environment variables
- Impact: Kernel operations silently fall back to CPU or fail
- Migration plan: Compile kernels at build time and embed in binary, or bundle with release

**GGUF Format Compatibility:**
- Risk: llama.cpp may change GGUF spec without warning
- Impact: New models may not load
- Migration plan: Pin tested GGUF versions, add CI tests against official llama.cpp

**Tokenizers Crate:**
- Risk: `tokenizers = "0.15"` from Hugging Face has transitive dependencies
- Impact: Security vulnerabilities in transitive deps
- Migration plan: Consider alternative tokenizers, or regular dependency updates

## Missing Critical Features

**No Resource Limits on Model Loading:**
- Problem: No limits on model file size, tensor count, or allocation size
- Blocks: Safe deployment in multi-tenant environments
- Impact: Could exhaust system memory or VRAM

**No Graceful Shutdown:**
- Problem: HTTP server and CLI don't handle signals cleanly
- Blocks: Production deployment
- Impact: May corrupt state or leave GPU in bad state

**No Request Cancellation:**
- Problem: Once inference starts, cannot be cancelled
- Blocks: Interactive use cases, long-running requests
- Impact: Poor user experience for long prompts

**No Metrics/Telemetry Export:**
- Problem: Metrics collected but not exported externally
- Blocks: Production monitoring
- Files: `src/metrics.rs`, `src/otel_traces.rs` - collection exists but export incomplete

**No Authentication/Authorization:**
- Problem: HTTP server has no auth
- Blocks: Public-facing deployment
- Files: `src/http/server.rs` - no middleware for auth

## Test Coverage Gaps

**End-to-End Inference Path:**
- What's not tested: Complete inference from prompt to generated text
- Files: `src/engine.rs`, `src/model/simple_transformer.rs`
- Risk: Integration failures between components
- Priority: High

**HTTP Server Endpoints:**
- What's not tested: `/v1/completions`, SSE streaming, error responses
- Files: `src/http/server.rs`
- Risk: API contract violations
- Priority: High

**Concurrent Request Handling:**
- What's not tested: Multiple simultaneous inference requests
- Files: `src/scheduler/scheduler.rs`, `src/engine.rs`
- Risk: Race conditions, deadlocks under load
- Priority: Medium

**GPU Memory Leak Detection:**
- What's not tested: Long-running inference sessions
- Files: `src/backend/gpu_test_common.rs:94-120` (infrastructure exists but not used in integration tests)
- Risk: Memory exhaustion over time
- Priority: Medium

**Error Recovery Paths:**
- What's not tested: GPU errors, OOM, model load failures
- Files: All error handling
- Risk: Crashes instead of graceful degradation
- Priority: Medium

**Quantization Format Coverage:**
- What's not tested: Q4_1, Q5_0, Q5_1 (tests have compilation errors)
- Files: `tests/q_dequant_tests.rs`
- Risk: These formats don't work at all
- Priority: Low (formats less commonly used)

---

*Concerns audit: 2026-01-20*
