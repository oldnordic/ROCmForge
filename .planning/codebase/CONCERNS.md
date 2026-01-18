# Codebase Concerns

**Analysis Date:** 2026-01-18

## Tech Debt

**Large Complex Files:**
- Issue: 3 files exceed 3,000 LOC (violates 300 LOC limit)
- Files:
  - `src/model/execution_plan.rs` (4,410 lines)
  - `src/backend/hip_backend.rs` (3,625 lines)
  - `src/loader/gguf.rs` (2,832 lines)
- Why: Organic growth during development
- Impact: Difficult to navigate, maintain, and test
- Fix approach: Split into smaller, focused modules

**Test Infrastructure Debt:**
- Issue: 20+ commented tests needing rewrite for new GGUF API
- Files: `tests/loader_tests.rs`, `tests/embedding_to_lmhead_tests.rs`
- Why: API changes outpaced test updates
- Impact: Reduced test coverage, potential regressions
- Fix approach: Rewrite tests using current GgufLoader API

**Debug Output in Production Code:**
- Issue: 42 files contain `eprintln!` statements
- Why: Transition to `tracing` logging incomplete
- Impact: Inconsistent logging, stdout pollution
- Fix approach: Replace all `eprintln!` with `tracing::debug/info`

## Known Bugs

**GPU Stream Synchronization Bug:**
- Symptoms: Inference hangs during execution
- File: `src/ggml/hip_backend/ops/matmul.rs`
- Root cause: hipBLAS operations on custom stream vs hipMemcpy on default stream
- Fix: Ensure all GPU operations use consistent stream
- Status: Known issue awaiting fix

**Race Condition in Inference Loop:**
- Symptoms: Unpredictable hangs during inference
- File: `tests/inference_loop_spawn_race_condition_test.rs`
- Root cause: Concurrent task spawn without proper synchronization
- Fix: Add proper synchronization primitives

**Engine Cleanup Issues:**
- File: `src/bin/rocmforge_cli.rs`
- Symptoms: Resource leaks, improper shutdown
- Fix: Multiple BUG fixes needed for proper engine cleanup

## Security Considerations

**Environment Variable Validation:**
- Risk: Env vars used without validation for kernel paths
- Files: `src/attention/kernels.rs`
- Current mitigation: None documented
- Recommendations: Add path validation and sanitization

**Missing .env.example:**
- Risk: Users don't know required environment variables
- Current mitigation: Documentation in code
- Recommendations: Create `.env.example` with all required vars

## Performance Bottlenecks

**Excessive Cloning:**
- Problem: Heavy use of `Arc::clone()` throughout codebase
- Files: Multiple files (widespread pattern)
- Impact: Potential memory overhead, reference counting overhead
- Improvement path: Audit and reduce unnecessary cloning, use references where possible

**Large File Operations:**
- Problem: Single-file modules cause long compile times
- Files: `execution_plan.rs`, `hip_backend.rs`, `gguf.rs`
- Impact: Incremental compilation less effective
- Improvement path: Modularization improves compile times

## Fragile Areas

**Test Fixture Duplication:**
- File: `tests/e2e_suite.rs`, `tests/common/mod.rs`
- Why fragile: Integration tests cannot import from `tests/common/` directly
- Impact: Changes must be duplicated in both locations
- Safe modification: Always update both locations, add comment explaining duplication

**GPU Kernel Management:**
- Files: `build.rs`, `src/ggml/hip_backend/`
- Why fragile: Manual kernel compilation and linking
- Common failures: ROCm version mismatches, architecture misconfiguration
- Safe modification: Test on target GPU hardware, verify kernel compilation

## Scaling Limits

**Single GPU Only:**
- Current capacity: One AMD GPU
- Limit: No multi-GPU support
- Symptoms at limit: Cannot use multiple GPUs for larger models
- Scaling path: Implement model parallelism or tensor parallelism

**No Request Batching:**
- Current capacity: Single request at a time (scheduler incomplete)
- Limit: Low throughput for multiple concurrent users
- Scaling path: Complete `src/scheduler/` implementation for dynamic batching

## Dependencies at Risk

**Generic Version Constraints:**
- Risk: Dependencies use generic versions (e.g., "0.7", "1.0")
- File: `Cargo.toml`
- Impact: Potential breakage from dependency updates
- Migration plan: Pin to specific patch versions

**CodeMCP Integration:**
- Risk: Third-party AI assistant integration
- File: `.codemcp/config.toml`
- Impact: Development workflow relies on external tool
- Migration plan: Tool is optional, core development unaffected

## Missing Critical Features

**Native HIP Dequantization Kernel:**
- Problem: Dequantization happens on CPU before GPU matmul
- Files: `src/ggml/hip_backend/ops/quantized_matmul.rs`, `src/ggml/op.rs`
- Current workaround: CPU dequantization (slow)
- Blocks: Efficient quantized inference
- Implementation complexity: High (requires HIP kernel development)

**Flash Attention Detection:**
- Problem: Flash support hardcoded to false
- File: `src/attention/backend_registry.rs`
- Current workaround: CPU attention fallback
- Blocks: Optimized attention computation
- Implementation complexity: Medium (runtime detection)

## Test Coverage Gaps

**Commented Tests:**
- What's not tested: 20+ tests in `tests/loader_tests.rs` commented out
- Risk: GGUF loading regressions
- Priority: High
- Difficulty to test: Tests written, need rewrite for new API

**End-to-End Inference:**
- what's not tested: Full HTTP request → GPU inference → response flow
- Risk: Integration issues between components
- Priority: Medium
- Difficulty to test: Requires GPU and model files

**Error Recovery:**
- What's not tested: Behavior after GPU errors, OOM conditions
- Risk: Crashes instead of graceful degradation
- Priority: Medium
- Difficulty to test: Hard to simulate GPU errors

---

*Concerns audit: 2026-01-18*
*Update as issues are fixed or new ones discovered*
