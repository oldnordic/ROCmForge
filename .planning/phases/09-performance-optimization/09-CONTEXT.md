# Phase 09: Performance Optimization - Context

**Gathered:** 2026-01-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Balanced optimization of throughput, latency, and memory efficiency for LLM inference on AMD GPUs. Profile current performance, identify bottlenecks, add benchmarks, and optimize until targets are met.

**Scope:**
- Throughput optimization (tokens/second)
- Latency optimization (time to first token)
- Memory efficiency (KV cache, allocations)
- Performance benchmarks and regression tests

**Out of scope:**
- New quantization formats (Phase 8)
- New model architectures (Phase 8)
- Production monitoring/observability (Phase 10)

</domain>

<decisions>
## Implementation Decisions

### Profiling Strategy
- **Tools:** ROCm tools (rocprof, rocperf) + standard Linux tools (perf, flamegraphs)
- **Hardware target:** RDNA2 and RDNA3 GPUs (test on both if available)
- **Model sizes:** Small (1-3B) and Medium (7-13B) parameters
- **Workload patterns:** Prompt processing + generation (end-to-end inference)

### Optimization Priorities
- **Approach:** Balanced — optimize throughput, latency, and memory based on context
- **Tradeoffs:** Context-dependent (no blanket preference)
- **Performance target:** Match llama.cpp on similar hardware
- **Depth:** Full optimization cycle until targets met (not just profiling)

### Benchmark Coverage
- **Operations:** Matmul variants (dense, quantized, batched), Attention ops (standard, flash, causal), Dequantization (all 15 formats)
- **Quantization formats:** Popular quants (Q4_0, Q4_K, Q6_K), K-quants (Q2_K, Q3_K, Q5_K), Unquantized (F16, F32 for baseline)
- **Scenarios:** Single-token generation, long prompt encoding, batch sizes (1, 4, 8 tokens)
- **Execution:** Manual benchmark runs only (not CI on every commit)

### Claude's Discretion
- Exact benchmark harness implementation (Criterion vs custom)
- Profiling workflow and tool integration
- How to measure and report results (format, verbosity)
- Specific optimizations to apply based on profiling data

</decisions>

<specifics>
## Specific Ideas

- Target llama.cpp performance — it's the reference implementation for GGUF inference
- Test on real GGUF models (small and medium)
- Profile both prompt processing (encoding) and generation (autoregressive)
- Benchmark should cover the hot path: matmul, attention, dequantization
- Manual benchmarks are fine — don't slow down CI with benchmark runs

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within performance optimization scope.

</deferred>

---

*Phase: 09-performance-optimization*
*Context gathered: 2026-01-18*
