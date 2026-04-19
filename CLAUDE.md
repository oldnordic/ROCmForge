# ROCmForge - LLM Inference on AMD GPUs

## Mission

ROCmForge is a high-performance LLM inference engine optimized for AMD GPUs using pure HIP/ROCm. Our mission is to bring competitive inference performance to AMD hardware through systematic optimization of quantization kernels, memory access patterns, and compute utilization.

**Current focus**: Q4_0 decode throughput optimization for Qwen2.5 models on RDNA2/3 architectures.

**Reference implementations**:
- llama.cpp - Quantization algorithms and GGUF format
- hipfire (https://github.com/Kaden-Schutt/hipfire) - Advanced HIP optimization techniques

## Project Status

**Working**:
- ✅ GPU decode path for Qwen2.5 Q4_0/Q4_K/Q6_K/Q8_0 models
- ✅ CPU fallback implementation
- ✅ Graph-captured decode replay
- ✅ Architecture-aware feature detection (RDNA1/2/3)
- ✅ Packed load optimizations for Q4_0 GEMV
- ✅ Portable DP4A kernel with software fallback (compiles on RDNA1/2/3, CDNA)
- ✅ Hardware DP4A acceleration on RDNA2 (gfx1030)

**In progress**:
- 🚧 Performance optimization of `dot4_manual()` for RDNA3
- 🚧 Investigation of RDNA3-specific vectorization patterns
- 🚧 WMMA optimization for RDNA3 (gfx1100+)
- 🚧 Automatic kernel dispatch based on detected features

**Measured performance** (Qwen2.5-7B-Instruct Q4_0, RX 7900 XT):
- Decode: ~106 tok/s (with graph + Q8 fastpath)
- Target: 150-200 tok/s through hipfire-derived optimizations

## Hardware Context

**Development machine**:
- OS: Linux 7.0.0-1-cachyos
- GPU: AMD RX 7900 XT (gfx1100 - RDNA3)
- ROCm: 7.2.0
- CPU: AMD Ryzen (details in `/proc/cpuinfo`)
- Compiler: clang 18

**Validation targets**:
- Qwen2.5-0.5B-Instruct-Q4_0 - `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`
- Qwen2.5-7B-Instruct-Q4_0 - `/home/feanor/Projects/Memoria/models/Qwen2.5-7B-Instruct-Q4_0-Pure.gguf`

## Repository Structure

```
rocmforge/
├── src/
│   ├── gpu/              # AMD GPU inference path (HIP)
│   │   ├── kernels/      # Kernel wrappers and launch logic
│   │   │   └── quant/    # Modularized quantization kernels
│   │   │       ├── mod.rs
│   │   │       ├── q4_0.rs
│   │   │       ├── q4_1.rs
│   │   │       ├── q4_k.rs
│   │   │       ├── q5_k.rs
│   │   │       ├── q6_k.rs
│   │   │       ├── q8_0.rs
│   │   │       └── legacy.rs
│   │   ├── ops.rs        # High-level GPU operations
│   │   ├── forward.rs    # Layer forward pass
│   │   └── features.rs   # GPU architecture detection
│   ├── cpu/              # CPU fallback implementation
│   └── main.rs           # CLI entry point
├── hip_kernels/quant/    # HIP kernel sources (.hip files)
├── tests/                # Integration and correctness tests
├── benches/              # Performance benchmarks
├── AGENTS.md             # Subagent quality standards (READ THIS)
└── CHANGELOG.md          # Detailed commit history

External references:
- /home/feanor/Projects/llama.cpp - Reference quantization implementation
- /home/feanor/Projects/rocm-examples/ - ROCm code samples
```

## Development Phases

### Phase 1: Infrastructure & Correctness (COMPLETE)
- [x] Modularized quantization kernels by format (q4_0, q4_1, q4_k, etc.)
- [x] GPU architecture detection and feature queries
- [x] Kernel correctness testing infrastructure
- [x] CPU fallback implementation for validation

### Phase 2: hipfire Technique Port (IN PROGRESS)
- [x] Packed 32-bit loads for Q4_0 GEMV
- [x] DP4A-optimized fusion kernel implementation
- [ ] DP4A kernel integration into decode pipeline
- [ ] WMMA optimization for RDNA3 (gfx1100+)
- [ ] Automatic kernel dispatch by architecture

### Phase 3: Profiling & Optimization (PLANNED)
- [ ] ROCm profiler integration (rocprofv3)
- [ ] Kernel launch autotuning
- [ ] Memory access pattern optimization
- [ ] Shared memory tiling strategies

### Phase 4: Production Readiness (PLANNED)
- [ ] Comprehensive model compatibility testing
- [ ] Performance regression testing
- [ ] Documentation and examples
- [ ] Release engineering

## Quality Gates

**All code must pass**:

1. **Compilation**: `cargo check --features gpu` (no errors)
2. **Testing**: `cargo test --features gpu -- --test-threads=1` (all pass)
3. **Formatting**: `cargo fmt --check`
4. **Linter**: `cargo clippy --all-targets --all-features` (no warnings)
5. **Standards**: Follow AGENTS.md ZERO TOLERANCE policy

**Zero tolerance for**:
- Stub implementations (`unimplemented!()`, `todo!()`)
- Placeholder code (`// TODO`, `// fixme`, `// for now`)
- Mock implementations marked as "temporary"
- Code written without mandatory tools (LSP, Magellan, llmgrep, Mirage)

## Multi-Agent Coordination

### Handover Protocol

**When to handover**:
- Context usage >80% (MANDATORY - do not wait for block)
- Task requires specialized expertise (HIP kernels, quantization algorithms)
- Agent cannot proceed without blocking

**Handover message format**:
```
HANDOVER: Context limit approaching

Completed: [task description]
Next task: [specific next step]
Git state: [commit SHA or status]
Notes: [context for next subagent]

Project docs: /home/feanor/Projects/rocmforge/CLAUDE.md
Resume from: [specific location]
```

### Subagent Instructions

**Before dispatching subagents**:
1. Read AGENTS.md completely
2. Prepare explicit file-by-file plan
3. Require use of mandatory tools: LSP (Rust Analyzer), Magellan, llmgrep, Mirage
4. Emphasize: NO stub implementations, NO placeholders, NO code without tool verification

**Subagent quality check**:
- Did they use LSP to verify function signatures before editing?
- Did they use Magellan to find all references before refactoring?
- Did they use llmgrep to search for existing patterns before implementing?
- Did they use Mirage to analyze control flow before optimizing?
- Are there ANY stub implementations? (REFUSE if yes)
- Are there ANY placeholder comments? (REFUSE if yes)

## GPU Lock Protocol

**Problem**: Multiple sessions/tests can deadlock GPU by acquiring exclusive locks

**Solution**:
- Use `ROCMFORGE_GPU_LOCK_TIMEOUT=30` to set lock timeout (seconds)
- Kill stuck processes: `pkill -9 rocmforge`
- Check lock status: `lsof /tmp/rocmforge_gpu_lock*`

**Testing with GPU**:
- Use `--test-threads=1` for serial test execution
- Prefer Criterion benchmarks for stable measurements
- Use rocprofv3 for profiling, not informal timing

## Rules (adapted from hipfire)

### DO:
- **Measure everything**: Use Criterion, rocprofv3, and built-in profiling
- **Profile before optimizing**: rocprofv3 traces reveal actual bottlenecks
- **Test on real hardware**: All GPU code must run on actual AMD GPUs
- **Validate correctness**: Kernel tests verify numerical accuracy
- **Git everything**: Commit frequently with conventional commits
- **Follow AGENTS.md**: Zero tolerance for stubs and placeholders

### DON'T:
- **No Python in hot path**: GPU kernels must be pure HIP, no PyTorch/NumPy
- **No untested merges**: All code must pass tests before main
- **No premature optimization**: Profile first, optimize hotspots only
- **No hardcoded paths**: Use relative paths or environment variables
- **No silent failures**: All errors must propagate, no `unwrap()` in hot path
- **No cross-vendor code**: This is AMD-only, no CUDA/Cross-platform abstractions

## Performance Workflow

### 1. Establish Baseline
```bash
# Build release binary
cargo build --release --features gpu

# Run Criterion benchmark
cargo bench --bench gpu_decode --features gpu -- --noplot

# Run real-model test
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```

### 2. Profile with rocprofv3
```bash
# Use repo wrapper script
./.rocprofv3/profile_decode.sh runtime

# Or manual invocation
/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace \
  --output-directory /tmp/rocprof-decode \
  --output-format csv -- \
  ./target/release/rocmforge --gpu --model <path> --prompt "Hello"
```

### 3. Analyze Results
- Check `/tmp/rocprof-decode/*.csv` for kernel timings
- Identify top 5 kernels by duration
- Check launch bounds (grid size, block size)
- Verify memory access patterns

### 4. Optimize
- Implement hipfire-derived techniques (DP4A, WMMA, packed loads)
- Update kernel dispatch in `src/gpu/ops.rs`
- Test with kernel correctness suite
- Measure improvement with Criterion

### 5. Validate
- Ensure numerical accuracy (no regressions in correctness tests)
- Check real-model throughput (tok/s)
- Profile again to verify improvement
- Commit with performance data in CHANGELOG

## Success Criteria

**Phase 2 (hipfire port)**:
- [ ] DP4A kernel integrated and dispatching on RDNA2/3
- [ ] Decode throughput ≥150 tok/s on Qwen2.5-0.5B Q4_0 (RX 7900 XT)
- [ ] No accuracy regression (kernel tests pass)
- [ ] rocprofv3 shows kernel time reduction in hot paths

**Phase 3 (production readiness)**:
- [ ] Decode throughput ≥200 tok/s on Qwen2.5-0.5B Q4_0
- [ ] Comprehensive model compatibility (all Qwen2.5 variants)
- [ ] Performance regression testing CI
- [ ] Documentation for external contributors

## Contact & Context

**Primary development context**: AGENTS.md, hipfire_quick_start.md, CHANGELOG.md

**External references**:
- llama.cpp: `/home/feanor/Projects/llama.cpp`
- ROCm 7.2 docs: https://rocm.docs.amd.com/en/docs-7.2.0/
- hipfire: https://github.com/Kaden-Schutt/hipfire (optimization reference)

**Session memory**: `~/.claude/projects/-home-feanor-Projects-rocmforge/memory/`

**Before starting work**: Read AGENTS.md completely. Zero tolerance policy applies to ALL code.
