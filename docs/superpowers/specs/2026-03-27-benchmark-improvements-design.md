# Benchmark Improvements Design

**Date:** 2026-03-27
**Status:** Approved Design
**Related Plans:**
- 2026-03-26-avx2-q4k-q8k-gemm.md
- 2026-03-26-q5_0-and-benchmark.md

## Overview

Add professional benchmark infrastructure to ROCmForge with three components:
1. Criterion-based kernel benchmarks with statistical rigor
2. Real model benchmark with end-to-end and kernel-level profiling
3. Automated performance comparison reports

This completes the original plans' "could be improved" items and provides
regression detection, professional documentation, and comprehensive performance
analysis.

## Goals

- **Regression detection:** Catch performance degradation during development
- **Professional reports:** Publication-ready performance comparisons
- **Comprehensive coverage:** Kernel-level + end-to-end + quantization type comparison
- **No CI/CD:** Manual execution, but with automated report generation

## Architecture

### Component 1: Criterion Kernel Benchmarks

**Location:** `benches/kernels.rs` (Criterion harness)

**Purpose:** Statistical benchmarking of core kernels with regression detection

**Benchmarks:**
| Benchmark | Description | Comparison |
|-----------|-------------|------------|
| `gemv_q4k_q8_avx2` | Q4_K × Q8_K GEMV (AVX2) | vs scalar |
| `gemv_q4k_q8_scalar` | Q4_K × Q8_K GEMV (scalar) | baseline |
| `gemm_q4k_q8_avx2` | Q4_K × Q8_K GEMM (AVX2) | vs scalar |
| `gemm_q5_0_q8_0` | Q5_0 × Q8_0 operations | quantization comparison |
| `dequant_q4k` | Q4_K dequantization | per-type comparison |
| `dequant_q5_0` | Q5_0 dequantization | per-type comparison |

**Output:**
- Console: Live progress + summary table
- Files: `target/criterion/` with HTML reports

**Usage:**
```bash
cargo bench --bench kernels
```

### Component 2: Real Model Benchmark

**Location:** `examples/benchmark_real_model.rs`

**Purpose:** End-to-end inference performance on real GGUF models with profiling

**Model Discovery:**
- Searches `/home/feanor/Projects/Memoria/models/` automatically
- Filters by pattern (e.g., `*q4_k*`, `qwen*`)
- Supports multiple quantization types for comparison

**Measurements:**
- **End-to-end:** Total time, tokens/sec, prefill ms, decode ms
- **Per-layer:** Forward pass timing for each transformer layer
- **Memory:** Peak memory usage
- **Kernel:** Which kernel was selected (AVX2, Scalar, AVX-512)

**Command Line:**
```bash
cargo run --release --example benchmark_real_model -- --model "*q4_k*" --iterations 3 --tokens 10 --profile
```

**Output:**
- Console: Real-time progress + summary
- File: `docs/benchmarks/real-model-YYYY-MM-DD.md`

### Component 3: Performance Comparison Report

**Location:** `src/bench/reporter.rs`, `examples/generate_report.rs`

**Purpose:** Aggregate benchmark results into publication-ready report

**Report Sections:**
1. **Executive Summary** - Key findings, recommendations
2. **Kernel Performance** - AVX2 vs Scalar speedup tables
3. **Quantization Comparison** - Q4_K vs Q5_0 vs Q6_K vs Q8_0
4. **Real Model Results** - End-to-end latency by model
5. **Recommendations** - Which quantization for each use case

**Output:**
- Markdown: `docs/benchmarks/PERFORMANCE_REPORT_YYYY-MM-DD.md`
- CSV: `docs/benchmarks/data.csv` (raw data)

## File Structure

```
rocmforge/
├── benches/
│   ├── kernels.rs              # Criterion benchmarks (NEW)
│   ├── cpu_gemv.rs             # Existing
│   └── gemm_q4k_q8.rs          # Existing
├── examples/
│   ├── benchmark_real_model.rs # Real model benchmark (NEW)
│   ├── generate_report.rs      # Report generator (NEW)
│   └── ...
├── src/
│   └── bench/
│       ├── mod.rs              # Benchmark utilities (NEW)
│       ├── reporter.rs         # Report generation (NEW)
│       └── discovery.rs        # Model discovery (NEW)
├── docs/
│   └── benchmarks/
│       ├── real-model-YYYY-MM-DD.md
│       ├── PERFORMANCE_REPORT_YYYY-MM-DD.md
│       └── data.csv
└── Cargo.toml                  # Add Criterion dependency
```

## Dependencies

**Add to `Cargo.toml`:**

```toml
[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "kernels"
harness = false
```

## Data Flow

```
┌─────────────────┐
│ Criterion       │
│ Benchmarks      │──┐
│ (benches/)      │  │
└─────────────────┘  │
                      │
┌─────────────────┐  │    ┌──────────────────┐
│ Real Model      │  ├────►│ Reporter         │
│ Benchmark       │  │    │ (aggregates)     │
│ (examples/)     │  │    └──────────────────┘
└─────────────────┘  │              │
                      │              ▼
┌─────────────────┐  │      ┌──────────────────┐
│ Model Files     │  │      │ Markdown Report  │
│ (Memoria/models)│─┘      │ + CSV Data       │
└─────────────────┘         └──────────────────┘
```

## Error Handling

All errors:
- Go to stderr with clear actionable messages
- Include context (which file, which operation)
- Suggest fixes when possible

**Specific cases:**
- No models found → List supported formats, check path
- Invalid dimensions → Skip with warning, log dimensions
- Feature detection failed → Fall back to scalar, log features
- Out of memory → Suggest reducing batch size

## Testing

**Unit tests** (`src/bench/`):
- Model discovery finds expected models
- Report generation produces valid markdown
- Timing utilities are accurate

**Integration test:**
- Run benchmark suite on small model (qwen2.5-0.5b)
- Verify all benchmarks complete
- Check report markdown validity

**Validation criteria:**
- Criterion: All finish, plausible results
- Real model: Runs on ≥2 different files
- Report: Creates both markdown and CSV
- No regression: AVX2 faster than scalar

## Usage Examples

**Run kernel benchmarks:**
```bash
cargo bench --bench kernels
# View: target/criterion/report/index.html
```

**Benchmark real models:**
```bash
cargo run --release --example benchmark_real_model
# Creates: docs/benchmarks/real-model-2026-03-27.md
```

**Generate full report:**
```bash
cargo run --release --example generate_report
# Creates: docs/benchmarks/PERFORMANCE_REPORT_2026-03-27.md
```

## Success Criteria

1. Criterion benchmarks produce statistical reports with confidence intervals
2. Real model benchmark successfully benchmarks ≥2 models from Memoria/models
3. Performance report generates valid markdown with comparison tables
4. All benchmarks complete without errors on AMD Ryzen (test system)
5. AVX2 kernels show measurable speedup over scalar baseline

## Implementation Notes

- No changes to existing kernel code
- Criterion runs in dev mode (fast) and release mode (accurate)
- Real model benchmark requires release build for meaningful timing
- Reports are version-controlled (commit with date stamp)
- Can re-run and regenerate reports incrementally
