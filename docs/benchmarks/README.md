# Benchmark Results

This directory contains automated benchmark reports for ROCmForge.

## Files

- `PERFORMANCE_REPORT_*.md` - Comprehensive performance comparison reports
- `real-model-*.md` - Real model inference benchmark results
- `data.csv` - Raw benchmark data in CSV format

## Running Benchmarks

### Kernel Benchmarks

```bash
cargo bench --bench kernels
```

View HTML report: `target/criterion/report/index.html`

### Real Model Benchmark

```bash
# Default (all models)
cargo run --release --example benchmark_real_model

# Specific models
cargo run --release --example benchmark_real_model -- --model "*q4_k*"
```

### Generate Report

```bash
cargo run --release --example generate_report
```

## Interpreting Results

- **Speedup**: How much faster AVX2 is compared to scalar baseline
- **Tokens/sec**: Higher is better (more tokens generated per second)
- **Prefill/Decode ms**: Lower is better (faster inference)

## Regression Detection

To detect performance regressions:

```bash
# Save baseline
cargo bench --bench kernels -- --save-baseline main

# Compare against baseline
cargo bench --bench kernels -- --baseline main
```
