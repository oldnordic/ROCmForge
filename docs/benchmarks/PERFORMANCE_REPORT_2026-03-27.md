# Performance Comparison Report

**Generated:** 2026-03-27 11:11

**Git Commit:** 75b21e9

## Executive Summary

TODO: Add key findings and recommendations.

## Kernel Performance

### Q4_K × Q8_K GEMV

| AVX2 | 0.045 ms | 2.91x speedup |
| Scalar | 0.131 ms | baseline |
### Q4_K × Q8_K GEMM

| AVX2 | 0.045 ms | 2.91x speedup |
| Scalar | 0.131 ms | baseline |
## Real Model Results

| Model | Quantization | Prefill (ms) | Decode (ms) | Tok/s |
|-------|--------------|--------------|-------------|-------|
| qwen2.5-0.5b-q4_k_m.gguf | Q4_K | 981.5 | 1135.5 | 4.4 |
