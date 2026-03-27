# Real Model Benchmark Results

**Date:** 2026-03-27

**CPU Kernel:** Avx512Vnni

## Results

| Model | Quantization | Layers | Hidden | Load (ms) | Prefill (ms) | Decode (ms) | Tok/s |
|-------|--------------|--------|--------|-----------|--------------|-------------|-------|
| TinyLlama.Q4_K_M.gguf | Q4_K | 22 | 2048 | 244.1 | 4809.7 | 782.8 | 2.6 |
| Qwen2.5-14B-Instruct-1M-q6_k_m.gguf | Unknown | 48 | 5120 | 3754.4 | 41742.2 | 27797.2 | 0.1 |
| ggml-model-Q8_0.gguf | Q8_0 | 32 | 2560 | 1309.2 | 10641.3 | 1768.8 | 1.1 |
| qwen2.5-7b-instruct-q4_k_m.gguf | Q4_K | 28 | 3584 | 2449.8 | 8518.4 | 5635.9 | 0.4 |
| qwen2.5-1.5b-instruct-f32.gguf | Unknown | 28 | 1536 | 3132.0 | 3858.6 | 2573.8 | 0.8 |
| llama3-8b-q4_k_m.gguf | Q4_K | 32 | 4096 | 2248.7 | 8582.9 | 5727.2 | 0.3 |
| qwen2-7b-instruct-q4_0.gguf | Q4_0 | 28 | 3584 | 2973.4 | 5010.7 | 3351.6 | 0.6 |
| qwen2.5-0.5b-q4_k_m.gguf | Q4_K | 24 | 896 | 281.3 | 664.8 | 449.4 | 4.5 |
| qwen2.5-3b-instruct-q4_k_m.gguf | Q4_K | 36 | 2048 | 1100.4 | 3786.2 | 2642.1 | 0.8 |
| qwen2.5-0.5b-instruct-q4_0.gguf | Q4_0 | 24 | 896 | 238.5 | 487.2 | 323.7 | 6.2 |
| Qwen2.5-7B-Instruct-Q4_0-Pure.gguf | Q4_0 | 28 | 3584 | 1895.1 | 3384.8 | 2263.6 | 0.9 |
| qwen3-30b-a3b-instruct-2507-q4_0.gguf | Q4_0 | 48 | 2048 | 10381.4 | 1142.4 | 755.5 | 2.6 |
| qwen2.5-0.5b-instruct-q4_0.gguf | Q4_0 | 24 | 896 | 165.4 | 239.2 | 161.7 | 12.4 |
