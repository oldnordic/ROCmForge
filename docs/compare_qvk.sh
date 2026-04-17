#!/bin/bash
# Compare CPU vs GPU Q values after QKV projection (before RoPE)

echo "=== CPU Q values (before RoPE) ==="
# Run CPU with diagnostic to print Q values before RoPE
./target/release/rocmforge \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1 2>&1 | \
  grep -E "Q.*BEFORE.*RoPE" | head -3

echo ""
echo "=== GPU Q values (before RoPE) ==="
# Run GPU with diagnostic to print Q values before RoPE
ROCMFORGE_DISABLE_DECODE_GRAPH=1 ./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "The" --no-template --top-p 1.0 --max-tokens 1 2>&1 | \
  grep -E "Q.*BEFORE.*RoPE" | head -3
