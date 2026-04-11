#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PERF_BIN="${PERF_BIN:-perf}"
BIN_PATH="${ROCMFORGE_BIN:-$ROOT_DIR/target/release/rocmforge}"
MODEL_PATH="${ROCMFORGE_BENCH_MODEL:-/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
PROMPT="${ROCMFORGE_BENCH_PROMPT:-Hello}"
TOKENS="${ROCMFORGE_BENCH_TOKENS:-64}"
REPEATS="${PERF_REPEATS:-3}"
EVENTS="${PERF_EVENTS:-task-clock,context-switches,cpu-migrations,page-faults,major-faults,minor-faults}"
PERF_ENABLE_DECODE_GRAPH="${PERF_ENABLE_DECODE_GRAPH:-1}"
PERF_ENABLE_Q8_ACT_FASTPATH="${PERF_ENABLE_Q8_ACT_FASTPATH:-1}"

DECODE_GRAPH_FLAG="${ROCMFORGE_ENABLE_DECODE_GRAPH:-$PERF_ENABLE_DECODE_GRAPH}"
Q8_FASTPATH_FLAG="${ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH:-$PERF_ENABLE_Q8_ACT_FASTPATH}"

if [[ ! -x "$BIN_PATH" ]]; then
    echo "release binary not found at $BIN_PATH" >&2
    echo "build it with: cargo build --release --features gpu" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "model not found at $MODEL_PATH" >&2
    echo "override with ROCMFORGE_BENCH_MODEL=/path/to/model.gguf" >&2
    exit 1
fi

echo "[perf_decode] ROCMFORGE_ENABLE_DECODE_GRAPH=$DECODE_GRAPH_FLAG" >&2
echo "[perf_decode] ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=$Q8_FASTPATH_FLAG" >&2

exec env \
    ROCMFORGE_ENABLE_DECODE_GRAPH="$DECODE_GRAPH_FLAG" \
    ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH="$Q8_FASTPATH_FLAG" \
    "$PERF_BIN" stat \
    -r "$REPEATS" \
    -e "$EVENTS" \
    -- \
    "$BIN_PATH" \
    --gpu \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --no-template \
    --top-p 1.0 \
    --max-tokens "$TOKENS"
