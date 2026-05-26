#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-runtime}"

ROCPROF_BIN="${ROCPROF_BIN:-/opt/rocm/bin/rocprofv3}"
ROCMFORGE_BIN="${ROCMFORGE_BIN:-./target/release/rocmforge}"
ROCMFORGE_MODEL="${ROCMFORGE_MODEL:-/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
ROCMFORGE_PROMPT="${ROCMFORGE_PROMPT:-The quick brown fox jumps over the lazy dog. Then what happened?}"
ROCMFORGE_MAX_TOKENS="${ROCMFORGE_MAX_TOKENS:-1}"
ROCMFORGE_EXTRA_ARGS="${ROCMFORGE_EXTRA_ARGS:-}"
ROCPROF_OUTDIR="${ROCPROF_OUTDIR:-/tmp/rocprof-prefill}"
ROCPROF_ENABLE_DECODE_GRAPH_DEFAULT="${ROCPROF_ENABLE_DECODE_GRAPH_DEFAULT:-0}"
ROCPROF_ENABLE_Q8_ACTIVATION_FASTPATH_DEFAULT="${ROCPROF_ENABLE_Q8_ACTIVATION_FASTPATH_DEFAULT:-0}"

export ROCMFORGE_ENABLE_DECODE_GRAPH="${ROCMFORGE_ENABLE_DECODE_GRAPH:-$ROCPROF_ENABLE_DECODE_GRAPH_DEFAULT}"
export ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH="${ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH:-$ROCPROF_ENABLE_Q8_ACTIVATION_FASTPATH_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "${ROCPROF_BIN}" ]]; then
    echo "rocprofv3 not found: ${ROCPROF_BIN}" >&2
    exit 1
fi

if [[ ! -x "${ROCMFORGE_BIN}" ]]; then
    echo "rocmforge binary not found: ${ROCMFORGE_BIN}" >&2
    echo "build it with: cargo build --release --features gpu" >&2
    exit 1
fi

mkdir -p "${ROCPROF_OUTDIR}"

BASE_CMD=(
    "${ROCMFORGE_BIN}"
    --gpu
    --model "${ROCMFORGE_MODEL}"
    --prompt "${ROCMFORGE_PROMPT}"
    --no-template
    --top-p 1.0
    --max-tokens "${ROCMFORGE_MAX_TOKENS}"
)

if [[ -n "${ROCMFORGE_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${ROCMFORGE_EXTRA_ARGS} )
    BASE_CMD+=( "${EXTRA_ARGS[@]}" )
fi

COMMON_TRACE_ARGS=(
    --stats
    --summary
    --summary-output-file stdout
    --summary-units usec
    --group-by-queue
    --output-config
    --output-directory "${ROCPROF_OUTDIR}"
    --output-format csv
)

case "${MODE}" in
    runtime)
        echo "[profile_prefill] mode=${MODE}" >&2
        echo "[profile_prefill] ROCMFORGE_ENABLE_DECODE_GRAPH=${ROCMFORGE_ENABLE_DECODE_GRAPH}" >&2
        echo "[profile_prefill] ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=${ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH}" >&2
        exec "${ROCPROF_BIN}" \
            --runtime-trace \
            "${COMMON_TRACE_ARGS[@]}" \
            -- "${BASE_CMD[@]}"
        ;;
    runtime-batched)
        echo "[profile_prefill] mode=${MODE}" >&2
        echo "[profile_prefill] forcing batched prefill" >&2
        BASE_CMD+=(--prefill-only-validate)
        exec "${ROCPROF_BIN}" \
            --runtime-trace \
            "${COMMON_TRACE_ARGS[@]}" \
            -- "${BASE_CMD[@]}"
        ;;
    system)
        echo "[profile_prefill] mode=${MODE}" >&2
        echo "[profile_prefill] ROCMFORGE_ENABLE_DECODE_GRAPH=${ROCMFORGE_ENABLE_DECODE_GRAPH}" >&2
        echo "[profile_prefill] ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=${ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH}" >&2
        exec "${ROCPROF_BIN}" \
            --sys-trace \
            "${COMMON_TRACE_ARGS[@]}" \
            -- "${BASE_CMD[@]}"
        ;;
    *)
        echo "unknown mode: ${MODE}" >&2
        echo "modes: runtime, runtime-batched, system" >&2
        exit 2
        ;;
esac
