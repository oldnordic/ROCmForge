#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LOCK_FILE="/tmp/rocmforge_gpu_lock"
GPU_LOCK_TIMEOUT="${ROCMFORGE_GPU_LOCK_TIMEOUT:-30}"

DEFAULT_TIMEOUT="${ROCMFORGE_DEFAULT_TIMEOUT:-120}"
DEFAULT_MAX_TOKENS="${ROCMFORGE_DEFAULT_MAX_TOKENS:-50}"
GPU_SAFE_RUN_LOCK_HELD=false

usage() {
    cat <<EOF
Usage: gpu_safe_run.sh [--timeout <seconds>] [--max-tokens <n>] <command>...

Run GPU CLI commands with safety harness.

Steps:
    1. Acquire GPU lock
    2. Run preflight checks
    3. Execute command with timeout
    4. Release lock

Options:
    --timeout <seconds>     Execution timeout (default: $DEFAULT_TIMEOUT)
    --max-tokens <n>        Max tokens for decode runs (default: $DEFAULT_MAX_TOKENS)

Environment:
    ROCMFORGE_DEFAULT_TIMEOUT      Default timeout (default: 120)
    ROCMFORGE_DEFAULT_MAX_TOKENS   Default max tokens (default: 50)
    ROCMFORGE_GPU_LOCK_TIMEOUT     Lock acquisition timeout (default: 30)

Exit codes:
    0    Success
    1-4  Preflight check failed (see gpu_preflight.sh)
    10   Lock acquisition timeout
    11   Command timeout
    12   Command execution failed
    255  Usage error

Examples:
    gpu_safe_run.sh ./target/release/rocmforge --gpu --model foo.gguf --prompt "Hi"
    gpu_safe_run.sh --timeout 60 --max-tokens 10 ./target/release/rocmforge --gpu ...
EOF
}

acquire_lock() {
    local start_time
    start_time=$(date +%s)

    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -ge $GPU_LOCK_TIMEOUT ]; then
            echo "gpu_lock: timeout after ${elapsed}s" >&2
            return 1
        fi

        if mkdir "$GPU_LOCK_FILE" 2>/dev/null; then
            echo $$ > "$GPU_LOCK_FILE/pid"
            echo "gpu_lock: acquired by PID $$" >&2
            return 0
        fi

        local lock_pid
        if [ -f "$GPU_LOCK_FILE/pid" ]; then
            lock_pid=$(cat "$GPU_LOCK_FILE/pid")
            if ! kill -0 "$lock_pid" 2>/dev/null; then
                echo "gpu_lock: stale lock detected (PID $lock_pid), removing" >&2
                rm -rf "$GPU_LOCK_FILE"
                continue
            fi
        fi

        sleep 0.1
    done
}

release_lock() {
    if [ ! -d "$GPU_LOCK_FILE" ]; then
        echo "gpu_lock: not held" >&2
        return 2
    fi

    local lock_pid
    if [ -f "$GPU_LOCK_FILE/pid" ]; then
        lock_pid=$(cat "$GPU_LOCK_FILE/pid")
        if [ "$lock_pid" != "$$" ]; then
            echo "gpu_lock: held by PID $lock_pid, not $$" >&2
            return 2
        fi
    fi

    rm -rf "$GPU_LOCK_FILE"
    echo "gpu_lock: released by PID $$" >&2
    return 0
}

run_preflight() {
    "${SCRIPT_DIR}/gpu_preflight.sh"
}

timeout_cmd() {
    local timeout="$1"
    shift

    if command -v timeout >/dev/null 2>&1; then
        timeout "$timeout" "$@"
    else
        echo "gpu_safe_run: 'timeout' command not found, running without timeout" >&2
        "$@"
    fi
}

main() {
    local timeout="$DEFAULT_TIMEOUT"
    local max_tokens="$DEFAULT_MAX_TOKENS"
    local cmd_args=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --timeout)
                if [ -z "${2:-}" ]; then
                    echo "gpu_safe_run: --timeout requires argument" >&2
                    exit 255
                fi
                timeout="$2"
                shift 2
                ;;
            --max-tokens)
                if [ -z "${2:-}" ]; then
                    echo "gpu_safe_run: --max-tokens requires argument" >&2
                    exit 255
                fi
                max_tokens="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --)
                shift
                cmd_args=("$@")
                break
                ;;
            *)
                cmd_args=("$@")
                break
                ;;
        esac
    done

    if [ ${#cmd_args[@]} -eq 0 ]; then
        echo "gpu_safe_run: no command specified" >&2
        usage >&2
        exit 255
    fi

    echo "gpu_safe_run: starting with timeout=${timeout}s, max-tokens=${max_tokens}"

    if ! acquire_lock; then
        echo "gpu_safe_run: failed to acquire GPU lock" >&2
        exit 10
    fi
    GPU_SAFE_RUN_LOCK_HELD=true

    cleanup() {
        if [ "${GPU_SAFE_RUN_LOCK_HELD:-false}" = true ]; then
            release_lock
            GPU_SAFE_RUN_LOCK_HELD=false
        fi
    }
    trap cleanup EXIT

    echo "gpu_safe_run: lock acquired, running preflight"

    if ! run_preflight; then
        local exit_code=$?
        echo "gpu_safe_run: preflight failed with code $exit_code" >&2
        exit $exit_code
    fi

    echo "gpu_safe_run: preflight passed, executing command"

    local full_cmd=("${cmd_args[@]}")

    if ! printf '%s\n' "${full_cmd[@]}" | grep -q -- '--max-tokens'; then
        full_cmd+=(--max-tokens "$max_tokens")
    fi

    if timeout_cmd "$timeout" "${full_cmd[@]}"; then
        echo "gpu_safe_run: command completed successfully"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
            echo "gpu_safe_run: command timed out after ${timeout}s" >&2
            exit 11
        else
            echo "gpu_safe_run: command failed with code $exit_code" >&2
            exit 12
        fi
    fi

    cleanup
    return 0
}

main "$@"
