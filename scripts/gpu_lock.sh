#!/usr/bin/env bash
set -euo pipefail

GPU_LOCK_FILE="/tmp/rocmforge_gpu_lock"
GPU_LOCK_TIMEOUT="${ROCMFORGE_GPU_LOCK_TIMEOUT:-30}"

usage() {
    cat <<EOF
Usage: gpu_lock.sh <command>

Commands:
    acquire    Acquire GPU lock (blocks until available or timeout)
    release    Release GPU lock
    status     Check lock status

Environment:
    ROCMFORGE_GPU_LOCK_TIMEOUT    Timeout in seconds (default: 30)

Exit codes:
    0    Success
    1    Lock acquisition failed (timeout)
    2    Lock not held
    3    Lock file corrupted
EOF
}

acquire() {
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

release() {
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

status() {
    if [ ! -d "$GPU_LOCK_FILE" ]; then
        echo "gpu_lock: available"
        return 0
    fi

    local lock_pid
    if [ -f "$GPU_LOCK_FILE/pid" ]; then
        lock_pid=$(cat "$GPU_LOCK_FILE/pid")
        if kill -0 "$lock_pid" 2>/dev/null; then
            echo "gpu_lock: held by PID $lock_pid"
            return 0
        else
            echo "gpu_lock: stale lock (PID $lock_pid not running)"
            return 3
        fi
    else
        echo "gpu_lock: corrupted (no pid file)"
        return 3
    fi
}

main() {
    if [ $# -lt 1 ]; then
        usage >&2
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        acquire)
            acquire
            ;;
        release)
            release
            ;;
        status)
            status
            ;;
        *)
            echo "gpu_lock: unknown command '$command'" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
