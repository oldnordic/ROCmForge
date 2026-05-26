#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: gpu_preflight.sh

Run staged GPU preflight checks before real-model execution.

Checks (in order):
    1. Driver/render node presence
    2. ROCm/HIP runtime device visibility
    3. Memory round-trip
    4. Trivial kernel launch

Exit codes:
    0    All checks passed
    1-4  Failed at check N
    255  Usage error
EOF
}

log_check() {
    local num="$1"
    local name="$2"
    echo "[check $num] $name"
}

check1_render_node() {
    log_check 1 "render node"

    if [ -d "/dev/dri" ]; then
        for node in /dev/dri/renderD*; do
            if [ -e "$node" ]; then
                echo "  found: $node"
                return 0
            fi
        done
    fi

    echo "  error: no render node found in /dev/dri" >&2
    return 1
}

check2_rocm_visibility() {
    log_check 2 "ROCm/HIP runtime"

    if ! command -v rocminfo >/dev/null 2>&1; then
        echo "  error: rocminfo not found" >&2
        return 2
    fi

    if ! rocminfo >/dev/null 2>&1; then
        echo "  error: rocminfo failed" >&2
        return 2
    fi

    local device_count
    device_count=$(rocminfo | grep -c "Marketing Name:" || true)

    if [ "$device_count" -lt 1 ]; then
        echo "  error: no devices visible" >&2
        return 2
    fi

    echo "  devices: $device_count"
    return 0
}

check3_memory_roundtrip() {
    log_check 3 "memory round-trip"

    local temp_binary
    temp_binary=$(mktemp /tmp/rocmforge_preflight_XXXXXX)

    cleanup() {
        rm -f "$temp_binary"
    }
    trap cleanup EXIT

    cat > "$temp_binary.cpp" <<'CPP_END'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
    const size_t size = 1024;
    float *d_data = nullptr;
    float *h_data = new float[size];

    for (size_t i = 0; i < size; i++) {
        h_data[i] = static_cast<float>(i);
    }

    hipError_t err = hipMalloc(&d_data, size * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        return 1;
    }

    err = hipMemcpy(d_data, h_data, size * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy H2D failed: %s\n", hipGetErrorString(err));
        hipFree(d_data);
        return 1;
    }

    err = hipMemcpy(h_data, d_data, size * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy D2H failed: %s\n", hipGetErrorString(err));
        hipFree(d_data);
        return 1;
    }

    for (size_t i = 0; i < size; i++) {
        if (h_data[i] != static_cast<float>(i)) {
            fprintf(stderr, "data mismatch at %zu: got %f, expected %f\n",
                    i, h_data[i], static_cast<float>(i));
            hipFree(d_data);
            return 1;
        }
    }

    hipFree(d_data);
    delete[] h_data;
    return 0;
}
CPP_END

    local hipcc_flags="-O2"
    if ! hipcc $hipcc_flags "$temp_binary.cpp" -o "${temp_binary}.bin" 2>/dev/null; then
        echo "  error: failed to compile preflight kernel" >&2
        return 3
    fi

    if ! "${temp_binary}.bin" >/dev/null 2>&1; then
        echo "  error: preflight kernel execution failed" >&2
        return 3
    fi

    rm -f "${temp_binary}.bin"
    echo "  ok"
    return 0
}

check4_trivial_kernel() {
    log_check 4 "trivial kernel launch"

    local temp_binary
    temp_binary=$(mktemp /tmp/rocmforge_preflight_XXXXXX)

    cleanup() {
        rm -f "$temp_binary"
    }
    trap cleanup EXIT

    cat > "$temp_binary.cpp" <<'CPP_END'
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void trivial_kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    const size_t size = 256;
    const size_t bytes = size * sizeof(float);
    float *d_data = nullptr;

    hipError_t err = hipMalloc(&d_data, bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipLaunchKernelGGL(trivial_kernel,
                       dim3(1), dim3(size),
                       0, 0,
                       d_data);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", hipGetErrorString(err));
        hipFree(d_data);
        return 1;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "kernel sync failed: %s\n", hipGetErrorString(err));
        hipFree(d_data);
        return 1;
    }

    hipFree(d_data);
    return 0;
}
CPP_END

    local hipcc_flags="-O2"
    if ! hipcc $hipcc_flags "$temp_binary.cpp" -o "${temp_binary}.bin" 2>/dev/null; then
        echo "  error: failed to compile trivial kernel" >&2
        return 4
    fi

    if ! "${temp_binary}.bin" >/dev/null 2>&1; then
        echo "  error: trivial kernel execution failed" >&2
        return 4
    fi

    rm -f "${temp_binary}.bin"
    trap - EXIT
    echo "  ok"
    return 0
}

main() {
    if [ $# -ne 0 ]; then
        usage >&2
        exit 255
    fi

    echo "gpu_preflight: starting staged checks"

    check1_render_node || exit $?
    check2_rocm_visibility || exit $?
    check3_memory_roundtrip || exit $?
    check4_trivial_kernel || exit $?

    echo "gpu_preflight: all checks passed"
    return 0
}

main "$@"
