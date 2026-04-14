#!/bin/bash
# Benchmark wrapper with GPU temperature monitoring
# Ensures GPU is not thermally throttled before running benchmarks

set -e

TEMP_LIMIT=83  # Maximum safe junction temperature (°C)
COOLDOWN_WAIT=60  # Seconds to wait if too hot

echo "=== Q6_K Benchmark with Temperature Monitoring ==="

# Function to check GPU junction temperature
check_temp() {
    local temp
    temp=$(rocm-smi --showtemp 2>/dev/null | \
           grep "Temperature (Sensor junction)" | \
           head -1 | \
           awk -F':' '{print $3}' | \
           sed 's/[^0-9.]//g')

    if [ -z "$temp" ] || [ "$temp" == "" ]; then
        echo "0"
        return 0
    fi

    # Round to integer
    temp=${temp%.*}
    echo "$temp"
}

# Function to wait for GPU cooldown
wait_for_cooldown() {
    local current_temp=$1
    local target_temp=$((TEMP_LIMIT - 5))
    local waited=0

    echo "GPU temperature: ${current_temp}°C (limit: ${TEMP_LIMIT}°C)"

    if [ "$current_temp" -ge "$TEMP_LIMIT" ]; then
        echo "⚠️  GPU TOO HOT - Waiting for cooldown to ${target_temp}°C..."

        while [ "$waited" -lt "$COOLDOWN_WAIT" ]; do
            sleep 10
            waited=$((waited + 10))
            current_temp=$(check_temp)
            echo "  [${waited}s] Temperature: ${current_temp}°C"

            if [ "$current_temp" -le "$target_temp" ]; then
                echo "✓ GPU cooled to ${current_temp}°C"
                break
            fi
        done

        if [ "$waited" -ge "$COOLDOWN_WAIT" ]; then
            echo "✗ Timeout waiting for cooldown (current: ${current_temp}°C)"
            echo "  Benchmark results may be throttled"
            return 1
        fi
    else
        echo "✓ Temperature OK: ${current_temp}°C"
    fi

    return 0
}

# Check temperature before benchmarking
echo "Checking GPU temperature..."
CURRENT_TEMP=$(check_temp)
wait_for_cooldown "$CURRENT_TEMP"

# Run benchmark command passed as arguments
echo ""
echo "=== Running Benchmark ==="
echo "Command: $@"
echo "$@"
echo ""

# Capture temperature during benchmark
TEMP_BEFORE=$(check_temp)
"$@"
RESULT=$?
TEMP_AFTER=$(check_temp)

echo ""
echo "=== Temperature Summary ==="
echo "Before: ${TEMP_BEFORE}°C"
echo "After:  ${TEMP_AFTER}°C"

if [ "$TEMP_AFTER" -ge "$TEMP_LIMIT" ]; then
    echo "⚠️  Warning: GPU reached ${TEMP_AFTER}°C during benchmark"
    echo "   Results may be throttled"
fi

exit $RESULT
