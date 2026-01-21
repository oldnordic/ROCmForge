# HIP Kernel Debugging Guide

**For ROCmForge Developers**

This guide explains how to debug HIP kernel launch failures and issues using the debug hygiene features built into the HIP backend.

---

## Table of Contents

1. [Overview](#overview)
2. [Debug Builds](#debug-builds)
3. [HIP_LAUNCH_BLOCKING](#hip_launch_blocking)
4. [Error Messages](#error-messages)
5. [Common Issues](#common-issues)
6. [Tools](#tools)

---

## Overview

ROCmForge's HIP backend includes several debugging features to help diagnose kernel launch failures:

- **Debug-only logging**: Kernel launch dimensions are logged in debug builds using `#[cfg(debug_assertions)]`
- **Enhanced error messages**: Error messages include kernel name, grid dimensions, and block dimensions
- **Async error detection**: Calls `hipGetLastError()` after kernel launches to catch asynchronous HIP errors
- **Synchronous execution**: `HIP_LAUNCH_BLOCKING` environment variable enables synchronous kernel execution for easier debugging

These features are implemented in `src/backend/hip_backend/backend.rs` in the `launch_kernel_with_module_shared()` method.

---

## Debug Builds

### Enabling Debug Logging

Debug logging is automatically enabled when building in debug mode:

```bash
# Debug build (enables debug_assertions)
cargo build

# Run with debug logging
RUST_LOG=debug cargo run --bin rocmforge_cli

# Run tests with debug output
RUST_LOG=debug cargo test --lib
```

### What Gets Logged

In debug builds, every kernel launch logs:

```
Launch: kernel=kernel_name, grid=(x,y,z), block=(x,y,z), shared=N bytes
```

Example output:
```
DEBUG Launch: kernel=dequantize_q4_0, grid=(140,1,1), block=(64,1,1), shared=0 bytes
DEBUG Launch: kernel=matmul_f32, grid=(28,1,1), block=(256,1,1), shared=0 bytes
```

### Release Builds

Debug logging is **completely disabled** in release builds for zero performance overhead:

```bash
# Release build (no debug logging)
cargo build --release
```

The `#[cfg(debug_assertions)]` attribute ensures all debug logging code is compiled out in release mode.

### Code Reference

Debug logging is implemented in `launch_kernel_with_module_shared()`:

```rust
#[cfg(debug_assertions)]
tracing::debug!(
    "Launch: kernel={}, grid=({},{},{}), block=({},{},{}), shared={} bytes",
    kernel.name(),
    grid_dim.0, grid_dim.1, grid_dim.2,
    block_dim.0, block_dim.1, block_dim.2,
    shared_mem_bytes
);
```

---

## HIP_LAUNCH_BLOCKING

### What It Does

`HIP_LAUNCH_BLOCKING` is an official HIP environment variable that forces synchronous kernel execution. When enabled:

1. Each kernel launch blocks until completion
2. Errors are reported immediately at the source
3. Debugging is easier because the crash location is exact

**Trade-off**: Significant performance degradation. Only use for debugging.

### Enabling Synchronous Execution

```bash
# Enable synchronous kernel execution
HIP_LAUNCH_BLOCKING=1 cargo run --bin rocmforge_cli

# Or with "true"
HIP_LAUNCH_BLOCKING=true cargo run --bin rocmforge_cli
```

### How It Works

The backend reads `HIP_LAUNCH_BLOCKING` during initialization and stores it as a boolean flag:

```rust
let debug_sync_launch = std::env::var("HIP_LAUNCH_BLOCKING")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);
```

When enabled, `hipDeviceSynchronize()` is called after every kernel launch:

```rust
if self.debug_sync_launch {
    let sync_result = unsafe { ffi::hipDeviceSynchronize() };
    if sync_result != ffi::HIP_SUCCESS {
        let error_msg = get_error_string(sync_result);
        return Err(HipError::KernelLaunchFailed(format!(
            "Device synchronization failed after kernel '{}': {}",
            kernel.name(),
            error_msg
        )));
    }
}
```

### When to Use HIP_LAUNCH_BLOCKING

Use `HIP_LAUNCH_BLOCKING=1` when:

- Investigating kernel launch failures
- Debugging GPU crashes or hangs
- Verifying kernel execution order
- Developing new kernels

**Never use in production** - it defeats the purpose of asynchronous GPU execution.

---

## Error Messages

### Error Message Format

When a kernel launch fails, the error message includes:

```
Kernel 'kernel_name' launch failed: error_message (grid=(x,y,z), block=(x,y,z))
```

Example:
```
Kernel 'transpose_kernel' launch failed: hipErrorInvalidValue (grid=(2800,1,1), block=(64,64,1))
```

### Reading HIP Error Codes

The backend uses `hipGetErrorString()` to convert error codes to human-readable messages. Common errors:

| Error Code | Meaning | Typical Cause |
|------------|---------|---------------|
| `hipErrorInvalidValue` | Invalid argument | Block/grid dimensions exceed limits |
| `hipErrorInvalidLaunchConfig` | Invalid launch configuration | Thread count exceeds maxThreadsPerBlock |
| `hipErrorMemoryAllocation` | Out of memory | GPU memory exhausted |
| `hipErrorNotInitialized` | HIP not initialized | Backend creation failed |
| `hipErrorInvalidDevicePointer` | Invalid device pointer | Null or corrupted buffer |

### Async Error Detection

After a successful kernel launch, the backend checks for asynchronous errors:

```rust
let async_error = unsafe { ffi::hipGetLastError() };
if async_error != ffi::HIP_SUCCESS {
    let error_msg = get_error_string(async_error);
    tracing::warn!(
        "Async HIP error detected after kernel launch: code={}, msg={}",
        async_error,
        error_msg
    );
}
```

Async errors are logged as warnings (not errors) because they may originate from previous operations.

### Example Debug Session

```bash
# Run with sync execution and debug logging
HIP_LAUNCH_BLOCKING=1 RUST_LOG=debug cargo test test_transpose

# Output shows:
# 1. Kernel launch dimensions
# 2. Error location if failure occurs
# 3. Device limits at startup
```

---

## Common Issues

### Issue: Block Dimension Too Large

**Symptom:**
```
Kernel 'transpose' launch failed: hipErrorInvalidValue (grid=(2800,1,1), block=(64,64,1))
```

**Cause:** `block=(64,64,1)` = 4096 threads exceeds `maxThreadsPerBlock=1024`

**Solution:** Reduce block dimensions:
```rust
// Wrong: 64 * 64 * 1 = 4096 threads
let block_dim = (64, 64, 1);

// Correct: 32 * 32 * 1 = 1024 threads
let block_dim = (32, 32, 1);
```

### Issue: Grid Dimension Exceeds Limit

**Symptom:**
```
grid.x 100000 invalid (limit: 1..2147483647)
```

**Cause:** Grid dimension calculation overflow or incorrect value

**Solution:** Use safe grid dimension helpers:
```rust
use crate::backend::hip_backend::backend::safe_grid_dim;

let grid_x = safe_grid_dim(elements as u64, tile_dim)?;
```

### Issue: Shared Memory Exceeded

**Symptom:**
```
Shared memory 70000 bytes exceeds limit 65536
```

**Cause:** Kernel requests more shared memory than device provides per block

**Solution:** Reduce shared memory usage or split into multiple kernels

### Issue: Driver Reports Invalid Limits

**Symptom:**
```
Device limits: maxThreadsPerBlock=1024, maxThreadsDim=[1024, 0, 0]
```

**Cause:** HIP driver bug reporting `maxThreadsDim[1]` and `maxThreadsDim[2]` as 0

**Workaround:** Tests detect this condition and skip with a clear message

### Issue: Kernel Not Found

**Symptom:**
```
Kernel loading failed: hipErrorNotFound
```

**Cause:** HSACO file not found or kernel name mismatch

**Solution:**
- Verify HSACO path in build.rs
- Check kernel name matches HSACO symbol
- Ensure ROCm architecture matches target GPU

---

## Tools

### Environment Variables

| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `HIP_LAUNCH_BLOCKING=1` | Synchronous kernel execution | Debugging kernel failures |
| `RUST_LOG=debug` | Enable debug logging | See kernel launch details |
| `RUST_LOG=trace` | Enable trace logging | Deep debugging |
| `RUST_BACKTRACE=1` | Enable backtraces | Investigate panics |

### rocprof-tool (ROCm Profiler)

Profile kernel performance:

```bash
# Profile kernel execution
rocprof --hip-trace --basenameson ./target/release/rocmforge_cli

# Output: kernel timing, memory usage, etc.
```

### rocm-smi (GPU Monitoring)

Monitor GPU status:

```bash
# Show GPU info
rocm-smi

# Show continuous memory usage
watch -n 1 rocm-smi
```

### Device Properties Query

Check device limits programmatically:

```rust
let backend = HipBackend::new()?;
let limits = backend.limits();

println!("maxThreadsPerBlock: {}", limits.max_threads_per_block);
println!("maxThreadsDim: {:?}", limits.max_threads_dim);
println!("maxGridSize: {:?}", limits.max_grid_size);
println!("sharedMemPerBlock: {}", limits.shared_mem_per_block);
```

---

## Phase 27 Integration

This debugging guide builds on Phase 27 (Device Property Infrastructure). Device limits are cached during backend initialization and used for launch validation:

```rust
// Validate launch configuration before kernel execution
backend.validate_launch_config(grid_dim, block_dim, shared_mem_bytes)?;

// Launch if validation passes
backend.launch_kernel_with_module_shared(kernel, grid_dim, block_dim, args, shared_mem_bytes)?;
```

See `.planning/phases/27-device-property-infrastructure/27-04-SUMMARY.md` for details on device limit validation.

---

## Code Reference

Key files for debugging:

- `src/backend/hip_backend/backend.rs` - Kernel launch implementation
- `src/backend/hip_backend/device.rs` - `get_error_string()` function
- `src/backend/hip_backend/error.rs` - `HipError` enum
- `src/backend/hip_backend/ffi.rs` - HIP FFI bindings

---

## Best Practices

1. **Always use debug builds for development**
   ```bash
   cargo build  # Not cargo build --release
   ```

2. **Enable HIP_LAUNCH_BLOCKING when investigating failures**
   ```bash
   HIP_LAUNCH_BLOCKING=1 cargo test
   ```

3. **Check device limits before launching kernels**
   ```rust
   backend.validate_launch_config(grid_dim, block_dim, shared_mem_bytes)?;
   ```

4. **Read error messages carefully** - they include kernel name and dimensions

5. **Use RUST_LOG for detailed output**
   ```bash
   RUST_LOG=debug cargo test
   ```

---

## Further Reading

- [HIP Error Handling Documentation](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/error_handling.html)
- [HIP Environment Variables](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html)
- Phase 27 Research: `.planning/phases/27-device-property-infrastructure/27-04-SUMMARY.md`
- Phase 28 Research: `.planning/phases/28-debug-hygiene/28-RESEARCH.md`
