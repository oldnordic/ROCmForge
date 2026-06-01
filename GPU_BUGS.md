# GPU Bugs and Fixes

This document documents GPU bugs that have been found and fixed in ROCm-Forge.
Each entry includes the bug, symptoms, root cause, and fix.

## April 2026: FFN Down Dimension Swap (Memory Corruption)

**Date:** 2026-04-14  
**Severity:** Critical (memory corruption, process crash)  
**Affected:** Non-graph decode path with stage profiling enabled

### Symptoms

- GPU memory access fault on decode
- Process abort with SIGABRT
- Error message: "Memory access fault by GPU node"
- Only affected hybrid path with stage profiling
- Graph path worked correctly

### Root Cause

In `src/gpu/forward.rs`, function `gpu_layer_forward_hybrid` had swapped
dimensions for the FFN down projection when stage profiling was enabled:

```rust
// WRONG (line 1450-1451):
gpu_dispatch_gemv_residual_on_stream(..., h, ff_size, ...)

// CORRECT (line 1166-1167 was already right):
gpu_dispatch_gemv_residual_on_stream(..., ff_size, h, ...)
```

### Why This Happened

Parameter order confusion between two similar functions:

```rust
// ops.rs signature:
pub fn gpu_dispatch_gemv_on_stream(..., out_dim: usize, in_dim: usize, ...)
pub fn gpu_dispatch_gemv_residual_on_stream(..., in_dim: usize, out_dim: usize, ...)
                                                    ^^^^^^^^  ^^^^^^^^^^
                                                    OPPOSITE ORDER!
```

For FFN down projection:
- Input: `swiglu` buffer (size = `ff_size`)
- Output: `hidden` buffer (size = `h`)
- Parameters: `(in_dim=ff_size, out_dim=h)`

### Impact

With swapped dimensions `(h, ff_size)` instead of `(ff_size, h)`:
- GEMV kernel reads only `h` elements from input (896 of 4864)
- GEMV kernel writes `ff_size` elements to output (4864 into 896-element buffer)
- Out-of-bounds memory write
- GPU memory access fault

### Fix

Swapped dimensions in `src/gpu/forward.rs:1450-1451`:

```diff
  gpu_dispatch_gemv_residual_on_stream(
      device,
      &gpu_layer.ffn_down,
      &gpu_layer.ffn_down_meta,
      scratch.swiglu.as_ptr() as *const f32,
      scratch.hidden.as_ptr() as *const f32,
      scratch.hidden.as_ptr() as *mut f32,
-     h,
-     ff_size,
+     ff_size,  // in_dim (input dimension)
+     h,        // out_dim (output dimension)
      device.stream(),
  )
```

### Prevention

1. **Regression test:** `tests/gpu_forward_regression.rs` ensures this doesn't regress
2. **Code review:** Pay attention to GEMV parameter order
3. **Type safety:** Consider wrappers to prevent confusion (future work)

### Related Issues

- RDNA4 support added at same time (Gfx1201 variant)
- WARP_SIZE default corrected from 64 to 32 for RDNA

---

## Template for Future Bugs

When documenting new bugs, include:

1. **Date and severity**
2. **Symptoms** (what went wrong)
3. **Root cause** (code location and bug)
4. **Why it happened** (design flaw, confusion, etc.)
5. **Impact** (user-facing consequences)
6. **Fix** (code changes)
7. **Prevention** (tests, documentation, design changes)
