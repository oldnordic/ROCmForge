# Model Loading Investigation Report

**Date**: 2026-01-20
**Investigator**: Claude (Session Context)
**Status**: INVESTIGATION ONLY - No code changes made

---

## Executive Summary

Investigated a reported model loading hang in ROCmForge CLI. **Did NOT reproduce a hang** - instead found a different error that prevents model loading from completing.

## Test Conditions

- **Hardware**: AMD Radeon RX 7900 XT (RDNA3)
- **ROCm Version**: 7.1 (per documentation)
- **Model Tested**: `models/qwen2.5-0.5b.gguf` (428MB, smallest available)
- **Command**: `cargo run --release --bin rocmforge_cli -- generate --gguf models/qwen2.5-0.5b.gguf --prompt "Hi" --max-tokens 1`
- **Timeout**: 15 seconds

## Actual Behavior (Not a Hang)

The CLI did NOT hang. It progressed through model loading but failed during inference with:

```
Error processing request 0: Inference failed: Generic error: GGML layer exec failed: InvalidShape("Missing output desc")
```

### Output Progression Observed

1. **CLI started** ✅
2. **Model loading began** ✅
3. **Weight preloading to GPU** ✅ (this is where the previous hang was reported)
4. **Inference loop started** ✅
5. **Layer 0-6 executed successfully** ✅
6. **Failed at Layer 0, Node 7 (Reshape operation)** ❌

```
>>> execute_graph_with_config: Executing node 7 (op=Reshape)
>>> execute_op: op=Reshape
>>> View/Reshape: inputs=[TensorId(4)], outputs=[TensorId(7)]
>>> View/Reshape: Available tensor IDs in backend: [...]
ERROR: InvalidShape("Missing output desc")
```

## Root Cause Analysis

### Error Location

**File**: `src/ggml/hip_backend/mod.rs:1053-1055`

```rust
let output_desc = self
    .tensor_desc(outputs[0])
    .ok_or_else(|| GgmlError::InvalidShape("Missing output desc".to_string()))?;
```

### What's Happening

The execution graph contains a `Reshape` operation (node 7) with:
- Input: `TensorId(4)`
- Output: `TensorId(7)`

However, `TensorId(7)` is **not present** in the backend's tensor descriptor map (`self.tensors`). This means:

1. The GGUF loader created the execution graph with a reference to `TensorId(7)`
2. The graph executor expects `TensorId(7)` to have a descriptor
3. The descriptor was never created during graph initialization
4. Result: `tensor_desc(7)` returns `None` → Error

### Why This Matters

This is NOT a hang. It's an **execution graph initialization bug**. The reshape operation references a tensor that was never registered with the backend.

## Potential GPU Hang Risk (Not Observed)

### Risk Location

**File**: `src/loader/gguf.rs:1197-1200` (inside `load_to_gpu_async`)

```rust
// Allocate GPU buffer
let total_elements: usize = shape.iter().product();
let buffer = HipBuffer::new(total_elements * std::mem::size_of::<f32>())
    .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
```

### Why This Could Hang

Per `docs/ROCM_HANG_INVESTIGATION_2026-01-07.md`:

> **Multiple Small Allocations Pathology**: 24-layer KV cache allocation (48 × 7MB contiguous buffers) succeeds, but layer norm weight allocation (many × 3584-byte small buffers) HANGS.

The `load_to_gpu_async` function allocates **one GPU buffer per tensor** in a loop. For a typical model:
- ~200-400 tensors
- Each allocation = individual `hipMalloc` call
- Triggers the "many small allocations" pathology on RDNA3 cards

### Why It Didn't Hang Today

1. **Smaller model tested** (0.5B vs 14B)
2. **System state may differ** (GPU driver warm, previous allocations freed)
3. **CWSR workaround may be applied** (unknown - not verified)

## Code Flow Analysis

### Model Loading Path

```
CLI: rocmforge_cli.rs:688 (create_engine)
  → InferenceEngine::from_gguf() (engine.rs:282)
    → GgufLoader::new() (gguf.rs)
    → load_to_gpu_async() (gguf.rs:1035) ⚠️ MULTIPLE SMALL ALLOCATIONS
    → ModelRuntime::load_from_gguf_with_loader() (backend.rs:3008)
      → HipBackend::new()
      → ScratchBufferManager::new()
      → KVCache::new() ⚠️ MULTIPLE SMALL ALLOCATIONS
      → ExecutionPlan::from_gguf()
```

### Inference Path (Where Error Occurred)

```
CLI: run_local_generate() (rocmforge_cli.rs:510)
  → engine.submit_request()
  → inference_loop() → process_batch() → process_single_request_impl()
    → run_forward_pass() (engine.rs:801)
      → decode_step() (backend.rs:3163)
        → forward_layer_ggml_decode() (execution_plan)
          → execute_graph_with_config() (ggml/hip_backend/mod.rs)
            → execute_op() for Reshape → ERROR ❌
```

## Missing Tensor Descriptor Problem

### Tensor IDs in Backend at Error Time

```
Available: [TensorId(33), TensorId(10), TensorId(30), TensorId(38),
           TensorId(26), TensorId(21), TensorId(39), TensorId(24),
           TensorId(19), TensorId(17), TensorId(29), TensorId(14),
           TensorId(23), TensorId(15), TensorId(3), TensorId(12),
           TensorId(28), TensorId(0), TensorId(5), TensorId(34),
           TensorId(11), TensorId(25), TensorId(36), TensorId(22),
           TensorId(18), TensorId(32), TensorId(20), TensorId(2),
           TensorId(4), TensorId(16), TensorId(27), TensorId(31),
           TensorId(6), TensorId(1), TensorId(35), TensorId(37),
           TensorId(13)]

Missing: TensorId(7) ← This is the output of the Reshape operation
```

### Input TensorId(4) IS Available

The input tensor `TensorId(4)` exists in the backend, but the output `TensorId(7)` was never registered.

## Related Documentation

- `docs/ROCM_HANG_INVESTIGATION_2026-01-07.md` - Known ROCm firmware bugs on RDNA3
- `docs/implementation_principles.md` - Development principles (correct → measurable → fast)
- `docs/MANUAL.md` - Known issues and troubleshooting

## Next Steps (Not Implemented - Investigation Only)

### To Fix "Missing output desc" Error

1. Investigate `ExecutionPlan::from_gguf()` - ensure all tensor IDs are registered
2. Check reshape/view operation registration in graph construction
3. Verify tensor descriptor initialization includes intermediate tensors

### To Prevent GPU Hang

1. **Implement memory pooling** (per ROCM_HANG_INVESTIGATION recommendations)
2. **Batch allocations** - single large buffer, subdivided
3. **Add explicit synchronization** after every N allocations
4. **Consider applying `amdgpu.cwsr_enable=0`** kernel parameter workaround

## Safety Concerns

### No Safety Checks In GPU Allocation Path

**File**: `src/loader/gguf.rs:1197-1200`

```rust
let buffer = HipBuffer::new(total_elements * std::mem::size_of::<f32>())
    .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
```

**Missing**:
- No bounds checking on `total_elements` (could overflow)
- No limit on total allocation count
- No progress logging during allocation loop
- No timeout protection

**Recommendation**: Add allocation limits, progress logging, and consider memory pooling architecture.

## Conclusion

**Did NOT observe a hang** - found a different bug. The "Missing output desc" error prevents inference from completing. The GPU hang risk from multiple small allocations remains but was not triggered in this test run.

---

**Report End** - Investigation only, no code changes made.
