# Plan 07-03: Cost Modeling for Backend Selection Summary

**Completed:** 2026-01-18
**Duration:** ~30 min
**Status:** Complete

## Accomplishments

### 1. Enhanced Cost Estimation (Task 1)
Implemented operation-aware cost model in `HybridScheduler::estimate_cost()`:
- **Base latency estimates per operation type:**
  - MatMul: 10us (large matrix operations)
  - QuantizedMatMul: 5us (faster due to fusion)
  - Softmax: 5us
  - Attention: 20us (most complex)
  - Add/Scale: 1us (element-wise)
  - Dequantize: 2us
- **Tensor size estimation:** Conservative estimates per operation type
  - MatMul: 2048 x 2048 elements
  - Softmax: 128 x 128 elements
  - QuantizedMatMul: 2048 x 2048 / 2 (compressed)
- **Memory estimation:** 4 bytes per F32 element
- **Transfer cost:** 10% overhead for CPU backends (simulates PCIe transfer)

### 2. Cost-Based Automatic Selection (Task 2)
Replaced simple GPU-first fallback with intelligent comparison:
- Compare estimated costs when both CPU and GPU available
- 2x threshold prevents oscillation between backends
- `SelectionReason::CostModel` records both costs for telemetry
- GPU preferred for large parallelizable operations
- CPU may be preferred for small ops where transfer cost dominates

### 3. HybridExecutor Implementation (Task 3)
Created `HybridExecutor` implementing `GgmlBackend`:
- Wraps CPU and optional GPU backends
- Uses `Box<dyn Any>` for buffer type erasure
- Heuristic-based operation routing:
  - MatMul, QuantizedMatMul, Attention → GPU
  - Add, Scale, Softmax → CPU
- Synchronizes all backends for consistency
- Public API: `new()`, `scheduler()`, `scheduler_mut()`

### 4. Test Coverage (Task 4)
Added 6 new tests for automatic selection (14 total tests):
- `test_automatic_prefers_gpu_for_large_ops` - GPU selection for MatMul
- `test_automatic_error_when_no_backends` - Error handling
- `test_cost_comparison` - 2x threshold validation
- `test_cost_model_with_transfer_penalty` - Transfer cost accounting
- `test_tensor_element_estimation` - Size estimates per op
- `test_enhanced_cost_estimation` - Cost by operation type

## Design Decisions

### Cost Model Approach
Used logarithmic scaling for tensor size to reflect sub-linear performance scaling on parallel hardware. Large operations don't scale linearly due to memory bandwidth limitations.

### Transfer Cost Modeling
Added 10% overhead for CPU backends to simulate PCIe transfer when data is on GPU. This creates a bias toward keeping computation on the device where data resides.

### 2x Threshold
CPU must be at least 2x faster than GPU to be preferred. This prevents oscillation between backends when costs are similar, and accounts for model inaccuracies.

### Buffer Type Erasure
`HybridExecutor` uses `Box<dyn Any>` for buffers since `CpuBackend::Buffer = Vec<f32>` and `HipGgmlBackend::Buffer = HipBuffer` are different types.

## Files Created/Modified

**Modified:**
- `src/ggml/hybrid_scheduler.rs` - Enhanced cost model, automatic selection, HybridExecutor (830 LOC total, +190 LOC)
- `src/ggml/mod.rs` - Added `HybridExecutor` export

## Test Coverage

**All tests passing: 315 total**
- `hybrid_scheduler`: 14/14 tests passing (6 new)
  - Original 8 tests still pass
  - New 6 automatic selection tests pass

### New Tests Added

1. `test_automatic_prefers_gpu_for_large_ops` - GPU selection for MatMul
2. `test_automatic_error_when_no_backends` - Error handling
3. `test_cost_comparison` - 2x threshold validation
4. `test_cost_model_with_transfer_penalty` - Transfer cost accounting
5. `test_tensor_element_estimation` - Size estimates per op
6. `test_enhanced_cost_estimation` - Cost by operation type

## Verification Status

- [x] Enhanced cost estimation based on operation type and tensor size
- [x] Automatic selection compares costs between backends
- [x] HybridExecutor delegates to appropriate backend
- [x] Tests validate selection logic
- [x] cargo check passes
- [x] All 315 tests passing

## Known Limitations

1. **Simplified tensor size estimation:** Uses conservative estimates rather than actual tensor shapes. Full implementation would need shape tracking through the graph.

2. **Static cost model:** Base latency estimates are hardcoded. Production system would use historical execution data for calibration.

3. **Limited operation coverage:** Some operations (LayerNorm, RmsNorm, Rope, SwiGlu, etc.) use default 1024-element estimate.

4. **HybridExecutor is basic:** Current implementation uses simple heuristics. Full integration would require:
   - Tensor data location tracking
   - Actual cost model integration in execute_op
   - Buffer transfer between backends

## Next Steps

Plan 07-04: Integration and Testing
- Integrate HybridExecutor with graph execution
- Add end-to-end tests for hybrid execution
- Benchmark CPU vs GPU backend selection
- Document usage patterns

## Commits

- `97f884e`: feat(07-03): implement enhanced cost estimation in HybridScheduler
- `73f009f`: feat(07-03): implement cost-based automatic backend selection
- `2d352b6`: feat(07-03): create HybridExecutor that wraps backends
- `73a8d1a`: test(07-03): add automatic selection tests
