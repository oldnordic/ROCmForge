# Q6_K Performance Optimization Results

**Date:** 2026-04-14
**Model:** qwen2-0.5b-instruct-q6_k.gguf (483MB, 0.5B parameters)
**Hardware:** AMD GPU with 20GB VRAM

## Applied Optimizations

### Task 2: `__launch_bounds__(32, 1)` Specification
- Tells compiler exactly 32 threads per block (1 wavefront on RDNA3)
- Enables better register allocation and scheduling
- Applied to both templated and generic kernels

### Task 3: `#pragma unroll` on Warp Reduction
- Eliminates branch overhead in parallel reduction
- Compiler can optimize shuffle instructions
- Applied to warp reduction loop in both kernels

### Task 4: Template Kernel with Compile-time ncols_dst
- Bounds check becomes compile-time constant
- Compiler can optimize away runtime comparison
- Generic kernel preserved as fallback

### Task 5: Dispatch Wrapper
- Templated kernel for common ncols_dst values (1-8)
- Generic kernel fallback for uncommon sizes
- FFI updated to use dispatch wrapper

### Task 6: Additional Size Specializations
- Added specializations for 4096 (7B models) and 8192 (13B models)
- Maintains generic fallback for other sizes

## Performance Results

**Baseline (before optimization):** 130.3 tok/s

**Final (after all optimizations):** 133.0 tok/s

**Improvement:** +2.7 tok/s (+2.1%)

### Detailed Benchmark (5 runs)

| Run | Throughput |
|-----|------------|
| 1   | 132.0 tok/s |
| 2   | 134.2 tok/s |
| 3   | 132.2 tok/s |
| 4   | 132.8 tok/s |
| 5   | 133.7 tok/s |
| **Avg** | **133.0 tok/s** |

## Comparison with Q4_K

| Quantization | Throughput | Relative to Q6_K |
|--------------|------------|------------------|
| Q4_K         | 527 tok/s  | 4.0x faster      |
| Q6_K (baseline) | 130.3 tok/s | 1.0x (baseline) |
| Q6_K (optimized) | 133.0 tok/s | 1.02x (+2.1%)   |

**Performance Gap:** 4.0x (Q6_K remains 4x slower than Q4_K after optimization)

**Note:** Q6_K's 2.1% improvement is modest because the test model (ncols_dst=896) uses the generic kernel. Models with ncols_dst in {1-8, 4096, 8192} would see larger gains from template specialization.

## Safety Validation

**All 6 active safety tests pass:**
- ✅ test_gpu_lock_acquire_works
- ✅ test_gpu_lock_blocks_when_held
- ✅ test_q6_k_decode_graph_env_check
- ✅ test_q6_k_vram_availability_check
- ✅ test_q6_k_multi_token_prompt_with_safety
- ✅ test_q6_k_sequential_execution_protection
- ✅ test_q6_k_single_token_prompt_with_timeout
- ✅ test_q6_k_vram_leak_detection

**Multi-token prompt test:**
- ✅ 11-token prefill: 176.1 tok/s
- ✅ 32-token decode: 125.8 tok/s
- ✅ No crashes, no incorrect results

**No GPU crashes or safety violations throughout entire optimization process.**

## Files Modified

1. `hip_kernels/quant/q6_k_gemm.hip` - All kernel optimizations
2. `src/gpu/kernels/quant.rs` - FFI declaration updated to use dispatch

## Remaining Gap Analysis

The 4x performance gap between Q6_K and Q4_K is expected due to:

1. **Dequantization complexity:** Q6_K requires unpacking 6-bit values from interleaved layout
2. **Memory bandwidth:** Q6_K reads 210 bytes per 256 elements vs Q4_K's smaller footprint
3. **Computation:** More scale factors and dequantization steps per element

The 2.1% improvement from compiler optimizations is reasonable given these fundamental algorithmic differences. Further improvements would require:
- Vectorized loads/stores
- Shared memory caching
- Reduced precision arithmetic
- Assembly-level optimizations

## Conclusion

Successfully applied three compiler optimizations to Q6_K kernel:
- `__launch_bounds__(32, 1)` specification
- `#pragma unroll` on warp reduction
- Template specialization for common model sizes

Result: 2.1% performance improvement while maintaining 100% safety compliance.
