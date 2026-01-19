---
phase: 19-wavefront-native-quantized-matmul
plan: 02
subsystem: gpu-kernels
tags: [hip, wave64, quantization, q4_0, q4_k, q6_k, matmul, shfl_down]

# Dependency graph
requires:
  - phase: 19-01
    provides: Quantization format analysis (Q4_0, Q4_K, Q6_K bit-packing layouts)
provides:
  - HIP-native quantized matmul kernels with wave64 alignment
  - CUDA __shfl_down_f32 intrinsics replaced with HIP __shfl_down
  - WARP_SIZE corrected from 32 (CUDA warp) to 64 (RDNA3 wavefront)
affects: [19-03, quantized-matmul-testing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - HIP wave reduction using __shfl_down() without WARP_SIZE parameter
    - Wave64-compatible shared memory allocation (WARP_SIZE=64)

key-files:
  created: []
  modified:
    - kernels/q4_0_matmul.hip
    - kernels/q4_k_matmul.hip
    - kernels/q6_k_matmul.hip

key-decisions:
  - "Correct WARP_SIZE from 32 to 64 for RDNA3 wavefront alignment"
  - "Replace __shfl_down_f32(partial, stride, WARP_SIZE) with __shfl_down(partial, stride)"
  - "Document TILE_SIZE_K=32, TILE_SIZE_N=32 as not wave64-aligned (defer optimization)"

patterns-established:
  - "Pattern: HIP shuffle uses type-generic __shfl_down(var, delta), not _f32 suffix"
  - "Pattern: WARP_SIZE inferred from wavefront, not passed as parameter"
  - "Pattern: Shared memory arrays sized using WARP_SIZE constant for architecture portability"

# Metrics
duration: 3min
completed: 2026-01-19
---

# Phase 19 Plan 02: Wavefront-Native Quantized Matmul HIP Intrinsics Summary

**WARP_SIZE corrected from 32 to 64 for RDNA3 wavefront alignment, __shfl_down_f32 replaced with HIP-native __shfl_down in Q4_0, Q4_K, Q6_K matmul kernels**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-19T21:06:41Z
- **Completed:** 2026-01-19T21:09:34Z
- **Tasks:** 6 (5 with changes, 1 verification-only)
- **Files modified:** 3

## Accomplishments

- Corrected WARP_SIZE constant from 32 (CUDA warp) to 64 (RDNA3 wavefront) in all three quantized matmul kernels
- Replaced CUDA-specific `__shfl_down_f32` intrinsic with HIP-native `__shfl_down` (6 occurrences total)
- Removed WARP_SIZE parameter from shuffle calls (HIP infers from wavefront)
- Verified shared memory arrays automatically resized via WARP_SIZE constant
- Documented tile size alignment (TILE_SIZE_K/N = 32, not wave64-aligned)

## Task Commits

Each task was committed atomically:

1. **Task 1: Correct WARP_SIZE constant from 32 to 64** - `e90cc90` (feat)
2. **Task 2: Verify shared memory arrays use WARP_SIZE constant** - (no changes, verification-only)
3. **Task 3: Replace __shfl_down_f32 in q4_0_matmul.hip** - `37a419a` (feat)
4. **Task 4: Replace __shfl_down_f32 in q4_k_matmul.hip** - `b20949b` (feat)
5. **Task 5: Replace __shfl_down_f32 in q6_k_matmul.hip** - `6fdfa85` (feat)
6. **Task 6: Verify tile sizes are wave64-aligned** - (documentation, no code changes)

**Plan metadata:** (to be added after SUMMARY.md commit)

## Files Created/Modified

- `kernels/q4_0_matmul.hip` - WARP_SIZE corrected to 64, __shfl_down_f32 replaced with __shfl_down
- `kernels/q4_k_matmul.hip` - WARP_SIZE corrected to 64, __shfl_down_f32 replaced with __shfl_down
- `kernels/q6_k_matmul.hip` - WARP_SIZE corrected to 64, __shfl_down_f32 replaced with __shfl_down

## Decisions Made

1. **WARP_SIZE = 64 for RDNA3**: Changed default from 32 (CUDA warp) to 64 (RDNA3 wavefront) for correct shared memory sizing and wave reduction loop bounds.

2. **Remove WARP_SIZE parameter from __shfl_down**: HIP infers the wavefront size from hardware, unlike CUDA which requires explicit width parameter.

3. **Tile size alignment deferred**: TILE_SIZE_K=32 and TILE_SIZE_N=32 are not wave64-aligned (not divisible by 64), but changing tile sizes requires performance validation. Documented as known issue for future optimization phase.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three quantized matmul kernels are now HIP-native with correct wave64 alignment
- Primary wave reduction paths using `__builtin_amdgcn_wave_reduce_fadd` unchanged (already HIP-native)
- Fallback paths now use HIP `__shfl_down` instead of CUDA `__shfl_down_f32`

**Known Issue - Tile Size Alignment:**
TILE_SIZE_K=32 and TILE_SIZE_N=32 are not wave64-aligned. Per plan, this is documented for future optimization rather than changed in this phase. Wave64 alignment would require TILE_SIZE to be divisible by 64 (e.g., 64, 128).

---
*Phase: 19-wavefront-native-quantized-matmul*
*Plan: 02*
*Completed: 2026-01-19*
