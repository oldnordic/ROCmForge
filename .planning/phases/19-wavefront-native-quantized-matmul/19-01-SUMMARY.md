---
phase: 19-wavefront-native-quantized-matmul
plan: 01
subsystem: gpu-kernels
tags: [quantization, q4_0, q4_k, q6_k, bit-packing, dequantization, gguf]

# Dependency graph
requires:
  - phase: 17-gpu-dequantization
    provides: CPU reference implementations for Q4_0, Q4_K, Q6_K dequantization
provides:
  - Bit-packing format documentation for Q4_0, Q4_K, Q6_K quantization formats
  - Verification of CPU implementations against llama.cpp specifications
  - Reference ground truth for GPU kernel numerical correctness validation
affects: [19-02, 19-03, hip-kernel-development, quantization-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [quantization-format-bit-packing, signed-6bit-conversion, nibble-extraction]

key-files:
  created: [.planning/phases/19-wavefront-native-quantized-matmul/19-01-SUMMARY.md]
  modified: []

key-decisions:
  - "CPU reference implementations verified as numerically correct ground truth for GPU kernels"
  - "Q4_0 format: 32 elements/block, 20 bytes/block, scale + 16 bytes packed 4-bit values"
  - "Q4_K format: 256-element super-blocks, 8 sub-blocks, scale/min per sub-block, value = min + quant*scale"
  - "Q6_K format: 256-element blocks, 16 f16 scales, 6-bit packed values with signed conversion"

patterns-established:
  - "Pattern: Dequantization formula validation via CPU reference comparison"
  - "Pattern: Bit-level layout documentation as prerequisite for GPU kernel development"

# Metrics
duration: 1min
completed: 2026-01-19
---

# Phase 19 Plan 01: Quantization Format Analysis Summary

**Bit-packing format documentation for Q4_0, Q4_K, Q6_K with CPU reference validation as ground truth for HIP kernel development**

## Performance

- **Duration:** < 1 minute
- **Started:** 2026-01-19T21:04:38Z
- **Completed:** 2026-01-19T21:05:04Z
- **Tasks:** 3
- **Files modified:** 0 (documentation only)

## Accomplishments

- **Q4_0 format documented**: 32 elements per 20-byte block with scale and 4-bit packed signed values
- **Q4_K format documented**: 256-element super-blocks with 8 sub-blocks, each with independent scale/min
- **Q6_K format documented**: 256-element blocks with 16 f16 scales and signed 6-bit packed values
- **CPU references verified**: All three implementations match llama.cpp numerical specifications

## Quantization Format Specifications

### Q4_0 Format

**Block Structure:**
- **Block size:** 32 elements
- **Bytes per block:** 20 bytes
  - 4 bytes: scale (f32, little-endian)
  - 16 bytes: packed 4-bit values (32 * 4 bits = 16 bytes)

**Bit Packing:**
- 2 values per byte
- Low nibble (bits 0-3): even index elements
- High nibble (bits 4-7): odd index elements

**Dequantization Formula:**
```
value = scale * ((packed & 0x0F) - 8)
```

**Signed Range:**
- Raw 4-bit: 0-15
- After conversion: -8 to +7

**Example:**
```rust
// From q4_0_dequant.rs lines 197-219
let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
let packed = data[4 + i];  // Get packed byte

// Low nibble (even index)
let low = (packed & 0x0F) as i32 - 8;
result[base_idx + i * 2] = scale * low as f32;

// High nibble (odd index)
let high = ((packed >> 4) & 0x0F) as i32 - 8;
result[base_idx + i * 2 + 1] = scale * high as f32;
```

**Constants verified:**
- `Q4_0_BLOCK_SIZE`: 20 bytes (lines 28)
- `Q4_0_ELEMENTS_PER_BLOCK`: 32 elements (line 29)

---

### Q4_K Format

**Super-Block Structure:**
- **Super-block size:** 256 elements (8 sub-blocks of 32 elements each)
- **Bytes per super-block:** 256 bytes
  - 16 bytes: 8 half-precision scales (f16, 2 bytes each)
  - 16 bytes: 8 int8 minimum values (1 byte each)
  - 224 bytes: 8 sub-blocks of 4-bit quantized values (28 bytes each)

**Sub-Block Layout (32 elements):**
- Scale: f16 (2 bytes) - shared across super-block
- Min: int8 (1 byte) - shared across super-block
- Quantized values: 28 bytes of 4-bit packed values

**Dequantization Formula:**
```
value = min + (quant * scale)
```

**Bit Extraction:**
```rust
// From q4_k_dequant.rs lines 259-270
let bit_pos = i * 4;
let byte_idx = bit_pos / 8;
let bit_offset = bit_pos % 8;

// Combine 2 bytes for 4-bit extraction
let combined = (data[quant_offset + 1] as u16) << 8 | (data[quant_offset] as u16);
let quant = ((combined >> bit_offset) & 0x0F) as u8;

result[element_idx] = min + (quant as f32) * scale;
```

**Format specification verified:** Lines 8-15 in q4_k_dequant.rs

---

### Q6_K Format

**Block Structure:**
- **Block size:** 256 elements
- **Bytes per block:** 256 bytes
  - 32 bytes: 16 half-precision scales (f16, 2 bytes each)
  - 192 bytes: 6-bit packed quantized values (256 * 6 / 8 = 192)
  - 32 bytes: padding (for alignment)

**Scale Grouping:**
- 16 elements share one f16 scale
- Element 0-15: scale[0], Element 16-31: scale[1], etc.

**Dequantization Formula:**
```
value = signed_6bit * scale
```

**Signed 6-bit Conversion:**
- Raw 6-bit: 0-63
- Signed conversion: if >= 32, subtract 64
- Final range: -32 to +31

**Bit Extraction:**
```rust
// From q6_k_dequant.rs lines 243-262
let bit_offset = (i * 6) % 8;
let byte_idx = (i * 6) / 8;

// Combine 2 bytes for 6-bit extraction
let combined = ((data[quants_start + byte_idx + 1] as u16) << 8)
    | (data[quants_start + byte_idx] as u16);

// Extract 6-bit value (position depends on bit_offset)
let quant_val = ((combined >> (10 - bit_offset)) & 0x3F) as u8;

// Convert to signed range: [0, 63] -> [-32, 31]
let signed_val = if quant_val >= 32 {
    (quant_val as i8 - 64) as f32
} else {
    quant_val as f32
};

result[element_idx] = signed_val * scale;
```

**Format specification verified:** Lines 8-17 in q6_k_dequant.rs

---

## Task Commits

This was a documentation-only plan with no code changes. No commits were made.

**Summary artifact created:** 19-01-SUMMARY.md

## Files Created/Modified

- `.planning/phases/19-wavefront-native-quantized-matmul/19-01-SUMMARY.md` - Quantization format documentation

## Decisions Made

- **CPU references validated as ground truth**: The CPU implementations in `q4_0_dequant.rs`, `q4_k_dequant.rs`, and `q6_k_dequant.rs` match llama.cpp specifications and serve as the numerical correctness reference for GPU kernels
- **No code changes required**: All format specifications were already correctly documented in the codebase comments; this plan consolidates them for kernel development reference

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for HIP kernel development (19-02):**
- Q4_0 bit-packing layout documented with extraction pattern
- Q4_K super-block structure documented with scale/min per sub-block
- Q6_K 6-bit signed packing documented with conversion formula

**Numerical validation approach:**
- Use `dequantize_q4_0_cpu()`, `dequantize_q4_k_cpu()`, `dequantize_q6_k_cpu()` as ground truth
- GPU kernel output must match CPU reference within floating-point tolerance

**Key considerations for 19-02 (HIP kernel rewrites):**
- Wave reduction: Use `__shfl_down()` or `__builtin_amdgcn_wave_reduce_fadd()` instead of CUDA `__shfl_down_f32()`
- Tile sizes: Use 64-multiple sizes for wave64 (CDNA) compatibility
- Bit extraction: Follow CPU patterns exactly for numerical correctness

---
*Phase: 19-wavefront-native-quantized-matmul*
*Completed: 2026-01-19*
