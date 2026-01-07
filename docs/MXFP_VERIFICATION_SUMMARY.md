# MXFP Verification Summary

**Status**: IMPLEMENTATION NOT FOUND - WAITING FOR AGENT 1
**Date**: 2026-01-06

## What I Checked

1. **src/loader/gguf.rs** - No MXFP types in enum
2. **kernels/mxfp_dequant.hip** - File does not exist
3. **src/loader/mxfp.rs** - File does not exist
4. **src/kv_cache/kv_cache.rs** - No MXFP6 dtype

## What Agent 1 Must Implement

### Critical Spec Details (Must Match Exactly)

#### MXFP4 (E2M1)
- **Bits**: 4 total (2 exponent, 1 mantissa, 1 sign)
- **Bit layout**: `[sign:1][exp:2][mant:1]` = `[bit3][bit2-1][bit0]`
- **Range**: [-6.0, 6.0]
- **Block size**: 32 elements
- **Scale**: E8M0 (8-bit exponent only)
- **Value**: `(-1)^sign * 2^(exp-1) * (1.mant)`

#### MXFP6-E2M3 (Recommended)
- **Bits**: 6 total (2 exponent, 3 mantissa, 1 sign)
- **Bit layout**: `[sign:1][exp:2][mant:3]` = `[bit5][bit4-3][bit2-0]`
- **Range**: [-7.5, 7.5]
- **Block size**: 32 elements
- **Scale**: E8M0 (8-bit exponent only)
- **Value**: `(-1)^sign * 2^(exp-1) * (1.mant/8)`

#### MXFP6-E3M2 (Optional - Not Fully Specified)
- **Bits**: 6 total (3 exponent, 2 mantissa, 1 sign)
- **Range**: NOT SPECIFIED in analysis doc
- **Action**: Must check AMD Quark source code

### Block Size (CRITICAL - Memory Corruption Risk)

```
MXFP4:  [scale: 1 byte] [32 x 4-bit elements = 16 bytes] = 17 bytes/block
MXFP6:  [scale: 1 byte] [32 x 6-bit elements = 24 bytes] = 25 bytes/block
```

### E8M0 Scale Format (CRITICAL - Scaling Errors)

```rust
// CORRECT:
pub struct E8M0 { exponent: i8 }  // Pure exponent, NO mantissa

// Value = 2^exponent
E8M0 { exponent: 0 }  ->  1.0
E8M0 { exponent: 1 }  ->  2.0
E8M0 { exponent: -1 } ->  0.5
```

## Verification Checklist (For When Agent 1 Completes)

### Part 1: GGUF Enum (src/loader/gguf.rs)
- [ ] MXFP4 enum value assigned (NOT 0-8)
- [ ] MXFP6_E2M3 enum value assigned
- [ ] MXFP6_E3M2 enum value assigned (if implemented)
- [ ] from_u32() handles MXFP types
- [ ] element_size() returns 32 for both

### Part 2: E8M0 Implementation (src/loader/mxfp.rs)
- [ ] E8M0 has exponent only (no mantissa)
- [ ] to_f32() returns 2^exponent
- [ ] from_f32() computes log2(value)
- [ ] Clamping to [-127, 127]

### Part 3: MXFP4 Decoding (kernels/mxfp_dequant.hip)
- [ ] Bit layout: `sign=(bits>>3)&1, exp=(bits>>1)&3, mant=bits&1`
- [ ] Value: `ldexpf(1.0+mant, exp-1)`
- [ ] Special case: `if(exp==0 && mant==0) return 0.0`
- [ ] Clipping: `fmaxf(-6.0, fminf(6.0, value))`

### Part 4: MXFP6-E2M3 Decoding (kernels/mxfp_dequant.hip)
- [ ] Bit layout: `sign=(bits>>5)&1, exp=(bits>>3)&3, mant=bits&7`
- [ ] Value: `ldexpf(1.0+mant/8.0, exp-1)`
- [ ] Special case: `if(exp==0 && mant==0) return 0.0`
- [ ] Clipping: `fmaxf(-7.5, fminf(7.5, value))`

### Part 5: Block Packing (kernels/mxfp_dequant.hip)
- [ ] MXFP4: 4-bit elements, 2 per byte
- [ ] MXFP6: 6-bit elements, across byte boundaries
- [ ] Scale is first byte in block

### Part 6: Test Coverage
- [ ] E8M0 round-trip test
- [ ] MXFP4 decode test (all 16 values)
- [ ] MXFP6-E2M3 decode test (all 64 values)
- [ ] Block packing test (edge cases)
- [ ] Accuracy validation (<0.1% error)

## Known Specification Ambiguities

1. **MXFP6-E3M2 Range** - Not specified, must check AMD docs
2. **GGUF Enum Values** - Conflicting proposals (32/33 vs 20/21/22)
3. **Block Size 64** - When to use vs 32? Not specified
4. **MXFP4 Value** - Analysis says "experimental", TODO includes it

## Discrepancy Severity Levels

**CRITICAL** (Must fix):
- Wrong bit layout → Data corruption
- Wrong block size → Memory corruption
- Wrong bias → Scaling errors
- Missing special cases → NaN propagation

**HIGH** (Should fix):
- Enum conflicts → Future compatibility
- Inefficient packing → Performance
- Missing tests → Quality risk

**MEDIUM** (Consider):
- Code style → Maintainability
- Missing docs → Usability

**LOW** (Nice to have):
- Variable names → Readability
- Optimizations → Performance

## Next Steps

1. **Agent 1**: Implement using checklist above
2. **Agent 1**: Address ambiguities by checking AMD docs
3. **Verification Agent**: Re-run verification when complete
4. **Verification Agent**: Create detailed discrepancy report

## Full Report

See: `/home/feanor/Projects/ROCmForge/docs/MXFP_VERIFICATION_REPORT_2026-01-06.md`
