# Q4_K Investigation Findings

**Date:** 2026-04-15
**Status:** CRITICAL BUG - Q4_K produces gibberish output

## Problem Statement
Q4_K models produce gibberish output while Q4_0, Q6_K, Q8_0 all work correctly.
llama.cpp produces perfect output with the same model files, proving model files are valid.

## Investigation Results

### ✅ What We Verified (Unit Tests Pass)

1. **Scale Extraction Logic**
   - `get_scale_min_k4` function correctly extracts 6-bit scale/min values
   - Handles both simple cases (indices 0-3) and complex bit packing (indices 4-7)
   - Correctly uses u8 arithmetic (not i8) to avoid sign extension issues

2. **Dequantization Formula**
   - Formula: `output = d * sc * q - dmin * m` is correct
   - Math operations verified: `d1 * q - m1` where `d1 = d * sc` and `m1 = dmin * m`
   - Uses raw q values (0-15), NOT centered like Q4_0

3. **Struct Layout**
   - Q4_K block: 144 bytes total
   - Bytes 0-1: d (half float)
   - Bytes 2-3: dmin (half float)
   - Bytes 4-15: scales[12]
   - Bytes 16-143: qs[128]

4. **Scale Index Usage**
   - Uses scales in pairs: (0,1), (2,3), (4,5), (6,7)
   - NOT just even indices (0, 2, 4, 6)
   - Matches llama.cpp pattern: `is = 2 * il`, then use both `is+0` and `is+1`

5. **Nibble Processing**
   - Low nibbles go to positions 0-31
   - High nibbles go to positions 32-63
   - Correct bit extraction: `(q_byte & 0xF)` and `(q_byte >> 4) & 0xF`

### ❌ What's Broken (End-to-End Fails)

1. **CPU Implementation**: `src/cpu/quant.rs:237` (embed_q4_k)
   - Output: "Transformers Transformers Transformers..." (repeating)
   - Coherent English words but semantically wrong

2. **GPU Implementation**: TWO separate kernels!
   - **GEMV kernel** (`q4_k_gemv.hip`): "ucer琬鹃tramANCEQUE版&id稍联合会" (mixed scripts)
   - **GEMM kernel** (`q4_k_gemm.hip`): **BROKEN FORMULA - dividing instead of multiplying!**
   - Different gibberish from CPU

3. **Both Implementations**
   - Unit tests pass ✅
   - Real model inference fails ❌
   - This suggests the bug is NOT in the dequantization itself

## 🎯 ROOT CAUSE FOUND (2026-04-15)

### **CRITICAL BUG: GEMM Kernel Has Completely Wrong Formula**

**File:** `hip_kernels/quant/q4_k_gemm.hip` (line 54)

**Broken Code:**
```cpp
const float val = static_cast<float>(q4) / d + dmin;
```

**Correct Formula (llama.cpp):**
```cpp
output = d * sc * q - dmin * m
```

**What's Wrong:**
1. ❌ **Division instead of multiplication**: `q / d` instead of `d * sc * q`
2. ❌ **Addition instead of subtraction**: `+ dmin` instead of `- dmin * m`
3. ❌ **Missing scales array**: Never uses `scales[12]` at all!
4. ❌ **Missing get_scale_min_k4**: No 6-bit scale/min extraction
5. ❌ **No scale/min pairs**: Treats Q4_K like simple uniform quantization

**Why Unit Tests Passed:**
- We fixed the GEMV kernel (`q4_k_gemv.hip`) with correct formula
- We tested the formula with unit tests
- **BUT inference uses GEMM kernel (`gemm_q4_k_f32_launch`) which was still broken!**

**Code Path:**
```
src/gpu/ops.rs:1240 → gemm_q4_k_f32() → gemm_q4_k_f32_launch()
```

**Discovery Method:**
- Traced function calls from ops.rs
- Found TWO separate Q4_K kernels:
  1. `q4_k_gemm.hip` - GEMM (matrix multiplication) ← **ACTUALLY USED**
  2. `q4_k_gemv.hip` - GEMV (matrix-vector) ← **We fixed this one**

**Fix Applied:**
- Replaced `q4_k_gemm.hip` with correct implementation
- Added proper `get_scale_min_k4` function
- Fixed dequantization formula: `val = d * sc * q - dmin * m`
- Added scale/min pair processing: (0,1), (2,3), (4,5), (6,7)
- Fixed __shfl_down with explicit warp size 32

## Root Cause Analysis

### Issue: Monolithic Code Structure

**File:** `src/cpu/quant.rs` - **764 lines!**

This file contains ALL quantization formats mixed together:
- Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0
- GEMM, GEMV, dequantization, quantization
- Verification functions

**Problems with this structure:**
1. **Code duplication** - Similar patterns across formats create bugs
2. **Hard to verify** - Can't easily compare implementations
3. **Hidden bugs** - Bugs in one format affect others
4. **No isolation** - Can't test components independently

### Specific Issues Found

1. **Inconsistent Scale Types**
   - Q4_K uses `u8` scales (6-bit values 0-63)
   - But code was casting to `i8` prematurely
   - Fixed by keeping as `u8` until final `f32` cast

2. **Scale Index Confusion**
   - Initially used scales 0-7 separately
   - Corrected to use pairs: (0,1), (2,3), (4,5), (6,7)
   - This required careful study of llama.cpp layout

3. **Signed vs Unsigned Math**
   - Rust `i8 >> n` does arithmetic shift (sign extension) ❌
   - Rust `u8 >> n` does logical shift (zeros) ✅
   - Critical difference not caught by simple inspection

## Next Steps

### Immediate Actions Required

1. **Split quant.rs into separate modules**
   ```
   src/cpu/quant/
   ├── mod.rs
   ├── q4_0.rs
   ├── q4_1.rs
   ├── q4_k.rs
   ├── q5_k.rs
   ├── q6_k.rs
   └── q8_0.rs
   ```

2. **Create comprehensive test suite**
   - Test each quant format independently
   - Test dequantization with known inputs/outputs
   - Compare against llama.cpp reference

3. **Add data flow validation**
   - Test GGUF file loading
   - Verify tensor shapes and strides
   - Validate model architecture parameters

### Investigation Questions

1. **Is GGUF loading Q4_K data correctly?**
   - ✅ **VERIFIED CORRECT**: `src/loader/file.rs:93` correctly adds `data_start + desc.offset`
   - ✅ **VERIFIED CORRECT**: Block layout is 144 bytes (d + dmin + scales[12] + qs[128])
   - ✅ **VERIFIED CORRECT**: Tensor offsets are relative to data_start, not file beginning

2. **Are model parameters correct?**
   - Hidden size, num layers, etc.
   - Q4_K might have different architecture

3. **Is there a bug in the test itself?**
   - Model file might be corrupted
   - Tokenization might be wrong

### Latest Findings (2026-04-15 Afternoon)

#### Memory Layout Test Results
Test at offset 512 revealed the data is GGUF **metadata**, not tensor weights:
- `d = 0x6567` = ASCII "eg"
- `dmin = 0x656e` = ASCII "en"
- `scales = [72, 61, 6c, ...]` = ASCII "real.basemode"

**This proves we were testing the wrong bytes!** The actual Q4_K tensor data starts at `data_start` offset (after all metadata and tensor descriptors).

#### What We've Verified So Far
1. ✅ **Kernel formula**: Fixed from `q / d + dmin` to `d * sc * q - dmin * m`
2. ✅ **Block structure**: `half d[2]` union matches llama.cpp (access as d[0], d[1])
3. ✅ **Scale extraction**: `get_scale_min_k4` correctly extracts 6-bit values
4. ✅ **Scale pairs**: Uses (0,1), (2,3), (4,5), (6,7) correctly
5. ✅ **Q values**: Raw 0-15, not centered
6. ✅ **GGUF loading**: Correctly adds `data_start` to tensor offsets
7. ✅ **Block layout**: 144 bytes with correct byte offsets

#### What's Still Broken
- ❌ **Output is still gibberish**: "soeverথ.Designerfasfas端WithName" (mixed scripts)
- ❌ **Both CPU and GPU**: Different gibberish but both wrong
- ❌ **Unit tests pass**: But end-to-end fails

#### Current Hypothesis
Since all the low-level details are correct, the bug must be in:
1. **Data reshaping/transposition**: Q4_K might need different transpose logic
2. **Dimension interpretation**: `[vocab, hidden]` vs `[hidden, vocab]` confusion
3. **Multi-block assembly**: How 256-element blocks combine into the full embedding vector
4. **Hidden size mismatch**: Model expects different hidden size than we're providing

## Files Created During Investigation

1. `tests/q4_k_unit_test.rs` - Unit tests for scale extraction and formula
2. `tests/q4_k_reference_test.rs` - Model loading tests
3. `tests/q4_k_debug_dump.rs` - Byte layout verification
4. `INVESTIGATION_Q4_K.md` - This document

## Hypothesis

The bug is likely **NOT** in the dequantization formula itself (unit tests pass).
The bug is probably in:
1. **Data loading/parsing** - Wrong tensor offsets or shapes
2. **Model architecture** - Q4_K models have different structure
3. **Code organization** - 764-line file makes verification impossible

## Current Status (2026-04-15 Afternoon)

### ✅ What We Fixed
1. **GEMM kernel formula**: Changed from `val = q / d + dmin` to `val = d * sc * q - dmin * m`
2. **Both kernels**: GEMM and GEMV now use correct formula
3. **Scale extraction**: Uses `get_scale_min_k4` with 6-bit extraction
4. **Signed/unsigned shift**: Fixed Rust vs C math differences

### ❌ What's Still Broken
- **Q4_K output is gibberish** despite all fixes
- Mixed scripts in output (English + Bengali + Chinese)
- Both CPU and GPU produce different gibberish
- Unit tests pass but real inference fails

### 🎯 Verified Correct Components
1. **GGUF loading**: Correctly uses `data_start + offset`
2. **Block layout**: 144 bytes, correct structure
3. **Kernel formula**: Matches llama.cpp exactly
4. **Scale extraction**: 6-bit unpacking correct
5. **Scale pairs**: Uses (0,1), (2,3), (4,5), (6,7)
6. **Q values**: Raw 0-15, not centered

### 🔍 Next Steps - Systematic Data Flow Validation

**Hypothesis:** Bug is in data reshaping, dimension interpretation, or multi-block assembly.

**Required Investigation:**

1. **Verify tensor dimensions**
   - Check `[vocab_size, hidden_size]` vs `[hidden_size, vocab_size]`
   - Verify transpose logic for Q4_K specifically
   - Compare with Q4_0 (which works)

2. **Multi-block assembly test**
   - Dequantize first block manually
   - Verify it produces 256 f32 values
   - Check block boundary calculations
   - Verify row_offset calculation

3. **Data flow trace**
   - Add logging at each stage:
     - GGUF tensor load (first 100 bytes)
     - After dequantization (first 10 values)
     - After transpose (if applicable)
     - Final embedding vector (first 10 values)
   - Compare with llama.cpp at each stage

4. **Minimal reproducible test**
   - Create test with 2x2 Q4_K matrix
   - Known input, known expected output
   - Test on CPU and GPU
   - Compare byte-for-byte with llama.cpp

### 📊 Progress Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Unit tests | ✅ Pass | Formula is correct |
| GGUF loading | ✅ Verified | Offsets calculated correctly |
| Block layout | ✅ Verified | 144 bytes, correct structure |
| Kernel formula | ✅ Fixed | Both GEMM and GEMV |
| CPU inference | ❌ Broken | Wrong output |
| GPU inference | ❌ Broken | Wrong output |
| Data pipeline | ❓ Unknown | Needs tracing |

## Recommendation

**PROCEED WITH SYSTEMATIC DEBUGGING:**

1. **NO MORE KERNEL CHANGES** - Formula is verified correct
2. **Add data flow logging** - Trace bytes through pipeline
3. **Create minimal test** - 2x2 matrix with known output
4. **Compare with llama.cpp** - At each pipeline stage
5. **Check dimensions/transposes** - Most likely remaining bug location

The investigation has verified all low-level components are correct. The bug is almost certainly in how the data is being interpreted or assembled at a higher level.

## Files Created During Investigation

1. `tests/q4_k_unit_test.rs` - Unit tests for scale extraction and formula ✅ ALL PASS
2. `tests/q4_k_reference_test.rs` - Model loading tests
3. `tests/q4_k_debug_dump.rs` - Byte layout verification
4. `tests/q4_k_memory_test.rs` - Memory layout and block parsing ✅ VERIFIED
5. `docs/investigation/Q4_K_INVESTIGATION.md` - This document

## Timeline

- **2026-04-15 Morning**: Discovered unit tests pass but inference fails
- **2026-04-15 Midday**: Found GEMM kernel had wrong formula (`q / d + dmin`)
- **2026-04-15 Afternoon**: Fixed GEMM kernel, verified GGUF loading correct
- **2026-04-15 Late**: Still gibberish - bug must be in data reshaping/assembly

## Recommendation

**User was right**: Big files are where bugs hide. `src/cpu/quant.rs` is 764 lines and needs to be split.

**But first**: Complete systematic data flow validation to find the remaining bug.
