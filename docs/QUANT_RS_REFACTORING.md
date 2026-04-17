# Quant.rs Refactoring Specification

**Date:** 2026-04-15  
**Goal:** Split `src/gpu/kernels/quant.rs` (3,750 LOC) into per-format modules  
**Approach:** One subagent per quantization format, double-check before proceeding

---

## Current Structure

The file contains functions for these quantization formats:
- **Q4_K** - 4-bit K-quants (lines 29-205)
- **Q5_K** - 5-bit K-quants (lines 237-403)
- **Q6_K** - 6-bit K-quants (lines 436-691)
- **Q8_0** - 8-bit quants (lines 691-866)
- **Q4_0** - 4-bit quants (lines 901-1137)
- **Q4_1** - 4-bit quants with bias (lines 1389-1782)
- **GEMV/GEMM** - Matrix operations (lines 1192-2830)
- **Fused kernels** - Specialized kernels (lines 2348-2632)

---

## Target Structure

```
src/gpu/kernels/quant/
├── mod.rs              # Public exports, re-exports (~100 LOC)
├── common.rs           # Shared utilities, types (~200 LOC)
├── q4_k.rs             # Q4_K quantization (~500 LOC)
├── q5_k.rs             # Q5_K quantization (~500 LOC)
├── q6_k.rs             # Q6_K quantization (~700 LOC)
├── q8_0.rs             # Q8_0 quantization (~400 LOC)
├── q4_0.rs             # Q4_0 quantization (~600 LOC)
├── q4_1.rs             # Q4_1 quantization (~600 LOC)
└── gemm.rs             # GEMM kernels (~500 LOC)
```

---

## Critical Rules (MUST FOLLOW)

### 1. Rust vs C/C++ Math Differences

**SIGNED vs UNSIGNED SHIFT:**
```rust
// ❌ WRONG - Rust i8 does arithmetic shift (sign extension)
let x: i8 = -128;
let y = x >> 2;  // Sign extends! y = -32

// ✅ CORRECT - Rust u8 does logical shift (zeros)
let x: u8 = 128;
let y = x >> 2;  // Zeros! y = 32
```

**Rule:** Always use `u8` for bit manipulation, cast to `i8`/`f32` ONLY at the end.

### 2. AMD HIP Standards

**Warp Size:**
```cpp
// ✅ CORRECT - Explicit warp size for AMD
__shfl_down(sum, offset, 32);  // 3rd parameter is REQUIRED

// ❌ WRONG - NVIDIA style (missing warp size)
__shfl_down(sum, offset);
```

**Launch Bounds:**
```cpp
// ✅ CORRECT - AMD HIP standard
__launch_bounds__(32, 1)

// ❌ WRONG - No launch bounds
```

### 3. Function Signatures

All wrapper functions MUST follow this pattern:
```rust
pub fn <operation>_<format>(
    input: *const <input_type>,
    output: *mut <output_type>,
    n: usize,
) -> GpuResult<()> {
    // Safety checks
    if n == 0 { return Ok(()); }
    
    // Call HIP kernel
    let result = unsafe { <operation>_<format>_launch(...) };
    
    if result != hipError_t::hipSuccess {
        return Err(GpuError::HipApiError { ... });
    }
    
    Ok(())
}
```

### 4. Extern Function Declarations

```rust
// Use link_name if the C symbol differs from Rust name
#[link_name = "<exact_c_symbol>"]
fn <operation>_<format>_launch(...) -> hipError_t;
```

### 5. Error Handling

```rust
// Always validate parameters
if n % <block_size> != 0 {
    return Err(GpuError::HipApiError {
        code: -1,
        description: format!("n must be multiple of {}", <block_size>),
    });
}
```

---

## Subagent Tasks

Each subagent will:

1. **Read the current `quant.rs` file**
2. **Extract ALL functions for one format:**
   - Quantize function
   - Dequantize function  
   - Batched dequantize function
   - Verify accuracy function
   - Finalize metrics function
   - Any GEMV/GEMM functions for that format
3. **Create new file in `src/gpu/kernels/quant/`**
4. **Add proper imports and use statements**
5. **Verify AMD HIP compliance:**
   - Check all `__shfl_down` calls have explicit warp size (32)
   - Check all kernels have `__launch_bounds__`
   - Verify no C/C++ math assumptions in Rust code
6. **Create test that the split works:**
   - Import from new module
   - Verify code compiles
   - Run existing tests
7. **Document the file:**
   - Add header with format description
   - Document any special considerations
8. **Report completion with verification**

---

## Subagent Template

Each subagent will be spawned with this prompt:

```
You are a quantization format specialist. Your task is to extract ALL code for <FORMAT> from src/gpu/kernels/quant.rs into a new file src/gpu/kernels/quant/<format>.rs

CRITICAL RULES:
1. Use u8 for bit manipulation, cast to i8/f32 ONLY at end
2. All __shfl_down calls MUST have explicit 32 as 3rd parameter (AMD HIP)
3. All kernels MUST have __launch_bounds__(32, 1)
4. Preserve ALL function signatures exactly as they are
5. Keep all safety checks and error handling
6. Add proper imports: use crate::gpu::error::{GpuError, GpuResult};
7. Add proper imports for FFI types

STEPS:
1. Read src/gpu/kernels/quant.rs
2. Extract ALL <FORMAT> functions:
   - quantize_<format>
   - dequantize_<format>
   - dequantize_<format>_batched
   - verify_<format>_accuracy
   - finalize_<format>_metrics
   - gemv_<format>_* (if any)
   - gemm_<format>_* (if any)
3. Create src/gpu/kernels/quant/<format>.rs
4. Add file header with format description
5. Verify AMD HIP compliance
6. Report what you extracted and where
7. DO NOT modify any other files

Return a summary of what you extracted and confirm AMD HIP compliance.
```

---

## Verification Checklist

After each subagent completes, verify:

- [ ] New file created in `src/gpu/kernels/quant/`
- [ ] All functions for format extracted
- [ ] AMD HIP standards followed (checked code)
- [ ] Rust math correctness verified (u8 for bit ops)
- [ ] Compilation succeeds
- [ ] All existing tests pass
- [ ] No broken imports/references

---

## Order of Operations

1. **Q4_K** ← Start here (we just investigated it, most familiar)
2. Verify Q4_K split
3. **Q6_K** ← Next largest
4. Verify Q6_K split
5. **Q8_0**
6. Verify Q8_0 split
7. **Q4_0**
8. Verify Q4_0 split
9. **Q4_1**
10. Verify Q4_1 split
11. **Q5_K**
12. Verify Q5_K split
13. **Common module** (shared utilities)
14. **mod.rs** (public exports)
15. **Final integration test**

---

## Expected Time per Format

- **Extraction:** 5-10 minutes
- **Verification:** 5 minutes
- **Testing:** 5 minutes
- **Total per format:** ~15-20 minutes

**Total estimated time:** ~2-3 hours for all formats

---

## Success Criteria

When complete:
- ✅ `quant.rs` reduced from 3,750 LOC to ~100 LOC (just mod.rs)
- ✅ Each format in separate file
- ✅ All code follows AMD HIP standards
- ✅ All Rust math is correct
- ✅ All tests pass
- ✅ Code is easier to navigate and maintain
