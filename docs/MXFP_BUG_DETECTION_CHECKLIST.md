# MXFP4/MXFP6 Bug Detection Checklist

**Status**: IMPLEMENTATION NOT FOUND - Waiting for Agent 1
**Last Updated**: 2026-01-06

## Quick Reference: Critical Bugs to Avoid

### Memory Safety Bugs

- [ ] **Buffer Overflow Prevention**
  - Use `checked_mul()` for all block offset calculations
  - Validate tensor size BEFORE allocating buffers
  - Check data size matches expected size upfront

- [ ] **Integer Overflow Prevention**
  - Block index: `block_idx.checked_mul(BLOCK_SIZE)?`
  - Element index: `block_start.checked_add(element_offset)?`
  - Use `usize::checked_mul` for all size calculations

- [ ] **Uninitialized Memory**
  - Initialize GPU buffers to zero
  - Validate all tensor data is present before upload
  - Check for partial reads from disk

### Float Precision Bugs

- [ ] **NaN/Inf Validation**
  - Reject NaN/Inf in input tensors
  - Validate computed scales are finite
  - Check dequantized outputs are finite

- [ ] **Precision Loss**
  - Document MXFP bit layout explicitly
  - Test round-trip accuracy (f32 → MXFP → f32)
  - Use round-to-nearest-even (IEEE 754)

- [ ] **Special Cases**
  - Zero handling (signed zero)
  - Subnormal numbers
  - Overflow/underflow saturation

### Edge Cases

- [ ] **Empty Tensors**
  - Test `total_elements = 0`
  - Handle gracefully without division by zero

- [ ] **Misaligned Sizes**
  - Test `size % BLOCK_SIZE != 0`
  - Validate alignment upfront or support padding

- [ ] **Extreme Values**
  - `f32::MAX`, `f32::MIN`
  - `f32::INFINITY`, `f32::NAN`
  - Should error or saturate correctly

- [ ] **Zero Values**
  - All-zero tensor
  - Zero scale
  - Should preserve semantics

### GPU Kernel Bugs

- [ ] **Thread Divergence**
  - Use stride-based access, not conditional branches
  - Avoid `if (tid % 2 == 0)` patterns

- [ ] **Reduction Loops**
  - Always use `BLOCK_SIZE / 2` or `blockDim.x / 2`
  - NEVER use hardcoded `stride = 16`

- [ ] **Shared Memory**
  - Validate size fits in shared memory limits
  - Use `__syncthreads()` correctly

- [ ] **Boundary Conditions**
  - Check `idx < n` before accessing arrays
  - Handle partial blocks at end of tensor

### Dequantization Logic Bugs

- [ ] **Sign Bit Handling**
  - Extract sign bit: `(bits >> 7) & 1`
  - Apply to value: `sign = if bit { -1.0 } else { 1.0 }`

- [ ] **Exponent Bias**
  - MXFP4: bias = 7 (4-bit exponent)
  - MXFP6: bias = 31 (6-bit exponent) or variant
  - Document which bias is used

- [ ] **Mantissa Scaling**
  - MXFP4 (3-bit): `mantissa / 8.0`
  - MXFP6 (variable): depends on format
  - Add implicit leading 1: `1.0 + mantissa_scaled`

- [ ] **Bit Packing Order**
  - Document endianness
  - Test with known values
  - Validate against spec

## Test Coverage Checklist

### Unit Tests (Required)

- [ ] Basic correctness (small tensors, 10x10)
- [ ] Large tensors (≥1M elements)
- [ ] Empty tensor (size = 0)
- [ ] Single element (size = 1)
- [ ] Misaligned sizes (31, 33, 65)
- [ ] Maximum values (f32::MAX)
- [ ] Minimum values (f32::MIN)
- [ ] Zero tensor (all elements = 0)
- [ ] Zero scale (scale = 0.0)
- [ ] Round-trip accuracy (< 1% error)

### Property Tests (Recommended)

- [ ] Round-trip property: f32 → MXFP → f32 ≈ original
- [ ] Linearity property: quant(a + b) ≈ quant(a) + quant(b)
- [ ] Monotonicity: if a > b then quant(a) ≥ quant(b)
- [ ] Saturation: MAX_VALUE round-trips correctly

### Integration Tests (Required)

- [ ] Load real GGUF file with MXFP tensors
- [ ] Upload to GPU and run inference
- [ ] Compare output accuracy
- [ ] Performance benchmark

## Code Review Checklist

### Before Submitting Implementation

- [ ] No `unwrap()` or `expect()` in hot paths
- [ ] All integer arithmetic uses `checked_*`
- [ ] All float operations check `is_finite()`
- [ ] Tensor alignment validated upfront
- [ ] Buffer bounds checked before access
- [ ] GPU kernels handle misaligned sizes
- [ ] Reduction loops use `BLOCK_SIZE / 2`
- [ ] All edge cases have explicit tests
- [ ] MXFP bit layout documented
- [ ] Round-trip accuracy measured

### Before Merging

- [ ] All tests passing
- [ ] Code reviewed by 2 people
- [ ] Benchmark results documented
- [ ] Error cases tested
- [ ] Documentation complete
- [ ] CHANGELOG.md updated

## Known Bug Patterns to Avoid

### Pattern 1: Silent Break in Loop

```rust
// WRONG
for block_idx in 0..blocks {
    if block_start + 4 > tensor.data.len() {
        break;  // Silent corruption!
    }
}

// RIGHT
let expected_size = blocks * bytes_per_block;
if tensor.data.len() != expected_size {
    return Err(anyhow!("Invalid data size"));
}

for block_idx in 0..blocks {
    // No break needed
}
```

### Pattern 2: Unchecked Multiplication

```rust
// WRONG
let offset = block_idx * block_size;  // May overflow

// RIGHT
let offset = block_idx.checked_mul(block_size)
    .ok_or_else(|| anyhow!("Overflow"))?;
```

### Pattern 3: Missing NaN Check

```rust
// WRONG
result[i] = quantized * scale;  // If scale=NaN, result=NaN

// RIGHT
let value = quantized * scale;
if !value.is_finite() {
    return Err(anyhow!("Non-finite value"));
}
result[i] = value;
```

### Pattern 4: Wrong Reduction Stride

```cpp
// WRONG
for (int stride = 16; stride > 0; stride >>= 1) { ... }

// RIGHT
for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) { ... }
```

### Pattern 5: Sign Bit Not Applied

```rust
// WRONG
let sign = (bits >> 7) & 1;
let value = compute_magnitude(bits);
// Forgot to apply sign!

// RIGHT
let sign = if (bits >> 7) & 1 == 1 { -1.0 } else { 1.0 };
let magnitude = compute_magnitude(bits);
let value = sign * magnitude;
```

## Quick Validation Commands

```bash
# Run all tests
cargo test --package rocmforge --lib mxfp

# Run with sanitizer (nightly Rust)
cargo +nightly test -Z sanitizer=address --lib mxfp

# Run with valgrind
cargo test --lib mxfp && valgrind ./target/debug/rocmforge-*

# Check for unwrap/expect in new code
grep -n "unwrap()\|expect()" src/loader/mxfp*.rs

# Run clippy
cargo clippy --package rocmforge -- -D warnings

# Format check
cargo fmt --check
```

## MXFP Specification Reference

### MXFP4 (Micro Floating Point 4-bit)
- **Format**: 1-bit sign + 4-bit exponent + 3-bit mantissa
- **Bias**: 7 (exponent range: [-7, +8])
- **Value**: `(-1)^sign * 2^(exp - 7) * (1 + mant/8)`
- **Max Value**: `2^8 * (1 + 7/8) = 480`
- **Min Positive**: `2^-7 * 1 = 0.0078125`
- **Special Encodings**:
  - Zero: exp=0, mant=0 (value=0)
  - May have Inf/NaN encodings (check spec)

### MXFP6 (Micro Floating Point 6-bit)
- **Format**: 1-bit sign + 6-bit exponent OR variable exponent/mantissa split
- **Bias**: Depends on variant (typically 31 for 6-bit exponent)
- **Value**: `(-1)^sign * 2^(exp - bias) * (1 + mant/max_mantissa)`
- **Max Value**: Depends on exact format
- **Min Positive**: Depends on exact format

### Block Size
- **Typical**: 32 or 64 elements per block
- **Scale**: Shared per block (1 float32 or float16)
- **Storage**: Block scale + packed MXFP values

## Bug Report Template

When reporting bugs in MXFP implementation, use this template:

```markdown
### Bug Title

**Severity**: CRITICAL/HIGH/MEDIUM/LOW
**Location**: file:line or function_name
**Type**: Memory Safety / Float Precision / Logic Error / Edge Case

**Description**:
[Brief description of the bug]

**Impact**:
[What goes wrong - crashes, corruption, silent errors]

**Reproduction**:
```rust
// Minimal test case that reproduces the bug
```

**Recommended Fix**:
```rust
// Corrected code
```

**Test Case Added**:
- [ ] Yes - link to test
- [ ] No - needs test
```

## References

- Main Report: `/home/feanor/Projects/ROCmForge/docs/DEBUG_MXFP_IMPLEMENTATION_REPORT.md`
- CHANGELOG (known bugs): `/home/feanor/Projects/ROCmForge/CHANGELOG.md`
- Existing Dequantization: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:765-849`

---

**Next Action**: Wait for Agent 1 to complete implementation, then re-run this checklist.
