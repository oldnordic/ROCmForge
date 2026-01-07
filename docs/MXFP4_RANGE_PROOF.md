# MXFP4 Range Proof: Why Max Value is 8.0, Not 6.0

## Executive Summary

The current implementation clamps MXFP4 values to `[0.5, 6.0]`, but the correct range per OCP MX Specification v1.0 is `[0.5, 8.0]`. This document provides mathematical proof and verification.

---

## 1. OCP MX Specification v1.0

### 1.1 MXFP4 Format Definition

**Format**: E2M1 (Exponent 2-bit, Mantissa 1-bit)
- Bit layout: `[sign:1][exp:2][mant:1]`
- Total: 4 bits

**Formula**:
```
value = (-1)^sign × 2^(exp-1) × (1 + mantissa)
```

Where:
- `sign`: 0 or 1 (bit 3)
- `exp`: exponent bits value (0-3, from bits [2:1])
- `mantissa`: 0 or 1 (bit 0)

### 1.2 Actual Exponent Calculation

The specification states `2^(exp-1)` where `exp` is the **2-bit exponent value**, not the biased exponent.

For exponent bits `[e1, e0]`:
```
exp_bits = e1×2 + e0
actual_exp = exp_bits - 1
```

---

## 2. Proof by Exhaustive Enumeration

Let's enumerate all possible values for E2M1:

### 2.1 All Positive Values

| exp_bits | actual_exp | mant_bits | mantissa | value = 2^(exp-1) × (1+mant) |
|:--------:|:----------:|:---------:|:--------:|:----------------------------:|
| 00 (0)   | -1         | 0         | 0.0      | 2^(-1) × 1.0 = 0.5          |
| 00 (0)   | -1         | 1         | 1.0      | 2^(-1) × 2.0 = 1.0          |
| 01 (1)   | 0          | 0         | 0.0      | 2^(0) × 1.0 = 1.0           |
| 01 (1)   | 0          | 1         | 1.0      | 2^(0) × 2.0 = 2.0           |
| 10 (2)   | 1          | 0         | 0.0      | 2^(1) × 1.0 = 2.0           |
| 10 (2)   | 1          | 1         | 1.0      | 2^(1) × 2.0 = 4.0           |
| 11 (3)   | 2          | 0         | 0.0      | 2^(2) × 1.0 = 4.0           |
| 11 (3)   | 2          | 1         | 1.0      | **2^(2) × 2.0 = 8.0**      |

**Maximum value**: 8.0 (when exp_bits=3, mant=1)

### 2.2 Encoding for Maximum Value

For value = 8.0:
```
sign_bit = 0 (positive)
exp_bits = 3 (binary: 11)
mant_bit = 1 (binary: 1)

encoding = [0][11][1] = 0b0111 = 7
```

Let's verify:
```rust
sign = 1.0 (bit 3 is 0)
exp = ((7 >> 1) & 0x03) - 1 = 3 - 1 = 2
mant = (7 & 0x01) = 1
value = 1.0 × 2^2 × (1 + 1) = 4 × 2 = 8.0 ✅
```

---

## 3. Current Implementation Bug

### 3.1 Bug Location 1: Encoding Clamping

**File**: `src/loader/gguf.rs:220`

```rust
// ❌ WRONG
let clamped = abs.max(0.5).min(6.0);

// ✅ CORRECT
let clamped = abs.max(0.5).min(8.0);
```

**Impact**:
- Input value 7.0 → Clamped to 6.0 → Encoded as exp=2, mant=1 → Decoded as 4.0
- **Error**: (7.0 - 4.0) / 7.0 = 42.9% quantization error!

### 3.2 Bug Location 2: Scale Calculation

**File**: `src/loader/gguf.rs:100`

```rust
// ❌ WRONG
E8M0::from_f32(max_val / 6.0)

// ✅ CORRECT
E8M0::from_f32(max_val / 8.0)
```

**Impact**:
- If `max_val = 7.0`:
  - Current: scale = 7.0 / 6.0 = 1.167 → E8M0(1.167) ≈ E8M0(0) = 2^0 = 1.0
  - Correct: scale = 7.0 / 8.0 = 0.875 → E8M0(0.875) ≈ E8M0(0) = 2^0 = 1.0
  - But wait, both give same scale here...

Actually, the issue is more subtle. Let's recalculate:

If we have values in range [0, 7.0]:
- Current (6.0 divisor): max_val / 6.0 = 7.0 / 6.0 ≈ 1.167 → encoded as ~1.0
  - Normalized values: [0, 7.0] / 1.0 = [0, 7.0]
  - But encoder clamps to 6.0, so values in (6.0, 7.0] get clamped!
- Correct (8.0 divisor): max_val / 8.0 = 7.0 / 8.0 = 0.875 → encoded as ~1.0
  - Normalized values: [0, 7.0] / 1.0 = [0, 7.0]
  - Encoder can represent up to 8.0, so no clamping!

### 3.3 Bug Location 3: Decoding Clamping

**File**: `src/loader/gguf.rs:147`

```rust
// ❌ WRONG
val = val.clamp(-6.0, 6.0);

// ✅ CORRECT
val = val.clamp(-8.0, 8.0);
```

**Impact**:
- Decoded value 8.0 → Clamped to 6.0
- Same 25% error as encoding bug

---

## 4. Mathematical Proof

### 4.1 Derivation of Maximum Value

Given the formula:
```
value = (-1)^sign × 2^(exp_bits-1) × (1 + mantissa)
```

To maximize value:
- `sign = 0` (positive)
- `exp_bits = 3` (maximum 2-bit value)
- `mantissa = 1` (maximum 1-bit value)

```
max_value = 1 × 2^(3-1) × (1 + 1)
           = 1 × 2^2 × 2
           = 4 × 2
           = 8
```

### 4.2 Range Verification

Minimum positive value:
- `exp_bits = 0`, `mantissa = 0`
- `value = 2^(0-1) × 1.0 = 2^(-1) = 0.5` ✅

Maximum value:
- `exp_bits = 3`, `mantissa = 1`
- `value = 2^(3-1) × 2.0 = 2^2 × 2 = 8.0` ✅

**Range**: [0.5, 8.0] for positive values
**Full range**: [-8.0, 8.0] including negative values

---

## 5. Comparison with MXFP6

### 5.1 MXFP6 Format

**Format**: E2M3 (Exponent 2-bit, Mantissa 3-bit)
- Bit layout: `[sign:1][exp:2][mant:3]`
- Total: 6 bits

**Formula**:
```
value = (-1)^sign × 2^(exp-1) × (1 + mantissa/8)
```

### 5.2 MXFP6 Maximum Value

```
exp_bits = 3, mantissa = 7
max_value = 2^(3-1) × (1 + 7/8)
          = 2^2 × (1.875)
          = 4 × 1.875
          = 7.5
```

### 5.3 Comparison Table

| Format | Bits | Min Pos | Max   | Range Size |
|:------:|:----:|:-------:|:-----:|:----------:|
| MXFP4  | 4    | 0.5     | 8.0   | 7.5        |
| MXFP6  | 6    | 0.5     | 7.5   | 7.0        |

**Interesting observation**: MXFP4 has a larger max value (8.0) than MXFP6 (7.5), but MXFP6 has better precision within its range.

---

## 6. Reference Implementation

### 6.1 Correct E2M1 Encoding

```rust
pub fn encode_e2m1(value: f32) -> u8 {
    if value == 0.0 {
        return 0b0000;
    }

    let sign = if value < 0.0 { 0b1000 } else { 0b0000 };
    let abs = value.abs();

    // ✅ CORRECT: 8.0 is the maximum
    let clamped = abs.max(0.5).min(8.0);

    // Brute-force search for best encoding
    let mut best_encoding = 0u8;
    let mut best_error = f32::MAX;

    for exp_bits in 0..4 {
        for mant_bits in 0..2 {
            let exp = exp_bits as i32 - 1;
            let mant = mant_bits as f32;
            let decoded = (1.0 + mant) * 2_f32.powi(exp);

            let error = (clamped - decoded).abs();
            if error < best_error {
                best_error = error;
                best_encoding = (exp_bits << 1) | mant_bits;
            }
        }
    }

    sign | best_encoding
}
```

### 6.2 Correct E2M1 Decoding

```rust
pub fn decode_e2m1(bits: u8) -> f32 {
    if bits == 0 {
        return 0.0;
    }

    let sign = if bits & 0x08 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 1) & 0x03) as i32 - 1;
    let mant = (bits & 0x01) as f32;

    sign * (1.0 + mant) * 2_f32.powi(exp)
}
```

The decoding is **already correct** in the current implementation!

---

## 7. Test Cases

### 7.1 Test Maximum Value

```rust
#[test]
fn test_mxfp4_max_value() {
    let block = MxfpBlock::pack_mxfp4(&[8.0; 32]);
    let unpacked = block.unpack_mxfp4();

    // Should be able to represent 8.0
    assert!((unpacked[0] - 8.0).abs() < f32::EPSILON);
}

#[test]
fn test_mxfp4_range() {
    // Test values at the edge of range
    let test_values = [0.5, 1.0, 2.0, 4.0, 8.0];

    for &val in &test_values {
        let block = MxfpBlock::pack_mxfp4(&[val; 32]);
        let unpacked = block.unpack_mxfp4();

        let error_pct = (unpacked[0] - val).abs() / val * 100.0;
        assert!(
            error_pct < 0.1,
            "Value {}: error {:.3}%",
            val, error_pct
        );
    }
}
```

### 7.2 Test Clamping

```rust
#[test]
fn test_mxfp4_clamping() {
    // Values above 8.0 should be clamped
    let block = MxfpBlock::pack_mxfp4(&[10.0; 32]);
    let unpacked = block.unpack_mxfp4();

    // Should clamp to 8.0
    assert!((unpacked[0] - 8.0).abs() < 0.1);
}
```

---

## 8. Impact Analysis

### 8.1 Quantization Error with Wrong Range

If we clamp to 6.0 instead of 8.0:

| Input Value | Encoded (6.0 clamp) | Decoded | Error   | Error % |
|:-----------:|:-------------------:|:-------:|:-------:|:-------:|
| 6.5         | 6.0                 | 4.0     | 2.5     | 38.5%   |
| 7.0         | 6.0                 | 4.0     | 3.0     | 42.9%   |
| 8.0         | 6.0                 | 4.0     | 4.0     | 50.0%   |

### 8.2 Real LLM Weight Impact

For typical LLM weights (std dev ~0.02-0.1):
- Most weights are in range [-0.5, 0.5]
- Scale factor handles the range
- **Bug impact**: Limited for typical weights, but violates spec

For outlier weights (rare but important):
- Critical weights > 6.0×scale will be incorrectly quantized
- **Impact**: Potential accuracy degradation

---

## 9. Conclusion

### 9.1 Fact Summary

1. ✅ **Formula is correct**: `value = (-1)^sign × 2^(exp-1) × (1 + mant)`
2. ❌ **Range is wrong**: Clamped to 6.0, should be 8.0
3. ✅ **Bit layout is correct**: [sign:1][exp:2][mant:1]
4. ❌ **Scale divisor is wrong**: 6.0, should be 8.0

### 9.2 Fix Required

**3 lines of code** need to change in `src/loader/gguf.rs`:

1. Line 220: `min(6.0)` → `min(8.0)`
2. Line 100: `/ 6.0` → `/ 8.0`
3. Line 147: `clamp(-6.0, 6.0)` → `clamp(-8.0, 8.0)`

### 9.3 Verification

After fixing:
```
exp_bits=3, mant=1 → value = 2^(3-1) × (1+1) = 4 × 2 = 8.0 ✅
```

---

## 10. References

- OCP MX Specification v1.0: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Section 3.1: MXFP4 Format Definition
- Section 3.2: MXFP6 Format Definition

---

**Document Version**: 1.0
**Date**: 2026-01-06
**Author**: Agent 2 (Double Check Agent)
**Status**: Mathematical proof complete
