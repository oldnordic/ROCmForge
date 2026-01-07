# MXFP Bit-Level Verification Checklist

**Purpose**: Exact bit-for-bit verification against OCP MX Specification v1.0
**Use When**: Agent 1 completes MXFP implementation
**Verification Method**: Line-by-line code review against this checklist

---

## MXFP4 (E2M1) - Bit Layout Verification

### Specification

```
MXFP4 4-bit encoding (E2M1):
Bit 3:     Sign (0 = positive, 1 = negative)
Bits 2-1:  Exponent (2 bits, bias = 1)
Bit 0:     Mantissa (1 bit)

Value formula:  (-1)^sign * 2^(exp-1) * (1.mantissa)
Range:          [-6.0, 6.0]
Special case:   exp=0 AND mant=0 → value = 0.0
```

### Kernel Code Verification

**File**: `kernels/mxfp_dequant.hip`

**Function**: `mxfp4_to_float(uint8_t bits)`

```cpp
// STEP 1: Extract bits
// CORRECT:
const uint32_t sign = (bits >> 3) & 0x01;   // ✓ Bit 3
const uint32_t exp  = (bits >> 1) & 0x03;   // ✓ Bits 1-2
const uint32_t mant = bits & 0x01;           // ✓ Bit 0

// WRONG (common errors):
const uint32_t sign = bits & 0x01;           // ✗ Wrong bit position
const uint32_t exp  = (bits >> 2) & 0x03;    // ✗ Wrong bit position
const uint32_t mant = (bits >> 1) & 0x01;    // ✗ Wrong bit position
```

**Verification**:
- [ ] Sign extracted from bit 3: `(bits >> 3) & 0x01`
- [ ] Exponent extracted from bits 1-2: `(bits >> 1) & 0x03`
- [ ] Mantissa extracted from bit 0: `bits & 0x01`

```cpp
// STEP 2: Special case check
// CORRECT:
if (exp == 0 && mant == 0) return 0.0f;     // ✓ Zero check

// WRONG:
if (exp == 0) return 0.0f;                  // ✗ Misses mantissa check
if (mant == 0) return 0.0f;                 // ✗ Misses exponent check
```

**Verification**:
- [ ] Zero check: `if (exp == 0 && mant == 0) return 0.0f;`

```cpp
// STEP 3: Compute value
// CORRECT:
float significand = 1.0f + (float)mant;     // ✓ 1.mantissa (NOT mant/X)
float exponent = (float)exp - 1.0f;         // ✓ Bias = 1
float value = ldexpf(significand, (int)exponent);

// WRONG:
float significand = (float)mant;            // ✗ Missing implicit 1
float significand = 1.0f + (float)mant / X; // ✗ No division for 1-bit mant
float exponent = (float)exp;                // ✗ Missing bias
float exponent = (float)exp - 2.0f;         // ✗ Wrong bias
```

**Verification**:
- [ ] Significand: `1.0f + (float)mant` (NOT `mant / X`)
- [ ] Exponent bias: `- 1.0f` (NOT `- 0.0f` or `- 2.0f`)
- [ ] Uses `ldexpf()` for power-of-two computation

```cpp
// STEP 4: Apply sign
// CORRECT:
return sign ? -value : value;               // ✓ Apply sign

// WRONG:
return value * (1.0f - 2.0f * sign);        // ✗ Overly complex
```

**Verification**:
- [ ] Sign application: `sign ? -value : value`

```cpp
// STEP 5: Range clipping
// CORRECT:
value = fmaxf(-6.0f, fminf(6.0f, value));   // ✓ Clip to [-6.0, 6.0]

// WRONG:
value = fmaxf(-7.5f, fminf(7.5f, value));   // ✗ MXFP6 range, not MXFP4
```

**Verification**:
- [ ] Range clipping: `fmaxf(-6.0f, fminf(6.0f, value))`

### Test Vectors

```rust
#[test]
fn test_mxfp4_decode() {
    // Test all 16 possible MXFP4 values
    let test_cases = [
        (0b0000, 0.0),      // exp=0, mant=0, sign=0 → 0.0
        (0b0001, 1.0),      // exp=1, mant=1, sign=0 → 2^0 * 1.1 = 1.0
        (0b0010, 2.0),      // exp=2, mant=0, sign=0 → 2^1 * 1.0 = 2.0
        (0b0011, 3.0),      // exp=2, mant=1, sign=0 → 2^1 * 1.1 = 3.0
        (0b1000, 0.0),      // exp=0, mant=0, sign=1 → -0.0 = 0.0
        (0b1001, -1.0),     // exp=1, mant=1, sign=1 → -2^0 * 1.1 = -1.0
        // ... etc
    ];

    for (bits, expected) in test_cases {
        let decoded = mxfp4_to_float(bits);
        assert!((decoded - expected).abs() < 0.01);
    }
}
```

**Verification**:
- [ ] Unit test covers all 16 possible 4-bit values
- [ ] Special case (exp=0, mant=0) tested
- [ ] Positive values tested
- [ ] Negative values tested
- [ ] Boundary values tested

---

## MXFP6-E2M3 - Bit Layout Verification

### Specification

```
MXFP6-E2M3 6-bit encoding:
Bit 5:     Sign (0 = positive, 1 = negative)
Bits 4-3:  Exponent (2 bits, bias = 1)
Bits 2-0:  Mantissa (3 bits)

Value formula:  (-1)^sign * 2^(exp-1) * (1.mantissa/8)
Range:          [-7.5, 7.5]
Special case:   exp=0 AND mant=0 → value = 0.0
```

### Kernel Code Verification

**File**: `kernels/mxfp_dequant.hip`

**Function**: `mxfp6_e2m3_to_float(uint8_t bits)`

```cpp
// STEP 1: Extract bits
// CORRECT:
const uint32_t sign = (bits >> 5) & 0x01;   // ✓ Bit 5
const uint32_t exp  = (bits >> 3) & 0x03;   // ✓ Bits 3-4
const uint32_t mant = bits & 0x07;           // ✓ Bits 0-2

// WRONG:
const uint32_t sign = (bits >> 3) & 0x01;   // ✗ Wrong position (MXFP4 layout)
const uint32_t exp  = (bits >> 2) & 0x07;   // ✗ Wrong position (E3M2 layout)
```

**Verification**:
- [ ] Sign extracted from bit 5: `(bits >> 5) & 0x01`
- [ ] Exponent extracted from bits 3-4: `(bits >> 3) & 0x03`
- [ ] Mantissa extracted from bits 0-2: `bits & 0x07`

```cpp
// STEP 2: Special case check
// CORRECT:
if (exp == 0 && mant == 0) return 0.0f;     // ✓ Zero check
```

**Verification**:
- [ ] Zero check: `if (exp == 0 && mant == 0) return 0.0f;`

```cpp
// STEP 3: Compute value
// CORRECT:
float significand = 1.0f + (float)mant / 8.0f;  // ✓ 1.mantissa/8
float exponent = (float)exp - 1.0f;             // ✓ Bias = 1
float value = ldexpf(significand, (int)exponent);

// WRONG:
float significand = 1.0f + (float)mant;     // ✗ Missing /8.0f
float significand = 1.0f + (float)mant / 4.0f; // ✗ Wrong divisor
float significand = 1.0f + (float)mant / 16.0f; // ✗ Wrong divisor
```

**Verification**:
- [ ] Significand: `1.0f + (float)mant / 8.0f` (NOT `mant / 4.0f` or `mant / 16.0f`)
- [ ] Exponent bias: `- 1.0f`
- [ ] Uses `ldexpf()` for power-of-two computation

```cpp
// STEP 4: Range clipping
// CORRECT:
value = fmaxf(-7.5f, fminf(7.5f, value));   // ✓ Clip to [-7.5, 7.5]

// WRONG:
value = fmaxf(-6.0f, fminf(6.0f, value));   // ✗ MXFP4 range, not MXFP6
```

**Verification**:
- [ ] Range clipping: `fmaxf(-7.5f, fminf(7.5f, value))`

### Test Vectors

```rust
#[test]
fn test_mxfp6_e2m3_decode() {
    // Test representative 6-bit values
    let test_cases = [
        (0b000000, 0.0),      // exp=0, mant=0 → 0.0
        (0b001000, 1.0),      // exp=1, mant=0 → 2^0 * 1.0 = 1.0
        (0b001111, 1.875),    // exp=1, mant=7 → 2^0 * 1.875 = 1.875
        (0b010000, 2.0),      // exp=2, mant=0 → 2^1 * 1.0 = 2.0
        (0b010111, 3.75),     // exp=2, mant=7 → 2^1 * 1.875 = 3.75
        (0b100000, 0.0),      // exp=0, mant=0, sign=1 → -0.0 = 0.0
        (0b101000, -1.0),     // exp=1, mant=0, sign=1 → -2^0 * 1.0 = -1.0
        // ... etc
    ];

    for (bits, expected) in test_cases {
        let decoded = mxfp6_e2m3_to_float(bits);
        assert!((decoded - expected).abs() < 0.01);
    }
}
```

**Verification**:
- [ ] Unit test covers representative 6-bit values
- [ ] Special case (exp=0, mant=0) tested
- [ ] Mantissa division by 8 verified
- [ ] Boundary values tested

---

## MXFP6-E3M2 - Bit Layout Verification

### Specification

**STATUS**: NOT FULLY SPECIFIED in MXFP_QUANTIZATION_ANALYSIS.md

```
MXFP6-E3M2 6-bit encoding (LIKELY - verify with AMD docs):
Bit 5:     Sign (0 = positive, 1 = negative)
Bits 4-2:  Exponent (3 bits, bias = ?)
Bits 1-0:  Mantissa (2 bits)

Value formula:  (-1)^sign * 2^(exp-?) * (1.mantissa/?)
Range:          ? (NOT SPECIFIED)
Special case:   exp=0 AND mant=0 → value = 0.0
```

**ACTION REQUIRED**: Check AMD Quark source code for:
1. Exponent bias (likely 3, but must verify)
2. Mantissa divisor (likely 4, but must verify)
3. Range (not specified anywhere)

### Kernel Code Verification (If Implemented)

**File**: `kernels/mxfp_dequant.hip`

**Function**: `mxfp6_e3m2_to_float(uint8_t bits)`

```cpp
// LIKELY (MUST VERIFY WITH AMD DOCS):
const uint32_t sign = (bits >> 5) & 0x01;   // ✓ Bit 5
const uint32_t exp  = (bits >> 2) & 0x07;   // ✓ Bits 2-4
const uint32_t mant = bits & 0x03;           // ✓ Bits 0-1

float significand = 1.0f + (float)mant / 4.0f;  // ⚠️ Verify divisor
float exponent = (float)exp - 3.0f;             // ⚠️ Verify bias
float value = ldexpf(significand, (int)exponent);

// ⚠️ RANGE NOT SPECIFIED - must check AMD docs
```

**Verification** (IF implemented):
- [ ] Sign extracted from bit 5: `(bits >> 5) & 0x01`
- [ ] Exponent extracted from bits 2-4: `(bits >> 2) & 0x07`
- [ ] Mantissa extracted from bits 0-1: `bits & 0x03`
- [ ] Significand divisor verified with AMD docs: `1.0f + mant / X`
- [ ] Exponent bias verified with AMD docs: `exp - X`
- [ ] Range clipping verified with AMD docs
- [ ] Reference: AMD Quark GitHub source code

---

## E8M0 Scale Format Verification

### Specification

```
E8M0 8-bit exponent-only format:
Bit 7:     Sign of exponent (0 = positive, 1 = negative)
Bits 6-0:  Exponent magnitude (7 bits)

Value formula:  2^exponent  (where exponent is signed 8-bit)
Range:          2^-127 to 2^+127
Special cases:  exponent = -128 → 0.0 (underflow)
                exponent = +127 → Inf (overflow)
```

### Rust Code Verification

**File**: `src/loader/mxfp.rs`

```rust
// CORRECT:
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct E8M0 {
    exponent: i8,  // ✓ Signed 8-bit exponent
}

impl E8M0 {
    pub fn to_f32(&self) -> f32 {
        // ✓ Direct power-of-two computation
        2.0_f32.powi(self.exponent as i32)
    }

    pub fn from_f32(value: f32) -> Self {
        // ✓ Log2 conversion with clamping
        let exp = value.log2().clamp(-127.0, 127.0) as i8;
        E8M0 { exponent: exp }
    }
}

// WRONG:
pub struct E8M0 {
    sign: bool,      // ✗ E8M0 has no explicit sign bit
    exponent: u8,    // ✗ Should be signed, not unsigned
}

pub fn to_f32(&self) -> f32 {
    ldexpf(1.0, self.exponent)  // ✗ Less accurate than powi()
}
```

**Verification**:
- [ ] E8M0 struct: `exponent: i8` (signed 8-bit)
- [ ] No separate sign field (E8M0 has NO mantissa bits)
- [ ] `to_f32()`: Returns `2.0_f32.powi(exponent as i32)`
- [ ] `from_f32()`: Computes `value.log2().clamp(-127.0, 127.0)`
- [ ] Clamping to prevent overflow/underflow

### Test Vectors

```rust
#[test]
fn test_e8m0_roundtrip() {
    let test_cases = [
        (1.0, 0),      // 2^0 = 1.0
        (2.0, 1),      // 2^1 = 2.0
        (0.5, -1),     // 2^-1 = 0.5
        (4.0, 2),      // 2^2 = 4.0
        (0.25, -2),    // 2^-2 = 0.25
        (128.0, 7),    // 2^7 = 128.0
    ];

    for (value, expected_exp) in test_cases {
        let e8m0 = E8M0::from_f32(value);
        assert_eq!(e8m0.exponent, expected_exp);

        let decoded = e8m0.to_f32();
        assert!((decoded - value).abs() < 0.001);
    }
}
```

**Verification**:
- [ ] Round-trip test: `f32 → E8M0 → f32`
- [ ] Powers of two tested (2^n)
- [ ] Fractional values tested (2^-n)
- [ ] Clamping at boundaries tested

---

## Block Packing Verification

### MXFP4 Block Layout

```
Block size: 32 elements
Layout:     [Scale: 1 byte E8M0] [Element 0-1: 1 byte] [Element 2-3: 1 byte] ...
Total:      1 + (32 * 4 / 8) = 1 + 16 = 17 bytes
```

### Kernel Code Verification

**File**: `kernels/mxfp_dequant.hip`

```cpp
// CORRECT:
const int block_idx = idx / 32;           // ✓ Block index
const int elem_idx = idx % 32;            // ✓ Element within block

// Load scale (first byte in block)
const int8_t scale_exp = ((int8_t*)mxfp4_data)[block_idx * 17];
const float scale = __exp2f((float)scale_exp);  // ✓ E8M0 → float

// Load element (4-bit packed, 2 per byte)
const uint8_t packed = mxfp4_data[block_idx * 17 + 1 + elem_idx / 2];
const uint8_t elem_4bit = (elem_idx % 2 == 0)
    ? (packed >> 4)      // ✓ High nibble (element 0, 2, 4, ...)
    : (packed & 0x0F);   // ✓ Low nibble (element 1, 3, 5, ...)

// WRONG:
const uint8_t elem_4bit = (elem_idx % 2 == 0)
    ? (packed & 0x0F)    // ✗ Swapped nibble order
    : (packed >> 4);
```

**Verification**:
- [ ] Block size: 32 elements (NOT 64)
- [ ] Scale position: `block_idx * 17 + 0` (first byte)
- [ ] Data position: `block_idx * 17 + 1 + elem_idx / 2`
- [ ] Even elements: High nibble `(packed >> 4)`
- [ ] Odd elements: Low nibble `(packed & 0x0F)`

### MXFP6 Block Layout

```
Block size: 32 elements
Layout:     [Scale: 1 byte E8M0] [Element 0-1: 2 bytes] [Element 2-3: 2 bytes] ...
            (6-bit elements cross byte boundaries)
Total:      1 + (32 * 6 / 8) = 1 + 24 = 25 bytes
```

### Kernel Code Verification

```cpp
// CORRECT (from TODO.md):
const int byte_idx = 1 + elem_idx * 6 / 8;     // ✓ Byte offset
const int bit_offset = (elem_idx * 6) % 8;     // ✓ Bit offset within byte

uint16_t elem_6bit;
if (bit_offset <= 2) {
    // ✓ 6 bits fit in one byte (bits 0-5 starting at bit_offset)
    elem_6bit = (mxfp6_data[byte_idx] >> bit_offset) & 0x3F;
} else {
    // ✓ 6 bits cross byte boundary
    elem_6bit = (mxfp6_data[byte_idx] >> bit_offset) & 0x3F;
    elem_6bit |= (mxfp6_data[byte_idx + 1] << (8 - bit_offset)) & 0x3F;
}

// WRONG:
const uint8_t elem_6bit = mxfp6_data[byte_idx] & 0x3F;  // ✗ Ignores bit_offset
```

**Verification**:
- [ ] Block size: 32 elements (NOT 64)
- [ ] Scale position: `block_idx * 25 + 0` (first byte)
- [ ] Byte offset: `1 + elem_idx * 6 / 8`
- [ ] Bit offset: `(elem_idx * 6) % 8`
- [ ] Handles cross-byte boundary correctly
- [ ] Masks with `0x3F` (6 bits)

---

## Accuracy Validation Verification

### Dequantization Error Test

```rust
#[test]
fn test_dequantization_accuracy() {
    // Generate random test values
    let mut rng = rand::thread_rng();
    let test_values: Vec<f32> = (0..1000)
        .map(|_| rng.gen_range(-5.0..5.0))
        .collect();

    // Quantize to MXFP6
    let mxfp6_data = quantize_to_mxfp6(&test_values);

    // Dequantize back to FP32
    let dequantized = dequantize_from_mxfp6(&mxfp6_data);

    // Compute relative error
    let max_error = test_values.iter()
        .zip(dequantized.iter())
        .map(|(orig, deq)| ((orig - deq) / orig).abs())
        .fold(0.0f32, f32::max);

    // MXFP6 requirement: <0.1% error
    assert!(max_error < 0.001, "MXFP6 error: {}%", max_error * 100.0);
}
```

**Verification**:
- [ ] Test uses random values (not just edge cases)
- [ ] Tests full quantization/dequantization pipeline
- [ ] Measures relative error (not absolute)
- [ ] MXFP6 requirement: `< 0.001` (0.1%)
- [ ] MXFP4 requirement: `< 0.002` (0.2%)

### Perplexity Validation

```bash
# Install evaluation harness
pip install lm-eval[api]

# Run perplexity test on quantized model
lm_eval --model vllm \
    --model_args pretrained=/models/Llama-70B-MXFP6 \
    --tasks wikitext \
    --batch_size auto

# Compare to FP16 baseline
# FP16 baseline: 10.54
# MXFP6 target: < 10.65 (0.1% increase = 0.0105)
```

**Verification**:
- [ ] Perplexity tested on standard benchmark (wikitext, lambada)
- [ ] Compared to FP16 baseline
- [ ] MXFP6: < 0.1% increase
- [ ] MXFP4: < 0.2% increase

---

## Final Verification Checklist

### Before Approving Implementation

- [ ] All MXFP4 bit layout checks pass
- [ ] All MXFP6-E2M3 bit layout checks pass
- [ ] All MXFP6-E3M2 bit layout checks pass (if implemented)
- [ ] All E8M0 format checks pass
- [ ] All block packing checks pass
- [ ] All unit tests pass (100% pass rate)
- [ ] Test coverage > 80%
- [ ] Accuracy validation passes (MXFP6: <0.1%, MXFP4: <0.2%)
- [ ] Perplexity validation passes (MXFP6: <0.1% increase)
- [ ] No CRITICAL discrepancies found
- [ ] No HIGH priority discrepancies found
- [ ] All MEDIUM issues documented and acceptable

### Sign-Off Criteria

**Implementation approved when ALL of**:
1. Zero critical bit layout errors
2. Zero high priority discrepancies
3. Accuracy validation passes
4. Unit tests pass with 100% success rate
5. Integration tests pass
6. Code reviewed against this checklist

**Implementation rejected if ANY of**:
1. Wrong bit layout (data corruption risk)
2. Wrong block size (memory corruption risk)
3. Wrong exponent bias (scaling errors)
4. Missing special case handling (NaN propagation)
5. Accuracy validation fails (>0.2% error)
6. Zero test coverage

---

**End of Bit-Level Verification Checklist**

**Usage**: Print this checklist and verify line-by-line against Agent 1's implementation
**Reference**: OCP MX Specification v1.0, AMD Quark source code
