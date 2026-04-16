# Q4_K Scale/Min Extraction Bug Analysis

**Date:** 2026-04-16  
**Issue:** Q4_K quantization produces garbage output: "Self few souls fewлоinsics..."  
**Function:** `get_scale_min_k4` in `src/cpu/kernels/q4.rs:231`

## Current Implementation (rocmforge)

```rust
fn get_scale_min_k4(j: usize, scales: [u8; 12]) -> (u8, u8) {
    let (d, m);
    if j < 4 {
        d = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        // For j >= 4, we need to be careful about array bounds
        // j+4 can be 8, 9, 10, 11 for j = 4, 5, 6, 7
        // j-4 can be 0, 1, 2, 3 for j = 4, 5, 6, 7
        let j_plus_4 = j + 4;
        let j_minus_4 = j - 4;

        if j_plus_4 < 12 {
            d = (scales[j_plus_4] & 0xF) | ((scales[j_minus_4] >> 6) << 4);
            m = (scales[j_plus_4] >> 4) | ((scales[j] >> 6) << 4);
        } else {
            // Fallback for safety
            d = 0;
            m = 0;
        }
    }
    (d, m)
}
```

## llama.cpp Reference Implementation

```c
static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

## Critical Analysis

### **IMPLEMENTATION COMPARISON: Both implementations are EQUIVALENT**

After careful analysis, the `get_scale_min_k4` function in rocmforge is **functionally identical** to llama.cpp for the valid input range.

**Bit-by-bit comparison:**

For j < 4:
- **rocmforge:** `d = scales[j] & 63`, `m = scales[j + 4] & 63`
- **llama.cpp:** `*d = q[j] & 63`, `*m = q[j + 4] & 63`
- **Result:** ✅ IDENTICAL

For j >= 4:
- **rocmforge:**
  - `d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)`
  - `m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)`

- **llama.cpp:**
  - `*d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)`
  - `*m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)` where `j-0 == j`

- **Result:** ✅ IDENTICAL (since `j-0 == j`)

### Input Range Validation

The function is called with `j` values: 0, 1, 2, 3, 4, 5, 6, 7
- From `get_scale_min_k4(is, self.scales)` where `is = 0, 2, 4, 6`
- From `get_scale_min_k4(is + 1, self.scales)` where `is + 1 = 1, 3, 5, 7`

For j = 7: `j_plus_4 = 11 < 12` ✅ bounds check passes
The bounds check is correct and will never trigger the fallback for valid inputs.

## Root Cause Hypothesis: Scales Array Population

**The bug is NOT in `get_scale_min_k4` - it's in how the scales array is populated from the Q4_K block structure.**

### Hypothesis: Incorrect Scales Array Extraction

The 12-byte `scales` array in the Q4_K block structure may be extracted incorrectly from the raw block data. Let me verify the Q4_K block structure:

**Q4_K Block Structure (llama.cpp):**
```c
typedef struct {
    uint8_t qs[QK_K/2];      // 128 bytes: quantized values
    uint8_t qh[QK_K/4];      // 64 bytes: high bits for Q5_K/Q6_K
    uint8_t scales[QK_K/16]; // 12 bytes: scale/min values
    uint8_t d;               // 2 bytes: FP16 delta
    uint8_t dmin;            // 2 bytes: FP16 delta min
} block_q4_K;
```

The `scales[12]` array should be read from bytes 192-203 in the block (after qs[128] and qh[64]).

### Field Order Comparison

**llama.cpp block_q4_K structure:**
```c
typedef struct {
    ggml_half d;         // 2 bytes: super-block scale
    ggml_half dmin;      // 2 bytes: super-block min scale
    uint8_t scales[12];  // 12 bytes: quantized scales/mins
    uint8_t qs[128];     // 128 bytes: 4-bit quantized values
} block_q4_K;
```

**rocmforge BlockQ4K structure:**
```rust
#[repr(C, align(16))]
pub struct BlockQ4K {
    pub d: [u8; 2],      // 2 bytes: FP16 scale
    pub dmin: [u8; 2],   // 2 bytes: FP16 min scale
    pub scales: [u8; 12],// 12 bytes: quantized scales/mins
    pub qs: [u8; 128],   // 128 bytes: 4-bit quantized values
}
```

**Field order: ✅ IDENTICAL**

Both structures have the same field order and memory layout. The `#[repr(C)]` attribute ensures correct alignment.

### Potential Issues

1. **Byte ordering:** FP16 values (d, dmin) may need byte swapping
2. **Array offset:** Scales array may be read from wrong offset
3. **Block alignment:** Padding bytes may be incorrectly skipped
4. **Endianness:** Multi-byte fields may be misinterpreted
5. **Scale calculation:** The way scales are extracted from the 12-byte array might be wrong

### Detailed Investigation

Let me examine how the scales array is actually used. Looking at the `embed_q4_k` function:

```rust
// From q4_k.rs:60-94
pub fn embed_q4_k(token_id: usize, emb: &[u8], out: &mut [f32], hidden_size: usize) {
    let block = &emb[row_offset + b * Q4_K_BLOCK_BYTES..];
    let d = load_f16_scale(&block[0..2]);        // Bytes 0-1: d
    let dmin = load_f16_scale(&block[2..4]);      // Bytes 2-3: dmin
    let scales = &block[4..16];                   // Bytes 4-15: scales[12]
    let mut qs = &block[16..144];                 // Bytes 16-143: qs[128]
```

This matches the structure layout! The scales array is correctly extracted from bytes 4-15.

### CRITICAL DISCOVERY: Two Different Implementations

I found **TWO** different implementations of the same function:

1. **`src/cpu/kernels/q4.rs:231`** - Used by dequantization
2. **`src/cpu/quant/q4_k.rs:12`** - Used by embedding lookup

Let me compare them:

**Implementation 1 (q4.rs:231):**
```rust
fn get_scale_min_k4(j: usize, scales: [u8; 12]) -> (u8, u8) {
    let (d, m);
    if j < 4 {
        d = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        let j_plus_4 = j + 4;
        let j_minus_4 = j - 4;
        if j_plus_4 < 12 {
            d = (scales[j_plus_4] & 0xF) | ((scales[j_minus_4] >> 6) << 4);
            m = (scales[j_plus_4] >> 4) | ((scales[j] >> 6) << 4);
        } else {
            d = 0; m = 0;
        }
    }
    (d, m)
}
```

**Implementation 2 (q4_k.rs:12):**
```rust
pub fn get_scale_min_k4(scales: &[u8], j: usize) -> (u8, u8) {
    debug_assert!(scales.len() >= 12);
    debug_assert!(j < 8);
    
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}
```

**Both implementations are IDENTICAL in logic!** The only difference is:
- Implementation 1 uses bounds checking and returns (0, 0) for invalid j
- Implementation 2 uses debug_assertions

### ROOT CAUSE HYPOTHESIS

Since both implementations are correct and match llama.cpp, the bug must be in:

1. **How the scales array is populated from the model file** (byte ordering issue)
2. **How the quantized weights (qs) are interpreted** (bit packing issue)
3. **The dequantization formula itself** (mathematical error)

### Recommended Investigation

1. **Add debug logging:** Log raw scales bytes and extracted (d, m) values
2. **Compare with llama.cpp:** Run identical blocks through both implementations
3. **Check FP16 loading:** Verify `load_f16_scale` correctly handles endianness
4. **Test with known values:** Create a test with known input/output pairs

## Recommendation

The `get_scale_min_k4` implementations appear equivalent for valid inputs. I recommend:

1. **Verify scales array population:** Check how the 12-byte scales array is extracted from the Q4_K block
2. **Add debug output:** Log the raw scales array and extracted (d, m) values for comparison with llama.cpp
3. **Check calling code:** Verify that j is always in the range [0, 7] when this function is called

## Next Steps

Before modifying the `get_scale_min_k4` function, we should:
1. Verify the scales array is being read correctly from the Q4_K block structure
2. Add debug logging to compare actual values with llama.cpp
3. Check if there are other differences in the dequantization pipeline

**The bug might not be in this function at all - it could be in how the scales array is populated from the block structure.**