# AMD MXFP4/MXFP6 Quantization Analysis

**Date**: 2026-01-20
**Source**: AMD ROCm Blogs - "High-Accuracy MXFP4, MXFP6, and Mixed-Precision Models on AMD GPUs"
**URL**: https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html
**Purpose**: Understanding AMD's low-bit quantization formats for LLM inference

---

## Executive Summary

AMD's MXFP4 and MXFP6 are **OCP (Open Compute Project) Microscaling Formats** - open, vendor-neutral standards for reduced-precision AI. These formats are **natively supported on AMD Instinct MI355 GPUs**, delivering up to **4× higher peak throughput** compared to FP16.

**Key Hardware Support**: AMD MI355X GPUs with CDNA4 architecture

---

## Part 1: MXFP Format Specifications

### MXFP4-E2M1 Format

| Property | Value |
|----------|-------|
| Element Type | FP4 (E2M1) |
| Bits | 4 |
| Range | [-6, 6] |
| Block Size | 32 elements |
| Scale Type | E8M0 (2^n) |
| Data Structure | 4-bit element + 1 shared E8M0 scale per 32 elements |

### MXFP6-E2M3 Format (Recommended)

| Property | Value |
|----------|-------|
| Element Type | FP6 (E2M3) |
| Bits | 6 |
| Range | [-7.5, 7.5] |
| Block Size | 32 elements |
| Scale Type | E8M0 (2^n) |
| Note | **Better accuracy than E3M2** for LLMs |

### MXFP6-E3M2 Format

| Property | Value |
|----------|-------|
| Element Type | FP6 (E3M2) |
| Bits | 6 |
| Range | [-28.0, 28.0] |
| Block Size | 32 elements |
| Scale Type | E8M0 (2^n) |
| Note | Wider range but typically worse accuracy than E2M3 |

### Comparison Table

| Format | Element Type | Bits | Range | Block Size | Scale Type |
|--------|--------------|------|-------|------------|------------|
| MXFP4 | FP4 (E2M1) | 4 | [-6, 6] | 32 | E8M0 (2^n) |
| MXFP6 | FP6 (E2M3) | 6 | [-7.5, 7.5] | 32 | E8M0 (2^n) |
| FP6 (alternative) | FP6 (E3M2) | 6 | [-28.0, 28.0] | 32 | E8M0 (2^n) |

---

## Part 2: Quantization Process

### Three Key Steps

1. **Scaling**: Each block of 32 values shares a scale factor (E8M0)
2. **Clipping**: Values outside the representable range are clipped
3. **Rounding**: Values are rounded to the nearest representable value

### MXFP4 Quantization Formula

```
x_q^MXFP4 = Rounding_E2M1( clip( x/scale, -6, 6 ) )
```

### MXFP6-E2M3 Quantization Formula

```
x_q^MXFP6-E2M3 = Rounding_E2M3( clip( x/scale, -7.5, 7.5 ) )
```

### Critical: Scale Rounding with RNE

The scale calculation **must use Round-to-Nearest-Even (Banker's rounding)**:

```
scale = 2^( clip( floor(log2_RNE(max_abs(x))), -127, 127 ) - 2 )
```

**Why RNE matters**:
- Plain floor rounding **underestimates** the numerical range
- Floor rounding introduces **negative bias**
- RNE reduces both bias and quantization noise
- **Omission of RNE directly degrades model accuracy**

### Accuracy Impact of RNE

From AMD's testing (Figure 3 in blog):
- RNE rounding: **significantly better results**
- Floor rounding: systematic accuracy degradation

---

## Part 3: Performance on AMD MI355

### FLOPS Speedup vs FP16

| Format | Speedup vs FP16 |
|--------|-----------------|
| FP4 | Up to 4× |
| FP6 | Up to 4× |
| FP8 | ~2-3× |

**Theoretical FLOPS for MI355X**:
- Frequency: 2500 MHz
- SIMDs: 192 (48 WGPs × 2 CUs × 2 SIMDs)
- FLOP/cycle: 128 (32-wide SIMD × 2 VALU × 2 FMA)
- **Total: 61.44 TFLOP/s**

### MXFP6 = MXFP4 on AMD MI GPUs

**Critical insight**: MXFP6 has the **same computational FLOPs** as MXFP4 on AMD MI GPUs, but offers better accuracy.

---

## Part 4: Accuracy Results on Large Models

### Models Tested

| Model | Size | Architecture |
|-------|------|--------------|
| DeepSeek-R1 | 528B | MoE |
| Llama-3.1 | 405B | Dense |
| Llama-3.3 | 70B | Dense |
| gpt-oss | 120B | MoE |

### Key Findings

1. **MXFP4**: Highly effective for **super-large models** (>70B)
2. **MXFP6**: Outperforms MXFP4 on **smaller models** (70B and below)
3. **Mixed Precision**: MXFP4+MXFP6 combinations provide best accuracy/efficiency trade-off

### DeepSeek-R1-0528 Results

| Metric | MXFP4 | MXFP6 | MXFP4-MXFP6 Mixed |
|--------|-------|-------|-------------------|
| AIME24 | >99.5% | Higher | Highest |
| GPQA Diamond | >99.5% | Higher | Highest |
| MATH-500 | >99.5% | Higher | Highest |

**Pattern**: MXFP6 > MXFP4-MXFP6 > MXFP4 for accuracy

### Llama-3.3-70B Results

- **MXFP4**: Noticeable accuracy degradation on smaller models
- **MXFP6**: Mitigates accuracy loss
- **Mixed precision**: Consistently outperforms MXFP4 alone

---

## Part 5: AMD Quark Toolkit

### What is Quark?

**AMD Quark** is AMD's model optimization toolkit for:
- Cross-platform optimized low-bit model deployment
- MXFP4/MXFP6 quantization with high accuracy retention
- Integration with advanced PTQ algorithms:
  - **GPTQ**
  - **SmoothQuant**
  - **AutoSmoothQuant** (optimized variant of SmoothQuant)
  - **Quarot**

### Deployment Compatibility

Quark-quantized models work with:
- **vLLM** (production inference)
- **SGLang** (production inference)
- AMD Instinct MI355 GPUs (native FP4/FP6 support)

---

## Part 6: E8M0 Scale Format

### Structure

```rust
#[repr(C)]
struct E8M0 {
    exponent: i8,  // 8-bit signed exponent
}
```

### Conversion

```rust
impl E8M0 {
    pub fn to_f32(&self) -> f32 {
        2.0_f32.powi(self.exponent as i32)
    }

    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 || value.is_nan() {
            return E8M0 { exponent: 0 };
        }
        if value.is_infinite() {
            return E8M0 { exponent: 127 };
        }
        let abs_val = value.abs();
        let exp = abs_val.log2().clamp(-127.0, 127.0).round() as i8;
        E8M0 { exponent: exp }
    }
}
```

### Key Properties

- Represents scale factor as `2^exponent`
- Range: `2^(-127)` to `2^(127)`
- Used as **shared scale** for blocks of 32 MXFP4/MXFP6 values

---

## Part 7: Implementation Requirements for ROCmForge

### For MXFP Support

1. **E8M0 scale handling** with RNE rounding
2. **Block-based processing** (32 elements per block)
3. **Clipping** to valid range before quantization
4. **Proper rounding** (not floor) for scale calculation

### For ROCmForge Architecture

```
src/kernels/
├── quantization/
│   ├── mod.rs
│   ├── mxfp4.rs        # MXFP4-E2M1 encode/decode
│   ├── mxfp6.rs        # MXFP6-E2M3 encode/decode
│   ├── e8m0.rs         # E8M0 scale handling
│   └── k_quants.rs     # K-quants (Q4_K, Q6_K, etc.)
```

### MXFP Block Structure

```rust
pub struct MxfpBlock {
    pub scale: E8M0,           // 1 byte per 32 elements
    pub elements: Vec<u8>,     // 32 elements * 4 bits = 16 bytes
    pub format: MxfpFormat,     // E2M1 or E2M3
}

pub enum MxfpFormat {
    Mxfp4E2m1,     // 4-bit, range [-6, 6]
    Mxfp6E2m3,     // 6-bit, range [-7.5, 7.5]
    Mxfp6E3m2,     // 6-bit, range [-28, 28]
}
```

---

## Part 8: Practical Considerations

### When to Use MXFP

| Scenario | Recommendation |
|----------|----------------|
| Models >100B | MXFP4 sufficient |
| Models 30-100B | MXFP6 recommended |
| Models <30B | MXFP6 or mixed precision |
| Accuracy-critical | MXFP6 or mixed |
| Memory-constrained | MXFP4 (smaller) |

### Current Model Support

AMD has publicly released MXFP4 models:
- DeepSeek-R1-0528
- Llama-3.1-405B
- Llama-3.3-70B
- More coming soon

---

## Part 9: Relationship to ROCmForge

### Current ROCmForge Status

ROCmForge already has MXFP definitions in:
- `src/loader/gguf.rs` (lines 78-238)
- `src/loader/mxfp.rs` (358 lines)
- `src/loader/dequant.rs`

### Issues Identified

1. **Duplicate code**: `MxfpBlock` defined in both `gguf.rs` and `mxfp.rs`
2. **RNE rounding**: Not explicitly mentioned in current implementation
3. **MXFP not production-ready**: Considered experimental

### Recommendations

1. **Consolidate MXFP code** into single location
2. **Add RNE rounding** for scale calculation
3. **Prioritize K-quants** (Q4_K, Q6_K) for current models
4. **Keep MXFP as future-ready** but not primary focus

---

## Part 10: Key Takeaways

1. **MXFP is OCP standard** - vendor-neutral, open
2. **Native MI355 support** - 4× speedup over FP16
3. **RNE rounding critical** - omission degrades accuracy
4. **MXFP6 > MXFP4 for accuracy** - same FLOPs on AMD
5. **Large models benefit most** - MXFP4 nearly lossless for >70B
6. **Quark toolkit** - AMD's official quantization tool

---

**End of Report** - Investigation only, no code changes.
