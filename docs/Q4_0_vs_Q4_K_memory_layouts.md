# Q4_0 vs Q4_K Memory Layout Comparison

## Overview

This document provides a detailed comparison of Q4_0 and Q4_K quantization formats, focusing on their memory layouts, compression efficiency, and use cases.

## Format Specifications

### Q4_0 Format

**Block Size:** 32 elements, 18 bytes

**Layout:**
- `d`: f16 scale (2 bytes)
- `qs`: 16 bytes of 4-bit quantized values (32 elements / 2 per byte)

**Compression Ratio:**
- Original: 32 × 4 bytes = 128 bytes
- Compressed: 18 bytes
- Ratio: 7.11× compression
- Bits per value: 4.5 bits

**Characteristics:**
- Simple, uniform quantization
- Single scale per block
- Best for: Small embeddings, fast inference
- Block alignment: 32 elements

### Q4_K Format

**Block Size:** 256 elements, 144 bytes

**Layout:**
- `d`: f16 scale (2 bytes)
- `dmin`: f16 minimum value (2 bytes)
- `scales`: 12 uint8_t scales (12 bytes)
- `qs`: 128 uint8_t quantized nibbles (128 bytes)

**Compression Ratio:**
- Original: 256 × 4 bytes = 1024 bytes
- Compressed: 144 bytes
- Ratio: 7.11× compression
- Bits per value: 4.5 bits

**Characteristics:**
- Non-uniform quantization with multiple scales
- Better compression for varied data distributions
- Best for: Large embeddings, higher accuracy requirements
- Block alignment: 256 elements

## Key Differences

### 1. Block Size

| Aspect | Q4_0 | Q4_K |
|--------|------|------|
| Elements per block | 32 | 256 |
| Block size | 18 bytes | 144 bytes |
| Alignment | 32-element alignment | 256-element alignment |

**Impact:** Q4_K requires larger tensors to be effective but provides better compression for large matrices.

### 2. Quantization Approach

**Q4_0:**
- Single scale factor per block
- All 32 values share the same scale
- Formula: `value = d * (q - 8)` (centered at 0)
- Simple and fast

**Q4_K:**
- Multiple scales (12 different scales)
- Non-uniform quantization across 256 values
- Formula: `value = d * scales[group] * q - d * dmin` (with minimum offset)
- More accurate but complex

### 3. Memory Layout

**Q4_0 Layout (18 bytes):**
```
Offset 0-1:   d (f16 scale)
Offset 2-17:  qs[16] (4-bit values)
```

**Q4_K Layout (144 bytes):**
```
Offset 0-1:    d (f16 scale)
Offset 2-3:    dmin (f16 minimum)
Offset 4-15:   scales[12] (uint8_t scales)
Offset 16-143: qs[128] (4-bit values in nibbles)
```

### 4. Dimension Compatibility

**Q4_0:**
- Compatible with dimensions that are multiples of 32
- Small models work well
- Less padding overhead

**Q4_K:**
- Requires dimensions that are multiples of 256
- Large models benefit most
- More padding overhead for small dimensions

## Performance Considerations

### Inference Speed

**Q4_0:**
- ✅ Faster dequantization (simpler formula)
- ✅ Less memory overhead per block
- ✅ Better cache locality (smaller blocks)
- ✅ More suitable for real-time inference

**Q4_K:**
- ⚠️ Slower dequantization (complex formula)
- ⚠️ More memory overhead per block
- ❌ Larger blocks can reduce cache efficiency
- ✅ Better for batch processing

### Memory Efficiency

**Compression Efficiency (both ~7.11×):**
- Q4_0: 128 → 18 bytes (7.11×)
- Q4_K: 1024 → 144 bytes (7.11×)

**Practical Memory Usage:**
- Q4_0: Better for small tensors (<1024 elements)
- Q4_K: Better for large tensors (>4096 elements)

### Accuracy

**Q4_0:**
- Uniform quantization can be less accurate for varied distributions
- Best for data with consistent magnitude ranges
- May require higher precision formats for sensitive layers

**Q4_K:**
- Non-uniform quantization adapts to data distribution
- Better accuracy for complex data patterns
- Often matches Q4_0 or Q8_0 accuracy with better compression

## Use Case Recommendations

### Use Q4_0 when:
1. **Real-time inference** is required
2. **Model size is small** (<1B parameters)
3. **Memory bandwidth is limited**
4. **Simple deployment** is preferred
5. **Cache efficiency** is critical

### Use Q4_K when:
1. **Accuracy is critical** and approaches Q8_0 quality
2. **Model size is large** (>3B parameters)
3. **Memory capacity is constrained**
4. **Batch processing** is common
5. **Complex data distributions** are present

## Dimension Compatibility Examples

### Compatible Dimensions

| Dimension | Q4_0 Blocks | Q4_K Blocks |
|-----------|-------------|-------------|
| 256 | 8 blocks | 1 block |
| 512 | 16 blocks | 2 blocks |
| 768 | 24 blocks | 3 blocks |
| 1024 | 32 blocks | 4 blocks |
| 4096 | 128 blocks | 16 blocks |

### Padding Requirements

**Q4_0:** Pads to next multiple of 32
- 100 → 128 (4 blocks)
- 500 → 512 (16 blocks)

**Q4_K:** Pads to next multiple of 256
- 100 → 256 (1 block)
- 500 → 512 (2 blocks)

## Implementation Notes

### Memory Access Patterns

**Q4_0:**
- Smaller blocks = more frequent block switches
- Better for sequential access patterns
- Lower latency per memory access

**Q4_K:**
- Larger blocks = fewer block switches
- Better for random access patterns
- Higher latency per memory access but fewer accesses

### GPU Considerations

**Q4_0:**
- ✅ Better for wavefront processing (32 threads = 32 elements)
- ✅ Less shared memory usage
- ✅ Simpler kernel implementation

**Q4_K:**
- ⚠️ Requires careful thread-to-element mapping
- ⚠️ More complex kernel logic
- ✅ Better memory coalescing for large tensors

## Conversion Between Formats

### Q4_0 to Q4_K
- Lossy conversion (Q4_K more accurate)
- Requires re-quantization with multiple scales
- Computationally expensive

### Q4_K to Q4_0
- Lossy conversion (Q4_0 less accurate)
- Requires scale simplification
- Reduces memory usage

### Both to Q8_0
- Lossless up-conversion (higher precision)
- Simple scale expansion
- Increases memory usage 2×

## Testing and Validation

Both formats have comprehensive verification tests:

**Q4_0 Tests:**
- `q4_0_correctness_test.rs` - Output coherence validation
- Block layout verification
- Dimension compatibility tests

**Q4_K Tests:**
- `q4_k_correctness_test.rs` - Output coherence validation
- `q4_k_layout_test.rs` - Memory layout verification
- `q4_k_dimension_test.rs` - Dimension compatibility tests

## Conclusion

**Q4_0 and Q4_K achieve similar compression ratios (~7.11×) but serve different purposes:**

- **Q4_0:** Speed and simplicity - ideal for small models and real-time inference
- **Q4_K:** Accuracy and efficiency - ideal for large models and memory-constrained environments

The choice between them depends on your specific requirements for latency, memory, and accuracy. Both formats are production-ready and thoroughly tested in rocmforge.