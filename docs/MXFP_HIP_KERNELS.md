# MXFP HIP Dequantization Kernels - Technical Summary

## Overview
Location: `/home/feanor/Projects/ROCmForge/kernels/mxfp_dequant.hip`

## Kernels Implemented

### 1. mxfp4_to_fp16_kernel
Dequantizes MXFP4 blocks (32 elements × 4-bit) to FP32 output.

**Signature**:
```cpp
extern "C" __global__ void mxfp4_to_fp16_kernel(
    const uint8_t* __restrict__ input,  // Packed MXFP4 data
    float* __restrict__ output,         // FP32 output
    const int num_blocks                 // Number of MXFP4 blocks
)
```

**Grid/Block Configuration**:
- Grid: `(num_blocks, 1, 1)` - one block per MXFP4 block
- Block: 256 threads (8 waves of 32 for RDNA3)

**Algorithm**:
1. Each thread processes 1 element (32 threads active per block)
2. Read E8M0 scale: `scale = 2^exponent`
3. Unpack 4-bit E2M1 from byte (2 elements per byte)
4. Decode E2M1: `value = (-1)^sign * 2^(exp-1) * (1 + mant)`
5. Apply scale: `output = scale * value`
6. Clamp to [-8, 8] per OCP MX Spec v1.0

---

### 2. mxfp6_to_fp16_kernel
Dequantizes MXFP6 blocks (32 elements × 6-bit) to FP32 output.

**Signature**:
```cpp
extern "C" __global__ void mxfp6_to_fp16_kernel(
    const uint8_t* __restrict__ input,  // Packed MXFP6 data
    float* __restrict__ output,         // FP32 output
    const int num_blocks                 // Number of MXFP6 blocks
)
```

**Grid/Block Configuration**:
- Grid: `(num_blocks, 1, 1)` - one block per MXFP6 block
- Block: 256 threads

**Algorithm**:
1. Each thread processes 1 element (32 threads active per block)
2. Read E8M0 scale: `scale = 2^exponent`
3. Unpack 6-bit E2M3 (packed across byte boundaries)
   - Bit offset: `(element_idx * 6) % 8`
   - Byte index: `(element_idx * 6) / 8`
   - Extract 6 bits from 2-byte combined value
4. Decode E2M3: `value = (-1)^sign * 2^(exp-1) * (1 + mant/8)`
5. Apply scale: `output = scale * value`
6. Clamp to [-7.5, 7.5]

---

### 3. mxfp4_to_fp32_batch_kernel
Batched MXFP4 dequantization for flexible grid/block configuration.

**Signature**:
```cpp
extern "C" __global__ void mxfp4_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
)
```

**Use Case**: High-throughput batch processing with coalesced memory access.

---

### 4. mxfp6_to_fp32_batch_kernel
Batched MXFP6 dequantization for flexible grid/block configuration.

**Signature**:
```cpp
extern "C" __global__ void mxfp6_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
)
```

**Use Case**: High-throughput batch processing with coalesced memory access.

---

## Device Helper Functions

### decode_e2m1(uint8_t bits) -> float
```cpp
__device__ __forceinline__ float decode_e2m1(uint8_t bits) {
    if (bits == 0) return 0.0f;

    int sign = (bits & 0x08) ? -1 : 1;      // Bit 3: sign
    int exp = ((bits >> 1) & 0x03) - 1;     // Bits 1-2: exponent
    int mant = bits & 0x01;                  // Bit 0: mantissa

    return sign * __int2float_rn(1 + mant) * exp2f(exp);
}
```

**E2M1 Format**: [sign(1) | exp(2) | mant(1)]
**Value**: `(-1)^sign * 2^(exp-1) * (1 + mant)`

---

### decode_e2m3(uint8_t bits) -> float
```cpp
__device__ __forceinline__ float decode_e2m3(uint8_t bits) {
    if (bits == 0) return 0.0f;

    int sign = (bits & 0x20) ? -1 : 1;      // Bit 5: sign
    int exp = ((bits >> 3) & 0x03) - 1;     // Bits 3-4: exponent
    int mant = bits & 0x07;                  // Bits 0-2: mantissa

    return sign * (1.0f + mant / 8.0f) * exp2f(exp);
}
```

**E2M3 Format**: [sign(1) | exp(2) | mant(3)]
**Value**: `(-1)^sign * 2^(exp-1) * (1 + mant/8)`

---

### e8m0_to_f32(int8_t exponent) -> float
```cpp
__device__ __forceinline__ float e8m0_to_f32(int8_t exponent) {
    return exp2f(static_cast<float>(exponent));
}
```

**E8M0 Format**: 8-bit exponent-only
**Value**: `2^exponent`

---

## Bit Packing Details

### MXFP4 (4-bit values)
Each byte contains 2 elements:
- **High nibble** (bits 4-7): element at even index
- **Low nibble** (bits 0-3): element at odd index

**Unpacking**:
```cpp
int byte_idx = element_idx / 2;
int nibble_idx = element_idx % 2;

uint8_t e2m1_bits;
if (nibble_idx == 0) {
    e2m1_bits = (data[byte_idx] >> 4) & 0x0F;  // High nibble
} else {
    e2m1_bits = data[byte_idx] & 0x0F;          // Low nibble
}
```

### MXFP6 (6-bit values)
Elements are packed across byte boundaries:
- 32 elements × 6 bits = 192 bits = 24 bytes
- Each element spans 2 bytes (except aligned ones)

**Unpacking**:
```cpp
int bit_offset = (element_idx * 6) % 8;
int byte_idx = (element_idx * 6) / 8;

uint16_t combined = (data[byte_idx + 1] << 8) | data[byte_idx];
uint8_t e2m3_bits = (combined >> (10 - bit_offset)) & 0x3F;
```

**Visualization**:
```
Byte 0: [E0(5:0)][E1(5:2)]
Byte 1: [E1(1:0)][E2(5:4)]
Byte 2: [E2(3:0)][E3(5:6)]
...
```

---

## Memory Layout

### MXFP4 Block (17 bytes)
```
Offset  Size  Description
------  ----  -----------
0       1     E8M0 scale (signed 8-bit exponent)
1-16    16    32 × 4-bit elements (packed 2 per byte)
```

### MXFP6 Block (25 bytes)
```
Offset  Size  Description
------  ----  -----------
0       1     E8M0 scale (signed 8-bit exponent)
1-24    24    32 × 6-bit elements (packed across bytes)
```

---

## Compilation

**Command**:
```bash
/opt/rocm/bin/hipcc -c \
    --genco \
    --offload-arch=gfx1100 \
    -O3 \
    kernels/mxfp_dequant.hip \
    -o mxfp4_to_fp16_kernel.hsaco
```

**Result**: 16KB HSACO file for gfx1100 (RDNA3)

---

## GPU Optimization Notes

### RDNA3 Architecture (gfx1100)
- **Wavefront size**: 32 threads
- **Block size**: 256 threads = 8 waves
- **LDS**: 128 KB per CU
- **L2 cache**: 6 MB (shared across shaders)

### Optimization Strategies Implemented
1. **Coalesced memory access**: Sequential thread IDs access sequential bytes
2. **Bit operations**: Efficient shift/mask instead of division
3. **__device__ __forceinline__**: All helper functions inlined
4. **Avoid bank conflicts**: Sequential access patterns
5. **Minimal register pressure**: Simple arithmetic operations

### Expected Performance
- **Memory bandwidth bound**: ~1.5 TB/s (RX 7900 XT theoretical)
- **Compute bound**: Bit unpacking + float conversion
- **Estimated throughput**: 100-500 GB/s (depending on tensor size)

---

## Integration Example

### Loading the Kernel
```rust
// In HipBackend initialization
let mxfp_hsaco_path = env::var("MXFP_DEQUANT_HSACO")?;
let mxfp_module = HipModule::load_from_file(&mxfp_hsaco_path)?;
let mxfp4_kernel = mxfp_module.get_kernel("mxfp4_to_fp16_kernel")?;
```

### Launching the Kernel
```rust
// Dequantize MXFP4 tensor
let num_blocks = (num_elements + 31) / 32;

let grid = hipDim3 { x: num_blocks, y: 1, z: 1 };
let block = hipDim3 { x: 256, y: 1, z: 1 };

unsafe {
    hipLaunchKernel!(
        mxfp4_kernel,
        grid,
        block,
        [input_ptr, output_ptr, num_blocks as i32],
        0,
        stream
    );
}
```

---

## Testing

### Correctness Test
Compare CPU vs GPU output:
```rust
// CPU dequantization
let cpu_output = dequantize_mxfp4_cpu(&tensor)?;

// GPU dequantization
let gpu_output = dequantize_mxfp4_gpu(&tensor)?;

// Verify match
assert_eq!(cpu_output.len(), gpu_output.len());
for (cpu, gpu) in cpu_output.iter().zip(gpu_output.iter()) {
    assert!((cpu - gpu).abs() < f32::EPSILON);
}
```

### Performance Test
```rust
let start = Instant::now();
for _ in 0..100 {
    dequantize_mxfp4_gpu(&tensor)?;
}
let elapsed = start.elapsed();

println!("MXFP4 GPU dequantization: {:.2} µs/block",
    elapsed.as_micros() as f32 / 100.0 / num_blocks as f32
);
```

---

## Future Optimizations

1. **Shared memory caching**: Cache scale value in __shared__ memory
2. **Vectorized loads**: Use uint4/float4 for memory loads
3. **Async copy**: Use HIP async copy for overlapping compute/memory
4. **Batch optimization**: Process multiple blocks per kernel launch
5. **FP16 output**: Use native FP16 math for faster computation (requires FP16 support)

---

## References

- **OCP MX Spec v1.0**: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- **ROCm HIP Programming Guide**: https://rocm.docs.amd.com/projects/HIP/en/latest/
- **RDNA3 Architecture**: AMD Radeon RX 7900 XT (gfx1100)
