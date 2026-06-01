# Quick Start: hipfire Techniques for Q4_0

**Fast-track implementation guide** - What to copy from hipfire, how to adapt it.

---

## 1. DP4A Optimization (Week 2)

### Add feature detection

**File:** `src/gpu/features.rs` (new)

```rust
use hip_backend::HipError;

pub struct GpuFeatures {
    pub arch: String,
    pub has_dp4a: bool,
    pub has_dot2_f32_f16: bool,
    pub has_wmma: bool,
}

impl GpuFeatures {
    pub fn detect(device: &hip_backend::Device) -> Result<Self, HipError> {
        let arch = device.get_architecture()?;
        
        // DP4a: gfx1030+ (RDNA2) and gfx1100+ (RDNA3)
        let has_dp4a = arch.starts_with("gfx103")
            || arch.starts_with("gfx110")
            || arch.starts_with("gfx115")
            || arch.starts_with("gfx120");
        
        // v_dot2_f32_f16 instruction support
        let has_dot2_f32_f16 = matches!(arch.as_str(),
            "gfx1011" | "gfx1012"
            | "gfx1030" | "gfx1031" | "gfx1032"
            | "gfx1100" | "gfx1101" | "gfx1102"
            | "gfx1150" | "gfx1151"
            | "gfx1200" | "gfx1201"
        );
        
        // WMMA: gfx1100+ only
        let has_wmma = arch.starts_with("gfx110")
            || arch.starts_with("gfx115")
            || arch.starts_with("gfx120");
        
        Ok(Self { arch, has_dp4a, has_dot2_f32_f16, has_wmma })
    }
}
```

### DP4A Kernel Pattern

**File:** `hip_kernels/quant/q4_0_fused_norm_qkv_rope_dp4a.hip` (new)

**Key change:** Replace scalar loop with dp4a

**Current scalar pattern (slow):**
```cpp
for (int l = 0; l < 16; ++l) {
    const uint8_t q = static_cast<uint8_t>(b->qs[l]);
    sums[c] += d * (static_cast<float>(q & 0x0F) - 8.0f) * s_input[row_offset + l];
    sums[c] += d * (static_cast<float>(q >> 4) - 8.0f) * s_input[row_offset + l + 16];
}
```

**DP4A pattern (fast):**
```cpp
// Pack 16 bytes into 4 × uint32_t
const unsigned char* nib = reinterpret_cast<const unsigned char*>(b->qs);
unsigned int pk0 = *(const unsigned int*)(nib + 0);   // Bytes 0-3
unsigned int pk1 = *(const unsigned int*)(nib + 4);   // Bytes 4-7
unsigned int pk2 = *(const unsigned int*)(nib + 8);   // Bytes 8-11
unsigned int pk3 = *(const unsigned int*)(nib + 12);  // Bytes 12-15

// Pack nibbles (even: 0,2,4,6 | odd: 1,3,5,7)
int nib_even = (pk0 & 0xF)
             | ((pk1 & 0xF) << 8)
             | ((pk2 & 0xF) << 16)
             | ((pk3 & 0xF) << 24);

int nib_odd  = (pk0 >> 4)
             | ((pk1 >> 4) << 8)
             | ((pk2 >> 4) << 16)
             | ((pk3 >> 4) << 24);

// Load and quantize x values
float x_scale = compute_scale(x_vals);  // Find amax, scale to int8
int xq0 = __float2int_rn(x_vals[0] / x_scale);
int xq1 = __float2int_rn(x_vals[1] / x_scale);
// ... etc (pack into xq_even, xq_odd)

// Two dp4a instructions = 8 multiply-accumulates
int dot_sum = __builtin_amdgcn_sdot4(nib_even, xq_even, 0, false);
dot_sum = __builtin_amdgcn_sdot4(nib_odd, xq_odd, dot_sum, false);

// Rescale and accumulate
float nib_dot_x = (float)dot_sum * x_scale;
sums[c] += d * (nib_dot_x - 8.0f * 16.0f);  // Q4_0 zero-point is 8
```

**Performance:** 2 instructions vs 32 scalar ops = **~4-6× faster** for this loop

---

## 2. Multi-Row GEMV (Week 3)

### Kernel change

**Current:** `q4_0_fused_norm_qkv_rope.hip` processes 1 output row per block

**Multi-row variant:** Process 2-4 rows per block

```cpp
// At kernel launch
const int row_start = blockIdx.x * ROWS_PER_BLOCK;
const int row_idx = threadIdx.x / 32;  // Which row in this block
const int tid_in_row = threadIdx.x % 32;

for (int r = 0; r < ROWS_PER_BLOCK; r++) {
    const int row = row_start + r;
    if (row >= M) break;
    
    // Process this row (same logic as current kernel)
    // Each wave handles one row
}
```

**Architecture tuning:**
```rust
// src/gpu/ops.rs
fn optimal_rows_per_block(arch: &str) -> usize {
    match arch {
        "gfx1100" | "gfx1101" | "gfx1102" => 1,  // RDNA3: single-row optimal
        "gfx1030" | "gfx1031" => 1,              // RDNA2: specialized kernels
        _ => 2,                                   // RDNA1/APU: multi-row helps
    }
}
```

---

## 3. WMMA for RDNA3 Prefill (Week 4)

### Detect prefill vs decode

```rust
// src/gpu/forward.rs
pub fn gpu_layer_forward_decode(...) {
    let batch_size = 1;  // Decode: always 1 token
    let features = device.features();
    
    if features.has_wmma && batch_size > 1 {
        // Prefill: use WMMA
        launch_wmma_kernel();
    } else {
        // Decode: use DP4A or scalar
        launch_optimized_kernel();
    }
}
```

### WMMA kernel structure

**File:** `hip_kernels/quant/q4_0_fused_norm_qkv_rope_wmma.hip` (new)

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;

__launch_bounds__(32, 8)
__global__ void fused_qkv_q4_0_wmma(
    const void* __restrict__ w_q, const void* __restrict__ w_k, const void* __restrict__ w_v,
    const float* __restrict__ x,
    float* __restrict__ out_q,
    // ...
) {
    // Stage 1: Dequant Q4_0 → FP16 in shared memory
    extern __shared__ half smem_weights[];
    
    const Q4_0_block* b = reinterpret_cast<const Q4_0_block*>(w_q);
    float d = __half2float(b->d);
    const uint8_t* qs = b->qs;
    
    // Each thread loads 2 Q4_0 values, dequant to FP16
    int smem_idx = threadIdx.x * 2;
    smem_weights[smem_idx] = __float2half(d * ((qs[0] & 0xF) - 8.0f));
    smem_weights[smem_idx + 1] = __float2half(d * ((qs[0] >> 4) - 8.0f));
    // ... etc for 16 values
    __syncthreads();
    
    // Stage 2: WMMA GEMM with FP16
    half16_t a = load_half16_from_shared(smem_weights, ...);
    half16_t b = load_half16_from_shared(x_fp16, ...);
    float8_t acc = {0.0f, 0.0f, ...};
    
    // 16×16×16 matrix multiply
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    
    // Write output
    store_results(acc, out_q);
}
```

**Key:** Two-stage dequant + WMMA

---

## 4. Packed Loads (Immediate)

### Fix Q4_0 loading pattern

**Current (what we fixed):**
```cpp
for (int l = 0; l < 16; ++l) {
    const uint8_t q = static_cast<uint8_t>(b->qs[l]);
    // ...
}
```

**Better (packed loads):**
```cpp
const unsigned char* qs = reinterpret_cast<const unsigned char*>(b->qs);

// Load 4 bytes at once
unsigned int pk0 = *(const unsigned int*)(qs + 0);
unsigned int pk1 = *(const unsigned int*)(qs + 4);
unsigned int pk2 = *(const unsigned int*)(qs + 8);
unsigned int pk3 = *(const unsigned int*)(qs + 12);

// Extract bytes, then extract nibbles from bytes
unsigned char b0 = pk0 & 0xFF;
unsigned char b1 = (pk0 >> 8) & 0xFF;
// ... etc

// Now extract nibbles
float n0 = static_cast<float>((b0 & 0x0F)) - 8.0f;
float n1 = static_cast<float>((b0 >> 4)) - 8.0f;
// ... etc
```

**Benefit:** 4 loads instead of 16 = better memory coalescing

---

## 5. Factored Dequantization (Immediate)

### Reduce multiplication count

**Current:**
```cpp
sums[c] += d * (nib - 8.0f) * x;  // 2 multiplies per element
```

**Optimized:**
```cpp
// Precompute sum of x values
float sum_x = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

// Compute dot(nib, x) once
float nib_dot_x = nib0 * x0 + nib1 * x1 + /*...*/;

// Single dequantization
sums[c] += d * (nib_dot_x - 8.0f * sum_x);  // 2 multiplies total
```

**Benefit:** 16 multiplies → 2 multiplies = **8× reduction**

---

## 6. Dispatch Logic (Week 2)

### Add kernel selection

**File:** `src/gpu/ops.rs`

```rust
pub fn gpu_dispatch_fused_norm_qkv_rope_kvwrite_on_stream(
    device: &GpuDevice,
    // ... existing params ...
) -> GpuResult<bool> {
    let features = device.features()?;
    
    // Check if we can use fusion at all
    if q_meta.wtype != GgmlType::Q4_0 {
        return Ok(false);
    }
    
    // Select kernel variant based on features
    if features.has_wmma && batch_size > 1 {
        // WMMA for prefill
        gemv_norm_qkv_rope_kvwrite_q4_0_wmma_on_stream(
            // ... params ...
        )?;
    } else if features.has_dp4a {
        // DP4A for RDNA2/3
        gemv_norm_qkv_rope_kvwrite_q4_0_dp4a_on_stream(
            // ... params ...
        )?;
    } else {
        // Scalar fallback
        gemv_norm_qkv_rope_kvwrite_q4_0_scalar_on_stream(
            // ... params ...
        )?;
    }
    
    Ok(true)
}
```

---

## 7. Testing (Week 5)

### Add correctness test

**File:** `tests/kernel_correctness.rs` (new)

```rust
#[test]
fn test_q4_0_dequantization_correctness() {
    // Load a small Q4_0 model
    let weights = load_test_weights();
    
    // Run CPU reference
    let cpu_output = cpu_forward(&weights, &input);
    
    // Run GPU kernel
    let gpu_output = gpu_forward(&weights, &input);
    
    // Compare (allow small floating-point differences)
    assert_close!(cpu_output, gpu_output, tol: 1e-5);
}

#[test]
fn test_dp4a_accuracy() {
    // Test DP4A variant vs scalar
    let scalar_out = scalar_kernel_forward(&input);
    let dp4a_out = dp4a_kernel_forward(&input);
    
    // DP4a has ~0.4% noise from x quantization
    assert_relative_error!(dp4a_out, scalar_out, max_percent: 0.5);
}
```

---

## 8. Profiling (Week 5)

### Add timing instrumentation

**File:** `src/gpu/profile.rs` (new)

```rust
use std::time::Instant;

pub struct KernelTimer {
    name: String,
    start: Instant,
}

impl KernelTimer {
    pub fn start(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for KernelTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        println!("[{}] {:.2} ms", self.name, elapsed.as_secs_f64() * 1000.0);
    }
}

// Usage in kernels
pub fn gpu_dispatch_fused(...) {
    let _timer = KernelTimer::start("fused_qkv");
    // ... launch kernel
    device.stream().synchronize().unwrap();
    // Timer prints on drop
}
```

---

## Implementation Checklist

- [ ] **Week 1**
  - [ ] Create `src/gpu/features.rs`
  - [ ] Add arch detection to `src/gpu/device.rs`
  - [ ] Test on different GPUs

- [ ] **Week 2**
  - [ ] Create `q4_0_fused_norm_qkv_rope_dp4a.hip`
  - [ ] Add DP4A dispatch logic
  - [ ] Benchmark on RDNA2
  - [ ] Test accuracy impact

- [ ] **Week 3**
  - [ ] Create multi-row variant
  - [ ] Add per-arch tuning
  - [ ] Benchmark on RDNA1/2/3

- [ ] **Week 4**
  - [ ] Create `q4_0_fused_norm_qkv_rope_wmma.hip`
  - [ ] Add prefill/decode split
  - [ ] Benchmark on RDNA3

- [ ] **Week 5**
  - [ ] Add correctness tests
  - [ ] Add performance benchmarks
  - [ ] Add profiling infrastructure
  - [ ] Update documentation

---

## Quick Wins (Do Today)

### 1. Fix loading pattern (10 minutes)
```cpp
// In q4_0_fused_norm_qkv_rope.hip, replace:
for (int l = 0; l < 16; ++l) {
    const uint8_t q = static_cast<uint8_t>(b->qs[l]);
    // ...
}

// With:
const unsigned char* qs = reinterpret_cast<const unsigned char*>(b->qs);
unsigned int pk = *(const unsigned int*)(qs + block * 4);
// Extract bytes, then nibbles
```

### 2. Add factored dequantization (15 minutes)
```cpp
// Precompute sum_x once
float sum_x = 0.0f;
for (int i = 0; i < 16; i++) {
    sum_x += s_input[row_offset + i];
}

// Use in accumulation
sums[c] += d * (nib_dot_x - 8.0f * sum_x);
```

### 3. Add arch detection (30 minutes)
```rust
// In src/gpu/device.rs
pub fn get_architecture(&self) -> Result<String, HipError> {
    // Query HIP device properties
    let name = self.get_name()?;
    
    // Map device name to gfx arch
    if name.contains("7900") {
        Ok("gfx1100".to_string())
    } else if name.contains("6900") {
        Ok("gfx1030".to_string())
    } else if name.contains("5700") {
        Ok("gfx1010".to_string())
    } else {
        // Fallback: query HIP directly
        self.hip.get_device_architecture(self.device_id)
    }
}
```

**Expected impact:** 10-20% speedup from these three changes alone.

---

**Summary:** Copy patterns from hipfire, adapt to Q4_0 format. DP4a gives 1.5-2×, WMMA gives 2-4× for prefill. Combined: 1.7-2.3× overall on RDNA2/3.
