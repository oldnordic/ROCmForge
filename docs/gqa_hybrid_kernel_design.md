# GQA Hybrid Kernel Implementation Guide

**Date:** 2026-04-15
**Status:** Design Specification
**Reference:** Implements Option 4 from research document

---

## Overview

This document provides the complete implementation specification for the hybrid Q+KV kernel approach to enable QKV fusion for GQA models while maintaining performance and correctness.

## Architecture

### Design Philosophy

Instead of one monolithic kernel that handles Q, K, and V asymmetrically, we split into two specialized kernels:

1. **Q-Only Fusion Kernel:** RMSNorm → Q projection → RoPE(Q) → output
2. **KV-Only Fusion Kernel:** RMSNorm → K/V projection → RoPE(K) → KV cache write

### Key Insight

Both kernels need RMSNorm, but we can:
- **Option A:** Recompute RMSNorm in both kernels (simple, shared memory reuse)
- **Option B:** Compute RMSNorm once, cache in shared memory (complex, requires synchronization)

**Recommendation:** Start with Option A (recompute), optimize to B if needed.

---

## Kernel 1: Q-Only Fusion

### Signature

```cpp
template<int N_WAVES>
__global__ void gemv_norm_q_rope_q4_0_f32_kernel(
    // RMSNorm parameters
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    // Q projection
    const void* __restrict__ w_q,
    const float* __restrict__ bias_q,
    // Q output
    float* __restrict__ out_q,
    // Dimensions
    int n_rows,         // hidden_size
    int n_q,            // num_heads * head_dim
    // RoPE parameters
    const int* __restrict__ pos_ptr,
    int head_dim,
    float theta_base,
    int neox
);
```

### Implementation

```cpp
template<int N_WAVES>
__global__ void gemv_norm_q_rope_q4_0_f32_kernel(
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    const void* __restrict__ w_q,
    const float* __restrict__ bias_q,
    float* __restrict__ out_q,
    int n_rows, int n_q,
    const int* __restrict__ pos_ptr,
    int head_dim,
    float theta_base,
    int neox
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    // Early exit if out of bounds
    if (col_base >= n_q) return;

    // Shared memory: [n_rows floats] + [32 floats reduction]
    extern __shared__ float s_data[];
    float* s_input = s_data;
    float* s_reduction = &s_data[n_rows];

    // ── Phase 1: RMSNorm (identical to current kernel) ────────────────

    float local_sum_sq = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {
        float val = raw_hidden[i];
        s_input[i] = val;
        local_sum_sq += val * val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down(local_sum_sq, offset);
    }

    // Cross-warp reduction
    if (lane_id == 0) {
        s_reduction[wave_id] = local_sum_sq;
    }
    __syncthreads();

    float final_sum_sq = (tid < (blockDim.x / 32)) ? s_reduction[tid] : 0.0f;
    if (wave_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum_sq += __shfl_down(final_sum_sq, offset);
        }
        if (tid == 0) {
            s_reduction[0] = final_sum_sq;
        }
    }
    __syncthreads();

    // Apply normalization
    float inv_rms = rsqrtf(s_reduction[0] / (float)n_rows + eps);
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = s_input[i] * inv_rms * norm_weight[i];
    }
    __syncthreads();

    // ── Phase 2: Q Projection (simplified from current) ───────────────

    // Setup weight column pointers for 4 columns
    const Q4_0_block_nqr* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        w_cols[c] = reinterpret_cast<const Q4_0_block_nqr*>(
            static_cast<const uint8_t*>(w_q) + ((col_base + c) * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
    }

    // GEMV dot product
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK4_0;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i + 16])[0];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const Q4_0_block_nqr* b = &w_cols[c][block_idx];
                const float d = __half2float(b->d);
                const uint32_t q = reinterpret_cast<const uint32_t*>(b->qs)[i];

                sums[c] += d * (static_cast<float>( q        & 0x0F) - 8.0f) * in_l.x;
                sums[c] += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * in_l.y;
                sums[c] += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * in_l.z;
                sums[c] += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * in_l.w;
                sums[c] += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * in_h.x;
                sums[c] += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * in_h.y;
                sums[c] += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * in_h.z;
                sums[c] += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * in_h.w;
            }
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    // ── Phase 3: Bias + RoPE + Output ─────────────────────────────────

    if (lane_id == 0) {
        const int pos = *pos_ptr;

        // Add bias
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            sums[c] += (bias_q ? bias_q[col_base + c] : 0.0f);
        }

        // Apply RoPE
        const int col_in_q = col_base;
        const int head = col_in_q / head_dim;
        const int idx_in_head = col_in_q - head * head_dim;
        const int half = head_dim / 2;

        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            int pair_idx;
            if (neox) {
                const int local_idx = idx_in_head + 2 * p;
                pair_idx = (local_idx < half) ? local_idx : (local_idx - half);
            } else {
                pair_idx = (idx_in_head + 2 * p) / 2;
            }

            const float exponent = (float)(2 * pair_idx) / (float)head_dim;
            const float theta = 1.0f / powf(theta_base, exponent);
            const float angle = (float)pos * theta;
            const float cos_val = cosf(angle);
            const float sin_val = sinf(angle);

            const float x0 = sums[2 * p];
            const float x1 = sums[2 * p + 1];

            if (neox) {
                const int local_idx = idx_in_head + 2 * p;
                if (local_idx < half) {
                    sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                    sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                } else {
                    sums[2 * p]     = x0 * sin_val + x1 * cos_val;
                    sums[2 * p + 1] = x0 * cos_val - x1 * sin_val;
                }
            } else {
                sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
            }
        }

        // Write to output
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            out_q[col_base + c] = sums[c];
        }
    }
}
```

---

## Kernel 2: KV-Only Fusion

### Signature

```cpp
template<int N_WAVES>
__global__ void gemv_norm_kv_rope_kvwrite_q4_0_f32_kernel(
    // RMSNorm parameters
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    // KV projection
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    // KV cache
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    // Dimensions
    int n_rows,         // hidden_size
    int n_kv,           // num_kv_heads * head_dim
    // RoPE parameters
    const int* __restrict__ pos_ptr,
    int head_dim,
    float theta_base,
    int neox
);
```

### Implementation

```cpp
template<int N_WAVES>
__global__ void gemv_norm_kv_rope_kvwrite_q4_0_f32_kernel(
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int n_rows, int n_kv,
    const int* __restrict__ pos_ptr,
    int head_dim,
    float theta_base,
    int neox
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;
    const int total_kv_cols = 2 * n_kv;  // K + V

    // Early exit if out of bounds
    if (col_base >= total_kv_cols) return;

    // Shared memory: [n_rows floats] + [32 floats reduction]
    extern __shared__ float s_data[];
    float* s_input = s_data;
    float* s_reduction = &s_data[n_rows];

    // ── Phase 1: RMSNorm (identical to Q kernel) ─────────────────────

    float local_sum_sq = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {
        float val = raw_hidden[i];
        s_input[i] = val;
        local_sum_sq += val * val;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down(local_sum_sq, offset);
    }

    // Cross-warp reduction
    if (lane_id == 0) {
        s_reduction[wave_id] = local_sum_sq;
    }
    __syncthreads();

    float final_sum_sq = (tid < (blockDim.x / 32)) ? s_reduction[tid] : 0.0f;
    if (wave_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum_sq += __shfl_down(final_sum_sq, offset);
        }
        if (tid == 0) {
            s_reduction[0] = final_sum_sq;
        }
    }
    __syncthreads();

    // Apply normalization
    float inv_rms = rsqrtf(s_reduction[0] / (float)n_rows + eps);
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = s_input[i] * inv_rms * norm_weight[i];
    }
    __syncthreads();

    // ── Phase 2: KV Projection (dual GEMV) ───────────────────────────

    // Determine if processing K or V
    const bool is_k = col_base < n_kv;
    const int out_col_base = is_k ? col_base : (col_base - n_kv);

    // Setup weight column pointers for 4 columns
    const void* weights_base = is_k ? w_k : w_v;
    const Q4_0_block_nqr* w_cols[4];

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        w_cols[c] = reinterpret_cast<const Q4_0_block_nqr*>(
            static_cast<const uint8_t*>(weights_base) + ((out_col_base + c) * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
    }

    // GEMV dot product
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK4_0;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i + 16])[0];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const Q4_0_block_nqr* b = &w_cols[c][block_idx];
                const float d = __half2float(b->d);
                const uint32_t q = reinterpret_cast<const uint32_t*>(b->qs)[i];

                sums[c] += d * (static_cast<float>( q        & 0x0F) - 8.0f) * in_l.x;
                sums[c] += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * in_l.y;
                sums[c] += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * in_l.z;
                sums[c] += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * in_l.w;
                sums[c] += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * in_h.x;
                sums[c] += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * in_h.y;
                sums[c] += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * in_h.z;
                sums[c] += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * in_h.w;
            }
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    // ── Phase 3: Post-Processing (K: RoPE+cache, V: cache) ─────────────

    if (lane_id == 0) {
        const int pos = *pos_ptr;
        const size_t cache_base = (size_t)pos * n_kv;

        if (is_k) {
            // K: Add bias, apply RoPE, write to k_cache
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                sums[c] += (bias_k ? bias_k[out_col_base + c] : 0.0f);
            }

            const int col_in_k = out_col_base;
            const int kv_head = col_in_k / head_dim;  // Only n_kv_heads heads!
            const int idx_in_head = col_in_k - kv_head * head_dim;
            const int half = head_dim / 2;

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                int pair_idx;
                if (neox) {
                    const int local_idx = idx_in_head + 2 * p;
                    pair_idx = (local_idx < half) ? local_idx : (local_idx - half);
                } else {
                    pair_idx = (idx_in_head + 2 * p) / 2;
                }

                const float exponent = (float)(2 * pair_idx) / (float)head_dim;
                const float theta = 1.0f / powf(theta_base, exponent);
                const float angle = (float)pos * theta;
                const float cos_val = cosf(angle);
                const float sin_val = sinf(angle);

                const float x0 = sums[2 * p];
                const float x1 = sums[2 * p + 1];

                if (neox) {
                    const int local_idx = idx_in_head + 2 * p;
                    if (local_idx < half) {
                        sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                        sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                    } else {
                        sums[2 * p]     = x0 * sin_val + x1 * cos_val;
                        sums[2 * p + 1] = x0 * cos_val - x1 * sin_val;
                    }
                } else {
                    sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                    sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                }
            }

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                k_cache[cache_base + out_col_base + c] = sums[c];
            }

        } else {
            // V: Add bias, write directly to v_cache (no RoPE)
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float val = sums[c] + (bias_v ? bias_v[out_col_base + c] : 0.0f);
                v_cache[cache_base + out_col_base + c] = val;
            }
        }
    }
}
```

---

## C++ Dispatch Functions

### Q-Only Kernel

```cpp
extern "C" hipError_t gemv_norm_q_rope_q4_0_f32_launch(
    const float* raw_hidden,
    const float* norm_weight,
    float eps,
    const void* w_q,
    const float* bias_q,
    float* out_q,
    int n_rows, int n_q,
    const int* pos_ptr,
    int head_dim,
    float theta_base,
    int neox,
    hipStream_t stream
) {
    if (pos_ptr == nullptr) return hipErrorInvalidValue;
    if (raw_hidden == nullptr || norm_weight == nullptr) return hipErrorInvalidValue;
    if (n_rows <= 0 || n_q <= 0) return hipErrorInvalidValue;
    if ((n_q % 4) != 0) return hipErrorInvalidValue;
    if (head_dim <= 0 || (head_dim % 2) != 0) return hipErrorInvalidValue;

    const int N_WAVES = 8;
    const size_t shared_mem = (n_rows + 32) * sizeof(float);

    if (shared_mem <= 32768) {
        const int n_blocks_x = (n_q + (N_WAVES * 4) - 1) / (N_WAVES * 4);
        gemv_norm_q_rope_q4_0_f32_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
            raw_hidden, norm_weight, eps,
            w_q, bias_q, out_q,
            n_rows, n_q,
            pos_ptr, head_dim, theta_base, neox
        );
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}
```

### KV-Only Kernel

```cpp
extern "C" hipError_t gemv_norm_kv_rope_kvwrite_q4_0_f32_launch(
    const float* raw_hidden,
    const float* norm_weight,
    float eps,
    const void* w_k, const void* w_v,
    const float* bias_k, const float* bias_v,
    float* k_cache, float* v_cache,
    int n_rows, int n_kv,
    const int* pos_ptr,
    int head_dim,
    float theta_base,
    int neox,
    hipStream_t stream
) {
    if (pos_ptr == nullptr) return hipErrorInvalidValue;
    if (raw_hidden == nullptr || norm_weight == nullptr) return hipErrorInvalidValue;
    if (n_rows <= 0 || n_kv <= 0) return hipErrorInvalidValue;
    if ((n_kv % 4) != 0) return hipErrorInvalidValue;
    if (head_dim <= 0 || (head_dim % 2) != 0) return hipErrorInvalidValue;

    const int N_WAVES = 8;
    const int total_kv_cols = 2 * n_kv;
    const size_t shared_mem = (n_rows + 32) * sizeof(float);

    if (shared_mem <= 32768) {
        const int n_blocks_x = (total_kv_cols + (N_WAVES * 4) - 1) / (N_WAVES * 4);
        gemv_norm_kv_rope_kvwrite_q4_0_f32_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
            raw_hidden, norm_weight, eps,
            w_k, w_v, bias_k, bias_v,
            k_cache, v_cache,
            n_rows, n_kv,
            pos_ptr, head_dim, theta_base, neox
        );
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}
```

---

## Rust FFI Declarations

### Add to `src/gpu/kernels/q8_decode.rs`

```rust
/// Q-only fusion: RMSNorm + Q projection + RoPE (GQA-compatible)
pub fn gemv_norm_q_rope_q4_0_f32_on_stream(
    raw_hidden: *const f32,
    norm_weight: *const f32,
    eps: f32,
    w_q: *const u8,
    bias_q: *const f32,
    out_q: *mut f32,
    n_rows: usize,
    n_q: usize,
    pos_ptr: *const i32,
    head_dim: usize,
    theta_base: f32,
    neox: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    unsafe {
        let err = gemv_norm_q_rope_q4_0_f32_launch(
            raw_hidden,
            norm_weight,
            eps,
            w_q as *const c_void,
            bias_q,
            out_q,
            n_rows as i32,
            n_q as i32,
            pos_ptr,
            head_dim as i32,
            theta_base,
            neox,
            stream,
        );

        if err != hipError_t::hipSuccess {
            Err(GpuError::KernelLaunch(format!(
                "gemv_norm_q_rope_q4_0_f32_launch failed: {}",
                error_message(err)
            )))
        } else {
            Ok(())
        }
    }
}

/// KV-only fusion: RMSNorm + KV projection + RoPE(K) + KV write (GQA-compatible)
pub fn gemv_norm_kv_rope_kvwrite_q4_0_f32_on_stream(
    raw_hidden: *const f32,
    norm_weight: *const f32,
    eps: f32,
    w_k: *const u8,
    w_v: *const u8,
    bias_k: *const f32,
    bias_v: *const f32,
    k_cache: *mut f32,
    v_cache: *mut f32,
    n_rows: usize,
    n_kv: usize,
    pos_ptr: *const i32,
    head_dim: usize,
    theta_base: f32,
    neox: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    unsafe {
        let err = gemv_norm_kv_rope_kvwrite_q4_0_f32_launch(
            raw_hidden,
            norm_weight,
            eps,
            w_k as *const c_void,
            w_v as *const c_void,
            bias_k,
            bias_v,
            k_cache,
            v_cache,
            n_rows as i32,
            n_kv as i32,
            pos_ptr,
            head_dim as i32,
            theta_base,
            neox,
            stream,
        );

        if err != hipError_t::hipSuccess {
            Err(GpuError::KernelLaunch(format!(
                "gemv_norm_kv_rope_kvwrite_q4_0_f32_launch failed: {}",
                error_message(err)
            )))
        } else {
            Ok(())
        }
    }
}

// Link to HIP kernels
extern "C" {
    fn gemv_norm_q_rope_q4_0_f32_launch(
        raw_hidden: *const f32,
        norm_weight: *const f32,
        eps: f32,
        w_q: *const c_void,
        bias_q: *const f32,
        out_q: *mut f32,
        n_rows: i32,
        n_q: i32,
        pos_ptr: *const i32,
        head_dim: i32,
        theta_base: f32,
        neox: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn gemv_norm_kv_rope_kvwrite_q4_0_f32_launch(
        raw_hidden: *const f32,
        norm_weight: *const f32,
        eps: f32,
        w_k: *const c_void,
        w_v: *const c_void,
        bias_k: *const f32,
        bias_v: *const f32,
        k_cache: *mut f32,
        v_cache: *mut f32,
        n_rows: i32,
        n_kv: i32,
        pos_ptr: *const i32,
        head_dim: i32,
        theta_base: f32,
        neox: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
```

---

## Ops Layer Integration

### Modify `src/gpu/forward.rs`

```rust
// In gpu_attention_decode_from_state, replace QKV projection:

// OLD: Single fused kernel (only works when n_q == n_kv)
// let use_fused_kernel = /* ... */ && q_size == kv_size;

// NEW: Hybrid approach (works for all models)
let use_hybrid_fusion = gpu_layer.attn_q_meta.wtype == GgmlType::Q4_0
    && gpu_layer.attn_k_meta.wtype == GgmlType::Q4_0
    && gpu_layer.attn_v_meta.wtype == GgmlType::Q4_0
    && q_size % 4 == 0 && kv_size % 4 == 0;

if use_hybrid_fusion {
    // Q-only fusion
    gemv_norm_q_rope_q4_0_f32_on_stream(
        raw_hidden,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        eps,
        gpu_layer.attn_q.as_ptr() as *const u8,
        gpu_layer.attn_q_bias
            .as_ref()
            .map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        scratch.q.as_ptr() as *mut f32,
        h,
        q_size,
        scratch.decode_pos_ptr(),
        config.head_dim,
        config.rope_theta,
        if config.rope_scaling { 1 } else { 0 },
        device.stream(),
    )?;

    // KV fusion
    gemv_norm_kv_rope_kvwrite_q4_0_f32_on_stream(
        raw_hidden,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        eps,
        gpu_layer.attn_k.as_ptr() as *const u8,
        gpu_layer.attn_v.as_ptr() as *const u8,
        gpu_layer.attn_k_bias
            .as_ref()
            .map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        gpu_layer.attn_v_bias
            .as_ref()
            .map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        kv.k_ptr_mut(layer_idx)? as *mut f32,
        kv.v_ptr_mut(layer_idx)? as *mut f32,
        h,
        kv_size,
        scratch.decode_pos_ptr(),
        config.head_dim,
        config.rope_theta,
        if config.rope_scaling { 1 } else { 0 },
        device.stream(),
    )?;
} else {
    // Fallback to separate kernels
    gpu_rms_norm_rows(/* ... */)?;
    gpu_project_rows(&device, &gpu_layer.attn_q, /* Q ... */)?;
    gpu_project_rows(&device, &gpu_layer.attn_k, /* K ... */)?;
    gpu_project_rows(&device, &gpu_layer.attn_v, /* V ... */)?;
    // ... RoPE, KV write, etc.
}
```

---

## CMakeLists Integration

### Add to `hip_kernels/quant/CMakeLists.txt`

```cmake
# Q-only fusion kernel (GQA-compatible)
add_library(q4_0_fused_q_only STATIC
    q4_0_fused_q_only.hip
)
target_link_libraries(q4_0_fused_q_only quant_common)
set_target_properties(q4_0_fused_q_only PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# KV-only fusion kernel (GQA-compatible)
add_library(q4_0_fused_kv_only STATIC
    q4_0_fused_kv_only.hip
)
target_link_libraries(q4_0_fused_kv_only quant_common)
set_target_properties(q4_0_fused_kv_only PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
```

---

## Testing Plan

### 1. Correctness Testing

```rust
#[test]
fn test_hybrid_qkv_fusion_correctness() {
    // Test with GQA model (qwen2.5-0.5b)
    // Compare outputs vs non-fused path
    // Verify bitwise match
}
```

### 2. Performance Testing

```bash
# Benchmark with GQA model
cargo bench --bench gpu_decode --features gpu -- --noplot

# Expected: 585-595 tok/s (vs 535 tok/s current)
```

### 3. Regression Testing

```bash
# Test with MHA model (n_q == n_kv)
# Ensure performance matches or exceeds current QKV fusion

# Test with various GQA ratios
# qwen2: n_q/n_kv = 7
# mixtral: n_q/n_kv = 8
```

---

## Performance Projection

### Expected Improvements

| Configuration | Current | Hybrid | Improvement |
|---------------|---------|--------|-------------|
| **GQA (qwen)** | 535 tok/s | 585-595 tok/s | +50-60 tok/s (+9-11%) |
| **MHA (llama)** | 646 tok/s | 640-650 tok/s | ~0% (2 kernels vs 1) |

### Why 2 Kernels ≈ 1 Kernel Performance

1. **Wave balance:** Q and KV can run in parallel (different CUDA streams)
2. **RMSNorm recomputation:** Small cost compared to memory bandwidth savings
3. **Better occupancy:** Specialized kernels have simpler control flow

### Optimization Opportunities

**Option B (Advanced):** Cache normalized hidden state

```cpp
// Kernel 1: RMSNorm + Q projection
// Write normalized hidden to global memory
// Kernel 2: Load normalized hidden + KV projection
```

**Trade-off:** Extra global memory read vs RMSNorm recomputation
- RMSNorm cost: ~50 FLOPs per element
- Global memory read: ~500 cycles latency
- **Verdict:** Recompute RMSNorm (Option A) is likely faster

---

## Conclusion

The hybrid approach provides:
- ✅ **GQA compatibility** (works for all models)
- ✅ **Modest performance gain** (~50-60 tok/s)
- ✅ **Lower complexity** than monolithic GQA kernel
- ✅ **Maintainable codebase** (separate concerns)

**Next Step:** Implement and benchmark to validate projections.

---

**Implementation Checklist:**
- [ ] Create `q4_0_fused_q_only.hip`
- [ ] Create `q4_0_fused_kv_only.hip`
- [ ] Update CMakeLists.txt
- [ ] Add Rust FFI declarations
- [ ] Modify ops layer integration
- [ ] Add correctness tests
- [ ] Run performance benchmarks
- [ ] Update documentation

**Estimated Effort:** 12 hours (2 design, 4 implementation, 4 testing, 2 tuning)
