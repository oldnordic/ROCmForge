#include "common.hip"

#define HD 128
#define TILE_ROWS 4

// ── Kernel Definitions ───────────────────────────────────────────────────────────

extern "C" __global__ void fused_sigmoid_alpha_gate_f32_kernel(
    float* __restrict__ beta,          // [batch × n] in-place sigmoid
    float* __restrict__ alpha,         // [batch × n] in-place alpha_gate
    const float* __restrict__ dt_bias, // [n]
    const float* __restrict__ a_log,   // [n]
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const long long batch_off = (long long)blockIdx.y * n;
    float* beta_b  = beta  + batch_off;
    float* alpha_b = alpha + batch_off;

    // Sigmoid on beta
    beta_b[i] = 1.0f / (1.0f + expf(-beta_b[i]));

    // Alpha gate on alpha
    float biased = alpha_b[i] + dt_bias[i];
    float sp = (biased > 20.0f) ? biased
             : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
    alpha_b[i] = sp * (-expf(a_log[i]));
}

extern "C" __global__ void conv1d_silu_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    const int n_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    float x = input[c];
    float s0 = state[c * 3];
    float s1 = state[c * 3 + 1];
    float s2 = state[c * 3 + 2];

    float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
            + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

    // SiLU fused
    output[c] = y / (1.0f + expf(-y));

    state[c * 3 + 2] = s1;
    state[c * 3 + 1] = s0;
    state[c * 3]     = x;
}

extern "C" __global__ void fused_qk_l2_norm_scale_f32_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const int n_heads,
    const int head_dim,
    const float q_scale,
    const float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    float* q_head = q + batch_off + h * head_dim;
    float* k_head = k + batch_off + h * head_dim;

    // Q: l2-normalize, then scale
    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_inv_norm;
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_scale;

    // K: l2-normalize (no scale)
    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        k_head[i] *= k_inv_norm;
}

extern "C" __launch_bounds__(256)
__global__ void repeat_interleave_qk_f32_kernel(
    const float* __restrict__ q_src,
    const float* __restrict__ k_src,
    float* __restrict__ q_dst,
    float* __restrict__ k_dst,
    const int n_key_heads,
    const int ratio,
    const int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_key_heads * ratio * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int kh_r = idx / head_dim;
    int kh = kh_r / ratio;

    int src_off = kh * head_dim + d;
    q_dst[idx] = q_src[src_off];
    k_dst[idx] = k_src[src_off];
}

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ output,
    const int n_tokens,
    const int n_heads,
    const int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;  // row tile (0..HD/TILE_ROWS-1)
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    // Tile of S in LDS: TILE_ROWS × 128 floats = 4KB
    __shared__ float S_tile[TILE_ROWS * HD];

    // Load tile from global → LDS
    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = S_global[i];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col+1], k2 = k_t[col+2], k3 = k_t[col+3];
        float q0 = q_t[col], q1 = q_t[col+1], q2 = q_t[col+2], q3 = q_t[col+3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD + col;
            float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    // Store tile back
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = S_tile[i];
}

extern "C" __global__ void gated_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n_heads,
    const int head_dim,
    const float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    const float* xh = x + batch_off + h * head_dim;
    const float* zh = z + batch_off + h * head_dim;
    float* oh = out + batch_off + h * head_dim;

    float sq_sum = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += 32) {
        float normed = xh[i] * inv_rms * weight[i];
        float z_val = zh[i];
        float silu_z = z_val / (1.0f + expf(-z_val));
        oh[i] = normed * silu_z;
    }
}

// ── Host-Side Launchers ──────────────────────────────────────────────────────────

extern "C" {

hipError_t gpu_fused_sigmoid_alpha_gate_f32(
    float* beta,
    float* alpha,
    const float* dt_bias,
    const float* a_log,
    int n,
    int batch_size,
    hipStream_t stream
) {
    CHECK_NULL(beta);
    CHECK_NULL(alpha);
    CHECK_NULL(dt_bias);
    CHECK_NULL(a_log);
    if (n <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    const int block = 256;
    const int grid_x = (n + block - 1) / block;

    hipLaunchKernelGGL(
        fused_sigmoid_alpha_gate_f32_kernel,
        dim3(grid_x, batch_size, 1), dim3(block, 1, 1), 0, stream,
        beta, alpha, dt_bias, a_log, n
    );

    return hipGetLastError();
}

hipError_t gpu_conv1d_silu_f32(
    float* output,
    const float* input,
    const float* weight,
    float* state,
    int n_channels,
    hipStream_t stream
) {
    CHECK_NULL(output);
    CHECK_NULL(input);
    CHECK_NULL(weight);
    CHECK_NULL(state);
    if (n_channels <= 0) return hipErrorInvalidValue;

    const int block = 256;
    const int grid = (n_channels + block - 1) / block;

    hipLaunchKernelGGL(
        conv1d_silu_f32_kernel,
        dim3(grid, 1, 1), dim3(block, 1, 1), 0, stream,
        output, input, weight, state, n_channels
    );

    return hipGetLastError();
}

hipError_t gpu_fused_qk_l2_norm_scale_f32(
    float* q,
    float* k,
    int n_heads,
    int head_dim,
    int batch_size,
    float q_scale,
    float eps,
    hipStream_t stream
) {
    CHECK_NULL(q);
    CHECK_NULL(k);
    if (n_heads <= 0 || head_dim <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        fused_qk_l2_norm_scale_f32_kernel,
        dim3(n_heads, batch_size, 1), dim3(32, 1, 1), 0, stream,
        q, k, n_heads, head_dim, q_scale, eps
    );

    return hipGetLastError();
}

hipError_t gpu_repeat_interleave_qk_f32(
    const float* q_src,
    const float* k_src,
    float* q_dst,
    float* k_dst,
    int n_key_heads,
    int ratio,
    int head_dim,
    hipStream_t stream
) {
    CHECK_NULL(q_src);
    CHECK_NULL(k_src);
    CHECK_NULL(q_dst);
    CHECK_NULL(k_dst);
    if (n_key_heads <= 0 || ratio <= 0 || head_dim <= 0) return hipErrorInvalidValue;

    int total = n_key_heads * ratio * head_dim;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    hipLaunchKernelGGL(
        repeat_interleave_qk_f32_kernel,
        dim3(grid, 1, 1), dim3(block, 1, 1), 0, stream,
        q_src, k_src, q_dst, k_dst, n_key_heads, ratio, head_dim
    );

    return hipGetLastError();
}

hipError_t gpu_gated_delta_net_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    int n_tokens,
    int n_heads,
    int head_dim,
    hipStream_t stream
) {
    CHECK_NULL(q);
    CHECK_NULL(k);
    CHECK_NULL(v);
    CHECK_NULL(gate);
    CHECK_NULL(beta);
    CHECK_NULL(state);
    CHECK_NULL(output);
    if (n_tokens <= 0 || n_heads <= 0 || head_dim != HD) return hipErrorInvalidValue;

    const int n_tiles = HD / TILE_ROWS;

    hipLaunchKernelGGL(
        gated_delta_net_f32_kernel,
        dim3(n_heads, n_tiles, 1), dim3(32, 1, 1), 0, stream,
        q, k, v, gate, beta, state, output, n_tokens, n_heads, head_dim
    );

    return hipGetLastError();
}

hipError_t gpu_gated_norm_f32(
    const float* x,
    const float* z,
    const float* weight,
    float* out,
    int n_heads,
    int head_dim,
    int batch_size,
    float eps,
    hipStream_t stream
) {
    CHECK_NULL(x);
    CHECK_NULL(z);
    CHECK_NULL(weight);
    CHECK_NULL(out);
    if (n_heads <= 0 || head_dim <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        gated_norm_f32_kernel,
        dim3(n_heads, batch_size, 1), dim3(32, 1, 1), 0, stream,
        x, z, weight, out, n_heads, head_dim, eps
    );

    return hipGetLastError();
}

// ── Batched Prefill Kernels ──────────────────────────────────────────────────────

extern "C" __global__ void batched_fused_sigmoid_alpha_gate_f32_kernel(
    float* __restrict__ beta,
    float* __restrict__ alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const int n,
    const int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (int b = blockIdx.y; b < batch_size; b += gridDim.y) {
        const long long off = (long long)b * n;
        float* beta_b  = beta  + off;
        float* alpha_b = alpha + off;

        beta_b[i] = 1.0f / (1.0f + expf(-beta_b[i]));

        float biased = alpha_b[i] + dt_bias[i];
        float sp = (biased > 20.0f) ? biased
                 : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
        alpha_b[i] = sp * (-expf(a_log[i]));
    }
}

extern "C" __global__ void batched_conv1d_silu_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    const int n_channels,
    const int batch_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    for (int b = blockIdx.y; b < batch_size; b += gridDim.y) {
        const long long in_off  = (long long)b * n_channels;
        const long long st_off  = (long long)b * n_channels * 3;

        float x = input[in_off + c];
        float s0 = state[st_off + c * 3];
        float s1 = state[st_off + c * 3 + 1];
        float s2 = state[st_off + c * 3 + 2];

        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;

        output[in_off + c] = y / (1.0f + expf(-y));

        state[st_off + c * 3 + 2] = s1;
        state[st_off + c * 3 + 1] = s0;
        state[st_off + c * 3]     = x;
    }
}

extern "C" __global__ void batched_fused_qk_l2_norm_scale_f32_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const int n_heads,
    const int head_dim,
    const float q_scale,
    const float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    float* q_head = q + batch_off + h * head_dim;
    float* k_head = k + batch_off + h * head_dim;

    float q_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        q_sq += q_head[i] * q_head[i];
    for (int o = 16; o > 0; o >>= 1)
        q_sq += __shfl_xor(q_sq, o);
    float q_inv_norm = rsqrtf(q_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        q_head[i] *= q_inv_norm * q_scale;

    float k_sq = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        k_sq += k_head[i] * k_head[i];
    for (int o = 16; o > 0; o >>= 1)
        k_sq += __shfl_xor(k_sq, o);
    float k_inv_norm = rsqrtf(k_sq + eps);
    for (int i = tid; i < head_dim; i += 32)
        k_head[i] *= k_inv_norm;
}

extern "C" __launch_bounds__(32, 8)
__global__ void batched_gated_delta_net_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ output,
    const int n_tokens,
    const int n_heads,
    const int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    __shared__ float S_tile[TILE_ROWS * HD];

    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = S_global[i];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[col], k1 = k_t[col+1], k2 = k_t[col+2], k3 = k_t[col+3];
        float q0 = q_t[col], q1 = q_t[col+1], q2 = q_t[col+2], q3 = q_t[col+3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            float* sr = S_tile + r * HD + col;
            float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            float kv = s0*k0 + s1*k1 + s2*k2 + s3*k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            float delta = (v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;

            float out_v = s0*q0 + s1*q1 + s2*q2 + s3*q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = out_v;
        }
    }

    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = S_tile[i];
}

extern "C" __global__ void batched_gated_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n_heads,
    const int head_dim,
    const float eps
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
    const float* xh = x + batch_off + h * head_dim;
    const float* zh = z + batch_off + h * head_dim;
    float* oh = out + batch_off + h * head_dim;

    float sq_sum = 0.0f;
    for (int i = tid; i < head_dim; i += 32)
        sq_sum += xh[i] * xh[i];
    for (int o = 16; o > 0; o >>= 1)
        sq_sum += __shfl_xor(sq_sum, o);
    float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += 32) {
        float normed = xh[i] * inv_rms * weight[i];
        float z_val = zh[i];
        float silu_z = z_val / (1.0f + expf(-z_val));
        oh[i] = normed * silu_z;
    }
}

// ── Batched Host-Side Launchers ──────────────────────────────────────────────────

hipError_t gpu_batched_fused_sigmoid_alpha_gate_f32(
    float* beta,
    float* alpha,
    const float* dt_bias,
    const float* a_log,
    int n,
    int batch_size,
    hipStream_t stream
) {
    CHECK_NULL(beta);
    CHECK_NULL(alpha);
    CHECK_NULL(dt_bias);
    CHECK_NULL(a_log);
    if (n <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    const int block = 256;
    const int grid_x = (n + block - 1) / block;
    const int grid_y = batch_size;

    hipLaunchKernelGGL(
        batched_fused_sigmoid_alpha_gate_f32_kernel,
        dim3(grid_x, grid_y, 1), dim3(block, 1, 1), 0, stream,
        beta, alpha, dt_bias, a_log, n, batch_size
    );

    return hipGetLastError();
}

hipError_t gpu_batched_conv1d_silu_f32(
    float* output,
    const float* input,
    const float* weight,
    float* state,
    int n_channels,
    int batch_size,
    hipStream_t stream
) {
    CHECK_NULL(output);
    CHECK_NULL(input);
    CHECK_NULL(weight);
    CHECK_NULL(state);
    if (n_channels <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    const int block = 256;
    const int grid_x = (n_channels + block - 1) / block;
    const int grid_y = batch_size;

    hipLaunchKernelGGL(
        batched_conv1d_silu_f32_kernel,
        dim3(grid_x, grid_y, 1), dim3(block, 1, 1), 0, stream,
        output, input, weight, state, n_channels, batch_size
    );

    return hipGetLastError();
}

hipError_t gpu_batched_fused_qk_l2_norm_scale_f32(
    float* q,
    float* k,
    int n_heads,
    int head_dim,
    int batch_size,
    float q_scale,
    float eps,
    hipStream_t stream
) {
    CHECK_NULL(q);
    CHECK_NULL(k);
    if (n_heads <= 0 || head_dim <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        batched_fused_qk_l2_norm_scale_f32_kernel,
        dim3(n_heads, batch_size, 1), dim3(32, 1, 1), 0, stream,
        q, k, n_heads, head_dim, q_scale, eps
    );

    return hipGetLastError();
}

hipError_t gpu_batched_gated_delta_net_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    int n_tokens,
    int n_heads,
    int head_dim,
    hipStream_t stream
) {
    CHECK_NULL(q);
    CHECK_NULL(k);
    CHECK_NULL(v);
    CHECK_NULL(gate);
    CHECK_NULL(beta);
    CHECK_NULL(state);
    CHECK_NULL(output);
    if (n_tokens <= 0 || n_heads <= 0 || head_dim != HD) return hipErrorInvalidValue;

    const int n_tiles = HD / TILE_ROWS;

    hipLaunchKernelGGL(
        batched_gated_delta_net_f32_kernel,
        dim3(n_heads, n_tiles, 1), dim3(32, 1, 1), 0, stream,
        q, k, v, gate, beta, state, output, n_tokens, n_heads, head_dim
    );

    return hipGetLastError();
}

hipError_t gpu_batched_gated_norm_f32(
    const float* x,
    const float* z,
    const float* weight,
    float* out,
    int n_heads,
    int head_dim,
    int batch_size,
    float eps,
    hipStream_t stream
) {
    CHECK_NULL(x);
    CHECK_NULL(z);
    CHECK_NULL(weight);
    CHECK_NULL(out);
    if (n_heads <= 0 || head_dim <= 0 || batch_size <= 0) return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        batched_gated_norm_f32_kernel,
        dim3(n_heads, batch_size, 1), dim3(32, 1, 1), 0, stream,
        x, z, weight, out, n_heads, head_dim, eps
    );

    return hipGetLastError();
}

} // extern "C"
