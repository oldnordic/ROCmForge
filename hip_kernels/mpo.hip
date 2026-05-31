#include <hip/hip_runtime.h>
#include <cmath>

#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

// ── MPO Apply Kernel ─────────────────────────────────────────────────────────────
// y = MPO * x where MPO is a chain of site tensors.
// Each site i has shape [chi_left, d_i, chi_right].
// We contract from right to left: start with x, contract with last site,
// then propagate through the chain.
//
// Layout:
//   sites: flattened array of all site tensor values
//   site_dims: [chi_0, d_0, chi_1, chi_1, d_1, chi_2, ...]
//   n_sites: number of sites
//   out_dim: d_0 (physical dimension of first site)
//   in_dim:  d_{n-1} (physical dimension of last site)

extern "C" __global__ void mpo_apply_f32_kernel(
    const float* __restrict__ sites,
    const unsigned int* __restrict__ site_dims,
    const int n_sites,
    const int out_dim,
    const int in_dim,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // site i: chi_left = site_dims[4*i], d_i = site_dims[4*i+1],
    //         chi_right = site_dims[4*i+2], padding = site_dims[4*i+3]
    // The Rust side packs dims as [chi_l, d, chi_r, 1] per site.

    if (n_sites == 2) {
        // Site 0: [1, d0, chi1, 1], Site 1: [chi1, d1, 1, 1]
        const int d0   = site_dims[1];
        const int chi1 = site_dims[2];
        const int d1   = site_dims[5];

        // Validate dimensions against buffer bounds
        if (d0 <= 0 || chi1 <= 0 || d1 <= 0 || out_dim > d0 || in_dim > d1) {
            if (out_idx < out_dim) y[out_idx] = 0.0f;
            return;
        }

        const float* site0 = sites;                     // [1, d0, chi1]
        const float* site1 = sites + d0 * chi1;         // [chi1, d1, 1]

        // Cooperative block-wide contraction:
        // temp[chi1] = sum_{d1} site1[chi1, d1, 1] * x[d1]
        __shared__ float s_temp[64];
        if (threadIdx.x < chi1) {
            float sum = 0.0f;
            for (int j = 0; j < d1; ++j) {
                sum += site1[threadIdx.x * d1 + j] * x[j];
            }
            s_temp[threadIdx.x] = sum;
        }
        __syncthreads();

        if (out_idx >= out_dim) return;

        // y[d0] = sum_{chi1} site0[1, d0, chi1] * temp[chi1]
        float sum = 0.0f;
        for (int c = 0; c < chi1; ++c) {
            sum += site0[out_idx * chi1 + c] * s_temp[c];
        }
        y[out_idx] = sum;
    }
    else if (n_sites == 3) {
        // Site 0: [1, d0, chi1, 1]
        // Site 1: [chi1, d1, chi2, 1]
        // Site 2: [chi2, d2, 1, 1]
        const int d0   = site_dims[1];
        const int chi1 = site_dims[2];
        const int d1   = site_dims[5];
        const int chi2 = site_dims[6];
        const int d2   = site_dims[9];

        // Validate dimensions
        if (d0 <= 0 || chi1 <= 0 || d1 <= 0 || chi2 <= 0 || d2 <= 0 ||
            out_dim > d0 || in_dim > d2) {
            if (out_idx < out_dim) y[out_idx] = 0.0f;
            return;
        }

        const float* site0 = sites;
        const float* site1 = sites + d0 * chi1;
        const float* site2 = site1 + chi1 * d1 * chi2;

        // Cooperative block-wide contraction:
        // t2[chi2] = sum_{d2} site2[chi2, d2, 1] * x[d2]
        __shared__ float s_t2[64];
        __shared__ float s_t1[64];

        if (threadIdx.x < chi2) {
            float sum = 0.0f;
            for (int j = 0; j < d2; ++j) {
                sum += site2[threadIdx.x * d2 + j] * x[j];
            }
            s_t2[threadIdx.x] = sum;
        }
        __syncthreads();

        // t1[chi1] = sum_{d1, chi2} site1[chi1, d1, chi2] * t2[chi2]
        if (threadIdx.x < chi1) {
            float sum = 0.0f;
            for (int j = 0; j < d1; ++j) {
                for (int c2 = 0; c2 < chi2; ++c2) {
                    sum += site1[threadIdx.x * d1 * chi2 + j * chi2 + c2] * s_t2[c2];
                }
            }
            s_t1[threadIdx.x] = sum;
        }
        __syncthreads();

        if (out_idx >= out_dim) return;

        // y[d0] = sum_{chi1} site0[1, d0, chi1] * t1[chi1]
        float sum = 0.0f;
        for (int c1 = 0; c1 < chi1; ++c1) {
            sum += site0[out_idx * chi1 + c1] * s_t1[c1];
        }
        y[out_idx] = sum;
    }
    else {
        // Generic n-site fallback (up to 8 sites, chi <= 64)
        int max_chi = 1;
        for (int s = 0; s < n_sites; ++s) {
            int chi = site_dims[4 * s + 2];
            if (chi > max_chi) max_chi = chi;
        }
        if (max_chi > 64) {
            if (out_idx < out_dim) y[out_idx] = 0.0f;
            return;
        }

        int offsets[8];
        offsets[0] = 0;
        for (int s = 1; s < n_sites; ++s) {
            int chi_l = site_dims[4 * (s - 1)];
            int d     = site_dims[4 * (s - 1) + 1];
            int chi_r = site_dims[4 * (s - 1) + 2];
            offsets[s] = offsets[s - 1] + chi_l * d * chi_r;
        }

        // Cooperative block-wide contraction generic loop
        __shared__ float s_bond[64];

        // Load initial x into s_bond
        if (threadIdx.x < in_dim && threadIdx.x < 64) {
            s_bond[threadIdx.x] = x[threadIdx.x];
        }
        __syncthreads();

        for (int s = n_sites - 1; s >= 1; --s) {
            int chi_l = site_dims[4 * s];
            int d     = site_dims[4 * s + 1];
            int chi_r = site_dims[4 * s + 2];
            const float* site = sites + offsets[s];

            __shared__ float s_new_bond[64];
            if (threadIdx.x < chi_l) {
                float sum = 0.0f;
                for (int j = 0; j < d; ++j) {
                    for (int c_r = 0; c_r < chi_r; ++c_r) {
                        sum += site[threadIdx.x * d * chi_r + j * chi_r + c_r] * s_bond[c_r];
                    }
                }
                s_new_bond[threadIdx.x] = sum;
            }
            __syncthreads();

            // Copy s_new_bond to s_bond for next iteration
            if (threadIdx.x < chi_l) {
                s_bond[threadIdx.x] = s_new_bond[threadIdx.x];
            }
            __syncthreads();
        }

        if (out_idx >= out_dim) return;

        // Final site 0 contraction: y = site0 * bond
        int chi_r = site_dims[2];
        const float* site0 = sites; // offset is 0
        float sum = 0.0f;
        for (int c = 0; c < chi_r; ++c) {
            sum += site0[out_idx * chi_r + c] * s_bond[c];
        }
        y[out_idx] = sum;
    }
}

// ── Host-Side Launcher ───────────────────────────────────────────────────────────

extern "C" hipError_t gpu_mpo_apply_f32(
    const float* sites,
    const unsigned int* site_dims,
    int n_sites,
    int out_dim,
    int in_dim,
    const float* x,
    float* y,
    hipStream_t stream
) {
    if (!sites || !site_dims || !x || !y) {
        return hipErrorInvalidValue;
    }
    if (n_sites <= 0 || out_dim <= 0 || in_dim <= 0) {
        return hipErrorInvalidValue;
    }

    const int threads = 256;
    const int blocks = (out_dim + threads - 1) / threads;

    hipLaunchKernelGGL(
        mpo_apply_f32_kernel,
        dim3(blocks, 1, 1),
        dim3(threads, 1, 1),
        0,
        stream,
        sites,
        site_dims,
        n_sites,
        out_dim,
        in_dim,
        x,
        y
    );

    return hipGetLastError();
}
