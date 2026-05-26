#include "common.hip"

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

extern "C" __global__ void unpack_q4_split_device_kernel(
    const void* __restrict__ input_split,
    void* __restrict__ output_q4_0,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const uint8_t* in_bytes = static_cast<const uint8_t*>(input_split);
    uint8_t* out_bytes = static_cast<uint8_t*>(output_q4_0) + block_idx * Q4_0_BLOCK_SIZE;

    // scales offset = 0, zero_points offset = num_blocks * 2, nibbles offset = num_blocks * 4
    const uint8_t* scale_ptr = in_bytes + block_idx * 2;
    const uint8_t* nibble_ptr = in_bytes + num_blocks * 4 + block_idx * 16;

    // Write scale (2 bytes)
    out_bytes[0] = scale_ptr[0];
    out_bytes[1] = scale_ptr[1];

    // Write nibbles (16 bytes)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out_bytes[2 + i] = nibble_ptr[i];
    }
}

extern "C" __global__ void unpack_q4_fused_gate_up_device_kernel(
    const void* __restrict__ input_fused,
    void* __restrict__ output_gate,
    void* __restrict__ output_up,
    void* __restrict__ output_interleaved,
    void* __restrict__ output_interleaved_tile4,
    int intermediate_size,
    int hidden_size
) {
    int rfm_blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_row = hidden_size / QK4_0;
    int rfm_blocks_row = num_blocks_row / 8;
    int rfm_blocks_total = intermediate_size * rfm_blocks_row;

    if (rfm_blk_idx >= rfm_blocks_total) return;

    int r = rfm_blk_idx / rfm_blocks_row; // row index (ranges 0..intermediate_size - 1)
    int b = rfm_blk_idx % rfm_blocks_row; // block-in-row index

    const uint8_t* in_bytes = static_cast<const uint8_t*>(input_fused);
    const uint8_t* scale_ptr = in_bytes + rfm_blk_idx * 32;
    const uint8_t* nibble_ptr = in_bytes + static_cast<size_t>(rfm_blocks_total) * 64 + rfm_blk_idx * 256;

    uint8_t* out_gate_base = static_cast<uint8_t*>(output_gate);
    uint8_t* out_up_base = static_cast<uint8_t*>(output_up);
    uint8_t* out_interleaved_base = static_cast<uint8_t*>(output_interleaved);
    uint8_t* out_tile4_base = static_cast<uint8_t*>(output_interleaved_tile4);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int col_block = b * 8 + i;
        int global_block = r * num_blocks_row + col_block;

        // Extract scale and nibbles for gate
        uint8_t gate_s0 = scale_ptr[i * 2];
        uint8_t gate_s1 = scale_ptr[i * 2 + 1];
        uint8_t gate_n[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            gate_n[j] = nibble_ptr[i * 16 + j];
        }

        // Extract scale and nibbles for up
        uint8_t up_s0 = scale_ptr[16 + i * 2];
        uint8_t up_s1 = scale_ptr[16 + i * 2 + 1];
        uint8_t up_n[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            up_n[j] = nibble_ptr[128 + i * 16 + j];
        }

        // 1. Write gate standard Q4_0
        if (out_gate_base) {
            uint8_t* gate_out = out_gate_base + global_block * Q4_0_BLOCK_SIZE;
            gate_out[0] = gate_s0;
            gate_out[1] = gate_s1;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                gate_out[2 + j] = gate_n[j];
            }
        }

        // 2. Write up standard Q4_0
        if (out_up_base) {
            uint8_t* up_out = out_up_base + global_block * Q4_0_BLOCK_SIZE;
            up_out[0] = up_s0;
            up_out[1] = up_s1;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                up_out[2 + j] = up_n[j];
            }
        }

        // 3. Write interleaved block pair (gate 18B, up 18B)
        if (out_interleaved_base) {
            uint8_t* int_out = out_interleaved_base + global_block * (2 * Q4_0_BLOCK_SIZE);
            int_out[0] = gate_s0;
            int_out[1] = gate_s1;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int_out[2 + j] = gate_n[j];
            }
            int_out[18] = up_s0;
            int_out[19] = up_s1;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int_out[20 + j] = up_n[j];
            }
        }

        // 4. Write interleaved tile4 block group
        if (out_tile4_base) {
            int ff_base = (r / 4) * 4;
            int tile_ff = r % 4;
            if (ff_base + tile_ff < intermediate_size) {
                size_t tile4_offset = ((static_cast<size_t>(ff_base / 4) * num_blocks_row + col_block) * 4 + tile_ff) * (2 * Q4_0_BLOCK_SIZE);
                uint8_t* tile4_out = out_tile4_base + tile4_offset;

                tile4_out[0] = gate_s0;
                tile4_out[1] = gate_s1;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    tile4_out[2 + j] = gate_n[j];
                }
                tile4_out[18] = up_s0;
                tile4_out[19] = up_s1;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    tile4_out[20 + j] = up_n[j];
                }
            }
        }
    }
}

extern "C" hipError_t unpack_q4_split_launch(
    const void* input_split,
    void* output_q4_0,
    int num_blocks,
    hipStream_t stream
) {
    if (num_blocks <= 0) return hipErrorInvalidValue;
    if (input_split == nullptr || output_q4_0 == nullptr) return hipErrorInvalidValue;

    int threads_per_block = 256;
    int blocks_x = (num_blocks + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(unpack_q4_split_device_kernel),
        dim3(blocks_x),
        dim3(threads_per_block),
        0, stream,
        input_split,
        output_q4_0,
        num_blocks
    );
    return hipGetLastError();
}

extern "C" hipError_t unpack_q4_fused_gate_up_launch(
    const void* input_fused,
    void* output_gate,
    void* output_up,
    void* output_interleaved,
    void* output_interleaved_tile4,
    int intermediate_size,
    int hidden_size,
    hipStream_t stream
) {
    if (intermediate_size <= 0 || hidden_size <= 0) return hipErrorInvalidValue;
    if (input_fused == nullptr) return hipErrorInvalidValue;

    int num_blocks_row = hidden_size / QK4_0;
    int rfm_blocks_row = num_blocks_row / 8;
    int rfm_blocks_total = intermediate_size * rfm_blocks_row;

    if (rfm_blocks_total <= 0) return hipErrorInvalidValue;

    int threads_per_block = 256;
    int blocks_x = (rfm_blocks_total + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(unpack_q4_fused_gate_up_device_kernel),
        dim3(blocks_x),
        dim3(threads_per_block),
        0, stream,
        input_fused,
        output_gate,
        output_up,
        output_interleaved,
        output_interleaved_tile4,
        intermediate_size,
        hidden_size
    );
    return hipGetLastError();
}
