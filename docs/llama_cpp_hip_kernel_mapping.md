# llama.cpp HIP Kernel Mapping (2026-04-03)

This note records a local `llama.cpp` HIP profiling pass and maps dominant kernels to `rocmforge`
hotpaths so optimization work can continue with concrete guidance.

## Scope and setup

- `llama.cpp` tree: `/home/feanor/Projects/llama.cpp`
- model: `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`
- binary used: `/home/feanor/Projects/llama.cpp/build/bin/llama-cli`
- GPU: RX 7900 XT (gfx1100), ROCm 7.2
- command shape:
  - `-ngl 99 -p Hello -n 64 -no-cnv --ignore-eos --no-warmup --temp 0 --top-p 1 --seed 123`

## Installed pacman binary note (important)

On this machine, the system `llama.cpp` package (`llama-cpp-git`) provides `/usr/bin/llama-cli` and
`/usr/bin/llama-completion`, and the runtime backend used by that package is Vulkan, not HIP.

Evidence from a fixed-shape run with `/usr/bin/llama-completion`:

- `load_tensors: Vulkan0 model buffer size = ...`
- `llama_context: Vulkan_Host output buffer size = ...`
- measured eval throughput in this run: about `564.79 tok/s`

Implication:

- `pacman llama.cpp` is valid as a practical user-facing baseline on this machine.
- It is **not** a HIP-kernel baseline, so Vulkan-vs-HIP conclusions must be treated separately.

## Why this path instead of hipify

`llama.cpp` already has a HIP backend that compiles the CUDA source set directly:

- HIP backend reuses `ggml-cuda/*.cu` in `ggml/src/ggml-hip/CMakeLists.txt`
- CUDA/HIP symbol mapping is in `ggml/src/ggml-cuda/vendors/hip.h`

So this gives a more production-realistic HIP view than one-off hipify output.

## Throughput snapshots (local)

- no flash-attn (`-fa` off):
  - eval: `215.75 tok/s` (`llama_perf_context_print: eval time = 296.64 ms / 64 runs`)
- flash-attn on (`-fa`):
  - eval: `215.26 tok/s` (`llama_perf_context_print: eval time = 297.31 ms / 64 runs`)

Both runs were near-identical on this local build, so kernel-level profile buckets are the main signal.

## rocprofv3 top kernels

Profiling commands used:

```bash
/opt/rocm/bin/rocprofv3 \
  --runtime-trace --kernel-trace --stats --summary \
  --summary-output-file stdout --summary-units usec --group-by-queue \
  --output-directory /tmp/rocprof-llama-hip-no-fa --output-format csv -- \
  /usr/bin/env LD_LIBRARY_PATH=/home/feanor/Projects/llama.cpp/build/bin:$LD_LIBRARY_PATH \
  /home/feanor/Projects/llama.cpp/build/bin/llama-cli \
  -m /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -ngl 99 -p Hello -n 64 -no-cnv --ignore-eos --no-warmup --temp 0 --top-p 1 --seed 123
```

```bash
/opt/rocm/bin/rocprofv3 \
  --runtime-trace --kernel-trace --stats --summary \
  --summary-output-file stdout --summary-units usec --group-by-queue \
  --output-directory /tmp/rocprof-llama-hip-fa --output-format csv -- \
  /usr/bin/env LD_LIBRARY_PATH=/home/feanor/Projects/llama.cpp/build/bin:$LD_LIBRARY_PATH \
  /home/feanor/Projects/llama.cpp/build/bin/llama-cli \
  -m /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -ngl 99 -p Hello -n 64 -no-cnv --ignore-eos --no-warmup --temp 0 --top-p 1 --seed 123 -fa
```

Top kernel buckets from CSV stats:

- no `-fa` (`/tmp/rocprof-llama-hip-no-fa/andromeda/449283_kernel_stats.csv`)
  - `mul_mat_vec_q<(ggml_type)2, 1>`: `25.64%`
  - `rms_norm_f32<32>`: `13.66%`
  - `quantize_q8_1`: `13.12%`
  - `k_bin_bcast<op_add>`: `11.04%`
  - `mul_mat_vec_q<(ggml_type)8, 1>`: `8.06%`
  - `k_bin_bcast<op_mul>`: `5.31%`
  - `rope_neox<...>`: `4.96%`
  - `cpy_f32_f16<...>`: `4.86%`

- with `-fa` (`/tmp/rocprof-llama-hip-fa/andromeda/449284_kernel_stats.csv`)
  - `mul_mat_vec_q<(ggml_type)2, 1>`: `23.47%`
  - `flash_attn_vec_ext_f32<64,...>`: `14.24%`
  - `rms_norm_f32<32>`: `12.41%`
  - `quantize_q8_1`: `11.95%`
  - `k_bin_bcast<op_add>`: `9.91%`
  - `mul_mat_vec_q<(ggml_type)8, 1>`: `7.43%`
  - `k_bin_bcast<op_mul>`: `4.66%`
  - `cpy_f32_f16<...>`: `4.58%`
  - `rope_neox<...>`: `4.50%`

HIP runtime/API buckets (same runs):

- no `-fa` (`449283_hip_api_stats.csv`)
  - `hipLaunchKernel`: `68.64%`
  - `hipMemcpyAsync`: `9.67%`
  - `hipStreamSynchronize`: `9.00%`
- with `-fa` (`449284_hip_api_stats.csv`)
  - `hipLaunchKernel`: `70.13%`
  - `hipStreamSynchronize`: `9.48%`
  - `hipMemcpyAsync`: `7.85%`

## Mapping to rocmforge

GGML type IDs from `ggml/include/ggml.h`:

- `(ggml_type)2` = `GGML_TYPE_Q4_0`
- `(ggml_type)8` = `GGML_TYPE_Q8_0`
- `(ggml_type)3` = `GGML_TYPE_Q4_1`

Kernel mapping:

- `mul_mat_vec_q<(ggml_type)2,1>` (`ggml-cuda/mmvq.cu`)
  - role: dominant Q4_0 mat-vec decode projection
  - nearest `rocmforge` analog:
    - `gemv_q4_0_f32_q8_inline_residual_multi_row_kernel<8>`
    - `gemv_gate_up_swiglu_q4_0_f32_q8_inline_vulkan_style_v2_kernel<4>`
- `quantize_q8_1` (`ggml-cuda/quantize.cu`)
  - role: activation quantization traffic on decode path
  - nearest `rocmforge` analog:
    - inline Q8 activation quantization fastpaths already added in gate/up + residual paths
- `k_bin_bcast<op_add/op_mul>` (`ggml-cuda/binbcast.cu`)
  - role: large count of elementwise launches (residual + swiglu-style ops)
  - nearest `rocmforge` analog:
    - `add_on_stream`, `mul_on_stream`, `silu_on_stream` chains
- `rms_norm_f32` + `rope_neox`
  - role: expected decode staples
  - nearest `rocmforge` analog:
    - `rms_norm_on_stream`, `rope_heads_on_stream`
- `flash_attn_vec_ext_f32` (when `-fa`)
  - role: attention kernel family
  - nearest `rocmforge` analog:
    - `flash_attn_decode_strided_multi_head_state_kernel`

## Actionable guidance for rocmforge next passes

1. Continue prioritizing Q4_0 decode GEMV buckets first; this matches both `llama.cpp` HIP and
   current `rocmforge` hotspots.
2. Reduce launch count for elementwise add/mul/silu chains where safe; `llama.cpp` also shows
   `k_bin_bcast` buckets as meaningful overhead.
3. Keep reducing decode-time standalone quantization launches; Q8 quantization remains a major
   bucket in `llama.cpp`.
4. Track host-side `hipLaunchKernel` + `hipStreamSynchronize` after each kernel win to avoid
   trading kernel speedups for launch/sync overhead.
5. Keep safety-first fallback behavior as non-negotiable on display-attached GPUs.

## Attempted MMQ vs CUBLAS forced builds (blocked)

I attempted separate fresh HIP builds with:

- `-DGGML_CUDA_FORCE_MMQ=ON`
- `-DGGML_CUDA_FORCE_CUBLAS=ON`

Both are currently blocked on this local `llama.cpp` checkout by a compile error in
`ggml/src/ggml-cuda/add-id.cu` (`GGML_TENSOR_TERNARY_OP_LOCALS` unresolved in this tree state).
So this note uses the existing working HIP build for empirical guidance.
