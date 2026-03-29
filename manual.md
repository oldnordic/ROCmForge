# ROCmForge User Manual

ROCmForge is a high-performance LLM inference engine optimized for AMD GPUs (via ROCm/HIP) and modern CPUs.

## 1. Prerequisites

### GPU (AMD)
- **ROCm Toolkit**: Version 6.0 or higher is recommended.
- **Hardware**: Radeon RX 7000 series (RDNA3) or Instinct (CDNA) GPUs. Optimized for RX 7900 XT.
- **CMake**: 3.18+ for compiling HIP kernels.

### CPU
- **Architecture**: x86_64 with AVX2 or AVX-512 support.
- **Rust**: 1.81 or higher.

## 2. Installation

Clone the repository and build using Cargo:

```bash
# Standard build (CPU only)
cargo build --release

# GPU-accelerated build (Recommended for AMD users)
cargo build --release --features gpu
```

## 3. Basic Usage

### Running on GPU
To run inference on the GPU, use the `--gpu` (or `--device gpu`) flag:

```bash
./target/release/rocmforge \
    --model models/qwen2.5-0.5b-q4_0.gguf \
    --prompt "What is the capital of France?" \
    --gpu
```

### Running on CPU
```bash
./target/release/rocmforge \
    --model models/qwen2.5-0.5b-q4_0.gguf \
    --prompt "What is the capital of France?" \
    --device cpu
```

### Configuration Options

| Option | Description |
|---------|-------------|
| `--model <path>` | Path to GGUF model file. |
| `--gpu` | Alias for `--device gpu`. |
| `--max-tokens N` | Limit generation length (default: 256). |
| `--top-p F` | Nucleus sampling threshold (default: 0.9). |
| `--temperature F` | Controls randomness (default: 1.0). |
| `--list-tensors` | List tensor names and types in the model file. |

## 4. Performance Tuning

### Batch Prefill
The GPU implementation uses batched GEMM for prompt processing. Larger prompts and models benefit significantly from GPU acceleration. Prefill throughput stays high (~40+ tok/s) even for 7B models on RX 7900 XT.

### Optimized Decode
Generation uses wavefront-parallel Multi-Column GEMV kernels. These are specifically tuned to minimize activation reloads by caching the input vector in LDS (Shared Memory). The implementation automatically handles large models (like 7B) by using chunked LDS loading to stay within hardware limits.

## 5. Troubleshooting

### Segmentation Fault on Startup
Ensure your ROCm environment is correctly sourced. If using a specific GPU, verify `HIP_VISIBLE_DEVICES`. The project includes stack-size fixes for ROCm 6.x compatibility.

### Low Performance
- Verify you are using the `--release` build.
- Ensure the model weights are quantized (Q4_0, Q4_K, etc.). F32 weights will fallback to slower paths.
- Check if your GPU architecture is supported by the pre-compiled kernels or if they are being recompiled on the fly.

## 6. Supported Quantization Types
- **Q4_0**: Sequential split packing (llama.cpp compatible).
- **Q4_1**: Affine scaling with min offset.
- **Q8_0**: 8-bit quantization.
- **Q4_K / Q5_K**: K-Quants with sub-block scaling.
