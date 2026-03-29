rocmforge - LLM inference on AMD GPUs (HIP) and CPUs.

## Building

```bash
cargo build --release
```

**Requirements**
- Rust 1.81+ edition 2021
- AMD ROCm HIP toolkit (for GPU)
- GGUF model file

## Usage

```bash
rocmforge --model path/to/model.gguf --device cpu \
    --prompt "text" --max-tokens 100
```

| Option | Description |
|---------|-------------|
| `--model <path>` | Path to GGUF model file |
| `--device <cpu|gpu>` | Execution device (CPU or GPU) |
| `--prompt <text>` | Input prompt |
| `--max-tokens N` | Max tokens to generate [default: 256] |
| `--temperature F` | Sampling temperature [default: 1.0] |
| `--top-p F` | Nucleus sampling threshold [default: 0.9] |
| `--no-template` | Disable chat template |
| `--list-tensors` | List all tensors in model |
| `--debug` | Show debug info |

## Real Models Tested

The following models have been verified for performance and output coherence on AMD hardware.

| Model | Quant | Device | Prefill | Decode | Coherence |
|-------|-------|--------|---------|--------|-----------|
| **Qwen2.5-0.5B-Instruct** | Q4_0 | RX 7900 XT | 160.9 tok/s | 188.2 tok/s | High |
| **Qwen2.5-0.5B-Instruct** | Q4_K | RX 7900 XT | 155.4 tok/s | 182.1 tok/s | High |

*Benchmarks conducted on ROCm 6.1 with RDNA3 architecture.*

## License

GPL-3.0

See [LICENSE](LICENSE) for details.
