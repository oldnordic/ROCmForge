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

## License

GPL-3.0

See [LICENSE](LICENSE) for details.
