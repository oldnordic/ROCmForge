rocmforge provides fast, efficient inference for large language models on AMD GPUs (via HIP) and CPUs, with explicit device selection and no fallback paths.

## Features

- **GGUF model loading** - Memory-mapped GGUF files with zero-copy TensorView API
- **BPE/SentencePiece tokenization** - Auto-detects and loads tokenizer from GGUF metadata
- **Metadata-driven configuration** - All model dimensions, vocabulary size, RoPE style discovered from GGUF
- **Dual execution paths**
  - **CPU**: AVX2+FMA SIMD kernels with runtime feature detection
  - **GPU**: AMD HIP (ROCm) - coming soon
  - **Explicit device choice** - User selects CPU or GPU at startup, no runtime switching
- **Q4_0/Q8_0 quantization support** - Efficient dequantization on-the-fly
- **Flash attention** - GQA-aware for efficient KV caching
- **SwiGLU activation** - Fast SiLU implementation

## Architecture

```
rocmforge/
├── src/
│   ├── config/          # Model config, chat templates
│   ├── cpu/            # CPU inference kernels & dispatch
│   ├── gpu/            # HIP kernels (planned)
│   ├── loader/          # GGUF loading, metadata parsing
│   └── tokenizer/       # BPE/SentencePiece tokenization
├── tests/             # Integration tests
└── main.rs            # CLI entry point
```

## Building

```bash
cargo build --release
```

**Requirements**
- Rust 1.81+ edition 2021
- AMD ROCm HIP toolkit (for GPU - coming soon)
- GGUF model file for the target model

## Usage

### Basic inference

```bash
rocmforge --model path/to/model.gguf --device cpu \
    --prompt "What is the capital of France?" --max-tokens 100
```

### List model tensors

```bash
rocmforge --model path/to/model.gguf --list-tensors
```

### CLI Options

| Option | Description |
|---------|-------------|
| `--model <path>` | Path to GGUF model file |
| `--device <cpu|gpu>` | Execution device (CPU or GPU) |
| `--prompt <text>` | Input prompt |
| `--max-tokens N` | Max tokens to generate |
| `--temperature F` | Sampling temperature |
| `--top-p F` | Nucleus sampling threshold |
| `--no-template` | Disable chat template |
| `--list-tensors` | List all tensors in model |

## License

GPL-3.0-or-later

See [LICENSE](LICENSE) for details.

## Status

**CPU inference**: ✅ Complete
**GPU inference**: 🚧 In development
