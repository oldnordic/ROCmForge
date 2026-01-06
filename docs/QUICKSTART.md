# ROCmForge Quick Start Guide

> Get up and running with AMD GPU LLM inference in minutes

---

## Prerequisites

### Hardware
- **AMD GPU** with ROCm support (tested: Radeon RX 7900 XT - gfx1100)
- At least 8GB VRAM for small models (0.5B-3B parameters)

### Software
```bash
# ROCm 7.x (required)
rocm-smi --showproductname  # Verify ROCm is installed

# Rust 1.70+ (required)
rustc --version

# cargo (comes with Rust)
cargo --version
```

### ROCm Installation

If ROCm is not installed:
```bash
# Ubuntu/Debian
sudo apt install rocm-dev rocm-libs rocm-hipsdk

# Verify installation
hipcc --version
rocm-smi
```

See `docs/rocm_setup_guide.md` for detailed setup instructions.

---

## Build

```bash
# Clone and navigate
cd /home/feanor/Projects/ROCmForge

# Build with ROCm feature
cargo build --features rocm --release

# Binary location
./target/release/rocmforge_cli --help
```

---

## Verify GPU Kernels

Run the kernel tests to verify GPU acceleration:

```bash
# All kernel tests (41 tests, should all pass)
cargo test --features rocm --lib

# Specific kernel categories
cargo test --features rocm --lib kernel           # Basic kernels
cargo test --features rocm --lib rope_gpu         # RoPE
cargo test --features rocm --lib flash_attention  # FlashAttention
cargo test --features rocm --lib mlp              # SwiGLU, RMSNorm
```

**Expected output:** `test result: ok. 41 passed; 0 failed`

---

## Load a GGUF Model

### Supported Models

| Model | GGUF Files | Status |
|-------|-----------|--------|
| LLaMA 2/3 | `llama-*.gguf` | ✅ Working |
| Qwen2 | `qwen2*.gguf` | ✅ Phase 4.6 complete |
| Mistral | `mistral*.gguf` | ⚠️ Pending tensor mapping |

### Example: Inspect a GGUF Model

```bash
# View model structure
./target/release/test_gguf_load ~/.config/syncore/models/qwen2.5-0.5b.gguf

# Output shows:
# - Tensor count, types, shapes
# - Model architecture
# - Vocab size
# - Total model size
```

---

## Generate Text (CLI)

**Note:** End-to-end inference is actively being debugged. CLI usage may crash.

```bash
# Basic generation
./target/release/rocmforge_cli generate \
  --gguf ~/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "Hello, world" \
  --max-tokens 10

# With temperature
./target/release/rocmforge_cli generate \
  --gguf ~/.config/syncore/models/model.gguf \
  --prompt "Explain quantum computing" \
  --max-tokens 50 \
  --temperature 0.8
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gguf` | Path to GGUF model file | Required |
| `--prompt` | Input text prompt | Required |
| `--max-tokens` | Maximum tokens to generate | 50 |
| `--temperature` | Sampling temperature (0.0-1.0) | 0.7 |
| `--top-p` | Nucleus sampling threshold | 0.9 |
| `--top-k` | Top-k sampling | 40 |

---

## HTTP Server

Start the inference server:

```bash
# Start server on port 8080
./target/release/rocmforge_cli serve --port 8080 --gguf ~/.config/syncore/models/model.gguf

# Or run in background
./target/release/rocmforge_cli serve --port 8080 --gguf ~/.config/syncore/models/model.gguf &
```

### API Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Generate completion (OpenAI-compatible)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "The capital of France is",
    "max_tokens": 20
  }'

# Stream completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Write a haiku about AI",
    "max_tokens": 50,
    "stream": true
  }'
```

---

## GPU Monitoring

Monitor GPU utilization during inference:

```bash
# Live GPU stats
watch -n 1 rocm-smi

# Detailed memory info
rocm-smi --showmem

# GPU utilization
rocm-smi --showuse
```

---

## Troubleshooting

### "ROCm not found"
```bash
# Set ROCm path
export HIP_PATH=/opt/rocm
export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH
```

### "GPU not detected"
```bash
# Check GPU visibility
rocm-smi --showproductname

# Verify HIP devices
hipconfig --verbose
```

### Tests fail
```bash
# Clean build
cargo clean
cargo build --features rocm

# Check ROCm version (requires 7.x)
rocm-smi | head -5
```

### GGUF loading fails
```bash
# Verify GGUF file integrity
file ~/.config/syncore/models/model.gguf

# Check tensor count
./target/release/test_gguf_load ~/.config/syncore/models/model.gguf
```

### "vocab_size = 0" error
This was fixed in Phase 4.5. If you see this:
```bash
# Pull latest changes
git pull

# Rebuild
cargo build --features rocm --release
```

### Qwen2 model fails to load
Fixed in Phase 4.6. Ensure you have the latest code:
```bash
# Check implementation
git log --oneline -1  # Should mention Phase 4.6

# Rebuild
cargo build --features rocm --release
```

---

## Performance Expectations

### Current Status (Phases 1-4 Complete)

| Component | Status | Notes |
|-----------|--------|-------|
| GPU Kernels | ✅ Complete | All attention, RoPE, FlashAttention, MLP ops |
| Tensor Layout | ✅ Correct | Row-major, explicit indexing |
| CPU Round-trips | ✅ Eliminated | Full layer stays on GPU |
| End-to-end | ⚠️ Debugging | CLI may crash, working on it |

### Expected Performance (once stable)

| GPU Model | Model Size | Tokens/sec (est.) |
|-----------|-----------|-------------------|
| RX 7900 XT | 0.5B | ~30-50 |
| RX 7900 XT | 3B | ~10-20 |
| RX 7900 XT | 7B | ~5-10 |

*Note: These are estimates. Actual performance depends on sequence length, batch size, and KV cache.*

---

## Next Steps

1. **Report Issues:** Include `rocm-smi` output and model details
2. **Check Documentation:** See `docs/README.md` for full documentation index
3. **Development:** See `docs/TODO.md` for current development tasks

---

## Current Project Status

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| 1 | Basic Kernels | ✅ Complete | 3/3 |
| 2 | RoPE | ✅ Complete | 5/5 |
| 3a | Non-Causal FlashAttention | ✅ Complete | 17/17 |
| 3b | Causal Masking | ✅ Complete | 8/8 |
| 4 | MLP Ops | ✅ Complete | 8/8 |
| 4.5 | GGUF Vocab Inference | ✅ Complete | - |
| 4.6 | Qwen2 Tensor Mapping | ✅ Complete | - |
| 5 | Performance Optimization | ⏳ Planned | - |

**Total:** 41/41 kernel tests passing

---

**Last Updated:** 2026-01-04
**GPU:** AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
**ROCm:** 7.1.52802
