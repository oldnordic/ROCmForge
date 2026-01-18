# ROCmForge User Guide

**Version:** 0.1.0
**Last Updated:** 2026-01-19

This guide provides comprehensive information for installing, configuring, and using ROCmForge, an AMD GPU inference engine for Large Language Models.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Common Use Cases](#common-use-cases)
6. [CLI Reference](#cli-reference)
7. [HTTP API](#http-api)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | AMD GPU with ROCm support (RDNA2/3 or CDNA2/3) | RX 7900 XT/XTX or better |
| VRAM | 8 GB | 16 GB+ |
| System RAM | 16 GB | 32 GB+ |
| Storage | 20 GB free | 50 GB+ SSD |

**Supported GPU Architectures:**
- **RDNA3**: RX 7000 series (gfx1100, gfx1101, gfx1102)
- **RDNA2**: RX 6000 series (gfx1030)
- **CDNA3**: MI300 series (gfx940, gfx942, gfx950)
- **CDNA2**: MI200 series (gfx90a)

Check your GPU:
```bash
lspci | grep -i vga
```

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Operating System | Ubuntu 22.04/24.04, RHEL 8/9 | ROCm is Linux-only |
| ROCm | 5.0 or later (6.0+ recommended) | [ROCm Installation Guide](./rocm_setup_guide.md) |
| Rust | 1.82 or later | Required for std::simd support |
| Git | Latest | For cloning repository |

### Verify ROCm Installation

```bash
# Check ROCm version
cat /opt/rocm/.info/version
# Or
rocminfo | head -20

# Verify GPU is visible
rocm-smi

# Check hipcc compiler
hipcc --version

# Verify device nodes exist
ls /dev/kfd
```

---

## Installation

### Step 1: Install ROCm (if not already installed)

Follow the [ROCm Setup Guide](./rocm_setup_guide.md) for detailed instructions.

Quick install for Ubuntu:
```bash
# Download AMDGPU install script
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb

# Install package
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm --no-dkms
```

### Step 2: Set ROCm Environment

Add to your `~/.bashrc`:
```bash
# ROCm environment
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/rocm
export HIPCC=/opt/rocm/bin/hipcc
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Clone and Build ROCmForge

```bash
# Clone the repository
git clone https://github.com/oldnordic/ROCmForge.git
cd ROCmForge

# Build release binary (with ROCm support)
cargo build --release --features rocm

# The binaries will be at:
# - target/release/rocmforge_cli
# - target/release/inspect_gguf
# - target/release/test_inference
```

### Step 4: Verify Installation

```bash
# Check CLI help
./target/release/rocmforge-cli --help

# Check ROCm is accessible
rocminfo
```

---

## Configuration

### Environment Variables

ROCmForge can be configured via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
# Edit .env with your settings
```

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCMFORGE_GGUF` | Path to GGUF model file | (required for generation) |
| `ROCMFORGE_TOKENIZER` | Path to tokenizer.json | Auto-detected |
| `ROCMFORGE_MODELS` | Directory containing GGUF files | `./models` |
| `ROCMFORGE_GPU_DEVICE` | GPU device number | 0 |
| `ROCFORGE_LOG_LEVEL` | Simple log level | info |
| `ROCFORGE_LOG_FORMAT` | Output format (human/json) | human |
| `ROCFORGE_LOG_FILE` | Optional log file path | none |
| `RUST_LOG` | Standard tracing filter | info |

### Logging Configuration

#### Simple Log Level

```bash
# Available levels: error, warn, info, debug, trace
export ROCFORGE_LOG_LEVEL=debug
```

#### Log Format

```bash
# Human-readable colored output (default)
export ROCFORGE_LOG_FORMAT=human

# JSON format for log aggregation
export ROCFORGE_LOG_FORMAT=json
```

#### File Output

```bash
# Log to a file (always in JSON format)
export ROCFORGE_LOG_FILE=/var/log/rocmforge/app.log

# Combined with console output
export ROCFORGE_LOG_LEVEL=info
export ROCFORGE_LOG_FORMAT=human
export ROCFORGE_LOG_FILE=/tmp/rocmforge.log
```

#### Advanced Module Filtering

```bash
# Set default level to info
export RUST_LOG=info

# Set module-specific levels
export RUST_LOG=rocmforge=debug,hyper=info

# Enable trace for everything
export RUST_LOG=trace

# Common pattern: debug for ROCmForge, info for dependencies
export RUST_LOG=rocmforge=debug,warn
```

### Log Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `error` | Only errors | Production (minimal output) |
| `warn` | Warnings and errors | Production (standard) |
| `info` | Normal operations | Development/testing |
| `debug` | Detailed diagnostics | Troubleshooting |
| `trace` | Everything including internal details | Deep debugging |

---

## Quick Start

### 1. Download a GGUF Model

Download a compatible LLaMA-based model in GGUF format. Example sources:
- [HuggingFace - TheBloke](https://huggingface.co/TheBloke)
- [HuggingFace - Quantized Models](https://huggingface.co/models?search=gguf)

Example models:
- `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (~300 MB, good for testing)
- `Llama-2-7B-Chat-Q4_K_M.gguf` (~4 GB, production)

Place the model in your models directory:
```bash
mkdir -p models
# Copy your downloaded model to models/
```

### 2. Verify Model

```bash
# List discovered models
./target/release/rocmforge-cli models --dir ./models

# Inspect a specific model
./target/release/inspect_gguf ./models/your-model.gguf
```

### 3. Generate Text (CLI)

```bash
# Basic generation
./target/release/rocmforge-cli generate \
  --gguf ./models/your-model.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 100

# Streaming generation
./target/release/rocmforge-cli generate \
  --gguf ./models/your-model.gguf \
  --prompt "Tell me a short story" \
  --stream \
  --max-tokens 200
```

### 4. Start HTTP Server

```bash
# Start server
./target/release/rocmforge-cli serve \
  --gguf ./models/your-model.gguf \
  --addr 127.0.0.1:8080

# In another terminal, test the server
curl http://localhost:8080/health
```

---

## Common Use Cases

### Text Generation

Generate text with various sampling parameters:

```bash
./target/release/rocmforge-cli generate \
  --gguf ./models/llama-2-7b.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 500 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9
```

**Sampling Parameters:**
- `--max-tokens`: Maximum tokens to generate (1-8192)
- `--temperature`: Randomness (0.0=deterministic, 1.0=creative, 2.0=very random)
- `--top-k`: Limit vocabulary to top K tokens
- `--top-p`: Nucleus sampling threshold (0.0-1.0)

### Streaming Generation

For real-time output as tokens are generated:

```bash
./target/release/rocmforge-cli generate \
  --gguf ./models/model.gguf \
  --prompt "Write a haiku about programming" \
  --stream \
  --max-tokens 100
```

Output appears token-by-token. Press Ctrl+C to cancel.

### Using Environment Variables

Set defaults to avoid repeating options:

```bash
# Set default model
export ROCMFORGE_GGUF=./models/my-model.gguf

# Set default models directory
export ROCMFORGE_MODELS=./models

# Now you can omit --gguf
./target/release/rocmforge-cli generate --prompt "Hello!"
```

### Interactive Chat Session

Create a simple chat loop (bash):

```bash
export ROCMFORGE_GGUF=./models/model.gguf
export ROCMFORGE_TOKENIZER=./models/tokenizer.json

while true; do
    read -p "You: " prompt
    if [ "$prompt" = "quit" ]; then
        break
    fi
    echo -n "Assistant: "
    ./target/release/rocmforge-cli generate \
      --prompt "$prompt" \
      --stream \
      --max-tokens 500
    echo
done
```

---

## CLI Reference

### Commands Overview

```bash
rocmforge-cli [OPTIONS] <COMMAND>
```

| Command | Description |
|---------|-------------|
| `serve` | Start HTTP inference server |
| `generate` | Generate text from a prompt |
| `status` | Query request status by ID |
| `cancel` | Cancel a running request |
| `models` | List available GGUF models |

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host <URL>` | HTTP server base URL | `http://127.0.0.1:8080` |

### Command: serve

Start the HTTP inference server.

```bash
rocmforge-cli serve [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--addr <ADDRESS>` | Bind address | `127.0.0.1:8080` |
| `--gguf <PATH>` | GGUF model to load | `$ROCMFORGE_GGUF` |
| `--tokenizer <PATH>` | Tokenizer JSON path | Auto-detected |

**Example:**
```bash
rocmforge-cli serve \
  --gguf ./models/qwen2-0.5b.gguf \
  --addr 0.0.0.0:8080
```

### Command: generate

Generate text from a prompt.

```bash
rocmforge-cli generate [OPTIONS] --prompt <TEXT>
```

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt <TEXT>` | Prompt text (required) | - |
| `--gguf <PATH>` | Local GGUF model (bypasses HTTP) | - |
| `--tokenizer <PATH>` | Tokenizer JSON for local mode | Auto-detected |
| `--max-tokens <N>` | Maximum tokens to generate | 128 |
| `--temperature <N>` | Sampling temperature | 1.0 |
| `--top-k <N>` | Top-k sampling | 50 |
| `--top-p <N>` | Top-p (nucleus) sampling | 0.9 |
| `--stream` | Stream tokens as generated | false |

**Examples:**

```bash
# Basic generation (via HTTP)
rocmforge-cli generate --prompt "Hello, world!"

# Local model with streaming
rocmforge-cli generate \
  --gguf ./models/model.gguf \
  --prompt "Tell me a joke" \
  --stream

# Creative writing (high temperature)
rocmforge-cli generate \
  --gguf ./models/model.gguf \
  --prompt "Write a sci-fi story" \
  --temperature 1.2 \
  --top-p 0.95 \
  --max-tokens 1000
```

### Command: status

Check the status of a previous request.

```bash
rocmforge-cli status --request-id <ID>
```

| Option | Description | Default |
|--------|-------------|---------|
| `--request-id <ID>` | Request ID from generate command | (required) |

**Example:**
```bash
rocmforge-cli status --request-id 42
```

**Output:**
```
request_id: 42, finished: true, reason: length
text:
Your generated text here...
```

### Command: cancel

Cancel a running request.

```bash
rocmforge-cli cancel --request-id <ID>
```

| Option | Description | Default |
|--------|-------------|---------|
| `--request-id <ID>` | Request ID to cancel | (required) |

**Example:**
```bash
rocmforge-cli cancel --request-id 42
```

### Command: models

List available GGUF models.

```bash
rocmforge-cli models [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--dir <PATH>` | Models directory | `$ROCMFORGE_MODELS` or `./models` |

**Example:**
```bash
rocmforge-cli models --dir ./models
```

**Output:**
```
Discovered models:
- qwen2-0.5b-instruct-q4
  gguf: ./models/qwen2-0.5b-instruct-q4.gguf
  tokenizer: ./models/tokenizer.json
  arch: qwen2 | layers: 24 | heads: 14 | hidden: 896 | ctx: 32768 | vocab: 151936 | file_type: 15 | embedded tokenizer: no
  cache: cached
```

### inspect_gguf Utility

Inspect GGUF model metadata without loading the full model.

```bash
./target/release/inspect_gguf <path-to-gguf-file>
```

**Example Output:**
```
Inspecting GGUF file: ./models/model.gguf

=== Metadata ===
Architecture: llama
Vocab size: 32000
Hidden size: 4096
Num layers: 32
Num heads: 32
Num KV heads: Some(4)
Intermediate size: 11008
Max position embeddings: 2048
RMS norm epsilon: 0.000001
Use rotary embeddings: true

=== Tensor Names (first 20) ===
1. tok_embeddings.weight [32000, 4096]
2. layers.0.attention.wq.weight [4096, 4096]
3. layers.0.attention.wk.weight [1024, 4096]
...

Total tensors: 361
```

---

## HTTP API

### Endpoints

The HTTP server exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Generate text (non-streaming) |
| POST | `/generate/stream` | Generate text (SSE streaming) |
| GET | `/status/:request_id` | Get request status |
| POST | `/cancel/:request_id` | Cancel a request |
| GET | `/models` | List available models |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |
| GET | `/traces` | OpenTelemetry traces |

### POST /generate

Generate text from a prompt.

**Request:**
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
  }'
```

**Response:**
```json
{
  "request_id": 1,
  "text": "The capital of France is Paris.",
  "tokens": [1234, 5678, ...],
  "finished": true,
  "finish_reason": "length"
}
```

### POST /generate/stream

Generate text with Server-Sent Events streaming.

**Request:**
```bash
curl -X POST http://localhost:8080/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a story",
    "max_tokens": 200
  }'
```

**Stream Events:**
```
data: {"request_id":1,"token":1234,"text":"Once","finished":false}

data: {"request_id":1,"token":5678,"text":" upon","finished":false}

data: {"request_id":1,"token":9012,"text":" a","finished":false}

...

data: {"request_id":1,"token":3456,"text":".","finished":true,"finish_reason":"length"}
```

### GET /status/:request_id

Get the status of a request.

**Request:**
```bash
curl http://localhost:8080/status/1
```

**Response:**
```json
{
  "request_id": 1,
  "text": "Generated text here...",
  "tokens": [1234, 5678, ...],
  "finished": true,
  "finish_reason": "length"
}
```

### POST /cancel/:request_id

Cancel a running request.

**Request:**
```bash
curl -X POST http://localhost:8080/cancel/1
```

**Response:**
```json
{
  "request_id": 1,
  "text": "Partial text...",
  "tokens": [1234, 5678],
  "finished": true,
  "finish_reason": "cancelled"
}
```

### GET /models

List discovered models.

**Request:**
```bash
curl http://localhost:8080/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "qwen2-0.5b-instruct",
      "path": "/path/to/model.gguf",
      "tokenizer": "/path/to/tokenizer.json",
      "metadata": {
        "architecture": "qwen2",
        "num_layers": 24,
        "num_heads": 14,
        "hidden_size": 896,
        "head_dim": 64,
        "max_position_embeddings": 32768,
        "vocab_size": 151936,
        "file_type": 15,
        "has_tokenizer": false
      },
      "cache_status": {
        "cached": true,
        "refreshed": false
      }
    }
  ],
  "tokenizer_cache": {
    "hits": 42,
    "misses": 1,
    "bytes": 1234567
  }
}
```

### GET /health

Health check with detailed status.

**Request:**
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "rocmforge",
  "version": "0.1.0",
  "checks": {
    "engine": {
      "running": true,
      "model_loaded": true
    },
    "gpu": {
      "available": true,
      "memory": {
        "free_bytes": 12345678901,
        "total_bytes": 25769803776,
        "free_mb": 11776,
        "total_mb": 24576,
        "used_mb": 12800,
        "utilization_percent": 52
      }
    },
    "requests": {
      "active": 1,
      "queued": 0
    },
    "cache": {
      "pages_used": 1024,
      "pages_total": 2048,
      "pages_free": 1024,
      "active_sequences": 1
    }
  }
}
```

### GET /ready

Readiness probe for Kubernetes.

**Request:**
```bash
curl http://localhost:8080/ready
```

**Success (200):**
```json
{
  "ready": true,
  "service": "rocmforge"
}
```

**Not Ready (503):** Returns empty response when engine is not initialized or model not loaded.

### GET /metrics

Prometheus metrics export.

**Request:**
```bash
curl http://localhost:8080/metrics
```

**Response (Prometheus text format):**
```
# HELP rocmforge_requests_started_total Total number of requests started
# TYPE rocmforge_requests_started_total counter
rocmforge_requests_started_total 42

# HELP rocmforge_requests_completed_total Total number of requests completed
# TYPE rocmforge_requests_completed_total counter
rocmforge_requests_completed_total{status="success"} 40
rocmforge_requests_completed_total{status="failed"} 1
rocmforge_requests_completed_total{status="cancelled"} 1

# HELP rocmforge_tokens_generated_total Total tokens generated
# TYPE rocmforge_tokens_generated_total counter
rocmforge_tokens_generated_total 15234

# HELP rocmforge_queue_length Current number of queued requests
# TYPE rocmforge_queue_length gauge
rocmforge_queue_length 0

# HELP rocmforge_active_requests Current number of active requests
# TYPE rocmforge_active_requests gauge
rocmforge_active_requests 2

# HELP rocmforge_tokens_per_second Tokens generated per second
# TYPE rocmforge_tokens_per_second gauge
rocmforge_tokens_per_second 28.5

# HELP rocmforge_prefill_duration_seconds Prefill phase duration
# TYPE rocmforge_prefill_duration_seconds histogram
rocmforge_prefill_duration_seconds_bucket{le="0.001"} 0
rocmforge_prefill_duration_seconds_bucket{le="0.01"} 2
...

# HELP rocmforge_ttft_seconds Time to first token
# TYPE rocmforge_ttft_seconds histogram
rocmforge_ttft_seconds_bucket{le="0.01"} 0
rocmforge_ttft_seconds_bucket{le="0.1"} 5
...
```

### GET /traces

OpenTelemetry trace export.

**Request:**
```bash
# Get all traces
curl http://localhost:8080/traces

# Limit to 10 traces
curl "http://localhost:8080/traces?limit=10"

# Get and clear traces
curl "http://localhost:8080/traces?clear=true"
```

**Query Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `limit` | Maximum traces to return | all |
| `clear` | Clear traces after returning | false |

---

## Troubleshooting

### Common Issues

#### "GPU not found" or "No AMD GPU detected"

**Symptoms:**
- Error: "GPU not found"
- `rocm-smi` shows no devices

**Solutions:**
```bash
# 1. Check GPU is visible
lspci | grep -i vga
ls /dev/kfd

# 2. Verify ROCm installation
rocminfo

# 3. Check driver is loaded
dmesg | grep -i amdgpu

# 4. Load driver if needed
sudo modprobe amdgpu

# 5. Verify environment
echo $ROCM_PATH
echo $PATH | grep rocm
```

#### "hipcc: command not found"

**Symptoms:**
- Build fails with "hipcc: command not found"

**Solutions:**
```bash
# 1. Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# 2. Add to ~/.bashrc permanently
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 3. Verify hipcc exists
ls -la /opt/rocm/bin/hipcc
```

#### "libamdhip64.so: cannot open shared object file"

**Symptoms:**
- Runtime error: cannot open shared object file

**Solutions:**
```bash
# 1. Add to library path
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 2. Add to ldconfig permanently
echo "/opt/rocm/lib" | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig

# 3. Verify library exists
ls -la /opt/rocm/lib/libamdhip64.so
```

#### "Failed to load model" or "Invalid GGUF file"

**Symptoms:**
- Error loading GGUF model
- Crash during model loading

**Solutions:**
```bash
# 1. Verify file is valid GGUF
file ./models/model.gguf
# Should show: "GGUF model"

# 2. Check file isn't corrupted
./target/release/inspect_gguf ./models/model.gguf

# 3. Verify architecture is supported
./target/release/inspect_gguf ./models/model.gguf | grep "Architecture:"

# 4. Re-download model if corrupted
```

#### Out of Memory (OOM)

**Symptoms:**
- Crash during inference
- "Out of memory" error
- System becomes unresponsive

**Solutions:**
```bash
# 1. Check available GPU memory
rocm-smi --showmem

# 2. Use a smaller quantized model
# Q4_K_M instead of Q8_0
# 7B model instead of 13B

# 3. Close other GPU applications
# Close other browser tabs, ML workloads, etc.

# 4. Check system memory
free -h

# 5. Monitor GPU memory during inference
watch -n 1 rocm-smi
```

#### Slow Generation Speed

**Symptoms:**
- Generation takes longer than expected
- Low tokens/second

**Solutions:**
```bash
# 1. Check GPU is being used
rocm-smi --showuse

# 2. Verify ROCm is properly installed
rocminfo | grep "Number of compute units"

# 3. Enable debug logging to check for CPU fallback
export ROCFORGE_LOG_LEVEL=debug
rocmforge-cli generate --gguf model.gguf --prompt "test"

# 4. Check for thermal throttling
rocm-smi --showtemp

# 5. Try a smaller model for comparison
```

#### HTTP Server Connection Refused

**Symptoms:**
- `curl: (7) Failed to connect to localhost port 8080`

**Solutions:**
```bash
# 1. Check if server is running
ps aux | grep rocmforge

# 2. Check port is not in use
netstat -tlnp | grep 8080

# 3. Verify firewall settings
sudo ufw status
sudo ufw allow 8080/tcp

# 4. Check server logs
export ROCFORGE_LOG_LEVEL=debug
rocmforge-cli serve --gguf model.gguf
```

#### Tokenizer Not Found

**Symptoms:**
- Warning: "tokenizer not provided"
- Fallback to hash-based tokenizer

**Solutions:**
```bash
# 1. Explicitly provide tokenizer
rocmforge-cli generate \
  --gguf ./models/model.gguf \
  --tokenizer ./models/tokenizer.json

# 2. Set environment variable
export ROCMFORGE_TOKENIZER=./models/tokenizer.json

# 3. Check if tokenizer is auto-detected
# ROCmForge will look for:
# - <model-dir>/tokenizer.json
# - <model-dir>/../tokenizer.json
# - Embedded in GGUF (if available)
```

### Debug Mode

Enable debug logging for detailed diagnostics:

```bash
# Enable debug output
export ROCFORGE_LOG_LEVEL=debug
export RUST_LOG=rocmforge=trace

# Run with debug
rocmforge-cli generate --gguf model.gguf --prompt "test"
```

### Getting Help

If issues persist:

1. **Check logs**: Enable debug logging and review output
2. **Verify environment**: Run all verification commands above
3. **Check GitHub Issues**: [ROCmForge Issues](https://github.com/oldnordic/ROCmForge/issues)
4. **Create an issue**: Include:
   - ROCm version (`cat /opt/rocm/.info/version`)
   - GPU model (`lspci | grep VGA`)
   - Rust version (`rustc --version`)
   - Full error message
   - Debug log output

---

## Additional Resources

- [ROCm Setup Guide](./rocm_setup_guide.md) - Detailed ROCm installation
- [README.md](../README.md) - Project overview and status
- [CLI Reference](./CLI_REFERENCE.md) - Detailed CLI documentation
- [API Documentation](./API_DOCUMENTATION.md) - HTTP API details
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment

---

## License

MIT License - See LICENSE file for details.
