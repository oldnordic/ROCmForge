# CLI Reference

Complete reference for the `rocmforge_cli` command-line interface.

**Version:** 0.1.0
**Binary Name:** `rocmforge_cli`

---

## Table of Contents

- [Global Options](#global-options)
- [Commands](#commands)
  - [serve](#serve)
  - [generate](#generate)
  - [status](#status)
  - [cancel](#cancel)
  - [models](#models)
- [Environment Variables](#environment-variables)
- [Exit Codes](#exit-codes)
- [Troubleshooting](#troubleshooting)

---

## Global Options

Options that apply to all commands.

| Option | Description | Default |
|--------|-------------|---------|
| `--host <HOST>` | Base URL of the ROCmForge HTTP server | `http://127.0.0.1:8080` |
| `-h, --help` | Print help information | - |
| `-V, --version` | Print version information | - |

---

## Commands

### serve

Start the built-in HTTP server. Similar to `ollama serve`, this command runs an inference server that accepts HTTP requests.

```bash
rocmforge_cli serve [OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--addr <ADDR>` | Address to bind the HTTP server to | `127.0.0.1:8080` |
| `--gguf <GGUF>` | Path to the GGUF model to load at startup | `$ROCMFORGE_GGUF` |
| `--tokenizer <TOKENIZER>` | Path to tokenizer JSON file | `$ROCMFORGE_TOKENIZER` |

#### Examples

**Start server on default address:**
```bash
rocmforge_cli serve --addr 127.0.0.1:8080 --gguf ./models/model.gguf
```

**Start server with custom port:**
```bash
rocmforge_cli serve --addr 0.0.0.0:9000 --gguf /path/to/model.gguf
```

**Start server with explicit tokenizer:**
```bash
rocmforge_cli serve --gguf model.gguf --tokenizer tokenizer.json
```

**Start server using environment variables:**
```bash
export ROCMFORGE_GGUF=./models/model.gguf
export ROCMFORGE_TOKENIZER=./tokenizer.json
rocmforge_cli serve
```

#### Notes

- The server loads the model at startup if `--gguf` is provided
- If no tokenizer is specified, the server attempts to:
  1. Use `ROCMFORGE_TOKENIZER` environment variable
  2. Infer tokenizer path from the GGUF file location
  3. Fall back to a basic hashing tokenizer
- The server listens on `127.0.0.1:8080` by default

---

### generate

Generate text once and print the final response. Can run inference locally or via HTTP.

```bash
rocmforge_cli generate [OPTIONS] --prompt <PROMPT>
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --prompt <PROMPT>` | Prompt text to feed the model | **Required** |
| `--max-tokens <MAX_TOKENS>` | Maximum number of new tokens to generate | `128` |
| `--temperature <TEMPERATURE>` | Sampling temperature (higher = more random) | `1.0` |
| `--top-k <TOP_K>` | Top-k sampling value | `50` |
| `--top-p <TOP_P>` | Top-p (nucleus) sampling value | `0.9` |
| `--stream` | Stream tokens as they are generated | `false` |
| `--gguf <GGUF>` | Run inference locally using this GGUF model | HTTP mode |
| `--tokenizer <TOKENIZER>` | Path to tokenizer JSON (local mode only) | Inferred |

#### Parameter Validation

| Parameter | Valid Range | Default |
|-----------|-------------|---------|
| `max_tokens` | 1 - 8192 | 128 |
| `temperature` | >= 0.0 | 1.0 |
| `top_k` | >= 1 | 50 |
| `top_p` | (0.0, 1.0] | 0.9 |

#### Examples

**Generate via HTTP (default):**
```bash
rocmforge_cli generate --prompt "What is the capital of France?"
```

**Generate with custom parameters:**
```bash
rocmforge_cli generate --prompt "Write a poem" --max-tokens 256 --temperature 0.8
```

**Generate with streaming (HTTP):**
```bash
rocmforge_cli generate --prompt "Tell me a story" --stream
```

**Generate locally (no server):**
```bash
rocmforge_cli generate --gguf ./models/model.gguf --prompt "Hello, world!"
```

**Generate locally with streaming:**
```bash
rocmforge_cli generate --gguf model.gguf --stream --prompt "Explain quantum computing"
```

**Generate with custom host (HTTP mode):**
```bash
rocmforge_cli --host http://192.168.1.100:8080 generate --prompt "Test prompt"
```

#### Output

**Non-streaming mode:**
```
request_id: 12345
finish_reason: Some("length")
text:
The capital of France is Paris. It is located in the north-central part of the country...
```

**Streaming mode:**
Tokens are printed to stdout as they are generated. Completion message:
```
[request 12345 finished: length]
```

---

### status

Query the status of a previous generation request by its request ID.

```bash
rocmforge_cli status --request-id <REQUEST_ID>
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--request-id <REQUEST_ID>` | Request identifier from a previous generate command | **Required** |

#### Examples

**Check request status:**
```bash
rocmforge_cli status --request-id 12345
```

**With custom host:**
```bash
rocmforge_cli --host http://localhost:9000 status --request-id 67890
```

#### Output

```
request_id: 12345, finished: true, reason: Some("length")
text:
The generated text appears here...
```

---

### cancel

Cancel a running request on the HTTP server.

```bash
rocmforge_cli cancel --request-id <REQUEST_ID>
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--request-id <REQUEST_ID>` | Request identifier to cancel | **Required** |

#### Examples

**Cancel a request:**
```bash
rocmforge_cli cancel --request-id 12345
```

#### Output

```
Cancelled request 12345 (finished: Some("cancelled"))
text:
Partially generated text...
```

---

### models

List available GGUF models in a directory. Scans recursively for `.gguf` files.

```bash
rocmforge_cli models [OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dir <DIR>` | Directory containing GGUF files | `$ROCMFORGE_MODELS` or `./models` |

#### Examples

**List models in default directory:**
```bash
rocmforge_cli models
```

**List models in custom directory:**
```bash
rocmforge_cli models --dir /path/to/models
```

**Using environment variable:**
```bash
export ROCMFORGE_MODELS=/mnt/storage/models
rocmforge_cli models
```

#### Output

```
Discovered models:
- llama-2-7b-chat
  gguf: ./models/llama-2-7b-chat.Q4_K_M.gguf
  tokenizer: ./models/tokenizer.json
  arch: llama | layers: 32 | heads: 32 | hidden: 4096 | ctx: 2048 | vocab: 32000 | file_type: 3 | embedded tokenizer: no
  cache: cached

- mistral-7b-instruct
  gguf: ./models/mistral-7b-instruct.Q5_K_S.gguf
  tokenizer: (no tokenizer)
  arch: mistral | layers: 32 | heads: 32 | hidden: 4096 | ctx: 32768 | vocab: 32000 | file_type: 4 | embedded tokenizer: yes
  cache: refreshed
```

#### Output Fields

| Field | Description |
|-------|-------------|
| `arch` | Model architecture (llama, mistral, yi, mixtral, etc.) |
| `layers` | Number of transformer layers |
| `heads` | Number of attention heads |
| `hidden` | Hidden size (embedding dimension) |
| `ctx` | Maximum context window (sequence length) |
| `vocab` | Vocabulary size |
| `file_type` | GGUF quantization type code |
| `embedded tokenizer` | Whether tokenizer is embedded in GGUF |
| `cache` | Tokenizer cache status (`cached`, `refreshed`, `direct`) |

---

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `ROCMFORGE_GGUF` | Default path to GGUF model file | `serve`, HTTP server |
| `ROCMFORGE_TOKENIZER` | Default path to tokenizer.json | `serve`, `generate` |
| `ROCMFORGE_MODELS` | Default directory for GGUF model discovery | `models` |
| `ROCMFORGE_MAX_TRACES` | Maximum number of OpenTelemetry traces to store | Internal |

#### Usage Examples

**Set model paths globally:**
```bash
export ROCMFORGE_GGUF=/opt/models/llama-2-7b.gguf
export ROCMFORGE_TOKENIZER=/opt/models/tokenizer.json
export ROCMFORGE_MODELS=/opt/models

# Now you can omit these paths
rocmforge_cli serve
rocmforge_cli models
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error (file not found, connection failed, etc.) |
| `2` | Invalid usage (missing required arguments, invalid parameter values) |

### Common Exit Scenarios

| Scenario | Exit Code |
|----------|-----------|
| Successful generation | 0 |
| Server returns error (4xx/5xx) | 1 |
| Model file not found | 1 |
| Invalid `max_tokens` (0 or >8192) | 2 |
| Invalid `temperature` (< 0) | 2 |
| Invalid `top_k` (= 0) | 2 |
| Invalid `top_p` (<= 0 or > 1) | 2 |
| No GGUF files found in models directory | 0 |
| Server not reachable | 1 |

---

## Troubleshooting

### Server Connection Issues

**Problem:** `Server returned error: Connection refused`

**Solutions:**
1. Ensure the server is running: `rocmforge_cli serve`
2. Check the host/port matches: `--host http://127.0.0.1:8080`
3. Verify firewall settings

### Model Loading Issues

**Problem:** `Failed to load GGUF: No such file or directory`

**Solutions:**
1. Verify the file path is correct
2. Use absolute paths: `--gguf /full/path/to/model.gguf`
3. Check file permissions

**Problem:** `WARN: tokenizer not provided and no tokenizer file inferred`

**Solutions:**
1. Provide explicit tokenizer: `--tokenizer path/to/tokenizer.json`
2. Ensure tokenizer.json is in the same directory as the GGUF
3. Use a model with embedded tokenizer (check with `models` command)

### Parameter Validation Errors

**Problem:** `Invalid max_tokens: must be greater than 0`

**Solutions:**
1. Use `--max-tokens 1` or higher
2. Maximum allowed value is 8192

**Problem:** `Invalid top_p: must be in range (0.0, 1.0]`

**Solutions:**
1. Use values like `0.9`, `0.95`, or `1.0`
2. Avoid `0.0` or negative values

### Local Mode Issues

**Problem:** Generation seems slow or hangs

**Solutions:**
1. Check GPU is available: `rocm-smi` (ROCm utility)
2. Ensure ROCm drivers are installed
3. Try a smaller model for testing

### Models Command Returns Nothing

**Problem:** `No GGUF models found in './models'`

**Solutions:**
1. Specify the directory: `--dir /path/to/models`
2. Set environment variable: `export ROCMFORGE_MODELS=/path/to/models`
3. Verify files have `.gguf` extension (case-insensitive)

### Streaming Issues

**Problem:** `stream error: unexpected end of stream`

**Solutions:**
1. Check server is still running
2. Verify network connection
3. Try non-streaming mode to isolate the issue

### Signal Handling

**Problem:** `Ctrl+C` doesn't cancel generation

**Notes:**
- In streaming mode (HTTP), `Ctrl+C` attempts to cancel the request
- In local mode, `Ctrl+C` cancels the local inference
- If cancellation fails, the server may continue processing

### Memory Issues

**Problem:** Out of memory errors during generation

**Solutions:**
1. Reduce `--max-tokens`
2. Use a smaller quantization (e.g., Q4_K instead of Q8_0)
3. Reduce context window by truncating prompts
4. Close other applications using GPU memory

---

## Advanced Usage

### Combining Commands

**Start server and test:**
```bash
# Terminal 1: Start server
rocmforge_cli serve --gguf model.gguf

# Terminal 2: Generate
rocmforge_cli generate --prompt "Test"
```

### Using with Custom Scripts

**Example bash script:**
```bash
#!/bin/bash
export ROCMFORGE_GGUF="./models/llama-2-7b.gguf"
export ROCMFORGE_TOKENIZER="./models/tokenizer.json"

# Start server in background
rocmforge_cli serve &
SERVER_PID=$!

# Wait for server to be ready
sleep 5

# Run generation
rocmforge_cli generate --prompt "$1" --stream

# Clean up
kill $SERVER_PID
```

---

## See Also

- [User Guide](USER_GUIDE.md) - Installation and usage instructions
- [API Documentation](API_DOCUMENTATION.md) - HTTP API reference
- [Deployment Guide](DEPLOYMENT.md) - Production deployment
