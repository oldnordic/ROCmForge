# ROCmForge User Manual

**AMD GPU Inference Engine for Large Language Models**

Version 0.1.0-dev | Development Stage | Last Updated: January 2026

**⚠️ Development Status: Not production-ready. Use for testing and development only.**

---

## Table of Contents

1. [Installation](#installation)
2. [Build](#build)
3. [Running Tests](#running-tests)
4. [Model Discovery](#model-discovery)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

| Requirement | Minimum Version | Tested On |
|-------------|-----------------|----------|
| Rust | 1.82+ | 1.82 |
| ROCm | 5.x | 5.0+ |
| AMD GPU | RDNA2+ | RX 7900 XT |
| RAM | 16GB | 32GB |

### Step 1: Install ROCm

```bash
# Ubuntu/Debian
wget https://repo.radeon.com/rocm/rocm.gpg.key
sudo apt-key add rocm.gpg.key
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.0 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-hip-sdk rocm-dev

# Verify ROCm installation
rocm-smi
```

### Step 2: Clone and Build

```bash
# Clone repository
cd /path/to/projects
git clone <repository-url>
cd ROCmForge

# Build release binary (~55 seconds)
cargo build --release

# Verify build
ls -la target/release/rocmforge_cli
```

---

## Build

### Build Commands

```bash
# Release build (optimized)
cargo build --release

# Development build (faster compilation)
cargo build

# Check compilation only
cargo check
```

### Build Output

The build produces several binaries:
- `rocmforge_cli` - Main CLI tool
- `inspect_gguf` - GGUF model inspector
- `test_inference` - Inference testing tool
- `test_gguf_load` - GGUF loading tests
- `run_simple_model` - Simple model runner

---

## Running Tests

### Lib Unit Tests

```bash
# Run all lib unit tests (~0.5s)
cargo test --lib

# Run lib tests with output
cargo test --lib -- --nocapture
```

**Expected Result**: `test result: ok. 578 passed`

### Integration Tests

```bash
# Run specific integration test suite
cargo test --test multilayer_pipeline_tests

# Run all integration tests (excluding those with compilation errors)
cargo test --test '*'

# Run tests sequentially (prevents GPU conflicts)
cargo test -- --test-threads=1
```

### GPU Tests

GPU tests require `#[serial]` protection to prevent conflicts:

```bash
# Run GPU tests sequentially
cargo test --test multilayer_pipeline_tests -- --test-threads=1
```

**Note**: GPU tests will skip gracefully if GPU is not available.

---

## Model Discovery

### List Available Models

```bash
cargo run --release --bin rocmforge_cli -- models
```

**Expected Output**:
```
- qwen2-0_5b-instruct-q5_k_m
  gguf: models/Qwen2-0.5B-Instruct-GGUF/qwen2.5-0.5b.Q4_K_M.gguf
  arch: qwen2 | layers: 24 | heads: 14 | hidden: 896 | ctx: 2048
```

### Supported Models

ROCmForge can read GGUF metadata from:
- Qwen2 / Qwen2.5
- LLaMA family
- GLM
- Mistral
- And others (auto-detected from GGUF)

**Note**: Full inference not yet validated - only metadata reading confirmed.

---

## Troubleshooting

### Common Issues

#### 1. "No HIP GPU found"

```bash
# Check GPU detection
rocm-smi

# Expected: Device list with GPU info
# If empty: ROCm not installed or GPU not detected
```

#### 2. Build Failures

```bash
# Clean and rebuild
cargo clean
cargo build --release
```

#### 3. Test Failures

**Known failing tests**:
- `test_token_embedding_lookup_f32` - GGUF metadata bug (vocab_size = 0)
- `test_batch_embedding_lookup` - Same metadata bug
- `test_lm_head_matmul_correctness` - Same metadata bug

**Known compilation errors**:
- `tests/q_dequant_tests.rs` - Type mismatches, needs fixes
- `tests/attention_gpu_tests.rs` - Borrow checker issues

**Workaround**: Run tests excluding these files:
```bash
mv tests/q_dequant_tests.rs tests/q_dequant_tests.rs.disabled
cargo test --test '*'
mv tests/q_dequant_tests.rs.disabled tests/q_dequant_tests.rs
```

#### 4. GPU Reset During Tests

**Fixed**: All GPU tests now have `#[serial]` attribute and run sequentially.

If you still experience GPU resets:
```bash
# Run tests with single thread
cargo test -- --test-threads=1
```

### Debug Mode

```bash
# Enable verbose logging
RUST_LOG=debug cargo test --lib

# Run with backtrace
RUST_BACKTRACE=1 cargo test --lib
```

---

## Known Limitations

| Issue | Status | Details |
|-------|--------|---------|
| GGUF vocab_size = 0 | ⚠️ Bug | Metadata parser returns wrong value |
| q_dequant_tests | ❌ Broken | Compilation errors |
| attention_gpu_tests | ❌ Broken | Compilation errors |
| HTTP server | ❓ Untested | Exists but not validated |
| CLI generate/serve | ❓ Untested | Commands exist but not tested |
| End-to-end inference | ❓ Untested | Full pipeline not validated |

---

## Development Focus

Current priorities based on actual testing:

1. **Fix GGUF metadata parser** - vocab_size returns 0
2. **Fix compilation errors** - q_dequant_tests, attention_gpu_tests
3. **Validate HTTP server** - Test `/v1/completions` endpoint
4. **Test end-to-end** - Run actual inference with real models
5. **Add CLI tests** - Validate generate/serve commands

---

## Getting Help

- **Build Issues**: Check ROCm installation, Rust version
- **Test Failures**: See Known Issues above
- **GPU Issues**: Run `rocm-smi` to verify GPU detection

---

## License

GPL-3.0

---

**Note**: This is a development-stage project. Many features are untested and may not work as expected.
