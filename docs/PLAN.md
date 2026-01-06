# ROCmForge Implementation Plan

> GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32) → AMD Instinct MI355 (CDNA4)
> Last Updated: 2026-01-06
> Rule: **Make it correct → make it measurable → then make it fast.**

---

## Current Status

| Phase | Description | Status | Tests | Date |
|-------|-------------|--------|-------|------|
| Phase 1 | Basic kernels (scale, mask, softmax) | ✅ Complete | 3/3 | 2025-01-03 |
| Phase 2 | RoPE + KV Append | ✅ Complete | 5/5 | 2025-01-03 |
| Phase 3a | Non-Causal FlashAttention | ✅ Complete | 17/17 | 2025-01-03 |
| Phase 3b | Causal Masking | ✅ Complete | 8/8 | 2025-01-03 |
| Phase 4 | MLP Ops (SwiGLU, RMSNorm) | ✅ Complete | 8/8 | 2026-01-03 |
| Phase 4.5 | GGUF Vocab Size Inference | ✅ Complete | - | 2026-01-04 |
| Phase 5 | Quantization (MXFP4/MXFP6) | 🔨 In Progress | - | 2026-01-06 |

**Progress**: Phases 1-4 complete (41/41 tests passing)

---

## Architecture Decision: Standard GGUF Format Only

### ❌ Rejected: Runtime Tensor Name Mapping

**Decision**: ROCmForge will **NOT** implement per-model tensor name mapping at runtime.

**Rationale**:
- Non-standard approach used by no other inference engine
- Unsustainable: every new model requires custom mapping code
- Creates maintenance burden
- Against industry best practices

**What this means**:
- Models must be converted TO standard GGUF format before use
- ROCmForge enforces standard tensor naming conventions
- Clear error messages when non-standard models are loaded

### ✅ Accepted: AMD Quark for Quantization

**Decision**: Use AMD Quark toolkit for model quantization.

**Rationale**:
- AMD's official quantization toolkit
- Supports MXFP4, MXFP6, FP8, and traditional quantization
- Integrates with vLLM (AMD-optimized version)
- Open source, actively maintained
- Follows OCP Microscaling Formats (MX) Specification v1.0

---

## Phase 5: AMD MXFP Quantization

> **Goal**: Enable state-of-the-art quantization for AMD GPUs using AMD Quark
> **Hardware Target**: AMD Instinct MI355 (CDNA4) with native MXFP support
> **Fallback**: Software simulation for MI300/MI250/RDNA3

### Overview

MXFP4 and MXFP6 are **block-scaled floating-point formats** defined by the OCP MX Specification:

| Format | Bits/Element | Range | Block Size | Scale Type | Memory vs FP16 |
|--------|--------------|-------|------------|------------|----------------|
| **MXFP4** | 4 (E2M1) | [-6, 6] | 32 | E8M0 (2^n) | 4x reduction |
| **MXFP6** | 6 (E2M3) | [-7.5, 7.5] | 32 | E8M0 (2^n) | 2.67x reduction |
| **FP8** | 8 (E4M3) | Various | Per-tensor | Various | 2x reduction |

**Performance on AMD MI355**:
- MXFP4/MXFP6: **4x throughput** vs FP16
- MXFP6: Near-lossless accuracy on models >70B parameters
- MXFP4: Best for very large models (>100B)

---

### Phase 5.1: Setup & Dependencies

#### 5.1.1: Install AMD Quark

```bash
# Method 1: PyPI (Recommended)
pip install amd-quark

# Method 2: From source
git clone --recursive https://github.com/AMD/Quark
cd Quark
pip install .

# Method 3: Download with examples
wget -O amd_quark-0.9.zip https://download.amd.com/opendownload/Quark/amd_quark-0.9.zip
unzip -o amd_quark-0.9.zip
pip install amd-quark==0.9
```

**Version**: 0.9+ (latest as of September 2025)

**Dependencies**:
- Python 3.x
- PyTorch (ROCm version for AMD GPUs)
- HuggingFace Transformers
- datasets
- NumPy

#### 5.1.2: ROCm Requirements

```bash
# ROCm 7.0+ required for MXFP support
rocm-smi  # Verify installation

# For development, use Docker
docker run -it --rm \
  --privileged \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 8G \
  -v $(pwd):/workspace \
  -w /workspace \
  rocm/vllm-dev:open-mi355-08052025 bash
```

---

### Phase 5.2: Model Quantization with AMD Quark

#### 5.2.1: Quantize a Model to MXFP4

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.quantization import Config, QuantizationConfig
from quark.torch.quantization import FP4PerGroupSpec, OCP_MXFP4Spec
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. Load model
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_SEQ_LEN = 512
GROUP_SIZE = 32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare calibration data
NUM_CALIBRATION = 512
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION]

tokenized = tokenizer(text_data, return_tensors="pt",
    padding=True, truncation=True, max_length=MAX_SEQ_LEN)
calib_dataloader = DataLoader(tokenized['input_ids'],
    batch_size=1, drop_last=True)

# 3. Configure MXFP4 quantization
def FP4_PER_GROUP_SYM_SPEC(group_size, scale_format="e8m0",
                           scale_calculation_mode="even", is_dynamic=True):
    return FP4PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=is_dynamic
    ).to_quantization_spec()

global_quant_config = QuantizationConfig(
    input_tensors=FP4_PER_GROUP_SYM_SPEC(GROUP_SIZE, "e8m0", "even", True),
    weight=FP4_PER_GROUP_SYM_SPEC(GROUP_SIZE, "e8m0", "even", False)
)

# Load algorithm config
algo_config = {
    "quant_algo": "autosmoothquant",  # Recommended for LLMs
    "layer_quant_config": {},
    "exclude_layers": ["lm_head"]
}

quant_config = Config(
    global_quant_config=global_quant_config,
    exclude=["lm_head"],
    algo_config=algo_config
)

# 4. Quantize
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

# 5. Export
export_path = "/workspace/models/Llama-3.3-70B-Instruct-MXFP4"
exporter = ModelExporter(config=export_config, export_dir=export_path)
model = exporter.get_export_model(quant_model, quant_config=quant_config,
                                   custom_mode="quark", add_export_info_for_hf=True)
model.save_pretrained(export_path)
tokenizer.save_pretrained(export_path)
```

#### 5.2.2: Command-Line Quantization

```bash
cd ./amd_quark-0.9/examples/torch/language_modeling/llm_ptq/

python3 quantize_quark.py \
    --model_dir /workspace/models/Llama-3.3-70B-Instruct \
    --model_attn_implementation "sdpa" \
    --dataset /workspace/data/pile-val-backup \
    --quant_scheme w_mxfp4_a_mxfp4 \
    --group_size 32 \
    --kv_cache_dtype fp8 \
    --quant_algo autosmoothquant \
    --min_kv_scale 1.0 \
    --model_export hf_format \
    --output_dir /workspace/models/Llama-3.3-70B-Instruct-MXFP4 \
    --multi_gpu
```

**Available Quantization Schemes**:
- `w_mxfp4_a_mxfp4`: MXFP4 weights + MXFP4 activations
- `w_mxfp4_a_fp6_e2m3`: MXFP4 weights + FP6 activations
- `w_mxfp4_a_mxfp6`: MXFP4 weights + MXFP6 activations (recommended for smaller models)

**Supported Algorithms**:
- `autosmoothquant`: AutoSmoothQuant (recommended for LLMs)
- `gptq`: GPTQ
- `awq`: AWQ
- `quarot`: Quarot

---

### Phase 5.3: GGUF Format with MXFP Support

#### 5.3.1: Add MXFP Tensor Types to GGUF Loader

**File**: `src/loader/gguf.rs`

```rust
// Add to GGML_TYPE enum
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    // Existing types...
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,

    // NEW: MXFP types
    MXFP4 = 20,     // OCP MXFP4-E2M1 (4-bit)
    MXFP6_E2M3 = 21, // OCP MXFP6-E2M3 (6-bit, recommended)
    MXFP6_E3M2 = 22, // OCP MXFP6-E3M2 (6-bit)
}
```

#### 5.3.2: MXFP Data Structures

```rust
// Block-scaled format: scale + elements
#[repr(C)]
pub struct MxfpBlock {
    scale: E8M0,  // 8-bit exponent-only (2^scale)
    elements: [u8; 32],  // 32 elements packed
}

// E8M0 scale format (exponent only)
#[repr(C)]
pub struct E8M0 {
    exponent: i8,  // Power of two: value = 2^exponent
}
```

#### 5.3.3: Dequantization Kernels

**File**: `kernels/mxfp_dequant.hip` (NEW)

```cpp
// Dequantize MXFP4 to FP16
__global__ void mxfp4_to_fp16_kernel(
    const uint8_t* __restrict__ mxfp4_data,
    half* __restrict__ fp16_output,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_idx = idx / 32;
    const int elem_idx = idx % 32;

    // Load scale (E8M0 format)
    const int8_t scale_exp = ((int8_t*)mxfp4_data)[block_idx * 33];
    const float scale = __exp2f((float)scale_exp);

    // Load element (4-bit E2M1)
    const uint8_t packed = mxfp4_data[block_idx * 33 + 1 + elem_idx / 2];
    const uint8_t elem_4bit = (elem_idx % 2 == 0) ? (packed >> 4) : (packed & 0x0F);

    // Decode E2M1 to float
    const uint32_t sign = (elem_4bit >> 3) & 0x01;
    const uint32_t exp = (elem_4bit >> 1) & 0x03;
    const uint32_t mant = elem_4bit & 0x01;

    float value;
    if (exp == 0 && mant == 0) {
        value = 0.0f;
    } else {
        value = scale * ((sign ? -1.0f : 1.0f) *
                 ldexpf((float)(mant | 0x10), (int)exp - 5));
    }

    // Clip to MXFP4 range
    value = fmaxf(-6.0f, fminf(6.0f, value));

    fp16_output[idx] = __float2half(value);
}
```

---

### Phase 5.4: KV Cache Quantization

#### 5.4.1: MXFP6 for KV Cache

**Rationale**: MXFP6 provides better accuracy than MXFP4 for KV cache while still offering significant memory savings.

**File**: `src/kv_cache/kv_cache.rs`

```rust
pub enum KvCacheDtype {
    F32,
    F16,
    FP8,
    MXFP6,  // NEW
}

pub struct KvCache {
    dtype: KvCacheDtype,
    // ... existing fields
}
```

**Memory Savings for LLaMA-2 70B**:
- FP16 KV cache: ~28 GB
- MXFP6 KV cache: ~10.5 GB (62.5% reduction)
- Enables single-GPU inference on 24GB cards

---

### Phase 5.5: Implementation Tasks

| Task | Description | File | Estimate |
|------|-------------|------|----------|
| 5.5.1 | Add MXFP tensor types to GGUF enum | `src/loader/gguf.rs` | 1 day |
| 5.5.2 | Implement E8M0 scale struct | `src/loader/gguf.rs` | 0.5 day |
| 5.5.3 | Add MXFP block dequantization | `kernels/mxfp_dequant.hip` | 2 days |
| 5.5.4 | Implement MXFP4 decoding kernel | `kernels/mxfp_dequant.hip` | 2 days |
| 5.5.5 | Implement MXFP6 decoding kernel | `kernels/mxfp_dequant.hip` | 2 days |
| 5.5.6 | Add KV cache MXFP6 support | `src/kv_cache/kv_cache.rs` | 2 days |
| 5.5.7 | Update build.rs for MXFP kernels | `build.rs` | 0.5 day |
| 5.5.8 | Write MXFP dequantization tests | `src/loader/mxfp_tests.rs` | 2 days |
| 5.5.9 | Integration tests with Quark models | `tests/mxfp_integration.rs` | 3 days |
| 5.5.10 | Documentation and examples | `docs/MXFP_GUIDE.md` | 1 day |

**Total Estimate**: 16 days (3 weeks)

---

### Phase 5.6: Accuracy Validation

#### 5.6.1: Perplexity Testing

```bash
# Install lm-evaluation-harness
pip install lm-eval[api]

# Evaluate quantized model
lm_eval --model vllm \
    --model_args pretrained=amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8 \
    --tasks mmlu \
    --batch_size auto
```

#### 5.6.2: Acceptance Criteria

| Metric | Target |
|--------|--------|
| Perplexity increase (vs FP16) | <0.1 for MXFP6, <0.2 for MXFP4 |
| MMLU score (70B models) | Within 1% of FP16 |
| Memory reduction | >60% for KV cache, >75% for weights |
| Throughput improvement | >2x on MI355 |

---

### Phase 5.7: Go/No-Go Criteria

**Proceed if ALL of**:
- ROCm 7.0+ stable on target hardware
- AMD Quark quantization produces valid models
- Dequantization kernels pass accuracy tests
- <0.1% perplexity increase for MXFP6

**Cancel if ANY of**:
- >0.2% perplexity increase
- Performance worse than FP16
- ROCm MXFP implementation buggy
- Cannot load Quark-quantized models

---

## Phase 4.5: GGUF Vocab Size Inference ✅ COMPLETE

See TODO.md for details.

---

## Future Work (Beyond Phase 5)

### Multi-GPU Support
- Tensor parallelism for models >70B
- Pipeline parallelism for layer distribution

### Performance Optimization
- GPU sampler (top-k/top-p on device)
- Custom GEMM kernels (if profiling shows need)
- Kernel fusion for entire transformer layers

### Advanced Features
- Speculative decoding
- Prefix caching
- Batching optimization

---

## Hardware Reference

| Component | RDNA3 (Current) | CDNA4 (Target) |
|-----------|-----------------|----------------|
| **GPU** | AMD Radeon RX 7900 XT | AMD Instinct MI355 |
| **Architecture** | gfx1100 | gfx950 |
| **Wavefront Size** | 32 | 64 (optional) |
| **MXFP Support** | Software simulation | Native hardware |
| **Matrix Cores** | None | 1,024 MX cores |

### Block Size Formula

```cpp
// For RDNA3 (wave32)
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;

// Wave32 reduction
for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
    __syncthreads();
}
```

---

## Test Coverage

### Current Tests (41 total)

| Category | Tests | File |
|----------|-------|------|
| Basic kernels | 3 | `kernel_tests.rs` |
| RoPE | 5 | `rope_gpu_tests.rs` |
| QK^T matmul | 4 | `qkt_matmul_tests.rs` |
| Softmax explicit | 4 | `softmax_explicit_tests.rs` |
| Weighted matmul | 4 | `weighted_matmul_tests.rs` |
| Flash non-causal | 5 | `flash_nocausal_tests.rs` |
| Causal mask | 4 | `causal_mask_tests.rs` |
| Flash causal | 4 | `flash_causal_tests.rs` |
| SwiGLU | 5 | `swiglu_tests.rs` |
| RMSNorm | 3 | `rms_norm_tests.rs` |

### Running Tests

```bash
# All tests
cargo test --features rocm

# Specific phase
cargo test --features rocm --lib mlp

# Specific test
cargo test --features rocm --lib test_swiglu_matches_cpu_small

# With output
cargo test --features rocm --lib -- --nocapture
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `CHANGELOG.md` | Chronological history of all changes |
| `docs/TODO.md` | Detailed task tracking with progress |
| `docs/PLAN.md` | This file - roadmap and future work |
| `docs/QUICKSTART.md` | Quick start guide |
| `docs/CODEBASE_AUDIT_REPORT_2026-01-06.md` | Comprehensive audit |

---

## Quick Command Reference

```bash
# Build
cargo build --features rocm

# Clean build
cargo clean && cargo build --features rocm

# Release build
cargo build --features rocm --release

# Run tests
cargo test --features rocm

# GPU monitoring
watch -n 1 rocm-smi

# Check GPU info
rocm-smi --showproductname
rocm-smi --showmem
rocm-smi --showuse
```

---

## References

### AMD MXFP Documentation
- [AMD MXFP4/MXFP6 Blog Post](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [OCP MX Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AMD Quark Documentation](https://quark.docs.amd.com/)
- [AMD Quark GitHub](https://github.com/AMD/Quark)

### SDK Downloads
- [amd-quark PyPI](https://pypi.org/project/amd-quark/)
- [AMD Quark Download](https://download.amd.com/opendownload/Quark/amd_quark-0.9.zip)
- [vLLM AMD Integration](https://docs.vllm.ai/en/stable/features/quantization/quark/)

### Pre-Quantized Models
- `amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
- `amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
- `amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
