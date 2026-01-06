# ROCmForge TODO

> GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32) → AMD Instinct MI355 (CDNA4)
> Last Updated: 2026-01-06 (Phase 5: MXFP Quantization)

## Overall Progress

| Phase | Description | Status | Completion Date | Tests |
|-------|-------------|--------|-----------------|-------|
| Phase 1 | Replace GPU Kernel Stubs (scale, mask, softmax) | ✅ Complete | 2025-01-03 | 3/3 |
| Phase 2 | RoPE + KV Append | ✅ Complete | 2025-01-03 | 5/5 |
| Phase 3a | Non-Causal FlashAttention (divide & conquer) | ✅ Complete | 2025-01-03 | 17/17 |
| Phase 3b | Causal Masking (sequential) | ✅ Complete | 2025-01-03 | 8/8 |
| Phase 4 | MLP Ops (SwiGLU, RMSNorm) | ✅ Complete | 2026-01-03 | 8/8 |
| Phase 4.5 | GGUF Vocab Size Inference | ✅ Complete | 2026-01-04 | - |
| Phase 5 | MXFP Quantization (AMD Quark) | 🔨 In Progress | 2026-01-06 | - |

**Total**: 41/41 kernel tests passing (100%)

---

## Architecture Decision: Standard GGUF Format Only

### ❌ REJECTED: Runtime Tensor Name Mapping

**Decision**: ROCmForge will **NOT** implement per-model tensor name mapping at runtime.

**Why**:
- Non-standard - llama.cpp, vLLM, Ollama don't do this
- Unsustainable - every new model needs custom code
- Wrong approach - models should be CONVERTED to standard format

**Correct Approach**:
1. Use llama.cpp's `convert.py` to create standard GGUF files
2. Use AMD Quark to quantize to MXFP formats
3. ROCmForge enforces standard tensor naming
4. Fail with clear error message for non-standard models

### ✅ ACCEPTED: AMD Quark for Quantization

**Decision**: Use AMD's official Quark toolkit for all quantization.

**Why**:
- AMD's official solution (not reinventing the wheel)
- Follows OCP Microscaling Formats (MX) Specification v1.0
- Supports MXFP4, MXFP6, FP8, and traditional quantization
- Integrates with vLLM AMD
- Open source, actively maintained

---

## Phase 5: AMD MXFP Quantization

> **Goal**: Enable state-of-the-art quantization for AMD GPUs
> **Hardware Target**: AMD Instinct MI355 (CDNA4) with native MXFP support
> **Fallback**: Software simulation for MI300/MI250/RDNA3

### MXFP4/MXFP6 Overview

Block-scaled floating-point formats per OCP MX Specification v1.0:

| Format | Bits | Range | Block Size | Memory Reduction | Accuracy |
|--------|------|-------|------------|------------------|----------|
| **MXFP4** | 4 (E2M1) | [-6, 6] | 32 | 4x vs FP16 | Best for >100B models |
| **MXFP6** | 6 (E2M3) | [-7.5, 7.5] | 32 | 2.67x vs FP16 | Near-lossless on >70B |
| **FP8** | 8 (E4M3) | Various | Per-tensor | 2x vs FP16 | Good for KV cache |

**Performance on AMD MI355**:
- 4x throughput improvement vs FP16
- Near-lossless accuracy for large models with MXFP6
- Native hardware acceleration via 1,024 MX cores

---

### Phase 5.1: SDK Installation & Setup

#### Task 5.1.1: Install AMD Quark

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

- [ ] Verify installation: `python -c "import quark; print(quark.__version__)"`
- [ ] Download example scripts from AMD Quark repo
- [ ] Test with sample model

**Links**:
- [AMD Quark PyPI](https://pypi.org/project/amd-quark/)
- [AMD Quark GitHub](https://github.com/AMD/Quark)
- [AMD Quark Docs](https://quark.docs.amd.com/)

#### Task 5.1.2: Install ROCm 7.0+

```bash
# Verify ROCm version
rocm-smi --showversion

# For native MXFP support, need:
# - ROCm 7.0+
# - AMD Instinct MI355 (CDNA4) OR
# - Software simulation for older GPUs
```

- [ ] Verify ROCm 7.0+ installed
- [ ] Check GPU compatibility: `rocm-smi --showproductname`

---

### Phase 5.2: Model Quantization Workflow

#### Task 5.2.1: Quantize Model with AMD Quark

**Goal**: Create MXFP4/MXFP6 quantized model using AMD Quark

```python
# quantize_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.quantization import Config, QuantizationConfig, FP4PerGroupSpec
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_SEQ_LEN = 512
GROUP_SIZE = 32
NUM_CALIBRATION = 512

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token

# Calibration data
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION]
tokenized = tokenizer(text_data, return_tensors="pt",
    padding=True, truncation=True, max_length=MAX_SEQ_LEN)
calib_dataloader = DataLoader(tokenized['input_ids'], batch_size=1, drop_last=True)

# MXFP4 configuration
def FP4_PER_GROUP_SYM_SPEC(group_size):
    return FP4PerGroupSpec(
        ch_axis=-1, group_size=group_size,
        scale_format="e8m0", scale_calculation_mode="even",
        is_dynamic=True
    ).to_quantization_spec()

global_quant_config = QuantizationConfig(
    input_tensors=FP4_PER_GROUP_SYM_SPEC(GROUP_SIZE),
    weight=FP4_PER_GROUP_SYM_SPEC(GROUP_SIZE)
)

quant_config = Config(
    global_quant_config=global_quant_config,
    exclude=["lm_head"],
    algo_config={"quant_algo": "autosmoothquant"}
)

# Quantize
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)
freezed_model = quantizer.freeze(model)

# Export
export_path = f"/models/{MODEL_ID.replace('/', '-')}-MXFP4"
exporter = ModelExporter(config=export_config, export_dir=export_path)
model = exporter.get_export_model(freezed_model, quant_config=quant_config,
                                   custom_mode="quark", add_export_info_for_hf=True)
model.save_pretrained(export_path)
tokenizer.save_pretrained(export_path)
```

- [ ] Create `scripts/quantize_model.py` script
- [ ] Test with small model (e.g., Qwen 0.5B)
- [ ] Verify output format is HuggingFace-compatible

#### Task 5.2.2: Command-Line Quantization

```bash
cd amd_quark-0.9/examples/torch/language_modeling/llm_ptq/

# Quantize to MXFP4
python3 quantize_quark.py \
    --model_dir /models/Llama-3.3-70B-Instruct \
    --dataset /data/pile-val-backup \
    --quant_scheme w_mxfp4_a_mxfp4 \
    --group_size 32 \
    --kv_cache_dtype fp8 \
    --quant_algo autosmoothquant \
    --model_export hf_format \
    --output_dir /models/Llama-3.3-70B-MXFP4 \
    --multi_gpu
```

**Quantization schemes**:
- `w_mxfp4_a_mxfp4`: MXFP4 weights + MXFP4 activations
- `w_mxfp4_a_mxfp6`: MXFP4 weights + MXFP6 activations (recommended)
- `w_mxfp4_a_fp6_e2m3`: MXFP4 weights + FP6-E2M3 activations

- [ ] Create `scripts/quantize_cli.sh` wrapper script
- [ ] Test with different quantization schemes
- [ ] Document best practices

---

### Phase 5.3: GGUF MXFP Support

#### Task 5.3.1: Add MXFP Tensor Types

**File**: `src/loader/gguf.rs`

```rust
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    // Existing types
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,

    // NEW: MXFP types (OCP MX Specification v1.0)
    MXFP4 = 20,      // OCP MXFP4-E2M1 (4-bit)
    MXFP6_E2M3 = 21,  // OCP MXFP6-E2M3 (6-bit, recommended)
    MXFP6_E3M2 = 22,  // OCP MXFP6-E3M2 (6-bit)
}
```

- [ ] Add MXFP variants to `GgufTensorType` enum
- [ ] Update `tensor_type_from_u32()` function
- [ ] Update `u32_from_tensor_type()` function
- [ ] Add format documentation

#### Task 5.3.2: MXFP Data Structures

```rust
// E8M0 scale format (8-bit exponent only)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct E8M0 {
    exponent: i8,  // value = 2^exponent
}

impl E8M0 {
    pub fn to_f32(&self) -> f32 {
        2.0_f32.powi(self.exponent as i32)
    }

    pub fn from_f32(value: f32) -> Self {
        let exp = value.log2().clamp(-127.0, 127.0) as i8;
        E8M0 { exponent: exp }
    }
}

// MXFP block (32 elements + scale)
#[repr(C)]
pub struct MxfpBlock {
    scale: E8M0,
    elements: [u8; 16],  // 32 x 4-bit elements packed
}
```

- [ ] Implement `E8M0` struct with conversion methods
- [ ] Implement `MxfpBlock` struct
- [ ] Add unit tests for E8M0 conversion
- [ ] Add unit tests for MXFP block packing/unpacking

---

### Phase 5.4: MXFP Dequantization Kernels

#### Task 5.4.1: Create MXFP Dequantization Kernel

**File**: `kernels/mxfp_dequant.hip` (NEW)

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

// Decode E2M1 (4-bit) to float
__device__ __forceinline__ float mxfp4_to_float(uint8_t bits) {
    const uint32_t sign = (bits >> 3) & 0x01;
    const uint32_t exp = (bits >> 1) & 0x03;
    const uint32_t mant = bits & 0x01;

    if (exp == 0 && mant == 0) return 0.0f;

    // E2M1: value = (-1)^sign * 2^(exp-1) * (1.mant)
    float significand = 1.0f + (float)mant;
    float exponent = (float)exp - 1.0f;
    float value = ldexpf(significand, (int)exponent);
    return sign ? -value : value;
}

// Dequantize MXFP4 to FP16
extern "C" __global__ void mxfp4_to_fp16_kernel(
    const uint8_t* __restrict__ mxfp4_data,
    half* __restrict__ fp16_output,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const int block_idx = idx / 32;
    const int elem_idx = idx % 32;

    // Load scale (E8M0 format)
    const int8_t scale_exp = ((int8_t*)mxfp4_data)[block_idx * 33];
    const float scale = __exp2f((float)scale_exp);

    // Load element (4-bit)
    const uint8_t packed = mxfp4_data[block_idx * 33 + 1 + elem_idx / 2];
    const uint8_t elem_4bit = (elem_idx % 2 == 0) ? (packed >> 4) : (packed & 0x0F);

    // Decode and apply scale
    float value = mxfp4_to_float(elem_4bit);
    value = scale * value;

    // Clip to MXFP4 range
    value = fmaxf(-6.0f, fminf(6.0f, value));

    fp16_output[idx] = __float2half(value);
}

// Decode E2M3 (6-bit) to float
__device__ __forceinline__ float mxfp6_to_float(uint8_t bits) {
    const uint32_t sign = (bits >> 5) & 0x01;
    const uint32_t exp = (bits >> 3) & 0x03;
    const uint32_t mant = bits & 0x07;

    if (exp == 0 && mant == 0) return 0.0f;

    // E2M3: value = (-1)^sign * 2^(exp-1) * (1.mant/8)
    float significand = 1.0f + (float)mant / 8.0f;
    float exponent = (float)exp - 1.0f;
    float value = ldexpf(significand, (int)exponent);
    return sign ? -value : value;
}

extern "C" __global__ void mxfp6_to_fp16_kernel(
    const uint8_t* __restrict__ mxfp6_data,
    half* __restrict__ fp16_output,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const int block_idx = idx / 32;
    const int elem_idx = idx % 32;

    // Load scale
    const int8_t scale_exp = ((int8_t*)mxfp6_data)[block_idx * 25];  // 1 + 24 bytes
    const float scale = __exp2f((float)scale_exp);

    // Load element (6-bit)
    const int byte_idx = 1 + elem_idx * 6 / 8;
    const int bit_offset = (elem_idx * 6) % 8;

    uint16_t elem_6bit;
    if (bit_offset <= 2) {
        elem_6bit = (mxfp6_data[block_idx * 25 + byte_idx] >> bit_offset) & 0x3F;
    } else {
        elem_6bit = (mxfp6_data[block_idx * 25 + byte_idx] >> bit_offset) & 0x3F;
        elem_6bit |= (mxfp6_data[block_idx * 25 + byte_idx + 1] << (8 - bit_offset)) & 0x3F;
    }

    float value = mxfp6_to_float((uint8_t)elem_6bit);
    value = scale * value;

    // Clip to MXFP6 range
    value = fmaxf(-7.5f, fminf(7.5f, value));

    fp16_output[idx] = __float2half(value);
}
```

- [ ] Create `kernels/mxfp_dequant.hip`
- [ ] Implement `mxfp4_to_fp16_kernel`
- [ ] Implement `mxfp6_to_fp16_kernel`
- [ ] Add to build.rs compilation list
- [ ] Create Rust wrappers in `src/loader/mxfp.rs`

#### Task 5.4.2: Rust Wrapper Functions

**File**: `src/loader/mxfp.rs` (NEW)

```rust
use crate::backend::hip_backend::HipBackend;

pub unsafe fn mxfp4_to_fp16(
    backend: &HipBackend,
    mxfp4_data: &[u8],
    num_elements: usize,
) -> Result<Vec<half::f16>, String> {
    // Allocate output buffer
    // Launch kernel
    // Copy result back
    todo!()
}

pub unsafe fn mxfp6_to_fp16(
    backend: &HipBackend,
    mxfp6_data: &[u8],
    num_elements: usize,
) -> Result<Vec<half::f16>, String> {
    todo!()
}
```

- [ ] Create `src/loader/mxfp.rs` module
- [ ] Implement `mxfp4_to_fp16` function
- [ ] Implement `mxfp6_to_fp16` function
- [ ] Add error handling
- [ ] Integrate into GGUF loader

---

### Phase 5.5: KV Cache MXFP Support

#### Task 5.5.1: Add MXFP6 KV Cache Dtype

**File**: `src/kv_cache/kv_cache.rs`

```rust
pub enum KvCacheDtype {
    F32,
    F16,
    FP8,
    MXFP6,  // NEW
}

impl KvCacheDtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            KvCacheDtype::F32 => 4,
            KvCacheDtype::F16 => 2,
            KvCacheDtype::FP8 => 1,
            KvCacheDtype::MXFP6 => 1,  // Packed: 32 x 6-bit + 8-bit scale
        }
    }

    pub fn memory_reduction_vs_f16(&self) -> f64 {
        match self {
            KvCacheDtype::F32 => 0.0,
            KvCacheDtype::F16 => 0.0,
            KvCacheDtype::FP8 => 0.5,
            KvCacheDtype::MXFP6 => 0.625,  // 37.5% of F16
        }
    }
}
```

- [ ] Add `MXFP6` variant to `KvCacheDtype`
- [ ] Update `size_bytes()` method
- [ ] Add `memory_reduction_vs_f16()` method
- [ ] Update KV cache allocation logic

#### Task 5.5.2: KV Cache Quantization/Dequantization

- [ ] Implement `quantize_kv_to_mxfp6()` kernel
- [ ] Implement `dequantize_kv_from_mxfp6()` kernel
- [ ] Add tests for KV cache round-trip accuracy
- [ ] Measure memory usage before/after

---

### Phase 5.6: Testing & Validation

#### Task 5.6.1: Unit Tests for MXFP

**File**: `src/loader/mxfp_tests.rs` (NEW)

```rust
#[cfg(test)]
mod mxfp_tests {
    #[test]
    fn test_e8m0_roundtrip() {
        // Test E8M0 conversion
    }

    #[test]
    fn test_mxfp4_decode() {
        // Test MXFP4 decoding
    }

    #[test]
    fn test_mxfp6_decode() {
        // Test MXFP6 decoding
    }

    #[test]
    fn test_mxfp_block_packing() {
        // Test block packing/unpacking
    }

    #[test]
    fn test_dequantization_accuracy() {
        // Test <0.1% error requirement
    }
}
```

- [ ] Create `src/loader/mxfp_tests.rs`
- [ ] Implement E8M0 round-trip test
- [ ] Implement MXFP4 decode test
- [ ] Implement MXFP6 decode test
- [ ] Implement dequantization accuracy test

#### Task 5.6.2: Integration Tests

**File**: `tests/mxfp_integration.rs` (NEW)

```rust
#[test]
fn test_load_quark_quantized_model() {
    // Test loading HuggingFace model quantized with Quark
}

#[test]
fn test_mxfp_kv_cache_roundtrip() {
    // Test KV cache quantization/dequantization
}

#[test]
fn test_end_to_end_inference_mxfp() {
    // Test full inference with MXFP weights
}
```

- [ ] Create integration test file
- [ ] Test Quark-quantized model loading
- [ ] Test KV cache round-trip
- [ ] Test end-to-end inference

#### Task 5.6.3: Accuracy Validation

- [ ] Run perplexity tests on quantized models
- [ ] Compare to FP16 baseline
- [ ] Verify <0.1% increase for MXFP6
- [ ] Verify <0.2% increase for MXFP4

```bash
# Install lm-eval
pip install lm-eval[api]

# Run evaluation
lm_eval --model vllm \
    --model_args pretrained=amd/Llama-2-70b-WMXFP4FP8,tensor_parallel_size=4 \
    --tasks mmlu \
    --batch_size auto
```

---

### Phase 5.7: Documentation

#### Task 5.7.1: Create MXFP Guide

**File**: `docs/MXFP_GUIDE.md` (NEW)

- [ ] Write comprehensive MXFP guide
- [ ] Include installation instructions
- [ ] Include quantization examples
- [ ] Include troubleshooting section
- [ ] Add reference to OCP MX Specification

#### Task 5.7.2: Update README.md

- [ ] Add MXFP support to README
- [ ] Link to MXFP guide
- [ ] Update hardware requirements
- [ ] Add quantization examples

---

### Phase 5.8: Go/No-Go Evaluation

#### Task 5.8.1: Pre-Implementation Checks

- [ ] Verify ROCm 7.0+ stability
- [ ] Test AMD Quark installation
- [ ] Verify GGUF MX spec compatibility
- [ ] Test dequantization accuracy on sample data

**Proceed if ALL pass**:
- ROCm 7.0+ stable
- AMD Quark produces valid models
- Sample dequantization <0.1% error
- Can load Quark HuggingFace format

#### Task 5.8.2: Post-Implementation Validation

- [ ] Run accuracy validation (Task 5.6.3)
- [ ] Measure performance improvement
- [ ] Verify memory reduction targets

**Success Criteria**:
- <0.1% perplexity increase (MXFP6)
- >2x throughput improvement (on MI355)
- >60% KV cache memory reduction

---

## Phase 1: Replace GPU Kernel Stubs ✅ COMPLETE

See archived sections below for details.

---

## Phase 2: RoPE + KV Append ✅ COMPLETE

See archived sections below for details.

---

## Phase 3: FlashAttention ✅ COMPLETE

See archived sections below for details.

---

## Phase 4: MLP Ops ✅ COMPLETE

See archived sections below for details.

---

## Phase 4.5: GGUF Vocab Size Inference ✅ COMPLETE

**Status**: COMPLETE
**Completed**: 2026-01-04

### Summary

Added vocab_size inference from tensor shapes when GGUF metadata is missing.

**Files Modified**:
- `src/loader/gguf.rs` - Added `infer_vocab_size_from_tensors()` method

**Exit Criteria**:
- [x] Helper method implemented
- [x] `to_model_config()` uses inference fallback
- [x] Code compiles without errors

---

## ARCHIVED: Phase 3a - Non-Causal FlashAttention

**Status**: ✅ COMPLETE

All sub-tasks completed. See Phase 3 section for details.

---

## ARCHIVED: Phase 3b - Causal Masking

**Status**: ✅ COMPLETE

All sub-tasks completed. See Phase 3 section for details.

---

## ARCHIVED: Phase 3 Retrospective

**Status**: ✅ COMPLETE

Lessons learned about scope and test isolation documented.

---

## Quick Reference

### Build Commands

```bash
# Build with ROCm feature
cargo build --features rocm

# Clean build
cargo clean && cargo build --features rocm

# Release build
cargo build --features rocm --release
```

### Test Commands

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

### GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 rocm-smi

# Check GPU info
rocm-smi --showproductname
rocm-smi --showmem
rocm-smi --showuse
```

---

## References

### AMD MXFP Resources
- [AMD MXFP4/MXFP6 Blog](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [OCP MX Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AMD Quark Docs](https://quark.docs.amd.com/)
- [AMD Quark GitHub](https://github.com/AMD/Quark)

### SDK Downloads
- [amd-quark PyPI](https://pypi.org/project/amd-quark/)
- [Quark Download](https://download.amd.com/opendownload/Quark/amd_quark-0.9.zip)
- [Docker Image](https://hub.docker.com/r/rocm/vllm-dev)

### Pre-Quantized Models (HuggingFace)
- `amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
- `amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
- `amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`
