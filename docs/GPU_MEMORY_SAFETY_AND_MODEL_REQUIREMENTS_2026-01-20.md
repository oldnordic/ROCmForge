# GPU Memory Safety and Model-Specific Requirements

**Date**: 2026-01-20
**Source**: AMD ROCm Examples + ROCmForge Code Analysis
**Purpose**: Document safe GPU memory allocation patterns and model-specific requirements

---

## Executive Summary

**CRITICAL FINDING**: We have safe allocation methods (`allocate_buffer_safe()` with 70% safety margin) BUT **they are NOT used in the model loading path**. This is why your desktop crashes.

---

## Part 1: AMD's Recommended Memory Query Pattern

### AMD Reference: Check Memory Before Allocating

**Source**: `HIP-Basic/device_query/main.cpp:134-141`

```cpp
// AMD's pattern for checking available memory
size_t free, total;
HIP_CHECK(hipMemGetInfo(&free, &total));

// Display results
std::cout << "memInfo.total: " << bytes_to_gib(total) << " GiB\n";
std::cout << "memInfo.free:  " << bytes_to_gib(free) << " GiB ("
          << (free / total * 100) << "%)\n";
```

**Key Insight**: Always query `hipMemGetInfo()` BEFORE allocating to know what's available.

### AMD Reference: Conservative Allocation

AMD doesn't explicitly show "allocate only X%" in basic examples, but their graph API tutorial shows checking memory before operations:

**Source**: `HIP-Doc/Tutorials/graph_api/src/main_graph_capture.hip:131-133`

```cpp
size_t devMemFree, devMemTotal;
hip_check(hipMemGetInfo(&devMemFree, &devMemTotal));
// Then use devMemFree to determine if operation is possible
```

---

## Part 2: Our Implementation - What We Have vs What We Use

### What We Have (Correct Safety Pattern)

**Source**: `src/backend/hip_backend/backend.rs:1474-1482`

```rust
/// Check if an allocation of given size is safe
///
/// Returns true if size < 70% of currently free GPU memory.
/// This prevents exhausting GPU memory needed by desktop/compositor.
pub fn can_allocate(&self, size: usize) -> HipResult<bool> {
    let (free, _total) = self.get_memory_info()?;

    // Safety margin: use only 70% of free memory
    // Leave 30% for desktop/compositor/driver overhead
    let safe_threshold = (free * 7) / 10;

    Ok(size <= safe_threshold)
}
```

**Source**: `src/backend/hip_backend/backend.rs:1497-1510`

```rust
/// Allocate buffer with conservative memory check
///
/// Returns error if requested size exceeds 70% of free GPU memory.
pub fn allocate_buffer_safe(&self, size: usize) -> HipResult<HipBuffer> {
    // First check if allocation is safe
    if !self.can_allocate(size)? {
        return Err(HipError::MemoryAllocationFailed(format!(
            "Allocation of {} bytes would exceed safe threshold (70% of free memory)",
            size
        )));
    }

    // Then allocate normally
    HipBuffer::new(size)
}
```

### What We Actually Use in Model Loading (UNSAFE!)

**Source**: `src/loader/gguf.rs:1197-1200` (in `load_to_gpu_async()` loop)

```rust
// allocate GPU buffer
let total_elements: usize = shape.iter().product();
let buffer = HipBuffer::new(total_elements * std::mem::size_of::<f32>())
    .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
```

**PROBLEM**:
- Uses `HipBuffer::new()` directly
- NO call to `can_allocate()` or `allocate_buffer_safe()`
- NO 70% safety margin
- Loops over ~200-400 tensors
- Each allocation is unchecked

### Where Safe Allocation IS Used

**Source**: `src/model/kv_cache.rs:79-81`

```rust
let keys_buffer = backend.allocate_buffer_safe(total_keys_bytes)
    .map_err(|e| ModelError::KvCacheAllocationFailed(e.to_string()))?;
let values_buffer = backend.allocate_buffer_safe(total_values_bytes)
    .map_err(|e| ModelError::KvCacheAllocationFailed(e.to_string()))?;
```

**Source**: `src/backend/hip_backend/backend.rs:2337-2347` (scratch buffer)

```rust
if !backend.can_allocate(total_bytes)? {
    return Err(HipError::MemoryAllocationFailed(format!(
        "Scratch buffer allocation ({}) would exceed safe memory threshold",
        bytes_to_gib(total_bytes)
    )));
}
let buffer = backend.allocate_buffer_safe(total_bytes)?;
```

**Status**: Safe allocation used in KV cache and scratch buffers, but NOT in model weight loading.

---

## Part 3: Model-Specific Requirements

### Supported Model Types

**Source**: `src/model/config.rs:5-13`

```rust
pub enum ModelType {
    Llama,    // Meta LLaMA family
    Qwen,     // Alibaba Qwen/Qwen2 family
    Mistral,  // Mistral AI models
    Yi,       // Yi models (similar to LLaMA)
    Mixtral,  // Mistral's MoE architecture
}
```

### Model-Specific Metadata Keys

**Source**: `src/loader/metadata.rs:50-200`

| Model Type | Key Pattern | Unique Requirements |
|------------|------------|-------------------|
| **Qwen2** | `qwen2.*` | `head_count_kv` (GQA/MQA), `rope.dimension_count` |
| **LLaMA** | `llama.*` | `head_count_kv`, `rope.dimension_count` |
| **Mistral** | `mistral.*` | Sliding Window Attention (in some versions) |
| **Yi** | `yi.*` | Similar to LLaMA/Mistral |
| **Mixtral** | `mixtral.*` | MoE: `n_experts`, `n_experts_per_tok` |
| **Gemma3** | `gemma3.*` | Google Gemma architecture |
| **GLM** | `glm.*` | ChatGLM architecture (different KV layout) |

### Critical Model Differences

#### 1. Attention Architecture (MQA/GQA vs MHA)

| Model | KV Heads | Query Heads | Ratio | Memory Impact |
|-------|----------|-------------|-------|---------------|
| LLaMA (7B) | 32 | 32 | 1:1 (MHA) | Standard |
| Qwen2 (0.5B) | 2 | 14 | 7:1 (GQA) | **Less KV memory** |
| Qwen2 (7B) | 4 | 28 | 7:1 (GQA) | **Less KV memory** |
| Mistral (7B) | 8 | 32 | 4:1 (GQA) | **Less KV memory** |
| Mixtral (8x7B) | 8 | 32 | 4:1 (GQA) | **Less KV memory** |

**Implication**:
- KV cache size depends on `num_kv_heads`, NOT `num_attention_heads`
- For Qwen2 with 28 query heads but only 4 KV heads, KV cache is 1/7th the size
- We MUST query `head_count_kv` from GGUF metadata

#### 2. Position Embedding Types

| Model | RoPE Type | Dimension | Notes |
|-------|-----------|-----------|-------|
| Qwen2 | Rotatory | Configurable | `qwen2.rope.dimension_count` |
| LLaMA | Rotatory | head_dim | Usually hidden_size / num_heads |
| Mistral | Rotatory | head_dim | Sliding window in some versions |
| GLM | 2D positional | Different | Separate implementation needed |

#### 3. Quantization-Specific Memory

| Quantization | Bits per Element | Memory vs FP16 |
|--------------|------------------|-----------------|
| Q4_0 | 4.5 | ~28% of FP16 |
| Q4_K | 4.5 | ~28% of FP16 |
| Q5_K | 5 | ~31% of FP16 |
| Q6_K | 6 | ~38% of FP16 |
| Q8_0 | 8 | ~50% of FP16 |
| FP16 | 16 | 100% (baseline) |

**Implication**: Smaller quantization doesn't just reduce weights - it reduces ALL memory including KV cache if quantized KV is used.

---

## Part 4: Memory Requirements by Model Size

### Estimated GPU Memory Requirements

**Formula**:
```
Total Memory ≈ Weights + KV Cache + Scratch + Overhead

Weights = model_size_bytes × quantization_ratio
KV Cache = 2 × num_layers × num_kv_heads × context_len × head_dim × batch_size × sizeof(float16)
Scratch ≈ 3 × hidden_size × batch_size × sizeof(float16)
Overhead ≈ 500MB (driver, display, etc.)
```

### Examples (Batch=1, Context=2048)

| Model | Quantization | Weights | KV Cache (24L) | Scratch | Total |
|-------|--------------|---------|----------------|---------|-------|
| Qwen2 0.5B | Q4_K | ~300 MB | ~40 MB | ~50 KB | ~350 MB |
| Qwen2 7B | Q4_K | ~4.2 GB | ~340 MB | ~400 KB | ~5 GB |
| Qwen2 14B | Q6_K | ~11 GB | ~680 MB | ~800 KB | ~12 GB |
| LLaMA 7B | Q4_K | ~4.2 GB | ~1.3 GB (MHA) | ~400 KB | ~6 GB |

**Key Insight**: Models with MQA/GQA (like Qwen2) need MUCH less KV cache memory.

---

## Part 5: Safe Allocation Strategy

### Recommended Pattern

```rust
// BEFORE loading model, check if we have enough memory
pub fn check_model_memory_requirements(
    backend: &HipBackend,
    config: &ModelConfig,
    quantization_bits: usize,
    context_len: usize,
    batch_size: usize,
) -> HipResult<()> {
    // 1. Get current available memory
    let (free, total) = backend.get_memory_info()?;

    // 2. Calculate weight memory (with quantization ratio)
    let weight_ratio = quantization_bits as f64 / 16.0; // vs FP16
    let estimated_weights = estimate_model_weights(config) * weight_ratio;

    // 3. Calculate KV cache memory (using num_kv_heads!)
    let kv_heads = config.num_kv_heads.unwrap_or(config.num_attention_heads);
    let kv_memory = 2 * config.num_hidden_layers * kv_heads
                    * context_len * config.head_dim * batch_size * 2; // sizeof(float16)

    // 4. Calculate scratch buffer
    let scratch_memory = 3 * config.hidden_size * batch_size * 2;

    // 5. Add safety margin (20% overhead)
    let total_needed = (estimated_weights + kv_memory + scratch_memory) as f64 * 1.2;

    // 6. Check against 70% of free memory
    let safe_threshold = (free * 7) / 10;

    if total_needed > safe_threshold as f64 {
        return Err(HipError::MemoryAllocationFailed(format!(
            "Model requires {:.2} GB but only {:.2} GB safely available",
            total_needed / 1e9, safe_threshold / 1e9
        )));
    }

    Ok(())
}
```

### Memory Pool Pattern (To Prevent Multiple Allocations)

```rust
pub struct ModelWeightArena {
    buffer: HipBuffer,  // Single large allocation
    offsets: HashMap<String, (usize, usize)>,  // name -> (offset, size)
}

impl ModelWeightArena {
    pub fn new(backend: &HipBackend, total_size: usize) -> HipResult<Self> {
        // Check if we can allocate this much
        if !backend.can_allocate(total_size)? {
            return Err(HipError::MemoryAllocationFailed(
                "Cannot allocate model weight arena".to_string()
            ));
        }

        // Single allocation instead of hundreds
        let buffer = backend.allocate_buffer_safe(total_size)?;

        Ok(ModelWeightArena {
            buffer,
            offsets: HashMap::new(),
        })
    }

    pub fn get_tensor_view(&self, name: &str) -> Option<&[u8]> {
        // Return sub-slice of the big buffer
        self.offsets.get(name).map(|(offset, size)| unsafe {
            &self.buffer.as_bytes()[*offset..*offset + *size]
        })
    }
}
```

---

## Part 6: Model Loading Checklist

### Before Loading Any Model

- [ ] Query `hipMemGetInfo()` for available GPU memory
- [ ] Parse model metadata to identify architecture
- [ ] Extract `num_kv_heads` (critical for KV cache sizing)
- [ ] Calculate total memory needed
- [ ] Check against 70% safety threshold
- [ ] If insufficient, fail EARLY with clear error message

### During Model Loading

- [ ] Use memory pool (single allocation) instead of per-tensor allocations
- [ ] Track cumulative allocation size
- [ ] Re-query `hipMemGetInfo()` periodically if loading large models
- [ ] Use `allocate_buffer_safe()` for any additional buffers

### Model-Specific Checks

| Model | Special Consideration |
|-------|----------------------|
| Qwen2 | Check `head_count_kv` for GQA ratio |
| Mixtral | MoE: Multiple expert weights per layer |
| GLM | Different KV layout (may need special handling) |
| LLaMA | Standard MHA (baseline for others) |

---

## Part 7: Immediate Fixes Needed

### Fix #1: Use Safe Allocation in Model Loading

**Location**: `src/loader/gguf.rs:1197-1200`

**Current (UNSAFE)**:
```rust
let buffer = HipBuffer::new(total_elements * std::mem::size_of::<f32>())
    .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
```

**Should Be**:
```rust
let buffer = backend.allocate_buffer_safe(total_elements * std::mem::size_of::<f32>())
    .map_err(|e| anyhow!("Failed to allocate GPU buffer for '{}': {}", name, e))?;
```

**But this still has the multiple allocation problem!** Need memory pool instead.

### Fix #2: Calculate Memory Requirements Before Loading

**Location**: `src/backend/hip_backend/backend.rs:3069-3076`

**Add BEFORE loading**:
```rust
// Check memory requirements before preload
eprintln!("load_from_gguf: Checking memory requirements...");
let (free, total) = backend.get_memory_info()?;
eprintln!("GPU memory: {:.2} GB free / {:.2} GB total",
         free as f64 / 1e9, total as f64 / 1e9);

// Calculate estimated requirements
let estimated_weights = /* sum of all tensor sizes */;
let kv_cache = /* based on num_kv_heads, layers, context */;
let total_needed = estimated_weights + kv_cache + scratch;

let safe_threshold = (free * 7) / 10;
if total_needed > safe_threshold {
    return Err(HipError::MemoryAllocationFailed(format!(
        "Model requires {:.2} GB but only {:.2} GB safely available. \
         Reduce model size, context length, or close other GPU applications.",
        total_needed / 1e9, safe_threshold / 1e9
    )));
}
```

### Fix #3: Use num_kv_heads for KV Cache Sizing

**Current Issue**: KV cache might be sized for `num_attention_heads` instead of `num_kv_heads`.

**Verify** `src/model/kv_cache.rs` uses the correct head count.

---

## Part 8: Reference Code Snippets

### AMD's Full Memory Query Pattern

```cpp
// Query device properties
hipDeviceProp_t props{};
HIP_CHECK(hipGetDeviceProperties(&props, device_id));

// Query memory info
size_t free, total;
HIP_CHECK(hipMemGetInfo(&free, &total));

// Check if allocation is safe
size_t allocation_size = /* calculate needed size */;
size_t safe_threshold = (free * 7) / 10;  // 70% safety margin

if (allocation_size > safe_threshold) {
    std::cerr << "ERROR: Would exceed safe memory threshold\n";
    return error_exit_code;
}

// Allocate
HIP_CHECK(hipMalloc(&d_buffer, allocation_size));
```

### Our Pattern (Should Be)

```rust
// Query memory info
let (free, total) = backend.get_memory_info()?;

// Check if safe
let allocation_size = /* calculate needed size */;
if !backend.can_allocate(allocation_size)? {
    return Err(HipError::MemoryAllocationFailed(
        "Would exceed safe memory threshold".to_string()
    ));
}

// Allocate safely
let buffer = backend.allocate_buffer_safe(allocation_size)?;
```

---

## Summary

| Issue | Current State | Required State | Priority |
|-------|---------------|----------------|----------|
| Memory check before load | ❌ Not done | ✅ Add check | 🔴 Critical |
| Safe allocation in load_to_gpu_async | ❌ Uses `HipBuffer::new()` | ✅ Use memory pool | 🔴 Critical |
| Model-specific KV sizing | ⚠️ Uses num_kv_heads? | ✅ Verify | 🟡 Medium |
| Multiple allocations | ❌ ~200-300 calls | ✅ Single pool | 🔴 Critical |
| 70% safety margin | ⚠️ Only in some paths | ✅ All paths | 🔴 Critical |

---

**End of Report** - Investigation only, no code changes.
