# llama.cpp GPU Memory Management Analysis

**Date**: 2026-01-20
**Source**: `/home/feanor/Projects/llama.cpp`
**Purpose**: Extract safe GPU memory allocation patterns for consumer hardware
**Status**: INVESTIGATION ONLY - No code changes

---

## Executive Summary

**KEY FINDING**: llama.cpp does NOT use percentage-based safety margins or check `hipMemGetInfo` before every allocation. Instead, it uses a **memory pool architecture** that:
1. Pre-calculates total memory needed
2. Allocates in **large chunks** (max 16 chunks)
3. Subdivides internally using free block tracking
4. Reuses buffers across inferences

This approach **eliminates the "multiple small allocations" pathology** that causes GPU hangs on RDNA3.

---

## Part 1: llama.cpp's Memory Architecture

### Overview: Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: ggml_backend_sched (Scheduler)                 │
│ - Queries device memory for tensor split decisions      │
│ - Calls reserve() to pre-allocate                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ Layer 2: ggml_gallocr (Graph Allocator)                │
│ - Pre-calculates total memory needed                    │
│ - Calls ggml_vbuffer_alloc() for each chunk            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ Layer 3: ggml_dyn_tallocr (Dynamic Tensor Allocator)   │
│ - Free block tracking within chunks                     │
│ - Best-fit allocation from free blocks                  │
└─────────────────────────────────────────────────────────┘
```

---

## Part 2: Memory Query Pattern (How They Check Available Memory)

### Location: `ggml/src/ggml-cuda/ggml-cuda.cu:4215-4243`

```c
static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);

    // Get current GPU memory info
    CUDA_CHECK(cudaMemGetInfo(free, total));

#if defined(__linux__)
    // For UMA systems (integrated GPUs), use system memory instead
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx->device));

    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        }
    }
#endif
}
```

**Key Insight**: This is called via `ggml_backend_dev_memory()` to make **decisions** about tensor splitting, NOT for per-allocation safety checks.

### Usage in Model Loading: `src/llama-model.cpp:2464-2478`

```cpp
// default split, by free memory
for (size_t i = 0; i < n_devices(); ++i) {
    ggml_backend_dev_t dev = devices[i];
    size_t total;
    size_t free;
    ggml_backend_dev_memory(dev, &free, &total);

    // devices can return 0 bytes for free and total memory if they do not
    // have any to report. in this case, we will use the host memory as a fallback
    if (free == 0 && total == 0) {
        ggml_backend_dev_memory(cpu_dev, &free, &total);
    }
    splits[i] = free;  // Use free memory to determine tensor split
}
```

---

## Part 3: Memory Pool Architecture (The Critical Pattern)

### Virtual Buffer with Chunks

**Location**: `ggml/src/ggml-alloc.c:99-100, 401-443`

```c
#define GGML_VBUFFER_MAX_CHUNKS 16

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};
```

**Key Points**:
- Max 16 chunks per buffer
- Each chunk = ONE large `hipMalloc` call
- Virtual buffer spans all chunks
- Tensors are allocated from chunks using offsets

### Chunk Allocation

**Location**: `ggml/src/ggml-alloc.c:427-443`

```c
static struct vbuffer * ggml_vbuffer_alloc(
    ggml_backend_buffer_type_t buft,
    const struct ggml_dyn_tallocr * talloc,
    enum ggml_backend_buffer_usage usage
) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;

        // SINGLE large allocation per chunk
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);  // Clean up on failure
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}
```

### Actual HIP Allocation

**Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:699-716`

```c
static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size
) {
    ggml_backend_cuda_buffer_type_context * buft_ctx =
        (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx =
        new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}
```

**IMPORTANT**: This is called **once per chunk**, NOT once per tensor!

---

## Part 4: Free Block Tracking (How Subdivision Works)

### Dynamic Tensor Allocator Structure

**Location**: `ggml/src/ggml-alloc.c:132-165`

```c
struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    size_t max_size;
    size_t n_free_blocks;
    struct free_block * free_blocks;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_size;  // 0 = unlimited
    int n_chunks;
    struct tallocr_chunk ** chunks;
};
```

### Allocation from Free Blocks

**Location**: `ggml/src/ggml-alloc.c:205-275`

```c
static struct buffer_address ggml_dyn_tallocr_alloc(
    struct ggml_dyn_tallocr * alloc,
    size_t size,
    const struct ggml_tensor * tensor
) {
    // Find best fitting free block across all chunks
    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t best_reuse = -1;

    for (int c = 0; c < alloc->n_chunks; c++) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        for (int b = 0; b < chunk->n_free_blocks; b++) {
            struct free_block * block = &chunk->free_blocks[b];
            if (block->size >= size) {
                size_t diff = block->size - size;
                if (diff < best_reuse) {
                    best_reuse = diff;
                    best_fit_chunk = c;
                    best_fit_block = b;
                    if (!best_reuse) break;  // Perfect fit
                }
            }
        }
    }

    if (best_fit_block == -1) {
        // No existing chunk has space - create new chunk
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }

    // Allocate from the free block
    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address addr = {.chunk = best_fit_chunk, .offset = block->offset};

    // Update free block
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    return addr;
}
```

---

## Part 5: Reserve Pattern (Pre-Allocation Strategy)

### API Usage Example

**Location**: `ggml/include/ggml-backend.h:254-293`

```c
// 1. Create scheduler with backends
sched = ggml_backend_sched_new(
    {backend_gpu, backend_cpu},
    NULL,
    num_backends,
    GGML_DEFAULT_GRAPH_SIZE,
    false,
    true
);

// 2. Build a "max batch size" graph for measurement
reserve_graph = build_graph(sched, max_batch_size);

// 3. Reserve memory based on max graph
ggml_backend_sched_reserve(sched, reserve_graph);

// 4. Use pre-reserved memory for actual inference
for (int iter = 0; iter < n_iter; iter++) {
    graph = build_graph(sched);
    ggml_backend_sched_graph_compute(sched, graph);
}
```

### Reserve Implementation

**Location**: `ggml/src/ggml-alloc.c:828-951`

```c
static bool ggml_gallocr_reserve_n_impl(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    bool no_alloc
) {
    // 1. Reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // 2. Allocate in hash table (calculates needed sizes)
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // 3. Reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;

        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ?
                ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }

        if (realloc) {
            ggml_vbuffer_free(galloc->buffers[i]);
            if (!no_alloc) {
                // Allocate ALL chunks at once
                galloc->buffers[i] = ggml_vbuffer_alloc(
                    galloc->bufts[i],
                    galloc->buf_tallocs[i],
                    GGML_BACKEND_BUFFER_USAGE_COMPUTE
                );
                if (galloc->buffers[i] == NULL) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    return false;
                }
            }
        }
    }

    return true;
}
```

**Key Insight**: Reserve is called **once at startup** with the maximum expected graph size. After that, all inferences reuse the pre-allocated memory.

---

## Part 6: Buffer Reuse Pattern

### Buffer Pool (Alternative to Free/Realloc)

**Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:329-416`

```c
struct ggml_cuda_buffer_pool {
    static const int MAX_BUFFERS = 16;

    struct ggml_cuda_buffer {
        void * ptr;
        size_t size;
    };

    int device;
    ggml_cuda_buffer buffer_pool[MAX_BUFFERS];
    size_t pool_size;

    void * alloc(size_t size) {
        // Try to reuse from pool
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr && b.size >= size) {
                // Found a reusable buffer
                void * ptr = b.ptr;
                b.ptr = nullptr;  // Mark as used
                b.size = 0;
                return ptr;
            }
        }

        // No reusable buffer - allocate new with 5% look-ahead
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        pool_size += look_ahead_size;
        return ptr;
    }

    void free(void * ptr, size_t size) {
        // Return to pool instead of actually freeing
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        // Pool full - actually free
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};
```

**Key Points**:
- Buffers are returned to pool instead of freed
- 5% look-ahead reduces need for reallocation
- Only actually frees when pool is full

---

## Part 7: What llama.cpp Does NOT Do

### ❌ No Percentage-Based Safety Margins

llama.cpp does NOT:
- Calculate "70% of free memory" as a threshold
- Check `hipMemGetInfo` before each allocation
- Use fixed safety percentages

### ❌ No Per-Tensor hipMalloc Calls

llama.cpp does NOT:
- Call `hipMalloc` once per tensor (that's 200-400 calls!)
- Allocate small buffers individually
- Free/reallocate during normal inference

### ❌ No Runtime Memory Checking During Allocation

The allocator trusts that:
1. The reserve phase calculated correctly
2. The large chunk allocations succeeded
3. Free block tracking will handle subdivision

---

## Part 8: Reusable Patterns for ROCmForge

### Pattern 1: Memory Pool Arena

```rust
/// Single large buffer subdivided for multiple tensors
pub struct ModelWeightArena {
    buffer: HipBuffer,        // ONE large allocation
    chunks: Vec<MemoryChunk>, // Subdivisions within the buffer
    free_blocks: Vec<FreeBlock>, // Free block tracking
}

struct MemoryChunk {
    offset: usize,
    size: usize,
}

struct FreeBlock {
    offset: usize,
    size: usize,
}

impl ModelWeightArena {
    pub fn new(backend: &HipBackend, total_size: usize) -> HipResult<Self> {
        // Check if we can allocate this much
        let (free, _total) = backend.get_memory_info()?;
        if total_size > free {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Model requires {} MB but only {} MB available",
                total_size / 1_048_576, free / 1_048_576
            )));
        }

        // SINGLE allocation instead of hundreds
        let buffer = HipBuffer::new(total_size)?;

        Ok(ModelWeightArena {
            buffer,
            chunks: Vec::new(),
            free_blocks: vec![FreeBlock { offset: 0, size: total_size }],
        })
    }

    pub fn allocate_tensor(&mut self, name: &str, size: usize) -> HipResult<usize> {
        // Find best fitting free block
        let best_idx = self.free_blocks.iter().position(|b| b.size >= size)
            .ok_or_else(|| HipError::MemoryAllocationFailed(
                format!("No free block large enough for '{}', needed {} bytes", name, size)
            ))?;

        let block = &mut self.free_blocks[best_idx];
        let offset = block.offset;

        // Split the block
        block.offset += size;
        block.size -= size;

        if block.size == 0 {
            self.free_blocks.remove(best_idx);
        }

        self.chunks.push(MemoryChunk { offset, size });
        Ok(offset)
    }
}
```

### Pattern 2: Reserve Before Load

```rust
/// Pre-calculate memory requirements before loading
pub fn check_model_memory_requirements(
    backend: &HipBackend,
    gguf: &GgufLoader,
    context_len: usize,
) -> HipResult<usize> {
    // 1. Query current available memory
    let (free, total) = backend.get_memory_info()?;

    // 2. Calculate weight memory from GGUF tensor info
    let weight_size = calculate_total_weight_size(gguf)?;

    // 3. Calculate KV cache (using num_kv_heads!)
    let metadata = gguf.metadata();
    let kv_heads = metadata.num_kv_heads.unwrap_or(metadata.num_heads);
    let kv_size = 2 * metadata.num_layers * kv_heads
                    * context_len * metadata.head_dim * 2; // f16

    // 4. Calculate scratch buffer
    let scratch_size = 3 * metadata.hidden_size * 2;

    // 5. Total with 10% overhead for alignment/padding
    let total_needed = (weight_size + kv_size + scratch_size) as f64 * 1.10;

    if total_needed as usize > free {
        return Err(HipError::MemoryAllocationFailed(format!(
            "Model requires {:.2} GB but only {:.2} GB available. \
             Try: smaller model, reduced context, or close other GPU apps.",
            total_needed / 1e9, free as f64 / 1e9
        )));
    }

    Ok(total_needed as usize)
}
```

### Pattern 3: Single Allocation Path

```rust
// BEFORE (current - BAD):
for (name, data) in dequantized.iter() {
    let buffer = HipBuffer::new(size)?;  // 200-300 allocations!
}

// AFTER (proposed):
// 1. Pre-calculate total size
let total_size = check_model_memory_requirements(&backend, &gguf, ctx_len)?;

// 2. Allocate once
let arena = ModelWeightArena::new(&backend, total_size)?;

// 3. Subdivide for each tensor
for (name, data) in dequantized.iter() {
    let offset = arena.allocate_tensor(name, size)?;
    // Copy data to buffer + offset
}
```

---

## Part 9: Comparison Summary

| Aspect | ROCmForge (Current) | llama.cpp | ROCmForge (Proposed) |
|--------|---------------------|-----------|---------------------|
| Allocation count | ~200-300 hipMalloc calls | ~16 hipMalloc calls (chunks) | ~1 hipMalloc call |
| Memory checking | 70% threshold (flawed) | Query for decisions only | Query once, then trust |
| Free block tracking | None | Yes, per chunk | Yes, in arena |
| Buffer reuse | None | Yes, pool + reserve | Yes, arena persists |
| Per-tensor overhead | High (alloc/free per tensor) | Low (offset calc) | Low (offset calc) |

---

## Part 10: Implementation Priority

### Critical (Prevents GPU Hang)

1. **Memory pool arena** - Replace loop of `HipBuffer::new()` with single allocation
2. **Pre-calculate size** - Check memory requirements before ANY allocation
3. **Free block tracking** - Subdivide single buffer instead of multiple allocations

### Important (Stability)

4. **Buffer reuse** - Keep arena alive across inferences
5. **Use `num_kv_heads`** - Correct KV cache sizing for GQA/MQA models

### Nice to Have

6. **Buffer pool** - Reuse allocations across model loads
7. **Look-ahead allocation** - 5% margin reduces reallocation

---

## Conclusion

llama.cpp's approach to GPU memory safety:

1. **NOT percentage-based margins** - Those don't work with concurrent GPU processes
2. **NOT per-allocation checks** - Too slow, doesn't prevent hangs
3. **YES memory pooling** - Single large allocation, subdivided
4. **YES reserve pattern** - Pre-calculate and pre-allocate
5. **YES buffer reuse** - Avoid free/realloc cycles

The key insight: **Avoid many small allocations**. Each `hipMalloc` call is a risk on RDNA3. Pooling eliminates this risk.

---

**End of Report** - Investigation only, no code changes made.
