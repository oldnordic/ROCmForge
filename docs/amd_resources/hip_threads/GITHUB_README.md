# HIP Threads - GitHub Repository Analysis

**Date:** 2026-04-14
**Source:** https://github.com/ROCm/hipThreads
**Status:** Early-access preview (production use not recommended)

---

## Overview

HIP Threads is a **C++-style concurrency library** for AMD GPUs that brings familiar threading abstractions to GPU programming by implementing C++ threading and synchronization primitives for GPU code.

**Key Promise:** Port CPU `std::thread` code to GPU with minimal changes.

## System Requirements

### Critical Version Constraints

⚠️ **hipThreads currently works ONLY with ROCm 7.0.2**
- Other ROCm versions (including newer ones) are NOT supported
- This is a hard requirement - the library will not work with other versions

### Full Prerequisites

1. **Linux OS** (Ubuntu 24.04 recommended)
2. **CMake 3.21+**
3. **Build tools** (make or ninja)
4. **ROCm 7.0.2** (HIP runtime and hipcc) - specific version required
5. **libhipcxx v2.7** - must build from source
6. **rocThrust 4.2.0** - required for code examples

## Installation

```bash
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads
cmake -B build
cmake --build ./build
sudo cmake --install ./build
```

Default install location: `/opt/rocm`

## Usage in CMake

```cmake
find_package(hipthreads REQUIRED)
target_link_libraries(<your_target> hipthreads::hipthreads)
```

## Porting from CPU to GPU

### Minimal Changes Required

To port existing CPU code using `std::thread` to GPU:

1. Replace `std::thread` with `hip::thread`
2. Add `__device__` annotation to lambdas/functions running on GPU
3. Handle GPU memory allocation (CPU and GPU have separate memory pools)

**The familiar threading model remains the same** - no need to rewrite concurrency logic.

## Code Examples

### 1. SAXPY - Incremental GPU Porting

Location: `examples/saxpy/step3-simdize/`

Demonstrates natural progression from `std::thread` to optimized GPU execution.

```bash
cd examples/saxpy/step3-simdize/
cmake -B build
cmake --build ./build
./build/bin/saxpy
```

### 2. llama3.c - LLM Inference (🎯 HIGHLY RELEVANT)

Location: `examples/llama3.c/step4-simdize/`

**Port of llama3.c** (minimal LLaMA 3 inference engine in C) to `hip::thread`.

```bash
cd examples/llama3.c/step4-simdize/
cmake -B build
cmake --build ./build
```

**Model export and inference:**
```bash
# Export model (requires Meta LLaMA 3 weights)
python export.py llama3.2_3b_instruct_fp32.bin --meta-llama ../llama3.2-3b-instruct/

# Run inference
./build/bin/llama3 ~/models/llama3.2_3b_instruct_fp32.bin \
    -z ~/models/tokenizer.bin \
    -i "My car" \
    -n 100

# Start chat session
./build/bin/llama3 ~/models/llama3.2_3b_instruct_fp32.bin \
    -z ~/models/tokenizer.bin \
    -m chat
```

**Command-line options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-t <float>` | Temperature (0 to inf) | `1.0` |
| `-p <float>` | Top-p sampling (0 to 1) | `0.9` |
| `-s <int>` | Random seed | `time(NULL)` |
| `-n <int>` | Number of steps | `4096` |
| `-i <string>` | Input prompt | — |
| `-z <string>` | Path to tokenizer | — |
| `-m <string>` | Mode: `generate` or `chat` | `generate` |
| `-y <string>` | System prompt (chat mode) | — |

**This is directly relevant to Q6_K work** - demonstrates LLM inference on AMD GPUs using high-level abstractions.

## Key Limitations and Best Practices

### 1. Avoiding Deadlocks: Synchronous Calls and Scoping

**Problem:** `hip::thread` launches a persistent kernel (scheduler) that polls for work. Calling synchronous HIP functions causes deadlocks because they wait for ALL GPU tasks to finish—including the persistent idle kernel.

**Functions that cause deadlock:**
- `hipDeviceSynchronize`
- Synchronous `hipMemcpy`
- `thrust::copy`

**Solution A: Use Async APIs**
```cpp
// Use async versions
hipMemcpyAsync(dst, src, size, direction, stream);
hipMemsetAsync(dst, value, size, stream);
```

**Solution B: Scoping**
```cpp
// Wrap hip::thread objects in scoped block
{
    hip::thread t1([] __device__ { /* ... */ });
    hip::thread t2([] __device__ { /* ... */ });
    // Threads joined and persistent kernel destroyed here
}
// NOW safe to use synchronous calls
hipDeviceSynchronize();
```

### 2. Lambda Annotations

**Requirement:** Must use **extended lambdas** annotated with `__device__`.

```cpp
// ❌ WRONG - standard lambda
hip::thread t([]() { /* ... */ });

// ✅ CORRECT - extended lambda with __device__
hip::thread t([] __device__ () { /* ... */ });
```

**Device Functions:**
Host code cannot reference `__device__` functions directly. Wrap in lambda:

```cpp
__device__ float my_device_func(float x) {
    return x * 2.0f;
}

// ❌ WRONG - host can't call device function directly
// hip::thread t(my_device_func);

// ✅ CORRECT - wrap in __device__ lambda
hip::thread t([] __device__ () {
    float result = my_device_func(42.0f);
});
```

### 3. Memory and Data Transfer

**Requirements:**
- Arguments must be **TriviallyCopyable** (copied by value to device)
- **No complex types** (no `std::vector` or standard containers)
- **Raw pointers must point to GPU memory** (allocated via `hipMalloc`)
- **Never capture by reference `[&]`** if variable is on launching thread's stack
- **Shared data must exist in heap/global memory**

```cpp
// ❌ WRONG - passing host pointer
float* host_data = new float[100];
hip::thread t([host_data] __device__ () {
    float val = host_data[0]; // CRASH
});

// ❌ WRONG - capturing by reference
float value = 42.0f;
hip::thread t([&value] __device__ () {
    float val = value; // CRASH - can't access host stack
});

// ✅ CORRECT - GPU memory, capture by value
float* device_data;
hipMalloc(&device_data, 100 * sizeof(float));
hipMemcpy(device_data, host_data, 100 * sizeof(float), hipMemcpyHostToDevice);

hip::thread t([device_data] __device__ () {
    float val = device_data[0]; // OK
});
```

### 4. Synchronization Behavior

**GPU synchronization primitives are approximations:**

- **No Preemption or Blocking**: GPU does not support blocking or hardware preemption
- **`condition_variable::wait` spins or yields** (does not actually block)
- **`this_thread::yield` only returns control when yieldee has finished**
  - Yieldee will not be interrupted
  - Cannot yield back to caller

## API Reference

Documentation available in multiple forms:
- **Doxygen docs** in `docs/` directory
- **Source documentation** - Doxygen-style comments in `inc/` and `src/`
- **Tutorials and examples** - ROCm Blogs post

**View API docs locally:**
```bash
python3 -m http.server 5500 --directory docs/doxygen/html
# Open http://localhost:5500
```

## Implementation Details

### Cooperative Threading

The library implements:
- `hip::thread` - GPU thread primitive
- `hip::mutex` - Mutual exclusion
- `hip::lock_guard` - RAII-style locking
- `hip::condition_variable` - Condition synchronization
- Other C++ standard library threading primitives

### Multi-Fiber Execution

Supports **width parameter** to leverage GPU SIMD architecture.

## Key Insights for Q6_K Work

### 1. LLM Inference Example Exists

**`examples/llama3.c`** demonstrates:
- LLM inference on AMD GPUs using high-level abstractions
- No low-level kernel programming required
- Achieves competitive performance with C++-style code

**Relevance to Q6_K:**
- If llama3.c can run LLM inference with `hip::thread`, maybe quantization kernels can also benefit
- Worth investigating if `hip::thread` can simplify Q6_K implementation

### 2. Device Function Pattern is Native

HIP Threads **requires** `__device__` extended lambdas:

```cpp
hip::thread t([] __device__ () {
    // All GPU code here
    float result = my_device_function(x);
});
```

**This validates our Q6_K refactoring strategy:**
- Device functions (`vec_dot_q6_k`) are the correct pattern
- Main kernel should just call device functions
- Keeps main kernel simple for graph capture

### 3. Simplicity Wins

HIP Threads philosophy:
- "Familiar threading model remains the same"
- "No need to rewrite concurrency logic"
- Minimal changes from CPU to GPU

**Aligns with our Q6_K approach:**
- Device function pattern = simple, familiar
- Don't need complex inline bit manipulation
- Graph capture = simpler architecture

### 4. Performance Claims

Blog post claims **2.9x-6.4x speedups**.

**Our Q6_K goal: 2.2-3.7x improvement** by making it graph-compatible.

**This suggests:**
- Proper GPU utilization yields massive gains
- Graph compatibility is key to unlocking performance
- Our target is realistic and achievable

### 5. Critical Caveat: ROCm Version

⚠️ **hipThreads ONLY works with ROCm 7.0.2**

**Current system check needed:**
```bash
hipcc --version  # Check ROCm version
```

**If not ROCm 7.0.2:**
- Cannot use hipThreads directly
- But can still learn from the patterns and philosophy
- Device function approach is universal, not hipThreads-specific

## Next Steps

1. ✅ Read blog post (complete)
2. ✅ Read GitHub README (complete)
3. ⏳ Check current ROCm version
4. ⏳ If ROCm 7.0.2, experiment with hipThreads
5. ⏳ Investigate llama3.c example implementation
6. ⏳ Determine if hipThreads can simplify Q6_K refactoring
7. ⏳ Extract code patterns applicable to graph-compatible kernels

## License

Apache License v2.0 with LLVM Exceptions

## Disclaimers

Early-access preview - production use not recommended.

---

**Status:** GitHub README analyzed ✅ | Implementation investigation pending ⏳
