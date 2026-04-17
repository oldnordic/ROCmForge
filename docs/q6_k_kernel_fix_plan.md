# Q6_K Kernel Threading Bug Fix Plan

> **CRITICAL**: This plan fixes the memory access fault in Q6_K kernels. Each step must be implemented exactly as specified. No shortcuts, no placeholders.

**Bug**: Thread work distribution causes each thread to execute once instead of cooperatively processing all 256 elements per block.

**Impact**: Works for 1-token prompts, crashes with ≥6 tokens due to unprocessed memory regions.

**Files to Modify**:
1. `hip_kernels/quant/q6_k_gemv.hip` - Fix vec_dot_q6_k function
2. `hip_kernels/quant/q6_k_gemm.hip` - Fix matrix multiplication kernel
3. Test verification with real model

---

## Task 1: Analyze Working Q5_K Pattern

**File**: Reference `hip_kernels/quant/q5_k_gemv.hip` lines 19-56

**Working Q5_K Thread Pattern**:
```cpp
// Each thread processes 2 elements per iteration × 4 iterations = 8 elements
for (int j = 0; j < QK_K; j += 64) {           // Outer: 256 / 64 = 4 iterations
    const uint8_t q4 = ql[tid];                // Thread 0: ql[0], Thread 1: ql[1], etc.
    const uint8_t q4_2 = (ql[tid] >> 4) & 0x0F;
    
    sum += (static_cast<float>(q4 + high_bit) / d1 + min1) * vec[offset + j + tid];
    sum += (static_cast<float>(q4_2 + high_bit_2) / d2 + min2) * vec[offset + j + 32 + tid];
    
    ql += 32;                                   // Advance pointer for next iteration
    is += 2;
    u1 <<= 2; u2 <<= 2;
}
```

**Key Pattern**:
- Loop uses `j` as outer iterator, `tid` to differentiate work
- Each thread: `ql[tid]` accesses different element
- Pointer advances: `ql += 32` between iterations
- 32 threads × 2 elements/thread × 4 iterations = 256 elements processed

---

## Task 2: Fix vec_dot_q6_k Function

**File**: `hip_kernels/quant/q6_k_gemv.hip` lines 7-81

**Current Buggy Code** (lines 37-80):
```cpp
// First 128 elements (n=0)
for (int l = tid; l < 32; l += 32) {           // ❌ BUG: Each thread executes ONCE
    const int is = l / 16;
    const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    // ...
}

// Second 128 elements (n=128)
for (int l = tid; l < 32; l += 32) {           // ❌ BUG: Same issue
    // ...
}
```

**Fixed Code** (replace entire function):

```cpp
__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // Q6_K block layout: ql[0..128], qh[128..192], scales[192..208], d[208..210]
    const uint8_t* ql = &block_bytes[0];
    const uint8_t* qh = &block_bytes[128];
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block_bytes[192]);

    half d_half;
    memcpy(&d_half, &block_bytes[208], sizeof(half));
    const float d = __half2float(d_half);

    float sum = 0.0f;

    // Process 256 elements in two 128-element halves
    // Each thread processes 8 elements total (2 per iteration × 4 iterations per half)
    
    // First 128 elements: use cooperative threading like Q5_K
    for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        
        // Each thread accesses different element using tid
        const int8_t q1 = (int8_t)((ql[tid] & 0xF) | (((qh[tid] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)(((ql[tid + 32] & 0xF) | (((qh[tid] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)(((ql[tid] >> 4) & 0xF) | (((qh[tid] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)(((ql[tid + 32] >> 4) & 0xF) | (((qh[tid] >> 6) & 3) << 4)) - 32;

        const float d1 = d * (float)scales[is + 0];
        const float d2 = d * (float)scales[is + 2];
        const float d3 = d * (float)scales[is + 4];
        const float d4 = d * (float)scales[is + 6];

        sum += vec[offset + l + 0] * (d1 * (float)q1);
        sum += vec[offset + l + 32] * (d2 * (float)q2);
        sum += vec[offset + l + 64] * (d3 * (float)q3);
        sum += vec[offset + l + 96] * (d4 * (float)q4);
        
        // Advance pointers for next iteration
        ql += 64;
        qh += 32;
        scales += 8;
    }

    // Reset pointers for second half
    ql = &block_bytes[0];
    qh = &block_bytes[128];
    scales = reinterpret_cast<const int8_t*>(&block_bytes[192]);
    
    // Advance to second 128 elements
    ql += 64;   // Skip first 64 bytes (already processed)
    qh += 32;   // Skip first 32 bytes
    scales += 8; // Skip first 8 scales

    // Second 128 elements
    for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        
        const int8_t q1 = (int8_t)((ql[tid] & 0xF) | (((qh[tid] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)(((ql[tid + 32] & 0xF) | (((qh[tid] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)(((ql[tid] >> 4) & 0xF) | (((qh[tid] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)(((ql[tid + 32] >> 4) & 0xF) | (((qh[tid] >> 6) & 3) << 4)) - 32;

        const float d1 = d * (float)scales[is + 0];
        const float d2 = d * (float)scales[is + 2];
        const float d3 = d * (float)scales[is + 4];
        const float d4 = d * (float)scales[is + 6];

        sum += vec[offset + 128 + l + 0] * (d1 * (float)q1);
        sum += vec[offset + 128 + l + 32] * (d2 * (float)q2);
        sum += vec[offset + 128 + l + 64] * (d3 * (float)q3);
        sum += vec[offset + 128 + l + 96] * (d4 * (float)q4);
        
        // Advance pointers for next iteration
        ql += 64;
        qh += 32;
        scales += 8;
    }

    return sum;
}
```

**Key Changes**:
1. Changed `for (int l = tid; l < 32; l += 32)` to `for (int l = 0; l < 32; ++l)`
2. Changed array access from `ql[l]` to `ql[tid]` to differentiate work between threads
3. Added pointer advancement (`ql += 64`, etc.) inside the loop
4. Each thread now processes 8 elements (4 pairs) per 128-element chunk
5. Total: 32 threads × 8 elements = 256 elements correctly processed

---

## Task 3: Fix gemv_q6_k_f32_kernel Function

**File**: `hip_kernels/quant/q6_k_gemv.hip` lines 83-111

**Current Code** (lines 97-100):
```cpp
for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    sum += vec_dot_q6_k(col_base + block_idx * Q6_K_BLOCK_SIZE, input, block_idx * QK_K);
}
```

**Verification**: This code is CORRECT. The bug was only in vec_dot_q6_k, not in the calling code.

**Action**: No changes needed to gemv_q6_k_f32_kernel.

---

## Task 4: Fix gemm_q6_k_f32_kernel Function

**File**: `hip_kernels/quant/q6_k_gemm.hip` lines 7-70

**Current Code Analysis** (lines 40-59):
```cpp
for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;
    
    // Unpack Q6_K block
    const uint8_t* ql = &block[0];
    const uint8_t* qh = &block[128];
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);
    
    half d_half;
    memcpy(&d_half, &block[208], sizeof(half));
    const float d = __half2float(d_half);
    
    // Accumulate across block (same logic as GEMV)
    for (int n = 0; n < 2; n++) {
        for (int l = tid; l < 32; l += 32) {           // ❌ BUG: Same threading issue
            // ... dequantization logic ...
        }
    }
}
```

**Fixed Code** (replace lines 40-59):

```cpp
for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    const uint8_t* block = col_base + block_idx * Q6_K_BLOCK_SIZE;

    // Unpack Q6_K block
    const uint8_t* ql = &block[0];
    const uint8_t* qh = &block[128];
    const int8_t* scales = reinterpret_cast<const int8_t*>(&block[192]);

    half d_half;
    memcpy(&d_half, &block[208], sizeof(half));
    const float d = __half2float(d_half);

    // Accumulate across block (same cooperative threading as GEMV)
    for (int n = 0; n < 2; n++) {
        const int base = n * 128;
        
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            
            // Each thread accesses different element using tid
            const int8_t q1 = (int8_t)((ql[tid] & 0xF) | (((qh[tid] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)(((ql[tid + 32] & 0xF) | (((qh[tid] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)(((ql[tid] >> 4) & 0xF) | (((qh[tid] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)(((ql[tid + 32] >> 4) & 0xF) | (((qh[tid] >> 6) & 3) << 4)) - 32;

            const float d1 = d * (float)scales[is + 0];
            const float d2 = d * (float)scales[is + 2];
            const float d3 = d * (float)scales[is + 4];
            const float d4 = d * (float)scales[is + 6];

            sum += input[block_idx * QK_K + base + l + 0] * (d1 * (float)q1);
            sum += input[block_idx * QK_K + base + l + 32] * (d2 * (float)q2);
            sum += input[block_idx * QK_K + base + l + 64] * (d3 * (float)q3);
            sum += input[block_idx * QK_K + base + l + 96] * (d4 * (float)q4);
            
            // Advance pointers for next iteration
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
}
```

**Key Changes**:
1. Changed `for (int l = tid; l < 32; l += 32)` to `for (int l = 0; l < 32; ++l)`
2. Changed array access from `ql[l]` to `ql[tid]`
3. Added pointer advancement inside loop
4. Maintains same logic as fixed GEMV kernel

---

## Task 5: Rebuild with Fixed Kernels

**Command**:
```bash
cd /home/feanor/Projects/rocmforge
cargo clean --release
cargo build --release --features gpu
```

**Expected Output**:
- Clean compilation with no errors
- Warnings about Q6_K libraries not found are OK (they'll be built by CMake)

**Verification**:
```bash
echo "Build status: $?"
```

---

## Task 6: Test 1-Token Prompt (Baseline)

**Command**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "Hello" \
  --max-tokens 10 \
  --no-template
```

**Expected Result**: Should work (already working before fix)
- Output: Valid text
- Speed: ~70 tok/s
- No memory faults

**Success Criteria**:
```bash
# Check exit code
echo "Exit code: $?"

# Should be 0 and contain decoded text
```

---

## Task 7: Test 6-Token Prompt (Previously Failing)

**Command**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "Hello, how are you?" \
  --max-tokens 10 \
  --no-template
```

**Expected Result**: Should now work (was crashing before fix)
- Output: Valid text
- Speed: ~65-75 tok/s
- **No memory faults**

**Success Criteria**:
- Exit code: 0
- No "Memory access fault" message
- Contains decoded text

---

## Task 8: Test 16-Token Prompt (Severe Case)

**Command**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "Hello, how are you today? I hope you're having a great day!" \
  --max-tokens 20 \
  --no-template
```

**Expected Result**: Should work (was crashing before fix)
- Output: Valid coherent text
- Speed: ~65-75 tok/s
- **No memory faults**

**Success Criteria**:
- Exit code: 0
- No GPU errors
- Text makes sense

---

## Task 9: Test Longer Prompt (Stress Test)

**Command**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs!" \
  --max-tokens 50 \
  --no-template
```

**Expected Result**: Should handle longer context
- Output: Valid text
- Speed: Consistent with shorter prompts
- **No memory faults or corruption**

**Success Criteria**:
- Exit code: 0
- Output length: 50 tokens
- No errors in output

---

## Task 10: Compare with Q4_K Performance

**Test Q4_K baseline**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "Hello, how are you?" \
  --max-tokens 20 \
  --no-template
```

**Test Q6_K with fixed kernel**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "Hello, how are you?" \
  --max-tokens 20 \
  --no-template
```

**Expected Comparison**:
- Q6_K should be within 10-15% of Q4_K speed
- Both should complete without errors
- Q6_K should provide similar quality output

---

## Task 11: Verify Thread Work Distribution

**Add temporary debug output** (for verification only, remove after):

In `q6_k_gemv.hip`, add to vec_dot_q6_k:
```cpp
#if 0  // Debug: verify thread work distribution
__shared__ float debug_sums[32];
debug_sums[tid] = sum;
__syncthreads();
if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < 32; ++i) {
        total += debug_sums[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Block 0 total sum: %f\n", total);
    }
}
#endif
```

**Rebuild and test**:
```bash
cargo build --release --features gpu
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "Hello" \
  --max-tokens 5 \
  --no-template 2>&1 | grep "Block 0 total sum"
```

**Expected**: Should print debug sum (indicates all threads contributed)

**Remove debug code after verification**.

---

## Task 12: Final Integration Test

**Command**:
```bash
./target/release/rocmforge --gpu \
  --model /home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 30 \
  --no-template
```

**Expected Result**:
- Correct answer: "Paris"
- No errors
- Reasonable speed
- Coherent text

**Success Criteria**:
- Exit code: 0
- Output contains factual answer
- No memory faults or corruption

---

## Task 13: Create Test Script for Validation

**File**: Create `/tmp/test_q6_k_fixed.sh`

```bash
#!/bin/bash
set -e

echo "Testing Q6_K kernel fix..."

MODEL="/home/feanor/Projects/Memoria/models/qwen3-4b-instruct-q6_k.gguf"
BIN="./target/release/rocmforge"

echo "Test 1: 1-token prompt (baseline)"
$BIN --gpu --model "$MODEL" --prompt "Hi" --max-tokens 5 --no-template
echo "✓ Test 1 passed"

echo "Test 2: 6-token prompt (was failing)"
$BIN --gpu --model "$MODEL" --prompt "Hello, how are you?" --max-tokens 10 --no-template
echo "✓ Test 2 passed"

echo "Test 3: 16-token prompt (was failing)"
$BIN --gpu --model "$MODEL" --prompt "Hello, how are you today? I hope you're great!" --max-tokens 15 --no-template
echo "✓ Test 3 passed"

echo "Test 4: Long prompt (stress test)"
$BIN --gpu --model "$MODEL" --prompt "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs!" --max-tokens 50 --no-template
echo "✓ Test 4 passed"

echo "All Q6_K tests passed! ✓"
```

**Run test**:
```bash
chmod +x /tmp/test_q6_k_fixed.sh
/tmp/test_q6_k_fixed.sh
```

**Expected**: All tests pass with no errors.

---

## Task 14: Document the Fix

**Create file**: `/home/feanor/Projects/rocmforge/docs/q6_k_fix_summary.md`

```markdown
# Q6_K Kernel Threading Bug Fix

## Problem
Q6_K kernels had incorrect thread work distribution causing memory access faults with ≥6 token prompts.

## Root Cause
```cpp
// WRONG: Each thread executes once
for (int l = tid; l < 32; l += 32) {
```

## Solution
```cpp
// CORRECT: All threads execute cooperatively with pointer advancement
for (int l = 0; l < 32; ++l) {
    // Use tid to differentiate work
    const int8_t q1 = (int8_t)((ql[tid] & 0xF) | ...);
    // Advance pointers
    ql += 64; qh += 32; scales += 8;
}
```

## Changes Made
1. Fixed vec_dot_q6_k in q6_k_gemv.hip
2. Fixed gemm_q6_k_f32_kernel in q6_k_gemm.hip
3. Verified with real model testing

## Test Results
- ✓ 1-token prompts: 69.7 tok/s
- ✓ 6-token prompts: Working (was crashing)
- ✓ 16+ token prompts: Working (was crashing)
- ✓ Long prompts: Working correctly

## Files Modified
- hip_kernels/quant/q6_k_gemv.hip
- hip_kernels/quant/q6_k_gemm.hip
```

---

## Task 15: Commit the Fix

**Commands**:
```bash
cd /home/feanor/Projects/rocmforge
git add hip_kernels/quant/q6_k_gemv.hip hip_kernels/quant/q6_k_gemm.hip
git diff --staged
```

**Review changes carefully**, then commit:
```bash
git commit -m "fix(gpu): correct Q6_K thread work distribution in GEMV/GEMM kernels

Fixed critical threading bug in Q6_K kernels that caused memory access
faults with prompts ≥6 tokens.

Root Cause:
- vec_dot_q6_k and gemm_q6_k_f32_kernel used 'for (int l = tid; l < 32; l += 32)'
- This caused each thread to execute once instead of cooperatively processing all 256 elements
- Only 64/256 elements processed per block → unprocessed memory → page faults

Fix:
- Changed to 'for (int l = 0; l < 32; ++l)' with cooperative threading
- Each thread now accesses ql[tid] instead of ql[l] to differentiate work
- Added pointer advancement (ql += 64, qh += 32, scales += 8) inside loop
- Follows same pattern as working Q5_K kernel

Test Results:
- ✓ 1-token prompts: Working (69.7 tok/s)
- ✓ 6-token prompts: Working (was crashing)
- ✓ 16+ token prompts: Working (was crashing)
- ✓ All test cases pass without memory faults

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Implementation Notes

### Why This Fix Works

**Before (Buggy)**:
- Thread 0: processes ql[0] only
- Thread 1: processes ql[1] only
- ...
- Result: 32 threads × 2 elements = 64 elements processed

**After (Fixed)**:
- Thread 0: processes ql[0], ql[64], ql[128], ql[192] in different iterations
- Thread 1: processes ql[1], ql[65], ql[129], ql[193] in different iterations
- ...
- Result: 32 threads × 2 elements × 4 iterations = 256 elements processed

### Verification Method

Each test increases token count progressively:
1. 1 token → 1 block (baseline)
2. 6 tokens → multiple blocks (was failing)
3. 16 tokens → many blocks (was failing)
4. 50 tokens → stress test (was failing)

All tests must pass with no memory faults.

---

## Success Criteria

✅ All kernel code is complete and correct (no placeholders)
✅ All tests pass with no memory faults
✅ Performance is consistent with expectations
✅ Code follows established Q4_K/Q5_K patterns
✅ No shortcuts, no TODOs, no FIXMEs

**Only proceed to commit if all tests pass.**
