# GPU Safety Protocols - ROCmForge Project

## 🚨 CRITICAL: All GPU code creation is protected by safety hooks

**DO NOT bypass these safety measures.** They exist to prevent GPU crashes and hardware damage.

---

## Automatic Code Quality Protections (Enforced by Claude Code hooks)

**The hooks prevent WRITING unsafe GPU code, not just running it.** When you create or modify GPU kernels or tests, the hooks verify:

### 1. **Kernel Safety Checks** (`.hip` and `.cpp` files)
- ✅ **Early bounds check required:** `if (col >= ncols_dst) return;`
- ✅ **Parameter validation required:** `CHECK_NULL(weights/input/output)`
- ✅ **Q6_K linear processing required:** No nested loops, no pointer arithmetic

**What gets blocked:**
```cpp
// ❌ BLOCKED: Missing bounds check
__global__ void unsafe_kernel(...) {
    const int col = blockIdx.x;
    // No bounds check - will crash GPU!
    sum += data[col];
}

// ✅ ALLOWED: Safe kernel with bounds check
__global__ void safe_kernel(...) {
    const int col = blockIdx.x;
    if (col >= ncols_dst) return;  // Required safety check
    sum += data[col];
}
```

### 2. **Launch Function Safety** (`.hip` and `.cpp` files)
- ✅ **Parameter validation:** `if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;`
- ✅ **Null pointer checks:** `CHECK_NULL(weights); CHECK_NULL(input); CHECK_NULL(output);`

**What gets blocked:**
```cpp
// ❌ BLOCKED: Missing parameter validation
extern "C" hipError_t unsafe_launch(...) {
    // No validation - will crash GPU on invalid input!
    kernel<<<...>>>(...);
    return hipSuccess;
}

// ✅ ALLOWED: Safe launch with validation
extern "C" hipError_t safe_launch(...) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    CHECK_NULL(weights);
    CHECK_NULL(input);
    CHECK_NULL(output);
    kernel<<<...>>>(...);
    return hipGetLastError();
}
```

### 3. **Q6_K Linear Processing** (`q6_k*.hip` files)
- ✅ **Linear processing only:** `for (int l = 0; l < 8; ++l) { const int i = tid * 8 + l; ... }`
- ❌ **Nested loops blocked:** `for (int n = 0; n < QK_K; n += 128) { for (int l = 0; l < 32; ++l) { ... } }`
- ❌ **Pointer arithmetic blocked:** `ql += 64; qh += 32; scales += 8;`

**What gets blocked:**
```cpp
// ❌ BLOCKED: Nested loops with pointer arithmetic
for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; ++l) {
        ql += 64;  // Pointer arithmetic - fails graph capture
        // ...
    }
}

// ✅ ALLOWED: Linear processing
for (int l = 0; l < 8; ++l) {
    const int i = tid * 8 + l;  // Linear calculation
    // Direct array access only
}
```

### 4. **Test Safety Measures** (`test*gpu*.rs` files)
- ✅ **Q6_K tests must disable graph:** `ROCMFORGE_DISABLE_DECODE_GRAPH=1`
- ✅ **Timeout required:** `timeout 30 <command>`
- ✅ **Token limits required:** `--max-tokens 10`

**What gets blocked:**
```rust
// ❌ BLOCKED: GPU test without safety measures
#[test]
fn test_gpu_unsafe() {
    // No graph disable, no timeout, no token limits - will crash GPU!
    run_gpu_test();
}

// ✅ ALLOWED: Safe GPU test
#[test]
fn test_gpu_safe() {
    // ROCMFORGE_DISABLE_DECODE_GRAPH=1 for Q6_K
    // timeout to prevent hangs
    // --max-tokens to prevent unbounded execution
}
```

---

## How The Hooks Work

The following safety checks are **automatically enforced** before any GPU command runs:

### 1. **Q6_K Graph Capture Block**
- **Blocks:** Any Q6_K model execution with graph capture enabled
- **Reason:** Q6_K with graph capture crashes on multi-token prompts (memory access fault)
- **Required:** Must use `ROCMFORGE_DISABLE_DECODE_GRAPH=1`

### 2. **Multi-Token Prompt Protection**
- **Blocks:** Prompts >5 characters with graph capture enabled
- **Reason:** Multi-token prefill with graph capture causes GPU crashes
- **Required:** Use `ROCMFORGE_DISABLE_DECODE_GRAPH=1` for multi-token prompts

### 3. **Timeout Enforcement**
- **Blocks:** Any GPU command without `timeout` wrapper
- **Reason:** Prevents GPU hangs and reset crashes from hung processes
- **Required:** Must use `timeout 30 <command>`

### 4. **Token Limit Enforcement**
- **Blocks:** GPU commands without `--max-tokens` limit
- **Reason:** Prevents unbounded GPU execution crashes
- **Required:** Must use `--max-tokens 10` (or similar reasonable limit)

---

## Safe GPU Command Patterns

### ✅ Q6_K Models (Graph Disabled)
```bash
# Safe: Single-token prompt with timeout
ROCMFORGE_DISABLE_DECODE_GRAPH=1 timeout 30 \
  ./target/release/rocmforge --gpu \
  --model /path/to/q6_k.gguf \
  --prompt "Hi" \
  --max-tokens 10

# Safe: Multi-token prompt with timeout
ROCMFORGE_DISABLE_DECODE_GRAPH=1 timeout 60 \
  ./target/release/rocmforge --gpu \
  --model /path/to/q6_k.gguf \
  --prompt "Hello world" \
  --max-tokens 20
```

### ✅ Q4_K / Q8_0 Models (Graph Enabled)
```bash
# Safe: Single-token prompt with graph
timeout 30 ROCMFORGE_DISABLE_DECODE_GRAPH=0 \
  ./target/release/rocmforge --gpu \
  --model /path/to/q4_k.gguf \
  --prompt "Hi" \
  --max-tokens 10

# Safe: Multi-token prompt WITHOUT graph
timeout 60 ROCMFORGE_DISABLE_DECODE_GRAPH=1 \
  ./target/release/rocmforge --gpu \
  --model /path/to/q4_k.gguf \
  --prompt "Longer prompt here" \
  --max-tokens 20
```

### ❌ UNSAFE PATTERNS (Blocked by hooks)
```bash
# BLOCKED: No timeout
./target/release/rocmforge --gpu --model x.gguf --prompt "Hi" --max-tokens 10

# BLOCKED: No max-tokens
timeout 30 ./target/release/rocmforge --gpu --model x.gguf --prompt "Hi"

# BLOCKED: Q6_K with graph enabled
timeout 30 ROCMFORGE_DISABLE_DECODE_GRAPH=0 \
  ./target/release/rocmforge --gpu \
  --model q6_k.gguf --prompt "Hi" --max-tokens 10

# BLOCKED: Multi-token prompt with graph
timeout 30 ROCMFORGE_DISABLE_DECODE_GRAPH=0 \
  ./target/release/rocmforge --gpu \
  --model q4_k.gguf --prompt "Long prompt text" --max-tokens 10
```

---

## GPU Test Development

When developing GPU tests, follow this workflow:

### 1. Development Phase (Graph Disabled)
```bash
# Develop with graph disabled to avoid crashes
export ROCMFORGE_DISABLE_DECODE_GRAPH=1
cargo test --release --features gpu --test gpu_decode_real
```

### 2. Single-Token Testing (Graph Enabled)
```bash
# Test graph capture with single-token prompts only
timeout 30 ROCMFORGE_DISABLE_DECODE_GRAPH=0 \
  ./target/release/rocmforge --gpu \
  --model test.q4_k.gguf \
  --prompt "X" \
  --max-tokens 5
```

### 3. Production Testing (Graph Disabled for Q6_K)
```bash
# Final testing with real prompts (graph disabled for Q6_K)
timeout 60 ROCMFORGE_DISABLE_DECODE_GRAPH=1 \
  ./target/release/rocmforge --gpu \
  --model production.q6_k.gguf \
  --prompt "Real user prompt" \
  --max-tokens 50
```

---

## Current GPU Status (2026-04-14)

| Quantization | Graph Capture | Multi-Token | Notes |
|--------------|---------------|-------------|-------|
| **Q4_K**     | ✅ Compatible  | ✅ Safe     | Works with graph for all prompts |
| **Q8_0**     | ✅ Compatible  | ✅ Safe     | Works with graph for all prompts |
| **Q6_K**     | ⚠️ Partial     | ❌ CRASHES  | Single-token OK, multi-token crashes |

**Q6_K Status Details:**
- Kernel rewritten with linear processing (no nested loops)
- Works with graph capture for single-token prompts (82 tok/s)
- **CRASHES** with graph capture for multi-token prompts (memory access fault)
- Safe operation: **Always use `ROCMFORGE_DISABLE_DECODE_GRAPH=1` for Q6_K**
- Performance: 95 tok/s without graph (still excellent)

---

## Emergency GPU Recovery

If GPU crashes despite protections:

```bash
# 1. Check GPU status
dmesg | grep -i "amdgpu\|reset\|gpu" | tail -20

# 2. Check for GPU hang
rocminfo | grep "GPU Reset"

# 3. If GPU is hung, reset ROCm
sudo /opt/rocm/bin/rocminfo --reset

# 4. Check ROCm installation
/opt/rocm/bin/rocminfo --showall
```

---

## Safety Hook Configuration

The safety hooks are defined in `.claude/settings.json` (gitignored for local customization).

**To disable protections (NOT RECOMMENDED):**
```bash
# Temporarily disable hooks (DANGEROUS)
mv .claude/settings.json .claude/settings.json.disabled
# ... do dangerous GPU work ...
mv .claude/settings.json.disabled .claude/settings.json
```

**To customize protections:**
Edit `.claude/settings.json` and adjust the `hooks.PreToolUse` section.

---

## Reference Documentation

- **GPU Kernel Design Guidelines:** `docs/gpu_kernel_design_guidelines.md`
- **Q6_K Implementation:** `hip_kernels/quant/q6_k_gemv.hip`
- **Graph Capture API:** `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html`
- **ROCm Profiling:** `.rocprofv3/profile_decode.sh`

---

## Rule of Thumb

> **"When in doubt, disable graph. When developing, use timeout. When testing Q6_K, no graph."**

**Before running any GPU command, ask:**
1. ✅ Is `timeout` set?
2. ✅ Is `--max-tokens` set?
3. ✅ Is graph disabled for Q6_K?
4. ✅ Is prompt short for graph testing?

If any answer is "NO", **add the safety measure first**.

---

## History of GPU Crashes (Learnings)

1. **Q6_K Original Kernel** - Nested loops with pointer arithmetic → HIP error 901
2. **Q6_K Linear Rewrite v1** - Wrong element distribution → Memory access fault
3. **Q6_K Linear Rewrite v2** - Correct interleaved distribution → Graph capture works for single-token, **CRASHES for multi-token**
4. **Current Protection** - Safety hooks prevent all known crash scenarios

**Each crash taught us a new safety rule.** The hooks prevent repeating these mistakes.
