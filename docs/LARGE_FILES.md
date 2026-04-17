# Large Source Files Audit (>1,000 LOC)

**Date:** 2026-04-15  
**Purpose:** Identify source files that need refactoring due to size  
**Threshold:** >1,000 lines of code (LOC)  
**Scope:** Production source code only (excludes tests)

---

## Summary

- **Total source files over 1,000 LOC:** 7
- **Total lines in large source files:** 13,309 LOC
- **Largest file:** `src/gpu/kernels/quant.rs` (3,750 LOC)
- **Languages:** Rust

---

## Files Over 1,000 LOC (Source Code Only)

| Rank | File | LOC | Type | Recommended Split |
|------|------|-----|------|-------------------|
| 1 | `src/gpu/kernels/quant.rs` | 3,750 | Quant kernels | `quantq4.rs`, `quantq4_k.rs`, `quantq6.rs`, etc. |
| 2 | `src/cpu/ops.rs` | 2,323 | CPU operations | By operation type |
| 3 | `src/gpu/quant_wrapper.rs` | 1,712 | GPU FFI wrappers | Match quant.rs structure |
| 4 | `src/gpu/forward.rs` | 1,630 | GPU forward pass | By stage (prefill/decode) |
| 5 | `src/gpu/weights.rs` | 1,425 | GPU weights | By functionality |
| 6 | `src/gpu/ops.rs` | 1,274 | GPU operations | By operation type |
| 7 | `src/config.rs` | 1,095 | Configuration | By config section |

---

## Detailed Refactoring Plan

### 1. `src/gpu/kernels/quant.rs` → **3,750 LOC**

**Current State:** All quantization kernel FFI declarations in one file
- Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 kernels
- GEMM, GEMV, quantize, dequantize, verify operations
- Mixed formats make it hard to find/fix bugs

**Target Structure:**
```
src/gpu/kernels/quant/
├── mod.rs              # Public exports, ~50 LOC
├── q4_0.rs             # Q4_0 kernels (~400 LOC)
├── q4_1.rs             # Q4_1 kernels (~400 LOC)
├── q4_k.rs             # Q4_K kernels (~500 LOC)
├── q5_k.rs             # Q5_K kernels (~500 LOC)
├── q6_k.rs             # Q6_K kernels (~600 LOC)
├── q8_0.rs             # Q8_0 kernels (~400 LOC)
└── common.rs           # Shared utilities (~400 LOC)
```

**Benefits:**
- ✅ Easier to find format-specific code
- ✅ Isolated bug fixes per format
- ✅ Smaller PRs for format changes
- ✅ Faster compilation when changing one format

**User Quote from Q4_K Investigation:**
> "why you have a huge quant.rs to hold all quants, intead of quantq4.rs quantq4_k.rs etc... ?"

---

### 2. `src/cpu/ops.rs` → **2,323 LOC**

**Current State:** All CPU operations in one file
- GEMM, GEMV operations
- FFN (feed-forward network)
- Attention mechanisms
- Normalization layers

**Target Structure:**
```
src/cpu/ops/
├── mod.rs              # Public exports (~50 LOC)
├── gemm.rs             # GEMM operations (~600 LOC)
├── gemv.rs             # GEMV operations (~400 LOC)
├── ffn.rs              # FFN operations (~400 LOC)
├── attention.rs        # Attention mechanisms (~500 LOC)
├── norms.rs            # RMS norm, layer norm (~200 LOC)
└── dispatch.rs         # Operation dispatch (~200 LOC)
```

**Benefits:**
- ✅ Optimize specific operations independently
- ✅ Clear operation boundaries
- ✅ Easier to add new operation types

---

### 3. `src/gpu/quant_wrapper.rs` → **1,712 LOC**

**Current State:** All quantization wrapper functions mixed together
- Wrappers for all quant formats
- Similar structure to quant.rs problem

**Target Structure:**
```
src/gpu/quant_wrapper/
├── mod.rs              # Public exports (~50 LOC)
├── q4_0.rs             # Q4_0 wrappers (~200 LOC)
├── q4_1.rs             # Q4_1 wrappers (~200 LOC)
├── q4_k.rs             # Q4_K wrappers (~250 LOC)
├── q5_k.rs             # Q5_K wrappers (~250 LOC)
├── q6_k.rs             # Q6_K wrappers (~300 LOC)
├── q8_0.rs             # Q8_0 wrappers (~200 LOC)
└── common.rs           # Shared utilities (~250 LOC)
```

**Benefits:**
- ✅ Matches quant.rs structure
- ✅ Parallel organization
- ✅ Easier to add format-specific optimizations

---

### 4. `src/gpu/forward.rs` → **1,630 LOC**

**Current State:** GPU forward pass implementation
- Prefill stage
- Decode stage
- Attention computation
- FFN computation

**Target Structure:**
```
src/gpu/forward/
├── mod.rs              # Public exports (~50 LOC)
├── prefill.rs          # Prefill operations (~400 LOC)
├── decode.rs           # Decode operations (~400 LOC)
├── attention.rs        # Attention mechanisms (~400 LOC)
├── ffn.rs              # FFN operations (~300 LOC)
└── common.rs           # Shared utilities (~100 LOC)
```

**Benefits:**
- ✅ Optimize prefill/decode independently
- ✅ Stage-specific performance tuning
- ✅ Clear stage boundaries

---

### 5. `src/gpu/weights.rs` → **1,425 LOC**

**Current State:** GPU weight loading and management
- CPU weight loading
- GPU weight uploading
- Weight transposition
- Memory management

**Target Structure:**
```
src/gpu/weights/
├── mod.rs              # Public exports (~50 LOC)
├── loading.rs          # Weight loading (~400 LOC)
├── upload.rs           # GPU upload (~300 LOC)
├── transpose.rs        # Weight transposition (~300 LOC)
├── memory.rs           # Memory management (~200 LOC)
└── meta.rs             # Weight metadata (~150 LOC)
```

**Benefits:**
- ✅ Separate CPU/GPU weight paths
- ✅ Clear loading pipeline
- ✅ Easier to optimize upload strategies

---

### 6. `src/gpu/ops.rs` → **1,274 LOC**

**Current State:** GPU operation dispatch logic
- Large switch/match statements
- Operation selection
- Fallback logic

**Target Structure:**
```
src/gpu/ops/
├── mod.rs              # Public exports (~50 LOC)
├── gemm_dispatch.rs    # GEMM operation selection (~300 LOC)
├── gemv_dispatch.rs    # GEMV operation selection (~300 LOC)
├── attention_dispatch.rs  # Attention selection (~200 LOC)
├── ffn_dispatch.rs     # FFN selection (~200 LOC)
└── fallback.rs         # Fallback logic (~200 LOC)
```

**Benefits:**
- ✅ Clear dispatch paths
- ✅ Easier to add new operations
- ✅ Better operation-specific optimization

---

### 7. `src/config.rs` → **1,095 LOC**

**Current State:** Configuration and tensor naming
- Multiple naming schemes (GGUF, HuggingFace, MoE)
- Tensor name resolution
- Model configuration

**Target Structure:**
```
src/config/
├── mod.rs              # Public exports (~50 LOC)
├── model.rs            # Model configuration (~300 LOC)
├── tensor_names.rs     # Naming schemes (~400 LOC)
├── gguf.rs             # GGUF-specific config (~200 LOC)
└── huggingface.rs      # HF-specific config (~100 LOC)
```

**Benefits:**
- ✅ Clear config sections
- ✅ Easier to add new naming schemes
- ✅ Better config validation

---

## Implementation Priority

### Phase 1: Split Quantization Files (High Priority)
**Why:** Directly addresses Q4_K investigation findings

1. **Split `src/gpu/kernels/quant.rs`**
   - Create `src/gpu/kernels/quant/` directory
   - Split into `quantq4.rs`, `quantq4_k.rs`, `quantq6.rs`, etc.
   - Update all imports
   - Verify tests pass

2. **Split `src/gpu/quant_wrapper.rs`**
   - Create `src/gpu/quant_wrapper/` directory
   - Match quant.rs structure
   - Update FFI declarations
   - Verify GPU tests pass

**Expected Outcome:**
- ✅ Quant format isolation
- ✅ Easier Q4_K debugging
- ✅ Faster format-specific development

### Phase 2: Split Operation Files (Medium Priority)
**Why:** Improve maintainability and performance optimization

3. **Split `src/cpu/ops.rs`**
   - Create `src/cpu/ops/` directory
   - Split by operation type (GEMM, GEMV, FFN, attention)
   - Update dispatch logic
   - Performance test

4. **Split `src/gpu/ops.rs`**
   - Create `src/gpu/ops/` directory
   - Split by operation type
   - Update dispatch logic
   - Performance test

**Expected Outcome:**
- ✅ Operation-specific optimization
- ✅ Clearer code organization
- ✅ Better performance profiling

### Phase 3: Split Forward Pass Files (Medium Priority)
**Why:** Stage-specific optimization and maintenance

5. **Split `src/gpu/forward.rs`**
   - Create `src/gpu/forward/` directory
   - Split by stage (prefill, decode)
   - Update integration
   - Benchmark both stages

**Expected Outcome:**
- ✅ Stage-specific optimization
- ✅ Easier stage debugging
- ✅ Better performance analysis

### Phase 4: Split Remaining Files (Low Priority)
**Why:** General code organization

6. **Split `src/gpu/weights.rs`**
7. **Split `src/config.rs`**

**Expected Outcome:**
- ✅ Better code organization
- ✅ Easier maintenance

---

## Benefits of Splitting Large Files

### From Q4_K Investigation Experience

The **monolithic `quant.rs` file (3,750 LOC)** directly contributed to debugging difficulties:

❌ **Problems encountered:**
- Couldn't easily identify which code path was being used
- GEMM vs GEMV confusion (two separate kernels)
- Scale extraction bugs hidden among 6 other quant formats
- Hard to verify one format without risking others
- Code review required understanding entire file

✅ **After splitting (expected):**
- Open `q4_k.rs` and see only Q4_K code
- Verify Q4_K changes don't affect Q4_0, Q6_K
- Faster code review (only relevant file)
- Easier to understand Q4_K data flow
- Format-specific testing in isolation

### General Benefits

1. **Easier Navigation** - Find code faster
2. **Reduced Compilation Time** - Smaller files compile faster
3. **Better Code Review** - Smaller, focused PRs
4. **Fewer Merge Conflicts** - Less contention on shared files
5. **Clearer Ownership** - Each file has single responsibility
6. **Easier Testing** - Test modules in isolation
7. **Better IDE Performance** - Smaller files index faster

---

## Naming Convention

Following the user's suggested pattern:

- `quant.rs` → `quant/` with `quantq4.rs`, `quantq4_k.rs`, `quantq6.rs`, etc.
- Pattern: `quant<format>.rs` where format is `q4`, `q4_k`, `q6_k`, etc.
- Consistent for both `quant.rs` and `quant_wrapper.rs`

This pattern:
- ✅ Clear file purpose from name
- ✅ Easy to find format-specific code
- ✅ Consistent naming across codebase
- ✅ Matches user expectations

---

## Notes

- **LOC counted as:** Lines of code (includes comments and blank lines)
- **Date:** 2026-04-15
- **Repository:** rocmforge
- **Tool:** `wc -l` (line count)
- **Excluded:** Tests, target/, .git/, .cargo/, worktrees/, old/
- **Included:** Production source code only

---

## Related Documentation

- [Q4_K Investigation](./investigation/Q4_K_INVESTIGATION.md) - Example of bugs hiding in large files
- [CLAUDE.md](../CLAUDE.md) - Project structure and conventions
