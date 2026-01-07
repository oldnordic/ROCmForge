# Code Cleanup Quickstart Guide

**TL;DR**: 81 warnings to fix. Quick wins take ~4 hours. Full cleanup ~10 hours.

---

## 5-Minute Audit

```bash
# See all warnings
cargo build --workspace 2>&1 | grep "warning:"

# Count by type
cargo build --workspace 2>&1 | grep "warning:" | wc -l  # Total: 81
cargo build --workspace 2>&1 | grep "dead_code" | wc -l  # Dead code: 12
cargo build --workspace 2>&1 | grep "unused_imports" | wc -l  # Unused imports: 12
cargo build --workspace 2>&1 | grep "unused variable" | wc -l  # Unused vars: 36
```

---

## 30-Minute Quick Wins

```bash
# 1. Auto-fix unused imports and variables (15 min)
cargo fix --lib --allow-dirty
cargo fix --bin rocmforge_cli --allow-dirty

# 2. Auto-fix clippy warnings (10 min)
cargo clippy --fix --allow-dirty

# 3. Format code (5 min)
cargo fmt

# 4. Verify
cargo build --workspace
cargo test --workspace
```

**Result**: ~50 warnings eliminated automatically.

---

## Remaining Manual Fixes (31 warnings)

### Critical Dead Code (12 items)

**Remove these entirely**:
```rust
// src/attention/kernels.rs
const BLOCK_SIZE: u32 = 256;  // Line 13
const WARP_SIZE: u32 = 32;    // Line 14
struct KernelCache { /* ... */ }  // Line 18
static GLOBAL_CACHE: Mutex<...>  // Line 43
fn get_or_init_cache() { /* ... */ }  // Line 46

// src/backend/hip_backend.rs (unused FFI)
fn hipSetDevice(...)  // Line 15
fn hipMemcpyHtoD(...)  // Line 19
fn hipMemcpyDtoH(...)  // Line 20
fn hipGetLastError()  // Line 41

// src/tensor/matmul.rs
fn transpose_in_place_gpu(...)  // Line 199

// src/http/server.rs
fn token_to_text(...)  // Line 165
```

**Or keep with `#[allow(dead_code)]`** if planned for future use:
```rust
#[allow(dead_code)]
#[cfg(feature = "kernel_cache")]
struct KernelCache { /* ... */ }
```

---

### Naming Violations (5 items)

```rust
// src/loader/gguf.rs:1329
- struct f16(u16);
+ struct F16(u16);

// src/backend/hip_backend.rs
- const hipMemcpyHostToDevice: i32 = 1;
- const hipMemcpyDeviceToHost: i32 = 2;
- const hipMemcpyDeviceToDevice: i32 = 3;
- const hipSuccess: i32 = 0;
+ const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
+ const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
+ const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
+ const HIP_SUCCESS: i32 = 0;

// FFI exceptions - add #[allow(...)]:
#[allow(non_camel_case_types)]
pub struct hipUUID { /* ... */ }

#[allow(non_snake_case)]
fn cblas_sgemm(..., A: *const f32, B: *const f32, C: *mut f32) { /* ... */ }
```

---

### Unused Variables (36 items) - Prefix with `_`

**Top offenders**:
```rust
// src/model/execution_plan.rs (16 instances)
- layer_idx: usize,
+ _layer_idx: usize,

- hidden_size = self.config.hidden_size;
+ _hidden_size = self.config.hidden_size;

// src/ops/attention_gpu.rs (9 instances)
- seq_q, num_heads_q, head_dim_q = ...
+ _seq_q, num_heads_q, head_dim_q = ...

// src/backend/scratch.rs (5 instances)
- head_dim: usize,
+ _head_dim: usize,
```

---

### Unused Struct Fields (4 items)

```rust
// src/ops/attention_gpu.rs:35-37
pub struct HipAttentionKernels {
    // ...
-   qk_kernel: Option<HipModule>,
-   softmax_kernel: Option<HipModule>,
-   v_kernel: Option<HipModule>,
    // Remove or implement usage
}

// src/loader/onnx_loader.rs:82
pub struct OnnxSession {
-   model_path: String,
    // Remove unused field
}

// src/bin/rocmforge_cli.rs:102,110
struct GenerateResponse {
-   tokens: Vec<u32>,
    // Remove unused field
}

struct TokenStream {
-   token: u32,
    // Remove unused field
}
```

---

## File-by-File Priority

### Fix First (High Warning Count)
1. **src/model/execution_plan.rs** (16 warnings)
   - Prefix 12 unused variables with `_`
   - Add `#[allow(dead_code)]` to 6 weight mapping functions

2. **src/ops/attention_gpu.rs** (9 warnings)
   - Prefix 9 unused variables with `_`
   - Remove 3 unused struct fields

3. **src/backend/scratch.rs** (5 warnings)
   - Prefix 4 unused variables with `_`
   - Remove 1 unused import

### Next Priority
4. **src/backend/hip_backend.rs** (4 warnings)
   - Remove 4 unused FFI functions
   - Fix 4 constant names

5. **src/model/kv_cache.rs** (3 warnings)
   - Prefix 2 unused variables with `_`
   - Remove 1 unused import

6. **src/attention/cpu.rs** (2 warnings)
   - Remove 2 unused imports

7. **build.rs** (2 warnings)
   - Remove 2 unused imports

---

## One-Shot Cleanup Script

```bash
#!/bin/bash
set -e

echo "=== ROCmForge Cleanup ==="

# Phase 1: Auto-fix (30 min)
echo "Phase 1: Auto-fixing imports and variables..."
cargo fix --lib --allow-dirty --allow-staged
cargo fix --bin rocmforge_cli --allow-dirty --allow-staged

# Phase 2: Clippy fixes (15 min)
echo "Phase 2: Running clippy auto-fix..."
cargo clippy --fix --allow-dirty --allow-staged

# Phase 3: Format (5 min)
echo "Phase 3: Formatting code..."
cargo fmt

# Phase 4: Verify
echo "Phase 4: Building and testing..."
cargo build --workspace
cargo test --workspace --quiet

# Report remaining warnings
echo "=== Remaining warnings ==="
cargo build --workspace 2>&1 | grep "warning:" | wc -l

echo "Done! Check git diff and commit changes."
```

---

## Verification Checklist

After cleanup, verify:

```bash
# Should output 0 (or only #[allow(...)] items)
cargo build --workspace 2>&1 | grep "warning:" | wc -l

# Should output 0
cargo clippy --workspace 2>&1 | grep "warning:" | wc -l

# All tests pass
cargo test --workspace

# Code formatted
cargo fmt --check
```

---

## Preventing Future Warnings

### CI Check
Add to `.github/workflows/ci.yml`:
```yaml
- name: Check warnings
  run: cargo build --workspace --deny warnings
```

### Pre-commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace --quiet
```

---

## Common Patterns

### Pattern 1: Unused Function Parameter
```rust
// BAD
fn process(data: &[f32], _unused_param: usize) -> Result<f32> {
    // Compiler warns about _unused_param
}

// GOOD
fn process(data: &[f32], _unused_param: usize) -> Result<f32> {
    let _ = _unused_param;  // Explicitly ignore
    // or
}

// BETTER
fn process(data: &[f32]) -> Result<f32> {
    // Remove unused parameter entirely
}
```

### Pattern 2: Dead Code That Should Be Kept
```rust
// For future features or alternative implementations
#[allow(dead_code)]
#[cfg(feature = "experimental")]
fn experimental_feature() {
    // ...
}

// For FFI bindings
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct hipUUID { /* ... */ }
```

### Pattern 3: Unused Import
```rust
// BAD
use std::collections::HashMap;  // Never used

// GOOD
// Remove the line entirely
```

---

## Timeline

| Phase | Time | Warnings Fixed | Automation |
|-------|------|----------------|------------|
| 1. Auto-fix | 30 min | ~50 | 100% |
| 2. Dead code | 2 hours | 12 | Manual |
| 3. Unused vars | 1 hour | 36 | Semi-auto |
| 4. Naming | 30 min | 5 | Manual |
| 5. Clippy | 1 hour | ~20 | Semi-auto |
| **Total** | **5 hours** | **~123** | **70%** |

---

## Resources

- **Detailed Plan**: See `CODE_CLEANUP_PLAN_DETAILED.md`
- **Rust Style Guide**: https://rust-lang.github.io/style-guide/
- **Clippy Lints**: https://rust-lang.github.io/rust-clippy/
- **FFI Best Practices**: https://michael-f-bryan.github.io/rust-ffi-guide/

---

**Version**: 1.0
**Last Updated**: 2025-01-06
