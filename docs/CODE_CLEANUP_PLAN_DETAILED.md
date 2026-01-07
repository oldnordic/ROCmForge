# Code Cleanup Plan - ROCmForge

**Date**: 2025-01-06
**Project**: ROCmForge - AMD GPU LLM Inference Engine
**Total Rust Files**: 104 source files, 41 test files
**Total Public API Items**: 437
**Total Compiler Warnings**: 76 (library) + 3 (CLI) + 2 (build script) = 81

---

## Executive Summary

ROCmForge currently has **81 compiler warnings** that need to be addressed. The cleanup is divided into **5 priority phases** based on severity and impact:

1. **Critical (Phase 1)**: Dead code removal, unused FFI bindings - **HIGH IMPACT**
2. **High (Phase 2)**: Unused variables/imports, naming conventions - **MEDIUM IMPACT**
3. **Medium (Phase 3)**: Code smell fixes, Clippy warnings - **LOW IMPACT**
4. **Low (Phase 4)**: Documentation improvements, file organization - **COSMETIC**
5. **Optional (Phase 5)**: Architectural cleanup, code deduplication - **LONG-TERM**

**Estimated Total Time**: 12-15 hours across all phases
**Quick Wins**: Phases 1-2 can be done in ~4 hours with automated tools

---

## Phase 1: Critical - Dead Code & Unused FFI (Priority: HIGHEST)

**Estimated Time**: 2-3 hours
**Impact**: Reduces binary size, removes confusion, improves security
**Severity**: Critical
**Auto-Fixable**: Yes (with `cargo fix` and manual verification)

### 1.1 Unused FFI Bindings (Security Risk)

**Issue Description**: Unused HIP FFI function bindings create attack surface and code confusion.

**Files Affected**:
- `src/backend/hip_backend.rs:15` - `hipSetDevice`
- `src/backend/hip_backend.rs:19` - `hipMemcpyHtoD`
- `src/backend/hip_backend.rs:20` - `hipMemcpyDtoH`
- `src/backend/hip_backend.rs:41` - `hipGetLastError`

**Fix Required**:
```rust
// BEFORE (lines 15, 19, 20, 41)
extern "C" {
    fn hipSetDevice(deviceId: i32) -> i32;           // REMOVE
    fn hipMemcpyHtoD(dst: *mut c_void, src: *const c_void, count: usize) -> i32; // REMOVE
    fn hipMemcpyDtoH(dst: *mut c_void, src: *const c_void, count: usize) -> i32; // REMOVE
    fn hipGetLastError() -> i32;                     // REMOVE
}

// AFTER
extern "C" {
    // ... keep only used functions
}
```

**Verification**: Ensure `hipMemcpy` (wrapper) is still used instead of direct HtoD/DtoH calls.

---

### 1.2 Dead Kernel Cache Infrastructure (200+ lines)

**Issue Description**: Entire kernel caching system in `attention/kernels.rs` is defined but never used.

**Files Affected**:
- `src/attention/kernels.rs:13-43` - Lines 13-43 (constants, struct, static, function)

**Dead Code**:
```rust
const BLOCK_SIZE: u32 = 256;          // Line 13 - NEVER USED
const WARP_SIZE: u32 = 32;            // Line 14 - NEVER USED

struct KernelCache { /* ... */ }       // Line 18 - NEVER CONSTRUCTED
static GLOBAL_CACHE: Mutex<Option<KernelCache>> = Mutex::new(None); // Line 43 - NEVER USED

fn get_or_init_cache() -> Result<&'static Mutex<Option<KernelCache>>, HipError> { /* ... */ } // Line 46 - NEVER CALLED
```

**Fix Required**: Remove lines 13-66 (approximately) or mark with `#[allow(dead_code)]` if planned for future use.

**Decision Required**: Is this infrastructure planned for Phase 2+ GPU kernel caching?
- If **YES**: Add `#[cfg_attr(not(feature = "kernel_cache"), allow(dead_code))]`
- If **NO**: Delete entirely

**Recommended**: DELETE unless kernel caching is actively being worked on.

---

### 1.3 Unused Weight Mapping Functions (400+ lines)

**Issue Description**: Six complex weight-mapping functions are implemented but never called. These represent significant code debt.

**Files Affected**:
- `src/model/execution_plan.rs:1097` - `try_map_qwen2_attention_weights` (~60 lines)
- `src/model/execution_plan.rs:1157` - `map_llama_attention_weights` (~60 lines)
- `src/model/execution_plan.rs:1496` - `try_map_qwen2_mlp_weights` (~60 lines)
- `src/model/execution_plan.rs:1555` - `map_llama_mlp_weights` (~60 lines)
- `src/model/execution_plan.rs:1806` - `try_map_qwen2_layer_norm_weights` (~50 lines)
- `src/model/execution_plan.rs:1895` - `map_llama_layer_norm_weights` (~50 lines)
- `src/model/execution_plan.rs:2158` - `from_gguf_tensors` (~50 lines in LayerPlan)

**Fix Required**:
1. **If these are for future model support**: Add `#[allow(dead_code)]` with TODO comment
2. **If these were superseded**: Delete entirely
3. **If these should be called**: Wire them up to `ExecutionPlan::from_gguf`

**Recommended**: Keep with `#[allow(dead_code)]` and add TODO:
```rust
/// TODO: Wire up these alternative weight mappings for:
/// - Qwen2 models (use try_map_qwen2_*)
/// - LLaMA models (use map_llama_*)
#[allow(dead_code)]
fn try_map_qwen2_attention_weights(...) { /* ... */ }
```

---

### 1.4 Unused Struct Fields

**Issue Description**: Struct fields that are never read indicate incomplete implementation or dead code.

**Files Affected**:
- `src/loader/onnx_loader.rs:82` - `model_path: String` in `OnnxSession`
- `src/ops/attention_gpu.rs:35-37` - Three kernel fields in `HipAttentionKernels`:
  ```rust
  qk_kernel: Option<HipModule>,
  softmax_kernel: Option<HipModule>,
  v_kernel: Option<HipModule>,
  ```
- `src/bin/rocmforge_cli.rs:102` - `tokens: Vec<u32>` in `GenerateResponse`
- `src/bin/rocmforge_cli.rs:110` - `token: u32` in `TokenStream`

**Fix Required**:
1. **OnnxSession.model_path**: Remove field (placeholder implementation)
2. **HipAttentionKernels.qk_kernel/softmax_kernel/v_kernel**: Remove or implement usage
3. **CLI structs**: Remove unused fields

---

### 1.5 Unused Functions

**Files Affected**:
- `src/tensor/matmul.rs:199` - `transpose_in_place_gpu` (entire function)
- `src/http/server.rs:165` - `token_to_text` method (unused helper)
- `src/attention/cpu.rs:4` - `cpu_matmul_f32` import (unused)

**Fix Required**: Remove or add `#[allow(dead_code)]` with TODO comments.

---

## Phase 2: High Priority - Unused Variables & Imports (Priority: HIGH)

**Estimated Time**: 1-2 hours
**Impact**: Cleaner code, better IDE autocomplete, faster compilation
**Severity**: High (but not critical)
**Auto-Fixable**: 90% with `cargo fix --allow-dirty`

### 2.1 Unused Variables (36 instances)

**Top Offenders**:
1. `src/model/execution_plan.rs` - **16 warnings** (highest count)
2. `src/ops/attention_gpu.rs` - **9 warnings**
3. `src/backend/scratch.rs` - **5 warnings**
4. `src/backend/hip_backend.rs` - **4 warnings**
5. `src/model/kv_cache.rs` - **3 warnings**

**Fix Required**: Prefix with underscore `_` to indicate intentional unused:

```rust
// BEFORE
fn layer_norm(
    &self,
    layer_idx: usize,           // WARNING: unused
    scratch_buffers: &ScratchBufferManager,  // WARNING: unused
    kv_cache: &mut KVCache,     // WARNING: unused
) -> Result<()> { /* ... */ }

// AFTER
fn layer_norm(
    &self,
    _layer_idx: usize,          // FIXED: prefixed with underscore
    _scratch_buffers: &ScratchBufferManager,  // FIXED
    _kv_cache: &mut KVCache,     // FIXED
) -> Result<()> { /* ... */ }
```

**Specific Files to Fix**:
- `src/backend/hip_backend.rs:1436,1441,1442` - 3 unused params
- `src/backend/scratch.rs:36,40,41,119` - 4 unused variables
- `src/model/execution_plan.rs:298,438,439,1158,1280,1283,1284,1556,1896,2030,2164,2165` - 12 variables
- `src/kv_cache/kv_cache.rs:90` - `token` parameter
- `src/loader/gguf.rs:323` - `bits_in_second_byte` variable
- `src/loader/mmap_loader.rs:97` - `file_size` variable
- `src/model/glm_position.rs:137` - `window_center` variable
- `src/model/kv_cache.rs:54` - `layer` loop variable
- `src/ops/attention_gpu.rs:105,107,750,770,771,817,818,932` - 9 variables
- `src/tensor/matmul.rs:200` - `handle` parameter

**Automated Fix**:
```bash
cargo fix --lib --allow-dirty --allow-staged
```

Then manually verify that these are truly unused (not placeholders for future work).

---

### 2.2 Unnecessary `mut` Keywords (6 instances)

**Issue Description**: Variables marked `mut` but never mutated.

**Files Affected**:
- `src/model/execution_plan.rs:434` - `mut kv_cache`
- `src/model/execution_plan.rs:753` - `mut attention_scores`
- `src/model/execution_plan.rs:1295` - `mut padded`
- `src/model/execution_plan.rs:1320` - `mut qkv_weight`
- `src/model/kv_cache.rs:52` - `mut current_seq_len`
- `src/model/simple_transformer.rs:56` - `mut linear`

**Fix Required**: Remove `mut` keyword:

```rust
// BEFORE
let mut kv_cache = Option<&mut KVCache>;

// AFTER
let kv_cache = Option<&mut KVCache>;
```

---

### 2.3 Unused Imports (12 instances)

**Build Script**:
- `build.rs:2` - `PathBuf`, `Path`
- `build.rs:3` - `std::process::Command`

**Library Files**:
- `src/attention/cpu.rs:3` - `mask`
- `src/attention/cpu.rs:4` - `cpu_matmul_f32`
- `src/attention/kernels.rs:6` - `std::ffi::c_void`
- `src/backend/hip_backend.rs:1162` - `HIPBLAS_OP_N`, `HIPBLAS_OP_T`, `sgemm`
- `src/backend/scratch.rs:4` - `HipResult`
- `src/http/server.rs:11` - `HeaderMap`, `HeaderValue`, `header`
- `src/model/execution_plan.rs:601` - `HIPBLAS_OP_N`, `HIPBLAS_OP_T`, `sgemm`
- `src/model/kv_cache.rs:4` - `HipResult`
- `src/model/simple_transformer.rs:4` - `AttentionBackend`
- `src/ops/attention_gpu.rs:23` - `std::ffi::c_void`
- `src/scheduler/scheduler.rs:457` - `std::thread`
- `src/hip_isolation_test.rs:3,4` - `std::ffi::c_void`, `std::ptr`
- `src/lib.rs:36` - `super::*`

**CLI Binary**:
- `src/bin/rocmforge_cli.rs:4` - `RequestBuilderExt`

**Automated Fix**:
```bash
cargo fix --lib --bin rocmforge_cli --allow-dirty
```

---

## Phase 3: Medium Priority - Naming Convention Violations (Priority: MEDIUM)

**Estimated Time**: 30-45 minutes
**Impact**: Code consistency, Rust idioms compliance
**Severity**: Medium
**Auto-Fixable**: 50% (manual review needed for FFI)

### 3.1 Non-CamelCase Types

**Issue 1**: `struct f16` should be `F16`
- **File**: `src/loader/gguf.rs:1329`
- **Current**:
  ```rust
  struct f16(u16);
  ```
- **Fix**:
  ```rust
  struct F16(u16);
  ```
- **Impact**: Must update all references to use `F16` instead of `f16`

**Issue 2**: `struct hipUUID` should be `HipUuid`
- **File**: `src/backend/hip_backend.rs:105`
- **Current**:
  ```rust
  pub struct hipUUID {
  ```
- **Fix**:
  ```rust
  pub struct HipUuid {
  ```
- **Impact**: FFI binding - must match C struct name, add `#[repr(C)]` comment

**Recommended**: Keep `hipUUID` with `#[allow(non_camel_case_types)]` for FFI compatibility:

```rust
/// FFI-compatible UUID struct (must match hipUUID from HIP API)
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct hipUUID {
    // ... fields
}
```

---

### 3.2 Non-UpperCase Constants (HIP FFI)

**Files Affected**:
- `src/backend/hip_backend.rs:48` - `hipMemcpyHostToDevice`
- `src/backend/hip_backend.rs:49` - `hipMemcpyDeviceToHost`
- `src/backend/hip_backend.rs:50` - `hipMemcpyDeviceToDevice`
- `src/backend/hip_backend.rs:51` - `hipSuccess`

**Current Code**:
```rust
const hipMemcpyHostToDevice: i32 = 1;
const hipMemcpyDeviceToHost: i32 = 2;
const hipMemcpyDeviceToDevice: i32 = 3;
const hipSuccess: i32 = 0;
```

**Fix Required**:
```rust
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
const HIP_SUCCESS: i32 = 0;
```

**Impact**: Must update all references in `hip_backend.rs` to use new names.

---

### 3.3 Non-SnakeCase Variables (hipBLAS FFI)

**File**: `src/backend/hip_blas.rs:127-132`

**Current Code**:
```rust
unsafe extern "C" fn cblas_sgemm(
    // ... other params
    A: *const f32,     // WARNING: should be `a`
    B: *const f32,     // WARNING: should be `b`
    C: *mut f32,       // WARNING: should be `c`
) { /* ... */ }
```

**Fix Required**:
```rust
unsafe extern "C" fn cblas_sgemm(
    // ... other params
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) { /* ... */ }
```

**Impact**: FFI function signature - must match C BLAS convention.

**Recommended**: Keep as-is with `#[allow(non_snake_case)]` for FFI compatibility:

```rust
#[allow(non_snake_case)]
unsafe extern "C" fn cblas_sgemm(
    // ...
    A: *const f32,  // Matches BLAS convention
    B: *const f32,
    C: *mut f32,
) { /* ... */ }
```

---

## Phase 4: Low Priority - Code Smell & Clippy Warnings (Priority: LOW)

**Estimated Time**: 1-2 hours
**Impact**: Code quality, maintainability
**Severity**: Low (cosmetic but good practice)

### 4.1 Unnecessary Parentheses

**File**: `src/loader/gguf.rs:352`

**Current**:
```rust
(packed[byte_idx + 1] & ((1 << bits_from_second_byte) - 1))
```

**Fix**:
```rust
packed[byte_idx + 1] & ((1 << bits_from_second_byte) - 1)
```

---

### 4.2 Derivable Impl

**File**: `src/attention/backend.rs:13-17`

**Current**:
```rust
impl Default for AttentionBackend {
    fn default() -> Self {
        AttentionBackend::Cpu
    }
}
```

**Fix**:
```rust
#[derive(Default)]
pub enum AttentionBackend {
    #[default]
    Cpu,
    Gpu,
}
```

---

### 4.3 Needless Range Loops

**File**: `src/attention/compute.rs:54`

**Current**:
```rust
for i in 0..data.len() {
    data[i] = /* ... */;
}
```

**Fix**:
```rust
for item in &mut data {
    *item = /* ... */;
}
```

**File**: `src/attention/softmax.rs:22,28` (2 instances)

Similar pattern - replace with iterators.

---

### 4.4 Manual Implementations of Standard Traits

**Files**: `src/attention/multi_query.rs` (7 instances)

**Current**:
```rust
if self.num_query_heads % self.num_kv_heads != 0 {
    // ...
}
```

**Fix**:
```rust
if !self.num_query_heads.is_multiple_of(self.num_kv_heads) {
    // ...
}
```

**Locations**:
- Line 74: `num_query_heads % num_kv_heads != 0`
- Line 223: Three occurrences of `len() % expected != 0`
- Line 246: `q.len() % expected_per_token != 0`
- Line 257: Two occurrences of `k.len() % expected != 0`

---

### 4.5 Too Many Arguments

**Files**:
- `src/backend/gpu_executor.rs:107` - `layer_norm` has 8 arguments
- `src/backend/gpu_executor.rs:145` - `rope` has 9 arguments

**Fix Required**: Refactor into parameter structs:

```rust
// BEFORE
pub fn layer_norm(
    &self,
    input: &HipBuffer,
    output: &HipBuffer,
    weight: &HipBuffer,
    bias: &HipBuffer,
    epsilon: f32,
    hidden_size: usize,
    seq_len: usize,
) -> Result<()> { /* ... */ }

// AFTER
pub struct LayerNormParams<'a> {
    pub input: &'a HipBuffer,
    pub output: &'a HipBuffer,
    pub weight: &'a HipBuffer,
    pub bias: &'a HipBuffer,
    pub epsilon: f32,
    pub hidden_size: usize,
    pub seq_len: usize,
}

pub fn layer_norm(&self, params: LayerNormParams) -> Result<()> { /* ... */ }
```

---

### 4.6 Unnecessary Casts

**File**: `src/backend/gpu_executor.rs:123-126`

**Current**:
```rust
input.as_ptr() as *mut std::ffi::c_void,  // Already c_void*
output.as_ptr() as *mut std::ffi::c_void,
weight.as_ptr() as *mut std::ffi::c_void,
bias.as_ptr() as *mut std::ffi::c_void,
```

**Fix**:
```rust
input.as_ptr(),
output.as_ptr(),
weight.as_ptr(),
bias.as_ptr(),
```

---

### 4.7 Manual Div Ceil

**File**: `src/backend/gpu_executor.rs:136`

**Current**:
```rust
((hidden_size + 31) / 32 * 32) as u32
```

**Fix**:
```rust
hidden_size.div_ceil(32) * 32
```

---

## Phase 5: Documentation & Organization (Priority: LOWEST)

**Estimated Time**: 1-2 hours
**Impact**: Developer experience, onboarding
**Severity**: Cosmetic (but valuable)

### 5.1 Documentation Warnings (13 unresolved links)

**Issues**:
- 9 instances of unresolved link to `seq_len`
- 6 instances of unresolved link to `hidden_size`
- 1 instance of unresolved link to `intermediate_size`
- 1 unclosed HTML tag `HipBackend`

**Fix Required**: Update doc comments to use proper intra-doc links:

```rust
/// Processes input with shape `[seq_len, hidden_size]`
///
/// # Arguments
/// * `input` - Input tensor of shape [`seq_len`][`hidden_size`]
///
/// [`seq_len`]: Self::seq_len()
/// [`hidden_size`]: ModelConfig::hidden_size
```

---

### 5.2 Test File Organization

**Current State**:
- 41 test files in `tests/` directory (integration tests)
- 14 test files in `src/` (unit tests with `*_tests.rs` naming)

**Issue**: Test files scattered across both locations can be confusing.

**Recommendation**:
1. Keep `tests/` for integration tests and smoke tests
2. Move unit tests into modules they test (e.g., `src/attention/mod.rs` contains tests)
3. Consider consolidating `*_tests.rs` files into their parent modules

**Test Files in src/**:
```
src/attention/flash_attention_tests.rs
src/attention/rope_gpu_tests.rs
src/attention/softmax_explicit_tests.rs
src/attention/qkt_matmul_tests.rs
src/attention/weighted_matmul_tests.rs
src/attention/flash_nocausal_tests.rs
src/attention/causal_mask_tests.rs
src/attention/flash_causal_tests.rs
src/attention/kernel_tests.rs
src/mlp/swiglu_tests.rs
src/mlp/rms_norm_tests.rs
src/mlp/gpu_path_regression_tests.rs
src/loader/mxfp_tests.rs
src/hip_backend_debug_tests.rs
```

**Action**: These are fine as-is - they're focused unit tests. Consider adding `#[cfg(test)]` module organization comments.

---

### 5.3 Orphaned/Temporary Files

**Files to Review**:
- `src/hip_isolation_test.rs` - Temporary debug test?
- `src/hip_backend_debug_tests.rs` - Debugging code
- `tests/debug_test.rs` - Temporary?
- `tests/test_*.rs` (multiple) - Naming inconsistent

**Recommendation**: Consolidate or remove temporary debug files.

---

### 5.4 TODO/FIXME Comments (6 instances)

**Locations**:
1. `src/loader/gguf.rs` - "TODO: Implement dequantization for these types"
2. `src/attention/multi_query.rs:74` - "TODO: Implement full GPU pipeline for MQA"
3. `src/model/execution_plan.rs` - "TODO: Replace with GPU attention kernel"
4. `src/model/glm_position.rs:137` - "TODO: Implement full GPU position embedding application"
5. `src/ops/attention_gpu.rs` - "TODO: Implement GPU causal mask kernel (Phase 2+)"
6. `src/mlp/gpu_path_regression_tests.rs` - "TODO: Add actual mlp_swiglu call once the API is exposed"

**Action**: Track these in project management system (GitHub Issues, TODO.md, etc.)

---

## Phase 6: Optional - Architectural Cleanup (Priority: OPTIONAL)

**Estimated Time**: 4-6 hours
**Impact**: Long-term maintainability
**Severity**: Architectural debt (not urgent)

### 6.1 Code Duplication

**Potential Duplication Areas**:
- Weight mapping functions (6 similar functions in execution_plan.rs)
- Error handling patterns across modules
- Tensor shape validation logic

**Action**: Consider refactoring after Phase 1-5 complete.

---

### 6.2 Large Files

**Files Over 1000 Lines**:
1. `src/model/execution_plan.rs` - **2276 lines** (LARGEST)
2. `src/backend/hip_backend.rs` - **1955 lines**
3. `src/loader/gguf.rs` - **1507 lines**
4. `src/ops/attention_gpu.rs` - **1095 lines**

**Recommendation**: Consider splitting `execution_plan.rs` into modules:
```
src/model/execution_plan/
  ├── mod.rs
  ├── weight_mapping.rs
  ├── layer_plans.rs
  ├── forward_pass.rs
  └── mlp_ops.rs
```

---

### 6.3 Public API Surface

**Current**: 437 public items (functions, structs, enums, traits, constants)

**Action**: Audit public API for:
- Internal helpers that should be private
- Missing `pub(crate)` for items used across modules but not externally
- Inconsistent visibility patterns

---

## Prioritized Cleanup Roadmap

### Week 1: Quick Wins (4 hours)

**Day 1**: Phase 1 - Dead Code (2 hours)
1. Remove unused FFI bindings (30 min)
2. Remove dead kernel cache (30 min)
3. Add `#[allow(dead_code)]` to weight mapping functions (1 hour)

**Day 2**: Phase 2 - Unused Variables/Imports (2 hours)
1. Run `cargo fix` automated fixes (30 min)
2. Manually verify fixes (1 hour)
3. Commit changes (30 min)

---

### Week 2: Code Quality (2-3 hours)

**Day 1**: Phase 3 - Naming Conventions (1 hour)
1. Fix `f16` -> `F16` (15 min)
2. Fix HIP constants naming (15 min)
3. Add `#[allow(...)]` for FFI compatibility (30 min)

**Day 2**: Phase 4 - Clippy Warnings (1-2 hours)
1. Fix needless range loops (30 min)
2. Fix manual implementations of standard traits (30 min)
3. Fix unnecessary casts/mut (30 min)
4. Consider refactoring too-many-arguments (30 min - optional)

---

### Week 3: Polish (1-2 hours)

**Day 1**: Phase 5 - Documentation (1 hour)
1. Fix unresolved doc links (30 min)
2. Review test file organization (30 min)

**Day 2**: Phase 6 - Architectural (Optional)
1. Split large files (2-4 hours - optional, can defer)

---

## Metrics & Success Criteria

### Before Cleanup
- Total warnings: 81
- Dead code warnings: 12
- Unused imports: 12
- Unused variables: 36
- Naming violations: 5
- Clippy warnings: ~20

### After Cleanup (Target)
- Total warnings: **< 10** (only FFI `allow` attributes)
- Dead code warnings: **0** (or properly marked with `#[allow(dead_code)]`)
- Unused imports: **0**
- Unused variables: **0** (prefixed with `_`)
- Naming violations: **0** (or FFI with `#[allow(...)]`)
- Clippy warnings: **< 5** (only intentional deviations)

### Quality Gates
- [ ] `cargo build --workspace` produces 0 warnings (excluding `#[allow(...)]`)
- [ ] `cargo clippy --workspace` produces 0 warnings (excluding allowed lints)
- [ ] `cargo doc --workspace` produces 0 warnings
- [ ] All tests pass: `cargo test --workspace`
- [ ] No dead code in production paths (FFI, hot loops)

---

## Automated Cleanup Script

```bash
#!/bin/bash
# cleanup_rocmforge.sh - Automated cleanup for ROCmForge

set -e

echo "=== ROCmForge Code Cleanup ==="
echo "Phase 1: Auto-fix unused imports and variables"
cargo fix --lib --allow-dirty --allow-staged
cargo fix --bin rocmforge_cli --allow-dirty --allow-staged

echo "Phase 2: Run clippy and auto-fix"
cargo clippy --fix --allow-dirty --allow-staged

echo "Phase 3: Format code"
cargo fmt

echo "Phase 4: Build and check for remaining warnings"
cargo build --workspace 2>&1 | grep "warning:" > /tmp/remaining_warnings.txt

echo "Remaining warnings:"
cat /tmp/remaining_warnings.txt | wc -l

echo "Done! Review changes before committing."
```

**Usage**:
```bash
chmod +x cleanup_rocmforge.sh
./cleanup_rocmforge.sh
git diff  # Review changes
cargo test  # Verify tests still pass
```

---

## Risk Mitigation

### Low Risk Changes (Auto-Apply)
- Unused imports removal
- Unused variable prefixing (`_var`)
- Unnecessary `mut` removal
- Unnecessary parentheses removal

### Medium Risk Changes (Manual Review Required)
- Dead code removal (verify not used in tests)
- Naming convention changes (check FFI compatibility)
- Refactoring parameter structs

### High Risk Changes (Extensive Testing Required)
- Splitting large files
- Public API visibility changes
- Weight mapping function removal

### Testing Strategy
1. Run `cargo test --workspace` after each phase
2. Run integration tests: `cargo test --test '*'`
3. Run specific smoke tests:
   ```bash
   cargo test --test inference_smoke_tests
   cargo test --test execution_plan_forward_pass_tests
   cargo test --test mlp_validation_tests
   ```

---

## Maintenance Strategy

### Preventing Warning Accumulation

**CI Integration**:
```yaml
# .github/workflows/ci.yml
- name: Check for warnings
  run: |
    cargo build --workspace --all-targets 2>&1 | tee /tmp/build.log
    if grep -q "warning:" /tmp/build.log; then
      echo "Warnings detected - failing CI"
      exit 1
    fi

- name: Run clippy
  run: |
    cargo clippy --workspace --all-targets -- -D warnings
```

**Pre-commit Hook**:
```bash
#!/bin/bash
# .git/hooks/pre-commit
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --quiet
```

### Regular Cleanup Schedule
- **Weekly**: Run `cargo fix` and `cargo clippy --fix`
- **Monthly**: Review dead code warnings
- **Quarterly**: Audit public API surface

---

## Appendix: File-by-File Warning Breakdown

### Top 10 Files by Warning Count

| File | Warnings | Primary Issues |
|------|----------|----------------|
| `src/model/execution_plan.rs` | 16 | Unused variables, dead code |
| `src/ops/attention_gpu.rs` | 9 | Unused variables, unused imports |
| `src/backend/scratch.rs` | 5 | Unused variables, unused imports |
| `src/backend/hip_backend.rs` | 4 | Dead code (FFI), unused imports |
| `src/model/kv_cache.rs` | 3 | Unused variables, unused imports |
| `src/attention/cpu.rs` | 2 | Unused imports |
| `build.rs` | 2 | Unused imports |
| `src/kv_cache/kv_cache.rs` | 1 | Unused variable |
| `src/loader/gguf.rs` | 1 | Naming convention |
| `src/tensor/matmul.rs` | 1 | Dead code |

### Files with Zero Warnings (Clean)

These files have **0 warnings** and serve as examples of clean code:
- `src/attention/backend.rs` (1 clippy suggestion - derivable impl)
- `src/attention/mask.rs`
- `src/attention/compare.rs`
- `src/attention/rope.rs`
- Most test files in `tests/` directory

---

## Conclusion

This cleanup plan provides a **systematic, prioritized approach** to eliminating all 81 compiler warnings in ROCmForge. The **quick wins (Phases 1-2)** can be completed in **~4 hours** with mostly automated fixes, while the **full cleanup (Phases 1-5)** takes **~10 hours**.

**Recommended Starting Point**:
```bash
# Quick audit - see what needs fixing
cargo build --workspace 2>&1 | grep "warning:" | sort | uniq -c

# Automated fixes (90% of warnings)
cargo fix --lib --allow-dirty
cargo clippy --fix --allow-dirty

# Manual fixes (remaining 10%)
# See Phase 1-5 for detailed instructions
```

**Next Steps**:
1. Run automated fixes (30 min)
2. Review and commit (30 min)
3. Tackle Phase 1 dead code (2 hours)
4. Complete remaining phases incrementally

---

**Document Version**: 1.0
**Last Updated**: 2025-01-06
**Maintained By**: Development Team
**Review Frequency**: Monthly
