# Coding Conventions

**Analysis Date:** 2026-01-20

## Naming Patterns

**Files:**
- `snake_case.rs` for all Rust source files
- `mod.rs` for module entry points in directories
- Tests co-located with source: `*_tests.rs` or inline `#[cfg(test)]` modules
- Integration tests in `tests/` directory at project root

**Functions:**
- `snake_case` for all functions and methods
- `async fn` prefix for async functions
- Builder pattern: `with_*` for chaining configuration (e.g., `with_repetition_penalty()`)

**Variables:**
- `snake_case` for local variables and parameters
- Descriptive names over abbreviations (e.g., `token_id` not `tid`, `batch_size` not `bs`)
- Single underscore `_` for intentionally unused variables

**Types:**
- `PascalCase` for structs, enums, and type aliases
- `PascalCase` for trait names
- `SCREAMING_SNAKE_CASE` for constants (rare - consts usually use PascalCase or snake_case)
- `Result<T, CustomError>` type aliases for domain-specific results

**Modules:**
- `snake_case` for module directories and files
- Public re-exports at module top for clean API surface

**Generics:**
- Single-letter: `T`, `E`, `K`, `V`
- Descriptive when clarity needed: `Backend`, `Config`

## Code Style

**Formatting:**
- `rustfmt` for code formatting (via `make fmt` or `cargo fmt`)
- No explicit `.rustfmt.toml` - uses Rust defaults
- 100-character line target (soft limit)

**Linting:**
- `clippy` via `make clippy` or `cargo clippy --features rocm -- -D warnings`
- Clippy warnings treated as errors in CI
- Standard clippy lints allowed in `src/lib.rs`:
  ```rust
  #![allow(clippy::too_many_arguments)]
  #![allow(clippy::manual_slice_size_calculation)]
  #![allow(clippy::needless_range_loop)]
  #![allow(clippy::collapsible_else_if)]
  #![allow(clippy::collapsible_if)]
  #![allow(clippy::bool_comparison)]
  #![allow(clippy::let_and_return)]
  #![allow(clippy::clone_on_copy)]
  #![allow(clippy::type_complexity)]
  #![allow(clippy::missing_safety_doc)]
  #![allow(clippy::bool_to_int_with_if)]
  #![allow(clippy::if_same_then_else)]
  #![allow(clippy::redundant_clone)]
  #![allow(clippy::manual_memcpy)]
  ```
- These allowances accommodate FFI bindings, GPU kernel code, and ML patterns

**Braces:**
- Opening braces on same line for structs, enums, functions
- Closing braces on new line

**Spacing:**
- Spaces around operators: `a + b`, `x == y`
- No trailing whitespace
- Blank line between functions and major logical blocks

## Import Organization

**Order:**
1. `std` imports
2. Third-party crate imports
3. `crate::` imports (internal)
4. `use super::*` for child modules

**Example from `src/attention/mod.rs`:**
```rust
use thiserror::Error;
use crate::backend::{DeviceTensor, HipBackend};
```

**Path Aliases:**
- `crate::` prefix for all internal imports
- `use crate::module::item;` pattern (no `super::` outside tests)
- Re-exports at crate root via `src/lib.rs` for public API

**Feature-gated imports:**
```rust
#[cfg(feature = "rocm")]
use crate::backend::DeviceTensor;
```

**Common internal re-exports:**
```rust
pub use error::{ErrorCategory, ForgeResult, RocmForgeError};
pub use kv_cache::KvCache;
pub use sampler::Sampler;
```

## Error Handling

**Centralized Error Type:**
- Single unified error enum: `RocmForgeError` in `src/error.rs`
- All domain-specific errors consolidated into one type
- Uses `thiserror` for error display and `From` implementations

**Error Categories:**
```rust
pub enum ErrorCategory {
    User,       // Invalid input/config - show to user
    Recoverable, // Temporary condition - may retry
    Internal,   // Bug - report to developers
    Backend,    // GPU/HIP failure
    Model,      // File or model issue
}
```

**Error creation macros:**
```rust
user_error!("Invalid temperature value")
user_error!("value: {}", 42)
internal_error!("Unexpected state in tokenizer")
backend_error!("HIP kernel launch failed for matmul")
model_error!("Tensor not found: token_embeddings")
```

**Helper functions:**
```rust
user_err("Value not found")
internal_err("Invariant violated")
backend_err("GPU not available")
```

**Pattern: Return `ForgeResult<T>`**
```rust
pub type ForgeResult<T> = std::result::Result<T, RocmForgeError>;

pub fn sample(&mut self, logits: &[f32]) -> SamplerResult<u32> {
    if logits.is_empty() {
        return Err(SamplerError::EmptyLogits);
    }
    // ...
    Ok(token_id)
}
```

**Context preservation:**
```rust
let result = risky_operation()
    .map_err(|e| context(e, "during model loading"))?;
```

**Allowed unwrap/expect:**
- `unwrap()` used extensively (695 occurrences) - mostly in tests and initialization
- `expect()` with descriptive messages (435 occurrences)
- Pattern: `expect("Failed to...")` for truly invariant conditions

**Domain-specific error types:**
- `SamplerError` in `src/sampler/sampler.rs`
- `AttentionError` in `src/attention/mod.rs`
- `KvCacheError` in `src/kv_cache/page_table.rs`
- These use `thiserror::Error` and convert to `RocmForgeError` at boundaries

## Logging

**Framework:** `tracing` and `tracing-subscriber`

**Initialization:**
```rust
pub use logging::{
    init_logging_default, init_logging_from_env, init_tracing, is_initialized,
    LogLevel, LogFormat, LoggingConfig, LoggingError,
};
```

**Usage patterns:**
```rust
use tracing::{info, warn, error, debug};

debug!("DEBUG: Attention::forward called with backend: {:?}", self.backend);
debug!("DEBUG: Input lengths - q: {}, k: {}, v: {}", q.len(), k.len(), v.len());
```

**Levels:**
- `error!`: User-facing errors, failures
- `warn!`: Recoverable issues, deprecated usage
- `info!`: Normal operation, startup messages
- `debug!`: Detailed operation info
- `trace!`: Very verbose, per-operation details

**Debug print statements:**
- `println!` and `eprintln!` used in some GPU tests for visibility
- Pattern: `eprintln!("WARNING: GPU not available...")` in test fixtures

## Comments

**Module-level documentation:**
```rust
//! MLP (Multi-Layer Perceptron) operations for ROCmForge
//!
//! Implements GPU kernels for MLP components:
//! - SwiGLU activation
//! - RMSNorm normalization
```

**Function documentation:**
- JSDoc/TSDoc-style comments using `///` for public items
- `///` with description for exported functions
- Example from `src/error.rs`:
```rust
/// Categorize the error for handling decisions
///
/// Returns the error category, which can be used to determine
/// whether an error is recoverable, user-facing, or internal.
pub fn category(&self) -> ErrorCategory { ... }
```

**Inline comments:**
- `//` for implementation notes
- Explaining non-obvious logic
- GPU kernel parameters often commented

**TODO/FIXME markers:**
- 21 occurrences across 17 files (as of 2026-01-20)
- Pattern: `// TODO: add error context` in test files
- Pattern: `// TODO: Migrate from to_host_vec() to copy_from_device_safe()`

**When to Comment:**
- Non-trivial algorithm steps
- FFI boundary descriptions
- GPU memory layout explanations
- Workarounds for hardware/driver issues

## Function Design

**Size:**
- Target: Keep functions under 50 lines
- Larger functions acceptable for:
  - GPU kernel launches (many parameters)
  - Complex state machines
  - Table-driven initialization

**Parameters:**
- Many parameters accepted for GPU kernels (allowed via clippy::too_many_arguments)
- Struct parameters for >3 related fields
- Builder pattern for configuration

**Return Values:**
- `Result<T, E>` for fallible operations
- `Option<T>` for truly optional values
- `Vec<T>` or `Box<[T]>` for owned collections
- `&[T]` for slices (preferred over `Vec<T>` reference)

**Self parameter:**
- `&self` for read-only methods
- `&mut self` for state mutation
- `self` for consuming builders
- `Arc<Self>` shared for GPU backend (reference counting)

## Module Design

**Exports:**
- Public items explicitly listed with `pub use`
- Private items by default
- Tests in `#[cfg(test)]` modules within source files

**Barrel Files:**
- `mod.rs` files re-export public API
- `src/lib.rs` re-exports crate-level API
- Example from `src/attention/mod.rs`:
```rust
pub use backend::AttentionBackend;
pub use backend_registry::{
    AttentionBackendError, AttentionBackendRegistry, ...
};
```

**Test co-location:**
```rust
// Phase 1 kernel tests (CPU vs GPU comparison)
#[cfg(test)]
#[cfg(feature = "rocm")]
mod kernel_tests;

// Phase 2 RoPE GPU tests
#[cfg(test)]
#[cfg(feature = "rocm")]
mod rope_gpu_tests;
```

**Feature-gated modules:**
```rust
#[cfg(feature = "rocm")]
pub mod gpu;

#[cfg(feature = "simd")]
pub use cpu::{simd_matmul_f32, ...};
```

**Public vs Internal:**
- `pub` for library API surface
- `pub(crate)` for cross-module internal use
- Private for module-local implementation

## Concurrency Patterns

**Arc Usage:**
- `Arc<HipBackend>` for shared GPU backend across tests
- `std::sync::Arc` for reference-counted shared ownership
- Pattern in test fixtures:
```rust
pub static GPU_FIXTURE: Lazy<Option<GpuTestFixture>> = Lazy::new(|| {
    if !HipBackend::gpu_available() {
        return None;
    }
    match GpuTestFixture::new() {
        Ok(fixture) => Some(fixture),
        Err(e) => None,
    }
});
```

**Lock Handling:**
- `PoisonError<T>` converted to `RocmForgeError::LockPoisoned`
- Pattern in `src/error.rs`:
```rust
impl<T> From<std::sync::PoisonError<T>> for RocmForgeError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        RocmForgeError::LockPoisoned(err.to_string())
    }
}
```

**Async:**
- `tokio` for async runtime
- `async fn` for async operations
- `futures` and `async-stream` for utilities

## Unsafe Code

**FFI Boundaries:**
- GPU kernel launches use `unsafe`
- HIP API calls use `unsafe`
- Pattern:
```rust
unsafe {
    let result = flash_attention_gpu_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, mask_ptr,
        scale, batch_size, seq_len, num_heads, head_dim,
    );
    assert_eq!(result, 0, "FlashAttention kernel failed");
}
```

**Raw Pointers:**
- Used for GPU device memory
- Cast via `as_ptr()`, `as_mut_ptr()`
- Pattern: `buffer.as_mut_ptr() as *mut f32`

**Safety Documentation:**
- Not consistently documented (allowed via `clippy::missing_safety_doc`)
- FFI safety documented at module level

## Constants and Magic Numbers

**Named Constants:**
```rust
const TEST_TOLERANCE: f32 = 1e-3;
```

**Inline Values:**
- GPU kernel parameters often inline (e.g., grid sizes, block sizes)
- File format constants inline in parsers

---

*Convention analysis: 2026-01-20*
