# Coding Conventions

**Analysis Date:** 2026-01-20

## Naming Patterns

**Files:**
- `snake_case.rs` for all Rust source files
- `mod.rs` for module definitions in directories
- Test files: `*_tests.rs` or `*_test.rs` for test modules within `src/`
- Integration tests: `<topic>_tests.rs` in `tests/` directory

**Functions:**
- `snake_case` for all functions and methods
- Constructor pattern: `new()`, `with_<variant>()`, `from_<source>()`
- Getter pattern: `<property>()` (no `get_` prefix)
- Setter pattern: `set_<property>()` or `with_<property>(self, ...)` for builder-style
- Predicate pattern: `is_<state>()`, `has_<property>()`, `can_<action>()`

**Variables:**
- `snake_case` for locals and fields
- Capitalized acronyms: `HipBackend`, `MxfpBlock`, `RocmForgeError` (not `HIPBackend`)
- Short names in tight loops: `i`, `j`, `k` for indices; `b`, `h` for batch/heads
- Descriptive names elsewhere: `input_tokens`, `max_sequence_length`

**Types:**
- `PascalCase` for structs, enums, and type aliases
- `PascalCaseError` for error types (e.g., `SamplerError`, `ModelError`)
- `PascalCaseResult<T>` for Result type aliases (e.g., `ForgeResult<T>`)
- Traits: `PascalCase` (no suffix convention observed)

**Constants:**
- `SCREAMING_SNAKE_CASE` for const values
- Test constants: `TestTolerance`, `TEST_TOLERANCE` - both patterns observed

## Code Style

**Formatting:**
- Rust standard formatting via `cargo fmt`
- No explicit `.rustfmt.toml` - uses Rust defaults

**Linting:**
- Clippy with many allowed lints via `#![allow(...)]` in `src/lib.rs`:
  - `too_many_arguments` - FFI and kernel functions need many args
  - `manual_slice_size_calculation` - Common in GPU code
  - `needless_range_loop` - Clearer for GPU operations
  - `collapsible_else_if`, `collapsible_if` - Sometimes clearer for control flow
  - `bool_comparison` - Sometimes clearer for intent
  - `let_and_return` - Sometimes clearer for debugging
  - `clone_on_copy` - Sometimes needed for API clarity
  - `type_complexity` - Complex types common in ML
  - `missing_safety_doc` - FFI documented at module level
  - `bool_to_int_with_if`, `if_same_then_else`, `redundant_clone` - Style choices
  - `manual_memcpy` - GPU memory operations often manual
- Compiler warnings: Some unused imports present (c_void, super::*)

**Line Length:**
- No strict limit enforced
- GPU kernel declarations can be very long (many parameters)

**File Size:**
- Some files exceed 300 LOC guideline:
  - `src/model/execution_plan/execution_plan_src.rs`: 4213 lines
  - `src/backend/hip_backend/backend.rs`: 4039 lines
  - `src/loader/gguf.rs`: 2846 lines
  - `src/kv_cache/kv_cache.rs`: 2094 lines
- Large files typically contain multiple concerns or FFI bindings

## Import Organization

**Order:**
1. Standard library: `use std::...`
2. External crates: `use rand::...`, `use anyhow::...`
3. Internal modules: `use crate::...`
4. Feature-gated imports: `#[cfg(feature = "rocm")] use crate::backend::...`

**Path Aliases:**
- No `use crate::prelude::*` pattern observed
- Re-exports at module level: `pub use ...` in `mod.rs` files
- Example from `src/loader/mod.rs`:
  ```rust
  pub use gguf::{GgufLoader, F16};
  pub use mxfp::{E8M0, MxfpBlock};
  pub use tensor_type::GgufTensorType;
  ```

**Common patterns:**
- `use super::*` in test modules (but warns when unused)
- `use crate::<module>::<Type>` for cross-module references

## Error Handling

**Pattern:** Centralized error type via `thiserror`

**Main error type:** `RocmForgeError` in `src/error.rs`
- Enum-based with domain variants
- `#[error(...)]` attributes for Display messages
- `#[from]` for automatic conversions (e.g., `std::io::Error`)

**Result type alias:** `ForgeResult<T> = Result<T, RocmForgeError>`

**Module-specific errors:**
- `SamplerResult<T>` in `src/sampler/sampler.rs`
- `ModelResult<T>` in `src/model/simple_transformer.rs`
- `SchedulerResult<T>` in `src/scheduler/scheduler.rs`
- `AttentionResult<T>` in `src/attention/mod.rs`
- `EngineResult<T>` in `src/engine.rs`

**Error macros** (defined in `src/error.rs`):
```rust
user_error!("message")           // InvalidRequest variant
internal_error!("message")        // InternalError variant
backend_error!("message")         // HipError variant
model_error!("message")           // ModelLoadFailed variant

// With formatting:
user_error!("Invalid temperature: {}", temp)
```

**Helper functions:**
```rust
user_err("msg")       // -> RocmForgeError
internal_err("msg")   // -> RocmForgeError
backend_err("msg")    // -> RocmForgeError

io_context(io_err, "context")           // -> RocmForgeError::IoError
context(any_err, "context")             // -> RocmForgeError::InternalError
check(result)                           // -> ForgeResult<T>
```

**Error categorization:**
- `RocmForgeError::category()` returns `ErrorCategory`
- Categories: `User`, `Recoverable`, `Internal`, `Backend`, `Model`
- Helper methods: `is_recoverable()`, `is_user_error()`, `is_internal_error()`

**In production paths:**
- Prefer `?` operator for error propagation
- Avoid `unwrap()` and `expect()` in production code (per CLAUDE.md)
- Use `ok_or_else(|| internal_err("context"))?` for Option-to-Result conversion

**In tests:**
- `unwrap()` acceptable for test setup
- `expect()` with clear messages

## Logging

**Framework:** `tracing` crate (not `log`)

**Initialization:**
- `init_logging_default()` - Default setup
- `init_logging_from_env()` - From environment variables
- `init_tracing()` - Manual configuration

**Levels:** `LogLevel` enum: `Trace`, `Debug`, `Info`, `Warn`, `Error`

**Usage patterns:**
```rust
use tracing::{debug, error, info, warn};

debug!("Detailed diagnostic: {}", value);
info!("Normal operation message");
warn!("Something unexpected but recoverable: {}", e);
error!("Operation failed: {}", e);
```

**Conditional debug logging:**
```rust
#[cfg(debug_assertions)]
tracing::debug!("Expensive debug info");
```

**Structured logging:**
```rust
use tracing::info;
info!(request_id = %id, "Processing request");
```

## Comments

**When to Comment:**
- Module-level documentation (all modules have `//!` doc comments)
- Complex algorithms (attention kernels, matmul implementations)
- FFI boundaries (unsafe blocks)
- Performance-critical sections

**JSDoc/TSDoc equivalent:** Rust doc comments (`///` for items, `//!` for modules)

**Documentation patterns:**
- Module header: `//! <Purpose>`
- Function docs: `/// <Summary>` then `/// # Arguments` etc.
- Example from `src/error.rs`:
  ```rust
  /// Unified error type for ROCmForge
  ///
  /// This enum consolidates all domain-specific errors into a single type
  /// that can be used throughout the codebase.
  ```

**TODO comments:**
- Present in codebase (grep count: 27 ignored tests indicate some)
- Use standard Rust TODO/FIXME comments

**Phase markers:**
- Comments like `// Phase 24 FIX:` indicate development phase
- `// Phase X:` comments track implementation progress

## Function Design

**Size:** No strict limit but prefer:
- Public functions: < 50 lines
- Private helpers: < 30 lines
- Large functions are common in:
  - GPU kernel launch code
  - FFI wrapper functions

**Parameters:**
- Many parameters accepted for GPU kernels (via `#![allow(clippy::too_many_arguments)]`)
- Builder pattern for complex initialization (e.g., `RetryConfig`)
- Configuration structs for multi-parameter functions

**Return values:**
- `Result<T, E>` for fallible operations
- `Option<T>` for optional values
- Tuple returns for multiple outputs (e.g., `(free, total)`)
- `ForgeResult<T>` preferred for library code

## Module Design

**Exports:**
- Public re-exports in `mod.rs` files
- `pub use` to flatten module hierarchy
- Example from `src/lib.rs`:
  ```rust
  pub use attention::Attention;
  pub use backend::HipBackend;
  pub use error::{ErrorCategory, ForgeResult, RocmForgeError};
  ```

**Barrel files:**
- `src/lib.rs` acts as main barrel file
- Each module has `mod.rs` for re-exports

**Feature flags:**
- `#[cfg(feature = "rocm")]` for GPU-dependent code
- `#[cfg(test)]` for test modules
- `#[cfg(feature = "simd")]` for SIMD CPU backend

**Test modules:**
- Co-located: `mod tests { #[cfg(test)] ... }` within source
- Separate files: `*_tests.rs` in same directory as module
- Integration tests: `tests/*.rs`

**Unsafe code:**
- FFI bindings to HIP/ROCm
- GPU kernel launches
- Raw pointer operations
- SAFETY comments required for unsafe blocks

---

*Convention analysis: 2026-01-20*
