# Coding Conventions

**Analysis Date:** 2026-01-18

## Naming Patterns

**Files:**
- `snake_case.rs` for all module files
- `kebab-case` for directories
- `*_test.rs` for test files
- `mod.rs` for directory exports

**Functions:**
- `snake_case` for all functions
- No special prefix for async functions
- Descriptive names: `matmul_f32`, `validate_matmul_dims`

**Variables:**
- `snake_case` for variables
- `SCREAMING_SNAKE_CASE` for constants (`GGUF_MAGIC`)
- No underscore prefix for private members (Rust convention)

**Types:**
- PascalCase for structs: `HipBackend`, `KvCache`
- PascalCase for type aliases: `Result<T>`
- PascalCase for enum names, SCREAMING_SNAKE_CASE for values

## Code Style

**Formatting:**
- rustfmt with default settings (no explicit config)
- 4-space indentation (Rust standard)
- Double quotes for strings
- Trailing semicolons required
- Line length: Default rustfmt (100 chars)

**Linting:**
- Rust compiler warnings (`cargo clippy`)
- No explicit linter config detected
- Run: `cargo clippy` for lint suggestions

## Import Organization

**Order:**
1. External crates (`std`, `tokio`, `serde`)
2. Internal modules (`crate::`)
3. Relative imports (`super::`, `use super::*`)
4. Type imports (`use crate::module::Type`)

**Grouping:**
- Blank lines between groups
- Alphabetical within each group (not strictly enforced)

**Path Aliases:**
- `crate::` for absolute paths from lib root
- No external path aliases configured

## Error Handling

**Patterns:**
- `Result<T, Error>` return types
- `anyhow::Error` for application errors
- Custom error types with `thiserror`
- `unwrap()` used heavily in tests (not production)

**Error Types:**
- Custom errors end with `Error`: `HipError`, `KvCacheError`
- `?` operator for propagation
- `panic!` only for truly unrecoverable errors

**Logging:**
- Transitioning from `eprintln!` to `tracing`
- Structured logging: `tracing::info!`, `tracing::error!`

## Logging

**Framework:**
- `tracing` and `tracing-subscriber`
- Levels: error, warn, info, debug, trace

**Patterns:**
- Structured logging: `tracing::info!(context, "message")`
- 42 files still contain `eprintln!` (transition in progress)
- Log at service boundaries, not in utilities

## Comments

**When to Comment:**
- Explain why, not what (Rust is self-documenting)
- Document business logic, algorithms, edge cases
- `///` for public API documentation
- `//!` for module-level docs

**JSDoc/TSDoc:**
- Rustdoc: `///` for functions, `//!` for modules
- `///` with examples for public APIs

**TODO Comments:**
- Format: `// TODO: description` or `// FIXME: description`
- Also: `// BUG:` for known issues
- `// PHASE X:` for tracking development phases

## Function Design

**Size:**
- Keep under 300 LOC (600 with justification)
- Extract helpers for complex logic
- 3 files exceed 3000 LOC (known tech debt)

**Parameters:**
- Use structs for 3+ parameters
- Destructure in parameter list

**Return Values:**
- `Result<T, E>` for fallible operations
- Explicit returns
- Return early for guard clauses

## Module Design

**Exports:**
- Named exports preferred (`pub use`)
- `mod.rs` for directory exports
- `pub(crate)` for internal visibility

**Barrel Files:**
- `mod.rs` re-exports public API
- Keep internal helpers private
- Avoid circular dependencies

---

*Convention analysis: 2026-01-18*
*Update when patterns change*
