# Repository Guidelines

## Project Structure & Module Organization
Core inference code lives in `src/`, split by subsystem (`attention`, `backend`, `http`, `kv_cache`, `sampler`, etc.) so GPU, scheduler, and HTTP work stay isolated. GPU kernels and HIP helpers sit in `kernels/`, model fixtures in `models/`, integration tests in `tests/`, and microbenchmarks in `benches/`. Reference materials, bug hunts, and architecture notes are collected inside `docs/`, while helper automation and cleanup utilities live under `scripts/`.

## Build, Test, and Development Commands
- `make build` / `cargo build --features rocm --release`: Produce the optimized ROCm binary.
- `make test` or `cargo test --features rocm --lib -- --test-threads=1`: Run the serial GPU-safe suite.
- `make test-verbose`: Same tests with full stdout for diagnosing kernel issues.
- `make check`: Lightweight signature check before large edits.
- `make fmt` and `make clippy` (`cargo fmt`, `cargo clippy --features rocm -- -D warnings`): Formatting and lint gates that must pass before review.

## Coding Style & Naming Conventions
This repo targets Rust 2021 with 4-space indentation, snake_case modules, PascalCase types, and SCREAMING_SNAKE_CASE constants. Always format via `cargo fmt` rather than manual alignment. Favor existing `tracing` macros over `println!`, annotate every unsafe block with a brief `// SAFETY:` comment, and keep HIP bindings thin.

## Testing Guidelines
Tests require an AMD GPU on ROCm 5.x+, so keep `--test-threads=1` to avoid device contention. Add unit tests next to the module that owns the logic (`#[cfg(test)]` in `src/...`) and use the `tests/` crate for integration scenarios. Name tests after the behavior they prove (e.g., `test_swiglu_matches_cpu_small`), include doc tests when editing README/docs snippets, and record CPU-vs-GPU parity plus any new benchmark in `benches/` for kernel work.

## Commit & Pull Request Guidelines
History favors descriptive subjects tied to phases or subsystems (`Phase 23: hipDeviceSynchronize fix`, `CLI Bug Fixes`). Mirror that clarity or use concise `feat/fix/chore` prefixes, and list the validation commands (`cargo test --features rocm ...`, `cargo clippy`). Pull requests should supply a summary, reproduction or validation steps, GPU model + driver versions used, and logs or screenshots for CLI/HTTP changes. Keep patches scoped to a single subsystem so reviewers can diff quickly.

## GPU & Configuration Notes
Develop on Linux with ROCm 5.x and confirm `hipconfig` detects the card before testing. Keep large GGUF weights outside the repo (only store trimmed fixtures under `models/`) and document any env vars you introduce in `README.md` or `docs/` so other agents can reproduce results.
