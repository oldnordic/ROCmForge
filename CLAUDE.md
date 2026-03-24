<!-- GSD:project-start source:PROJECT.md -->
## Project

**rocmforge**

rocmforge is an AMD-first LLM inference engine for the Qwen2.5 model family. It provides two execution hotpaths (CPU and AMD GPU via HIP) with explicit device selection and no fallback. Users run inference through a CLI interface, choosing CPU or GPU at startup.

Ported from Memoria with learnings applied: metadata-driven configuration, modular architecture, and clean separation between backends.

**Core Value:** **Correct, fast inference on AMD hardware with zero hardcoded assumptions.**

All configuration flows from GGUF metadata. Model dimensions, vocabulary, RoPE style — everything discovered, nothing assumed.

### Constraints

- **LOC limit**: Max 1000 lines per file — forces modularity
- **TDD approach**: Tests written first, implementation follows
- **Metadata-driven**: All config from GGUF, no hardcoded model dimensions
- **Explicit device**: No runtime CPU↔GPU fallback, user chooses at startup
- **HIP only**: No CUDA, AMD GPU compute via ROCm HIP runtime
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

**Language:** Rust (edition 2021)

**Core Dependencies:**
- `memmap2` — GGUF memory-mapped file access
- `half` — f16 support for GPU operations
- `rayon` — CPU parallelism
- `libloading` — Dynamic HIP library loading
- `regex`, `once_cell` — Tokenization

**GPU Runtime:** ROCm HIP (via librocm_smi64, hiprtc)
**Model Format:** GGUF (GGML quantization types)
**Target Models:** Qwen2.5 family

**Build Profile:**
```toml
[profile.release]
opt-level = 3
lto = "thin"
```
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
