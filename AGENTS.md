# Repository Guidelines

## Project Structure & Module Organization
`src/lib.rs` exposes `loader`, `config`, `tokenizer`, `cpu`, `hardware`, `bench`, and feature-gated `gpu`. The two hotpaths are CPU in `src/cpu/` and AMD GPU in `src/gpu/`; HIP kernels live in `hip_kernels/` and `hip_kernels/quant/`. Keep CLI-only logic in `src/main.rs`. Tests live in `tests/`, helpers in `tests/common/`, examples in `examples/`, and benchmarks in `docs/benchmarks/`.

## Build, Test, and Development Commands
`cargo build --release` builds the CPU path and CLI. `cargo build --release --features gpu` enables ROCm/HIP and compiles native HIP kernels. `cargo test` runs the suite. `cargo test --features gpu -- --test-threads=1` is the recommended GPU run because several tests use `serial_test`. `cargo bench --bench kernels` writes Criterion reports to `target/criterion/`. `cargo bench --bench gpu_decode --features gpu -- --noplot` runs the real-model GPU Criterion decode bench. `cargo run --release -- --model path/to/model.gguf --prompt "Hello"` runs the CLI.

## Coding Style & Naming Conventions
Follow Rust 2021 conventions: 4-space indentation, `snake_case` for files and functions, `CamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Prefer focused modules. Run `cargo fmt` and `cargo clippy --all-targets --all-features` before submitting. Comments should explain quantization details or FFI safety boundaries.

## Code Quality Standards for All Agents

**⚠️ ZERO TOLERANCE POLICY: The following code WILL BE IMMEDIATELY REFUSED**

### Prohibited Patterns (will result in immediate rejection):
- **Stub implementations**: functions that return `unimplemented!()`, `todo!()`, dummy values
- **Mock implementations**: simplified/test versions marked as "for now" or "temporary"
- **Placeholder code**: comments like `// TODO: implement later`, `// fixme`, `// for now`, `// placeholder`
- **Code written without tool verification**: ANY code not generated using mandatory tools

### Mandatory Tools for Code Generation:
ALL code must be verified using these tools before being written:
1. **LSP (Rust Analyzer)** - REQUIRED before ANY code change:
   - Use `LSP goToDefinition` before modifying any function/type
   - Use `LSP findReferences` before changing signatures
   - Use `LSP hover` to understand types and documentation
   - Use `LSP documentSymbol` to understand file structure

2. **Magellan** - REQUIRED for symbol navigation:
   - Use `magellan find` to locate symbol definitions
   - Use `magellan refs` to find callers/callees
   - NEVER grep or find files manually

3. **llmgrep** - REQUIRED for code searching:
   - Use `llmgrep search` for all code searching (NEVER `grep` or `find`)
   - Use `--output human` for readable results
   - Use filters: `--kind`, `--path`, `--language`

4. **Mirage** - REQUIRED for CFG analysis:
   - Use `mirage cfg` for control flow questions
   - Use `mirage paths` for execution paths
   - Use `mirage blast-zone` for impact analysis

### Refusal Policy:
If you submit code containing:
- `unimplemented!()`, `todo!()`, or dummy placeholders → **WILL BE REFUSED**
- "for now" comments, "FIXME", "TODO" in production code → **WILL BE REFUSED**
- Code written without LSP/Magellan/llmgrep/Mirage verification → **WILL BE REFUSED**
- Mock implementations or simplified test code marked as temporary → **WILL BE REFUSED**

**No exceptions. No "just this once". No "I'll fix it later".**

**If you cannot verify your approach with tools, DO NOT WRITE CODE. Ask for clarification instead.**

### File-by-File Plan Requirements:
ALL code changes must follow this planning process:
- **Create explicit file-by-file plan** before writing any code
- **Specify exact files** to create or modify
- **List exact functions/types** to change with signatures
- **Include line numbers** when modifying existing code
- **Get plan approved** before starting implementation
- **Update plan as needed** when discovering new information

**Example proper plan:**
```
Files to modify:
1. src/gpu/kernels/quant/q4_0.rs:234-267 - Function: gemv_q4_0_f32
   - Change signature: add stream: &HipStream parameter
   - Update extern "C" declaration
   - Modify kernel launch call

2. src/gpu/ops.rs:892-945 - Function: gpu_gemv_q4_0
   - Add stream parameter passing
   - Update all call sites

Files to create:
1. src/gpu/kernels/quant/q4_0_stream.rs (new file)
   - Purpose: Stream-based Q4_0 operations
   - Functions: gemv_q4_0_f32_on_stream, gemm_q4_0_f32_on_stream
```

**Example improper plan (will be refused):**
```
"Add stream support to Q4_0 kernels"  ← Too vague
"Fix the GPU stuff"  ← No specifics
"Update related files"  ← Which files?
```

### Token Limit Handover Procedure:

**⚠️ CRITICAL: PROACTIVE HANDOVER IS MANDATORY**

- Check context remaining AFTER COMPLETING EACH TASK
- If you've used >80% of your context, **YOU MUST STOP AND INITIATE HANDOVER**
- DO NOT proceed to the next task
- DO NOT wait until you're blocked from responding
- DO NOT attempt "one more quick thing"

**Handover is NOT optional when approaching 80% context usage.**

**Handover message format:**
```
HANDOVER: Context limit approaching

Completed: [task description]
Next task: [specific next step]
Git state: [commit SHA or status]
Notes: [context for next subagent]

Project docs: /home/feanor/Projects/rocmforge/AGENTS.md
Resume from: [specific location]
```

## Testing Guidelines
Keep unit tests near implementation and use `tests/` for coverage. Match existing names such as `quant_unit.rs`, `quant_integration.rs`, and `integration_gpu.rs`. GPU-only tests should use `#![cfg(feature = "gpu")]`; add `#[serial]` when a test touches shared device state or VRAM-sensitive setup. Include benchmark output when changing hot paths.

## Code Graph Workflow
Use `.magellan/rocmforge.db` before editing. Tested here: `magellan status`, `magellan find|refs|query`, `llmgrep search`, and `mirage paths|blast-zone`, all with `--db .magellan/rocmforge.db`. Workflow:
1. confirm the index with `magellan status --db .magellan/rocmforge.db`
2. locate the target symbol with `magellan find --db .magellan/rocmforge.db --name <symbol>`
3. inspect callers and references with `magellan refs --db .magellan/rocmforge.db --name <symbol>` and `magellan query --db .magellan/rocmforge.db --file src/gpu/forward.rs`
4. use `llmgrep search --db .magellan/rocmforge.db "<term>"`
5. run `mirage paths --db .magellan/rocmforge.db <from> <to>` and `mirage blast-zone --db .magellan/rocmforge.db --use-call-graph <file:line>` before changing CPU or GPU hotpaths

Do not use `llmgrep lookup` with this DB; it requires native-v3, while this repository uses SQLite. `mirage` CFG data can be incomplete for some large GPU functions; if `paths` or `blast-zone` fails, fall back to `magellan find|refs|query` plus direct file inspection instead of assuming the graph is wrong.

## ROCm Profiling Workflow
ROCm 7.2 tooling on this machine lives under `/opt/rocm/bin/`. Use `/opt/rocm/bin/rocprofv3` for traces and `/opt/rocm/bin/rocprofv3-avail` to inspect counters and agents. Start with trace mode before attempting PMCs. The baseline decode trace command is:

`/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace --stats --summary --summary-output-file stdout --summary-units usec --group-by-queue --output-directory /tmp/rocprof-decode --output-format csv -- ./target/release/rocmforge --gpu --model /home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf --prompt Hello --no-template --top-p 1.0 --max-tokens 64`

Prefer the repo-native wrapper and configs under `.rocprofv3/`:

- `./.rocprofv3/profile_decode.sh runtime`
- `./.rocprofv3/profile_decode.sh runtime-gate-up`
- `./.rocprofv3/profile_decode.sh runtime-ffn-down`
- `./.rocprofv3/profile_decode.sh system`

The wrapper keeps `--output-config` enabled, so inspect the generated `*_config.json` before assuming a run used the options you intended.

Use `tests/gpu_decode_real.rs::test_gpu_greedy_decode_profile_real_model` for the built-in stage profiler:

`cargo test --release --features gpu --test gpu_decode_real test_gpu_greedy_decode_profile_real_model -- --ignored --nocapture --test-threads=1`

On this ROCm 7.2.0 setup, `rocprofv3` kernel/runtime trace works, and `--output-config` resolves cleanly. PMC collection against the decode path still aborts `rocmforge`, even with `ROCMFORGE_DISABLE_DECODE_GRAPH=1` and `--disable-signal-handlers`. Use timing traces first and treat PMCs as experimental until that interaction is understood.

Criterion and `perf` complement `rocprofv3`:

- Criterion real-model decode bench: `cargo bench --bench gpu_decode --features gpu -- --noplot`
- Host-side `perf` wrapper: `./.perf/perf_decode.sh`

Use Criterion to reduce end-to-end decode noise and compare regressions over time. Use `perf` for host-side counters such as `task-clock`, `page-faults`, `context-switches`, and `cpu-migrations`. On this machine, hardware-event sets can fail with `No supported events found.`, so the wrapper defaults to software counters.

Scope matters:

- `cargo bench --bench gpu_decode --features gpu -- --noplot` loads the model once and measures repeated prompt+decode iterations inside one process.
- `./.perf/perf_decode.sh` measures the full CLI process, so it includes startup and model loading.

Recommended measurement workflow:

1. Build the release binary first: `cargo build --release --features gpu`
2. Establish the steady-state baseline with Criterion: `cargo bench --bench gpu_decode --features gpu -- --noplot`
3. If throughput regresses or a GPU kernel changes, run `./.rocprofv3/profile_decode.sh runtime` before touching launch geometry
4. Use the built-in stage profiler when you need bucketed decode timing by subsystem:
   `cargo test --release --features gpu --test gpu_decode_real test_gpu_greedy_decode_profile_real_model -- --ignored --nocapture --test-threads=1`
5. Run `./.perf/perf_decode.sh` only to check host-side overhead, page faults, and scheduling noise
6. Do not compare raw `perf` throughput numbers against Criterion without noting that `perf` includes startup and model loading
7. Treat PMCs as a last step on this machine because they still abort against the decode path

What each tool answers:

- Criterion: "Did end-to-end graph-backed decode get faster or slower in a loaded process?"
- `rocprofv3`: "Which HIP kernels and launch gaps dominate the GPU timeline?"
- built-in stage profiler: "Which decode stage bucket moved?"
- `perf`: "Did host-side overhead move in the wrong direction?"

## Model Paths
Local test models live in `/home/feanor/Projects/Memoria/models/`. The primary GPU regression/perf model used in this repository is `/home/feanor/Projects/Memoria/models/qwen2.5-0.5b-instruct-q4_0.gguf`. `tests/gpu_decode_real.rs` uses that path directly, so keep it available when running the real-model GPU tests.

## ROCm/HIP References
Local ROCm examples and reference code are in `/home/feanor/Projects/rocm-examples/`. Official ROCm 7.2 references used in this repo:
- ROCm 7.2 release notes: `https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html`
- HIP graphs: `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html`
- HIP Graph API tutorial: `https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/tutorial/graph_api.html`
- HIP performance guidelines: `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html`
- HIP hardware implementation notes: `https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html`
- HIP occupancy API: `https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/occupancy.html`

## Architecture & Safety Notes
The GPU path is AMD-only and uses ROCm/HIP natively. Do not introduce CUDA, NVIDIA-specific code, or cross-vendor abstraction layers unless the project direction changes. When touching `src/gpu/` or HIP kernels, prefer explicit validation and conservative fallbacks to avoid desktop instability, GPU resets, or invalid memory access. The CPU path remains the fallback reference.

## Commit & Pull Request Guidelines
Use the repository’s Conventional Commit pattern, for example `feat(gpu): add Q4_1 methods`, `fix(build): ...`, or `test(q4_0): ...`. Keep scopes specific to the subsystem or quantization type. PRs should describe the affected path, list the commands you ran, and link the relevant issue. Include benchmark deltas when throughput changes.

## Session Continuity Notes (Codex)
Use this block when reopening work and the user asks to continue the last optimization session.

- Last updated: April 10, 2026
- Focus branch/work area:
  - `src/gpu/launch_autotune.rs`
  - `src/gpu/forward.rs`
  - `src/gpu/ops.rs`
- Current state:
  - launch autotune cache persistence is fixed and uses a persisted list schema (`entries: [{key, variant}]`) rather than JSON object keys.
  - expected cache file: `~/.cache/rocmforge/launch_autotune_v1.json`
  - full-decode graph warmup now triggers whenever cache entries are missing and no decode graph is cached (not tied to `pos == 0`).
  - full-decode graph update now only attempts in-place update for existing `FullGreedyDecode` scope; otherwise it instantiates a new full-decode graph.
- Latest measured throughput snapshot (Qwen2.5-7B-Instruct-Q4_0-Pure.gguf, `--max-tokens 64`):
  - graph + q8 fastpath + launch autotune: about `105-106 tok/s`
  - no graph + q8 fastpath + launch autotune: about `104 tok/s`
- Known next step if user says "continue":
  1. profile with `./.rocprofv3/profile_decode.sh runtime`
  2. target top kernels (`QKV`, `gate_up`, residual) with additional launch variants and re-measure
  3. keep decode graph + autotune safety behavior intact while tuning
