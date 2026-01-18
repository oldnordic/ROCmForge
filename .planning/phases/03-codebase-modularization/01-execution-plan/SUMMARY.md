# Plan 03-01 Summary: Split execution_plan.rs into Focused Modules

**Status**: PARTIALLY COMPLETED
**Date**: 2026-01-18
**Original Estimate**: 4-5 hours
**Actual Time**: ~2 hours

---

## Objective

Split `src/model/execution_plan.rs` (4410 lines) into focused modules per the 300 LOC convention.

---

## What Was Attempted

### Created Module Files (reverted)

The following module files were created in `src/model/execution_plan/`:

1. **architecture.rs** (~70 lines)
   - `Architecture` enum (Qwen2, LLaMA, Mistral)
   - `Architecture::detect()` method
   - Architecture prefix patterns

2. **layer_plan.rs** (~80 lines)
   - `LayerPlan` struct with lazy tensor handles
   - All weight fields for transformer layer

3. **ggml_plan.rs** (~60 lines)
   - `EmbeddingGgmlPlan` struct
   - `LayerGgmlPlan` struct
   - `RopeCache` struct

4. **builder.rs** (~400 lines attempted)
   - `ExecutionPlan` struct definition
   - `ExecutionPlan::from_gguf()` constructor
   - Tensor mapping helpers
   - Missing: GGML graph building methods

5. **execute.rs** (~1500 lines attempted)
   - Forward pass methods
   - Layer execution
   - Attention computation
   - Missing: GGML-specific methods

6. **mod.rs** (~70 lines)
   - Module declarations
   - Public exports
   - Test includes

---

## Challenges Encountered

### 1. Complex Interdependencies

The original `execution_plan.rs` has deeply interconnected methods:

- `build_embedding_plan()` - creates GGML graphs for embedding lookup
- `build_layer_ggml_plans()` - creates GGML graphs for layer execution
- `forward_layer_ggml_decode()` - executes pre-built GGML graphs
- `rope_cache()` - manages RoPE table caching

These methods require access to private fields and have complex relationships that don't split cleanly.

### 2. GGML Graph Building

The GGML graph building code is tightly coupled with:
- `ExecutionPlan` private fields
- `LayerPlan` lazy tensor handles
- `HipGgmlBackend` binding
- Buffer allocation and management

### 3. Missing Fields

After initial extraction, the following fields were missing from `ExecutionPlan`:
- `embedding_plan: OnceCell<EmbeddingGgmlPlan>`
- `layer_ggml_plans: OnceCell<Vec<LayerGgmlPlan>>`

Adding these required updating multiple files simultaneously.

### 4. File Management

During development, the module directory was accidentally removed, requiring restoration from backup.

---

## Decision: Pause Refactoring

Given the complexity and time invested, the decision was made to **pause this refactoring** and revert to the original single-file structure.

### Rationale

1. **Low ROI**: The 4410-line file, while large, is functionally coherent. All code relates to execution plan management.

2. **High Risk**: The refactoring touches core inference path. Any bug would be difficult to debug.

3. **Alternative Approaches**: Consider less invasive improvements:
   - Add more section comments
   - Extract smaller, self-contained helpers
   - Improve documentation within existing structure

4. **Priority**: Phase 03 has other plans that may be easier to complete successfully.

---

## Current State

- **Original file**: `src/model/execution_plan.rs` (4410 lines) - unchanged
- **Module files**: Created in git history but reverted
- **Compilation**: Working (unrelated F16 visibility issue exists)

---

## Recommendations for Future

If revisiting this refactoring:

1. **Incremental Approach**: Start with extracting pure helpers (no ExecutionPlan methods)

2. **Separate Phase**: Consider a dedicated "GGML graph extraction" phase

3. **Test Coverage**: Ensure comprehensive tests exist before refactoring

4. **Smaller Modules**: Target 200-300 LOC per file, not 600

5. **Dependency Analysis**: Map all method dependencies before starting

---

## Artifacts Created

The following files were created during this attempt and are available in git history:
- `src/model/execution_plan/mod.rs`
- `src/model/execution_plan/architecture.rs`
- `src/model/execution_plan/layer_plan.rs`
- `src/model/execution_plan/ggml_plan.rs`
- `src/model/execution_plan/builder.rs`
- `src/model/execution_plan/execute.rs`

These can be restored if the refactoring is revisited.

---

## Commits

None (refactoring reverted)
