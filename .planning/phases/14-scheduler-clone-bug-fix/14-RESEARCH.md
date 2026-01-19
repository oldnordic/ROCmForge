# Phase 14: Scheduler Clone Bug Fix - Research

**Researched:** 2026-01-19
**Domain:** Scheduler state management and continuous batching
**Confidence:** HIGH (based on source code analysis)

## Summary

The scheduler clone bug is a **CRITICAL** issue where `update_iteration_batch` can overwrite scheduler state with stale batch clones, causing generated tokens to be lost during multi-token generation. A **workaround** exists at lines 636-640 in `scheduler.rs` that compares token counts to detect and skip stale clones, but this is a defensive measure rather than a proper architectural fix.

**Primary recommendation:** The workaround is functional and passes tests. Phase 14 should focus on (1) verifying the fix is robust, (2) adding the test `test_update_iteration_batch_cannot_clobber_new_tokens` mentioned in requirements, and (3) considering the HashMap::entry API approach for a cleaner implementation.

## Standard Stack

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| std::collections::HashMap | built-in | Core data structure for request storage | Standard library, no alternatives |
| proptest | 3.x | Property-based testing for scheduler | Already in use for scheduler tests |

**No external dependencies needed** - this is a pure Rust logic fix using standard library APIs.

## Architecture Patterns

### Scheduler State Management

The scheduler maintains three collections for request lifecycle:

```rust
pub struct Scheduler {
    pending_queue: VecDeque<GenerationRequest>,      // Requests waiting to start
    processing_requests: HashMap<u32, GenerationRequest>,  // Active generation
    completed_requests: HashMap<u32, GenerationRequest>,    // Finished requests
    // ...
}
```

**Key invariant:** `processing_requests` HashMap is the **source of truth** for active request state. `IterationBatch` contains clones that can become stale.

### Continuous Batching Flow

```
1. scheduler.get_next_iteration_batch()
   -> Returns IterationBatch with clones of processing_requests

2. engine.process_batch()
   -> Processes each request
   -> Calls scheduler.add_generated_token() directly (updates processing_requests)

3. scheduler.snapshot_request()
   -> Reads updated request from scheduler

4. scheduler.update_iteration_batch()
   -> Receives batch (potentially stale clones)
   -> Must NOT overwrite fresh state in processing_requests
```

### Pattern 1: Token Count Comparison (Current Workaround)

**What:** Compare `generated_tokens.len()` before inserting to detect stale clones

**When to use:** When `update_iteration_batch` receives batch that may contain stale data

**Example:**
```rust
// Source: src/scheduler/scheduler.rs:636-640
if let Some(existing) = self.processing_requests.get(&request.request_id) {
    if existing.generated_tokens.len() > request.generated_tokens.len() {
        // Keep the existing request with more tokens (skip the stale clone)
        continue;
    }
}
```

### Pattern 2: HashMap::entry API (Recommended Cleaner Approach)

**What:** Use entry API to only update if batch has fresher data

**When to use:** Refactoring `update_iteration_batch` for cleaner implementation

**Example:**
```rust
// Recommended: Use Entry API for clarity
use std::collections::hash_map::Entry;

for request in batch.requests {
    if !request.is_complete() && request.state != RequestState::Failed {
        match self.processing_requests.entry(request.request_id) {
            Entry::Vacant(e) => {
                e.insert(request);
            }
            Entry::Occupied(mut e) => {
                // Only update if batch has more tokens (prevents stale overwrite)
                if e.get().generated_tokens.len() <= request.generated_tokens.len() {
                    e.insert(request);
                }
                // If batch has fewer tokens, keep existing (it's fresher)
            }
        }
    }
}
```

### Anti-Patterns to Avoid

- **Unconditional insert:** `self.processing_requests.insert(id, request)` without token count check
- **Assuming batch is fresh:** The batch contains clones from BEFORE token generation
- **Ignoring snapshot_request:** Engine MUST call `snapshot_request()` to get updated state

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Token count comparison | Custom comparison logic with branches | `HashMap::entry` API | Clearer intent, standard pattern |
| State tracking | New state management structures | Existing `processing_requests` HashMap | Already the source of truth |

**Key insight:** The workaround works but is defensive. A better approach would be to restructure the API so stale batches cannot be passed, but that requires engine changes.

## Common Pitfalls

### Pitfall 1: Assuming Batch is Always Fresh

**What goes wrong:** Assuming `batch.requests` always contains the latest state

**Why it happens:** `get_next_iteration_batch()` returns clones made BEFORE processing

**How to avoid:** Always compare token counts or use entry API

**Warning signs:** Tokens disappear between iterations, `generated_tokens.len()` decreases

### Pitfall 2: Breaking Test Coverage

**What goes wrong:** Fix breaks existing scheduler tests

**Why it happens:** Scheduler has complex state machine with many invariants

**How to avoid:** Run ALL scheduler tests before and after fix

**Warning signs:** `test_update_iteration_batch`, `test_tokens_preserved_after_update`, `test_continuous_batching_mixed` fail

### Pitfall 3: Incomplete Fix

**What goes wrong:** Fix only handles one code path, misses another

**Why it happens:** `update_iteration_batch` has multiple update paths

**How to avoid:** Review ALL places where `processing_requests` is modified

**Warning signs:** Tests pass but integration tests fail

## Code Examples

### Current Workaround (Lines 636-640)

```rust
// Source: src/scheduler/scheduler.rs:626-645
// Update remaining processing requests
// Preserve tokens from processing_requests to avoid losing data from stale batch clones
for request in batch.requests {
    if !self.processing_requests.contains_key(&request.request_id) {
        // Request was removed (completed), don't re-insert stale clone
        continue;
    }
    if !request.is_complete() && request.state != RequestState::Failed {
        // Check if we have an existing request with more tokens than the batch
        // This can happen if the batch has a stale clone from before token generation
        if let Some(existing) = self.processing_requests.get(&request.request_id) {
            if existing.generated_tokens.len() > request.generated_tokens.len() {
                // Keep the existing request with more tokens (skip the stale clone)
                continue;
            }
        }
        // Otherwise, insert/overwrite with the batch's version
        self.processing_requests.insert(request.request_id, request);
    }
}
```

### Test That Validates the Fix (Lines 989-1030)

```rust
// Source: src/scheduler/scheduler.rs:989-1030
#[test]
fn test_stale_batch_clone_does_not_overwrite_scheduler() {
    let config = SchedulerConfig::default();
    let mut scheduler = Scheduler::new(config);

    // Submit a request
    scheduler.submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9).unwrap();

    // Get iteration batch (creates clones)
    let batch = scheduler.get_next_iteration_batch().unwrap();
    assert_eq!(batch.requests[0].generated_tokens.len(), 0);

    let req_id = batch.requests[0].request_id;

    // Simulate the engine: add tokens directly to scheduler
    scheduler.add_generated_token(req_id, 100).unwrap();
    scheduler.add_generated_token(req_id, 101).unwrap();

    // Verify scheduler has the tokens
    let req = scheduler.get_request(req_id).unwrap();
    assert_eq!(req.generated_tokens.len(), 2);
    assert_eq!(req.generated_tokens, vec![100, 101]);

    // Simulate the bug: Pass the OLD batch (with stale clones)
    let completed = scheduler.update_iteration_batch(batch).unwrap();
    assert_eq!(completed.len(), 0);

    // Check if tokens were preserved
    let req = scheduler.get_request(req_id).unwrap();
    assert_eq!(req.generated_tokens.len(), 2);
    assert_eq!(req.generated_tokens, vec![100, 101]);
}
```

### Engine Integration (src/engine.rs:698-729)

```rust
// Source: src/engine.rs:698-729
// Process each request in the batch while keeping scheduler state in sync
let original_requests = iteration_batch.requests.clone();
let mut refreshed_requests = Vec::with_capacity(original_requests.len());

for request in &original_requests {
    match self.process_single_request(request).await {
        Ok(_) => {
            if let Some(updated) = self.snapshot_request(request.request_id).await {
                refreshed_requests.push(updated);  // Fresh state from scheduler
            } else {
                refreshed_requests.push(request.clone());  // Fallback to original
            }
        }
        // ...
    }
}

// Update the iteration batch with refreshed requests
let mut updated_batch = iteration_batch;
updated_batch.requests = refreshed_requests;

// CRITICAL: update_iteration_batch receives refreshed batch
let _completed = scheduler.update_iteration_batch(updated_batch)
    .map_err(|e| EngineError::SchedulerError(e.to_string()))?;
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Unconditional insert | Token count comparison | 2026-01-11 (FIX-3) | Tokens preserved in workaround |
| Workaround with `continue` | Entry API (recommended) | Phase 14 | Cleaner implementation |

**Status:** Workaround is IMPLEMENTED and TESTED but marked as UNCOMMITTED in docs.

## Open Questions

1. **Test naming:** Requirements mention `test_update_iteration_batch_cannot_clobber_new_tokens` but existing test is `test_stale_batch_clone_does_not_overwrite_scheduler`. Should this be a new test or a rename?

2. **Entry API vs current workaround:** The current workaround works. Is the Entry API approach worth the refactor, or should we verify the current implementation is sufficient?

3. **Engine integration:** Does the engine always call `snapshot_request()` correctly? If yes, the workaround is defensive. If no, the workaround is critical.

## Sources

### Primary (HIGH confidence)

| Source | Confidence | Notes |
|--------|------------|-------|
| `src/scheduler/scheduler.rs:602-648` | HIGH | Read `update_iteration_batch` implementation |
| `src/scheduler/scheduler.rs:989-1030` | HIGH | Read test for bug scenario |
| `src/scheduler/scheduler.rs:636-640` | HIGH | Verified workaround is in place |
| `src/engine.rs:698-729` | HIGH | Verified engine calls `snapshot_request()` |
| `.planning/codebase/CONCERNS.md:7-12` | HIGH | Bug documented in concerns |
| `docs/FIX_3_SCHEDULER_TOKEN_PRESERVATION_IMPLEMENTATION.md` | HIGH | Detailed fix documentation |
| `docs/CHANGELOG.md:1517-1533` | HIGH | Fix logged as COMPLETE |

### Secondary (MEDIUM confidence)

| Source | Confidence | Notes |
|--------|------------|-------|
| `tests/scheduler_tests.rs` | HIGH | Verified existing test coverage |
| `.planning/research/PITFALLS_v1.2.md:433-483` | HIGH | Pitfall analysis for scheduler bug |

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no external dependencies needed
- Architecture: HIGH - source code fully analyzed
- Pitfalls: HIGH - documented in FIX-3 and PITFALLS_v1.2

**Research date:** 2026-01-19
**Valid until:** 60 days (scheduler logic is stable)

## Appendix: File Locations

| Item | Location |
|------|----------|
| Bug location | `src/scheduler/scheduler.rs:602-648` |
| Workaround | `src/scheduler/scheduler.rs:636-640` |
| Existing test | `src/scheduler/scheduler.rs:989-1030` |
| Engine call site | `src/engine.rs:728` |
| Test suite | `tests/scheduler_tests.rs` |
| Fix documentation | `docs/FIX_3_SCHEDULER_TOKEN_PRESERVATION_IMPLEMENTATION.md` |
| Changelog entry | `docs/CHANGELOG.md:1517-1533` |
