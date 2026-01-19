---
phase: 20-code-hygiene-completion
plan: 05
type: execution
completed: 2026-01-19

one_liner: "Fixed unused mut and unused assignment warnings in multi_query.rs and ttft.rs"

subsystem: "code-hygiene"
tags: ["hygiene", "warnings", "rust-compiler"]

tech-stack:
  added: []
  patterns: []

requires:
  - "20-01" # Baseline hygiene established

provides:
  - "Reduced unused_mut warnings from 2 to 0"
  - "Reduced unused_assignment warnings from 1 to 0"

affects:
  - [] # No structural changes affecting future phases

key-files:
  created: []
  modified:
    - path: "src/attention/multi_query.rs"
      change: "Removed mut from 2 variables (scores, output)"
    - path: "src/profiling/ttft.rs"
      change: "Replaced unused max_time assignment with explicit let _"
---

# Phase 20 Plan 05: Simple Hygiene Fixes Summary

## Objective

Fix unused mut, unused assignment, and remaining privacy warnings.

## Tasks Completed

| Task | Name | Commit | Files Modified |
| ---- | ----- | ------ | -------------- |
| 1 | Fix unused mut variables in multi_query.rs | 3bca7e9 | src/attention/multi_query.rs |
| 2 | Fix unused max_time assignment in ttft.rs | 8709a6a | src/profiling/ttft.rs |
| 3 | Verify Q4_0DequantCache privacy warning | 5db42b1 | N/A (already fixed) |
| 4 | Verify all simple hygiene warnings fixed | 9109e3b | N/A (verification) |

## Changes Made

### 1. Removed unused mut keywords (multi_query.rs)

**Line 311:** `let mut scores` -> `let scores`
- Variable is never modified after initialization
- Allocated as immutable DeviceTensor

**Line 341:** `let mut output` -> `let output`
- Variable is never modified after initialization
- Allocated as immutable DeviceTensor

### 2. Fixed unused assignment (ttft.rs)

**Line 178:** `max_time = self.d2h_transfer_ms;` -> `let _ = self.d2h_transfer_ms;`
- The last comparison in `dominant_component()` updated max_time but immediately returned
- Changed to explicit `let _` to show intentional non-use of the value
- Only the component name matters for the final return value

### 3. Privacy warning verification

- Q4_0DequantCache was already made `pub` in plan 20-02 (commit b0fa088)
- No privacy warnings exist in current cargo check output
- This task from plan 20-05 was already addressed in plan 20-02

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| unused_mut warnings | 2 | 0 |
| unused_assignment warnings | 1 | 0 |
| is_more_private warnings | 0 | 0 |
| **Total simple hygiene warnings** | **3** | **0** |

## Verification Results

```bash
# Unused mut - zero results
$ cargo check --features rocm 2>&1 | grep "does not need to be mutable"
# (no output)

# Unused assignments - zero results
$ cargo check --features rocm 2>&1 | grep "value assigned.*is never read"
# (no output)

# Privacy - zero results
$ cargo check --features rocm 2>&1 | grep "is more private"
# (no output)
```

HYGIENE-05 requirement satisfied: All simple hygiene warnings (unused mut, unused assignment, privacy) eliminated.

## Remaining Warnings

54 warnings remain after this plan:
- 16 "field X is never read" warnings (dead_code)
- 21 "method X is never used" warnings (dead_code)
- 17 deprecated method warnings (copy_to_host, to_host_vec)

These are different warning categories not targeted by this plan. Dead code and deprecated API usage will be addressed in future plans.

## Deviations from Plan

None - plan executed exactly as written.

## Duration

- Start: 2026-01-19T22:02:09Z
- End: 2026-01-19T22:04:56Z
- Duration: ~3 minutes
