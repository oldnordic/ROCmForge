# Phase 02: Test Infrastructure

**Goal**: Restore commented tests and improve test coverage

**Status**: In Planning

**Dependencies**: Phase 1 (Critical Bug Fixes)

## Overview

Phase 2 focuses on restoring test infrastructure that was commented out due to API changes and improving overall test coverage. This phase ensures the codebase has solid testing foundations before proceeding with codebase modularization (Phase 3).

## Plans

| Plan | Title | Status |
|------|-------|--------|
| 02-01 | Rewrite 20+ commented GGUF loader tests for new API | Pending |
| 02-02 | Restore embedding_to_lmhead tests | Pending |
| 02-03 | Add end-to-end inference tests | Pending |
| 02-04 | Replace unwrap() with proper error handling in tests | Pending |

## Key Issues Addressed

1. **Commented GGUF Loader Tests** (02-01)
   - `tests/loader_tests.rs` has 5 commented-out tests using old `GgufModel` API
   - Need to rewrite to use new `GgufLoader` API

2. **Missing embedding_to_lmhead Tests** (02-02)
   - `tests/embedding_to_lmhead_tests.rs` is entirely commented out (436 lines)
   - Tests for token embedding lookup and LM head computation

3. **Missing End-to-End Tests** (02-03)
   - No comprehensive end-to-end inference flow tests
   - Need to test full request → response pipeline

4. **Error Handling in Tests** (02-04)
   - 463+ `unwrap()` calls in test files
   - Need proper `Result<T>` return types and assertions

## Dependencies

```
Phase 1 (Critical Bug Fixes) → Phase 2 (Test Infrastructure) → Phase 3 (Codebase Modularization)
```

All Phase 2 plans can execute in parallel (no inter-dependencies).

## Success Criteria

- [ ] All commented tests rewritten and passing
- [ ] New end-to-end inference tests added
- [ ] Test error handling improved (unwrap() replaced)
- [ ] Overall test coverage increased
- [ ] All tests pass with `cargo test`

---

*Created: 2026-01-18*
*Phase: 02-test-infrastructure*
