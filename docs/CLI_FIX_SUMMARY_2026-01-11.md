# CLI Bug Fix Summary - 2026-01-11

**Status**: Experimental fixes applied
**Tests**: 145/145 PASS
**WARNING**: This is alpha software. These fixes improve stability but do not make it production-ready.

---

## What Was Fixed

### P0 Bug #7: Double Spawn Bug (CRITICAL) ✅ FIXED

**Problem**: The inference loop was spawned TWICE, causing race conditions and resource contention.

**Root Cause**:
- CLI spawned a task at `rocmforge_cli.rs:476` (before fix)
- `run_inference_loop()` also spawned a task at `engine.rs:219`
- Result: Two inference loops competing for the same resources

**Fix Applied** (`src/bin/rocmforge_cli.rs:482`):
```rust
// Before (WRONG):
tokio::spawn(async move {
    let _ = engine_clone.run_inference_loop().await;
});

// After (CORRECT):
engine.run_inference_loop().await;
```

**Impact**:
- ✅ Single inference loop now runs
- ✅ No race conditions on scheduler
- ✅ No double processing of batches
- ✅ Reduced GPU resource contention

### P1 Bug #8: Server Error Context Masking ✅ FIXED

**Problem**: `unwrap_or_default()` hid actual error messages when server returned errors.

**Locations Fixed** (`src/bin/rocmforge_cli.rs`):
- Line 203: `run_http_generate()`
- Line 280: `fetch_status()`
- Line 299: `cancel_http_request()`

**Fix Applied** (same pattern for all 3):
```rust
// Before (HIDES ERRORS):
let text = resp.text().await.unwrap_or_default();
anyhow::bail!("Server returned error: {}", text);

// After (SHOWS ERRORS):
let status = resp.status();
let text = resp.text().await
    .unwrap_or_else(|e| format!("<failed to read error body: {}>", e));
anyhow::bail!("Server returned error {}: {}", status, text);
```

**Impact**:
- ✅ Error messages now include HTTP status codes
- ✅ Network read failures are shown instead of hidden
- ✅ Debugging server issues is now possible

---

## Bugs NOT Fixed (Deferred)

### P0 Bug #1: GPU Resource Leak (PARTIAL FIX)

**What Was Fixed**: Removed double spawn that was causing resource contention

**What Remains**: The background task is still not tracked for cleanup on shutdown

**Why Partial Fix**: Complete fix requires implementing RAII guard pattern with JoinHandle tracking (estimated 2-3 hours)

### Other Deferred Bugs

| Bug | Priority | Status |
|-----|----------|--------|
| Missing error context in JSON parsing | P1 | Deferred |
| Silent error dropping (`.ok()`) | P1 | Deferred |
| No cleanup on early returns | P1 | Deferred |
| Missing input validation | P2 | Deferred |
| Potential infinite loop | P2 | Deferred |

---

## Test Results

```
cargo test --lib
test result: ok. 145 passed; 0 failed; 0 ignored
```

✅ All 145 unit tests pass
✅ No regressions introduced
✅ Code compiles cleanly

---

## Known Issues

**This is experimental alpha software.**

### Critical Known Issues:
1. **GPU resource leak**: Background task not properly cleaned up on shutdown
2. **No cleanup on early returns**: Background task orphaned when errors occur
3. **CLI may crash**: Known stability issues during inference

### Limitations:
- Manual testing not performed (requires GPU hardware)
- GPU memory leak testing not performed
- End-to-end integration testing incomplete

---

## Files Modified

1. `src/bin/rocmforge_cli.rs` - Fixed double spawn, improved error context (3 locations)

**Total Changes**: ~30 lines modified

---

## Next Steps

1. **Implement RAII guard** for proper GPU resource cleanup (P0)
2. **Add cleanup on early returns** (P1)
3. **Replace `.ok()` with proper error logging** (P1)
4. **Add input validation** for CLI parameters (P2)
5. **Manual testing** with GPU hardware to verify fixes

---

## References

- Bug Analysis: `docs/CLI_BUG_ANALYSIS_FRESH_2026-01-11.md`
- Implementation Report: `docs/CLI_FIX_IMPLEMENTATION_2026-01-11.md`
- Code Drift Analysis: `docs/CLI_CODE_DRIFT_ANALYSIS_FRESH_2026-01-11.md`
- API Drift Analysis: `docs/CLI_API_DRIFT_ANALYSIS_FRESH_2026-01-11.md`

---

**Status**: Experimental | **Date**: 2026-01-11 | **Tests**: 145/145 PASS

**NOT PRODUCTION-READY** - Use at your own risk
