# Subagent Handoff Protocol for ROCmForge

**Status:** ENFORCEABLE - All subagents MUST follow this protocol

**Last Updated:** 2026-04-16

---

## The Golden Rule

```
When a subagent runs out of context, it MUST:
1. Create a completion report
2. Generate structured handoff
3. Leave NO stubs or TODOs
4. Enable seamless continuation
```

---

## Handoff Trigger Conditions

Subagent MUST initiate handoff when:

1. **Context threshold:** >80% of context window used
2. **Task incomplete:** Primary goal not achieved
3. **Complex subtask:** Requires more than simple completion
4. **Documentation needed:** Changelog, comments, or docs required

## When NOT to Handoff

- ✅ Task is 100% complete (tested, documented, committed)
- ✅ Simple atomic operation (rename, small fix)
- ✅ Informational query (no code changes)

---

## Handoff Structure

### 1. COMPLETION REPORT

Every handoff MUST include:

```markdown
# Subagent Completion Report

**Task:** [Original task description]
**Subagent ID:** [Unique identifier]
**Status:** 🟡 PARTIAL COMPLETION (requires continuation)

## What Was Completed

- [x] Subtask 1 (fully complete, tested, documented)
- [x] Subtask 2 (fully complete, tested, documented)
- [x] Subtask 3 (fully complete, tested, documented)

## What Remains

- [ ] Subtask 4 (ready to continue)
- [ ] Subtask 5 (ready to continue)
- [ ] Subtask 6 (blocked by Subtask 4)

## Files Modified

| File | Changes | Status | Notes |
|------|---------|--------|-------|
| `src/cpu/ops.rs` | Split into 6 modules | ✅ Complete | All tests passing |
| `src/cpu/ops/mod.rs` | Created main dispatch | ✅ Complete | Re-exports all submodules |
| `src/cpu/ops/activation.rs` | Extracted silu, softmax | ⚠️ Needs docs | Requires documentation |

## Technical Context

### Architecture Decisions
- **Why 6 modules?** Clear separation of concerns
- **Dispatch pattern:** Uses trait-based dispatch for extensibility
- **Performance:** No measurable overhead from modularization

### Dependencies
- `src/cpu/ops/activation.rs` depends on `src/cpu/simd.rs`
- `src/cpu/ops/gemm.rs` depends on `src/cpu/kernels/` modules
- Update order: activation → gemm → others

### Testing Status
- ✅ All existing tests pass
- ⚠️ New module structure needs integration tests
- ❌ Performance benchmarks not yet updated

## Next Steps for Continuation

1. **Complete documentation:** Add module-level docs to activation.rs
2. **Integration tests:** Test modular dispatch works correctly
3. **Performance validation:** Run benchmarks to ensure no regression
4. **Update CHANGELOG:** Document the refactoring
5. **Commit:** Create atomic commit for the split

## Context Handoff

### Code Context
- **Active work:** Splitting `src/cpu/ops.rs` into modules
- **Pattern:** Following quant.rs modularization approach
- **Files to focus on:** `src/cpu/ops/activation.rs`, integration tests

### Mental Model
- **Goal:** Make cpu/ops.rs maintainable (2323 lines is too large)
- **Approach:** Extract logical groupings into separate files
- **Constraint:** Maintain backward compatibility (no API changes)
- **Success criteria:** All tests pass, no performance regression

### Key Insights
- **RMS norm variants** should stay together (they're related)
- **GEMV dispatch** is complex, extract carefully
- **Flash attention** is standalone, can move easily

## Artifact References

- **Branch:** `refactor/cpu-ops-modularization`
- **Commits:** `abc123` (initial split), `def456` (activation module)
- **Issues:** #234 (cpu/ops refactoring)
- **Related:** `src/cpu/quant.rs` modularization (completed 2026-04-15)

## Known Issues

⚠️ **INCOMPLETE:** The following items need attention:

1. **Missing docs:** `activation.rs` needs module-level documentation
2. **Test coverage:** GEMV dispatch needs integration tests
3. **Performance:** Baseline benchmarks before/after modularization
4. **CHANGELOG:** Entry not yet written

## Continuation Instructions

For the next subagent:

1. **Start from:** `src/cpu/ops/activation.rs` documentation
2. **Use this pattern:** Follow existing doc style in `src/cpu/quant.rs`
3. **Run tests:** `cargo test --test ops::activation`
4. **Update CHANGELOG:** When fully complete

**DO NOT:**
- ❌ Start over from scratch
- ❌ Ignore the architectural decisions made
- ❌ Create stub functions (implement for real)

**DO:**
- ✅ Continue from where this left off
- ✅ Follow the established patterns
- ✅ Complete the remaining tasks
- ✅ Update CHANGELOG when done

---

## Handoff File Format

Save handoff as: `.claude/handoffs/[task-id]-[timestamp].md`

```markdown
# Handoff: [Task Name]

**Original Task:** [Description]
**Subagent:** [Agent ID]
**Timestamp:** [ISO 8601]
**Context Window:** [X% used]

## Continuation Point

**Next subagent should start with:**
- File: `path/to/file.rs`
- Function: `function_name`
- Line: [approximate]

**Why here:** This is the next logical step in the refactoring.

## Progress Report

[Detailed completion report as shown above]

## Ready for Continuation

✅ This handoff contains complete context for next subagent
✅ All work is saved and committed
✅ No stubs or TODOs left behind
✅ Tests pass for completed work
✅ Architecture decisions documented

```

---

## Subagent Instructions

### When Context is Running Low (80%+)

1. **STOP current work**
2. **Create handoff document** (following format above)
3. **Save to `.claude/handoffs/`**
4. **Report to user:** "Context running low. Created handoff: `[filename]`"
5. **WAIT for user confirmation** before exiting

### On Task Completion

1. **Update CHANGELOG** with completed work
2. **Verify all tests pass**
3. **Check for stubs/TODOs** (must be zero)
4. **Create final summary report**
5. **Report to user:** "Task complete. Summary: `[filename]`"

### On Blocked Task

1. **Document the blocker** clearly
2. **Create partial handoff** (with blocker info)
3. **Report to user:** "Task blocked. Handoff: `[filename]`. Blocker: `[description]`"
4. **DO NOT create stubs** - leave work incomplete but documented

---

## User Instructions

### When Subagent Reports Handoff

1. **Review the handoff document**
2. **Check for stubs/TODOs** (reject if present)
3. **Verify architecture decisions** make sense
4. **Approve continuation** with: `Continue with new subagent`
5. **Alternative:** Take manual control if needed

### When Subagent Completes Task

1. **Review completion report**
2. **Run tests to verify**
3. **Check CHANGELOG entry**
4. **Approve merge** if satisfied

---

## Handoff Quality Checklist

Before subagent exits, verify:

- [ ] No stub functions created
- [ ] No TODO comments left behind
- [ ] All completed work is tested
- [ ] Architecture decisions are documented
- [ ] Next steps are clear
- [ ] Files are committed (not just modified)
- [ ] CHANGELOG is updated (if task complete)
- [ ] Handoff file is created

---

## Emergency Continuation

If subagent crashes unexpectedly:

1. **Check `.claude/handoffs/`** for latest handoff
2. **Review git history** for recent commits
3. **Check test status** (`cargo test`)
4. **Resume from handoff point**

---

## Examples

### Example 1: Successful Modularization Handoff

```markdown
# Handoff: cpu/ops.rs Modularization

**Status:** 🟡 PARTIAL (40% complete)

## Completed
- ✅ Analyzed cpu/ops.rs structure (2323 lines)
- ✅ Created module structure: ops/{normalization,rope,activation,attention,gemm,bias}.rs
- ✅ Extracted normalization.rs (rms_norm variants) - tested & documented
- ✅ Extracted rope.rs (positional encoding) - tested & documented

## Remaining
- [ ] Extract activation.rs (silu, softmax) - code moved, needs docs
- [ ] Extract attention.rs (flash_attn) - not started
- [ ] Extract gemm.rs (dispatch logic) - not started
- [ ] Extract bias.rs (add_bias, residual) - not started
- [ ] Update CHANGELOG
- [ ] Integration testing
- [ ] Performance validation

## Continuation Point
**Start with:** `src/cpu/ops/activation.rs` documentation
**Pattern:** Follow doc style in `src/cpu/quant/q4_0.rs`

## Architecture
**Decision:** Flat module structure (no sub-subdirectories)
**Reason:** Simpler imports, clearer dependency graph
**Override:** Use nested if depth >3 modules needed
```

### Example 2: Blocked Task Handoff

```markdown
# Handoff: GPU Kernel Optimization

**Status:** 🟡 BLOCKED (30% complete)

## Completed
- ✅ Profiled kernel performance
- ✅ Identified bottleneck in memory access pattern
- ✅ Designed new memory access pattern

## Blocker
**Issue:** Need AMD-specific intrinsic `__builtin_amdgcn_buffer_load`
**Status:** Unknown if available in HIP
**Impact:** Cannot proceed without confirming intrinsic availability

## What Needs Research
1. Check HIP documentation for buffer load intrinsics
2. Look for AMD ROCm examples using similar patterns
3. Consider alternative: shared memory + manual prefetch

## Next Steps
**When unblocked:** Resume with memory access pattern implementation
**Research needed:** Confirm HIP intrinsic availability
**Fallback:** Use shared memory approach if intrinsic unavailable
```

---

## Implementation Notes

### Subagent Prompt Template

When starting a task that may need handoff:

```markdown
You are working on a potentially long-running task. If you approach 80% context usage:

1. STOP current work
2. Create handoff document using the format in docs/SUBAGENT_HANDOFF.md
3. Save to .claude/handoffs/[task-name]-[timestamp].md
4. Report to user: "Context low. Handoff created."

DO NOT create stubs. DO NOT leave TODOs. Document progress and exit cleanly.

Your goal is to enable seamless continuation, not create technical debt.
```

### Handoff Detection

Subagents should monitor:

```python
# Pseudo-code for context monitoring
if context_usage > 0.80:
    if task_status not in ["COMPLETE", "BLOCKED"]:
        create_handoff()
        report_to_user("Handoff created: .claude/handoffs/latest.md")
        await_user_confirmation()
```

---

## Success Metrics

**Good handoff:**
- ✅ Next subagent can continue in <5 minutes
- ✅ Zero stub functions created
- ✅ All work is committed and tested
- ✅ Architecture is clear and documented

**Bad handoff:**
- ❌ TODO comments everywhere
- ❌ Stub functions that need implementation
- ❌ Unclear what was done vs what remains
- ❌ Next subagent has to start over

---

**Remember:** A handoff is not a failure. It's a successful pause that enables completion.**

**The goal:** Work gets finished properly, not quickly.