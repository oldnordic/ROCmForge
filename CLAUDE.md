# Development Rules - NON-NEGOTIABLE

**Last Updated:** 2026-01-08
**Status**: MANDATORY for ALL code changes

---

## Rule #1: NEVER GUESS - ALWAYS VERIFY

Before ANY code change:

```bash
# 1. READ the source code first
Read /path/to/file.rs

# 2. CHECK the actual database schema
sqlite3 path/to.db ".schema"

# 3. VERIFY with compiler/tests
cargo test
cargo check
```

**FORBIDDEN**:
- ❌ Assuming what a function does without reading it
- ❌ Guessing database table/column names
- ❌ Writing code based on "how it should work"
- ❌ Using grep to "find" symbols (use `find_symbols` instead)

---

## Rule #2: STORE ARCHITECTURAL DECISIONS

Before ANY code change, document:

### For Bugfixes:
```sql
INSERT INTO architectural_decisions (
    project, decision, reasoning, alternatives, trade_offs,
    session_id, created_at_epoch, created_at
) VALUES (
    'project-name',
    'Fix: [brief description]',
    'Root cause: [PROVEN with evidence]. Fix addresses [specific issue].',
    'Alternative approaches considered...',
    'Risk: [what could break]. Mitigation: [how we prevent regression].',
    'session-id',
    strftime('%s', 'now'),
    datetime('now')
);
```

### For Features/Refactoring:
```sql
INSERT INTO architectural_decisions (
    project, decision, reasoning, alternatives, trade_offs,
    session_id, created_at_epoch, created_at
) VALUES (
    'project-name',
    'Add: [feature name]',
    'Why: [business/technical reason]. Approach: [specific implementation].',
    'Alternative 1: [description] Alternative 2: [description]',
    'Trade-off 1: [cost] Trade-off 2: [complexity]',
    'session-id',
    strftime('%s', 'now'),
    datetime('now')
);
```

---

## Rule #3: TDD - PROVE IT FIRST

1. Write a test that FAILS for the expected reason
2. Show the failure output
3. Fix the code
4. Show the test now passes

```bash
# Step 1: Run test - should FAIL with expected error
cargo test test_name

# Step 2: Fix code

# Step 3: Run test again - should PASS
cargo test test_name
```

---

## Rule #4: USE PROPER TOOLS

| Task | Use This | NEVER Use |
|------|----------|-----------|
| Find symbols | `find_symbols(query="...")` | grep/rg |
| Read code | `Read` tool | cat/head/tail |
| Edit code | `refactor_rename` / `refactor_delete` | manual Edit |
| Understand function | `discover_summary(symbol="...")` | assumptions |
| Check schema | `sqlite3 .db ".schema"` | guessing |
| Check row counts | `sqlite3 .db "SELECT COUNT(*)..."` | assumptions |

---

## Rule #5: CITE YOUR SOURCES

Before making changes, cite EXACTLY what you read:

```
I read /home/feanor/Projects/project/src/file.rs:123-456
The function `do_thing` takes parameters X, Y, Z
I checked .codemcp/operations.db schema
Table `reinjection_briefs` has columns: project, session_id, current_task, ...
Therefore I will change...
```

---

## Rule #6: NO DIRTY FIXES

- ❌ "TODO: fix later"
- ❌ `#[allow(dead_code)]` to silence warnings
- ❌ Commenting out broken code
- ❌ Minimal/half-hearted fixes
- ❌ Mocks, stubs, placeholders in core code

**ONLY**: Complete, tested, documented code.

---

## Rule #7: HONEST STATUS - NO "PRODUCTION-READY" CLAIMS

**CRITICAL**: NEVER claim code is "production-ready" or "ready for production use."

### Why This Matters

"Production-ready" is the most common bullshit claim in LLM-generated code. It leads to:
- Users deploying broken code to real systems
- Loss of trust when claims prove false
- Legal liability for failures
- Wasted time debugging issues that were claimed to not exist

### FORBIDDEN Phrases

- ❌ "production-ready"
- ❌ "ready for production"
- ❌ "production-grade"
- ❌ "enterprise-ready"
- ❌ "battle-tested" (unless you actually have battle data)
- ❌ "100% bug-free"
- ❌ "guaranteed to work"

### ALLOWED Alternatives

Be specific and honest about what's actually working:

| Instead of... | Use... |
|---------------|--------|
| "Production-ready" | "Development / Testing" |
| "Ready for production" | "Ready for testing" |
| "100% complete" | "Phase X complete" |
| "No known issues" | "No critical known issues" (if true) |
| "Guaranteed" | "Designed to..." |

### What You CAN Say

- "145/145 unit tests passing"
- "Code quality grade: B+ (82/100)"
- "GPU kernels: Complete and tested"
- "HTTP server: Functional for testing"
- "CLI: Experimental - may crash"
- "Known issues: [list them honestly]"

### When Describing Components

Use these status levels:
- ✅ **Complete** - Feature is implemented
- ⚠️ **Experimental** - Works but has known issues
- ❌ **Not Implemented** - Feature doesn't exist yet
- 🔄 **In Progress** - Being worked on

### Example Honest Status

```markdown
## Project Status
**Phase 15 Complete**

| Component | Status | Notes |
|-----------|--------|-------|
| GPU Kernels | ✅ Complete | 41/41 tests passing |
| HTTP Server | ✅ Complete | Functional for testing |
| CLI | ⚠️ Experimental | May crash - known issues |
| End-to-End | ❌ Not Tested | Integration incomplete |

**Known Issues**:
- CLI may crash during inference
- End-to-end execution not fully tested
- ~50 compiler warnings remaining
```

### Documentation Requirements

When updating README.md or other docs:
1. List known issues honestly
2. Mark experimental features clearly
3. Never hide bugs behind vague language
4. Include test counts, not claims
5. Separate "what works" from "what's experimental"

**Remember**: Users will trust honest assessments more than false claims of perfection.

---

## Session Start Checklist

When starting work on a project:

1. [ ] Read the project's CLAUDE.md
2. [ ] Check docs/DATABASE_SCHEMA.md for schema
3. [ ] Read relevant source files
4. [ ] Run `cargo check` or equivalent
5. [ ] Store architectural decision before coding
6. [ ] TDD: write failing test first
7. [ ] Implement fix/feature
8. [ ] Prove it passes with full output
9. [ ] Update documentation

---

## Code Quality Standards

- Max 300 LOC per file (600 with justification)
- No `unwrap()` in prod paths
- Proper error handling
- No state artifacts in src/
- Modules map to docs/

---

## When In Doubt

1. Read the source code
2. Check the database schema
3. Run tests
4. Store a decision
5. Ask for clarification

**DO NOT GUESS.**
