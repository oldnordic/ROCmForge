# Agent Instructions

This file is intentionally small. The canonical workflow is:

`/home/feanor/Projects/AGENTS.md`

Follow that shared standard before making code changes: query with Magellan / llmgrep / Mirage first, edit surgically, then run the standard verification gates and refresh the graph.

Project: `rocmforge`
Scope: the repository root
Default database: `/home/feanor/Projects/rocmforge/.magellan/rocmforge.db`

Local notes:

- Preserve existing dirty worktree changes; assume they belong to the user or another active agent.
- Prefer repo-local `.claude/scripts/quality-gate.sh` when present.
- Use `magellan watch --root ./src --db .magellan/rocmforge.db --scan-initial` if the database is missing or stale.
- Keep `AGENTS.md` / `CLAUDE.md` out of public packages unless the user explicitly asks otherwise.
