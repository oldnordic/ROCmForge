# Plan 10-14 Summary: Write CLI Reference

**Phase:** 10 (Production Hardening)
**Plan:** 10-14
**Status:** Complete
**Completed:** 2026-01-19

---

## Accomplishments

1. **CLI Reference Documentation Created** - Comprehensive `docs/CLI_REFERENCE.md` covering all CLI functionality
2. **Complete Command Documentation** - All 5 commands documented with options, examples, and notes
3. **Environment Variable Reference** - All `ROCMFORGE_*` environment variables documented
4. **Exit Code Documentation** - All exit codes with common scenarios listed
5. **Troubleshooting Section** - 10+ common issues with solutions

---

## Commands Documented

| Command | Description | Options Count |
|---------|-------------|---------------|
| `serve` | Start HTTP server | 3 |
| `generate` | Generate text (HTTP or local) | 8 |
| `status` | Query request status | 1 |
| `cancel` | Cancel running request | 1 |
| `models` | List available GGUF models | 1 |

---

## Documentation Sections

### Global Options
- `--host` - Server URL for HTTP mode
- `--help`, `--version` - Standard help options

### Per-Command Documentation
Each command includes:
- Description and usage syntax
- Options table with defaults
- Multiple usage examples
- Output format examples
- Notes on behavior

### Environment Variables
| Variable | Purpose |
|----------|---------|
| `ROCMFORGE_GGUF` | Default model path |
| `ROCMFORGE_TOKENIZER` | Default tokenizer path |
| `ROCMFORGE_MODELS` | Default models directory |
| `ROCMFORGE_MAX_TRACES` | OpenTelemetry trace limit |

### Exit Codes
- `0` - Success
- `1` - General error (connection, file not found)
- `2` - Invalid usage (parameter validation)

### Troubleshooting
10+ issues documented:
- Server connection issues
- Model loading problems
- Parameter validation errors
- Local mode GPU issues
- Memory issues

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `docs/CLI_REFERENCE.md` | ~360 | Complete CLI reference |

---

## Validation Against Source

Documentation verified against:
- `/home/feanor/Projects/ROCmForge/src/bin/rocmforge_cli.rs` (603 LOC)
- `/home/feanor/Projects/ROCmForge/src/http/server.rs` (environment variable usage)
- `/home/feanor/Projects/ROCmForge/src/models.rs` (model discovery)

### Parameter Validation Ranges (from CLI source)

| Parameter | Min | Max | Default |
|-----------|-----|-----|---------|
| `max_tokens` | 1 | 8192 | 128 |
| `temperature` | 0.0 | - | 1.0 |
| `top_k` | 1 | - | 50 |
| `top_p` | 0.0 (exclusive) | 1.0 (inclusive) | 0.9 |

---

## Acceptance Criteria Status

- [x] All commands documented (5/5)
- [x] All flags listed with descriptions (13 flags + 3 global options)
- [x] Examples for each command (25+ examples total)
- [x] Exit codes documented (0, 1, 2 with scenarios)
- [x] Matches actual CLI behavior (verified via source code)

---

## Related Documentation

- `docs/USER_GUIDE.md` - Installation and usage (task 10-13)
- `docs/API_DOCUMENTATION.md` - HTTP API reference (task 10-15)
- `docs/DEPLOYMENT.md` - Production deployment (task 10-16)

---

## Next Steps

- Task 10-15: Write API documentation
- Task 10-16: Write deployment guide

---

## Commit Information

```
git add docs/CLI_REFERENCE.md
git commit -m "docs(10-14): add CLI reference documentation"
```
