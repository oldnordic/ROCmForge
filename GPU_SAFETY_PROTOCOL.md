# GPU Safety Protocol (Critical!)

## CRASH CAUSES (Not Temperature!)

**What actually crashes the desktop:**
- Running tests in parallel (`cargo test` runs multiple tests concurrently)
- VRAM exhaustion (allocating more than available)
- Memory leaks (not releasing GPU resources)
- Race conditions (allocating VRAM that's in use)

**What does NOT crash it:**
- Temperature (GPU stays cool even during gaming)
- Overclocking (we don't do it)
- Power limits

## SAFE Operations

### ✓ Safe (Single-threaded, sequential)
```bash
# Criterion benchmarks (single-threaded by design)
cargo bench --bench gpu_decode --features gpu

# Manual testing (one at a time)
./target/release/rocmforge --gpu --model ...

# Single test with explicit thread limit
cargo test --test gpu_decode_real -- --test-threads=1 --ignored
```

### ✗ DANGEROUS (Parallel execution)
```bash
# ALL tests run in parallel - CRASH RISK!
cargo test

# Multiple GPU tests without thread limit
cargo test --test gpu_*

# Running multiple GPU processes simultaneously
./target/release/rocmforge --gpu ... &  # Background process
./target/release/rocmforge --gpu ...     # Second process = CRASH
```

## VRAM Budget

**Available:** ~20GB VRAM
**Used by model:** ~693MB
**Safe margin:** Keep at least 2-3GB free

**Red flags:**
- VRAM usage > 18GB
- Multiple GPU allocations without release
- Long-running processes without cleanup

## Development Safety Rules

1. **Always use `--test-threads=1` for GPU tests**
2. **Never run `cargo test` without filtering** (too many parallel GPU ops)
3. **Check VRAM usage before benchmarks:** `rocm-smi --showmeminfo vram`
4. **Run one operation at a time** (no background GPU processes)
5. **Clean up between major tests** (restart process if VRAM looks high)

## Recovery

If GPU crashes or acts weird:
```bash
# Check VRAM state
/opt/rocm/bin/rocm-smi --showmeminfo vram

# If needed, kill any stuck GPU processes
pkill -9 rocmforge

# Worst case: reboot
```

---

**Remember:** Temperature monitoring is useless for our crash prevention. The danger is **parallel execution and memory management**, not heat!
