# Session Context: ROCm CLI Inference Hang Investigation

**Date**: 2026-01-07
**Status**: WAITING FOR REBOOT - CWSR Disable Fix Applied
**Next Action**: Test CLI after reboot to verify fix works

---

## Quick Summary

We investigated a CLI hang during model inference and discovered it's **NOT a code bug** but a **known ROCm driver/firmware issue** affecting RDNA3 consumer cards (RX 7900 XT/gfx1100).

**Fix Applied**: Added `amdgpu.cwsr_enable=0` to GRUB cmdline
**Action Required**: Reboot and test

---

## The Problem

### Symptoms
- Model loads successfully (all 24 layers KV cache allocated)
- Inference hangs during weight tensor loading (many 3584-byte buffers)
- GPU has plenty of memory (6GB used / 21GB total)
- Process hangs for 180+ seconds without progress

### Hang Point Identified
```
DEBUG: KVCache::new() completed all 24 layers  ✅ SUCCESS
DEBUG: load_from_gguf: Creating execution plan...
allocate_buffer: created buffer with size 3584 bytes (repeated many times)
... HANGS HERE ...
```

---

## Root Cause: ROCm Driver Bug

### Primary Issues
1. **MES Firmware Bug** - Microcode Execution Scheduler causing GPU hangs
   - [ROCm/ROCm#5724](https://github.com/ROCm/ROCm/issues/5724) - MES 0x83 firmware bug
   - [ROCm/ROCm#5590](https://github.com/ROCm/ROCm/issues/5590) - CWSR triggering MES 0x80 hang

2. **Memory Allocation Pathology** - Multiple small `hipMalloc` calls trigger hangs
   - [ROCm/hip#3370](https://github.com/ROCm/hip/issues/3370) - `hipFreeAsync` hangs on RX 7900 XT
   - [ROCm/ROCm#5581](https://github.com/ROCm/ROCm/issues/5581) - ROCm 7.0+ exhibits stalls (6.4.4 works)

### Why This Happens
- Large allocations (KV cache): ✅ Work
- Many small allocations (layer norm weights): ❌ Hang
- This is a **ROCm driver bug**, NOT a ROCmForge code bug

---

## The Fix: Disable CWSR

### What is CWSR?
**CWSR = Compute Wave Store and Resume**
- GPU preemption feature for **compute workloads only**
- Does NOT affect gaming/graphics (separate GFX queues)
- Can be disabled safely for single-user inference

### Fix Applied
**File**: `/etc/default/grub`
```bash
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.cwsr_enable=0 ..."
```

**GRUB Updated**: ✅ Complete
**Reboot Required**: Yes (parameter is boot-time only)

### After Reboot: Verify
```bash
# Check that CWSR is disabled
cat /sys/module/amdgpu/parameters/cwsr_enable
# Should output: 0

# Check kernel cmdline
cat /proc/cmdline | grep cwsr_enable
# Should show: amdgpu.cwsr_enable=0
```

---

## Test After Reboot

### Command to Run
```bash
# Kill any existing processes
pkill -9 rocmforge_cli

# Test CLI inference (should complete in <60 seconds)
RUST_LOG=warn timeout 180 ./target/x86_64-unknown-linux-gnu/release/rocmforge_cli generate \
  --gguf "/home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf" \
  --prompt "Hello" \
  --max-tokens 5
```

### Expected Result
- Model loads: 24 layers
- Inference completes: Generates 5 tokens
- Total time: <60 seconds
- Exit code: 0 (success)

### If It Still Hangs
1. Check `dmesg` for MES timeout errors:
   ```bash
   dmesg | tail -50 | grep -i "mes\|timeout\|hang"
   ```

2. Consider **Option B**: Memory Pooling Architecture (2-3 days dev work)

---

## Code Changes Made (Phase 10)

### Files Modified
1. **`src/backend/hip_backend.rs:1695-1748`**
   - Added `ModelRuntime::load_from_gguf()` static method
   - Eliminates wasteful 32-layer KV cache allocation
   - Direct loading from GGUF without intermediate runtime

2. **`src/engine.rs:146-174`**
   - Updated `load_gguf_model()` to use `load_from_gguf()`
   - Added `tokio::task::spawn_blocking` wrapper

3. **`src/loader/gguf.rs:1109-1153`**
   - Added `infer_intermediate_size_from_tensors()` method
   - Auto-detects intermediate_size from tensor shapes

4. **`src/ops/attention_gpu.rs:653`**
   - Fixed HIP kernel INFINITY macro (replaced with literal float)

### Bugs Fixed (Code Issues)
- ✅ BUG-005: CLI weight shape mismatch (intermediate_size=0)
- ✅ BUG-006: HIP kernel INFINITY macro
- ✅ BUG-007: Redundant KV cache allocation

---

## Documentation Created

1. **`/docs/ROCM_HANG_INVESTIGATION_2026-01-07.md`**
   - Comprehensive investigation report
   - All GitHub issues and research findings
   - Workaround options with citations

2. **`/docs/TODO.md`** - Updated Phase 10 section
   - Root cause documentation
   - Fix options explained

3. **This file** - Session context for continuity

---

## Project Status

### Test Health
- **Unit Tests**: 190/190 passing (100%)
- **Build**: Successful with ROCm feature
- **Warnings**: 15 build warnings (down from 84)

### Phases Complete
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1-9.5 | GPU Kernels, MLP, Quantization, Code Quality | ✅ Complete |
| Phase 10 | CLI Inference Fix | ⚠️ Waiting for reboot |

---

## System Information

**Hardware**: AMD Radeon RX 7900 XT (gfx1100, RDNA3)
**ROCm Version**: 7.1
**Kernel**: Linux 6.12.63-2-cachyos-lts
**Model**: Qwen2.5-0.5B Instruct (GGUF)

---

## If Fix Works After Reboot

1. Update docs: Mark Phase 10 as complete
2. Consider implementing **Option B** (Memory Pooling) for long-term stability
3. Document the workaround in README.md for other RDNA3 users

---

## If Fix Still Fails

**Next Step**: Implement Memory Pooling Architecture
- **File**: `src/backend/memory_pool.rs` (NEW)
- **Design**: Pre-allocate large buffer arena, use offset-based indexing
- **Effort**: 2-3 days
- **Benefit**: Addresses root cause without system configuration changes

---

## References

### GitHub Issues
- [ROCm/ROCm#5724](https://github.com/ROCm/ROCm/issues/5724) - MES firmware bug
- [ROCm/ROCm#5590](https://github.com/ROCm/ROCm/issues/5590) - CWSR hang workaround
- [ROCm/hip#3370](https://github.com/ROCm/hip/issues/3370) - hipFreeAsync hangs
- [ROCm/ROCm#5581](https://github.com/ROCm/ROCm/issues/5581) - ROCm 7.0+ stalls

### Linux Kernel Docs
- [AMDGPU Module Parameters](https://docs.kernel.org/gpu/amdgpu/module-parameters.html) - cwsr_enable documentation

---

## Conversation History

### What We Did
1. User reported CLI inference hang after model loading
2. Initial debugging: Found hang during weight buffer allocation
3. User requested proper online investigation (no more trial-and-error)
4. Launched research agent to investigate ROCm/HIP issues
5. Discovered: Known ROCm driver bug, not code issue
6. Explained CWSR and its effects
7. Applied fix: `amdgpu.cwsr_enable=0` in GRUB
8. Created this context file for post-reboot continuity

### User Feedback Highlights
- "you are not following the rules, did you research online"
- Emphasized: Proper investigation, no dirty fixes, TDD approach
- Approved Option A (CWSR disable) after understanding it doesn't affect gaming

---

**End of Session Context**
