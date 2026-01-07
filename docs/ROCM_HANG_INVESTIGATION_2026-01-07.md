# ROCm HIP Hang Investigation Report

**Date**: 2026-01-07
**Hardware**: AMD Radeon RX 7900 XT (gfx1100, RDNA3)
**ROCm Version**: 7.1
**Issue**: GPU hangs during model weight tensor allocation

## Executive Summary

The inference hang is **NOT a code bug** but a **known ROCm driver/firmware issue** affecting RDNA3 consumer cards. The root cause is:

1. **MES Firmware Bug** (Microcode Execution Scheduler) - causes GPU hangs on memory operations
2. **CWSR (Compute Wave Store and Resume)** - triggers firmware 0x80 hang
3. **Multiple Small Allocations** - pathology with many `hipMalloc` calls for small buffers

**Most Effective Workaround**: Disable CWSR with kernel parameter `amdgpu.cwsr_enable=0`

---

## Root Cause Analysis

### 1. MES Firmware Bugs (Primary Cause)

**GitHub Issues**:
- [ROCm/ROCm#5724](https://github.com/ROCm/ROCm/issues/5724) - MES 0x83 firmware causing memory access faults
- [ROCm/ROCm#5590](https://github.com/ROCm/ROCm/issues/5590) - CWSR triggering MES 0x80 hangs

**Symptoms**:
- GPU hangs during `hipMalloc` / `hipMemcpy` operations
- No error messages returned - calls simply never return
- Affects RX 7900 XT (gfx1100) and other RDNA3 consumer cards
- Occurs even with sufficient GPU memory available

### 2. Memory Allocation Pathology

**GitHub Issues**:
- [ROCm/hip#3370](https://github.com/ROCm/hip/issues/3370) - `hipFreeAsync` hangs on RX 7900 XT
- [ROCm/hip#3384](https://github.com/ROCm/hip/issues/3384) - Memory not properly freed despite `hipFree()` calls
- [ROCm/ROCm#5581](https://github.com/ROCm/ROCm/issues/5581) - ROCm 6.4.4 works, but 7.0+ exhibits stalls/hangs

**Our Case**:
- 24-layer KV cache allocation: **SUCCESS** (48 × 7MB contiguous buffers)
- Layer norm weight allocation: **HANGS** (many × 3584-byte small buffers)
- This pattern matches the "multiple small allocations" pathology

### 3. Consumer Card Limitations

**AMD Statement**: ROCm support for consumer RDNA3 cards is "not officially supported"

**Community Consensus**:
- Primary development focus on Instinct datacenter GPUs (MI200/MI300 series)
- Consumer cards (RX 7000 series) have reduced testing
- Multiple reports of crashes leading to removed support for some consumer cards

---

## Confirmed Workarounds

### Priority 1: Disable CWSR (Most Effective)

**Method**: Add kernel command line parameter

```bash
# Add to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub
amdgpu.cwsr_enable=0

# Then update grub
sudo update-grub
# Or on Fedora
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
```

**Citation**: [ROCm/ROCm#5590](https://github.com/ROCm/ROCm/issues/5590)
**Status**: Confirmed working even with latest ROCm 7.1.1 firmware
**Reboot Required**: Yes

### Priority 2: Downgrade to ROCm 6.4.4

**Citation**: [ROCm/ROCm#5581](https://github.com/ROCm/ROCm/issues/5581)
**Status**: User-reported stability improvements
**Trade-off**: Loses ROCm 7.x features

### Priority 3: Batch Allocations (Code-Level)

**Strategy**:
- Pre-allocate large buffers at startup
- Subdivide internally for layer weights
- Eliminates multiple `hipMalloc` calls

**Implementation**:
```rust
// Instead of:
for tensor in tensors {
    let buffer = hipMalloc(tensor.size)?;  // Many small allocations
}

// Use:
let giant_buffer = hipMalloc(total_size)?;  // One large allocation
for (offset, tensor) in tensors.iter().enumerate() {
    use_sub_buffer(giant_buffer, offset, tensor.size);
}
```

### Priority 4: Increase GTT Size

**Citation**: Framework Laptop community reports
**Method**: Set `gttsize` to 60GB (requires reboot)

### Priority 5: Use NULL Stream

**Citation**: [ROCm/hip#3384](https://github.com/ROCm/hip/issues/3384)
**Status**: Reported workaround for allocation failures

---

## Recommended Solutions for ROCmForge

### Solution A: Memory Pooling Architecture (Recommended)

**File**: `src/backend/memory_pool.rs` (new file)

**Design**:
1. Pre-allocate weight buffer arena at model load time
2. Use offset-based indexing instead of individual allocations
3. Eliminates thousands of `hipMalloc` calls

**Benefits**:
- Addresses the root cause (many small allocations)
- No system configuration changes required
- Improves performance (fewer driver calls)

**Effort**: 2-3 days

### Solution B: System Configuration (Quick Win)

**Action**: Apply `amdgpu.cwsr_enable=0` kernel parameter

**Steps**:
1. Edit `/etc/default/grub`
2. Add `amdgpu.cwsr_enable=0` to `GRUB_CMDLINE_LINUX_DEFAULT`
3. Run `sudo update-grub`
4. Reboot

**Effort**: 10 minutes

### Solution C: Explicit Synchronization (Partial Mitigation)

**File**: `src/loader/gguf.rs:588-597`

**Change**: Add `hipDeviceSynchronize()` after every N allocations

**Limitation**: Only reduces frequency of hangs, doesn't fix root cause

---

## Test Commands

### Before Any Fix (Reproduce Hang)

```bash
# Kill existing processes
pkill -9 rocmforge_cli

# Run with 60 second timeout (will hang)
timeout 60 ./target/release/rocmforge_cli generate \
  --gguf "/path/to/model.gguf" \
  --prompt "Hello" \
  --max-tokens 5
```

### After Applying CWSR Disable

```bash
# Verify kernel parameter is applied
cat /proc/cmdline | grep cwsr_enable

# Should see: amdgpu.cwsr_enable=0

# Run inference (should complete)
timeout 180 ./target/release/rocmforge_cli generate \
  --gguf "/path/to/model.gguf" \
  --prompt "Hello" \
  --max-tokens 5
```

---

## References

### GitHub Issues
- [ROCm/ROCm#5724](https://github.com/ROCm/ROCm/issues/5724) - MES 0x83 firmware causing GPU Hang
- [ROCm/ROCm#5590](https://github.com/ROCm/ROCm/issues/5590) - CWSR causing MES firmware 0x80 hang
- [ROCm/ROCm#5581](https://github.com/ROCm/ROCm/issues/5581) - ComfyUI stall/hang on gfx1201
- [ROCm/hip#3370](https://github.com/ROCm/hip/issues/3370) - hipFreeAsync hangs on RX 7900 XT
- [ROCm/hip#3384](https://github.com/ROCm/hip/issues/3384) - hipMalloc fails despite enough memory
- [ROCm/ROCm#3644](https://github.com/ROCm/ROCm/issues/3644) - Segmentation faults with dual RX 7900 GRE
- [ROCm/ROCm#4903](https://github.com/ROCm/ROCm/issues/4903) - llama.cpp shared memory issue

### Documentation
- [ROCm consolidated changelog](https://rocm.docs.amd.com/en/latest/release/changelog.html)
- [ROCm 7.1.1 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm environment variables](https://rocm.docs.amd.com/en/latest/reference/env-variables.html)

### Community Discussions
- [Is everything broken with RDNA3?](https://www.reddit.com/r/ROCm/comments/1agh38b/is_everything_actually_this_broken_especially/)
- [ROCm on Framework Laptop](https://community.frame.work/t/experiments-with-using-rocm-on-the-fw16-amd/62189)
- [Faster llama.cpp ROCm performance for RDNA3](https://www.reddit.com/r/LocalLLaMA/comments/1ok7hd4/faster_llamacpp_rocm_performance_for_amd_rdna3/)

---

## Conclusion

This is **not a bug in ROCmForge code**. It is a **known ROCm driver/firmware issue** affecting RDNA3 consumer cards.

**Recommended Action Plan**:
1. **Immediate**: Apply `amdgpu.cwsr_enable=0` kernel parameter (10 minutes, reboot required)
2. **Short-term**: Implement memory pooling architecture (2-3 days development)
3. **Long-term**: Consider ROCm 6.4.4 for stability if 7.1 remains problematic

**No code changes will fix this** without either:
- System configuration change (disable CWSR)
- Architecture change (memory pooling)
- ROCm version downgrade
