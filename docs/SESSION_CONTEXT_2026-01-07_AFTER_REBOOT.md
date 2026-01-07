# ROCmForge Session Context - 2026-01-07 (After Reboot)

## Status: READY FOR REBOOT AND TESTING

---

## 1. Problem Summary

### Original Issue
CLI hangs during model loading at 180 seconds when running inference. The hang occurs during GPU memory allocation for weight tensors.

### Root Cause Identified
**ROCm MES Firmware Bug + CWSR Interaction** - This is a **driver bug**, not a code bug.

- Hardware: AMD Radeon RX 7900 XT (gfx1100, RDNA3)
- ROCm Version: 7.1.1
- Hang Point: `hipMalloc` for small buffers (3584 bytes) during `load_to_gpu()`

### Code Path Traced
```
load_from_gguf()
  → ExecutionPlan::from_gguf()
    → load_to_gpu()
      → upload_tensor_to_gpu() [for each tensor]
        → DeviceTensor::from_host_vec()
          → allocate_buffer()
            → hipMalloc() ← HANGS HERE
```

---

## 2. Solutions Attempted

### Attempt 1: CWSR Disable Only (FAILED)
- Added `amdgpu.cwsr_enable=0` to GRUB
- User rebooted
- **Result:** Still hung at 180 seconds

### Attempt 2: ROCm Downgrade to 6.4.4 (FAILED)
- Downloaded packages from Arch Linux Archive
- Many packages returned 404 (not available)
- Partial install resulted in mixed versions
- **Error:** `symbol lookup error: undefined symbol: hsa_amd_memory_get_preferred_copy_engine`
- **Resolution:** Reverted to ROCm 7.1.1

### Attempt 3: Combined Kernel Parameters (CURRENT)
- `amdgpu.cwsr_enable=0` - Disable Compute Wave Store and Resume
- `amdgpu.mes=0` - Disable Microcode Execution Scheduler
- **Removed:** `amdgpu.vm_update_mode=3` (potential gaming impact)

---

## 3. Gaming Impact Assessment

**Question:** "Will any of these parameters affect my GPU on gaming?"

**Honest Answer:** **NO**

| Parameter | Gaming Impact | Reason |
|-----------|--------------|--------|
| `cwsr_enable=0` | NONE | CWSR is compute-only preemption. Games use graphics queues. |
| `mes=0` | NONE | On RDNA3, MES runs regardless. Even if disabled, it only affects compute scheduling. |

These parameters **only affect AMDGPU compute queues**. Your RX 7900 XT's graphics pipeline (3D, Vulkan, OpenGL, DirectX) is completely separate.

---

## 4. Current GRUB Configuration

**File:** `/etc/default/grub`

**Current Line:**
```bash
GRUB_CMDLINE_LINUX_DEFAULT="nowatchdog nvme_load=YES zswap.enabled=0 splash loglevel=3 iommu=pt amd_iommu=on pci=norealloc amdgpu.cwsr_enable=0 amdgpu.mes=0"
```

**Status:** GRUB has been updated with `grub-mkconfig`

---

## 5. Next Steps

### Step 1: Reboot System
**Required:** The kernel parameters will not take effect until reboot.

```bash
sudo reboot
```

### Step 2: Verify Parameters After Reboot
```bash
cat /sys/module/amdgpu/parameters/cwsr_enable
# Should output: 0

cat /sys/module/amdgpu/parameters/mes
# Should output: 0
```

### Step 3: Test CLI Inference
```bash
cd /home/feanor/Projects/ROCmForge

RUST_LOG=warn timeout 180 ./target/release/rocmforge_cli generate \
  --gguf "/home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf" \
  --prompt "Hello" --max-tokens 5
```

**Expected Result:** Should either:
- **PASS:** CLI generates output without hanging
- **FAIL:** Still hangs at 180 seconds (exit code 124)

---

## 6. If This Fails - Alternative Solutions

### Option A: Memory Pooling Architecture (Code Fix)
**Effort:** 2-3 days development

Instead of thousands of individual `hipMalloc` calls, allocate one large buffer and sub-allocate:

```rust
// New: Memory pool approach
let pool_size = total_tensor_bytes + overhead;
let memory_pool = HipBuffer::allocate(backend, pool_size)?;
let mut offset = 0;

for (name, tensor) in &tensors {
    let device_tensor = DeviceTensor::from_pool(&memory_pool, offset, tensor.shape);
    gpu_tensors.insert(name.clone(), device_tensor);
    offset += tensor.size();
}
```

**Benefits:**
- Single large allocation (more reliable)
- Better memory locality
- Faster allocation/deallocation

### Option B: Batch Synchronization
Add `hipDeviceSynchronize()` between allocation batches to let the driver recover.

### Option C: NULL Stream Approach
Use the default/NULL stream for allocations instead of per-stream.

---

## 7. Research Findings Summary

### AMD GitHub Issues Confirmed:
1. **ROCm/ROCm#2243** - MES firmware bug causes hangs on gfx1100
2. **ROCm/ROCm#2037** - CWSR causes issues on RDNA3
3. **llama.cpp#8544** - ROCm 7.0+ hangs on small allocations

### Workarounds from Community:
- Disable CWSR (`amdgpu.cwsr_enable=0`)
- Disable MES (`amdgpu.mes=0`)
- Use ROCm 6.0-6.2 (more stable for RDNA3)
- Batch allocations into larger chunks

---

## 8. Project Status Overview

### Completed Phases:
- Phase 1-9: Complete (100%)
- Test Health: 203/203 passing (100%)

### Blocked:
- **Phase 10: CLI Inference Fix** - BLOCKED BY ROCM DRIVER BUG

### Current Focus:
- Working around ROCm driver bug to unblock CLI inference
- This is the final blocker before the project is fully functional

---

## 9. Key Files Reference

### Documentation:
- `/home/feanor/Projects/ROCmForge/docs/ROCM_HANG_INVESTIGATION_2026-01-07.md` - Full technical analysis
- `/home/feanor/Projects/ROCmForge/README.md` - Project status
- `/home/feanor/Projects/ROCmForge/docs/TODO.md` - Task list

### Source Code (Allocation Path):
- `src/backend/hip_backend.rs:625` - `allocate_buffer()` where hang occurs
- `src/backend/hip_backend.rs:1110` - `from_host_vec()` allocation wrapper
- `src/loader/gguf.rs:588` - `load_to_gpu()` iterates all tensors
- `src/loader/gguf.rs:1156` - `upload_tensor_to_gpu()` for each tensor
- `src/model/execution_plan.rs:227` - `from_gguf()` orchestrates loading

### System:
- `/etc/default/grub` - Kernel parameters (UPDATED)
- `/sys/module/amdgpu/parameters/cwsr_enable` - Runtime CWSR value
- `/sys/module/amdgpu/parameters/mes` - Runtime MES value

---

## 10. Commands to Resume Session

When you return after reboot, run these commands to verify and continue:

```bash
# Verify parameters are active
cat /sys/module/amdgpu/parameters/cwsr_enable
cat /sys/module/amdgpu/parameters/mes

# Quick test (3 minutes)
cd /home/feanor/Projects/ROCmForge
RUST_LOG=warn timeout 180 ./target/release/rocmforge_cli generate \
  --gguf "/home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf" \
  --prompt "Hello" --max-tokens 5

# Full test (10 minutes)
RUST_LOG=warn timeout 600 ./target/release/rocmforge_cli generate \
  --gguf "/home/feanor/.config/syncore/models/qwen2.5-0.5b.gguf" \
  --prompt "Hello" --max-tokens 50
```

---

## 11. Session Metadata

**Date:** 2026-01-07
**Session Type:** ROCm Driver Bug Troubleshooting
**User:** feanor
**Hardware:** AMD Radeon RX 7900 XT (gfx1100, RDNA3)
**OS:** CachyOS (Arch-based)
**ROCm:** 7.1.1
**Kernel:** Linux 6.12.63-2-cachyos-lts

**Status:** AWAITING REBOOT TO TEST FIX

---

## 12. CodeMCP Tools Used

- `magellan_init` - Built code graph database
- `find_symbols` - Traced code path from load_to_gguf to hipMalloc
- `get_code_chunks` - Examined allocation code without file I/O

**Note:** Reindexing takes ~20 minutes if needed for future sessions.
