---
phase: 28-rocm-compilation-fix
plan: 05
subsystem: Kernels
tags: [cfg-gates, rocm, compilation, kernels]

one_liner: Removed all #[cfg(feature = "rocm")] conditional compilation gates from src/kernels/ directory, making GPU kernels always available.

completed: 2026-01-20
duration: 8 minutes

requires:
  - Phase 28-01: Import fixes (completed)
  - Phase 28-02: Variable naming fixes (completed)

provides:
  - Unconditionally compiled GPU kernel modules in src/kernels/
  - Kernel loading and execution always available at runtime
  - Eliminated conditional compilation complexity from kernel code

tech-stack:
  added: []
  patterns: [unconditional-kernel-compilation, runtime-kernel-loading]

key-files:
  created: []
  modified:
    - src/kernels/attention/flash.rs
    - src/kernels/attention/mask.rs
    - src/kernels/attention/matmul.rs
    - src/kernels/attention/mod.rs
    - src/kernels/attention/rope.rs
    - src/kernels/attention/softmax.rs
    - src/kernels/element/rms_norm.rs
    - src/kernels/element/swiglu.rs
    - src/kernels/matmul/quantized/mod.rs
    - src/kernels/matmul/quantized/q4_0.rs
    - src/kernels/matmul/quantized/q4_k.rs
    - src/kernels/matmul/quantized/q6_k.rs
    - src/kernels/quant/mod.rs
    - src/kernels/quant/q4_0.rs
    - src/kernels/quant/q4_k.rs
    - src/kernels/quant/q6_k.rs
    - src/kernels/transpose/mod.rs
    - src/attention/kernels/kernels_cache/mod.rs
    - src/attention/kernels/kernels_cache/kernels_basic.rs
    - src/attention/kernels/kernels_cache/kernels_flash.rs
    - src/attention/kernels/mod.rs
    - src/ops/attention/kernels.rs
    - src/attention/backend_registry.rs

decisions:
  - Kernel code should always be compiled: GPU kernels are core to ROCmForge's functionality, not optional features. Conditional compilation gates were removed to ensure kernels are always available for runtime loading.
  - Add missing imports: When removing cfg gates, discovered missing `std::sync::Mutex`, `std::path::Path`, and `std::ffi::c_void` imports that were previously conditionally compiled.

deviations:
  Work completed in prior plan (28-03): The cfg gate removal from src/kernels/ was completed as part of plan 28-03 ("remove cfg(feature = "rocm") gates from src/attention/"). That commit (b4c10ae) included all 16 src/kernels/ files specified in this plan.
  
  Additional fixes made during this session:
  - Fixed src/attention/backend_registry.rs: Added `mut` keyword to `backends` variable in `AttentionBackendRegistry::new()` (pre-existing compilation error)
  - Fixed src/ops/attention/kernels.rs: Made `CompiledKernel.module` field public to allow struct construction from other modules
  - Added missing imports when removing cfg gates: `std::sync::Mutex`, `std::path::Path`, `std::ffi::c_void`
  - Simplified `can_use_flash_attention()` and `supports_causal_mask()` functions to remove `cfg!(feature = "rocm")` macro calls (now always returns based on actual constraints)

verification:
  - grep -r "cfg.*feature.*rocm" src/kernels/ returns 0 results
  - cargo check passes successfully (13 warnings, 0 errors)
  - All 16 kernel modules compile unconditionally

next_phase_ready: true
---
