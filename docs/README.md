# ROCmForge Documentation Index

> **Start here:** Read this index to understand what each document contains.

---

## Quick Links

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **rocm_setup_guide.md** | ROCm/HIP installation & compilation | **Start here** - verify your environment |
| **implementation_principles.md** | Methodology, mindset, rules | Read before implementing anything |
| **implementation_roadmap.md** | Phase-by-phase tasks | Read before starting each phase |
| **codebase_audit.md** | Current state: what exists/missing | Understand what we're working with |
| **kernel_research.md** | HIP code patterns and snippets | Reference while writing kernels |
| **nvidia_stack_decomposed.md** | NVIDIA → AMD component mapping | Understand the target architecture |
| **practical_order_of_attack.md** | Original execution strategy | Historical context |

---

## Document Summaries

### 0. rocm_setup_guide.md

**The environment setup.** Verify your ROCm/HIP installation.

Contains:
- ROCm 7.1 installation instructions
- hipcc compilation commands
- HSACO generation for runtime kernel loading
- Rust + HIP FFI integration patterns
- Common commands and troubleshooting

**Start here** to confirm your development environment works.

---

### 1. implementation_principles.md

**The methodology document.** Read this first.

Contains 10 principles that guide all implementation:
1. Start with contracts, not kernels
2. Fix biggest latency leaks first
3. Implement stub GPU path before optimizing
4. Use rocBLAS before custom MFMA
5. One kernel = one responsibility
6. Build system discipline
7. Profiling rules
8. Debugging mindset
9. LLM usage strategy
10. Roadmap order (don't reorder)

**Key quote:** "Make it correct → make it measurable → then make it fast."

---

### 2. implementation_roadmap.md

**The execution plan.** Follow this phase by phase.

```
Phase 1: Replace stubs (scale, mask, softmax)
Phase 2: RoPE + KV append (biggest latency win)
Phase 3: FlashAttention (performance unlock)
Phase 4: MLP Ops (SwiGLU + RMSNorm)
Phase 5: Optional (GPU sampler, MFMA, FP16, tuning)
```

Each phase contains:
- Exact contracts (input/output layouts)
- Complete HIP kernel implementations
- CPU vs GPU test templates
- Exit criteria

---

### 3. codebase_audit.md

**The current state.** What we have vs what's missing.

**Key findings:**
- Infrastructure: ✅ Complete (backend, scheduler, HTTP, loaders)
- GPU kernels: ❌ No-op stubs only
- CPU fallback: Every decode does ~4-5 GPU↔CPU round-trips

**What's mocked:**
- `scale_gpu_kernel` - returns 0, does nothing
- `mask_gpu_kernel` - returns 0, does nothing
- `softmax_gpu_kernel` - returns 0, does nothing
- RoPE GPU - falls back to CPU
- SwiGLU - CPU activation
- LayerNorm - CPU only

---

### 4. kernel_research.md

**Reference patterns.** Use when writing kernels.

Contains:
- FlashAttention implementation pattern
- Softmax kernel with LDS reduction
- SwiGLU activation
- RoPE GPU kernel
- MFMA matrix multiplication
- RMSNorm/LayerNorm
- Build system patterns (build.rs, HSACO loading)
- Reference resources (AMD docs, GitHub repos)

---

### 5. nvidia_stack_decomposed.md

**Architecture context.** What we're building against.

NVIDIA's 6-component stack:
1. Linear Algebra (cuBLAS → rocBLAS)
2. Attention Kernels (FlashAttention)
3. Memory Model (KV cache, paged attention)
4. Execution Model (streams, kernel fusion)
5. Sampler/Decode (top-k/top-p)
6. Glue Layer (tokenizer, loaders, API)

Maps each to AMD equivalents.

---

### 6. practical_order_of_attack.md

**Original strategy.** Historical context.

Documents the "execution sequence" and what was already done (infrastructure) vs what's missing (kernels).

---

## How to Use These Docs

### If You're Implementing a Kernel

1. **Read** `implementation_principles.md` (sections 1-5)
2. **Read** the relevant phase in `implementation_roadmap.md`
3. **Reference** `kernel_research.md` for code patterns
4. **Verify** against `codebase_audit.md` (understand the call site)

### If You're Debugging

1. **Read** `implementation_principles.md` (section 8: Debugging mindset)
2. **Check** `codebase_audit.md` for current state
3. **Consult** `kernel_research.md` for similar working patterns

### If You're Planning Architecture

1. **Read** `nvidia_stack_decomposed.md` for component breakdown
2. **Read** `practical_order_of_attack.md` for execution sequence
3. **Review** `codebase_audit.md` for existing patterns

### If You're Profiling/Optimizing

1. **Read** `implementation_principles.md` (section 7: Profiling rules)
2. **Follow** `implementation_roadmap.md` phase order exactly
3. **Measure** before optimizing (rule of thumb)

---

## The Implementation Contract

Every kernel implementation must satisfy:

1. **Exact signature** matching Rust FFI call site
2. **Tensor layout** contract (row-major, documented strides)
3. **CPU reference** function for correctness validation
4. **Test** that compares CPU vs GPU within tolerance (1e-5)
5. **Exit criteria** documented and satisfied

**Nothing ships without these.**

---

## Quick Reference: Tensor Layouts

All tensors use **row-major (C-style)** layout:

```
Q, K, V:    [batch, seq_len, head_dim]
            Access: q[b*S*D + s*D + d]

Scores:     [batch, seq_len, seq_len]
            Access: scores[b*S*S + i*S + j]

Output:     [batch, seq_len, head_dim]
            Access: out[b*S*D + s*D + d]
```

---

## File Structure

```
docs/
├── README.md                        (this file)
├── implementation_principles.md     (methodology - START HERE)
├── implementation_roadmap.md        (phase-by-phase tasks)
├── codebase_audit.md                (current state)
├── kernel_research.md               (code patterns)
├── nvidia_stack_decomposed.md       (architecture context)
└── practical_order_of_attack.md     (original strategy)

src/
├── attention/
│   ├── kernels.rs                   (no-op stubs - Phase 1)
│   ├── gpu.rs                       (calls kernels - needs updating)
│   ├── rope.rs                      (CPU fallback - Phase 2)
│   ├── softmax.rs                   (CPU reference)
│   └── mask.rs                      (CPU reference)
└── backend/
    └── hip_backend.rs               (FFI, kernel loading)
```

---

## Status

| Phase | Kernel | Status | Documented In |
|-------|--------|--------|---------------|
| 1 | scale_kernel | ⚪ Not started | roadmap.md Task 1.1 |
| 1 | mask_kernel | ⚪ Not started | roadmap.md Task 1.2 |
| 1 | softmax_kernel | ⚪ Not started | roadmap.md Task 1.3 |
| 2 | rope_kernel | ⚪ Not started | roadmap.md Task 2.3 |
| 2 | rope_kv_append_fused | ⚪ Not started | roadmap.md Task 2.4 |
| 3 | flash_attention | ⚪ Not started | roadmap.md Task 3.3 |
| 4 | swiglu_kernel | ⚪ Not started | roadmap.md Task 4.2 |
| 4 | rms_norm_kernel | ⚪ Not started | roadmap.md Task 4.3 |

---

## Critical Crash Fix (2025-01-03)

**CRITICAL FIX REQUIRED** - Read `CRASH_FIX_SUMMARY.md` for immediate fix instructions.

**Quick Fix (2 minutes):**
```bash
# File: src/backend/hip_backend.rs, Line 499
# Remove #[repr(C)] from HipBackend struct

#[repr(C)]  # ❌ REMOVE THIS LINE
#[derive(Debug)]
pub struct HipBackend {
    device: HipDevice,
    stream: Arc<HipStream>,
}
```

**Root Cause:** ABI violation when returning 24-byte struct containing `Arc<T>` with `repr(C)` annotation.

**Documents:**
- `CRASH_FIX_SUMMARY.md` - Quick fix guide
- `deep_crash_analysis.md` - Detailed technical analysis
- `ABI_RESEARCH_WITH_CITATIONS.md` - Research with sources

**Confidence:** 100% - Root cause definitively identified

---

> **Remember:** "Make it correct → make it measurable → then make it fast."
