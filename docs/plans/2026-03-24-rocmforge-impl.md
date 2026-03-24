# rocmforge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build AMD GPU/CPU inference engine for Qwen2.5 with metadata-driven configuration, TDD approach, and HTTP server.

**Architecture:** Direct port from Memoria with modularization for 1K LOC limit. Two explicit execution paths (CPU/GPU), no fallback. Metadata-driven config from GGUF files.

**Tech Stack:** Rust 2021, HIP (AMD GPU), memmap2, half, regex, rayon, axum (HTTP server)

---

## Status

| Phase | Status | Lines |
|-------|--------|-------|
| 1. Project Scaffold | ✅ Done | - |
| 2. Loader | ✅ Done | 979 |
| 3. Config | ✅ Done | 579 |
| 4. Tokenizer | ✅ Done | 577 |
| 5. CPU Backend | ❌ TODO | 0 |
| 6. GPU Backend | ❌ TODO | 0 |
| 7. CLI Entry Point | ❌ TODO | 0 |
| 8. HTTP Server | ❌ TODO | 0 |

---

## Phase 5: CPU Backend

### Task 5.1: CPU Weights

**Files:**
- Create: `src/cpu/mod.rs`
- Create: `src/cpu/weights.rs`

**Port from Memoria:** `src/cpu/weights.rs`

**Key types:**
```rust
pub struct CpuModelWeights {
    pub token_embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: Vec<f32>,  // lm_head
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_q_bias: Option<Vec<f32>>,
    pub attn_k_bias: Option<Vec<f32>>,
    pub attn_v_bias: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}
```

**Step 1: Create src/cpu/mod.rs**

```rust
pub mod weights;
pub mod cache;
pub mod forward;
pub mod prefill;
pub mod sampler;
pub mod ops;

use crate::loader::GgmlType;

#[derive(Debug)]
pub enum CpuError {
    UnsupportedWeightType(GgmlType),
    DimensionMismatch(&'static str),
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuError::UnsupportedWeightType(t) => {
                write!(f, "unsupported weight type for CPU backend: {:?}", t)
            }
            CpuError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
        }
    }
}

impl std::error::Error for CpuError {}
```

**Step 2: Create src/cpu/weights.rs** (port from Memoria)

**Step 3: Add dequantization for Q4_0, Q4_1, Q8_0**

**Step 4: Test weight loading**

**Step 5: Commit**

---

### Task 5.2: CPU KV Cache

**Files:**
- Create: `src/cpu/cache.rs`

**Key types:**
```rust
pub struct CpuKvCache {
    pub k: Vec<Vec<Vec<f32>>>,  // [layers][seq_len][kv_heads * head_dim]
    pub v: Vec<Vec<Vec<f32>>>,
    pub seq_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl CpuKvCache {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self;
    pub fn clear(&mut self);
}
```

---

### Task 5.3: CPU Ops

**Files:**
- Create: `src/cpu/ops.rs`

**Key functions:**
```rust
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32);
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], rows: usize, cols: usize);
pub fn rope_neox(x: &mut [f32], pos: usize, head_dim: usize, theta: f32);
pub fn softmax(x: &mut [f32]);
pub fn silu(x: f32) -> f32;
```

---

### Task 5.4: CPU Forward Pass

**Files:**
- Create: `src/cpu/forward.rs`

**Key functions:**
```rust
pub fn cpu_embed_token(token: u32, weights: &CpuModelWeights, out: &mut [f32], config: &ModelConfig);
pub fn layer_forward(hidden: &mut [f32], layer: &LayerWeights, kv: &mut CpuKvCache, scratch: &mut CpuForwardScratch, pos: usize, config: &ModelConfig);
pub fn cpu_full_forward(hidden: &mut [f32], weights: &CpuModelWeights, kv: &mut CpuKvCache, scratch: &mut CpuForwardScratch, pos: usize, config: &ModelConfig);

pub struct CpuForwardScratch {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub logits: Vec<f32>,
}
```

---

### Task 5.5: CPU Prefill

**Files:**
- Create: `src/cpu/prefill.rs`

**Key functions:**
```rust
pub fn cpu_prefill_forward(
    tokens: &[u32],
    weights: &CpuModelWeights,
    kv: &mut CpuKvCache,
    scratch: &mut CpuForwardScratch,
    config: &ModelConfig,
) -> Result<(), CpuError>;
```

---

### Task 5.6: CPU Sampler

**Files:**
- Create: `src/cpu/sampler.rs`

**Key functions:**
```rust
pub fn cpu_sample_greedy(logits: &[f32]) -> u32;
pub fn cpu_sample_top_p(logits: &[f32], temperature: f32, top_p: f32, seed: u64) -> u32;
```

---

## Phase 6: GPU Backend (HIP)

### Task 6.1: HIP Context
### Task 6.2: HIP Kernels
### Task 6.3: GPU Weights
### Task 6.4: GPU KV Cache
### Task 6.5: GPU Forward/Prefill/Sampler

---

## Phase 7: CLI Entry Point

### Task 7.1: Main CLI with --device cpu|gpu

---

## Phase 8: HTTP Server

### Task 8.1: Server Scaffold
### Task 8.2: OpenAI API
### Task 8.3: Claude API

---

## Next Action

**Start Phase 5: CPU Backend**

Task 5.1: Create `src/cpu/mod.rs` and `src/cpu/weights.rs`
