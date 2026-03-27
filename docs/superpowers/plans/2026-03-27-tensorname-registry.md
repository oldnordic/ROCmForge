# TensorNameRegistry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable metadata-driven tensor name resolution to support Qwen3 models without hardcoded "blk.N." prefix

**Architecture:** Extend ModelTraits with TensorNamingScheme; create TensorNameRegistry to format semantic tensor names; update weight loading to use registry

**Tech Stack:** Rust, HashMap, existing GGUF loader

---

## File Structure

**Files to modify:**
1. `src/config.rs` - Add TensorNamingScheme, TensorName enum, TensorNameRegistry, extend ModelTraits, update ModelConfig
2. `src/cpu/weights.rs` - Replace hardcoded `p` closure with `tensor_registry.resolve()`
3. `src/gpu/weights.rs` - Same changes as CPU (if exists)

**New code units:**
- `TensorNamingScheme` enum (2 variants)
- `TensorName` enum (15 variants)
- `TensorNameRegistry` struct with HashMap

---

## Task 1: Add TensorNamingScheme and TensorName enums

**Files:**
- Modify: `src/config.rs:33-47` (after AttentionLayout, before ModelTraits)

- [ ] **Step 1: Write TensorNamingScheme enum**

Add after line 32 (after AttentionLayout enum):

```rust
/// Tensor naming convention used by GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorNamingScheme {
    /// GGUF-style: `blk.N.ffn_gate.weight`
    Gguf,
    /// HuggingFace-style: `model.layers.N.mlp.gate_proj.weight`
    HuggingFace,
}
```

- [ ] **Step 2: Write TensorName enum**

Add after TensorNamingScheme enum:

```rust
/// Semantic tensor names that are resolved to actual tensor names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorName {
    // Attention weights
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    // Attention biases (optional)
    AttnQBias,
    AttnKBias,
    AttnVBias,
    // FFN weights
    FfnGate,
    FfnUp,
    FfnDown,
    // Normalization
    AttnNorm,
    FfnNorm,
    // Embeddings
    TokenEmb,
    LmHead,
    // Output norm
    OutputNorm,
}
```

- [ ] **Step 3: Verify code compiles**

Run: `cargo check --lib 2>&1 | grep -E "error|Finished" | head -3`

Expected: "Finished" or no compilation errors

---

## Task 2: Add TensorNameRegistry struct

**Files:**
- Modify: `src/config.rs:47-52` (after ModelTraits struct, before static REGISTRY)

- [ ] **Step 1: Add TensorNameRegistry struct**

Add after line 46 (after ModelTraits struct closing brace):

```rust
/// Registry for resolving semantic tensor names to actual tensor names
/// based on the model's naming convention.
#[derive(Debug, Clone)]
pub struct TensorNameRegistry {
    /// The naming scheme this registry uses
    pub scheme: TensorNamingScheme,
    /// Map of semantic names to format templates (use {} for layer number)
    templates: HashMap<TensorName, String>,
}

impl TensorNameRegistry {
    /// Create a registry for a specific naming scheme.
    pub fn from_scheme(scheme: &TensorNamingScheme) -> Self {
        match scheme {
            TensorNamingScheme::Gguf => Self::gguf_format(),
            TensorNamingScheme::HuggingFace => Self::huggingface_format(),
        }
    }

    /// Resolve a semantic tensor name for a specific layer.
    ///
    /// # Panics
    /// Panics if the tensor name is not in the registry (indicates bug).
    pub fn resolve(&self, name: TensorName, layer: usize) -> String {
        let template = self.templates.get(&name).expect("Tensor name should be in registry");
        template.replace("{}", &layer.to_string())
    }

    /// Resolve an optional tensor name (for bias tensors that may not exist).
    ///
    /// Returns None if the naming scheme doesn't support this tensor.
    pub fn resolve_optional(&self, name: TensorName, layer: usize) -> Option<String> {
        self.templates.get(&name).cloned().map(|t| t.replace("{}", &layer.to_string()))
    }

    /// Create GGUF-format registry.
    fn gguf_format() -> Self {
        let mut templates = HashMap::new();

        // Attention weights
        templates.insert(TensorName::AttnQ, "blk.{}.attn_q.weight".to_string());
        templates.insert(TensorName::AttnK, "blk.{}.attn_k.weight".to_string());
        templates.insert(TensorName::AttnV, "blk.{}.attn_v.weight".to_string());
        templates.insert(TensorName::AttnOutput, "blk.{}.attn_output.weight".to_string());

        // Attention biases
        templates.insert(TensorName::AttnQBias, "blk.{}.attn_q.bias".to_string());
        templates.insert(TensorName::AttnKBias, "blk.{}.attn_k.bias".to_string());
        templates.insert(TensorName::AttnVBias, "blk.{}.attn_v.bias".to_string());

        // FFN weights
        templates.insert(TensorName::FfnGate, "blk.{}.ffn_gate.weight".to_string());
        templates.insert(TensorName::FfnUp, "blk.{}.ffn_up.weight".to_string());
        templates.insert(TensorName::FfnDown, "blk.{}.ffn_down.weight".to_string());

        // Normalization
        templates.insert(TensorName::AttnNorm, "blk.{}.attn_norm.weight".to_string());
        templates.insert(TensorName::FfnNorm, "blk.{}.ffn_norm.weight".to_string());

        // Embeddings (no layer number)
        templates.insert(TensorName::TokenEmb, "token_embd.weight".to_string());
        templates.insert(TensorName::LmHead, "output.weight".to_string());
        templates.insert(TensorName::OutputNorm, "output_norm.weight".to_string());

        Self {
            scheme: TensorNamingScheme::Gguf,
            templates,
        }
    }

    /// Create HuggingFace-format registry.
    fn huggingface_format() -> Self {
        let mut templates = HashMap::new();

        // Attention weights
        templates.insert(TensorName::AttnQ, "model.layers.{}.self_attn.q_proj.weight".to_string());
        templates.insert(TensorName::AttnK, "model.layers.{}.self_attn.k_proj.weight".to_string());
        templates.insert(TensorName::AttnV, "model.layers.{}.self_attn.v_proj.weight".to_string());
        templates.insert(TensorName::AttnOutput, "model.layers.{}.self_attn.o_proj.weight".to_string());

        // Attention biases
        templates.insert(TensorName::AttnQBias, "model.layers.{}.self_attn.q_proj.bias".to_string());
        templates.insert(TensorName::AttnKBias, "model.layers.{}.self_attn.k_proj.bias".to_string());
        templates.insert(TensorName::AttnVBias, "model.layers.{}.self_attn.v_proj.bias".to_string());

        // FFN weights
        templates.insert(TensorName::FfnGate, "model.layers.{}.mlp.gate_proj.weight".to_string());
        templates.insert(TensorName::FfnUp, "model.layers.{}.mlp.up_proj.weight".to_string());
        templates.insert(TensorName::FfnDown, "model.layers.{}.mlp.down_proj.weight".to_string());

        // Normalization
        templates.insert(TensorName::AttnNorm, "model.layers.{}.input_layernorm.weight".to_string());
        templates.insert(TensorName::FfnNorm, "model.layers.{}.post_attention_layernorm.weight".to_string());

        // Embeddings (no layer number)
        templates.insert(TensorName::TokenEmb, "model.embed_tokens.weight".to_string());
        templates.insert(TensorName::LmHead, "lm_head.weight".to_string());
        templates.insert(TensorName::OutputNorm, "model.norm.weight".to_string());

        Self {
            scheme: TensorNamingScheme::HuggingFace,
            templates,
        }
    }

    /// Get all expected tensor names for a layer (for debugging).
    pub fn expected_tensors(&self, layer: usize) -> Vec<String> {
        self.templates.keys()
            .filter(|name| !matches!(*name, TensorName::TokenEmb | TensorName::LmHead | TensorName::OutputNorm))
            .map(|name| self.resolve(*name, layer))
            .collect()
    }

    /// List all supported tensor names (for debugging).
    pub fn list_all_names(&self) -> Vec<&'static str> {
        vec![
            "AttnQ", "AttnK", "AttnV", "AttnOutput",
            "AttnQBias", "AttnKBias", "AttnVBias",
            "FfnGate", "FfnUp", "FfnDown",
            "AttnNorm", "FfnNorm",
            "TokenEmb", "LmHead", "OutputNorm",
        ]
    }
}
```

- [ ] **Step 2: Verify code compiles**

Run: `cargo check --lib 2>&1 | grep -E "error|Finished" | head -3`

Expected: "Finished" or no compilation errors

---

## Task 3: Extend ModelTraits with tensor_naming field

**Files:**
- Modify: `src/config.rs:37-46` (ModelTraits struct definition)
- Modify: `src/config.rs:51-57` (DEFAULT_TRAITS)
- Modify: `src/config.rs:84-93` (qwen2 family)
- Modify: `src/config.rs:51-57` (DEFAULT_TRAITS)

- [ ] **Step 1: Add tensor_naming field to ModelTraits**

Update line 46 (add after default_norm_eps field):

```rust
pub struct ModelTraits {
    pub rope_style: RopeStyle,
    pub attention_layout: AttentionLayout,
    /// Whether Q/K/V projections have explicit bias terms
    pub use_attention_bias: bool,
    /// Fallback rope theta if absent from GGUF (should be in GGUF, but just in case)
    pub default_rope_theta: f32,
    /// Fallback RMS norm epsilon if absent from GGUF
    pub default_norm_eps: f32,
    /// Tensor naming convention used by this architecture
    pub tensor_naming: TensorNamingScheme,
}
```

- [ ] **Step 2: Update DEFAULT_TRAITS**

Update lines 51-57:

```rust
/// Default traits for unknown architectures (LLaMA-compatible)
static DEFAULT_TRAITS: ModelTraits = ModelTraits {
    rope_style: RopeStyle::Normal,
    attention_layout: AttentionLayout::SplitQkv,
    use_attention_bias: false,
    default_rope_theta: 10000.0,
    default_norm_eps: 1e-5,
    tensor_naming: TensorNamingScheme::Gguf,
};
```

- [ ] **Step 3: Split qwen2 and qwen3 registry entries**

Replace lines 84-93 with:

```rust
// Qwen2 family - NeoX RoPE, QKV bias, split QKV, high rope theta, GGUF naming
let qwen2 = ModelTraits {
    rope_style: RopeStyle::NeoX,
    attention_layout: AttentionLayout::SplitQkv,
    use_attention_bias: true,
    default_rope_theta: 1_000_000.0,
    default_norm_eps: 1e-6,
    tensor_naming: TensorNamingScheme::Gguf,
};
for arch in &["qwen2", "qwen2moe"] {
    m.insert(*arch, qwen2.clone());
}

// Qwen3 family - NeoX RoPE, QKV bias, split QKV, high rope theta, HuggingFace naming
let qwen3 = ModelTraits {
    rope_style: RopeStyle::NeoX,
    attention_layout: AttentionLayout::SplitQkv,
    use_attention_bias: true,
    default_rope_theta: 1_000_000.0,
    default_norm_eps: 1e-6,
    tensor_naming: TensorNamingScheme::HuggingFace,
};
for arch in &["qwen3", "qwen3moe"] {
    m.insert(*arch, qwen3.clone());
}
```

- [ ] **Step 4: Update llama family registry entry**

Add `tensor_naming: TensorNamingScheme::Gguf` to llama definition (line 64-70).

Find the llama definition and add the field:

```rust
let llama = ModelTraits {
    rope_style: RopeStyle::Normal,
    attention_layout: AttentionLayout::SplitQkv,
    use_attention_bias: false,
    default_rope_theta: 10000.0,
    default_norm_eps: 1e-5,
    tensor_naming: TensorNamingScheme::Gguf,
};
```

- [ ] **Step 5: Update phi3 registry entry**

Add `tensor_naming: TensorNamingScheme::Gguf` to phi3 definition (lines 106-114).

- [ ] **Step 6: Update gemma registry entry**

Add `tensor_naming: TensorNamingScheme::Gguf` to gemma definition (lines 118-127).

- [ ] **Step 7: Update glm registry entry**

Add `tensor_naming: TensorNamingScheme::Gguf` to glm definition (lines 130-139).

- [ ] **Step 8: Update yi registry entry**

Add `tensor_naming: TensorNamingScheme::Gguf` to yi definition (lines 74-81).

- [ ] **Step 9: Verify code compiles**

Run: `cargo check --lib 2>&1 | grep -E "error|Finished" | head -5`

Expected: "Finished" or errors only about missing tensor_naming in other registry entries (will fix next)

- [ ] **Step 10: Commit registry changes**

```bash
git add src/config.rs
git commit -m "feat(config): add TensorNamingScheme and TensorNameRegistry

- Add TensorNamingScheme enum (Gguf, HuggingFace)
- Add TensorName enum with 15 semantic tensor names
- Add TensorNameRegistry with format templates
- Extend ModelTraits with tensor_naming field
- Separate qwen3 (HuggingFace) from qwen2 (Gguf)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Add tensor_registry to ModelConfig

**Files:**
- Modify: `src/config.rs:161-185` (ModelConfig struct definition)

- [ ] **Step 1: Add tensor_registry field to ModelConfig struct**

Add after line 184 (after architecture field):

```rust
pub struct ModelConfig {
    // ... existing fields ...
    pub architecture: String,

    /// Tensor name registry for this model
    pub tensor_registry: TensorNameRegistry,
}
```

- [ ] **Step 2: Update ModelConfig::from_gguf to create tensor_registry**

Modify lines 220-235 in from_gguf function to include tensor_registry creation:

```rust
let config = Self {
    num_layers,
    hidden_size,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    vocab_size,
    max_seq_len,
    rms_norm_eps,
    rope_theta,
    rope_neox: traits.rope_style == RopeStyle::NeoX,
    use_attention_bias: traits.use_attention_bias,
    attention_layout: traits.attention_layout,
    architecture: meta.architecture.clone(),
    tensor_registry: TensorNameRegistry::from_scheme(&traits.tensor_naming),
};
```

Note: Add the tensor_registry field at the end of the struct initialization.

- [ ] **Step 3: Verify code compiles**

Run: `cargo check --lib 2>&1 | grep -E "error|Finished" | head -5`

Expected: "Finished" or no compilation errors

- [ ] **Step 4: Commit ModelConfig changes**

```bash
git add src/config.rs
git commit -m "feat(config): add tensor_registry to ModelConfig

- Add tensor_registry field to ModelConfig struct
- Initialize from_gguf to create registry from ModelTraits
- Enables metadata-driven tensor name resolution

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Update CPU weight loading to use TensorNameRegistry

**Files:**
- Modify: `src/cpu/weights.rs:163-217` (CpuLayerWeights::load method)

- [ ] **Step 1: Remove hardcoded p closure**

Delete lines 165-165 (the `let p = |s:&str| format!("blk.{}.{}", layer, s);` line)

- [ ] **Step 2: Replace tensor name resolution with registry calls**

Replace the hardcoded tensor name calls (lines 185-209) with registry calls:

Find and replace:

```rust
// OLD:
let (attn_q, attn_q_meta) = load_weight(&p("attn_q.weight"))?;
let (attn_k, attn_k_meta) = load_weight(&p("attn_k.weight"))?;
let (attn_v, attn_v_meta) = load_weight(&p("attn_v.weight"))?;
let (attn_o, attn_o_meta) = load_weight(&p("attn_output.weight"))?;
let (ffn_gate, ffn_gate_meta) = load_weight(&p("ffn_gate.weight"))?;
let (ffn_up, ffn_up_meta) = load_weight(&p("ffn_up.weight"))?;
let (ffn_down, ffn_down_meta) = load_weight(&p("ffn_down.weight"))?;

// NEW:
use crate::config::TensorName;

let (attn_q, attn_q_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
let (attn_k, attn_k_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
let (attn_v, attn_v_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
let (attn_o, attn_o_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnOutput, layer))?;
let (ffn_gate, ffn_gate_meta) = load_weight(&config.tensor_registry.resolve(TensorName::FfnGate, layer))?;
let (ffn_up, ffn_up_meta) = load_weight(&config.tensor_registry.resolve(TensorName::FfnUp, layer))?;
let (ffn_down, ffn_down_meta) = load_weight(&config.tensor_registry.resolve(TensorName::FfnDown, layer))?;
```

- [ ] **Step 3: Replace normalization tensor calls**

Replace lines 196 and 208:

```rust
// OLD:
attn_norm: copy_f32(file, &p("attn_norm.weight"))?,
// ...
ffn_norm: copy_f32(file, &p("ffn_norm.weight"))?,

// NEW:
use crate::config::TensorName;

attn_norm: copy_f32(file, &config.tensor_registry.resolve(TensorName::AttnNorm, layer))?,
// ...
ffn_norm: copy_f32(file, &config.tensor_registry.resolve(TensorName::FfnNorm, layer))?,
```

- [ ] **Step 4: Replace bias tensor calls with resolve_optional**

Replace lines 199-205:

```rust
// OLD:
attn_q_bias: optional_f32(copy_tensor_optional(file, &p("attn_q.bias"))?),
attn_k_bias: optional_f32(copy_tensor_optional(file, &p("attn_k.bias"))?),
attn_v_bias: optional_f32(copy_tensor_optional(file, &p("attn_v.bias"))?),

// NEW:
attn_q_bias: match config.tensor_registry.resolve_optional(TensorName::AttnQBias, layer) {
    Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
    None => None,
},
attn_k_bias: match config.tensor_registry.resolve_optional(TensorName::AttnKBias, layer) {
    Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
    None => None,
},
attn_v_bias: match config.tensor_registry.resolve_optional(TensorName::AttnVBias, layer) {
    Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
    None => None,
},
```

- [ ] **Step 5: Update CpuModelWeights::load for embedding tensors**

Find the CpuModelWeights::load function (around line 247) and update embedding tensor names:

```rust
// OLD lines for token_emb and lm_head:
let (token_emb, token_emb_meta) = load_weight_with_meta(file, "token_embd.weight", false)?;
let (lm_head, lm_head_meta) = load_weight_with_meta(file, "output.weight", lm_head_tied, false)?;

// NEW:
use crate::config::TensorName;

let token_emb_name = config.tensor_registry.resolve(TensorName::TokenEmb, 0);
let (token_emb, token_emb_meta) = load_weight_with_meta(file, &token_emb_name, false)?;

let lm_head_name = config.tensor_registry.resolve(TensorName::LmHead, 0);
let (lm_head, lm_head_meta) = load_weight_with_meta(file, &lm_head_name, lm_head_tied, false)?;
```

- [ ] **Step 6: Update output_norm tensor name**

Find output_norm loading and update:

```rust
// OLD:
output_norm: copy_f32(file, "output_norm.weight")?,

// NEW:
use crate::config::TensorName;

let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
output_norm: copy_f32(file, &output_norm_name)?,
```

- [ ] **Step 7: Verify code compiles**

Run: `cargo check --lib 2>&1 | grep -E "error|warning.*weights" | head -10`

Expected: No errors, may have warnings

- [ ] **Step 8: Run tests to verify no regressions**

Run: `cargo test --lib 2>&1 | tail -10`

Expected: "test result: ok. X passed; 0 failed"

- [ ] **Step 9: Commit CPU weight loading changes**

```bash
git add src/cpu/weights.rs
git commit -m "feat(cpu): use TensorNameRegistry for weight loading

- Replace hardcoded blk.N. prefix with registry.resolve()
- Support both GGUF and HuggingFace tensor naming schemes
- Use resolve_optional for optional bias tensors
- Update embedding and output norm tensor names

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Update GPU weight loading

**Files:**
- Modify: `src/gpu/weights.rs:256-351` (GpuLayerWeights::load method)
- Modify: `src/gpu/weights.rs:396-433` (GpuModelWeights::load method, if embedding tensors loaded there)

- [ ] **Step 1: Remove hardcoded p closure**

Delete line 256: `let p = |s: &str| format!("blk.{}.{}", layer, s);`

- [ ] **Step 2: Add TensorName import at top of file**

Add to line 10 (after ModelConfig import):

```rust
use crate::config::{TensorName, TensorNameRegistry};
```

- [ ] **Step 3: Replace weight tensor calls with registry**

Replace lines 318-328 with:

```rust
let (attn_q, attn_q_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
let attn_q_bias = load_f32_opt(&config.tensor_registry.resolve_optional(TensorName::AttnQBias, layer))?;
let (attn_k, attn_k_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
let attn_k_bias = load_f32_opt(&config.tensor_registry.resolve_optional(TensorName::AttnKBias, layer))?;
let (attn_v, attn_v_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
let attn_v_bias = load_f32_opt(&config.tensor_registry.resolve_optional(TensorName::AttnVBias, layer))?;
let (attn_o, attn_o_meta) = load_weight(&config.tensor_registry.resolve(TensorName::AttnOutput, layer))?;
let ffn_norm = load_f32(&config.tensor_registry.resolve(TensorName::FfnNorm, layer))?;
let (ffn_gate, ffn_gate_meta) = load_weight(&config.tensor_registry.resolve(TensorName::FfnGate, layer))?;
let (ffn_up, ffn_up_meta) = load_weight(&config.tensor_registry.resolve(TensorName::FfnUp, layer))?;
let (ffn_down, ffn_down_meta) = load_weight(&config.tensor_registry.resolve(TensorName::FfnDown, layer))?;
```

- [ ] **Step 4: Replace normalization tensor calls**

Replace lines 317 and 325:

```rust
let attn_norm = load_f32(&config.tensor_registry.resolve(TensorName::AttnNorm, layer))?;
// ...
let ffn_norm = load_f32(&config.tensor_registry.resolve(TensorName::FfnNorm, layer))?;
```

- [ ] **Step 5: Check if GpuModelWeights::load loads embedding tensors**

Search for token_emb.weight in GpuModelWeights::load (around line 400+).

If embedding tensors are loaded there, update:

```rust
let token_emb_name = config.tensor_registry.resolve(TensorName::TokenEmb, 0);
let (token_emb, token_emb_meta) = load_weight(&token_emb_name)?;

let lm_head_name = config.tensor_registry.resolve(TensorName::LmHead, 0);
let (lm_head, lm_head_meta) = load_weight(&lm_head_name)?;

let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
output_norm: load_f32(&output_norm_name)?,
```

- [ ] **Step 6: Verify code compiles**

Run: `cargo check --lib 2>&1 | grep -E "error|warning.*gpu" | head -10`

Expected: No errors, may have warnings

- [ ] **Step 7: Run GPU tests if any**

Run: `cargo test --lib gpu:: 2>&1 | tail -10`

Expected: Tests pass (if any exist)

- [ ] **Step 8: Commit GPU weight loading changes**

```bash
git add src/gpu/weights.rs
git commit -m "feat(gpu): use TensorNameRegistry for weight loading

- Replace hardcoded blk.N. prefix with registry.resolve()
- Support both GGUF and HuggingFace tensor naming schemes
- Use resolve_optional for optional bias tensors
- Update embedding and output norm tensor names

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 9: If file doesn't exist, mark task complete**

Run: `ls -la src/gpu/weights.rs 2>&1`

If no file exists, no changes needed.

---

## Task 7: Add unit tests for TensorNameRegistry

**Files:**
- Modify: `src/config.rs:432-580` (tests module)

- [ ] **Step 1: Add tests for TensorNameRegistry**

Add before the closing brace of the tests module (after line 579):

```rust
// TensorNameRegistry tests
#[test]
fn gguf_scheme_formats_correctly() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
    assert_eq!(registry.resolve(TensorName::FfnGate, 0), "blk.0.ffn_gate.weight");
    assert_eq!(registry.resolve(TensorName::AttnQ, 5), "blk.5.attn_q.weight");
    assert_eq!(registry.resolve(TensorName::AttnNorm, 2), "blk.2.attn_norm.weight");
}

#[test]
fn huggingface_scheme_formats_correctly() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
    assert_eq!(registry.resolve(TensorName::FfnGate, 0), "model.layers.0.mlp.gate_proj.weight");
    assert_eq!(registry.resolve(TensorName::AttnQ, 5), "model.layers.5.self_attn.q_proj.weight");
    assert_eq!(registry.resolve(TensorName::AttnNorm, 2), "model.layers.2.input_layernorm.weight");
}

#[test]
fn gguf_scheme_maps_all_tensors() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
    let all_names = [
        TensorName::AttnQ, TensorName::AttnK, TensorName::AttnV, TensorName::AttnOutput,
        TensorName::AttnQBias, TensorName::AttnKBias, TensorName::AttnVBias,
        TensorName::FfnGate, TensorName::FfnUp, TensorName::FfnDown,
        TensorName::AttnNorm, TensorName::FfnNorm,
        TensorName::TokenEmb, TensorName::LmHead, TensorName::OutputNorm,
    ];
    for name in all_names {
        let result = registry.resolve(*name, 0);
        assert!(!result.is_empty(), "TensorName::{:?} should produce non-empty string", name);
    }
}

#[test]
fn huggingface_scheme_maps_all_tensors() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
    let all_names = [
        TensorName::AttnQ, TensorName::AttnK, TensorName::AttnV, TensorName::AttnOutput,
        TensorName::AttnQBias, TensorName::AttnKBias, TensorName::AttnVBias,
        TensorName::FfnGate, TensorName::FfnUp, TensorName::FfnDown,
        TensorName::AttnNorm, TensorName::FfnNorm,
    ];
    for name in all_names {
        let result = registry.resolve(*name, 0);
        assert!(!result.is_empty(), "TensorName::{:?} should produce non-empty string", name);
    }
}

#[test]
fn qwen2_uses_gguf_scheme() {
    let traits = ModelTraits::for_arch("qwen2");
    assert_eq!(traits.tensor_naming, TensorNamingScheme::Gguf);
}

#[test]
fn qwen3_uses_huggingface_scheme() {
    let traits = ModelTraits::for_arch("qwen3");
    assert_eq!(traits.tensor_naming, TensorNamingScheme::HuggingFace);
}

#[test]
fn unknown_arch_falls_back_to_gguf() {
    let traits = ModelTraits::for_arch("unknown_future_arch");
    assert_eq!(traits.tensor_naming, TensorNamingScheme::Gguf);
}

#[test]
fn optional_bias_resolution() {
    let gguf = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
    assert!(gguf.resolve_optional(TensorName::AttnQBias, 0).is_some());
}

#[test]
fn layer_number_formatting() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
    assert!(registry.resolve(TensorName::FfnGate, 0).contains(".0."));
    assert!(registry.resolve(TensorName::FfnGate, 99).contains(".99."));
    assert!(registry.resolve(TensorName::FfnGate, 999).contains(".999."));
}

#[test]
fn non_layer_tensors_format_correctly() {
    let gguf = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
    assert_eq!(gguf.resolve(TensorName::TokenEmb, 0), "token_embd.weight");
    assert_eq!(gguf.resolve(TensorName::OutputNorm, 0), "output_norm.weight");

    let hf = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
    assert_eq!(hf.resolve(TensorName::TokenEmb, 0), "model.embed_tokens.weight");
    assert_eq!(hf.resolve(TensorName::OutputNorm, 0), "model.norm.weight");
}
```

- [ ] **Step 2: Run new tests**

Run: `cargo test --lib config::tests::gguf_scheme config::tests::huggingface 2>&1 | tail -15`

Expected: All tests pass

- [ ] **Step 3: Run all config tests**

Run: `cargo test --lib config::tests 2>&1 | tail -10`

Expected: "test result: ok. X passed; 0 failed" (X includes new + existing tests)

- [ ] **Step 4: Commit tests**

```bash
git add src/config.rs
git commit -m "test(config): add TensorNameRegistry unit tests

- Test GGUF and HuggingFace naming schemes
- Test all tensor names have valid mappings
- Test qwen2/qwen3 scheme detection
- Test optional bias resolution
- Test layer number formatting

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Run full test suite and verify regressions

**Files:**
- None (verification only)

- [ ] **Step 1: Run full library test suite**

Run: `cargo test --lib 2>&1 | tail -15`

Expected: "test result: ok. 124 passed; 0 failed; 0 ignored"

Note: All 124 existing tests should still pass.

- [ ] **Step 2: Verify compilation in release mode**

Run: `cargo build --release --lib 2>&1 | grep -E "Compiling rocmforge|Finished" | tail -3`

Expected: "Finished release profile [optimized]"

- [ ] **Step 3: Run doc tests if any**

Run: `cargo test --doc 2>&1 | tail -5`

Expected: No errors

---

## Task 9: Integration test with real qwen3 model

**Files:**
- Test: Manual verification with actual model

- [ ] **Step 1: Test with actual qwen3 model**

Run: `cargo run --release --example benchmark_real_model -- --model "*qwen3*" --iterations 1 --tokens 2 2>&1 | grep -A5 "qwen3"`

Expected: Model loads without "tensor not found" errors, shows timing results

- [ ] **Step 2: Verify qwen2 still works**

Run: `cargo run --release --example benchmark_real_model -- --model "*qwen2*" --iterations 1 --tokens 2 2>&1 | grep -A5 "qwen2"`

Expected: qwen2 models still load and run correctly (no regression)

---

## Task 10: Final cleanup and documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-03-27-tensorname-registry-design.md`
- Update: Implementation notes if needed

- [ ] **Step 1: Update design spec with implementation notes**

Add any lessons learned or deviations from the original design to the spec.

- [ ] **Step 2: Verify git history**

Run: `git log --oneline -5`

Expected: Clean commit history with descriptive messages

- [ ] **Step 3: Final verification**

Run: `echo "TensorNameRegistry Implementation Complete" && cargo test --lib 2>&1 | tail -3`

Expected: All tests pass, ready for use

---

## Success Criteria Verification

After completing all tasks, verify:

1. ✅ Qwen3 models load without `tensor not found` errors
2. ✅ All 124 existing model tests still pass
3. ✅ New architecture added to registry enables that model family
4. ✅ Weight loading uses `tensor_registry.resolve()`
5. ✅ All 15 `TensorName` enum variants have complete template mappings

Run: `cargo test --lib 2>&1 | grep "test result:"`

Expected: "test result: ok. 130+ passed; 0 failed" (124 original + 6+ new tests)
