# TensorNameRegistry Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this spec task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable metadata-driven tensor name resolution to support Qwen3 and future models without hardcoding assumptions

**Architecture:** Extend `ModelTraits` with `TensorNamingScheme` enum; create `TensorNameRegistry` that formats semantic tensor names based on architecture metadata

**Tech Stack:** Rust, existing GGUF loader, HashMap for name mapping

---

## Problem

Current weight loading code hardcodes `blk.N.` tensor name prefix:
```rust
let p = |s: &str| format!("blk.{}.{}", layer, s);
```

Qwen3 models use HuggingFace-style names (`model.layers.N.mlp.gate_proj.weight`), causing `tensor not found` errors.

The system should use GGUF metadata to determine tensor naming schemes, not hardcode assumptions.

---

## Solution

### Architecture

**Tensor naming is a property of architecture metadata:**

1. **Extend `ModelTraits` with `TensorNamingScheme`**
2. **Create `TensorNameRegistry` as a formatter**
3. **Integrate into `ModelConfig`**
4. **Update weight loading to use semantic names**

### Components

#### 1. TensorNamingScheme Enum (config.rs)

```rust
pub enum TensorNamingScheme {
    Gguf,           // blk.N.ffn_gate.weight
    HuggingFace,    // model.layers.N.mlp.gate_proj.weight
}
```

#### 2. Extend ModelTraits

```rust
pub struct ModelTraits {
    pub rope_style: RopeStyle,
    pub attention_layout: AttentionLayout,
    pub use_attention_bias: bool,
    pub default_rope_theta: f32,
    pub default_norm_eps: f32,
    pub tensor_naming: TensorNamingScheme,  // NEW
}
```

Registry entries:
```rust
let qwen2 = ModelTraits {
    tensor_naming: TensorNamingScheme::Gguf,
    // ... existing fields
};

let qwen3 = ModelTraits {
    tensor_naming: TensorNamingScheme::HuggingFace,
    // ... existing fields
};
```

#### 3. TensorNameRegistry (config.rs)

```rust
#[derive(Debug, Clone)]
pub struct TensorNameRegistry {
    scheme: TensorNamingScheme,
    templates: HashMap<TensorName, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorName {
    // Attention weights
    AttnQ, AttnK, AttnV, AttnOutput,
    // Attention biases (optional)
    AttnQBias, AttnKBias, AttnVBias,
    // FFN weights
    FfnGate, FfnUp, FfnDown,
    // Normalization
    AttnNorm, FfnNorm,
    // Embeddings
    TokenEmb, LmHead,
    // Output norm
    OutputNorm,
}

impl TensorNameRegistry {
    pub fn from_scheme(scheme: &TensorNamingScheme) -> Self {
        match scheme {
            TensorNamingScheme::Gguf => Self::gguf_format(),
            TensorNamingScheme::HuggingFace => Self::huggingface_format(),
        }
    }

    pub fn resolve(&self, name: TensorName, layer: usize) -> String {
        let template = self.templates.get(&name).unwrap();
        template.replace("{}", &layer.to_string())
    }

    pub fn resolve_optional(&self, name: TensorName, layer: usize) -> Option<String> {
        // For optional tensors like bias - returns None if not supported by scheme
        self.templates.get(&name).cloned()
    }
}
```

**Complete template mappings:**

| TensorName | Gguf | HuggingFace |
|-----------|------|-------------|
| AttnQ | `blk.{}.attn_q.weight` | `model.layers.{}.self_attn.q_proj.weight` |
| AttnK | `blk.{}.attn_k.weight` | `model.layers.{}.self_attn.k_proj.weight` |
| AttnV | `blk.{}.attn_v.weight` | `model.layers.{}.self_attn.v_proj.weight` |
| AttnOutput | `blk.{}.attn_output.weight` | `model.layers.{}.self_attn.o_proj.weight` |
| AttnQBias | `blk.{}.attn_q.bias` | `model.layers.{}.self_attn.q_proj.bias` |
| AttnKBias | `blk.{}.attn_k.bias` | `model.layers.{}.self_attn.k_proj.bias` |
| AttnVBias | `blk.{}.attn_v.bias` | `model.layers.{}.self_attn.v_proj.bias` |
| FfnGate | `blk.{}.ffn_gate.weight` | `model.layers.{}.mlp.gate_proj.weight` |
| FfnUp | `blk.{}.ffn_up.weight` | `model.layers.{}.mlp.up_proj.weight` |
| FfnDown | `blk.{}.ffn_down.weight` | `model.layers.{}.mlp.down_proj.weight` |
| AttnNorm | `blk.{}.attn_norm.weight` | `model.layers.{}.input_layernorm.weight` |
| FfnNorm | `blk.{}.ffn_norm.weight` | `model.layers.{}.post_attention_layernorm.weight` |
| TokenEmb | `token_embd.weight` | `model.embed_tokens.weight` |
| LmHead | `output.weight` | `lm_head.weight` |
| OutputNorm | `output_norm.weight` | `model.norm.weight` |

#### 4. ModelConfig Integration

```rust
pub struct ModelConfig {
    // ... existing fields ...
    pub tensor_registry: TensorNameRegistry,  // NEW
}

impl ModelConfig {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ConfigError> {
        let traits = ModelTraits::for_arch(&meta.architecture);
        let tensor_registry = TensorNameRegistry::from_scheme(&traits.tensor_naming);

        Ok(Self {
            // ... existing fields ...
            tensor_registry,
        })
    }
}
```

#### 5. Weight Loading Changes (cpu/weights.rs, gpu/weights.rs)

**Before:**
```rust
let p = |s: &str| format!("blk.{}.{}", layer, s);
let (ffn_gate, _) = load_weight(&p("ffn_gate.weight"))?;
```

**After:**
```rust
let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
let (ffn_gate, _) = load_weight(&ffn_gate_name)?;
```

### Data Flow

```
GGUF file → metadata.architecture → ModelTraits → TensorNamingScheme
                                                          ↓
                                                   TensorNameRegistry
                                                          ↓
                                        weight_loader.resolve(TensorName, layer)
                                                          ↓
                                                   actual tensor name
```

### Error Handling

**Clear error messages:**
```rust
WeightError::TensorNotFound {
    name: "model.layers.0.mlp.gate_proj.weight",
    expected_by: "FfnGate",
    layer: 0,
    scheme: "HuggingFace"
}
```

**Optional tensor handling:**
```rust
// For bias tensors that may not exist
let attn_q_bias_name = config.tensor_registry.resolve_optional(TensorName::AttnQBias, layer);
let attn_q_bias = match attn_q_bias_name {
    Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
    None => None,
};
```

**Fail fast:** Unknown architectures fall back to DEFAULT_TRAITS (Gguf naming).

**Debugging aid:**
```rust
impl TensorNameRegistry {
    pub fn expected_tensors(&self, layer: usize) -> Vec<String> {
        // Return expected names for debugging
    }

    pub fn list_all_names(&self) -> Vec<&'static str> {
        // List all TensorName variants supported
    }
}
```

**Scope decisions:**
- **IN SCOPE:** Weight loading in `cpu/weights.rs` and `gpu/weights.rs`
- **OUT OF SCOPE:** `infer_intermediate_size()` keeps its candidate list (already works)
- **OUT OF SCOPE:** Transpose logic in `cpu/transpose.rs` (weight name checks remain as-is)

### Testing

#### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Test each naming scheme produces correct names
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

    // Test all tensor names are mapped
    #[test]
    fn gguf_scheme_maps_all_tensors() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        let all_names = [
            TensorName::AttnQ, TensorName::AttnK, TensorName::AttnV, TensorName::AttnOutput,
            TensorName::FfnGate, TensorName::FfnUp, TensorName::FfnDown,
            TensorName::AttnNorm, TensorName::FfnNorm,
            TensorName::TokenEmb, TensorName::LmHead, TensorName::OutputNorm,
        ];
        for name in all_names {
            let result = registry.resolve(name, 0);
            assert!(!result.is_empty(), "TensorName::{:?} should produce non-empty string", name);
        }
    }

    // Test huggingface maps all tensors
    #[test]
    fn huggingface_scheme_maps_all_tensors() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
        let all_names = [
            TensorName::AttnQ, TensorName::AttnK, TensorName::AttnV, TensorName::AttnOutput,
            TensorName::FfnGate, TensorName::FfnUp, TensorName::FfnDown,
            TensorName::AttnNorm, TensorName::FfnNorm,
        ];
        for name in all_names {
            let result = registry.resolve(name, 0);
            assert!(!result.is_empty(), "TensorName::{:?} should produce non-empty string", name);
        }
    }

    // Test architecture traits have correct schemes
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

    // Test unknown architectures fall back to gguf
    #[test]
    fn unknown_arch_falls_back_to_gguf() {
        let traits = ModelTraits::for_arch("unknown_future_arch");
        assert_eq!(traits.tensor_naming, TensorNamingScheme::Gguf);
    }

    // Test optional tensor resolution
    #[test]
    fn optional_bias_resolution() {
        let gguf = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert!(gguf.resolve_optional(TensorName::AttnQBias, 0).is_some());

        // If a scheme doesn't support bias tensors
        // assert!(scheme.resolve_optional(TensorName::AttnQBias, 0).is_none());
    }

    // Test layer number formatting
    #[test]
    fn layer_number_formatting() {
        let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert!(registry.resolve(TensorName::FfnGate, 0).contains(".0."));
        assert!(registry.resolve(TensorName::FfnGate, 99).contains(".99."));
        assert!(registry.resolve(TensorName::FfnGate, 999).contains(".999."));
    }

    // Test non-layer tensors (no layer number)
    #[test]
    fn non_layer_tensors_format_correctly() {
        let gguf = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
        assert_eq!(gguf.resolve(TensorName::TokenEmb, 0), "token_embd.weight");
        assert_eq!(gguf.resolve(TensorName::OutputNorm, 0), "output_norm.weight");

        let hf = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
        assert_eq!(hf.resolve(TensorName::TokenEmb, 0), "model.embed_tokens.weight");
        assert_eq!(hf.resolve(TensorName::OutputNorm, 0), "model.norm.weight");
    }
}
```

#### Integration Tests

```rust
#[test]
#[ignore = "requires qwen3 model file"]
fn load_qwen3_model_successfully() {
    let path = "/path/to/qwen3-30b-q4_0.gguf";
    let file = GgufFile::open(path).unwrap();
    let config = ModelConfig::from_gguf(&file).unwrap();

    // Should use HuggingFace scheme
    assert_eq!(config.tensor_registry.scheme, TensorNamingScheme::HuggingFace);

    // Should successfully load weights
    let weights = CpuModelWeights::load(&file, &config);
    assert!(weights.is_ok());
}

#[test]
fn load_qwen2_model_still_works() {
    // Verify existing qwen2 support not broken
}
```

**Note:** FusedQKV support (Phi3, Falcon, GLM) is **out of scope** for this initial implementation. Current `AttentionLayout::FusedQkv` models will continue to work with their existing weight loading code. Future work can extend `TensorNameRegistry` with `FusedQkv` naming templates.

### Files to Modify

1. **src/config.rs** - Add `TensorNamingScheme`, `TensorNameRegistry`, extend `ModelTraits`
2. **src/cpu/weights.rs** - Use `tensor_registry.resolve()` instead of hardcoded format
3. **src/gpu/weights.rs** - Same as CPU
4. **Remove hardcoded detection** from `infer_intermediate_size()` if present

---

## Success Criteria

1. ✅ Qwen3 models load without `tensor not found` errors
2. ✅ All 124 existing model tests still pass (Qwen2, LLaMA, etc.)
3. ✅ New architecture added to registry enables that model family
4. ✅ Weight loading in `cpu/weights.rs` and `gpu/weights.rs` uses `tensor_registry.resolve()`
5. ✅ All 15 `TensorName` enum variants have complete template mappings for both schemes

## Out of Scope (Future Work)

1. **FusedQKV tensor naming** - Phi3, Falcon, GLM use fused QKV (`attn_qkv.weight`)
   - Current: These models continue working with existing code
   - Future: Add `FusedQkv` variant to `TensorName` enum

2. **Refactoring `infer_intermediate_size()`** - Currently uses hardcoded candidate list
   - Current: Keep as-is (already works for both naming schemes)
   - Future: Could use registry for cleaner code

3. **Refactoring transpose logic** - `cpu/transpose.rs` checks weight names
   - Current: Keep hardcoded weight name checks
   - Future: Could use semantic matching

## Open Questions

None - scope is clearly defined for initial implementation.

---

## Implementation Notes

**Why Gguf is the fallback:**
- GGUF format was originally designed with "blk.N." prefix
- Most GGUF-converted models use this convention
- Safer default than assuming HuggingFace format

**Why HuggingFace for Qwen3:**
- Qwen3 models released directly in HuggingFace format
- GGUF conversion preserves original tensor names
- Architecture string "qwen3" indicates this difference

**Registry in config.rs:**
- Keeps all architecture-specific behavior in one place
- Consistent with existing `ModelTraits` pattern
- Available to both CPU and GPU weight loaders
