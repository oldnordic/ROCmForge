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
pub struct TensorNameRegistry {
    scheme: TensorNamingScheme,
    templates: HashMap<TensorName, String>,
}

pub enum TensorName {
    AttnQ, AttnK, AttnV, AttnOutput,
    FfnGate, FfnUp, FfnDown,
    AttnNorm, FfnNorm,
    TokenEmb, OutputNorm, LmHead,
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
}
```

Templates:
- **Gguf**: `blk.{}.ffn_gate.weight`
- **HuggingFace**: `model.layers.{}.mlp.gate_proj.weight`

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

**Fail fast:** Unknown architectures fall back to DEFAULT_TRAITS (Gguf naming).

**Debugging aid:**
```rust
impl TensorNameRegistry {
    pub fn expected_tensors(&self, layer: usize) -> Vec<String> {
        // Return expected names for debugging
    }
}
```

### Testing

#### Unit Tests

```rust
#[test]
fn gguf_scheme_formats_correctly() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf);
    assert_eq!(registry.resolve(TensorName::FfnGate, 0), "blk.0.ffn_gate.weight");
    assert_eq!(registry.resolve(TensorName::AttnQ, 5), "blk.5.attn_q.weight");
}

#[test]
fn huggingface_scheme_formats_correctly() {
    let registry = TensorNameRegistry::from_scheme(&TensorNamingScheme::HuggingFace);
    assert_eq!(registry.resolve(TensorName::FfnGate, 0), "model.layers.0.mlp.gate_proj.weight");
    assert_eq!(registry.resolve(TensorName::AttnQ, 5), "model.layers.5.self_attn.q_proj.weight");
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
```

#### Integration Test

Load actual qwen3 GGUF file, verify weight loading succeeds.

### Files to Modify

1. **src/config.rs** - Add `TensorNamingScheme`, `TensorNameRegistry`, extend `ModelTraits`
2. **src/cpu/weights.rs** - Use `tensor_registry.resolve()` instead of hardcoded format
3. **src/gpu/weights.rs** - Same as CPU
4. **Remove hardcoded detection** from `infer_intermediate_size()` if present

---

## Success Criteria

1. Qwen3 models load without `tensor not found` errors
2. All existing model tests still pass (Qwen2, LLaMA, etc.)
3. New architectures can be added by updating registry only
4. Weight loading code has no hardcoded tensor name patterns

---

## Open Questions

1. **Are there other tensor naming schemes** beyond Gguf and HuggingFace we should support now?
2. **Should `infer_intermediate_size()`** also use the registry instead of its current candidate list?
