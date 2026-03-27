//! Model configuration - all hyperparameters needed for inference.
//!
//! Two concerns are strictly separated:
//!   - **Values** (sizes, dimensions, epsilons): always read from GGUF metadata.
//!   - **Behaviors** (RoPE style, attention layout): hardcoded per architecture
//!     in the `ModelTraits` registry below.
//!
//! `ModelConfig::from_gguf()` combines both into one validated struct.

use crate::loader::{GgufFile, LoadError};
use std::collections::HashMap;
use std::sync::OnceLock;

// ── Architecture traits registry ──────────────────────────────────────────────

/// How RoPE rotations are applied to head dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Consecutive pairs: (0,1),(2,3),... - LLaMA, Mistral
    Normal,
    /// Split-half pairs: (0,head_dim/2),(1,head_dim/2+1),... - Qwen2, GPT-NeoX
    NeoX,
}

/// How the Q/K/V weight tensors are laid out in the GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayout {
    /// Separate `blk.N.attn_q`, `blk.N.attn_k`, `blk.N.attn_v`
    SplitQkv,
    /// Single fused tensor `blk.N.attn_qkv` (Phi3, Falcon, etc.)
    FusedQkv,
}

/// Tensor naming convention used by GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorNamingScheme {
    /// GGUF-style: `blk.N.ffn_gate.weight`
    Gguf,
    /// HuggingFace-style: `model.layers.N.mlp.gate_proj.weight`
    HuggingFace,
}

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

/// Hardcoded structural/behavioral differences between architecture families.
/// All numeric *values* (sizes, epsilons) still come from GGUF metadata.
#[derive(Debug, Clone)]
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

static REGISTRY: OnceLock<HashMap<&'static str, ModelTraits>> = OnceLock::new();

/// Default traits for unknown architectures (LLaMA-compatible)
static DEFAULT_TRAITS: ModelTraits = ModelTraits {
    rope_style: RopeStyle::Normal,
    attention_layout: AttentionLayout::SplitQkv,
    use_attention_bias: false,
    default_rope_theta: 10000.0,
    default_norm_eps: 1e-5,
    tensor_naming: TensorNamingScheme::Gguf,
};

fn registry() -> &'static HashMap<&'static str, ModelTraits> {
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();

        // LLaMA family - consecutive RoPE, no bias, split QKV
        let llama = ModelTraits {
            rope_style: RopeStyle::Normal,
            attention_layout: AttentionLayout::SplitQkv,
            use_attention_bias: false,
            default_rope_theta: 10000.0,
            default_norm_eps: 1e-5,
            tensor_naming: TensorNamingScheme::Gguf,
        };
        for arch in &["llama", "mistral", "baichuan", "internlm2", "deepseek"] {
            m.insert(*arch, llama.clone());
        }
        m.insert(
            "yi",
            ModelTraits {
                default_norm_eps: 1e-6,
                ..llama.clone()
            },
        );
        m.insert("mixtral", llama.clone()); // MoE variant, same behaviors

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
        // Legacy Qwen1: lower rope theta, no QK norm
        m.insert(
            "qwen",
            ModelTraits {
                default_rope_theta: 10000.0,
                use_attention_bias: true,
                ..qwen2.clone()
            },
        );

        // Phi family
        m.insert(
            "phi3",
            ModelTraits {
                rope_style: RopeStyle::Normal,
                attention_layout: AttentionLayout::FusedQkv,
                use_attention_bias: false,
                default_rope_theta: 10000.0,
                default_norm_eps: 1e-5,
                tensor_naming: TensorNamingScheme::Gguf,
            },
        );
        m.insert("phi2", llama.clone());

        // Gemma family
        let gemma = ModelTraits {
            rope_style: RopeStyle::Normal,
            attention_layout: AttentionLayout::SplitQkv,
            use_attention_bias: false,
            default_rope_theta: 10000.0,
            default_norm_eps: 1e-6,
            tensor_naming: TensorNamingScheme::Gguf,
        };
        for arch in &["gemma", "gemma2", "gemma3"] {
            m.insert(*arch, gemma.clone());
        }

        // GLM
        m.insert(
            "glm",
            ModelTraits {
                rope_style: RopeStyle::NeoX,
                attention_layout: AttentionLayout::FusedQkv,
                use_attention_bias: false,
                default_rope_theta: 10000.0,
                default_norm_eps: 1e-5,
                tensor_naming: TensorNamingScheme::Gguf,
            },
        );

        m
    })
}

impl ModelTraits {
    /// Look up traits for an architecture string, falling back to LLaMA defaults
    /// for unknown architectures rather than failing.
    pub fn for_arch(arch: &str) -> &'static ModelTraits {
        registry().get(arch).unwrap_or(&DEFAULT_TRAITS)
    }
}

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// All hyperparameters needed to run inference.
///
/// Values come from GGUF metadata; behaviors come from the traits registry.
/// `vocab_size` comes from `tokenizer_data.tokens.len()` - not GGUF metadata,
/// which returns 0 for Qwen2.5.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    // Transformer dimensions
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    /// KV heads for GQA/MQA. Equals `num_heads` for standard MHA.
    pub num_kv_heads: usize,
    /// Dimension of each attention head. Should equal hidden_size / num_heads.
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,

    // Numerical parameters
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    // Behavioral flags (from ModelTraits)
    pub rope_neox: bool,
    pub use_attention_bias: bool,
    pub attention_layout: AttentionLayout,

    /// The raw architecture string from GGUF (e.g. "qwen2", "llama")
    pub architecture: String,

    /// Tensor name registry for this model
    pub tensor_registry: TensorNameRegistry,
}

impl ModelConfig {
    /// Build `ModelConfig` from an open GGUF file.
    ///
    /// `vocab_size` is taken from `tokenizer_data.tokens.len()` because GGUF
    /// metadata `vocab_size` key returns 0 for Qwen2.5 and similar models.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ConfigError> {
        let meta = &file.metadata;
        let traits = ModelTraits::for_arch(&meta.architecture);

        // CRITICAL: vocab_size from tokenizer tokens length, NOT metadata key (per D-05)
        let vocab_size = file.tokenizer_data().tokens.len();

        // All dimensions from metadata
        let num_layers = meta.block_count();
        let hidden_size = meta.embedding_length();
        let num_heads = meta.attention_head_count();
        let num_kv_heads = meta.attention_head_count_kv().unwrap_or(num_heads);
        let head_dim = meta.head_dim();
        let rms_norm_eps = meta.rms_norm_eps(traits.default_norm_eps);
        let rope_theta = meta.rope_freq_base(traits.default_rope_theta);
        let max_seq_len = meta.context_length();

        // intermediate_size: try metadata first, then tensor shape inference (per D-04, CONF-04)
        let intermediate_size = {
            let from_meta = meta.feed_forward_length();
            if from_meta > 0 {
                from_meta
            } else {
                infer_intermediate_size(file, hidden_size)
                    .ok_or(ConfigError::MissingField("intermediate_size"))?
            }
        };

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

        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        macro_rules! require_nonzero {
            ($field:expr, $name:literal) => {
                if $field == 0 {
                    return Err(ConfigError::Missing($name));
                }
            };
        }

        require_nonzero!(self.num_layers, "num_layers");
        require_nonzero!(self.hidden_size, "hidden_size");
        require_nonzero!(self.num_heads, "num_heads");
        require_nonzero!(self.num_kv_heads, "num_kv_heads");
        require_nonzero!(self.head_dim, "head_dim");
        require_nonzero!(self.intermediate_size, "intermediate_size");
        require_nonzero!(self.vocab_size, "vocab_size");
        require_nonzero!(self.max_seq_len, "max_seq_len");

        // GQA check: num_heads must be divisible by num_kv_heads
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(ConfigError::Invalid(format!(
                "num_heads ({}) not divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }

        // head_dim check: allow mismatch if head_dim came from explicit GGUF key
        // (some models like Phi3 have head_dim != hidden_size / num_heads)
        let _computed_head_dim = self.hidden_size / self.num_heads;
        // Not an error - just a note. Some models specify head_dim explicitly.
        // GPU kernels should use self.head_dim, not compute it.

        Ok(())
    }
}

/// Infer `intermediate_size` from MLP gate tensor shape when not in metadata.
/// Tries common tensor naming patterns for Qwen2/LLaMA/Phi.
fn infer_intermediate_size(file: &GgufFile, hidden_size: usize) -> Option<usize> {
    let candidates = [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];
    for name in &candidates {
        if let Ok(Some(tensor)) = file.tensor(name) {
            // dims are innermost-first: [hidden_size, intermediate_size]
            // or [intermediate_size, hidden_size] - use hidden_size to disambiguate
            if tensor.dims.len() >= 2 {
                let (d0, d1) = (tensor.dims[0] as usize, tensor.dims[1] as usize);
                if d0 == hidden_size && d1 != hidden_size {
                    return Some(d1);
                }
                if d1 == hidden_size && d0 != hidden_size {
                    return Some(d0);
                }
            }
        }
    }
    None
}

// ── Validation Errors ────────────────────────────────────────────────────────

/// Errors that can occur when building a `ModelConfig`.
#[derive(Debug)]
pub enum ConfigError {
    Missing(&'static str),
    MissingField(&'static str),
    Invalid(String),
    Load(LoadError),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Missing(field) => write!(f, "config missing: {}", field),
            ConfigError::MissingField(field) => write!(f, "field not found: {}", field),
            ConfigError::Invalid(msg) => write!(f, "invalid config: {}", msg),
            ConfigError::Load(e) => write!(f, "load error: {}", e),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<LoadError> for ConfigError {
    fn from(e: LoadError) -> Self {
        ConfigError::Load(e)
    }
}

// ── Chat templates ────────────────────────────────────────────────────────────

/// Prompt wrapping format for instruction-tuned models.
///
/// Each variant corresponds to one architecture family's instruct format.
/// Selection uses both the GGUF architecture string AND the tokenizer type,
/// because LLaMA2 and LLaMA3 share `architecture = "llama"` but use
/// different templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// No wrapping - raw completion mode.
    None,
    /// Qwen2/3 ChatML: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
    ChatML,
    /// LLaMA3: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>`
    LLaMA3,
    /// LLaMA2 / Mistral v0.1: `[INST] ... [/INST]`
    LLaMA2,
    /// Phi3: `<|user|>\n...<|end|>\n<|assistant|">\n`
    Phi3,
    /// Gemma: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
    Gemma,
}

impl ChatTemplate {
    /// Wrap `user_text` in the appropriate prompt format.
    /// Returns the text unchanged when `self == None`.
    pub fn apply(&self, user_text: &str) -> String {
        match self {
            ChatTemplate::None => user_text.to_string(),

            ChatTemplate::ChatML => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_text
            ),

            ChatTemplate::LLaMA3 => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user_text
            ),

            ChatTemplate::LLaMA2 => format!("[INST] {} [/INST]", user_text),

            ChatTemplate::Phi3 => format!(
                "<|user|>\n{}<|end|>\n<|assistant|\">\n",
                user_text
            ),

            ChatTemplate::Gemma => format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                user_text
            ),
        }
    }

    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            ChatTemplate::None => "none (raw completion)",
            ChatTemplate::ChatML => "ChatML (Qwen2/3)",
            ChatTemplate::LLaMA3 => "LLaMA3",
            ChatTemplate::LLaMA2 => "LLaMA2/Mistral",
            ChatTemplate::Phi3 => "Phi3",
            ChatTemplate::Gemma => "Gemma",
        }
    }
}

/// Detect the appropriate chat template from architecture + tokenizer type.
///
/// `tokenizer_model` is the value of `tokenizer.ggml.model` from the GGUF
/// KV section (e.g. `"gpt2"`, `"llama"`, `"spm"`).
///
/// The distinction between LLaMA2 and LLaMA3 requires the tokenizer type:
/// - LLaMA3 uses BPE (`"gpt2"`) with 128K vocab
/// - LLaMA2 uses SentencePiece (`"llama"` / `"spm"`)
pub fn detect_chat_template(architecture: &str, tokenizer_model: Option<&str>) -> ChatTemplate {
    match architecture {
        "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" | "qwen" => ChatTemplate::ChatML,

        "llama" | "mistral" | "yi" | "baichuan" | "internlm2" | "deepseek" => {
            // Distinguish LLaMA3 (BPE) from LLaMA2/Mistral (SPM)
            match tokenizer_model {
                Some("gpt2") | Some("bpe") => ChatTemplate::LLaMA3,
                _ => ChatTemplate::LLaMA2,
            }
        }

        "phi3" => ChatTemplate::Phi3,
        "gemma" | "gemma2" | "gemma3" => ChatTemplate::Gemma,
        "mixtral" => ChatTemplate::LLaMA2,

        // Unknown architecture: no template, raw completion
        _ => ChatTemplate::None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traits_qwen2_neox() {
        let t = ModelTraits::for_arch("qwen2");
        assert_eq!(t.rope_style, RopeStyle::NeoX);
        assert!(t.use_attention_bias);
        assert_eq!(t.default_rope_theta, 1_000_000.0);
    }

    #[test]
    fn traits_llama_normal() {
        let t = ModelTraits::for_arch("llama");
        assert_eq!(t.rope_style, RopeStyle::Normal);
        assert!(!t.use_attention_bias);
    }

    #[test]
    fn traits_unknown_falls_back_to_llama() {
        let t = ModelTraits::for_arch("some_future_arch");
        let ll = ModelTraits::for_arch("llama");
        assert_eq!(t.rope_style, ll.rope_style);
        assert_eq!(t.default_rope_theta, ll.default_rope_theta);
    }

    #[test]
    fn traits_phi3_fused_qkv() {
        let t = ModelTraits::for_arch("phi3");
        assert_eq!(t.attention_layout, AttentionLayout::FusedQkv);
        assert_eq!(t.rope_style, RopeStyle::Normal);
    }

    #[test]
    fn validation_rejects_zero_layers() {
        let cfg = ModelConfig {
            num_layers: 0,
            hidden_size: 896,
            num_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 4864,
            vocab_size: 151936,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            rope_neox: true,
            use_attention_bias: true,
            attention_layout: AttentionLayout::SplitQkv,
            architecture: "qwen2".to_string(),
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validation_rejects_bad_gqa() {
        // 14 heads, 3 kv_heads - 14 % 3 != 0
        let cfg = ModelConfig {
            num_layers: 24,
            hidden_size: 896,
            num_heads: 14,
            num_kv_heads: 3,
            head_dim: 64,
            intermediate_size: 4864,
            vocab_size: 151936,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            rope_neox: true,
            use_attention_bias: true,
            attention_layout: AttentionLayout::SplitQkv,
            architecture: "qwen2".to_string(),
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn template_qwen2_is_chatml() {
        assert_eq!(
            detect_chat_template("qwen2", Some("gpt2")),
            ChatTemplate::ChatML
        );
        assert_eq!(detect_chat_template("qwen3", None), ChatTemplate::ChatML);
    }

    #[test]
    fn template_llama3_detected_by_bpe() {
        // LLaMA3 has BPE tokenizer
        assert_eq!(
            detect_chat_template("llama", Some("gpt2")),
            ChatTemplate::LLaMA3
        );
    }

    #[test]
    fn template_llama2_detected_by_spm() {
        // LLaMA2 / Mistral use SentencePiece
        assert_eq!(
            detect_chat_template("llama", Some("llama")),
            ChatTemplate::LLaMA2
        );
        assert_eq!(
            detect_chat_template("mistral", Some("llama")),
            ChatTemplate::LLaMA2
        );
        assert_eq!(detect_chat_template("mixtral", None), ChatTemplate::LLaMA2);
    }

    #[test]
    fn template_phi3() {
        assert_eq!(detect_chat_template("phi3", None), ChatTemplate::Phi3);
    }

    #[test]
    fn template_gemma() {
        assert_eq!(detect_chat_template("gemma2", None), ChatTemplate::Gemma);
    }

    #[test]
    fn template_unknown_arch_is_none() {
        assert_eq!(
            detect_chat_template("future_arch", None),
            ChatTemplate::None
        );
    }

    #[test]
    fn chatml_apply_wraps_correctly() {
        let t = ChatTemplate::ChatML;
        let out = t.apply("Hello");
        assert!(out.starts_with("<|im_start|>user\n"));
        assert!(out.contains("Hello"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama2_apply() {
        let out = ChatTemplate::LLaMA2.apply("Hi");
        assert_eq!(out, "[INST] Hi [/INST]");
    }

    #[test]
    fn none_apply_passthrough() {
        let text = "raw prompt";
        assert_eq!(ChatTemplate::None.apply(text), text);
    }
}
