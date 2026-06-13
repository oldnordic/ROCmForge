use crate::loader::{GgufFile, LoadError};

use super::tensor_names::TensorNameRegistry;
use super::traits::{AttentionLayout, FfnLayout, ModelTraits, RopeStyle};

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
    /// Precomputed RoPE frequencies: freq[i] = 1/theta^(2i/head_dim).
    /// Length = head_dim/2. Eliminates powf per pair at inference time.
    pub rope_freq: Vec<f32>,

    // Behavioral flags (from ModelTraits)
    pub rope_neox: bool,
    pub use_attention_bias: bool,
    pub attention_layout: AttentionLayout,
    pub ffn_layout: FfnLayout,

    /// The raw architecture string from GGUF (e.g. "qwen2", "llama")
    pub architecture: String,

    /// Tensor name registry for this model
    pub tensor_registry: TensorNameRegistry,

    // LFM2 MoE-specific parameters
    pub shortconv_l_cache: Option<usize>,
    pub num_dense_layers: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub use_expert_bias: bool,
    pub expert_weights_scale: f32,

    // Research & advanced compression synergy parameters
    pub kv_lora_dim: Option<usize>,
    pub kv_frame_codec_enabled: Option<bool>,
    pub adastate_anchors_enabled: Option<bool>,
    pub kv_quant_bits: Option<usize>,
    pub turboquant_centroids: Option<Vec<f32>>,
    pub qjl_scale: Option<f32>,
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

        // Precompute RoPE frequencies to eliminate powf at inference time
        let half = head_dim / 2;
        let rope_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

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
            rope_freq,
            rope_neox: traits.rope_style == RopeStyle::NeoX,
            use_attention_bias: traits.use_attention_bias,
            attention_layout: traits.attention_layout,
            ffn_layout: traits.ffn_layout,
            architecture: meta.architecture.clone(),
            tensor_registry: TensorNameRegistry::from_scheme(&traits.tensor_naming),
            shortconv_l_cache: meta.shortconv_l_cache(),
            num_dense_layers: meta.num_dense_layers(),
            num_experts_per_tok: meta.num_experts_per_tok(),
            use_expert_bias: meta.use_expert_bias(),
            expert_weights_scale: meta.expert_weights_scale(),
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
        };

        config.validate()?;
        Ok(config)
    }

    /// Build `ModelConfig` from parsed RFM file metadata.
    pub fn from_rfm(meta: &crate::loader::RfmMetadata) -> Result<Self, ConfigError> {
        let traits = ModelTraits::for_arch(&meta.architecture);

        let half = meta.head_dim / 2;
        let rope_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / meta.rope_theta.powf((2 * i) as f32 / meta.head_dim as f32))
            .collect();

        let config = Self {
            num_layers: meta.num_layers,
            hidden_size: meta.hidden_size,
            num_heads: meta.num_heads,
            num_kv_heads: meta.num_kv_heads,
            head_dim: meta.head_dim,
            intermediate_size: meta.intermediate_size,
            vocab_size: meta.vocab_size,
            max_seq_len: meta.max_seq_len,
            rms_norm_eps: meta.rms_norm_eps,
            rope_theta: meta.rope_theta,
            rope_freq,
            rope_neox: meta.rope_neox,
            use_attention_bias: meta.use_attention_bias,
            attention_layout: traits.attention_layout,
            ffn_layout: traits.ffn_layout,
            architecture: meta.architecture.clone(),
            tensor_registry: TensorNameRegistry::from_scheme(&traits.tensor_naming),
            shortconv_l_cache: None,
            num_dense_layers: None,
            num_experts_per_tok: None,
            use_expert_bias: false,
            expert_weights_scale: 1.0,
            kv_lora_dim: meta.kv_lora_dim.map(|d| d.next_power_of_two()),
            kv_frame_codec_enabled: meta.kv_frame_codec_enabled,
            adastate_anchors_enabled: meta.adastate_anchors_enabled,
            kv_quant_bits: meta.kv_quant_bits,
            turboquant_centroids: meta.turboquant_centroids.clone(),
            qjl_scale: meta.qjl_scale,
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
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
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
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_gate_exps.weight",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tensor_names::TensorNamingScheme;

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
            ffn_layout: FfnLayout::SwiGLU,
            architecture: "qwen2".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
            rope_freq: vec![1.0, 0.5],
            shortconv_l_cache: None,
            num_dense_layers: None,
            num_experts_per_tok: None,
            use_expert_bias: false,
            expert_weights_scale: 1.0,
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
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
            ffn_layout: FfnLayout::SwiGLU,
            architecture: "qwen2".to_string(),
            tensor_registry: TensorNameRegistry::from_scheme(&TensorNamingScheme::Gguf),
            rope_freq: vec![1.0, 0.5],
            shortconv_l_cache: None,
            num_dense_layers: None,
            num_experts_per_tok: None,
            use_expert_bias: false,
            expert_weights_scale: 1.0,
            kv_lora_dim: None,
            kv_frame_codec_enabled: None,
            adastate_anchors_enabled: None,
            kv_quant_bits: None,
            turboquant_centroids: None,
            qjl_scale: None,
        };
        assert!(cfg.validate().is_err());
    }
}
