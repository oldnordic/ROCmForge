use crate::loader::{GgufFile, LoadError};

use super::tensor_names::TensorNameRegistry;
use super::traits::{AttentionLayout, FfnLayout, ModelTraits, RopeStyle};

/// All hyperparameters needed to run inference.
///
/// Values come from GGUF metadata; behaviors come from the traits registry.
/// `vocab_size` comes from `tokenizer_data.tokens.len()` - not GGUF metadata,
/// which returns 0 for Qwen2.5.
#[derive(Debug, Clone, Default)]
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

    // Gemma4-specific per-layer parameters
    /// Per-layer query projection size (num_heads * q_head_dim).
    pub per_layer_q_sizes: Vec<usize>,
    /// Per-layer KV projection size (num_kv_heads * kv_head_dim).
    pub per_layer_kv_sizes: Vec<usize>,
    /// Per-layer query head dimension.
    pub per_layer_head_dims: Vec<usize>,
    /// Per-layer KV head dimension.
    pub per_layer_kv_head_dims: Vec<usize>,
    /// Per-layer FFN intermediate size.
    pub per_layer_intermediate_sizes: Vec<usize>,
    /// Per-layer RoPE theta (sliding vs full attention).
    pub per_layer_rope_thetas: Vec<f32>,
    /// Per-layer RoPE partial rotary factor (1.0 for sliding, 0.25 for full).
    pub per_layer_rope_partial_factors: Vec<f32>,
    /// Per-layer sliding-window size (0 for full attention).
    pub per_layer_sliding_windows: Vec<usize>,
    /// True for sliding/local attention layers, false for full/global layers.
    pub per_layer_is_sliding: Vec<bool>,
    /// Precomputed RoPE frequencies per layer (length depends on rotated dims).
    pub per_layer_rope_freqs: Vec<Vec<f32>>,
    /// Number of consecutive final layers that share KV projections.
    pub num_kv_shared_layers: usize,
    /// Per-layer embedding (PLE) dimension, if present.
    pub hidden_size_per_layer_input: usize,
    /// Final logit softcapping value (e.g., 30.0 for Gemma4).
    pub final_logit_softcapping: Option<f32>,
    /// Attention logit softcapping value (e.g., 50.0 for Gemma4).
    pub attention_logit_cap: Option<f32>,
    /// Attention score scale (1.0 for Gemma4 because it uses QK norm;
    /// 1/sqrt(head_dim) otherwise).
    pub attention_scale: f32,
    /// Token embedding scale (sqrt(hidden_size) for Gemma4, 1.0 otherwise).
    pub embedding_scale: f32,
    /// Use GELU instead of SiLU in the SwiGLU FFN (Gemma4).
    pub use_gelu_swiglu: bool,
}

impl ModelConfig {
    /// Query projection size for a layer (handles Gemma4 per-layer dims).
    pub fn q_size(&self, layer: usize) -> usize {
        self.per_layer_q_sizes
            .get(layer)
            .copied()
            .unwrap_or(self.num_heads * self.head_dim)
    }

    /// KV projection size for a layer.
    pub fn kv_size(&self, layer: usize) -> usize {
        self.per_layer_kv_sizes
            .get(layer)
            .copied()
            .unwrap_or(self.num_kv_heads * self.head_dim)
    }

    /// Query head dimension for a layer.
    pub fn head_dim_for_layer(&self, layer: usize) -> usize {
        self.per_layer_head_dims
            .get(layer)
            .copied()
            .unwrap_or(self.head_dim)
    }

    /// KV head dimension for a layer.
    pub fn kv_head_dim_for_layer(&self, layer: usize) -> usize {
        self.per_layer_kv_head_dims
            .get(layer)
            .copied()
            .unwrap_or(self.head_dim)
    }

    /// Intermediate size for a layer.
    pub fn intermediate_size_for_layer(&self, layer: usize) -> usize {
        self.per_layer_intermediate_sizes
            .get(layer)
            .copied()
            .unwrap_or(self.intermediate_size)
    }

    /// RoPE theta for a layer.
    pub fn rope_theta_for_layer(&self, layer: usize) -> f32 {
        self.per_layer_rope_thetas
            .get(layer)
            .copied()
            .unwrap_or(self.rope_theta)
    }

    /// RoPE partial rotary factor for a layer.
    pub fn rope_partial_factor_for_layer(&self, layer: usize) -> f32 {
        self.per_layer_rope_partial_factors
            .get(layer)
            .copied()
            .unwrap_or(1.0)
    }

    /// Sliding-window size for a layer (0 means full causal).
    pub fn sliding_window_for_layer(&self, layer: usize) -> usize {
        self.per_layer_sliding_windows
            .get(layer)
            .copied()
            .unwrap_or(0)
    }

    /// True if the layer is a sliding/local attention layer.
    pub fn is_sliding_for_layer(&self, layer: usize) -> bool {
        self.per_layer_is_sliding
            .get(layer)
            .copied()
            .unwrap_or(false)
    }

    /// First layer index that participates in KV sharing (Gemma4).
    pub fn first_kv_shared_layer_idx(&self) -> usize {
        if self.num_kv_shared_layers > 0 && self.num_layers >= self.num_kv_shared_layers {
            self.num_layers - self.num_kv_shared_layers
        } else {
            self.num_layers
        }
    }

    /// Layer type string used for KV sharing (Gemma4).
    pub fn layer_type_for_layer(&self, layer: usize) -> &'static str {
        if self.is_sliding_for_layer(layer) {
            "sliding_attention"
        } else {
            "full_attention"
        }
    }

    /// True if this layer stores the full-length KV state shared by later layers.
    pub fn stores_shared_kv(&self, layer: usize) -> bool {
        let first = self.first_kv_shared_layer_idx();
        if layer >= first {
            return false;
        }
        // Last non-shared layer of its type stores the KV.
        let ty = self.layer_type_for_layer(layer);
        for l in (layer + 1)..first {
            if self.layer_type_for_layer(l) == ty {
                return false;
            }
        }
        true
    }

    /// Precomputed RoPE frequencies for a layer.
    pub fn rope_freq_for_layer(&self, layer: usize) -> &[f32] {
        self.per_layer_rope_freqs
            .get(layer)
            .map(|v| v.as_slice())
            .unwrap_or(self.rope_freq.as_slice())
    }

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

        let mut config = Self {
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
            per_layer_q_sizes: Vec::new(),
            per_layer_kv_sizes: Vec::new(),
            per_layer_head_dims: Vec::new(),
            per_layer_kv_head_dims: Vec::new(),
            per_layer_intermediate_sizes: Vec::new(),
            per_layer_rope_thetas: Vec::new(),
            per_layer_rope_partial_factors: Vec::new(),
            per_layer_sliding_windows: Vec::new(),
            per_layer_is_sliding: Vec::new(),
            per_layer_rope_freqs: Vec::new(),
            num_kv_shared_layers: 0,
            hidden_size_per_layer_input: 0,
            final_logit_softcapping: None,
            attention_logit_cap: None,
            attention_scale: 1.0 / (head_dim as f32).sqrt(),
            embedding_scale: 1.0,
            use_gelu_swiglu: false,
        };

        // Gemma4 has per-layer attention dims, RoPE params, and PLE.
        if config.architecture == "gemma4" {
            configure_gemma4(file, meta, &mut config)?;
        }

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
            per_layer_q_sizes: Vec::new(),
            per_layer_kv_sizes: Vec::new(),
            per_layer_head_dims: Vec::new(),
            per_layer_kv_head_dims: Vec::new(),
            per_layer_intermediate_sizes: Vec::new(),
            per_layer_rope_thetas: Vec::new(),
            per_layer_rope_partial_factors: Vec::new(),
            per_layer_sliding_windows: Vec::new(),
            per_layer_is_sliding: Vec::new(),
            per_layer_rope_freqs: Vec::new(),
            num_kv_shared_layers: 0,
            hidden_size_per_layer_input: 0,
            final_logit_softcapping: None,
            attention_logit_cap: None,
            attention_scale: 1.0 / (meta.head_dim as f32).sqrt(),
            embedding_scale: 1.0,
            use_gelu_swiglu: false,
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

/// Configure Gemma4-specific per-layer parameters from GGUF metadata and tensor shapes.
fn configure_gemma4(
    file: &GgufFile,
    meta: &crate::loader::GgufMetadata,
    config: &mut ModelConfig,
) -> Result<(), ConfigError> {
    let num_layers = config.num_layers;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;

    // Layer type pattern from metadata; fall back to shape-based heuristic.
    let sliding_pattern = meta
        .resolve_bool_array(&["attention.sliding_window_pattern"])
        .unwrap_or_else(Vec::new);

    let rope_theta_full = meta.rope_freq_base(config.rope_theta);
    let rope_theta_swa = meta.rope_freq_base_swa().unwrap_or(10_000.0);
    let sliding_window = meta.sliding_window();
    let q_head_dim = meta
        .rope_dimension_count_swa()
        .unwrap_or(config.head_dim.max(256));
    let global_q_head_dim = meta.rope_dimension_count().unwrap_or(q_head_dim * 2);

    config.num_kv_shared_layers = meta.shared_kv_layers();
    config.hidden_size_per_layer_input = meta.embedding_length_per_layer_input();
    config.final_logit_softcapping = meta.final_logit_softcapping();
    // Attention logit soft-capping is part of Gemma4's audio/vision layers, not
    // the text decoder. Leave it unset so flash_attn_decode does not transform
    // text attention scores.
    config.attention_logit_cap = None;
    // Gemma4 applies QK norm and does not use an additional 1/sqrt(head_dim)
    // attention scale; flash_attn_decode skips scaling when the value is 1.0.
    config.attention_scale = 1.0;
    // The GGUF token_embd.weight already contains the Gemma sqrt(hidden_size)
    // scaling baked in by the converter; do not scale it again at runtime.
    config.embedding_scale = 1.0;
    config.use_gelu_swiglu = true;

    for layer in 0..num_layers {
        let q_name = format!("blk.{}.attn_q.weight", layer);
        let k_name = format!("blk.{}.attn_k.weight", layer);
        let gate_name = format!("blk.{}.ffn_gate.weight", layer);

        let q_size = file
            .tensor(&q_name)
            .map_err(ConfigError::Load)?
            .map(|t| t.dims[1] as usize)
            .unwrap_or(num_heads * q_head_dim);
        let kv_size = file
            .tensor(&k_name)
            .map_err(ConfigError::Load)?
            .map(|t| t.dims[1] as usize)
            .unwrap_or(num_kv_heads * global_q_head_dim);
        let intermediate_size = file
            .tensor(&gate_name)
            .map_err(ConfigError::Load)?
            .map(|t| t.dims[1] as usize)
            .unwrap_or(config.intermediate_size);

        let q_head_dim_layer = if num_heads > 0 { q_size / num_heads } else { 0 };
        let kv_head_dim_layer = if num_kv_heads > 0 {
            kv_size / num_kv_heads
        } else {
            0
        };

        let is_sliding = sliding_pattern
            .get(layer)
            .copied()
            .unwrap_or_else(|| q_head_dim_layer == q_head_dim);

        let (theta, partial_factor, window) = if is_sliding {
            (rope_theta_swa, 1.0f32, sliding_window)
        } else {
            (rope_theta_full, 0.25f32, 0)
        };

        let rotated_dims = (q_head_dim_layer as f32 * partial_factor) as usize;
        let half = rotated_dims / 2;
        let rope_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / q_head_dim_layer as f32))
            .collect();

        config.per_layer_q_sizes.push(q_size);
        config.per_layer_kv_sizes.push(kv_size);
        config.per_layer_head_dims.push(q_head_dim_layer);
        config.per_layer_kv_head_dims.push(kv_head_dim_layer);
        config.per_layer_intermediate_sizes.push(intermediate_size);
        config.per_layer_rope_thetas.push(theta);
        config.per_layer_rope_partial_factors.push(partial_factor);
        config.per_layer_sliding_windows.push(window);
        config.per_layer_is_sliding.push(is_sliding);
        config.per_layer_rope_freqs.push(rope_freq);
    }

    Ok(())
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
            ..Default::default()
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
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }
}
