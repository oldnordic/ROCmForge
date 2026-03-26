//! CPU model weights — copies tensors from GGUF mmap into Vec<u8>.
//!
//! Weights are stored in their native quantized format (Q4_0, Q4_1, Q8_0, etc.)
//! and dequantized on-the-fly during inference.

use crate::config::ModelConfig;
use crate::loader::{GgmlType, GgufFile, LoadError};
use super::transpose::compute_transpose_flag;

// ── Error ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum WeightError {
    TensorNotFound(String),
    Load(LoadError),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightError::TensorNotFound(n) => write!(f, "tensor not found: {}", n),
            WeightError::Load(e) => write!(f, "GGUF load: {}", e),
        }
    }
}

impl std::error::Error for WeightError {}

impl From<LoadError> for WeightError {
    fn from(e: LoadError) -> Self {
        WeightError::Load(e)
    }
}

// ── Weight Metadata ─────────────────────────────────────────────────────────────

/// Metadata for a weight tensor, including its quantization type,
/// dimensions from GGUF, and whether it needs transposition.
#[derive(Clone, Debug)]
pub struct WeightMeta {
    /// Quantization type (F32, Q4_0, Q4_1, Q8_0, etc.)
    pub wtype: GgmlType,
    /// Dimensions from GGUF (innermost first, i.e., [cols, rows] for 2D matrices)
    pub dims: Vec<u64>,
    /// Whether this weight tensor needs transposed access
    pub needs_transpose: bool,
}

impl WeightMeta {
    /// Create metadata from a GGUF tensor descriptor.
    pub fn from_desc(desc: &crate::loader::TensorDesc, needs_transpose: bool) -> Self {
        Self {
            wtype: desc.ggml_type,
            dims: desc.dims.clone(),
            needs_transpose,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Copy tensor bytes from the mmap into a Vec<u8>.
fn copy_tensor(file: &GgufFile, name: &str) -> Result<Vec<u8>, WeightError> {
    let t = file
        .tensor(name)
        .map_err(WeightError::Load)?
        .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
    Ok(t.data.to_vec())
}

fn copy_tensor_optional(file: &GgufFile, name: &str) -> Result<Option<Vec<u8>>, WeightError> {
    match file.tensor(name).map_err(WeightError::Load)? {
        None => Ok(None),
        Some(t) => Ok(Some(t.data.to_vec())),
    }
}

/// Copy an always-F32 tensor as Vec<f32>.
fn copy_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, WeightError> {
    let bytes = copy_tensor(file, name)?;
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const f32, out.as_mut_ptr(), n);
    }
    Ok(out)
}

fn copy_f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const f32, out.as_mut_ptr(), n);
    }
    out
}

fn optional_f32(opt: Option<Vec<u8>>) -> Option<Vec<f32>> {
    opt.map(|b| copy_f32_from_bytes(&b))
}

/// Copy tensor bytes and create metadata.
fn copy_tensor_with_meta(
    file: &GgufFile,
    name: &str,
    needs_transpose: bool,
) -> Result<(Vec<u8>, WeightMeta), WeightError> {
    let t = file
        .tensor(name)
        .map_err(WeightError::Load)?
        .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
    let data = t.data.to_vec();
    let meta = WeightMeta {
        wtype: t.ggml_type,
        dims: t.dims.to_vec(),
        needs_transpose,
    };
    Ok((data, meta))
}

// ── Per-layer weights ─────────────────────────────────────────────────────────────────

/// Weights for a single transformer layer.
///
/// All weight tensors are stored in their native quantized format.
/// Dequantization happens during inference in the forward pass.
pub struct CpuLayerWeights {
    /// RMS norm weights for attention (always F32)
    pub attn_norm: Vec<f32>,
    /// Query projection weights (quantized)
    pub attn_q: Vec<u8>,
    pub attn_q_meta: WeightMeta,
    /// Query bias (optional, always F32 if present)
    pub attn_q_bias: Option<Vec<f32>>,
    /// Key projection weights (quantized)
    pub attn_k: Vec<u8>,
    pub attn_k_meta: WeightMeta,
    /// Key bias (optional, always F32 if present)
    pub attn_k_bias: Option<Vec<f32>>,
    /// Value projection weights (quantized)
    pub attn_v: Vec<u8>,
    pub attn_v_meta: WeightMeta,
    /// Value bias (optional, always F32 if present)
    pub attn_v_bias: Option<Vec<f32>>,
    /// Attention output projection (quantized)
    pub attn_o: Vec<u8>,
    pub attn_o_meta: WeightMeta,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: Vec<f32>,
    /// FFN gate projection (SwiGLU gate) (quantized)
    pub ffn_gate: Vec<u8>,
    pub ffn_gate_meta: WeightMeta,
    /// FFN up projection (SwiGLU up) (quantized)
    pub ffn_up: Vec<u8>,
    pub ffn_up_meta: WeightMeta,
    /// FFN down projection (quantized)
    pub ffn_down: Vec<u8>,
    pub ffn_down_meta: WeightMeta,
    /// General quantization type for this layer (legacy)
    pub weight_type: GgmlType,
}

impl CpuLayerWeights {
    fn load(file: &GgufFile, layer: usize, config: &ModelConfig) -> Result<Self, WeightError> {
        let p = |s: &str| format!("blk.{}.{}", layer, s);

        // Helper to get tensor type
        let get_type = |name: &str| -> GgmlType {
            file.tensor(name)
                .ok()
                .and_then(|opt| opt)
                .map(|t| t.ggml_type)
                .unwrap_or(GgmlType::F32)
        };

        // Helper to load weight with metadata
        let load_weight = |name: &str| -> Result<(Vec<u8>, WeightMeta), WeightError> {
            let desc = file.tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            let needs_transpose = compute_transpose_flag(name, &desc.dims, desc.ggml_type, config, false, false);
            copy_tensor_with_meta(file, name, needs_transpose)
        };

        let (attn_q, attn_q_meta) = load_weight(&p("attn_q.weight"))?;
        let (attn_k, attn_k_meta) = load_weight(&p("attn_k.weight"))?;
        let (attn_v, attn_v_meta) = load_weight(&p("attn_v.weight"))?;
        let (attn_o, attn_o_meta) = load_weight(&p("attn_output.weight"))?;
        let (ffn_gate, ffn_gate_meta) = load_weight(&p("ffn_gate.weight"))?;
        let (ffn_up, ffn_up_meta) = load_weight(&p("ffn_up.weight"))?;
        let (ffn_down, ffn_down_meta) = load_weight(&p("ffn_down.weight"))?;

        let weight_type = attn_q_meta.wtype; // Legacy: use attn_q type as general type

        Ok(Self {
            attn_norm: copy_f32(file, &p("attn_norm.weight"))?,
            attn_q,
            attn_q_meta,
            attn_q_bias: optional_f32(copy_tensor_optional(file, &p("attn_q.bias"))?),
            attn_k,
            attn_k_meta,
            attn_k_bias: optional_f32(copy_tensor_optional(file, &p("attn_k.bias"))?),
            attn_v,
            attn_v_meta,
            attn_v_bias: optional_f32(copy_tensor_optional(file, &p("attn_v.bias"))?),
            attn_o,
            attn_o_meta,
            ffn_norm: copy_f32(file, &p("ffn_norm.weight"))?,
            ffn_gate,
            ffn_gate_meta,
            ffn_up,
            ffn_up_meta,
            ffn_down,
            ffn_down_meta,
            weight_type,
        })
    }
}

// ── Full model weights ─────────────────────────────────────────────────────────────────

/// All weights for a transformer model, loaded into CPU memory.
pub struct CpuModelWeights {
    /// Per-layer weights
    pub layers: Vec<CpuLayerWeights>,
    /// Token embedding matrix (quantized)
    pub token_emb: Vec<u8>,
    pub token_emb_meta: WeightMeta,
    /// Final RMS norm weights (always F32)
    pub output_norm: Vec<f32>,
    /// Language model head / output projection (quantized)
    pub lm_head: Vec<u8>,
    pub lm_head_meta: WeightMeta,
    /// Whether LM head is tied to token embeddings (shared weights)
    pub lm_head_tied: bool,
}

impl CpuModelWeights {
    /// Load all weights from a GGUF file into CPU memory.
    ///
    /// # Arguments
    /// * `file` - Open GGUF file
    /// * `config` - Model configuration (determines number of layers, etc.)
    ///
    /// # Returns
    /// All model weights loaded into RAM.
    pub fn load(file: &GgufFile, config: &ModelConfig) -> Result<Self, WeightError> {
        let n = config.num_layers;

        // Load embedding weights with metadata
        let token_emb_desc = file.tensor("token_embd.weight")
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound("token_embd.weight".to_string()))?;
        let (token_emb, token_emb_meta) = copy_tensor_with_meta(file, "token_embd.weight", false)?;
        let output_norm = copy_f32(file, "output_norm.weight")?;

        // LM head: use output.weight if present, otherwise tie to embeddings
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor("output.weight") {
            let lm_view = file.tensor("output.weight")
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound("output.weight".to_string()))?;
            let needs_transpose = compute_transpose_flag("output.weight", &lm_view.dims, lm_view.ggml_type, config, true, false);
            let data = copy_tensor(file, "output.weight")?;
            let meta = WeightMeta {
                wtype: lm_view.ggml_type,
                dims: lm_view.dims.to_vec(),
                needs_transpose,
            };
            (data, meta, false)
        } else {
            // Weight tying: lm_head shares embedding weights
            // Tied embeddings need transposed access (W is [hidden_size, vocab_size])
            let tied_meta = WeightMeta {
                wtype: token_emb_meta.wtype,
                dims: token_emb_meta.dims.clone(),
                needs_transpose: true,  // Tied embeddings always need transpose
            };
            (token_emb.clone(), tied_meta, true)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let layer = CpuLayerWeights::load(file, i, config)?;
            if i == 0 || (i + 1) % 8 == 0 || i + 1 == n {
                eprintln!("[cpu weights] layer {}/{} loaded", i + 1, n);
            }
            layers.push(layer);
        }

        Ok(Self {
            layers,
            token_emb,
            token_emb_meta,
            output_norm,
            lm_head,
            lm_head_meta,
            lm_head_tied,
        })
    }

    /// Get weights for a specific layer.
    pub fn layer(&self, i: usize) -> &CpuLayerWeights {
        &self.layers[i]
    }
}
