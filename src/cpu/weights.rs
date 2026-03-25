//! CPU model weights — copies tensors from GGUF mmap into Vec<u8>.
//!
//! Weights are stored in their native quantized format (Q4_0, Q4_1, Q8_0, etc.)
//! and dequantized on-the-fly during inference.

use crate::config::ModelConfig;
use crate::loader::{GgmlType, GgufFile, LoadError};

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
    pub attn_q_type: GgmlType,
    /// Query bias (optional, always F32 if present)
    pub attn_q_bias: Option<Vec<f32>>,
    /// Key projection weights (quantized)
    pub attn_k: Vec<u8>,
    pub attn_k_type: GgmlType,
    /// Key bias (optional, always F32 if present)
    pub attn_k_bias: Option<Vec<f32>>,
    /// Value projection weights (quantized)
    pub attn_v: Vec<u8>,
    pub attn_v_type: GgmlType,
    /// Value bias (optional, always F32 if present)
    pub attn_v_bias: Option<Vec<f32>>,
    /// Attention output projection (quantized)
    pub attn_o: Vec<u8>,
    pub attn_o_type: GgmlType,
    /// RMS norm weights for FFN (always F32)
    pub ffn_norm: Vec<f32>,
    /// FFN gate projection (SwiGLU gate) (quantized)
    pub ffn_gate: Vec<u8>,
    pub ffn_gate_type: GgmlType,
    /// FFN up projection (SwiGLU up) (quantized)
    pub ffn_up: Vec<u8>,
    pub ffn_up_type: GgmlType,
    /// FFN down projection (quantized)
    pub ffn_down: Vec<u8>,
    pub ffn_down_type: GgmlType,
    /// General quantization type for this layer (legacy)
    pub weight_type: GgmlType,
}

impl CpuLayerWeights {
    fn load(file: &GgufFile, layer: usize) -> Result<Self, WeightError> {
        let p = |s: &str| format!("blk.{}.{}", layer, s);

        // Helper to get tensor type
        let get_type = |name: &str| -> GgmlType {
            file.tensor(name)
                .ok()
                .and_then(|opt| opt)
                .map(|t| t.ggml_type)
                .unwrap_or(GgmlType::F32)
        };

        let attn_q_type = get_type(&p("attn_q.weight"));
        let attn_k_type = get_type(&p("attn_k.weight"));
        let attn_v_type = get_type(&p("attn_v.weight"));
        let attn_o_type = get_type(&p("attn_output.weight"));
        let ffn_gate_type = get_type(&p("ffn_gate.weight"));
        let ffn_up_type = get_type(&p("ffn_up.weight"));
        let ffn_down_type = get_type(&p("ffn_down.weight"));
        let weight_type = attn_q_type; // Legacy: use attn_q type as general type

        Ok(Self {
            attn_norm: copy_f32(file, &p("attn_norm.weight"))?,
            attn_q: copy_tensor(file, &p("attn_q.weight"))?,
            attn_q_type,
            attn_q_bias: optional_f32(copy_tensor_optional(file, &p("attn_q.bias"))?),
            attn_k: copy_tensor(file, &p("attn_k.weight"))?,
            attn_k_type,
            attn_k_bias: optional_f32(copy_tensor_optional(file, &p("attn_k.bias"))?),
            attn_v: copy_tensor(file, &p("attn_v.weight"))?,
            attn_v_type,
            attn_v_bias: optional_f32(copy_tensor_optional(file, &p("attn_v.bias"))?),
            attn_o: copy_tensor(file, &p("attn_output.weight"))?,
            attn_o_type,
            ffn_norm: copy_f32(file, &p("ffn_norm.weight"))?,
            ffn_gate: copy_tensor(file, &p("ffn_gate.weight"))?,
            ffn_gate_type,
            ffn_up: copy_tensor(file, &p("ffn_up.weight"))?,
            ffn_up_type,
            ffn_down: copy_tensor(file, &p("ffn_down.weight"))?,
            ffn_down_type,
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
    /// Final RMS norm weights (always F32)
    pub output_norm: Vec<f32>,
    /// Language model head / output projection (quantized)
    pub lm_head: Vec<u8>,
    /// Quantization type for token embeddings
    pub token_emb_type: GgmlType,
    /// Quantization type for LM head
    pub lm_head_type: GgmlType,
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

        // Load embedding weights
        let token_emb_type = file
            .tensor("token_embd.weight")
            .map_err(WeightError::Load)?
            .map(|t| t.ggml_type)
            .unwrap_or(GgmlType::F32);
        let token_emb = copy_tensor(file, "token_embd.weight")?;
        let output_norm = copy_f32(file, "output_norm.weight")?;

        // LM head: use output.weight if present, otherwise tie to embeddings
        let (lm_head, lm_head_type) = if file.has_tensor("output.weight") {
            let lm_type = file
                .tensor("output.weight")
                .map_err(WeightError::Load)?
                .map(|t| t.ggml_type)
                .unwrap_or(GgmlType::F32);
            (copy_tensor(file, "output.weight")?, lm_type)
        } else {
            // Weight tying: lm_head shares embedding weights
            (token_emb.clone(), token_emb_type)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let layer = CpuLayerWeights::load(file, i)?;
            if i == 0 || (i + 1) % 8 == 0 || i + 1 == n {
                eprintln!("[cpu weights] layer {}/{} loaded", i + 1, n);
            }
            layers.push(layer);
        }

        Ok(Self {
            layers,
            token_emb,
            output_norm,
            lm_head,
            token_emb_type,
            lm_head_type,
        })
    }

    /// Get weights for a specific layer.
    pub fn layer(&self, i: usize) -> &CpuLayerWeights {
        &self.layers[i]
    }
}
