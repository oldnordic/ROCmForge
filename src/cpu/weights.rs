//! CPU model weights — copies tensors from GGUF mmap into Vec<u8>.
//!
//! Weights are stored in their native quantized format (Q4_0, Q4_1, Q8_0, etc.)
//! and dequantized on-the-fly during inference.

use super::transpose::compute_transpose_flag;
use crate::config::{ModelConfig, TensorName, TensorNamingScheme};
use crate::loader::{GgmlType, GgufFile, LoadError, RfmFile, RfmType};

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
            let desc = file
                .tensor(name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
            let needs_transpose =
                compute_transpose_flag(name, desc.dims, desc.ggml_type, config, false, false);
            copy_tensor_with_meta(file, name, needs_transpose)
        };

        // Helper to load weight with fallback names (for MoE models)
        let load_weight_fallback = |names: &[&str]| -> Result<(Vec<u8>, WeightMeta), WeightError> {
            for name in names {
                if let Ok(Some(desc)) = file.tensor(name) {
                    let needs_transpose = compute_transpose_flag(
                        name,
                        desc.dims,
                        desc.ggml_type,
                        config,
                        false,
                        false,
                    );
                    return copy_tensor_with_meta(file, name, needs_transpose);
                }
            }
            Err(WeightError::TensorNotFound(format!("tried {:?}", names)))
        };

        let (attn_q, attn_q_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnQ, layer))?;
        let (attn_k, attn_k_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnK, layer))?;
        let (attn_v, attn_v_meta) =
            load_weight(&config.tensor_registry.resolve(TensorName::AttnV, layer))?;
        let (attn_o, attn_o_meta) = load_weight(
            &config
                .tensor_registry
                .resolve(TensorName::AttnOutput, layer),
        )?;

        // For MoE models, try _exps tensors first, then fall back to standard names
        let ffn_gate_name = config.tensor_registry.resolve(TensorName::FfnGate, layer);
        let ffn_up_name = config.tensor_registry.resolve(TensorName::FfnUp, layer);
        let ffn_down_name = config.tensor_registry.resolve(TensorName::FfnDown, layer);

        let (ffn_gate, ffn_gate_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_gate_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnGateExps, layer);
                load_weight_fallback(&[&ffn_gate_exps_name, &ffn_gate_name])?
            } else {
                load_weight(&ffn_gate_name)?
            };

        let (ffn_up, ffn_up_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_up_exps_name = config.tensor_registry.resolve(TensorName::FfnUpExps, layer);
                load_weight_fallback(&[&ffn_up_exps_name, &ffn_up_name])?
            } else {
                load_weight(&ffn_up_name)?
            };

        let (ffn_down, ffn_down_meta) =
            if matches!(config.tensor_registry.scheme, TensorNamingScheme::GgufMoE) {
                let ffn_down_exps_name = config
                    .tensor_registry
                    .resolve(TensorName::FfnDownExps, layer);
                load_weight_fallback(&[&ffn_down_exps_name, &ffn_down_name])?
            } else {
                load_weight(&ffn_down_name)?
            };

        let weight_type = attn_q_meta.wtype; // Legacy: use attn_q type as general type

        Ok(Self {
            attn_norm: copy_f32(
                file,
                &config.tensor_registry.resolve(TensorName::AttnNorm, layer),
            )?,
            attn_q,
            attn_q_meta,
            attn_q_bias: match config
                .tensor_registry
                .resolve_optional(TensorName::AttnQBias, layer)
            {
                Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
                None => None,
            },
            attn_k,
            attn_k_meta,
            attn_k_bias: match config
                .tensor_registry
                .resolve_optional(TensorName::AttnKBias, layer)
            {
                Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
                None => None,
            },
            attn_v,
            attn_v_meta,
            attn_v_bias: match config
                .tensor_registry
                .resolve_optional(TensorName::AttnVBias, layer)
            {
                Some(name) => optional_f32(copy_tensor_optional(file, &name)?),
                None => None,
            },
            attn_o,
            attn_o_meta,
            ffn_norm: copy_f32(
                file,
                &config.tensor_registry.resolve(TensorName::FfnNorm, layer),
            )?,
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
        let token_emb_name = config.tensor_registry.resolve(TensorName::TokenEmb, 0);
        let token_emb_desc = file
            .tensor(&token_emb_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(token_emb_name.clone()))?;
        let (token_emb, token_emb_meta) = copy_tensor_with_meta(file, &token_emb_name, false)?;

        let output_norm_name = config.tensor_registry.resolve(TensorName::OutputNorm, 0);
        let output_norm = copy_f32(file, &output_norm_name)?;

        // LM head: use lm_head.weight if present, otherwise tie to embeddings
        let lm_head_name = config.tensor_registry.resolve(TensorName::LmHead, 0);
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor(&lm_head_name) {
            let lm_view = file
                .tensor(&lm_head_name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(lm_head_name.clone()))?;
            let needs_transpose = compute_transpose_flag(
                &lm_head_name,
                lm_view.dims,
                lm_view.ggml_type,
                config,
                true,
                false,
            );
            let data = copy_tensor(file, &lm_head_name)?;
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
                needs_transpose: true, // Tied embeddings always need transpose
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

// ── RFM Load Implementation ──────────────────────────────────────────────────

fn copy_rfm_tensor(file: &RfmFile, name: &str) -> Result<Vec<u8>, WeightError> {
    let t = file
        .tensor(name)
        .map_err(WeightError::Load)?
        .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
    Ok(t.data.to_vec())
}

fn rfm_type_to_ggml(t: &RfmType) -> GgmlType {
    match t {
        RfmType::F32 => GgmlType::F32,
        RfmType::Q4Split | RfmType::Q4FusedGateUp => GgmlType::Q4_0,
        RfmType::GgufPassthrough(v) => GgmlType::from_u32(*v).unwrap_or(GgmlType::Q4_0),
    }
}

fn unpack_q4_split(data: &[u8], num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);

    let scales_size = num_blocks * 2;
    let zp_size = num_blocks * 2;

    let scales = &data[0..scales_size];
    let nibbles = &data[scales_size + zp_size..];

    for i in 0..num_blocks {
        out.push(scales[i * 2]);
        out.push(scales[i * 2 + 1]);
        out.extend_from_slice(&nibbles[i * 16..(i + 1) * 16]);
    }
    out
}

fn unpack_q4_fused_gate_up(data: &[u8], gate_elements: usize) -> (Vec<u8>, Vec<u8>) {
    let num_blocks = gate_elements / 32;
    let rfm_blocks = num_blocks / 8;

    let mut gate_out = Vec::with_capacity(num_blocks * 18);
    let mut up_out = Vec::with_capacity(num_blocks * 18);

    let scales_total = rfm_blocks * 32;
    let zps_total = rfm_blocks * 32;

    let scales_offset = 0;
    let nibbles_offset = scales_total + zps_total;

    let scales = &data[scales_offset..scales_offset + scales_total];
    let nibbles = &data[nibbles_offset..];

    for b in 0..rfm_blocks {
        let gate_scale_chunk = &scales[b * 32..b * 32 + 16];
        let up_scale_chunk = &scales[b * 32 + 16..b * 32 + 32];

        let gate_nibble_chunk = &nibbles[b * 256..b * 256 + 128];
        let up_nibble_chunk = &nibbles[b * 256 + 128..b * 256 + 256];

        for i in 0..8 {
            gate_out.push(gate_scale_chunk[i * 2]);
            gate_out.push(gate_scale_chunk[i * 2 + 1]);
            gate_out.extend_from_slice(&gate_nibble_chunk[i * 16..(i + 1) * 16]);

            up_out.push(up_scale_chunk[i * 2]);
            up_out.push(up_scale_chunk[i * 2 + 1]);
            up_out.extend_from_slice(&up_nibble_chunk[i * 16..(i + 1) * 16]);
        }
    }
    (gate_out, up_out)
}

impl CpuLayerWeights {
    /// Load layer weights from an open RFM model file.
    pub fn load_rfm(
        file: &RfmFile,
        layer: usize,
        config: &ModelConfig,
    ) -> Result<Self, WeightError> {
        let load_rfm_weight =
            |name: &str, needs_transpose: bool| -> Result<(Vec<u8>, WeightMeta), WeightError> {
                let t = file
                    .tensor(name)
                    .map_err(WeightError::Load)?
                    .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;

                let data = match t.wtype {
                    RfmType::F32 => t.data.to_vec(),
                    RfmType::Q4Split => unpack_q4_split(t.data, t.element_count()),
                    RfmType::GgufPassthrough(_) => t.data.to_vec(),
                    _ => return Err(WeightError::Load(LoadError::UnknownTensorType(999))),
                };

                let meta = WeightMeta {
                    wtype: rfm_type_to_ggml(&t.wtype),
                    dims: t.dims.to_vec(),
                    needs_transpose,
                };
                Ok((data, meta))
            };

        let q_name = format!("blk.{}.attn_q.weight", layer);
        let k_name = format!("blk.{}.attn_k.weight", layer);
        let v_name = format!("blk.{}.attn_v.weight", layer);
        let o_name = format!("blk.{}.attn_output.weight", layer);

        let q_view = file
            .tensor(&q_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(q_name.clone()))?;
        let k_view = file
            .tensor(&k_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(k_name.clone()))?;
        let v_view = file
            .tensor(&v_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(v_name.clone()))?;
        let o_view = file
            .tensor(&o_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(o_name.clone()))?;

        let q_tr = compute_transpose_flag(
            &q_name,
            q_view.dims,
            rfm_type_to_ggml(&q_view.wtype),
            config,
            false,
            false,
        );
        let k_tr = compute_transpose_flag(
            &k_name,
            k_view.dims,
            rfm_type_to_ggml(&k_view.wtype),
            config,
            false,
            false,
        );
        let v_tr = compute_transpose_flag(
            &v_name,
            v_view.dims,
            rfm_type_to_ggml(&v_view.wtype),
            config,
            false,
            false,
        );
        let o_tr = compute_transpose_flag(
            &o_name,
            o_view.dims,
            rfm_type_to_ggml(&o_view.wtype),
            config,
            false,
            false,
        );

        let (attn_q, attn_q_meta) = load_rfm_weight(&q_name, q_tr)?;
        let (attn_k, attn_k_meta) = load_rfm_weight(&k_name, k_tr)?;
        let (attn_v, attn_v_meta) = load_rfm_weight(&v_name, v_tr)?;
        let (attn_o, attn_o_meta) = load_rfm_weight(&o_name, o_tr)?;

        let attn_q_bias = None;
        let attn_k_bias = None;
        let attn_v_bias = None;

        // FFN gate+up fusion
        let ffn_gate_up_name = format!("blk.{}.ffn_gate_up.weight", layer);
        let ffn_gate_up_view = file
            .tensor(&ffn_gate_up_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(ffn_gate_up_name.clone()))?;

        let (gate_data, up_data) =
            unpack_q4_fused_gate_up(ffn_gate_up_view.data, ffn_gate_up_view.element_count());

        let ffn_gate_name = format!("blk.{}.ffn_gate.weight", layer);
        let ffn_up_name = format!("blk.{}.ffn_up.weight", layer);

        let gate_tr = compute_transpose_flag(
            &ffn_gate_name,
            ffn_gate_up_view.dims,
            GgmlType::Q4_0,
            config,
            false,
            false,
        );
        let up_tr = compute_transpose_flag(
            &ffn_up_name,
            ffn_gate_up_view.dims,
            GgmlType::Q4_0,
            config,
            false,
            false,
        );

        let ffn_gate_meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: ffn_gate_up_view.dims.to_vec(),
            needs_transpose: gate_tr,
        };
        let ffn_up_meta = WeightMeta {
            wtype: GgmlType::Q4_0,
            dims: ffn_gate_up_view.dims.to_vec(),
            needs_transpose: up_tr,
        };

        // FFN down
        let ffn_down_name = format!("blk.{}.ffn_down.weight", layer);
        let ffn_down_view = file
            .tensor(&ffn_down_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(ffn_down_name.clone()))?;
        let down_tr = compute_transpose_flag(
            &ffn_down_name,
            ffn_down_view.dims,
            rfm_type_to_ggml(&ffn_down_view.wtype),
            config,
            false,
            false,
        );
        let (ffn_down, ffn_down_meta) = load_rfm_weight(&ffn_down_name, down_tr)?;

        // RMS norms
        let attn_norm_name = format!("blk.{}.attn_norm.weight", layer);
        let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer);

        let attn_norm = {
            let bytes = copy_rfm_tensor(file, &attn_norm_name)?;
            copy_f32_from_bytes(&bytes)
        };
        let ffn_norm = {
            let bytes = copy_rfm_tensor(file, &ffn_norm_name)?;
            copy_f32_from_bytes(&bytes)
        };

        let weight_type = attn_q_meta.wtype;

        Ok(Self {
            attn_norm,
            attn_q,
            attn_q_meta,
            attn_q_bias,
            attn_k,
            attn_k_meta,
            attn_k_bias,
            attn_v,
            attn_v_meta,
            attn_v_bias,
            attn_o,
            attn_o_meta,
            ffn_norm,
            ffn_gate: gate_data,
            ffn_gate_meta,
            ffn_up: up_data,
            ffn_up_meta,
            ffn_down,
            ffn_down_meta,
            weight_type,
        })
    }
}

impl CpuModelWeights {
    /// Load all model weights from an open RFM model file.
    pub fn load_rfm(file: &RfmFile, config: &ModelConfig) -> Result<Self, WeightError> {
        let n = config.num_layers;

        // Load token embedding weights
        let token_emb_name = "token_embd.weight";
        let token_emb_view = file
            .tensor(token_emb_name)
            .map_err(WeightError::Load)?
            .ok_or_else(|| WeightError::TensorNotFound(token_emb_name.to_string()))?;

        let token_emb_wtype = rfm_type_to_ggml(&token_emb_view.wtype);
        let token_emb = match token_emb_view.wtype {
            RfmType::Q4Split => {
                unpack_q4_split(token_emb_view.data, token_emb_view.element_count())
            }
            RfmType::GgufPassthrough(_) => token_emb_view.data.to_vec(),
            RfmType::F32 => token_emb_view.data.to_vec(),
            _ => return Err(WeightError::Load(LoadError::UnknownTensorType(999))),
        };
        let token_emb_meta = WeightMeta {
            wtype: token_emb_wtype,
            dims: token_emb_view.dims.to_vec(),
            needs_transpose: false,
        };

        // Output norm
        let output_norm_name = "output_norm.weight";
        let output_norm = {
            let bytes = copy_rfm_tensor(file, output_norm_name)?;
            copy_f32_from_bytes(&bytes)
        };

        // LM head
        let lm_head_name = "output.weight";
        let (lm_head, lm_head_meta, lm_head_tied) = if file.has_tensor(lm_head_name) {
            let lm_view = file
                .tensor(lm_head_name)
                .map_err(WeightError::Load)?
                .ok_or_else(|| WeightError::TensorNotFound(lm_head_name.to_string()))?;
            let lm_wtype = rfm_type_to_ggml(&lm_view.wtype);
            let needs_transpose =
                compute_transpose_flag(lm_head_name, lm_view.dims, lm_wtype, config, true, false);
            let data = match lm_view.wtype {
                RfmType::Q4Split => unpack_q4_split(lm_view.data, lm_view.element_count()),
                RfmType::GgufPassthrough(_) => lm_view.data.to_vec(),
                RfmType::F32 => lm_view.data.to_vec(),
                _ => return Err(WeightError::Load(LoadError::UnknownTensorType(999))),
            };
            let meta = WeightMeta {
                wtype: lm_wtype,
                dims: lm_view.dims.to_vec(),
                needs_transpose,
            };
            (data, meta, false)
        } else {
            // Weight tying
            let tied_meta = WeightMeta {
                wtype: token_emb_meta.wtype,
                dims: token_emb_meta.dims.clone(),
                needs_transpose: true,
            };
            (token_emb.clone(), tied_meta, true)
        };

        // Load all layers
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let layer = CpuLayerWeights::load_rfm(file, i, config)?;
            if i == 0 || (i + 1) % 8 == 0 || i + 1 == n {
                eprintln!("[cpu weights] layer {}/{} loaded from RFM", i + 1, n);
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
}
