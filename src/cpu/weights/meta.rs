use crate::loader::{GgmlType, LoadError};

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
    /// If this weight uses SVD outlier correction, this is the SVD rank k
    pub svd_k: Option<u32>,
}

impl WeightMeta {
    /// Create metadata from a GGUF tensor view.
    pub fn from_view(view: &crate::loader::TensorView<'_>, needs_transpose: bool) -> Self {
        Self {
            wtype: view.ggml_type,
            dims: view.dims.to_vec(),
            needs_transpose,
            svd_k: None,
        }
    }
}
