//! Weight loading errors.

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
