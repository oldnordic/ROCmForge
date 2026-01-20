//! Error types for SIMD matrix operations
//!
//! Provides error types for SIMD-accelerated matrix multiplication functions.

/// Matrix multiplication error types
#[derive(Debug, thiserror::Error)]
pub enum SimdMatmulError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeError { expected: usize, actual: usize },
}

/// Result type for SIMD matrix operations
pub type SimdMatmulResult<T> = Result<T, SimdMatmulError>;
