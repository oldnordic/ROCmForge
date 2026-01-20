//! HIP error types

use thiserror::Error;

/// HIP error types
#[derive(Error, Debug, Clone)]
pub enum HipError {
    #[error("HIP initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Kernel loading failed: {0}")]
    KernelLoadFailed(String),
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    #[error("Memory copy failed: {0}")]
    MemoryCopyFailed(String),
    #[error("Memory query failed: {0}")]
    MemoryQueryFailed(String),
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    #[error("Device not found")]
    DeviceNotFound,
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Generic error: {0}")]
    GenericError(String),
    #[error("Internal lock poisoned - this indicates a bug: {0}")]
    LockPoisoned(String),
}

impl<T> From<std::sync::PoisonError<T>> for HipError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        HipError::LockPoisoned(format!("Lock poisoned: {}", err))
    }
}

/// HIP result type
pub type HipResult<T> = Result<T, HipError>;

// Convert KVCacheError to HipError
impl From<crate::model::kv_cache::KVCacheError> for HipError {
    fn from(err: crate::model::kv_cache::KVCacheError) -> Self {
        HipError::GenericError(err.to_string())
    }
}

impl From<crate::ggml::GgmlError> for HipError {
    fn from(err: crate::ggml::GgmlError) -> Self {
        HipError::GenericError(format!("{:?}", err))
    }
}

impl HipError {
    /// Check if this error is recoverable (temporary condition)
    ///
    /// Recoverable errors may be retried with exponential backoff.
    /// These include:
    /// - Temporary device errors (GPU busy, driver resetting)
    /// - Memory allocation failures (may succeed after GC or waiting)
    /// - Memory copy failures (temporary driver issues)
    ///
    /// Non-recoverable errors (should NOT be retried):
    /// - DeviceNotFound (no GPU available)
    /// - InitializationFailed (HIP runtime broken)
    /// - KernelLoadFailed (corrupted kernel file)
    /// - LockPoisoned (data corruption bug)
    /// - GenericError (unknown errors)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            HipError::DeviceError(_)
                | HipError::MemoryAllocationFailed(_)
                | HipError::MemoryCopyFailed(_)
                | HipError::MemoryQueryFailed(_)
                | HipError::KernelLaunchFailed(_)
        )
    }

    /// Check if this error is permanent (should never retry)
    pub fn is_permanent(&self) -> bool {
        !self.is_recoverable()
    }
}
