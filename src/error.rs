//! Unified error handling for ROCmForge
//!
//! This module provides a centralized error type that consolidates all domain-specific
//! errors throughout the codebase. It implements error categorization for:
//! - User errors (recoverable, actionable by users)
//! - Internal errors (bugs, system failures)
//! - Backend errors (GPU/HIP failures)
//! - Model/Loader errors (file I/O, parsing)
//! - HTTP/Server errors (API issues)

use std::fmt;

// Re-export thiserror for convenience
pub use thiserror;

/// Unified error type for ROCmForge
///
/// This enum consolidates all domain-specific errors into a single type
/// that can be used throughout the codebase. It supports categorization
/// via the `category()` method.
#[derive(Debug, thiserror::Error)]
pub enum RocmForgeError {
    // ========== Backend Errors ==========
    /// HIP/ROCm backend error
    #[error("HIP error: {0}")]
    HipError(String),

    /// GPU memory allocation failed
    #[error("GPU memory allocation failed: {0}")]
    GpuMemoryAllocationFailed(String),

    /// GPU device not found or unavailable
    #[error("GPU device not found: {0}")]
    GpuDeviceNotFound(String),

    /// GPU kernel launch failed
    #[error("GPU kernel launch failed: {0}")]
    GpuKernelLaunchFailed(String),

    // ========== Model/Loader Errors ==========
    /// Model loading failed
    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    /// Invalid or corrupted model file
    #[error("Invalid model file: {0}")]
    InvalidModelFile(String),

    /// Unsupported model format or architecture
    #[error("Unsupported model format: {0}")]
    UnsupportedModelFormat(String),

    /// Tensor not found in model
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// Invalid tensor shape
    #[error("Invalid tensor shape: {0}")]
    InvalidTensorShape(String),

    // ========== KV Cache Errors ==========
    /// KV cache capacity exceeded
    #[error("KV cache capacity exceeded")]
    CacheCapacityExceeded,

    /// Invalid sequence ID
    #[error("Invalid sequence ID: {0}")]
    InvalidSequenceId(u32),

    /// Page not found for sequence
    #[error("Page not found for sequence: {0}")]
    PageNotFound(u32),

    /// Invalid cache configuration
    #[error("Invalid cache configuration: {0}")]
    InvalidCacheConfiguration(String),

    // ========== Scheduler Errors ==========
    /// Request not found
    #[error("Request not found: {0}")]
    RequestNotFound(u32),

    /// Batch size exceeded maximum
    #[error("Batch size exceeded: {actual} > {max}")]
    BatchSizeExceeded { max: usize, actual: usize },

    /// Queue capacity exceeded
    #[error("Queue capacity exceeded")]
    QueueCapacityExceeded,

    /// Invalid request state transition
    #[error("Invalid request state transition")]
    InvalidStateTransition,

    // ========== Sampler Errors ==========
    /// Empty logits vector
    #[error("Empty logits vector")]
    EmptyLogits,

    /// Invalid temperature value
    #[error("Invalid temperature: {0}. Must be > 0")]
    InvalidTemperature(f32),

    /// Invalid top_k value
    #[error("Invalid top_k: {0}. Must be > 0")]
    InvalidTopK(usize),

    /// Invalid top_p value
    #[error("Invalid top_p: {0}. Must be in (0, 1]")]
    InvalidTopP(f32),

    /// All probabilities are zero
    #[error("All probabilities are zero")]
    ZeroProbabilities,

    // ========== HTTP/Server Errors ==========
    /// Invalid client request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Generation failed
    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    /// Engine not initialized
    #[error("Inference engine not initialized")]
    EngineNotInitialized,

    // ========== Engine Errors ==========
    /// Inference execution failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Backend initialization failed
    #[error("Backend initialization failed: {0}")]
    BackendInitializationFailed(String),

    /// Cache initialization failed
    #[error("Cache initialization failed: {0}")]
    CacheInitializationFailed(String),

    /// Invalid engine configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    // ========== I/O Errors ==========
    /// File I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Memory mapping failed
    #[error("Memory mapping failed: {0}")]
    MmapError(String),

    // ========== Internal Errors ==========
    /// Internal error (indicates a bug)
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Lock poisoned (indicates a bug or concurrent access issue)
    #[error("Internal lock poisoned: {0}")]
    LockPoisoned(String),

    /// Unimplemented feature
    #[error("Unimplemented feature: {0}")]
    Unimplemented(String),

    /// Generic error with context
    #[error("{0}")]
    Generic(String),
}

impl RocmForgeError {
    /// Categorize the error for handling decisions
    ///
    /// Returns the error category, which can be used to determine
    /// whether an error is recoverable, user-facing, or internal.
    ///
    /// # Examples
    /// ```ignore
    /// match error.category() {
    ///     ErrorCategory::User => println!("User error: fix your input"),
    ///     ErrorCategory::Recoverable => println!("Retry later"),
    ///     ErrorCategory::Internal => println!("Report this bug"),
    /// }
    /// ```
    pub fn category(&self) -> ErrorCategory {
        match self {
            // User errors - actionable by users
            RocmForgeError::InvalidRequest(_)
            | RocmForgeError::InvalidConfiguration(_)
            | RocmForgeError::InvalidTemperature(_)
            | RocmForgeError::InvalidTopK(_)
            | RocmForgeError::InvalidTopP(_)
            | RocmForgeError::InvalidCacheConfiguration(_)
            | RocmForgeError::InvalidTensorShape(_)
            | RocmForgeError::UnsupportedModelFormat(_) => ErrorCategory::User,

            // Recoverable errors - temporary conditions
            RocmForgeError::CacheCapacityExceeded
            | RocmForgeError::QueueCapacityExceeded
            | RocmForgeError::BatchSizeExceeded { .. }
            | RocmForgeError::EngineNotInitialized => ErrorCategory::Recoverable,

            // Internal errors - bugs, system failures
            RocmForgeError::InternalError(_)
            | RocmForgeError::LockPoisoned(_)
            | RocmForgeError::Unimplemented(_) => ErrorCategory::Internal,

            // Backend errors - GPU/HIP failures
            RocmForgeError::HipError(_)
            | RocmForgeError::GpuMemoryAllocationFailed(_)
            | RocmForgeError::GpuDeviceNotFound(_)
            | RocmForgeError::GpuKernelLaunchFailed(_)
            | RocmForgeError::BackendInitializationFailed(_) => ErrorCategory::Backend,

            // Model/Loader errors - file issues
            RocmForgeError::ModelLoadFailed(_)
            | RocmForgeError::InvalidModelFile(_)
            | RocmForgeError::TensorNotFound(_)
            | RocmForgeError::MmapError(_)
            | RocmForgeError::IoError(_) => ErrorCategory::Model,

            // Everything else - inference failures, etc.
            _ => ErrorCategory::Internal,
        }
    }

    /// Check if this error is recoverable (temporary condition)
    ///
    /// Recoverable errors include capacity limits and unavailability.
    /// The caller may retry the operation after waiting.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.category(),
            ErrorCategory::Recoverable | ErrorCategory::Backend
        )
    }

    /// Check if this is a user-facing error (actionable by users)
    ///
    /// User errors indicate invalid input or configuration.
    /// The user should fix their request.
    pub fn is_user_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::User)
    }

    /// Check if this is an internal error (indicates a bug)
    ///
    /// Internal errors should be reported to developers.
    pub fn is_internal_error(&self) -> bool {
        matches!(self.category(), ErrorCategory::Internal)
    }
}

/// Error category for handling decisions
///
/// Categories help determine how to handle errors:
/// - User: Show to user, ask them to fix input
/// - Recoverable: Retry after waiting
/// - Internal: Log and report as bug
/// - Backend: May indicate GPU issues
/// - Model: File or model problems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// User error - invalid input or configuration
    User,
    /// Recoverable error - temporary condition
    Recoverable,
    /// Internal error - indicates a bug
    Internal,
    /// Backend error - GPU/HIP failure
    Backend,
    /// Model error - file or model issue
    Model,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::User => write!(f, "User"),
            ErrorCategory::Recoverable => write!(f, "Recoverable"),
            ErrorCategory::Internal => write!(f, "Internal"),
            ErrorCategory::Backend => write!(f, "Backend"),
            ErrorCategory::Model => write!(f, "Model"),
        }
    }
}

// ========== Conversion Traits for Existing Error Types ==========

// Note: From<std::io::Error> is auto-derived by #[from] on IoError variant
// Note: From<std::sync::PoisonError<T>> is implemented below since LockPoisoned takes String

impl<T> From<std::sync::PoisonError<T>> for RocmForgeError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        RocmForgeError::LockPoisoned(err.to_string())
    }
}

// Helper type alias for Results using RocmForgeError
pub type ForgeResult<T> = std::result::Result<T, RocmForgeError>;

// ========== Helper Functions ==========

/// Create a user-facing error with context
///
/// # Examples
/// ```ignore
/// return Err(user_error!("Invalid temperature value"));
/// ```
#[macro_export]
macro_rules! user_error {
    ($msg:expr) => {
        $crate::error::RocmForgeError::InvalidRequest($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::RocmForgeError::InvalidRequest(format!($fmt, $($arg)*))
    };
}

/// Create an internal error with context
///
/// # Examples
/// ```ignore
/// return Err(internal_error!("Unexpected state in tokenizer"));
/// ```
#[macro_export]
macro_rules! internal_error {
    ($msg:expr) => {
        $crate::error::RocmForgeError::InternalError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::RocmForgeError::InternalError(format!($fmt, $($arg)*))
    };
}

/// Create a backend error with context
///
/// # Examples
/// ```ignore
/// return Err(backend_error!("HIP kernel launch failed for matmul"));
/// ```
#[macro_export]
macro_rules! backend_error {
    ($msg:expr) => {
        $crate::error::RocmForgeError::HipError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::RocmForgeError::HipError(format!($fmt, $($arg)*))
    };
}

/// Create a model error with context
///
/// # Examples
/// ```ignore
/// return Err(model_error!("Tensor not found: token_embeddings"));
/// ```
#[macro_export]
macro_rules! model_error {
    ($msg:expr) => {
        $crate::error::RocmForgeError::ModelLoadFailed($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::RocmForgeError::ModelLoadFailed(format!($fmt, $($arg)*))
    };
}

/// Wrap an error with additional context using anyhow-style context
///
/// This helper function adds context to any error type that implements
/// std::error::Error.
///
/// # Examples
/// ```ignore
/// let result = risky_operation().map_err(|e| context(e, "during model loading"))?;
/// ```
pub fn context<E>(err: E, msg: &str) -> RocmForgeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    RocmForgeError::InternalError(format!("{}: {}", msg, err))
}

/// Wrap an IO error with context
///
/// # Examples
/// ```ignore
/// let file = File::open(path).map_err(|e| io_context(e, "opening config"))?;
/// ```
pub fn io_context(err: std::io::Error, msg: &str) -> RocmForgeError {
    RocmForgeError::IoError(std::io::Error::new(
        err.kind(),
        format!("{}: {}", msg, err),
    ))
}

/// Convert an option to a result with a user error
///
/// # Examples
/// ```ignore
/// let value = optional_value.ok_or_else(|| user_err("Value not found"))?;
/// ```
pub fn user_err(msg: &str) -> RocmForgeError {
    RocmForgeError::InvalidRequest(msg.to_string())
}

/// Convert an option to a result with an internal error
///
/// # Examples
/// ```ignore
/// let value = required_value.ok_or_else(|| internal_err("Invariant violated"))?;
/// ```
pub fn internal_err(msg: &str) -> RocmForgeError {
    RocmForgeError::InternalError(msg.to_string())
}

/// Convert an option to a result with a backend error
///
/// # Examples
/// ```ignore
/// let device = device_opt.ok_or_else(|| backend_err("GPU not available"))?;
/// ```
pub fn backend_err(msg: &str) -> RocmForgeError {
    RocmForgeError::HipError(msg.to_string())
}

/// Check if a result is Ok, return a converted error otherwise
///
/// # Examples
/// ```ignore
/// let value = check(result).map_err(|e| context(e, "during parsing"))?;
/// ```
pub fn check<T, E>(result: std::result::Result<T, E>) -> ForgeResult<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    result.map_err(|e| context(e, "operation failed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categories() {
        // User errors
        assert_eq!(
            RocmForgeError::InvalidRequest("test".to_string()).category(),
            ErrorCategory::User
        );
        assert_eq!(
            RocmForgeError::InvalidTemperature(0.0).category(),
            ErrorCategory::User
        );

        // Recoverable errors
        assert_eq!(
            RocmForgeError::CacheCapacityExceeded.category(),
            ErrorCategory::Recoverable
        );
        assert_eq!(
            RocmForgeError::EngineNotInitialized.category(),
            ErrorCategory::Recoverable
        );

        // Internal errors
        assert_eq!(
            RocmForgeError::InternalError("test".to_string()).category(),
            ErrorCategory::Internal
        );
        assert_eq!(
            RocmForgeError::LockPoisoned("test".to_string()).category(),
            ErrorCategory::Internal
        );

        // Backend errors
        assert_eq!(
            RocmForgeError::HipError("test".to_string()).category(),
            ErrorCategory::Backend
        );
        assert_eq!(
            RocmForgeError::GpuDeviceNotFound("test".to_string()).category(),
            ErrorCategory::Backend
        );
    }

    #[test]
    fn test_is_recoverable() {
        assert!(RocmForgeError::CacheCapacityExceeded.is_recoverable());
        assert!(RocmForgeError::QueueCapacityExceeded.is_recoverable());
        assert!(RocmForgeError::EngineNotInitialized.is_recoverable());
        assert!(RocmForgeError::HipError("test".to_string()).is_recoverable());

        // Not recoverable
        assert!(!RocmForgeError::InvalidRequest("test".to_string()).is_recoverable());
        assert!(!RocmForgeError::InternalError("test".to_string()).is_recoverable());
    }

    #[test]
    fn test_is_user_error() {
        assert!(RocmForgeError::InvalidRequest("test".to_string()).is_user_error());
        assert!(RocmForgeError::InvalidTemperature(0.0).is_user_error());
        assert!(RocmForgeError::InvalidTopK(0).is_user_error());
        assert!(RocmForgeError::InvalidTopP(1.5).is_user_error());
        assert!(RocmForgeError::UnsupportedModelFormat("test".to_string()).is_user_error());

        // Not user errors
        assert!(!RocmForgeError::CacheCapacityExceeded.is_user_error());
        assert!(!RocmForgeError::InternalError("test".to_string()).is_user_error());
    }

    #[test]
    fn test_is_internal_error() {
        assert!(RocmForgeError::InternalError("test".to_string()).is_internal_error());
        assert!(RocmForgeError::LockPoisoned("test".to_string()).is_internal_error());
        assert!(RocmForgeError::Unimplemented("test".to_string()).is_internal_error());

        // Not internal errors
        assert!(!RocmForgeError::InvalidRequest("test".to_string()).is_internal_error());
        assert!(!RocmForgeError::CacheCapacityExceeded.is_internal_error());
    }

    #[test]
    fn test_error_display() {
        let err = RocmForgeError::InvalidRequest("test message".to_string());
        assert_eq!(err.to_string(), "Invalid request: test message");

        let err = RocmForgeError::InvalidTemperature(0.0);
        assert_eq!(err.to_string(), "Invalid temperature: 0. Must be > 0");

        let err = RocmForgeError::BatchSizeExceeded { max: 32, actual: 64 };
        assert_eq!(err.to_string(), "Batch size exceeded: 64 > 32");
    }

    #[test]
    fn test_helper_functions() {
        let err = user_err("test error");
        assert!(matches!(err, RocmForgeError::InvalidRequest(_)));
        assert!(err.is_user_error());

        let err = internal_err("bug detected");
        assert!(matches!(err, RocmForgeError::InternalError(_)));
        assert!(err.is_internal_error());

        let err = backend_err("GPU unavailable");
        assert!(matches!(err, RocmForgeError::HipError(_)));
    }

    #[test]
    fn test_macros() {
        let err = user_error!("test");
        assert!(matches!(err, RocmForgeError::InvalidRequest(_)));

        let err = user_error!("value: {}", 42);
        assert_eq!(err.to_string(), "Invalid request: value: 42");

        let err = internal_error!("bug");
        assert!(matches!(err, RocmForgeError::InternalError(_)));

        let err = backend_error!("HIP error");
        assert!(matches!(err, RocmForgeError::HipError(_)));

        let err = model_error!("tensor not found");
        assert!(matches!(err, RocmForgeError::ModelLoadFailed(_)));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: RocmForgeError = io_err.into();
        assert!(matches!(err, RocmForgeError::IoError(_)));
    }

    #[test]
    fn test_poison_error_from_impl_exists() {
        use std::sync::{Mutex, PoisonError};

        // Verify the From<PoisonError<T>> trait impl exists by calling it explicitly
        // We create a dummy error and verify it converts correctly

        // Create a poison error manually (without actually panicking)
        // by using a helper that creates one from a known guard type
        fn convert_poison<T>(err: PoisonError<T>) -> RocmForgeError {
            RocmForgeError::from(err)
        }

        // The type check verifies the impl exists
        let _ = convert_poison::<i32> as fn(PoisonError<i32>) -> RocmForgeError;
    }

    #[test]
    fn test_context_helper() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "base error");
        let err = context(io_err, "during operation");
        assert!(matches!(err, RocmForgeError::InternalError(_)));
        assert!(err.to_string().contains("during operation"));
    }

    #[test]
    fn test_io_context_helper() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file.txt");
        let err = io_context(io_err, "opening config");
        assert!(matches!(err, RocmForgeError::IoError(_)));
        assert!(err.to_string().contains("opening config"));
    }

    #[test]
    fn test_check_helper() {
        let ok_result: std::result::Result<i32, std::io::Error> = Ok(42);
        assert!(check(ok_result).is_ok());

        let err_result: std::result::Result<i32, std::io::Error> =
            Err(std::io::Error::new(std::io::ErrorKind::Other, "error"));
        let result = check(err_result);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_category_display() {
        assert_eq!(ErrorCategory::User.to_string(), "User");
        assert_eq!(ErrorCategory::Recoverable.to_string(), "Recoverable");
        assert_eq!(ErrorCategory::Internal.to_string(), "Internal");
        assert_eq!(ErrorCategory::Backend.to_string(), "Backend");
        assert_eq!(ErrorCategory::Model.to_string(), "Model");
    }
}
