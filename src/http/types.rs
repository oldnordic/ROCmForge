//! HTTP types for ROCmForge inference server
//!
//! This module contains all HTTP-related types including:
//! - Error types (HttpError, ServerError)
//! - Request/Response structures (GenerateRequest, GenerateResponse, TokenStream)
//! - Generation state management
//! - Server state and query parameter types

use crate::error::{ErrorCategory, RocmForgeError};
use axum::{
    http::{header::HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Suggested retry delay for capacity-related errors (in seconds)
pub const RETRY_AFTER_SECONDS: u32 = 60;

/// HTTP error response with appropriate status code based on error category
#[derive(Debug)]
pub struct HttpError {
    /// The underlying error
    pub error: RocmForgeError,
    /// Suggested retry-after delay for recoverable errors
    pub retry_after: Option<u32>,
}

impl HttpError {
    /// Create a new HTTP error from a RocmForgeError
    pub fn new(error: RocmForgeError) -> Self {
        let retry_after = match error.category() {
            ErrorCategory::Recoverable | ErrorCategory::Backend => Some(RETRY_AFTER_SECONDS),
            _ => None,
        };

        Self { error, retry_after }
    }

    /// Get the appropriate HTTP status code for this error
    fn status_code(&self) -> StatusCode {
        match self.error.category() {
            ErrorCategory::User => StatusCode::BAD_REQUEST,           // 400
            ErrorCategory::Model => StatusCode::BAD_REQUEST,           // 400
            ErrorCategory::Recoverable => StatusCode::SERVICE_UNAVAILABLE, // 503
            ErrorCategory::Backend => StatusCode::SERVICE_UNAVAILABLE,    // 503
            ErrorCategory::Internal => StatusCode::INTERNAL_SERVER_ERROR, // 500
        }
    }

    /// Get the error message for the response
    fn error_message(&self) -> String {
        self.error.to_string()
    }

    /// Get the error category for the response
    fn error_category(&self) -> String {
        self.error.category().to_string()
    }

    /// Check if the error is recoverable (for client retry logic)
    fn is_recoverable(&self) -> bool {
        self.error.is_recoverable()
    }
}

impl From<RocmForgeError> for HttpError {
    fn from(error: RocmForgeError) -> Self {
        Self::new(error)
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        use axum::http::header::RETRY_AFTER;

        let status = self.status_code();
        let category = self.error_category();
        let message = self.error_message();
        let recoverable = self.is_recoverable();

        // Build response body
        let body = Json(serde_json::json!({
            "error": message,
            "category": category,
            "recoverable": recoverable,
            "status": "error"
        }));

        // Add Retry-After header for recoverable errors
        let mut headers = HeaderMap::new();
        if let Some(retry_after) = self.retry_after {
            headers.insert(RETRY_AFTER, retry_after.to_string().parse().unwrap());
        }

        (status, headers, body).into_response()
    }
}

/// Legacy ServerError for backward compatibility
/// Deprecated: Use RocmForgeError directly instead
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
    #[error("Request not found: {0}")]
    RequestNotFound(u32),
    #[error("Internal server error: {0}")]
    InternalError(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let http_error: HttpError = match self {
            ServerError::InvalidRequest(msg) => {
                RocmForgeError::InvalidRequest(msg).into()
            }
            ServerError::GenerationFailed(msg) => {
                RocmForgeError::GenerationFailed(msg).into()
            }
            ServerError::RequestNotFound(id) => {
                RocmForgeError::RequestNotFound(id).into()
            }
            ServerError::InternalError(msg) => {
                RocmForgeError::InternalError(msg).into()
            }
        };
        http_error.into_response()
    }
}

/// Request to generate text completion
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

/// Response containing generated text
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub request_id: u32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub finished: bool,
    pub finish_reason: Option<String>,
}

/// Single token in SSE stream
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenStream {
    pub request_id: u32,
    pub token: u32,
    pub text: String,
    pub finished: bool,
    pub finish_reason: Option<String>,
}

/// State for an in-progress generation
#[derive(Debug, Clone)]
pub struct GenerationState {
    pub request_id: u32,
    pub prompt: String,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub max_tokens: usize,
    pub finished: bool,
    pub finish_reason: Option<String>,
}

impl GenerationState {
    pub fn new(request_id: u32, prompt: String, max_tokens: usize) -> Self {
        GenerationState {
            request_id,
            prompt,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            max_tokens,
            finished: false,
            finish_reason: None,
        }
    }

    pub fn add_token(&mut self, token: u32, token_text: String) {
        self.generated_tokens.push(token);
        self.generated_text.push_str(&token_text);

        if self.generated_tokens.len() >= self.max_tokens {
            self.finished = true;
            self.finish_reason = Some("length".to_string());
        }
    }

    pub fn finish(&mut self, reason: String) {
        self.finished = true;
        self.finish_reason = Some(reason);
    }
}

/// Server state tracking in-progress generations
pub type ServerState = Arc<RwLock<HashMap<u32, GenerationState>>>;

/// Query parameters for the /traces endpoint
#[derive(Debug, Deserialize)]
pub struct TracesQuery {
    /// Maximum number of traces to return (default: all)
    pub limit: Option<usize>,
    /// Clear traces after returning them
    pub clear: Option<bool>,
}

/// Result type for server operations
pub type ServerResult<T> = anyhow::Result<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_state_creation() {
        let state = GenerationState::new(1, "Hello".to_string(), 10);

        assert_eq!(state.request_id, 1);
        assert_eq!(state.prompt, "Hello");
        assert_eq!(state.max_tokens, 10);
        assert!(!state.finished);
        assert!(state.generated_tokens.is_empty());
        assert!(state.generated_text.is_empty());
    }

    #[test]
    fn test_generation_state_add_token() {
        let mut state = GenerationState::new(1, "Hello".to_string(), 2);

        state.add_token(42, " world".to_string());
        assert_eq!(state.generated_tokens, vec![42]);
        assert_eq!(state.generated_text, " world");
        assert!(!state.finished);

        state.add_token(43, "!".to_string());
        assert_eq!(state.generated_tokens, vec![42, 43]);
        assert_eq!(state.generated_text, " world!");
        assert!(state.finished);
        assert_eq!(state.finish_reason, Some("length".to_string()));
    }

    #[test]
    fn test_generation_state_finish() {
        let mut state = GenerationState::new(1, "Hello".to_string(), 10);

        state.finish("stop".to_string());
        assert!(state.finished);
        assert_eq!(state.finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_traces_query_params() {
        let query = TracesQuery {
            limit: Some(10),
            clear: Some(false),
        };

        assert_eq!(query.limit, Some(10));
        assert_eq!(query.clear, Some(false));
    }

    #[test]
    fn test_traces_query_default() {
        let query = TracesQuery {
            limit: None,
            clear: None,
        };

        assert!(query.limit.is_none());
        assert!(query.clear.is_none());
    }

    #[test]
    fn test_http_error_user_category_returns_400() {
        // User errors should return 400 Bad Request
        let error = RocmForgeError::InvalidRequest("Invalid prompt".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
        assert!(!http_error.is_recoverable());
        assert!(http_error.retry_after.is_none());
    }

    #[test]
    fn test_http_error_model_category_returns_400() {
        // Model errors should return 400 Bad Request
        let error = RocmForgeError::InvalidModelFile("Corrupted model".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "Model");
        assert!(!http_error.is_recoverable());
        assert!(http_error.retry_after.is_none());
    }

    #[test]
    fn test_http_error_recoverable_category_returns_503_with_retry_after() {
        // Recoverable errors should return 503 Service Unavailable with Retry-After
        let error = RocmForgeError::CacheCapacityExceeded;
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(http_error.error_category(), "Recoverable");
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_queue_capacity_exceeded_returns_503() {
        let error = RocmForgeError::QueueCapacityExceeded;
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_batch_size_exceeded_returns_503() {
        let error = RocmForgeError::BatchSizeExceeded { max: 32, actual: 64 };
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_engine_not_initialized_returns_503() {
        let error = RocmForgeError::EngineNotInitialized;
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_backend_category_returns_503_with_retry_after() {
        // Backend errors (GPU/HIP) should return 503 with Retry-After
        let error = RocmForgeError::HipError("Device lost".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(http_error.error_category(), "Backend");
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_gpu_memory_failed_returns_503() {
        let error = RocmForgeError::GpuMemoryAllocationFailed("OOM".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_gpu_device_not_found_returns_503() {
        let error = RocmForgeError::GpuDeviceNotFound("No AMD GPU".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(http_error.is_recoverable());
        assert_eq!(http_error.retry_after, Some(RETRY_AFTER_SECONDS));
    }

    #[test]
    fn test_http_error_internal_category_returns_500() {
        // Internal errors should return 500 Internal Server Error
        let error = RocmForgeError::InternalError("Bug detected".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(http_error.error_category(), "Internal");
        assert!(!http_error.is_recoverable());
        assert!(http_error.retry_after.is_none());
    }

    #[test]
    fn test_http_error_lock_poisoned_returns_500() {
        let error = RocmForgeError::LockPoisoned("Mutex poisoned".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!http_error.is_recoverable());
        assert!(http_error.retry_after.is_none());
    }

    #[test]
    fn test_http_error_from_rocm_forge_error() {
        // Test the From trait implementation
        let error = RocmForgeError::InvalidTemperature(0.0);
        let http_error: HttpError = error.into();

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
    }

    #[test]
    fn test_http_error_into_response_has_correct_status() {
        // Test that IntoResponse produces correct status codes
        let error = RocmForgeError::CacheCapacityExceeded;
        let http_error = HttpError::new(error);
        let response = http_error.into_response();

        // Response should have 503 status
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_error_user_invalid_temperature_returns_400() {
        let error = RocmForgeError::InvalidTemperature(0.0);
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
    }

    #[test]
    fn test_http_error_user_invalid_top_k_returns_400() {
        let error = RocmForgeError::InvalidTopK(0);
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
    }

    #[test]
    fn test_http_error_user_invalid_top_p_returns_400() {
        let error = RocmForgeError::InvalidTopP(1.5);
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
    }

    #[test]
    fn test_http_error_user_unsupported_format_returns_400() {
        let error = RocmForgeError::UnsupportedModelFormat("unknown".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
    }

    #[test]
    fn test_http_error_invalid_configuration_returns_400() {
        let error = RocmForgeError::InvalidConfiguration("Invalid config".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(http_error.error_category(), "User");
    }

    #[test]
    fn test_legacy_server_error_invalid_request_maps_correctly() {
        // Legacy ServerError should map through to correct HTTP status
        let server_error = ServerError::InvalidRequest("Bad input".to_string());
        let response = server_error.into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_legacy_server_error_generation_failed_maps_to_500() {
        // GenerationFailed maps to Internal (500)
        let server_error = ServerError::GenerationFailed("Failed".to_string());
        let response = server_error.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_legacy_server_error_request_not_found_maps_to_500() {
        // RequestNotFound maps to Internal (500) - not explicitly categorized
        let server_error = ServerError::RequestNotFound(123);
        let response = server_error.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_legacy_server_error_internal_error_maps_to_500() {
        let server_error = ServerError::InternalError("Oops".to_string());
        let response = server_error.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_error_message_preserved() {
        let error = RocmForgeError::InvalidRequest("Test error message".to_string());
        let http_error = HttpError::new(error);

        assert_eq!(http_error.error_message(), "Invalid request: Test error message");
    }

    #[test]
    fn test_http_error_retry_after_constant() {
        // Verify the retry-after constant is set to 60 seconds
        assert_eq!(RETRY_AFTER_SECONDS, 60);
    }
}
