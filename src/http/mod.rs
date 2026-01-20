//! HTTP server module with SSE support
//!
//! This module provides the HTTP/SSE server for ROCmForge inference engine.
//! It is organized into focused submodules:
//!
//! - **types**: HTTP error types and request/response structures
//! - **routes**: Router setup and route definitions
//! - **handlers**: HTTP request handlers
//! - **server**: InferenceServer main implementation and server lifecycle

pub mod server;

// Public submodules (to be added)
pub mod types;
pub mod routes;
pub mod handlers;

#[cfg(feature = "context")]
pub mod context_handlers;

// Re-exports for backward compatibility
pub use server::{InferenceServer, run_server, ServerResult};
pub use types::{
    HttpError, ServerError, GenerateRequest, GenerateResponse, TokenStream,
    GenerationState, ServerState, TracesQuery,
};
pub use routes::create_router;
