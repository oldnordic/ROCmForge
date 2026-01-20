//! HTTP server module with SSE support

pub mod server;

#[cfg(feature = "context")]
pub mod context_handlers;

pub use server::*;
